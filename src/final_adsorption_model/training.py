"""Locked ten-fold LightGBM training and evaluation, independent of the parent project."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model import FinalAdsorptionModel, _patch_autogluon_metadata_scan, prepare_model_features

LABEL = "adsorption"
SPLITS = ("train", "valid", "test")  # Storage file keys, not evaluation partitions.
FOLD_COLUMN = "__development_cv_fold__"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_locked_data(project_dir: Path, data_dir: Path) -> dict[str, pd.DataFrame]:
    """Validate file identity, row order and the compound holdout before fitting."""
    manifest = json.loads((project_dir / "config/data_manifest.json").read_text(encoding="utf-8"))
    frames = {}
    for item in manifest["files"]:
        path = data_dir / item["file"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Locked data checksum mismatch: {path}")
        frame = pd.read_csv(path)
        if len(frame) != item["rows"] or not frame.row_id.is_unique:
            raise ValueError(f"Invalid row count or repeated row_id: {path}")
        if not np.isfinite(frame[LABEL].to_numpy(float)).all():
            raise ValueError(f"Non-finite labels: {path}")
        prepare_model_features(frame, project_dir)
        frames[item["split"]] = frame
    ids = pd.concat([frames[s].row_id for s in SPLITS], ignore_index=True)
    if not ids.is_unique:
        raise ValueError("A row appears in more than one storage file")
    group = "compound_group_key"
    development = set(frames["train"][group]) | set(frames["valid"][group])
    if development & set(frames["test"][group]):
        raise ValueError("Test compounds overlap the development set")
    return frames


def verify_artifacts(project_dir: Path) -> dict[str, int]:
    manifest = json.loads((project_dir / "config/artifact_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = project_dir / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Final artifact missing or changed: {path}")
    return {"verified_files": len(manifest["files"]),
            "verified_bytes": sum(item["bytes"] for item in manifest["files"])}


def development_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join the two storage files in locked order into one development set."""
    return pd.concat([frames[split] for split in ("train", "valid")],
                     ignore_index=True).assign(split="development")


def development_folds(development: pd.DataFrame, config: dict) -> np.ndarray:
    from sklearn.model_selection import KFold

    if (len(development) != 616 or config["num_bag_folds"] != 10
            or config["num_bag_sets"] != 1 or "development_cv_seed" not in config):
        raise ValueError("Expected 616 development records and one run of ten-fold CV")
    splitter = KFold(n_splits=config["num_bag_folds"], shuffle=True,
                     random_state=config["development_cv_seed"])
    folds = np.zeros(len(development), dtype=int)
    for fold, (_, indices) in enumerate(splitter.split(development), start=1):
        folds[indices] = fold
    return folds


def train_locked_model(
    project_dir: Path, data_dir: Path, model_dir: Path,
) -> FinalAdsorptionModel:
    """Reproduce the recipe in a NEW directory, preserving the confirmed artifacts."""
    if model_dir.exists():
        raise FileExistsError(f"Choose a new model directory: {model_dir}")
    frames = load_locked_data(project_dir, data_dir)
    config = json.loads((project_dir / "config/training_config.json").read_text(encoding="utf-8"))
    model = FinalAdsorptionModel(project_dir, model_dir=model_dir)
    development = development_frame(frames)
    folds = development_folds(development, config)
    component, = model.components
    recipe = config["baseline"]
    if (config["num_stack_levels"] != 0 or recipe["fit_weighted_ensemble"]
            or set(recipe["hyperparameters"]) != {"GBM"}
            or len(recipe["hyperparameters"]["GBM"]) != 1):
        raise ValueError("The default recipe requires one LightGBM without stacking or weighted ensembles")
    training = prepare_model_features(development, project_dir).copy()
    training[LABEL] = development[LABEL].to_numpy(float)
    training[FOLD_COLUMN] = folds
    _patch_autogluon_metadata_scan()
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor(
        label=config["label"], problem_type=config["problem_type"],
        eval_metric=config["eval_metric"], path=str(component.model_path),
        groups=FOLD_COLUMN, verbosity=2, learner_kwargs={"random_state": recipe["seed"]},
    )
    predictor.fit(
        train_data=training, presets=config["presets"],
        hyperparameters=recipe["hyperparameters"],
        num_bag_folds=config["num_bag_folds"], num_bag_sets=config["num_bag_sets"],
        num_stack_levels=0, fit_weighted_ensemble=False,
        fit_full_last_level_weighted_ensemble=False,
        full_weighted_ensemble_additionally=False,
        dynamic_stacking=False, calibrate_decision_threshold=False,
        save_bag_folds=True, refit_full=False, set_best_to_refit_full=False,
        num_gpus=0, num_cpus=recipe["num_cpus"],
    )
    if predictor.model_names() != [component.model_name]:
        raise RuntimeError(f"Expected only the configured LightGBM: {component.model_name}")
    model._predictors = [predictor]
    return model


def score(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if (y.ndim != 1 or not y.size or y.shape != prediction.shape
            or not np.isfinite(y).all() or not np.isfinite(prediction).all()):
        raise ValueError("Labels and predictions must be finite, nonempty and aligned vectors")
    error = y - prediction
    total = np.square(y - y.mean()).sum()
    return {"mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "r2": float(1.0 - np.square(error).sum() / total) if total else float("nan")}


def evaluate_locked_model(
    model: FinalAdsorptionModel, data_dir: Path, output_dir: Path,
    check_reference: bool = False,
) -> dict:
    """Report ten-fold mean training/validation scores and the fixed-test score."""
    frames = load_locked_data(model.project_dir, data_dir)
    model.load()
    config = json.loads((model.project_dir / "config/training_config.json").read_text(encoding="utf-8"))
    from .development_evaluation import evaluate_development_model

    return evaluate_development_model(model, frames, config, output_dir, check_reference)
