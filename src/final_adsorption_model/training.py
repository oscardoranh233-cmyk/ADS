"""Locked training and evaluation without dependencies on the parent project."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model import FinalAdsorptionModel, _patch_autogluon_metadata_scan, prepare_model_features

LABEL = "adsorption"
SPLITS = ("train", "valid", "test")


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
        raise ValueError("A row appears in more than one split")
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


def train_locked_model(
    project_dir: Path, data_dir: Path, model_dir: Path,
) -> FinalAdsorptionModel:
    """Reproduce the recipe in a NEW directory, preserving the confirmed artifacts."""
    if model_dir.exists():
        raise FileExistsError(f"Choose a new model directory: {model_dir}")
    frames = load_locked_data(project_dir, data_dir)
    config = json.loads((project_dir / "config/training_config.json").read_text(encoding="utf-8"))
    model = FinalAdsorptionModel(project_dir, model_dir=model_dir)
    _patch_autogluon_metadata_scan()
    from autogluon.tabular import TabularPredictor

    development = pd.concat([frames["train"], frames["valid"]], ignore_index=True)
    predictors = []
    for component, recipe_name, raw in zip(
        model.components, ("baseline", "second"), (development, frames["train"]),
    ):
        recipe = config[recipe_name]
        training = prepare_model_features(raw, project_dir).copy()
        training[LABEL] = raw[LABEL].to_numpy(float)
        predictor = TabularPredictor(
            label=config["label"], problem_type=config["problem_type"],
            eval_metric=config["eval_metric"], path=str(component.model_path),
            verbosity=2, learner_kwargs={"random_state": recipe["seed"]},
        )
        predictor.fit(
            train_data=training, presets=config["presets"],
            hyperparameters=recipe["hyperparameters"],
            num_bag_folds=config["num_bag_folds"], num_bag_sets=config["num_bag_sets"],
            num_stack_levels=config["num_stack_levels"],
            fit_weighted_ensemble=recipe["fit_weighted_ensemble"],
            fit_full_last_level_weighted_ensemble=False,
            full_weighted_ensemble_additionally=False,
            dynamic_stacking=False, calibrate_decision_threshold=False,
            save_bag_folds=True, refit_full=False, set_best_to_refit_full=False,
            num_gpus=0, num_cpus=recipe["num_cpus"],
        )
        if component.model_name not in predictor.model_names():
            raise RuntimeError(f"Expected model was not trained: {component.model_name}")
        predictors.append(predictor)
    model._predictors = predictors
    return model


def score(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("Predictions must be finite and aligned with labels")
    error = y - prediction
    total = np.square(y - y.mean()).sum()
    return {"mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "r2": float(1.0 - np.square(error).sum() / total) if total else float("nan")}


def evaluate_locked_model(
    model: FinalAdsorptionModel, data_dir: Path, output_dir: Path,
    check_reference: bool = False,
) -> dict:
    """Separate live fitted predictions from the historical OOF diagnostic."""
    frames = load_locked_data(model.project_dir, data_dir)
    model.load()
    baseline, second = model._predictors
    base_spec, second_spec = model.components
    base_oof = baseline.predict_oof(model=base_spec.model_name).to_numpy(float)
    second_oof = second.predict_oof(model=second_spec.model_name).to_numpy(float)
    n_train = len(frames["train"])
    n_development = n_train + len(frames["valid"])
    if len(base_oof) != n_development or len(second_oof) != n_train:
        raise ValueError("OOF predictions do not match the locked row counts")

    output, metrics = [], []
    for split in SPLITS:
        raw = frames[split]
        features = prepare_model_features(raw, model.project_dir)
        p_base = baseline.predict(features, model=base_spec.model_name).to_numpy(float)
        p_second = second.predict(features, model=second_spec.model_name).to_numpy(float)
        fitted = base_spec.weight * p_base + second_spec.weight * p_second
        historical_base = (base_oof[:n_train] if split == "train" else
                           base_oof[n_train:] if split == "valid" else p_base)
        diagnostic = base_spec.weight * historical_base + second_spec.weight * p_second
        result = raw[["row_id", "compound_id", LABEL]].copy()
        result["split"] = split
        result["inference_prediction"] = fitted
        result["historical_diagnostic_prediction"] = diagnostic
        result["train_oof_analogue"] = (
            base_spec.weight * historical_base + second_spec.weight * second_oof
            if split == "train" else np.nan
        )
        # The manuscript uses fitted Train and historical diagnostic Valid/Test.
        result["manuscript_prediction"] = fitted if split == "train" else diagnostic
        output.append(result)
        for kind, values in (("inference", fitted), ("historical_diagnostic", diagnostic)):
            metrics.append({"split": split, "kind": kind, "n": len(raw),
                            **score(raw[LABEL].to_numpy(float), values)})
        if split == "train":
            metrics.append({"split": split, "kind": "train_oof_analogue", "n": len(raw),
                            **score(raw[LABEL].to_numpy(float), result.train_oof_analogue.to_numpy())})
    if check_reference:
        reference = json.loads((model.project_dir / "reports/reference_metrics.json").read_text(encoding="utf-8"))
        for item in metrics:
            if item["kind"] == "historical_diagnostic":
                for key in ("mae", "rmse", "r2"):
                    np.testing.assert_allclose(item[key], reference["metrics"][item["split"]][key],
                                               rtol=0, atol=1e-6)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(output, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics).to_csv(output_dir / "metrics.csv", index=False)
    report = {"model_id": model.model_id, "feature_count": len(model.feature_columns),
              "reference_metrics_checked": check_reference, "metrics": metrics,
              "semantics": {
                  "inference": "Both saved components predict normally; Train and Valid include fitted baseline rows.",
                  "historical_diagnostic": "Train: baseline OOF + second fitted; Valid: baseline OOF + second held-out; Test: both held-out.",
                  "train_oof_analogue": "Both OOF arrays combined; fold partitions/training pools differ, so this is not nested CV.",
                  "test": "Compound-disjoint test was inspected before promotion; metrics are descriptive.",
              }}
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
