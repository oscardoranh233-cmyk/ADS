"""Evaluate ten-fold mean scores and the fixed compound holdout for LightGBM."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .model import FinalAdsorptionModel, prepare_model_features
from .training import LABEL, development_folds, development_frame, score


def _audit_folds(predictor, folds: np.ndarray, feature_columns: list[str], model_name: str) -> dict[str, int]:
    if predictor.features(feature_stage="original") != feature_columns:
        raise ValueError("Fitted predictor does not match the locked feature contract")
    if predictor.model_names() != [model_name]:
        raise ValueError("Expected only the configured standalone LightGBM")
    bag = predictor._trainer.load_model(model_name)
    if len(bag._cv_splitters) != 1 or bag.n_children != len(np.unique(folds)):
        raise ValueError(f"Expected one child per fold in {model_name}")
    np.testing.assert_array_equal(bag._cv_splitters[0].groups.to_numpy(), folds)
    return {model_name: bag.n_children}


def _single_predictor_fold_scores(
    predictor, model_name: str, role: str, features: pd.DataFrame,
    development: pd.DataFrame, folds: np.ndarray, oof: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    bag = predictor._trainer.load_model(model_name)
    transformed = predictor.transform_features(features, model=model_name)
    labels = development[LABEL].to_numpy(dtype=float)
    compounds = development["compound_group_key"]
    result, test_predictions = [], []
    seen_folds = set()
    for child_ref in bag.models:
        child = bag.load_child(child_ref)
        match = re.fullmatch(r"S1F(\d+)", child.name)
        if not match:
            raise ValueError(f"Unexpected fold model name: {child.name}")
        fold = int(match.group(1))
        if fold in seen_folds or not np.any(folds == fold):
            raise ValueError(f"Unexpected or duplicate fold: {fold}")
        seen_folds.add(fold)
        valid = folds == fold
        all_predictions = np.asarray(child.predict(transformed), dtype=float)
        prediction = all_predictions[:len(development)]
        test_predictions.append(all_predictions[len(development):])
        np.testing.assert_allclose(prediction[valid], oof[valid], atol=2e-5, rtol=0)
        overlap = len(set(compounds[valid]) & set(compounds[~valid]))
        for split, mask in (("train", ~valid), ("valid", valid)):
            result.append({
                "fold": fold, "split": split, "model": role, "n": int(mask.sum()),
                "compound_groups_overlapping_training": overlap,
                **score(labels[mask], prediction[mask]),
            })
    if seen_folds != set(folds):
        raise ValueError("Saved fold models do not cover the development folds")
    return result, np.mean(test_predictions, axis=0)


def evaluate_development_model(
    model: FinalAdsorptionModel, frames: dict[str, pd.DataFrame], config: dict,
    output_dir: Path, check_reference: bool,
) -> dict:
    development = development_frame(frames)
    folds = development_folds(development, config)
    test = frames["test"].assign(split="test")
    raw = pd.concat([development, test], ignore_index=True)
    data = prepare_model_features(raw, model.project_dir)
    model.load()
    component, = model.components
    predictor, = model._predictors
    counts = _audit_folds(predictor, folds, model.feature_columns, component.model_name)
    oof = predictor.predict_oof(model=component.model_name)
    if not oof.index.equals(development.index) or not np.isfinite(oof).all():
        raise ValueError("OOF index or predictions do not match the locked development data")
    inference = model.predict(raw)
    predictions = raw[["row_id", "compound_id", "compound_group_key", "split", LABEL]].copy()
    predictions["fold"] = pd.Series(folds, dtype="Int64")
    predictions[f"{component.role}_oof"] = oof.astype(float)
    predictions[f"{component.role}_prediction"] = inference.to_numpy(float)
    fold_metrics, test_child_mean = _single_predictor_fold_scores(
        predictor, component.model_name, component.role, data, development, folds, oof.to_numpy(float),
    )
    test_prediction = inference.iloc[len(development):].to_numpy(float)
    np.testing.assert_allclose(test_prediction, test_child_mean, atol=2e-5, rtol=0,
                               err_msg="Test predictions must equal the mean of ten fold predictions")
    metrics = []
    for split in ("train", "valid"):
        selected = [row for row in fold_metrics if row["split"] == split]
        metrics.append({
            "split": split, "kind": "fold_mean", "model": component.role,
            "n_folds": len(selected),
            "n_min": min(row["n"] for row in selected),
            "n_max": max(row["n"] for row in selected),
            **{key: float(np.mean([row[key] for row in selected]))
               for key in ("mae", "rmse", "r2")},
        })
    metrics.append({"split": "test", "kind": "test", "model": component.role, "n": len(test),
                    **score(test[LABEL].to_numpy(float), test_prediction)})
    if check_reference:
        path = model.project_dir / "reports/reference_metrics.json"
        reference = json.loads(path.read_text(encoding="utf-8"))
        if reference["model_id"] != model.model_id:
            raise ValueError("Reference model ID differs from the current default")
        actual = {(row["split"], row["kind"], row["model"]): row for row in metrics}
        expected = {(row["split"], row["kind"], row["model"]): row for row in reference["metrics"]}
        if actual.keys() != expected.keys():
            raise ValueError("Reference must contain exactly the fold means and the fixed-test metrics")
        for key, row in expected.items():
            np.testing.assert_allclose([actual[key][metric] for metric in ("mae", "rmse", "r2")],
                                       [row[metric] for metric in ("mae", "rmse", "r2")],
                                       rtol=0, atol=1e-6, err_msg=str(key))
    report = {
        "model_id": model.model_id, "feature_count": len(model.feature_columns),
        "reference_metrics_checked": check_reference, "fold_seed": config["development_cv_seed"],
        "development_rows": len(development), "test_rows": len(test),
        "development_compounds": int(development.compound_group_key.nunique()),
        "test_compounds": int(test.compound_group_key.nunique()),
        "components": {component.role: {"training_rows": len(oof), "bag_children": counts}},
        "test_prediction_vs_fold_mean_max_abs_error": float(np.max(np.abs(test_prediction - test_child_mean))),
        "metrics": metrics,
        "semantics": {
            "oof": "Row-wise ten-fold OOF predictions with validation-fold early stopping. Pooled OOF scores are not the reported validation fold means.",
            "test": "Metrics of the averaged fold-model predictions on the fixed compound-disjoint test set, which was previously inspected.",
            "fold_mean": "Arithmetic mean of ten separately computed fold scores. Train uses the nine fitting folds and Valid uses the held-out fold of each model.",
            "predictions": "Development rows have held-out OOF predictions and direct ten-fold inference predictions; only the former represent held-out predictions. Test rows have direct ten-fold inference predictions and no OOF assignment.",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics).to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(output_dir / "fold_metrics.csv", index=False)
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
