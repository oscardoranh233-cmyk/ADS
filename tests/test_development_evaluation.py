"""Regression tests for fold means, held-out alignment and fixed-test averaging."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from final_adsorption_model import FinalAdsorptionModel, feature_columns
from final_adsorption_model.development_evaluation import evaluate_development_model
from final_adsorption_model.training import development_folds, development_frame, score


class DevelopmentEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((PROJECT / "config/training_config.json").read_text(encoding="utf-8"))
        raw = pd.DataFrame(1.0, columns=feature_columns(), index=range(688))
        raw["row_id"] = np.arange(688)
        raw["compound_id"] = np.arange(688) // 8
        raw["compound_group_key"] = raw.compound_id.astype(str)
        raw["adsorption"] = np.arange(688) % 29 + 1.0
        self.frames = {"train": raw.iloc[:479].copy(), "valid": raw.iloc[479:616].copy(),
                       "test": raw.iloc[616:].copy()}
        folds = development_folds(development_frame(self.frames), self.config)
        self.folds = folds
        labels = raw.adsorption.to_numpy()
        children = []
        oof = np.empty(616)
        for fold in range(1, 11):
            prediction = labels.copy()
            prediction[:616] += np.where(folds == fold, fold / 2.0, fold / 10.0)
            # Opposing test errors cancel on averaging predictions, but not scores.
            prediction[616:] += fold - 5.5
            oof[folds == fold] = prediction[:616][folds == fold]
            child = SimpleNamespace(name=f"S1F{fold}", predict=lambda data, p=prediction: p[data.index])
            children.append(child)
        bag = SimpleNamespace(models=children, load_child=lambda child: child, n_children=10,
                              _cv_splitters=[SimpleNamespace(groups=pd.Series(folds))])
        self.predictor = SimpleNamespace(
            features=lambda **kwargs: feature_columns(),
            model_names=lambda: ["LightGBMSingle10_BAG_L1"],
            _trainer=SimpleNamespace(load_model=lambda name: bag),
            transform_features=lambda data, model: data,
            predict_oof=lambda model: pd.Series(oof),
            predict=lambda data, model: pd.Series(np.mean([c.predict(data) for c in children], axis=0),
                                                  index=data.index),
        )
        self.model = FinalAdsorptionModel(PROJECT)
        self.model._predictors = [self.predictor]

    def test_reports_arithmetic_fold_means_and_scores_averaged_test_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate_development_model(self.model, self.frames, self.config, Path(directory), False)
            metrics = {row["split"]: row for row in result["metrics"]}
            self.assertEqual(set(metrics), {"train", "valid", "test"})
            for split, expected_rmse in (("train", .55), ("valid", 2.75)):
                self.assertEqual(metrics[split]["n_folds"], 10)
                self.assertAlmostEqual(metrics[split]["rmse"], expected_rmse)
                self.assertAlmostEqual(metrics[split]["mae"], expected_rmse)
            oof = self.predictor.predict_oof(model="unused").to_numpy()
            development = development_frame(self.frames)
            pooled = score(development.adsorption.to_numpy(), oof)
            self.assertNotAlmostEqual(metrics["valid"]["rmse"], pooled["rmse"])
            fold_r2 = [score(development.adsorption.to_numpy()[self.folds == fold],
                             oof[self.folds == fold])["r2"] for fold in range(1, 11)]
            self.assertAlmostEqual(metrics["valid"]["r2"], np.mean(fold_r2))
            self.assertAlmostEqual(metrics["test"]["rmse"], 0.0)
            self.assertAlmostEqual(metrics["test"]["r2"], 1.0)
            predictions = pd.read_csv(Path(directory) / "predictions.csv")
            self.assertEqual(predictions.split.value_counts().to_dict(), {"development": 616, "test": 72})
            self.assertTrue(predictions.loc[predictions.split.eq("test"), "fold"].isna().all())
            self.assertEqual(len(pd.read_csv(Path(directory) / "fold_metrics.csv")), 20)

    def test_reordered_oof_is_rejected(self):
        oof = self.predictor.predict_oof(model="unused")
        self.predictor.predict_oof = lambda model: oof.iloc[::-1]
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "OOF index"):
            evaluate_development_model(self.model, self.frames, self.config, Path(directory), False)

    def test_test_prediction_must_match_fold_mean(self):
        original = self.predictor.predict
        self.predictor.predict = lambda data, model: original(data, model) + 1.0
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(AssertionError, "Test predictions"):
            evaluate_development_model(self.model, self.frames, self.config, Path(directory), False)


if __name__ == "__main__":
    unittest.main()
