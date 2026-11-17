from __future__ import annotations
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from final_adsorption_model import FinalAdsorptionModel, feature_columns, prepare_model_features
from final_adsorption_model.model import SURFACE, PORE, POLARITY, WATER
from final_adsorption_model.training import FOLD_COLUMN, load_locked_data, train_locked_model


def sample() -> pd.DataFrame:
    frame = pd.DataFrame(1.0, columns=feature_columns(), index=[8, 3])
    frame[SURFACE] = [400., 600.]
    frame[PORE] = [100., 200.]
    frame[POLARITY] = [0., 2.]
    frame[WATER] = [50., 60.]
    frame["SASA_3D"] = [100., 400.]
    frame["MolWt"] = [125., 1000.]
    frame["MolLogP"] = [2., -1.]
    return frame


class FinalModelTests(unittest.TestCase):
    def test_feature_contract_excludes_legacy_and_target(self):
        raw = sample()
        config = json.loads((PROJECT / "config/model_config.json").read_text(encoding="utf-8"))
        for column in config["excluded_features"] + ["adsorption", "row_id"]:
            raw[column] = "must not reach the model"
        original = raw.copy(deep=True)
        prepared = prepare_model_features(raw[raw.columns[::-1]])
        self.assertEqual(prepared.columns.tolist(), feature_columns())
        self.assertEqual(prepared.shape, (2, 125))
        pd.testing.assert_frame_equal(raw, original)
        np.testing.assert_allclose(prepared["resin_surface_x_SASA_3D"], [40000., 240000.])
        np.testing.assert_allclose(prepared["resin_pore_per_cuberoot_MolWt"], [20., 20.])
        np.testing.assert_allclose(prepared["resin_polarity_x_MolLogP_targeted"], [0., -2.])

    def test_missing_descriptor_or_fingerprint_is_rejected(self):
        for column in ("MolLogP", "maccs_104"):
            with self.subTest(column=column), self.assertRaises(KeyError):
                prepare_model_features(sample().drop(columns=column))

    def test_infinity_and_duplicate_columns_are_rejected(self):
        raw = sample()
        raw.loc[8, "MolLogP"] = np.inf
        with self.assertRaises(ValueError):
            prepare_model_features(raw)
        with self.assertRaises(ValueError):
            prepare_model_features(pd.concat([sample(), sample()[["MolLogP"]]], axis=1))

    def test_prediction_preserves_row_index(self):
        class Predictor:
            def __init__(self, values):
                self.values = values
            def predict(self, data, model):
                if data.columns.tolist() != feature_columns():
                    raise AssertionError("Unexpected model feature columns")
                return pd.Series(self.values, index=data.index)
        model = FinalAdsorptionModel(PROJECT)
        model._predictors = [Predictor([10., 30.])]
        prediction = model.predict(sample())
        np.testing.assert_allclose(prediction, [10., 30.])
        self.assertEqual(prediction.index.tolist(), [8, 3])

    def test_two_predictor_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(PROJECT / "config", root / "config")
            path = root / "config/model_config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            component = config["components"][0]
            config["components"] = [{**component, "weight": 0.75},
                                    {**component, "role": "second", "weight": 0.25}]
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one LightGBM predictor"):
                FinalAdsorptionModel(root)

    def test_invalid_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(PROJECT / "config", root / "config")
            path = root / "config/model_config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            for weight in (float("nan"), -.1, 1.000001):
                config["components"][0]["weight"] = weight
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.subTest(weight=weight), self.assertRaises(ValueError):
                    FinalAdsorptionModel(root)

    def test_missing_models_report_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            model = FinalAdsorptionModel(PROJECT, model_dir=directory)
            with self.assertRaisesRegex(FileNotFoundError, "Missing local AutoGluon"):
                model.load()

    def test_changed_training_data_is_rejected_before_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "train_model_rows.csv").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_locked_data(PROJECT, path)

    def test_training_never_overwrites_existing_models(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                train_locked_model(PROJECT, PROJECT / "data/local", Path(directory))

    def test_training_uses_locked_pools_and_fit_settings(self):
        import types
        fitted = []
        frames = {split: pd.concat([sample()] * ((n + 1) // 2), ignore_index=True).iloc[:n]
                  .assign(adsorption=label)
                  for split, n, label in (("train", 479, 1.), ("valid", 137, 2.), ("test", 72, 3.))}
        class Predictor:
            def __init__(self, **kwargs):
                self.constructor = kwargs
            def fit(self, **kwargs):
                fitted.append((self.constructor, kwargs))
            def model_names(self):
                return ["LightGBMSingle10_BAG_L1"]
        fake = types.ModuleType("autogluon.tabular")
        fake.TabularPredictor = Predictor
        with tempfile.TemporaryDirectory() as directory:
            with patch("final_adsorption_model.training.load_locked_data", return_value=frames), \
                 patch("final_adsorption_model.training._patch_autogluon_metadata_scan"), \
                 patch.dict(sys.modules, {"autogluon.tabular": fake}):
                train_locked_model(PROJECT, Path(directory), Path(directory) / "new")
        self.assertEqual(len(fitted), 1)
        (constructor, fit), = fitted
        training = fit["train_data"]
        self.assertEqual(len(training), 616)
        self.assertEqual(constructor["learner_kwargs"]["random_state"], 0)
        self.assertEqual(constructor["groups"], FOLD_COLUMN)
        self.assertEqual(training.columns.tolist(), feature_columns() + ["adsorption", FOLD_COLUMN])
        self.assertEqual(training[FOLD_COLUMN].value_counts().sort_values().tolist(), [61] * 4 + [62] * 6)
        self.assertEqual(training.adsorption.value_counts().to_dict(), {1.: 479, 2.: 137})
        self.assertEqual((fit["num_bag_folds"], fit["num_bag_sets"], fit["num_stack_levels"]), (10, 1, 0))
        self.assertEqual(fit["hyperparameters"]["GBM"][0]["min_data_in_leaf"], 6)
        self.assertFalse(fit["fit_weighted_ensemble"])
        self.assertFalse(fit["refit_full"])


if __name__ == "__main__":
    unittest.main()

