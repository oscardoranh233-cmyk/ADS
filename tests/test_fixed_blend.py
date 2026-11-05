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
from final_adsorption_model.training import load_locked_data, train_locked_model


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

    def test_blend_preserves_row_index(self):
        class Predictor:
            def __init__(self, values):
                self.values = values
            def predict(self, data, model):
                if data.columns.tolist() != feature_columns():
                    raise AssertionError("Unexpected model feature columns")
                return pd.Series(self.values, index=data.index)
        model = FinalAdsorptionModel(PROJECT)
        model._predictors = [Predictor([10., 30.]), Predictor([20., 50.])]
        prediction = model.predict(sample())
        np.testing.assert_allclose(prediction, [13.74, 37.48])
        self.assertEqual(prediction.index.tolist(), [8, 3])

    def test_invalid_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(PROJECT / "config", root / "config")
            path = root / "config/model_config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            for weights in ((float("nan"), .374), (-.1, 1.1), (.626001, .374)):
                for item, weight in zip(config["components"], weights):
                    item["weight"] = weight
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.subTest(weights=weights), self.assertRaises(ValueError):
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
        frames = {split: sample().assign(adsorption=[1., 2.]) for split in ("train", "valid", "test")}
        class Predictor:
            def __init__(self, **kwargs):
                self.constructor = kwargs
            def fit(self, **kwargs):
                fitted.append((self.constructor, kwargs))
            def model_names(self):
                return ["LightGBMSingle10_BAG_L1", "WeightedEnsemble_L2"]
        fake = types.ModuleType("autogluon.tabular")
        fake.TabularPredictor = Predictor
        with tempfile.TemporaryDirectory() as directory:
            with patch("final_adsorption_model.training.load_locked_data", return_value=frames), \
                 patch("final_adsorption_model.training._patch_autogluon_metadata_scan"), \
                 patch.dict(sys.modules, {"autogluon.tabular": fake}):
                train_locked_model(PROJECT, Path(directory), Path(directory) / "new")
        self.assertEqual([len(item[1]["train_data"]) for item in fitted], [4, 2])
        self.assertEqual([item[0]["learner_kwargs"]["random_state"] for item in fitted], [0, 986])
        self.assertEqual([item[1]["fit_weighted_ensemble"] for item in fitted], [False, True])
        for _, fit in fitted:
            self.assertEqual(fit["train_data"].columns.tolist(), feature_columns() + ["adsorption"])
            self.assertEqual((fit["num_bag_folds"], fit["num_bag_sets"], fit["num_stack_levels"]), (10, 1, 0))
            self.assertFalse(fit["refit_full"])


if __name__ == "__main__":
    unittest.main()

