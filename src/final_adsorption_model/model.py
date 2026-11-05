from __future__ import annotations

import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EPSILON = 1e-8
SURFACE = "树脂比表面积m2/g_mean"
PORE = "平均孔径Å_mean"
TRUE_DENSITY = "湿真密度g/ml_mean"
APPARENT_DENSITY = "湿视密度g/ml_mean"
BULK_DENSITY = "堆积密度mg/ml_mean"
WATER = "含水率%_mean"
POLARITY = "resin_polarity_code"
RESIN_COLUMNS = [
    SURFACE,
    PORE,
    TRUE_DENSITY,
    APPARENT_DENSITY,
    BULK_DENSITY,
    WATER,
    POLARITY,
]
PROJECT_DIR = Path(__file__).resolve().parents[2]


def feature_columns(project_dir: str | Path = PROJECT_DIR) -> list[str]:
    """Return the ordered, locked 125-feature input contract."""
    path = Path(project_dir) / "config" / "feature_columns_125.json"
    columns = json.loads(path.read_text(encoding="utf-8"))
    if len(columns) != 125 or len(set(columns)) != 125:
        raise ValueError("The final feature manifest must contain 125 unique columns")
    return columns


def _patch_autogluon_metadata_scan() -> None:
    """Avoid an environment-specific package metadata failure during model loading."""
    from autogluon.common.utils import utils as ag_utils

    def safe_get_package_versions() -> dict[str, str]:
        versions: dict[str, str] = {}
        for distribution in importlib.metadata.distributions():
            name = distribution.metadata.get("Name")
            if name:
                versions[name.lower()] = distribution.version
        return versions

    ag_utils.get_package_versions = safe_get_package_versions


def prepare_model_features(
    data: pd.DataFrame, project_dir: str | Path = PROJECT_DIR,
) -> pd.DataFrame:
    """Add the locked resin-derived features used by both final components.

    The input must already contain the molecular descriptors and MACCS keys used
    in training, together with the raw resin columns listed in ``RESIN_COLUMNS``.
    This function does not calculate molecular descriptors from SMILES.
    """
    if not data.columns.is_unique:
        raise ValueError("Input column names must be unique")
    frame = data.copy()
    missing = sorted(set(RESIN_COLUMNS) - set(frame.columns))
    if missing:
        raise KeyError(f"Missing raw resin columns: {missing}")

    for descriptor in ("MolLogP", "SASA_3D", "MolMR", "MolWt", "TPSA", "Min_Charge"):
        if descriptor not in frame.columns:
            raise KeyError(f"Missing molecular descriptor required for feature engineering: {descriptor}")
    frame["resin_surface_x_SASA_3D"] = frame[SURFACE] * frame["SASA_3D"]
    frame["resin_surface_x_MolMR"] = frame[SURFACE] * frame["MolMR"]
    frame["resin_pore_per_cuberoot_MolWt"] = frame[PORE] / (np.cbrt(frame["MolWt"].clip(lower=0.0)) + EPSILON)
    frame["resin_pore_per_sqrt_SASA"] = frame[PORE] / (np.sqrt(frame["SASA_3D"].clip(lower=0.0)) + EPSILON)
    frame["resin_polarity_x_MolLogP_targeted"] = frame[POLARITY] * frame["MolLogP"]
    frame["resin_polarity_x_TPSA_targeted"] = frame[POLARITY] * frame["TPSA"]
    frame["resin_polarity_x_MinCharge_targeted"] = frame[POLARITY] * frame["Min_Charge"]
    frame["resin_water_x_MolLogP"] = frame[WATER] * frame["MolLogP"]
    frame["resin_water_x_TPSA"] = frame[WATER] * frame["TPSA"]
    columns = feature_columns(project_dir)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"Missing model features: {missing}")
    features = frame.loc[:, columns]
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(features[c])]
    if non_numeric:
        raise ValueError(f"Model features must be numeric: {non_numeric}")
    # Missing values are handled by the saved AutoGluon preprocessing pipeline.
    if np.isinf(features.to_numpy(dtype=float)).any():
        raise ValueError("Model features must not contain infinity")
    return features


@dataclass(frozen=True)
class ComponentSpec:
    role: str
    model_path: Path
    model_name: str
    weight: float


class FinalAdsorptionModel:
    """Locked prediction blend: 0.626 LightGBM + 0.374 WeightedEnsemble_L2."""

    def __init__(
        self, project_dir: str | Path = PROJECT_DIR,
        model_dir: str | Path | None = None,
    ) -> None:
        self.project_dir = Path(project_dir).resolve()
        config_path = self.project_dir / "config" / "model_config.json"
        config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        self.model_id = str(config["model_id"])
        self.target = str(config["target"])
        self.feature_columns = feature_columns(self.project_dir)
        if config["feature_count"] != len(self.feature_columns):
            raise ValueError("Configuration and feature manifest disagree")
        self.components = [
            ComponentSpec(
                role=str(item["role"]),
                model_path=(Path(model_dir).resolve() / Path(item["model_path"]).name
                            if model_dir is not None
                            else self.project_dir / str(item["model_path"])),
                model_name=str(item["model_name"]),
                weight=float(item["weight"]),
            )
            for item in config["components"]
        ]
        weights = np.asarray([item.weight for item in self.components], dtype=float)
        if (len(self.components) != 2 or not np.isfinite(weights).all()
                or np.any(weights < 0)
                or not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=0)):
            raise ValueError("The final model requires exactly two non-negative weights summing to 1.0")
        self._predictors: list[Any] | None = None

    def load(self) -> "FinalAdsorptionModel":
        if self._predictors is not None:
            return self
        missing = [str(item.model_path) for item in self.components
                   if not (item.model_path / "predictor.pkl").is_file()]
        if missing:
            raise FileNotFoundError("Missing local AutoGluon model artifacts: " + ", ".join(missing))
        _patch_autogluon_metadata_scan()
        from autogluon.tabular import TabularPredictor

        predictors = [
            TabularPredictor.load(
                str(item.model_path),
                require_version_match=True,
                require_py_version_match=True,
                check_packages=False,
            )
            for item in self.components
        ]
        for predictor, component in zip(predictors, self.components):
            if predictor.features(feature_stage="original") != self.feature_columns:
                raise ValueError(f"Artifact feature contract mismatch: {component.role}")
            if component.model_name not in predictor.model_names():
                raise ValueError(f"Missing fitted component: {component.model_name}")
        self._predictors = predictors
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Return adsorption-capacity predictions in mg g^-1."""
        features = prepare_model_features(data, self.project_dir)
        if features.empty:
            return pd.Series(index=data.index, dtype=float,
                             name="predicted_adsorption_capacity_mg_per_g")
        self.load()
        assert self._predictors is not None
        prediction = np.zeros(len(features), dtype=float)
        for predictor, component in zip(self._predictors, self.components):
            prediction += component.weight * predictor.predict(features, model=component.model_name).to_numpy(dtype=float)
        if not np.isfinite(prediction).all():
            raise ValueError("Model returned non-finite predictions")
        return pd.Series(prediction, index=data.index, name="predicted_adsorption_capacity_mg_per_g")

