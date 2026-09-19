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
    "resin_polarity_中极性",
    "resin_polarity_弱极性",
    "resin_polarity_极性",
    "resin_polarity_非极性",
]


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


def prepare_model_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add the locked resin-derived features used by both final components.

    The input must already contain the molecular descriptors and MACCS keys used
    in training, together with the raw resin columns listed in ``RESIN_COLUMNS``.
    This function does not calculate molecular descriptors from SMILES.
    """
    frame = data.copy()
    missing = sorted(set(RESIN_COLUMNS) - set(frame.columns))
    if missing:
        raise KeyError(f"Missing raw resin columns: {missing}")

    frame["resin_density_gap"] = frame[TRUE_DENSITY] - frame[APPARENT_DENSITY]
    frame["resin_porosity_proxy"] = 1.0 - frame[APPARENT_DENSITY] / (frame[TRUE_DENSITY] + EPSILON)
    frame["resin_packing_ratio"] = frame[BULK_DENSITY] / (frame[APPARENT_DENSITY] + EPSILON)
    frame["resin_surface_x_pore"] = frame[SURFACE] * frame[PORE]
    frame["resin_surface_per_pore"] = frame[SURFACE] / (frame[PORE] + EPSILON)
    frame["resin_water_x_surface"] = frame[WATER] * frame[SURFACE]

    for descriptor in ("MolLogP", "SASA_3D", "MolMR", "MolWt", "TPSA", "Min_Charge"):
        if descriptor not in frame.columns:
            raise KeyError(f"Missing molecular descriptor required for feature engineering: {descriptor}")
    frame["resin_surface_x_MolLogP"] = frame[SURFACE] * frame["MolLogP"]
    frame["resin_surface_x_SASA_3D"] = frame[SURFACE] * frame["SASA_3D"]
    frame["resin_surface_x_MolMR"] = frame[SURFACE] * frame["MolMR"]
    frame["resin_pore_per_cuberoot_MolWt"] = frame[PORE] / (np.cbrt(frame["MolWt"].clip(lower=0.0)) + EPSILON)
    frame["resin_pore_per_sqrt_SASA"] = frame[PORE] / (np.sqrt(frame["SASA_3D"].clip(lower=0.0)) + EPSILON)
    frame["resin_polarity_x_MolLogP_targeted"] = frame[POLARITY] * frame["MolLogP"]
    frame["resin_polarity_x_TPSA_targeted"] = frame[POLARITY] * frame["TPSA"]
    frame["resin_polarity_x_MinCharge_targeted"] = frame[POLARITY] * frame["Min_Charge"]
    frame["resin_water_x_MolLogP"] = frame[WATER] * frame["MolLogP"]
    frame["resin_water_x_TPSA"] = frame[WATER] * frame["TPSA"]
    frame["resin_profile_category"] = frame[RESIN_COLUMNS].astype(str).agg("|".join, axis=1).astype("category")
    return frame


@dataclass(frozen=True)
class ComponentSpec:
    role: str
    model_path: Path
    model_name: str
    weight: float


class FinalAdsorptionModel:
    """Locked prediction blend: 0.626 LightGBM + 0.374 WeightedEnsemble_L2."""

    def __init__(self, project_dir: str | Path) -> None:
        self.project_dir = Path(project_dir).resolve()
        config_path = self.project_dir / "config" / "model_config.json"
        config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        self.model_id = str(config["model_id"])
        self.target = str(config["target"])
        self.components = [
            ComponentSpec(
                role=str(item["role"]),
                model_path=self.project_dir / str(item["model_path"]),
                model_name=str(item["model_name"]),
                weight=float(item["weight"]),
            )
            for item in config["components"]
        ]
        weights = np.asarray([item.weight for item in self.components], dtype=float)
        if len(self.components) != 2 or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-12):
            raise ValueError("The final model requires exactly two non-negative weights summing to 1.0")
        self._predictors: list[Any] | None = None

    def load(self) -> "FinalAdsorptionModel":
        if self._predictors is not None:
            return self
        _patch_autogluon_metadata_scan()
        from autogluon.tabular import TabularPredictor

        missing = [str(item.model_path) for item in self.components if not item.model_path.is_dir()]
        if missing:
            raise FileNotFoundError("Missing local AutoGluon model artifacts: " + ", ".join(missing))
        self._predictors = [
            TabularPredictor.load(
                str(item.model_path),
                require_version_match=False,
                require_py_version_match=False,
                check_packages=False,
            )
            for item in self.components
        ]
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Return adsorption-capacity predictions in mg g^-1."""
        self.load()
        assert self._predictors is not None
        features = prepare_model_features(data)
        prediction = np.zeros(len(features), dtype=float)
        for predictor, component in zip(self._predictors, self.components, strict=True):
            prediction += component.weight * predictor.predict(features, model=component.model_name).to_numpy(dtype=float)
        return pd.Series(prediction, index=data.index, name="predicted_adsorption_capacity_mg_per_g")

