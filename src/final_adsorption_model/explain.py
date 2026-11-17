"""TreeSHAP utilities used to explain the default ten-fold LightGBM."""
from __future__ import annotations
from functools import lru_cache
import numpy as np
import pandas as pd
import shap
from . import model as default_model
from .model import _patch_autogluon_metadata_scan

ADDITIVITY_TOLERANCE = 2e-5
CHILD_ADDITIVITY_TOLERANCE = 1e-4

@lru_cache(maxsize=2)
def load_predictor(path: str):
    from autogluon.tabular import TabularPredictor
    _patch_autogluon_metadata_scan()
    return TabularPredictor.load(path, require_version_match=True,
                                require_py_version_match=True, check_packages=False)

def numeric_feature_values(series: pd.Series) -> np.ndarray:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.cat.codes.to_numpy(dtype=float)
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(int).to_numpy(dtype=float)
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.isnan(values).any():
        finite = values[np.isfinite(values)]
        fill = float(np.median(finite)) if finite.size else 0.0
        values = np.nan_to_num(values, nan=fill)
    return values

def pretty_feature_name(name: str) -> str:
    exact = {
        default_model.SURFACE: "Resin surface area",
        default_model.PORE: "Mean pore diameter",
        default_model.TRUE_DENSITY: "Wet true density",
        default_model.APPARENT_DENSITY: "Wet apparent density",
        default_model.BULK_DENSITY: "Bulk density",
        default_model.WATER: "Water content",
        "resin_polarity_code": "Resin polarity code",
        "resin_density_gap": "Resin density gap",
        "resin_porosity_proxy": "Resin porosity proxy",
        "resin_packing_ratio": "Resin packing ratio",
        "resin_surface_x_pore": "Surface area × pore diameter",
        "resin_surface_per_pore": "Surface area / pore diameter",
        "resin_water_x_surface": "Water content × surface area",
        "resin_surface_x_SASA_3D": "Surface area × SASA (3D)",
        "resin_surface_x_MolMR": "Surface area × MolMR",
        "resin_pore_per_cuberoot_MolWt": "Pore diameter / cuberoot(MolWt)",
        "resin_pore_per_sqrt_SASA": "Pore diameter / sqrt(SASA)",
        "resin_polarity_x_MolLogP_targeted": "Resin polarity × MolLogP",
        "resin_polarity_x_TPSA_targeted": "Resin polarity × TPSA",
        "resin_polarity_x_MinCharge_targeted": "Resin polarity × minimum charge",
        "resin_water_x_MolLogP": "Water content × MolLogP",
        "resin_water_x_TPSA": "Water content × TPSA",
        "resin_profile_category": "Resin profile category",
    }
    if name in exact:
        return exact[name]
    if name.startswith("resin_polarity_"):
        return "Resin polarity category"
    if name.isascii():
        return name.replace("_", " ")
    return "Resin property"

def expand_leaf_components(config: dict) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    leaves: list[dict[str, object]] = []
    nesting_audit: list[dict[str, object]] = []
    for component in config["components"]:
        predictor_path = str(component["predictor_path"])
        model_name = str(component["model_name"])
        outer_weight = float(component["weight"])
        predictor = load_predictor(predictor_path)
        model = predictor._trainer.load_model(model_name)
        if model.__class__.__name__ == "WeightedEnsembleModel":
            internal_weights = {
                str(name): float(weight) for name, weight in model._get_model_weights().items()
            }
            if not np.isclose(sum(internal_weights.values()), 1.0, atol=1e-12):
                raise RuntimeError("WeightedEnsemble_L2 internal weights do not sum to one")
            nesting_audit.append(
                {
                    "role": component["role"],
                    "model_name": model_name,
                    "outer_weight": outer_weight,
                    "internal_weights": internal_weights,
                }
            )
            for base_name, internal_weight in internal_weights.items():
                leaves.append(
                    {
                        "role": f"{component['role']}/{base_name}",
                        "predictor_path": predictor_path,
                        "model_name": base_name,
                        "outer_weight": outer_weight,
                        "internal_weight": internal_weight,
                        "effective_weight": outer_weight * internal_weight,
                    }
                )
        else:
            nesting_audit.append(
                {
                    "role": component["role"],
                    "model_name": model_name,
                    "outer_weight": outer_weight,
                }
            )
            leaves.append(
                {
                    "role": component["role"],
                    "predictor_path": predictor_path,
                    "model_name": model_name,
                    "outer_weight": outer_weight,
                    "internal_weight": 1.0,
                    "effective_weight": outer_weight,
                }
            )
    if not np.isclose(sum(float(leaf["effective_weight"]) for leaf in leaves), 1.0, atol=1e-12):
        raise RuntimeError("Expanded leaf-model weights do not sum to one")
    return leaves, nesting_audit

def _child_input_and_columns(
    child, transformed: pd.DataFrame
) -> tuple[object, list[str], list[str]]:
    child_input = child.preprocess(transformed)
    if isinstance(child_input, pd.DataFrame):
        columns = child_input.columns.astype(str).tolist()
        original_columns = columns.copy()
    else:
        shape = getattr(child_input, "shape", None)
        if shape is None or len(shape) != 2:
            raise RuntimeError(
                f"Unexpected preprocessed input type for {child.__class__.__name__}: "
                f"{type(child_input).__name__}"
            )
        width = int(shape[1])
        if width == len(child.features):
            columns = [str(name) for name in child.features]
            original_columns = columns.copy()
        elif getattr(child, "_ohe_generator", None) is not None:
            generator = child._ohe_generator
            columns = [str(name) for name in generator.get_feature_names()]
            if len(columns) != width:
                raise RuntimeError(
                    f"OHE feature map has {len(columns)} columns but model input has {width}"
                )
            category_columns = [str(name) for name in generator.cat_cols]
            original_columns = []
            for encoded_name in columns:
                original_name = encoded_name
                for category_name in category_columns:
                    if encoded_name == category_name or encoded_name.startswith(category_name + "_"):
                        original_name = category_name
                        break
                original_columns.append(original_name)
            if set(original_columns) != set(str(name) for name in child.features):
                raise RuntimeError("OHE columns cannot be mapped back to the original model features")
        else:
            raise RuntimeError(f"Cannot map {width} child features to model feature names")
    return child_input, columns, original_columns

def tree_shap_for_leaf(
    leaf: dict[str, object], input_data: pd.DataFrame
) -> tuple[np.ndarray, float, np.ndarray, pd.DataFrame, dict[str, object]]:
    predictor = load_predictor(str(leaf["predictor_path"]))
    model_name = str(leaf["model_name"])
    transformed = predictor.transform_features(input_data, model=model_name)
    canonical_columns = transformed.columns.astype(str).tolist()
    bag = predictor._trainer.load_model(model_name)
    child_refs = list(bag.models)
    if not child_refs:
        raise RuntimeError(f"No child models found for {model_name}")

    shap_sum = np.zeros((len(input_data), len(canonical_columns)), dtype=float)
    pred_sum = np.zeros(len(input_data), dtype=float)
    expected_sum = 0.0
    max_child_additivity = 0.0
    child_types: list[str] = []
    shap_methods: list[str] = []
    for child_ref in child_refs:
        child = bag.load_child(child_ref)
        child_input, child_columns, original_columns = _child_input_and_columns(child, transformed)
        if set(original_columns) != set(canonical_columns):
            raise RuntimeError(f"Feature space changed inside {model_name}")
        if child.model.__class__.__name__ == "XGBRegressor":
            import xgboost as xgb

            best_iteration = getattr(child.model, "best_iteration", None)
            iteration_range = (
                (0, int(best_iteration) + 1) if best_iteration is not None else (0, 0)
            )
            contribution_matrix = child.model.get_booster().predict(
                xgb.DMatrix(child_input),
                pred_contribs=True,
                iteration_range=iteration_range,
            )
            values = np.asarray(contribution_matrix[:, :-1], dtype=float)
            expected = float(np.asarray(contribution_matrix[:, -1]).reshape(-1)[0])
            shap_method = "xgboost_native_pred_contribs"
        else:
            explainer = shap.TreeExplainer(
                child.model, feature_perturbation="tree_path_dependent"
            )
            values = explainer.shap_values(child_input)
            if isinstance(values, list):
                values = values[0]
            values = np.asarray(values, dtype=float)
            if values.ndim == 3 and values.shape[-1] == 1:
                values = values[:, :, 0]
            expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
            shap_method = "shap_tree_explainer_tree_path_dependent"
        if values.shape != (len(input_data), len(child_columns)):
            raise RuntimeError(
                f"Unexpected TreeSHAP shape for {model_name}: {values.shape}"
            )
        aligned = np.zeros((len(input_data), len(canonical_columns)), dtype=float)
        canonical_index = {name: index for index, name in enumerate(canonical_columns)}
        for encoded_index, original_name in enumerate(original_columns):
            aligned[:, canonical_index[original_name]] += values[:, encoded_index]
        child_prediction = np.asarray(child.model.predict(child_input), dtype=float).reshape(-1)
        reconstructed = expected + values.sum(axis=1)
        error = float(np.max(np.abs(reconstructed - child_prediction)))
        if error > CHILD_ADDITIVITY_TOLERANCE:
            raise RuntimeError(f"TreeSHAP additivity failed for {model_name}: {error}")
        shap_sum += aligned
        pred_sum += child_prediction
        expected_sum += expected
        max_child_additivity = max(max_child_additivity, error)
        child_types.append(child.__class__.__name__)
        shap_methods.append(shap_method)

    divisor = float(len(child_refs))
    shap_mean = shap_sum / divisor
    pred_mean = pred_sum / divisor
    expected_mean = expected_sum / divisor
    api_prediction = predictor.predict(input_data, model=model_name).to_numpy(dtype=float)
    bag_error = float(np.max(np.abs(api_prediction - pred_mean)))
    if bag_error > ADDITIVITY_TOLERANCE:
        raise RuntimeError(f"Child mean differs from bag prediction for {model_name}: {bag_error}")
    qa = {
        "role": leaf["role"],
        "model_name": model_name,
        "model_type": bag.__class__.__name__,
        "child_model_types": sorted(set(child_types)),
        "shap_methods": sorted(set(shap_methods)),
        "outer_weight": float(leaf["outer_weight"]),
        "internal_weight": float(leaf["internal_weight"]),
        "effective_weight": float(leaf["effective_weight"]),
        "n_child_models": len(child_refs),
        "max_child_additivity_error": max_child_additivity,
        "max_bag_prediction_vs_child_mean_error": bag_error,
    }
    return shap_mean, expected_mean, pred_mean, transformed, qa
