"""Export fold-averaged LightGBM SHAP values on the 616 development profiles."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
from final_adsorption_model import FinalAdsorptionModel, prepare_model_features
from final_adsorption_model.training import development_frame, load_locked_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate TreeSHAP averaged across the ten LightGBM fold models.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data/local")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs/shap")
    args = parser.parse_args()
    from final_adsorption_model.explain import (
        ADDITIVITY_TOLERANCE, expand_leaf_components, tree_shap_for_leaf, pretty_feature_name,
    )

    frames = load_locked_data(PROJECT_DIR, args.data_dir)
    raw = development_frame(frames)
    features = prepare_model_features(raw, PROJECT_DIR)
    model = FinalAdsorptionModel(PROJECT_DIR).load()
    config = {"components": [{"role": c.role, "predictor_path": str(c.model_path),
                               "model_name": c.model_name, "weight": c.weight}
                              for c in model.components]}
    leaves, nesting = expand_leaf_components(config)
    values = np.zeros(features.shape, dtype=float)
    expected, audits = 0.0, []
    for leaf in leaves:
        print(f"Computing SHAP: {leaf['model_name']}", flush=True)
        part, base, _, transformed, audit = tree_shap_for_leaf(leaf, features)
        columns = transformed.columns.tolist()
        if set(columns) != set(features.columns):
            raise ValueError("SHAP feature space differs from the final 125-feature contract")
        positions = [columns.index(c) for c in features.columns]
        values += float(leaf["effective_weight"]) * part[:, positions]
        expected += float(leaf["effective_weight"]) * base
        audits.append(audit)
    prediction = model.predict(raw).to_numpy(float)
    error = float(np.max(np.abs(expected + values.sum(axis=1) - prediction)))
    if error > ADDITIVITY_TOLERANCE:
        raise ValueError(f"LightGBM SHAP additivity failed: {error}")
    importance = pd.DataFrame({"feature": features.columns,
                               "display_name": [pretty_feature_name(c) for c in features.columns],
                               "mean_abs_shap": np.abs(values).mean(axis=0),
                               "max_abs_shap": np.abs(values).max(axis=0)})
    importance = importance.sort_values(["mean_abs_shap", "feature"], ascending=[False, True]).reset_index(drop=True)
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    importance["importance_share"] = importance.mean_abs_shap / importance.mean_abs_shap.sum()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(args.output_dir / "importance.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(args.output_dir / "shap_values.npz", values=values,
                        numeric=features.to_numpy(float), columns=np.array(features.columns.tolist(), dtype=str),
                        row_id=raw.row_id.to_numpy(), expected_value=expected)
    report = {"model_id": model.model_id, "n_rows": len(raw), "n_features": len(features.columns),
              "semantics": "Direct inference on the 616 development records, with TreeSHAP averaged over ten fold models. Test excluded from ranking.",
              "weighted_expected_value": expected, "max_additivity_error": error,
              "top_level_components": nesting, "expanded_tree_components": audits}
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"SHAP complete; maximum additivity error: {error:.8g}")


if __name__ == "__main__":
    main()
