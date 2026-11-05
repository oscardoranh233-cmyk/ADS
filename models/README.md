# Confirmed local model artifacts

These are the **125-feature models promoted on 2026-11-05**. The former 136-feature artifacts are incompatible and are rejected by the feature-contract check.

| Local destination | Source below the original research project |
| --- | --- |
| `baseline_single10fold/` | `outputs/resin_features_8_18_ablation_20261105/models/baseline_single10fold/` |
| `weighted_ensemble_single10fold/` | `outputs/resin_features_8_18_ablation_20261105/models/diverse_train_only_single10fold/` |

Run `python scripts/import_local_assets.py --source-root "path/to/original/project"` from the repository root. Every file is verified against `config/artifact_manifest.json` (100 files; 139,348,559 bytes). The import preserves all saved fold models and preprocessing. Files are local and ignored by Git; a GitHub clone alone does not include fitted predictors.

Use Python 3.9 and AutoGluon 1.4.0. `FinalAdsorptionModel.load()` checks both versions, fitted model names and the original ordered 125-feature contract. `scripts/evaluate.py --verify-artifacts` additionally validates all file hashes. Retraining must write to a new directory, such as `outputs/retrained_models/`.
