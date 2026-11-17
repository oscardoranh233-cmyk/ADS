# Confirmed local model artifacts

The current **standalone LightGBM** uses 125 features and ten folds of all 616 development records.

| Local destination | Source below the original research project |
| --- | --- |
| `single_baseline/baseline_single10fold/` | `model fix2/baseline LightGBM/models/baseline_single10fold/` |

Run `python scripts/import_local_assets.py --source-root "path/to/original/project"` from the repository root. The import checks 20 files against `config/artifact_manifest.json`, including the saved fold models and preprocessing. A fresh clone needs these fitted artifacts as local assets.

Use Python 3.9 and AutoGluon 1.4.0. `FinalAdsorptionModel.load()` checks both versions, fitted model names and the original ordered 125-feature contract. `scripts/evaluate.py --verify-artifacts` additionally validates all file hashes. Retraining must write to a new directory, such as `outputs/retrained_models/`.
