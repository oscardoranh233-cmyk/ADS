# Local model artifacts

This directory is intentionally excluded from Git. Before running predictions, copy the two saved AutoGluon predictor directories here:

| Destination | Source directory in the original project |
| --- | --- |
| `baseline_single10fold/` | `KADSpre/v6_green_single10fold_weighted_20260801/models/baseline_single10fold/` |
| `weighted_ensemble_single10fold/` | `KADSpre/v6_green_single10fold_diverse_dual_model_sweep_20260803/models/diverse_train_only_single10fold/` |

Keep each directory unchanged after copying. The model configuration names the fitted AutoGluon models that are used from the two predictors.

