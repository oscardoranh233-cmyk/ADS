# Final adsorption model

This repository contains the inference code for the locked final model:

`prediction = 0.626 × LightGBMSingle10_BAG_L1 + 0.374 × WeightedEnsemble_L2`

The two terms are saved AutoGluon predictors. `WeightedEnsemble_L2` is a fitted AutoGluon ensemble component; it is not retrained when this fixed outer blend is used.

## Open in VS Code

Open the `final_adsorption_model` folder in VS Code, then create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy the two local predictor folders described in [models/README.md](models/README.md) before running a prediction.

## Predict

The prediction input must be feature-complete: it must include the descriptor and MACCS columns used during the original model fitting, plus the raw resin columns. The code adds the locked resin interaction features automatically; it does not calculate molecular descriptors from SMILES.

```powershell
python scripts/predict.py --input data/features.csv --output outputs/predictions.csv
```

The output is the input table with `predicted_adsorption_capacity_mg_per_g` appended.

## Verify the configuration

```powershell
python -m pytest
```

## GitHub preparation

The `.gitignore` file excludes local model artifacts, virtual environments, prediction outputs, and Python cache files. Review and add an appropriate license before publishing.

