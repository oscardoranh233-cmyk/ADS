from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from final_adsorption_model import FinalAdsorptionModel  # noqa: E402


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Input must be a CSV or Excel workbook")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict adsorption capacity with the locked final blend.")
    parser.add_argument("--input", type=Path, required=True, help="Feature-complete CSV/XLSX input table")
    parser.add_argument("--output", type=Path, required=True, help="Prediction CSV output")
    args = parser.parse_args()

    data = read_table(args.input)
    prediction = FinalAdsorptionModel(PROJECT_DIR).predict(data)
    output = data.copy()
    output[prediction.name] = prediction
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output)} predictions to {args.output}")


if __name__ == "__main__":
    main()

