"""Build descriptors from supplied SMILES and supplied numeric resin profiles."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
from final_adsorption_model import prepare_model_features
from final_adsorption_model.model import RESIN_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand each supplied SMILES across the supplied resin profiles.")
    parser.add_argument("--compounds", type=Path, required=True, help="CSV with compound_id and smiles")
    parser.add_argument("--resins", type=Path, required=True, help="CSV with resin and the seven raw resin columns")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from final_adsorption_model.molecular import molecular_features

    compounds, resins = pd.read_csv(args.compounds), pd.read_csv(args.resins)
    if not {"compound_id", "smiles"}.issubset(compounds.columns):
        raise ValueError("Compounds require compound_id and smiles")
    if not {"resin", *RESIN_COLUMNS}.issubset(resins.columns):
        raise ValueError("Resins require resin and all seven raw resin columns")
    if compounds.empty or resins.empty or not compounds.compound_id.is_unique or not resins.resin.is_unique:
        raise ValueError("Input tables must be nonempty with unique compound/resin identifiers")
    rows = []
    for item in compounds.to_dict("records"):
        descriptors = molecular_features(item["smiles"])
        for resin in resins[["resin"] + RESIN_COLUMNS].to_dict("records"):
            rows.append({**item, **descriptors, **resin})
    result = pd.DataFrame(rows)
    prepare_model_features(result, PROJECT_DIR)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(result)} descriptor rows to {args.output}")


if __name__ == "__main__":
    main()
