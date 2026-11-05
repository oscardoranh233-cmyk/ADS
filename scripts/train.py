from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
from final_adsorption_model.training import train_locked_model, evaluate_locked_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the final 125-feature training recipe.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data/local")
    parser.add_argument("--model-dir", type=Path, required=True, help="A new directory; existing models are never overwritten")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs/retrained")
    args = parser.parse_args()
    model = train_locked_model(PROJECT_DIR, args.data_dir, args.model_dir.resolve())
    evaluate_locked_model(model, args.data_dir, args.output_dir)
    print(f"Saved retrained predictors to {args.model_dir}")


if __name__ == "__main__":
    main()
