from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
from final_adsorption_model import FinalAdsorptionModel
from final_adsorption_model.training import evaluate_locked_model, verify_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate development and fixed compound-test predictions.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data/local")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs/evaluation")
    parser.add_argument("--verify-artifacts", action="store_true", help="Verify the original local model files against SHA-256")
    parser.add_argument("--check-reference", action="store_true", help="Assert the original reported metrics within 1e-6")
    args = parser.parse_args()
    if args.verify_artifacts:
        if args.model_dir is not None:
            parser.error("Artifact hashes refer only to the original models/ directory")
        print(json.dumps(verify_artifacts(PROJECT_DIR)))
    result = evaluate_locked_model(FinalAdsorptionModel(PROJECT_DIR, args.model_dir),
                                   args.data_dir, args.output_dir, args.check_reference)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
