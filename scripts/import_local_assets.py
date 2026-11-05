"""Copy the confirmed artifacts and locked data from an existing research project."""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
from final_adsorption_model.training import sha256_file, verify_artifacts, load_locked_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Import local artifacts with checksum verification.")
    parser.add_argument("--source-root", type=Path, required=True, help="Original research project directory")
    args = parser.parse_args()
    source = args.source_root.resolve()
    model_source = source / "outputs/resin_features_8_18_ablation_20261105/models"
    artifacts = json.loads((PROJECT_DIR / "config/artifact_manifest.json").read_text(encoding="utf-8"))
    data = json.loads((PROJECT_DIR / "config/data_manifest.json").read_text(encoding="utf-8"))
    copies = []
    for item in artifacts["files"]:
        relative = Path(item["path"])
        name = relative.parts[1]
        source_name = "diverse_train_only_single10fold" if name == "weighted_ensemble_single10fold" else name
        copies.append((model_source / source_name / Path(*relative.parts[2:]),
                       PROJECT_DIR / relative, item["sha256"]))
    for item in data["files"]:
        copies.append((source / data["source"] / item["file"],
                       PROJECT_DIR / "data/local" / item["file"], item["sha256"]))
    for src, dst, digest in copies:
        if not src.is_file() or sha256_file(src) != digest:
            raise ValueError(f"Source is not the confirmed asset: {src}")
        if dst.exists() and sha256_file(dst) != digest:
            raise FileExistsError(f"Refusing to overwrite a different local asset: {dst}")
    for src, dst, _ in copies:
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    print(json.dumps(verify_artifacts(PROJECT_DIR)))
    load_locked_data(PROJECT_DIR, PROJECT_DIR / "data/local")
    print("Confirmed models and locked data are ready.")


if __name__ == "__main__":
    main()
