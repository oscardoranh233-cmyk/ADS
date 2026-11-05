# Locked research data (local only)

Place `train_model_rows.csv` (479 rows), `valid_model_rows.csv` (137 rows), and `test_model_rows.csv` (72 rows) in `data/local/`, or run `scripts/import_local_assets.py` from the repository root.

`config/data_manifest.json` records the exact source location and SHA-256 of each original CSV. Training and evaluation preserve CSV row order and check hashes, row counts, unique row IDs, finite targets, the feature schema, and compound separation between Test and development. Train and Valid use the historical row-level split; they are not compound-disjoint from one another.

The files already contain the historical descriptor values. They include `adsorption` as the target, `row_id`, `compound_id`, and `compound_group_key` as metadata. Only the ordered 125 features enter the estimator. Raw data and row-level predictions are excluded from Git. The public repository does not provide a standalone dataset download.
