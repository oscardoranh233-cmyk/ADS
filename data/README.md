# Locked research data (local only)

Place `train_model_rows.csv` (479 rows), `valid_model_rows.csv` (137 rows), and `test_model_rows.csv` (72 rows) in `data/local/`, or run `scripts/import_local_assets.py` from the repository root.

`train_model_rows.csv` and `valid_model_rows.csv` are storage partitions only. Concatenate them in that order to form the 616-record development set (77 compounds). Apply one shuffled record-level ten-fold partition to this combined set. Each fold fits on 554–555 records and validates on 61–62 records. Training and validation metrics are arithmetic means of ten separately calculated fold scores; no 7:2:1 reporting partition is used.

`config/data_manifest.json` records the exact source location and SHA-256 of each CSV. Training and evaluation preserve CSV row order and check hashes, row counts, unique row IDs, finite targets, the feature schema, and compound separation between test and development. The 72-record test set contains nine compounds absent from development. Compounds can overlap between the fitting and validation folds within development. The fixed test set has previously been inspected during model comparison.

Test predictions are averaged over the ten saved fold models before computing test metrics. Development OOF predictions retain their fold identifiers for audit; pooling them and calculating a single score is different from the reported arithmetic mean of fold scores.

The files already contain the historical descriptor values. They include `adsorption` as the target, `row_id`, `compound_id`, and `compound_group_key` as metadata. Only the ordered 125 features enter the estimator. Raw data and row-level predictions are excluded from Git. The public repository does not provide a standalone dataset download.
