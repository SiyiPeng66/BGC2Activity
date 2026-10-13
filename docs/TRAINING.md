# Training and representation export

This lightweight repository contains the model implementations and training entry points. Full training corpora and historical runs are intentionally not included. No automatic download of those datasets is provided. Training requires separately prepared data; inspect each script's `--help` before use.

Export representations without retraining:

```bash
python scripts/encode_peptides.py --checkpoint models/peptide_encoder.pt --peptides examples/activity/input.tsv --output results/embeddings.tsv
```

Output contains 96 embedding columns and a metadata file. This standalone encoder consumes tokens only. Its token reconstruction pretraining objective did not train the pooling head; exportability is not proof of embedding quality.

BGC training requires `train_samples.tsv`, `validation_samples.tsv`, `test_samples.tsv` and `dataset_manifest.json` in a supplied directory:

```bash
python scripts/train_bgc_v4.py --data-dir /path/to/bgc_data --pfam2vec models/pfam2vec.csv.gz --output-dir runs/bgc_new --epochs 80 --patience 10 --batch-size 64 --threads 2
```

`--resume` restores the full saved state and verifies configuration and data/code hashes. Workspace-specific historical data reconstruction scripts are not included. Supply separately prepared data in the required format.

Encoder pretraining requires a prepared corpus, matching evidence manifest and optional monomer registry:

```bash
python scripts/train_peptide_pretraining.py --corpus /path/to/pretraining.tsv --manifest /path/to/manifest.json --registry /path/to/registry.json --output-dir runs/encoder_new --protocol method10 --development --epochs 30
```

`--development` explicitly permits incomplete corpus evidence. `method10` and `table_s11` represent different manuscript settings, not interchangeable claims of exact replication.

All-data activity fitting:

```bash
python scripts/train_mic_all_data.py --data /path/to/activity.tsv --encoder models/peptide_encoder.pt --output-dir runs/activity_new --epochs 150 --lr 0.001 --batch-size 128
```

The input table must include sample_id, tokens, n_terminal, c_terminal, cyclization and log2_mic. This script uses every row regardless of the split column. It selects by training RMSE and is not an independent evaluation. `--warm-start` restores weights but reinitializes the optimizer; `--probe` runs a small fitting diagnostic. The fitting repair changed optimization settings and is not claimed to reproduce all manuscript hyperparameters.

Use new output directories. Training does not replace bundled weights automatically; pass `--model` to the activity CLI when trying a new checkpoint.
