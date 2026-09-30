# Cache coverage and restoration

The complete main and C++ cache directory inventories are in `cache_inventory.json` (relative paths, byte counts and original mtimes). The approximately 94 GB main cache was deliberately excluded from this upload; organizer parquet files are separately inventoried in `dataset_inventory.json`. Original copies remain on beleriand. Small chain/fold/transition/propensity metadata is included inside `snapshot/*/cache-metadata/`.

After restoring source and experiment files, copy the small metadata into the corresponding repository cache directory. Model predictions can be regenerated, but doing so consumes compute and was not run by the archive task. The project's documented examples are:

```bash
# From the restored main repository with its Python environment and organizer data available:
.venv/bin/python scripts/predict_run.py ref_f0 valid --epoch 3
.venv/bin/python scripts/predict_run.py ft128_f0h valid --output-name ft128H_f0
.venv/bin/python scripts/predict_run.py fin256_s3 train
```

Do not substitute `model.pt` for a preregistered fixed epoch: the former may have been selected on the evaluated fold. Consult archived run configs, logs and `fold_job.sh` / `fold_ft.sh` for exact run/cache mappings. `run_index.json` records the checkpoint names and recovered training arguments; `cache_inventory.json` identifies the previously existing outputs. Regeneration is not guaranteed bit-identical across hardware or software.

For an exact later transfer, subject to available destination disk and the user's chosen cache scope, use a separate destination rather than overwrite an active experiment directory:

```bash
mkdir -p /chosen/destination/cache
rsync -a --info=progress2 beleriand:/home/grinchenkoeg/WunderFunder/cache/ /chosen/destination/cache/
```

That command is documentation only and was not executed. Capture or compare a new source inventory if live jobs are still writing prediction files. No environment credentials, home configuration, or dataset bodies are in this archive.
