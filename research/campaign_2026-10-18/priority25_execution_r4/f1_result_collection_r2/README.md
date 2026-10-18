# F1 result collection R2

Status: **PREPARED_NOT_ADMITTED**. R2 is an isolated sibling of
`f1_result_collection_r1`; all R1 files and the recorded failure remain
unchanged. The original collection attempt failed before creating a snapshot or
success receipt because the validator looked up an absolute predictor command
path in an attempt source map whose keys are relative to the frozen source
root. This is a collector schema mismatch, not an export, grade, or model
failure. The failure receipt is
[`F1_RESULT_COLLECTION_FAILURE_V1.json`](../receipts/F1_RESULT_COLLECTION_FAILURE_V1.json),
SHA256 `6e0578aef73a5c70b2c0af40c0a11071c1ea5cbb52c0b81ea6b8dde86750b2af`.

The frozen export command names the absolute file
`/home/grinchenkoeg/computer-vision-homework-20261018/source-a266b036/scripts/predict_run.py`.
Each export attempt source map must bind that file with the exact source-root
relative key `scripts/predict_run.py` and SHA256
`750b9e77aa55673f61a4dd84d809d91dd315bf86eb36c36b065bc67a7d0f4b65`. R2
rejects a different command root, an absolute or root-prefixed alias for the
predictor, a wrong or missing predictor key, an unexpected hash, or changed
file bytes. Other producer source-map entries retain their existing key
formats, including absolute data-dependency paths; they are not reinterpreted
as source-root-relative predictor entries. The grader is held in the separate
external checkout, so its existing absolute source-map binding remains
unchanged. Focused fixtures exercise both formats.

R2 validates and archives the already-produced F1 exports and grade. It does
not run prediction export, grading, training, or checkpoint loading. The
captured grade is available locally, but the R1 failure receipt records no
snapshot or successful collection receipt; root must independently review R2
before admitting a bounded collection run. On admission, retain the established
caps: 75% CPU, 512 MB memory, no swap, hidden CUDA, physical temporary storage,
and 120 seconds. The exported arrays are only read by the admitted collector,
which streams their hashes and finiteness checks without deserializing them.

Run the standard-library fixtures from this directory:

```sh
python -m unittest -v test_f1_result_collection.py
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
```

The 15 tests use synthetic receipt and tiny NPY fixtures plus a copy of the
small pinned predictor source. They do not inspect F1 prediction arrays,
checkpoint tensors, targets, or parquet data. R2 contains no admission, export,
grade, or remote-execution claim. A successful collection would still cover
only one previously exposed fold/seed pair; it would not establish a width
winner, CP30, convergence, resumed/uninterrupted numerical parity, serving
performance, or public transfer.
