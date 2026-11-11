# L4 fixed-endpoint readout R1

`readout_r1.py` is a one-shot native receiver for the six already-admitted PR77
fits. It creates no model or training state. Each cell waits for ROOT's
`control/root-fits-r1/{cell}-result.json`, requires `status=fit_complete`, and
cross-checks its model bytes, run metadata, source/config/split identity,
terminal status, and all six fixed-E6 full-valid events before it exports. It
also requires the exact fit/evaluation row-group lists from the frozen official
chronological split. For train and valid parquet inputs it matches the fit's
provenance against the split manifest, then checks byte size, nanosecond mtime,
and the recorded Parquet footer SHA-256 against the actual file.

After a cell's fit result is complete, the receiver uses that cell's assigned
GPU for the canonical fixed-checkpoint command:

```text
python scripts/predict_sequence.py --checkpoint <fit/model.pt> --data-root <R/project> --split-manifest <R/project/chronological_primary.json> --output <R/runs/{cell}/readout-final/predictions> --membership evaluation
```

It then runs `scripts/score_sequence.py` on CPU. The canonical scorer validates
each listed prediction SHA, float32 `[20000, 2]` array, finite values and exact
full-valid roster before computing pooled `t0`, `t1`, and mean `wp`. Its score
receipt must match the training-time E6 monitor within absolute `1e-5` per
component. A mismatch is recorded as an export/metric-pipeline failure; it is
not treated as a method result.

ROOT's launch command on the native host is:

```bash
/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python \
  <deployed-readout-folder>/readout_r1.py \
  --root /data/lc-epanov_s/tabdata-next70-20261110-r1 \
  --source /data/lc-epanov_s/tabdata-next70-20261110-r1/source-specialist-r2 \
  --old-src /data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1/source-gru-monitor-resume-r1/src
```

The script imports only the existing owned-command executor from `--old-src`.
Prediction and scoring children receive only the frozen new source in
`PYTHONPATH`. Six disjoint workers wait on their own fit receipts; each stage
has a 600-second cap, a source-clock wave deadline, unique output paths,
receipt-bound PID/birth/PGID ownership, a 6 GB output cap and 4 GB free-space
reserve. Existing output or child receipts prevent reruns. No retry or cleanup
of foreign processes is implemented.

When all six scores pass, `control/readout-r1/complete.json` contains the
baseline plus three predeclared output pairings for each seed. Every complete
output is `0.5 * (left model official valid WP t0 + right model official valid
WP t1)`, with no validation-fitted weights. This recomputes the same repeatedly
exposed official validation partition; agreement checks the export/scorer
pipeline and is not independent validation or fresh confirmation.

Focused CPU tests do not load PyTorch or contact a GPU:

```bash
python -m pytest research/experiment_drafts/l4-next70-20261110/readout/test_readout_r1.py -q
```
