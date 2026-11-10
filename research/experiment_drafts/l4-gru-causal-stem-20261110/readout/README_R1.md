# Causal-stem fixed-endpoint readout R1

`readout_stem_r1.py` performs one final-checkpoint export and canonical CPU score
for each ROOT-admitted fit. It waits for the existing
`control/root-fits-r1/{cell}-result.json` receipts, verifies the ROOT admission
SHA supplied at launch, the frozen source capsule, model bytes, run metadata,
terminal record, all six full-valid monitor events, data fingerprints and
official split membership, then starts export and scoring. It never trains,
retries a failed stage, or opens a new deadline.

The receiver dynamically loads the pinned R2 helper (`readout_r2.py`,
SHA-256 `fadb34135dbf3ac6b5f3c5ab4d316e892d9ea16e6094bbc999cf5344ba94eb68`)
for the owned-command executor, GPU slot lock, disk guard, prediction-manifest
validator and score-receipt validator. It binds those APIs to source
`b9eddeb9a0733b8efa5a11bed24d03de53f94b83b03d72b5db09ab1adcd63188` and split
`0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`. The
interpreter argument is converted to an absolute lexical path and is not
dereferenced, preserving the virtual environment symlink.

Run on the ROOT-owned native host after ROOT provides the exact admission SHA:

```bash
/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python \
  /data/lc-epanov_s/tabdata-stem-20261110-r1/readout-receiver-r1/readout_stem_r1.py \
  --root /data/lc-epanov_s/tabdata-stem-20261110-r1 \
  --source /data/lc-epanov_s/tabdata-stem-20261110-r1/source-r1 \
  --python /tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python \
  --old-src /data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1/source-gru-monitor-resume-r1/src \
  --r2-helper /data/lc-epanov_s/tabdata-next70-20261110-r1/readout-receiver-r2/readout_r2.py \
  --admission-sha256 <ROOT-provided-64-hex-SHA256>
```

The receiver requires the original first-profile clock: fit results must be
published by `first_profile_epoch + 10,800s`; collection closes at
`first_profile_epoch + 14,400s`. Each export is capped at 1,200 seconds and
each score at 600 seconds. The aggregate output cap is 2.5 GB with a 4 GB free
space reserve. Prediction export is allowed only after an on-time successful
fit result and its independent evidence checks pass.

## Collection outputs

- Cell output: `runs/{cell}/final-readout-r1/` with `predictions/`, `score.json`
  and `readout-receipt.json`.
- Cell status: `control/readout-r1/{cell}-result.json`.
- Controller status and comparison: `control/readout-r1/complete.json`.
- Process ownership: `control/readout-r1/owner.json` and per-stage receipts under
  `control/readout-r1/owned/`.

The result schemas are `wnn.l4-gru-causal-stem-readout-cell.r1` and
`wnn.l4-gru-causal-stem-readout-summary.r1`; their formal JSON Schema is in
`readout_schema_r1.json`. A completed cell records the
full-valid score (`t0`, `t1`, `wp`), 1,873 prediction members, shape
`[20000, 2]`, dtype `float32`, scorer row count 3,528,273, model and provenance
digests, the monitor comparison at absolute tolerance `1e-5`, and both owned
stage receipts. The summary reports each model separately, then paired
same-seed K16-minus-K1 and K16-minus-plain deltas for all three score fields.
It does not combine models or compare to the earlier specialist screen.

The score is a re-export of the repeatedly exposed official validation split.
Agreement checks prediction bytes, coverage and scorer behavior; it is not
independent validation or fresh confirmation.

Focused local fixtures (no PyTorch, GPU, network or training):

```powershell
C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe -m pytest research/experiment_drafts/l4-gru-causal-stem-20261110/readout/test_readout_stem_r1.py -q
```
