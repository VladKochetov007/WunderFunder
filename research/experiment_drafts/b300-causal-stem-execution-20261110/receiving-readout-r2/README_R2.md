# B300 PR74 final readout receiver R2

This receiver exports and grades the six already-completed fixed 4E endpoints.
It starts no fits and does not modify any `runs/{cell}/fit` files. It accepts
only the frozen source capsule (`8a8afec964985a6ba8b13d1c9af1ab00e14c6504`,
manifest SHA-256 `1cb6bf60c668f34ea3c49e6f4c3b0c774ea6cdc7e8f8c5f29e5d9f04322f1936`),
the R1 authority, the original first-profile clock, and ROOT fit admission
`fa608f409499b67ef6c37b6d0e6a07f9c1a32392d321edbf669effd870c8ba18`.

Before any export, it requires the R2 fit owner/state to be complete, all six
fit results to be on time and zero-exit, their artifact inventories to remain
unchanged, their approved profiles to match admission, and all four full-valid
monitor events to validate. Prediction membership comes from the canonical
split's exact `train_groups` and `evaluation_groups`; no contiguous-ID or row
split assumption is used.

Each of the four approved GPU-pair streams runs its assigned readouts
sequentially. Each prediction uses the canonical two-rank `predict_sequence.py`
entrypoint. CPU scoring uses the canonical `score_sequence.py` entrypoint and
must agree with all three final full-valid monitor metrics within `1e-5`.
Finally, `compare_panel.py` runs against a separate shadow tree containing
hardlinks/copies of the model, terminal, approved-profile, and scored endpoint
files. Its hashes and scores must match the readout receipts. The actual fit
directories remain untouched.

The only deadline is the original six-hour wave clock. There is no resume or
automatic retry. Prediction, scoring, and panel stages have 2,400s, 1,200s,
and 600s per-stage caps. Outputs are written under
`runs/{cell}/final-readout-r2/` and
`control/b300-driver-r1/readout-r2/`.

Example native invocation after ROOT releases the completed fit controller
(use the exact R1 authority hash and staged receiver directories):

```bash
R=/data/epanov_s/tabdata-causal-stem-b300-20261110-r1
PY=/data/epanov_s/tabdata/.venv/bin/python
"$PY" "$R/receiver-readout-r2/b300_readout_r2.py" \
  --root "$R" \
  --source "$R/source-r1" \
  --manifest "$R/source-r1/source-manifest.json" \
  --manifest-sha256 1cb6bf60c668f34ea3c49e6f4c3b0c774ea6cdc7e8f8c5f29e5d9f04322f1936 \
  --source-commit 8a8afec964985a6ba8b13d1c9af1ab00e14c6504 \
  --python "$PY" \
  --authority "$R/control/b300-driver-r1/authority.json" \
  --authority-sha256 <R1-authority-sha256> \
  --fit-admission-sha256 fa608f409499b67ef6c37b6d0e6a07f9c1a32392d321edbf669effd870c8ba18 \
  --r1-receiver "$R/receiver-r1" \
  --r2-receiver "$R/receiver-r2"
```

The collector summary schema is in `readout_schema_r2.json`. A completed result
is an exposed fixed-endpoint screen, not fresh confirmation or a public score.

Focused local CPU checks use the pinned environment:

```powershell
C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe -m pytest C:/Users/admin/Documents/Codex/artifacts/b300-causal-stem-20261110/receiving-readout-r2/test_b300_readout_r2.py -q
```
