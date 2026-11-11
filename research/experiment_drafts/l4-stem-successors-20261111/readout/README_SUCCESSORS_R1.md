# L4 stem successor fixed-endpoint readout R1

`readout_successors_r1.py` reads the six completed successor fits from the
2026-11-11 cohort, exports each fixed E6 checkpoint once on its assigned L4,
and scores the official chronological validation with the source-pinned
canonical scorer. It never trains, resumes, profiles, or retries a fit. The
older causal-stem R1–R3 files stay frozen.

## Frozen identities and contract

- Successor commit: `6b10dfe71fb0ebec9b0fc76788e2de0fd13d1dba`.
- Source SHA-256: `38d6dd71ec29fb8d23b9df801a448f23fb8b78c1e0832883dda1d6f25095643d`.
- Source manifest SHA-256: `e2220d547dc92f791509e48762f164f70a3c064d01a4136156f8cc541dfb43d0`.
- Config manifest SHA-256: `2df8cba266f2ad5486e0b1de5370958666a915a6d22d45bfecd1e040af5ff6cf`.
- Official chronological split SHA-256: `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`.
- Cells: `k64`, `gru512_k16`, and `gru_dropout01_k16`, paired at seeds 203301 and 203302.
- Assigned GPUs: 0, 1, 2, 3, 5, and 6. GPU 4 is not used; GPU 7 remains spare.
- Endpoint: fixed 6 epochs / 38,760 updates; train only on the exact 10,325
  official fit sequences; score all 1,873 official valid sequences / 3,528,273
  selected rows with metric `global_wp_clip2_v1`.

The receiver verifies the source capsule before importing its config loader. It
binds raw config bytes to the frozen packet SHA-256 values and normalized config
hashes to ROOT's per-cell profile receipts. It checks the exact ROOT admission,
30-update actual-source profile/reload evidence, each fit result and model hash,
run metadata, data fingerprints, sparse official train/valid membership,
terminal identity, and all six full-valid monitor events. Readout workers wait
at a cohort barrier until all six fit results are terminal or the original fit
deadline arrives. Only individually validated `fit_complete` cells are read.

## ROOT launch

ROOT stages this file at
`/data/lc-epanov_s/tabdata-stem-successors-20261111-r1/readout-receiver-r1/readout_successors_r1.py`
and supplies the SHA-256 of the frozen
`control/ROOT-fit-admission-r1.json` after the profile/fit gate is complete.
The pinned shared executor/scorer validator remains
`/data/lc-epanov_s/tabdata-next70-20261110-r1/readout-receiver-r2/readout_r2.py`.
The two reused K16 control receipts are read-only from the completed
`/data/lc-epanov_s/tabdata-stem-20261110-r1` root.

```bash
/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python \
  /data/lc-epanov_s/tabdata-stem-successors-20261111-r1/readout-receiver-r1/readout_successors_r1.py \
  --root /data/lc-epanov_s/tabdata-stem-successors-20261111-r1 \
  --source /data/lc-epanov_s/tabdata-stem-successors-20261111-r1/source-r1 \
  --python /tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python \
  --old-src /data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1/source-gru-monitor-resume-r1/src \
  --r2-helper /data/lc-epanov_s/tabdata-next70-20261110-r1/readout-receiver-r2/readout_r2.py \
  --control-root /data/lc-epanov_s/tabdata-stem-20261110-r1 \
  --admission-sha256 <ROOT-provided-64-hex-SHA256> \
  --detach
```

`--detach` uses the existing owned process runtime and records the child PID and
birth identity in `control/readout-successors-r1/launcher.json`; omit it only
when ROOT intentionally runs the controller as its own detached owned worker.

The fit cutoff is the original campaign first-GPU epoch
`1794359428.5272088 + 14,400s`; the collection cutoff is the same first-GPU
epoch `+18,000s`. These are immutable. The readout uses six disjoint assigned
GPU slots, with canonical CPU scoring; each export is capped at 1,200 seconds,
each score at 600 seconds, aggregate root output at 3.4 GB, and free disk at a
2.5 GB minimum. It writes only `control/readout-successors-r1/` and
`runs/{cell}/final-readout-successors-r1/`. A prior owner, result, or output
prevents an automatic rerun.

## Reused controls and interpretation

The comparator receipts are the byte-pinned completed K16 fixed-E6 readouts
from source commit `336b1c436cf9f7a7d2196fb29eb0f55ca50e5da2`. Their exact
receipt hashes and model/config hashes are constants in the receiver. For each
successor arm, the summary reports both same-seed deltas and the arithmetic
mean of those two deltas, in the direction **successor minus reused K16** for
`t0`, `t1`, and `wp`. Contrasts are withheld unless all six new full-valid
scores and both reused controls validate.

The full-valid partition has already been repeatedly exposed during model
development. The canonical re-export confirms prediction/scorer agreement and
provides a descriptive paired screen; it is not an independent validation,
fresh confirmation, significance test, or public-score claim.

## Outputs

- Per-cell output: `runs/{cell}/final-readout-successors-r1/` with predictions,
  canonical `score.json`, and `readout-receipt.json`.
- Per-cell status and process stage receipts: `control/readout-successors-r1/`.
- Cohort owner and terminal summary: `control/readout-successors-r1/owner.json`
  and `complete.json`.
- Summary includes each model/config hash, full-valid endpoint metrics,
  reused-control receipt hashes/scores, per-seed deltas, and their two-seed
  means when all required evidence passes.

Focused CPU regression:

```powershell
C:/Users/admin/Documents/Codex/reviews/galam-test-env-20261027/Scripts/python.exe -m pytest research/experiment_drafts/l4-gru-causal-stem-20261110/readout/test_readout_successors_r1.py -q
```
