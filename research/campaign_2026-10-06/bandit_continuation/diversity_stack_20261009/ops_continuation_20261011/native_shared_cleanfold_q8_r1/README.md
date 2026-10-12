# Shared Q8 student/HFT clean-fold quality packet

This packet defines one fixed 50/50 **output** blend for each source-qualified outer fold: the fold's own `oldz4` student CP1 and HFT CP0. `shared_backbone=True` is requested explicitly; the native API default remains disabled. The models retain separate heads, and no parameters are averaged. Only first-two-output serving weights are exported, matching the frozen predictor's two-target endpoint.

Both fold evaluations were previously exposed during development. They are robustness measurements, not new holdouts: fold 0 scores its 468 excluded valid groups and fold 1 its 457 excluded valid groups. Their fitting ancestry, checkpoints, source, data footer, callback runner, metric, Python 3.11 wheel, and ELF are pinned in `CONTRACT_R1.json`. Full-valid/in-sample predictions are not part of this packet.

## Frozen commands

Run from this worktree. The contract builder is create-only; use a new packet ID/output if a source pin changes.

```sh
MAIN=/home/vlad/development/wunder/wnn_connectome_starterpack
PACKET=research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_shared_cleanfold_q8_r1
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s \
  "$MAIN/scripts/limited" "$MAIN/.venv/bin/python" "$PACKET/build_contract.py"
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s \
  "$MAIN/scripts/limited" "$MAIN/.venv/bin/python" "$PACKET/validate_packet.py"
```

Export the actual four checkpoint pairs to a new private output path. This reads checkpoints on CPU, verifies each student/HFT pair shares every non-head tensor, and writes only serving weights. It does not train or score.

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s \
  "$MAIN/scripts/limited" "$MAIN/.venv/bin/python" "$PACKET/export_pair_weights.py" \
  --weights /tmp/wnn_shared_cleanfold_q8_r3/weights.npz \
  --receipt /tmp/wnn_shared_cleanfold_q8_r3/WEIGHTS_RECEIPT_R1.json
```

Before any fold scoring, run the exact Py311 consumer preflight. It checks producer identities and actual Parquet metadata, imports the pinned ELF, loads the exported weights, constructs both shared Q8 ensembles, and requires the runtime precision string to be exactly `q8vnni`. This mode loads no labels and generates no predictions.

```sh
PY311=/tmp/wnn-projection-precompute-r1/py311/bin/python
NATIVE=/tmp/wnn-shared-f0-py311-package-r1
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s \
  "$MAIN/scripts/limited" env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONPATH="$NATIVE" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$PY311" "$PACKET/run_quality.py" \
  --weights /tmp/wnn_shared_cleanfold_q8_r3/weights.npz \
  --weights-receipt /tmp/wnn_shared_cleanfold_q8_r3/WEIGHTS_RECEIPT_R1.json \
  --preflight-output /tmp/wnn_shared_cleanfold_q8_r3/PREFLIGHT_R1.json
```

Full scoring is deliberately gated on a separate root-owned admission receipt. The receipt must use schema `wnn.root.native-shared-cleanfold-quality-admission.r1`, decision `PASS_FOR_ONE_FROZEN_FULL_CLEANFOLD_Q8_EVALUATION`, `status=admitted`, and `authorized=true`; its `bindings` must match the contract SHA, weights archive SHA, native ELF SHA, and validation parquet `{bytes, footer_bytes, footer_sha256_including_8_byte_trailer}`. The evaluator reads only the exact sorted outer-evaluation row groups, streams official `GlobalAccumulator` scoring with `need_prediction AND is_scored`, and writes fold-specific full-sequence predictions plus a receipt to a new output directory. It refuses to overwrite outputs.

Expected callback work is 18.5 million rows over both folds. The prior shared-Q8 microbenchmark observed 28.48–29.98 µs/row on an Intel i9-11900H; at 50% CPU, callback-only projection is about 1,110 seconds. A 1,800-second wall cap allows setup/decode headroom but is not a runtime guarantee or AMD forecast. No full-fold score has been produced by this packet yet.

After root has created an admission whose `bindings` include the exact contract SHA, NPZ SHA, weights-receipt SHA, ELF SHA, and footer identity above, the one admitted full-quality command is:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=1800s \
  "$MAIN/scripts/limited" env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONPATH="$NATIVE" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$PY311" "$PACKET/run_quality.py" \
  --weights /tmp/wnn_shared_cleanfold_q8_r3/weights.npz \
  --weights-receipt /tmp/wnn_shared_cleanfold_q8_r3/WEIGHTS_RECEIPT_R1.json \
  --root-admission /absolute/path/to/root-admission.json \
  --output-dir /tmp/wnn_shared_cleanfold_q8_r3/full_quality_output
```

The consumer refuses to run without that exact receipt. A completed score remains exposed development CV, not a public-score forecast or fresh-holdout result.
