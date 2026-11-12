# B300 receiver commands

These commands are prepared, not executed. Use new checkouts and persistent output;
keep existing jobs on their original source. Substitute provisioned paths, an owner
and actually free B300 devices. The provisioned environment must already include
Git in PATH for source admission. Do not change dependencies or rent new hardware.

```bash
export WF_REPO=/path/to/new-signal-routes-checkout
export WF_PACKET="$WF_REPO/research/experiment_drafts/causal-routing-b300-20261111"
export WF_RUNTIME=/path/to/new-pr75-runtime
export WF_DATA=/path/to/data-root
export WF_SPLIT=/path/to/original-split.json
export WF_RUNS=/path/to/persistent/signal-routes
export WF_PYTHON=/path/to/provisioned/bin/python
export WF_OWNER=actual-receiver-owner
git -C "$WF_REPO" fetch origin 8626000a4b965763a70011a0291a891aa6535d7e
git -C "$WF_REPO" worktree add --detach "$WF_RUNTIME" 8626000a4b965763a70011a0291a891aa6535d7e
```

`WF_DATA` contains `datasets/train.parquet` and `datasets/valid.parquet`. Original
split bytes must hash to
`0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`.
Restore the exact split, rather than generating a new one to pass admission.

## Choose one cell

The five variants are `plain`, `attention_additive`, `attention_differential`,
`ffn_static` and `ffn_conditional`. Run both seeds 205201/205202 for a complete
study. The two plain cells are shared anchors; do not run them twice.

```bash
WF_ID=sr_attention_differential_s205201
WF_CELL="$WF_PACKET/configs/$WF_ID.json"
WF_DEVICES=0,1
WF_CHECK_DEVICE=0
COMMON=(--runtime-root "$WF_RUNTIME" --cell "$WF_CELL" --data-root "$WF_DATA" \
  --split-manifest "$WF_SPLIT" --owner "$WF_OWNER")
```

The example GPU IDs are placeholders. Each profile and its admitted fit require
the exact same device/runtime allocation. Across cells, the comparison requires
matching software, precision, world size and GPU capabilities; physical UUIDs and
visible-device IDs may differ. There is no automatic ten-cell dispatch.

## Prepare data and run receiver probes

Preparation computes only fitting-data moments and verifies membership. Checks
run on B300 and cover all five variants, including initialization, gradients,
causality, cache/chunk equivalence, optimizer ownership and strict reload. Reuse
this exact-source preprocessing and acceptance receipt for other cells only when
the receipt declares the full variant set.

```bash
"$WF_PYTHON" "$WF_PACKET/runtime/launch.py" prepare "${COMMON[@]}" \
  --output "$WF_RUNS/preprocessing" --timeout-seconds 14400
"$WF_PYTHON" "$WF_PACKET/runtime/launch.py" check "${COMMON[@]}" \
  --preprocessing "$WF_RUNS/preprocessing/preprocessing.json" \
  --output "$WF_RUNS/acceptance/checks.json" --devices "$WF_CHECK_DEVICE" \
  --timeout-seconds 3600
```

## Profile, review, fit

Each full-size cell needs its own profile. A tiny model probe is not evidence that
H768/L24 fits. Preserve the profile's parameter counts, reload result, update/gradient
records, host RAM, VRAM, device/software identity and wall time.

```bash
MODEL_INPUTS=(--preprocessing "$WF_RUNS/preprocessing/preprocessing.json" \
  --checks-receipt "$WF_RUNS/acceptance/checks.json")
"$WF_PYTHON" "$WF_PACKET/runtime/launch.py" profile "${COMMON[@]}" "${MODEL_INPUTS[@]}" \
  --output "$WF_RUNS/$WF_ID/profile" --devices "$WF_DEVICES" \
  --profile-updates 50 --timeout-seconds 3600
```

Inspect the actual profile before approval. The ceilings below are examples and
must be replaced by the real allocation; do not automatically chain approval to
profiling. Admission must also leave space for four retained epoch checkpoints
and the resumable optimizer state.

```bash
"$WF_PYTHON" "$WF_PACKET/runtime/approve_profile.py" \
  --runtime-root "$WF_RUNTIME" --cell "$WF_CELL" \
  --profile "$WF_RUNS/$WF_ID/profile/profile_receipt.json" \
  --output "$WF_RUNS/$WF_ID/approved_profile.json" \
  --reviewer "$WF_OWNER" --note "Reviewed measured profile, reload and allocated resources" \
  --max-host-ram-gib 160 --max-vram-gib-per-rank 220 --approve
"$WF_PYTHON" "$WF_PACKET/runtime/launch.py" fit "${COMMON[@]}" "${MODEL_INPUTS[@]}" \
  --profile-receipt "$WF_RUNS/$WF_ID/approved_profile.json" \
  --output "$WF_RUNS/$WF_ID/fit" --devices "$WF_DEVICES" --timeout-seconds 28800
```

Eight hours is an attempt bound, not a runtime estimate. Resume only after
diagnosing the previous termination, from the same checkout and output, using
`--resume "$WF_RUNS/$WF_ID/fit/training_state.pt"`. Do not retry errors indefinitely.

## Fixed endpoint and full validation

```bash
"$WF_PYTHON" "$WF_PACKET/runtime/launch.py" export "${COMMON[@]}" \
  --checkpoint "$WF_RUNS/$WF_ID/fit/model.pt" --output "$WF_RUNS/$WF_ID/predictions" \
  --devices "$WF_DEVICES" --timeout-seconds 7200
"$WF_PYTHON" "$WF_PACKET/runtime/launch.py" score "${COMMON[@]}" \
  --predictions "$WF_RUNS/$WF_ID/predictions" --output "$WF_RUNS/$WF_ID/score.json" \
  --timeout-seconds 7200
```

Export requires the configured final E4 checkpoint. Score requires complete
1,873-sequence / 3,528,273-row coverage. Earlier epoch monitors remain diagnostics;
they cannot replace the fixed primary endpoint after seeing validation scores.

After all ten endpoints, compare the frozen panel:

```bash
PYTHONPATH="$WF_PACKET/runtime" timeout 300 "$WF_PYTHON" -m signal_routes.comparison \
  --output "$WF_RUNS/comparison.json" --scores \
  "$WF_RUNS/sr_plain_s205201/score.json" \
  "$WF_RUNS/sr_plain_s205202/score.json" \
  "$WF_RUNS/sr_attention_additive_s205201/score.json" \
  "$WF_RUNS/sr_attention_additive_s205202/score.json" \
  "$WF_RUNS/sr_attention_differential_s205201/score.json" \
  "$WF_RUNS/sr_attention_differential_s205202/score.json" \
  "$WF_RUNS/sr_ffn_static_s205201/score.json" \
  "$WF_RUNS/sr_ffn_static_s205202/score.json" \
  "$WF_RUNS/sr_ffn_conditional_s205201/score.json" \
  "$WF_RUNS/sr_ffn_conditional_s205202/score.json"
```

Export actual completed, negative or incomplete outcomes using the
[MIDAS handoff](midas/README.md). Preserve logs and checkpoints separately; this PR
does not package or upload a competition submission.
