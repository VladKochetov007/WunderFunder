# Receiver commands for B300 wave 2026-11-12

This packet reuses PR81's isolated card adapters and its `runtime/launch.py`,
`runtime/worker.py`, acceptance checks, profiler, fitter, exporter and scorer.
Use the supplied clean sparse Git worktrees at the exact pinned commits. All
model/data commands run on the receiving Linux host. Each command owns one
stage; no command queues the next stage or retries a failure.

## 1. Bind the exact runtimes and data

Cards 01?07 use Transformer commit
`8626000a4b965763a70011a0291a891aa6535d7e`; cards 08?10 use GRU commit
`67c3bd818ea243177eeb1dc789cd3d145eba0e81`. The prepared worktrees must retain
Git identity and pass `bindings.admit_runtime`: exact `HEAD`, clean tracked
`src`, `scripts` and `utils.py`, no untracked Python modules there, and no WNN
import before admission. A source archive alone fails this gate.

```bash
export WF_PACKET=/path/to/b300-all-20261112
export WF_RUNTIME_T=/path/to/clean-pr75-runtime-worktree
export WF_RUNTIME_G=/path/to/clean-pr79-runtime-worktree
export WF_DATA=/path/to/preserved-data-root
export WF_SPLIT=/path/to/frozen-split.json
export WF_RUNS=/path/to/persistent/b300-all-20261112-r1
export WF_PYTHON=/path/to/provisioned/bin/python
export WF_OWNER=actual-receiving-owner

# Inspect both sparse Git worktrees before preparation.
test "$(git -C "$WF_RUNTIME_T" rev-parse HEAD)" = 8626000a4b965763a70011a0291a891aa6535d7e
test "$(git -C "$WF_RUNTIME_G" rev-parse HEAD)" = 67c3bd818ea243177eeb1dc789cd3d145eba0e81
git -C "$WF_RUNTIME_T" diff --exit-code HEAD -- src scripts utils.py
git -C "$WF_RUNTIME_G" diff --exit-code HEAD -- src scripts utils.py
test -z "$(git -C "$WF_RUNTIME_T" ls-files --others --exclude-standard src scripts | grep '\.py$' || true)"
test -z "$(git -C "$WF_RUNTIME_G" ls-files --others --exclude-standard src scripts | grep '\.py$' || true)"
test "$(sha256sum "$WF_SPLIT" | cut -d' ' -f1)" = 0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c
```

The receiver must also bind actual `train.parquet` and `valid.parquet` hashes,
ordered features, whole-sequence membership, 10,325 fitting sequences and
1,873 validation sequences / 3,528,273 scored rows. Preserve the supplied
official file boundary. Every normalizer, PLE knot and other learned
preprocessor uses training members only. `seq_ix` is shuffled and is not time.

## 2. Prepare and check each runtime family

Run one family preparation and receiver check per exact packet, runtime, data
and split identity. The GRU preparation records the card 10 training-only tail
prevalence gate. If that fixed gate fails, preserve its receipt and omit both
card 10 arms; do not retune thresholds or inspect validation to change it.

```bash
# Transformer family: cards 01?07.
WF_RUNTIME="$WF_RUNTIME_T"
WF_FAMILY=transformer
WF_CELL="$WF_PACKET/configs/ip01_control_s204101.json"
COMMON=(--runtime-root "$WF_RUNTIME" --cell "$WF_CELL" --data-root "$WF_DATA" \
  --split-manifest "$WF_SPLIT" --owner "$WF_OWNER")
"$WF_PYTHON" "$WF_PACKET/runtime/launch.py" prepare "${COMMON[@]}" \
  --output "$WF_RUNS/$WF_FAMILY/preprocessing" --timeout-seconds 14400
"$WF_PYTHON" "$WF_PACKET/runtime/launch.py" check "${COMMON[@]}" \
  --preprocessing "$WF_RUNS/$WF_FAMILY/preprocessing/preprocessing.json" \
  --output "$WF_RUNS/$WF_FAMILY/acceptance/checks.json" \
  --devices <one-free-b300-index> --timeout-seconds 3600

# GRU family: cards 08?10. Use this separately after the Transformer pair.
WF_RUNTIME="$WF_RUNTIME_G"
WF_FAMILY=gru
WF_CELL="$WF_PACKET/configs/ip08_control_s204101.json"
COMMON=(--runtime-root "$WF_RUNTIME" --cell "$WF_CELL" --data-root "$WF_DATA" \
  --split-manifest "$WF_SPLIT" --owner "$WF_OWNER")
"$WF_PYTHON" "$WF_PACKET/runtime/launch.py" prepare "${COMMON[@]}" \
  --output "$WF_RUNS/$WF_FAMILY/preprocessing" --timeout-seconds 14400
"$WF_PYTHON" "$WF_PACKET/runtime/launch.py" check "${COMMON[@]}" \
  --preprocessing "$WF_RUNS/$WF_FAMILY/preprocessing/preprocessing.json" \
  --output "$WF_RUNS/$WF_FAMILY/acceptance/checks.json" \
  --devices <one-free-b300-index> --timeout-seconds 3600
```

A receiver or source failure is engineering evidence, not a negative mechanism
result. Do not proceed to profiles for a family until its check passes.

## 3. Profile the first eight exact cells

The initial cohort is two complete paired screens: cards 01 and 02 at seeds
204101 and 204102. Profile each of the eight exact configs on one distinct B300.
The profile must use the full model, data and recipe for that cell and complete
at least 50 updates with reload proof. Keep the selected device for that cell's
fit. The remaining 32 cells stay in the five-stage roster in
[`WAVE_20261112.json`](WAVE_20261112.json).

| Card | Seed | Exact cells |
|---|---:|---|
| 01 PLE | 204101 | `ip01_control_s204101`, `ip01_treatment_s204101` |
| 01 PLE | 204102 | `ip01_control_s204102`, `ip01_treatment_s204102` |
| 02 Same-row groups | 204101 | `ip02_control_s204101`, `ip02_treatment_s204101` |
| 02 Same-row groups | 204102 | `ip02_control_s204102`, `ip02_treatment_s204102` |

For each `WF_ID`, choose the matching runtime family and set `WF_CELL` to its
exact JSON file. Cards 01?07 use `WF_RUNTIME_T`, `WF_FAMILY=transformer`;
cards 08?10 use `WF_RUNTIME_G`, `WF_FAMILY=gru`.

```bash
WF_ID=ip01_control_s204101
WF_CELL="$WF_PACKET/configs/$WF_ID.json"
COMMON=(--runtime-root "$WF_RUNTIME" --cell "$WF_CELL" --data-root "$WF_DATA" \
  --split-manifest "$WF_SPLIT" --owner "$WF_OWNER")
MODEL_INPUTS=(--preprocessing "$WF_RUNS/$WF_FAMILY/preprocessing/preprocessing.json" \
  --checks-receipt "$WF_RUNS/$WF_FAMILY/acceptance/checks.json")
"$WF_PYTHON" "$WF_PACKET/runtime/launch.py" profile "${COMMON[@]}" "${MODEL_INPUTS[@]}" \
  --output "$WF_RUNS/$WF_ID/profile" --devices <assigned-free-b300-index> \
  --profile-updates 50 --timeout-seconds 3600
```

Review the profile and reload proof before recording approval. Check actual
update time, VRAM, host RSS, disk headroom, optimizer/master dtypes and all
50 update diagnostics. Set numeric approval ceilings from the live allocation
and measurements; they are not copied from another host or profile.

```bash
"$WF_PYTHON" "$WF_PACKET/runtime/approve_profile.py" \
  --runtime-root "$WF_RUNTIME" --cell "$WF_CELL" \
  --profile "$WF_RUNS/$WF_ID/profile/profile_receipt.json" \
  --output "$WF_RUNS/$WF_ID/approved_profile.json" --reviewer "$WF_OWNER" \
  --note "Reviewed exact-cell 50-update profile, reload and resource headroom" \
  --max-host-ram-gib <reviewed-host-limit> \
  --max-vram-gib-per-rank <reviewed-b300-limit> --approve
```

Review and approve each receipt as a separate operator decision. Never chain
approval to profile completion. Capture an exact full-shape profile for every
cell before fitting that cell.

## 4. Fit a reviewed cell

Transformer cards 01?07 use the amended one-B300 placement: batch 256,
accumulation 4, fixed endpoint E4, 3,280 planned updates. Their update sees the
same 256 sequence IDs as the former two-rank batch-128 recipe, with a different
floating-point reduction order; all require fresh paired controls. GRU cards
08?10 retain batch 32, accumulation 1, FP32/AdamW and fixed endpoint E6.
Retain every epoch's full official-valid score, complete resource and training
diagnostics, and resumable optimizer/RNG/data-position state.

```bash
"$WF_PYTHON" "$WF_PACKET/runtime/launch.py" fit "${COMMON[@]}" "${MODEL_INPUTS[@]}" \
  --profile-receipt "$WF_RUNS/$WF_ID/approved_profile.json" \
  --output "$WF_RUNS/$WF_ID/fit" --devices <same-profiled-b300-index> \
  --timeout-seconds <positive-seconds-remaining-before-CLOCK_R1-fit-cutoff>
```

Use the external campaign `CLOCK_R1.json` as deadline authority. Bound each
attempt by the remaining time and the launcher limit. There are no automatic
retries; preserve interruption evidence and resume only from a safe checkpoint
with the identical runtime, packet, config, split and preprocessing. A resume
does not extend the shared cutoff.

## 5. Export, score and compare

Export the fixed terminal model, then score the complete official valid file.
A complete receipt binds 1,873 sequences, 3,528,273 selected rows and
`global_wp_clip2_v1`.

```bash
"$WF_PYTHON" "$WF_PACKET/runtime/launch.py" export "${COMMON[@]}" \
  --checkpoint "$WF_RUNS/$WF_ID/fit/model.pt" \
  --output "$WF_RUNS/$WF_ID/predictions" --devices <profiled-device-index> \
  --timeout-seconds 7200
"$WF_PYTHON" "$WF_PACKET/runtime/launch.py" score "${COMMON[@]}" \
  --predictions "$WF_RUNS/$WF_ID/predictions" \
  --output "$WF_RUNS/$WF_ID/score.json" --timeout-seconds 7200
```

For a completed card, compare its four paired endpoints. The comparator validates
all four receipts and never fills a missing cell with zero. Its default mode
expects all 40 score receipts across cards 01?10.

```bash
PYTHONPATH="$WF_PACKET/runtime" "$WF_PYTHON" -m impact81.comparison \
  --card 01 --allow-one-card --output "$WF_RUNS/card01-comparison.json" \
  --scores "$WF_RUNS/ip01_control_s204101/score.json" \
    "$WF_RUNS/ip01_treatment_s204101/score.json" \
    "$WF_RUNS/ip01_control_s204102/score.json" \
    "$WF_RUNS/ip01_treatment_s204102/score.json"
```

Keep partial and negative outcomes. Export actual fit evidence through the
packet's MIDAS handoff only after the canonical source, packet-source map and
score identities validate.
