# B300 commands for cards01–06

Run only on the team's provisioned Linux host, using the same Python/CUDA/Torch
environment as PR75. These commands are prepared, **not executed**. Paths, owner,
free GPU IDs and numeric limits must refer to the actual allocation. Use persistent
disk; do not use `/tmp`, `/dev/shm`, a running checkout or another job's output.
No dependency/nightly upgrade, rental or automatic resource selection is included.

## Source and inputs

Fetch PR81 into its own checkout/worktree and record its exact commit. Separately
stage the immutable runtime; neither operation changes a running job:

```bash
# Set these to real provisioned paths first.
export WF_REPO=/path/to/pr81-checkout
export WF_RUNTIME=/path/to/new-pr75-runtime-worktree
export WF_DATA=/path/to/data-root
export WF_SPLIT=/path/to/original-split.json
export WF_RUNS=/path/to/persistent/pr81-results
export WF_PYTHON=/path/to/provisioned/bin/python
export WF_OWNER=actual-receiving-owner
export WF_PACKET="$WF_REPO/research/experiment_drafts/impact-portfolio-20261111"

git -C "$WF_REPO" fetch origin 8626000a4b965763a70011a0291a891aa6535d7e
git -C "$WF_REPO" worktree add --detach "$WF_RUNTIME" 8626000a4b965763a70011a0291a891aa6535d7e
```

`WF_DATA` contains `datasets/train.parquet` and `datasets/valid.parquet` under the
canonical contract. Split SHA must be
`0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`;
the preparation/fit code verifies identity and the complete memberships. Do not
recreate a different split to make a check pass.

The wrapper sets `CUBLAS_EMULATE_SINGLE_PRECISION=1` before any Torch import. Use
the same frozen source files for checks, profiles, fits and exports; source edits
invalidate those receipts. Runtime setup rejects dirty tracked source and unknown
Python modules in its code paths.

## One shared preprocessing artifact and server acceptance

```bash
WF_CELL="$WF_PACKET/configs/ip01_control_s204101.json"
COMMON=(--runtime-root "$WF_RUNTIME" --cell "$WF_CELL" --data-root "$WF_DATA" \
  --split-manifest "$WF_SPLIT" --owner "$WF_OWNER")

"$WF_PYTHON" "$WF_PACKET/runtime/launch.py" prepare "${COMMON[@]}" \
  --output "$WF_RUNS/preprocessing" --timeout-seconds 7200

"$WF_PYTHON" "$WF_PACKET/runtime/launch.py" check "${COMMON[@]}" \
  --preprocessing "$WF_RUNS/preprocessing/preprocessing.json" \
  --output "$WF_RUNS/acceptance/checks.json" --devices 0 --timeout-seconds 1800
```

GPU0 above is an example: use an actually free B300. A failed check is terminal
for this attempt. Preserve logs, fix the identified cause and declare a new
attempt; do not suppress it or loop retries. This preparation also verifies data
and normalizer identity; it reads no validation labels for preprocessing.

## One selected cell: profile, explicit review, fit

Choose **one** ID from the table; the same sequence applies to every selected cell.
No loop or automatic 24-fit admission is supplied.

| Card | Controls | Treatments |
|---|---|---|
|01 PLE | `ip01_control_s204101`, `ip01_control_s204102` | `ip01_treatment_s204101`, `ip01_treatment_s204102` |
|02 Groups | `ip02_control_s204101`, `ip02_control_s204102` | `ip02_treatment_s204101`, `ip02_treatment_s204102` |
|03 Adapters | `ip03_control_s204101`, `ip03_control_s204102` | `ip03_treatment_s204101`, `ip03_treatment_s204102` |
|04 Deep supervision | `ip04_control_s204101`, `ip04_control_s204102` | `ip04_treatment_s204101`, `ip04_treatment_s204102` |
|05 Copied / tied depth | `ip05_control_s204101`, `ip05_control_s204102` | `ip05_treatment_s204101`, `ip05_treatment_s204102` |
|06 Pairwise ranking | `ip06_control_s204101`, `ip06_control_s204102` | `ip06_treatment_s204101`, `ip06_treatment_s204102` |

```bash
WF_ID=ip04_treatment_s204101
WF_CELL="$WF_PACKET/configs/$WF_ID.json"
COMMON=(--runtime-root "$WF_RUNTIME" --cell "$WF_CELL" --data-root "$WF_DATA" \
  --split-manifest "$WF_SPLIT" --owner "$WF_OWNER")
MODEL_INPUTS=(--preprocessing "$WF_RUNS/preprocessing/preprocessing.json" \
  --checks-receipt "$WF_RUNS/acceptance/checks.json")

"$WF_PYTHON" "$WF_PACKET/runtime/launch.py" profile "${COMMON[@]}" "${MODEL_INPUTS[@]}" \
  --output "$WF_RUNS/$WF_ID/profile" --devices 0,1 \
  --profile-updates 50 --timeout-seconds 3600
```

Inspect the actual profile and reload evidence and available host/VRAM/disk budget.
The following command records the operator's decision; it must **not** be chained
blindly after profiling. Example RAM limits below are admission ceilings, not
measurements or predicted requirements. Replace them with the allocated limits.

```bash
"$WF_PYTHON" "$WF_PACKET/runtime/approve_profile.py" \
  --runtime-root "$WF_RUNTIME" --cell "$WF_CELL" \
  --profile "$WF_RUNS/$WF_ID/profile/profile_receipt.json" \
  --output "$WF_RUNS/$WF_ID/approved_profile.json" \
  --reviewer "$WF_OWNER" --note "Reviewed real profile/reload and available allocation" \
  --max-host-ram-gib 160 --max-vram-gib-per-rank 220 --approve

"$WF_PYTHON" "$WF_PACKET/runtime/launch.py" fit "${COMMON[@]}" "${MODEL_INPUTS[@]}" \
  --profile-receipt "$WF_RUNS/$WF_ID/approved_profile.json" \
  --output "$WF_RUNS/$WF_ID/fit" --devices 0,1 --timeout-seconds 28800
```

Use the **same physical GPU pair, runtime and source** as that cell's profile;
the canonical receipt binds them. Controls/treatments must be paired on the same
hardware/runtime. The8-hour bound is a declared attempt limit, not an ETA. Reserve
time and disk for collection. All four epoch snapshots and canonical resume state
remain enabled. Twenty-four independent fits are not a request to use48 GPUs at once.

For a diagnosed interruption, rerun that cell's fit command with its original
output and `--resume "$WF_RUNS/$WF_ID/fit/training_state.pt"`; a new launch receipt
is retained. Resume is epoch-safe and requires unchanged config/source/data/runtime.
No best-valid checkpoint is substituted for finalE4.

## Fixed endpoint, complete validation and comparison

```bash
"$WF_PYTHON" "$WF_PACKET/runtime/launch.py" export "${COMMON[@]}" \
  --checkpoint "$WF_RUNS/$WF_ID/fit/model.pt" \
  --output "$WF_RUNS/$WF_ID/predictions" --devices 0,1 --timeout-seconds 7200

"$WF_PYTHON" "$WF_PACKET/runtime/launch.py" score "${COMMON[@]}" \
  --predictions "$WF_RUNS/$WF_ID/predictions" \
  --output "$WF_RUNS/$WF_ID/score.json" --timeout-seconds 7200
```

Require canonical fixedE4/source checks and all1,873 sequences/3,528,273 scored
rows. On export failure preserve completed fit weights and failure evidence. A
consumer repair that changes the bound source needs a separately reviewed source
migration; never silently bypass the source check. Compare a completed card:

```bash
PYTHONPATH="$WF_PACKET/runtime" "$WF_PYTHON" -m impact81.comparison \
  --card 04 --allow-one-card --output "$WF_RUNS/card04-comparison.json" \
  --scores "$WF_RUNS/ip04_control_s204101/score.json" \
  "$WF_RUNS/ip04_treatment_s204101/score.json" \
  "$WF_RUNS/ip04_control_s204102/score.json" \
  "$WF_RUNS/ip04_treatment_s204102/score.json"
```

For the complete panel omit `--card/--allow-one-card` and supply exactly24 score
paths. Report every seed/target delta, failures and exposure; no significance or
leaderboard forecast follows from these two-seed screens.

Import real receipts through [the MIDAS commands](midas/README.md). Keep
`launch-receipts/`, acceptance/profile decisions, `fit/run_metadata.json`,
`fit/split.json`, progress/monitor logs, terminal receipt, epoch index, fixed
weights, prediction manifest and score together. The MIDAS evidence archive may
omit weight bytes; in that case hashes are producer claims, not byte verification.

Cards04–06 add twelve fits with exactly the same stage commands. Select card05 control by its explicit ID: it uses value-copied depth initialization, not the plain control of another card. For a single completed new card, use `--card 04`, `05` or `06` and supply that card's four score paths. The default full comparison now expects24 endpoints.

Use a fresh results directory and acceptance receipt at this source revision. Keep any earlier running first-three-card job on its original commit. The new receiver check includes loss-denominator and depth/cache proofs before the per-cell50-update profile; no acceptance has been run here.
