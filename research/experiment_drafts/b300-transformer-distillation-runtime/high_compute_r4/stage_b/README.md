# R4 Stage B: six learning-rate fits after the current B300 cohort

Prepared handoff, **not started or admitted on B300**. This is Stage B of
[E/H-B300-HIGH-COMPUTE-R4-20261024](../PR.md), not a new hypothesis or a change to
the active PR40 cohort. The runtime baseline is merged `research-team` commit
`9ad4263f255ea56775881eb48ca52de4b73e3a72`; execution must pin the later full commit
that contains this handoff. The original R3 configs and PR40 execution plan stay
unchanged. Preparation was limited to source/config inspection; no local tests,
training, inference, GPU checks or profiles were run.

## Fixed panel

| Cell / complete config | Outer fold | Training seed | Peak LR | New fits |
|---|---:|---:|---:|---:|
| [B_lr_1e-4_f0](configs/B_lr_1e-4_f0.json) | F0 | 201001 | 0.0001 | 1 |
| [B_lr_1e-4_f2](configs/B_lr_1e-4_f2.json) | F2 | 201002 | 0.0001 | 1 |
| [B_lr_3e-4_f0](configs/B_lr_3e-4_f0.json) | F0 | 201001 | 0.0003 | 1 |
| [B_lr_3e-4_f2](configs/B_lr_3e-4_f2.json) | F2 | 201002 | 0.0003 | 1 |
| [B_lr_1e-3_f0](configs/B_lr_1e-3_f0.json) | F0 | 201001 | 0.001 | 1 |
| [B_lr_1e-3_f2](configs/B_lr_1e-3_f2.json) | F2 | 201002 | 0.001 | 1 |

Six new teachers, **two per LR**, 180 teacher-epochs, zero students and zero
reused endpoints. The `3e-4` pair is the newly trained reference. Within each fold,
all three arms share seed, data, split, normalization, batch and update horizon;
only peak LR changes. Source and environment must also be identical.

All six configs specify:

- Causal raw112 Transformer, approximately 1.54B parameters: width 1792, 40
  layers, 28 heads, FF multiplier 4, dropout 0, no hints or teacher targets.
- 30 epochs, fixed final checkpoint index 29; no early stopping or best-epoch selection.
- `bf16_bf16x9`: BF16 forward autocast, efficient exact-window attention,
  compiled Transformer blocks, no activation recomputation. FP32 master weights,
  gradients, loss and state remain; process-wide cuBLAS emulation may affect
  Muon GEMMs, so this is not an IEEE-FP32 equivalence claim.
- Muon momentum 0.95, five Newton–Schulz iterations, epsilon `1e-7`; auxiliary
  AdamW beta `(0.9, 0.95)`, epsilon `1e-8`. Decay 0.01 for matrices, no vector/bias
  decay; gradient clipping 1.0. Same RMS-matched LR convention for every arm.
- Warmup 5% of the actual optimizer-update horizon, then cosine to 10% of peak
  LR (`1e-5`, `3e-5`, `1e-4` respectively). The trainer derives the update count
  from the bound inner fitting membership; do not copy the W2 cohort's counts.
- Forward context 1024, detached chunk 512, accumulation 8. Eight ranks,
  batch 2 per rank: 16 sequences and at most 65,536 new rows per full update.
  Partial sequence/batch tails retain the canonical trainer semantics.
- Complete inner fitting membership, all-fit normalization and all post-warmup
  label rows. The legacy `normalization_sequences=32` key is retained for schema
  compatibility; `normalization_scope=all_fit` determines the actual membership.

The inner split holds out 20% of source-qualified components from each outer
fitting pool (split seeds 41001/F0 and 41002/F2). Every remaining fitting sequence
is used. The outer evaluation membership is excluded throughout Stage B.
Inner scores are development evidence, not new outer F0/F2 scores. Neither these
folds nor a new internal partition restore pristine evaluation data.

**W8 × batch2 is the frozen R4 starting recipe, not a measured optimum.** The
active W2/global128 teacher's throughput, memory and profile receipts do not
validate this shape. If profiling calls for a larger batch or a different rank
count, amend the entire six-cell panel under a new source/attempt and recompute
its schedule before fitting; do not change individual LR arms mid-panel.

## Files and source binding

- `configs/` contains all six complete production JSONs, with no host paths or
  unresolved config placeholders. `stage_inner.py` checks each against its
  canonical R3-derived recipe before materialization.
- [SHA256SUMS](SHA256SUMS) binds the exact config bytes.
- [host.env.example](host.env.example) supplies the documented data/split paths
  and new output namespaces; the operator fills the source SHA, GPU UUIDs,
  actual host-memory allocation and operator name in a copy outside Git.
- [prepare.sh](prepare.sh) checks source/config/outer-split identities, creates
  new inner splits and uses the existing R4 materializer. It then byte-compares
  the resulting production configs with the committed six. It starts no tests,
  profiles or fits. Keep a failed partial preparation; use new output paths for
  a diagnosed retry instead of overwriting files.

The canonical outer hashes come from the unchanged
[PR40 plan](../../execution_20261024/plan.json). Paths can be rebound on another
host, but changed manifest bytes require a reviewed binding amendment. The
materializer additionally checks dataset identities and split ancestry.

## Receiver setup, after the four existing jobs release the GPUs

The operator must first verify terminal fit **and evaluation** evidence for
`t1280_f0`, `t1280_f2`, `t1792_f0`, `t1792_f2` in
`pr32-teachers-20261024-r4-batch`, archive their outputs and confirm release of
all eight devices. A progress percentage or an idle-looking device is insufficient.
If a predecessor failed, preserve and diagnose it before reallocating its group.
This packet does not monitor, stop, resume, replace, or attach a scheduler to them.

Use a **separate checkout and virtual environment** for the new reviewed commit.
Do not `git pull`, switch branches, or `uv sync` in the live
`/data/epanov_s/tabdata/runtime/pr32-source-f1beeb4c` checkout. Do not copy its
acceptance/profile receipts or training states into R4-B.

On the receiving Linux B300 host, from the new clean checkout:

```bash
set -euo pipefail
PACKET=research/experiment_drafts/b300-transformer-distillation-runtime/high_compute_r4

# Copy stage_b/host.env.example to an external file, fill every SET_* field,
# pin the full published handoff SHA, and bind the released eight GPU UUIDs.
source /actual/private/path/r4-b.env
test "$(git rev-parse HEAD)" = "$WNN_EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
uv sync --frozen
export PYTHONPATH="$PWD/src"

# Reads real data to make component-clean inner splits; no fitting or tests.
bash "$PACKET/stage_b/prepare.sh"

# Receiver only: CPU contracts, actual CUDA recovery/DDP and BF16 checks.
bash "$PACKET/run.sh" check --campaign-root "$WNN_CAMPAIGN_ROOT"
bash "$PACKET/run.sh" list --campaign-root "$WNN_CAMPAIGN_ROOT" --stage B
```

`check` intentionally disables emulation in isolated strict-reference checks;
the BF16-specific checks exercise the amended mode. All production and profile
cell commands require `CUBLAS_EMULATE_SINGLE_PRECISION=1`. Receiver acceptance
and each cell's one/eight-rank and peak-LR profiles remain unexecuted prerequisites.
Do not infer that publication or static inspection has satisfied them.

## Execute the six cells explicitly

All six fits use the same eight released devices **sequentially**. Start one cell
only after the preceding cell has finished evaluation and bundling. These are
explicit commands for the operator; publication installs no automatic queue.
The helper stops on the first error, never retries and never chooses another LR.

```bash
set -euo pipefail
PACKET=research/experiment_drafts/b300-transformer-distillation-runtime/high_compute_r4
source /actual/private/path/r4-b.env
export PYTHONPATH="$PWD/src"
run_b_cell() {
  local cell="$1"
  case "$cell" in
    B_lr_1e-4_f0|B_lr_1e-4_f2|B_lr_3e-4_f0|B_lr_3e-4_f2|B_lr_1e-3_f0|B_lr_1e-3_f2) ;;
    *) return 2 ;;
  esac
  cmp -- "$PACKET/stage_b/configs/$cell.json" "$WNN_CAMPAIGN_ROOT/cells/$cell/config.json" || return
  for action in profile-one profile-eight profile-peak admit fit evaluate bundle; do
    bash "$PACKET/run.sh" "$action" --campaign-root "$WNN_CAMPAIGN_ROOT" \
      --stage B --cell "$cell" || return
  done
}

run_b_cell B_lr_1e-4_f0
run_b_cell B_lr_1e-4_f2
run_b_cell B_lr_3e-4_f0
run_b_cell B_lr_3e-4_f2
run_b_cell B_lr_1e-3_f0
run_b_cell B_lr_1e-3_f2

uv run --no-sync python "$PACKET/campaign.py" compare \
  --campaign-root "$WNN_CAMPAIGN_ROOT" --stage B
```

Each profile executes 12 real updates on B300. Admission requires all three
profiles, near-peak LR stability, at least 20% GPU headroom, and the actual RAM/
disk checks. Fit starts from fresh weights, not a profile checkpoint. A genuine
interruption can use the existing `run.sh resume` for that same cell/source;
do not re-run the whole helper over occupied output paths.

On a terminal failure, retain its logs and use the existing failure command
before comparison; do not silently retry or substitute an earlier checkpoint:

```bash
uv run --no-sync python "$PACKET/campaign.py" record-failure \
  --campaign-root "$WNN_CAMPAIGN_ROOT" --stage B --cell '<failed cell ID>' \
  --phase '<profile_one|profile_eight|profile_peak|admission|fit|evaluate>' \
  --reason '<numerical or infrastructure diagnosis>' \
  --evidence '<absolute preserved log or terminal receipt path>'
```

## Results and next decision

Keep each cell's `fit/model.pt`, `training_state.pt`, metrics/rank logs,
`fit/score.json`, evaluation predictions, profile/check/admission receipts and
separate evidence/weights bundles. Archive the whole campaign metadata and
external inner manifests as well; the per-cell bundle is not a complete backup
of all those directories. Store large artifacts privately, outside Git.

Primary contrasts are `1e-4 − 3e-4` and `1e-3 − 3e-4`, paired within each fold.
Report every fold delta, equal-fold mean, both output scores, fixed-fit component
intervals, failures and compute cost. A mean WP gain of at least .001 with no
fold regression is a useful predeclared signal, assessed with uncertainty; an
interval crossing zero does not by itself reject the direction. Two coupled
fold/seed cells cannot identify fold versus seed variation or establish reliable
statistical significance. This panel establishes neither student benefit nor a
public score. Retain the third, secondary pairwise comparison without treating
three contrasts as independent confirmations.

After all six fixed endpoints or diagnosed failures are recorded, choose a
completed LR with an explicit reason through `campaign.py select --stage B`.
Selection requires both fold cells for that LR to succeed. Stage C preparation
is a separate next action; no later-stage fit is launched by this handoff.
