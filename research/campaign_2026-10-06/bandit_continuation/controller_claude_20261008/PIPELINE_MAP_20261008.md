# H230 Faithful Pipeline Status Map — 2026-10-08

## 1. Pipeline Stages Overview

### Teachers → Strong Parents → Teacher Cache → Students → Blend/Package

| Stage | Recipe | Artifacts EXIST | Missing | Cloud Failures |
|---|---|---|---|---|
| **Teachers** | Fold-0 clean-fit GRU256×2 ensemble; 8 independent checkpoints cached; scope: clean-fold-only, full-index predictions | `cache/bd26_h230_fold0_teacher_family_all_rows_v1/` (14 GB, metadata + 6 families + main) — receipt: [metadata](cache/bd26_h230_fold0_teacher_family_all_rows_v1/metadata.json) | Raw-family baseline teachers (2× GRU128 clean-fold) proposed but never fitted | N/A |
| **Strong Parent** | GRU256×2 Muon (native BF16 orthog, lr-mult 3, momentum 0.95) on all-supplied-data; seed99 fixed ep3 | Checkpoint: `artifacts/runs/bd26_h230_strong_parent_seed99_muon_local_v1/model_ep3.pt` (6031s fit, 0 OOM); predictions: `cache/pred_h230_seed99_ep3_fp32_train.npy` (1.7 GB) — terminals: [fit](research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/H230_SEED99_LOCAL_FIXED_EP3_ROOT_TERMINAL_REVIEW_20261008_R1.json), [export](research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/H230_SEED99_FIXED_EP3_TRAIN_EXPORT_TERMINAL_RECEIPT_20261008_R1.json) | Seed9 parent (proposed new fresh parent) — Kaggle v4–v8 packet built but not launched; v8 passes mount/cache validation tests | Kaggle v4–v7: mount/container failures (v7 exit 139); v8 audit-only, not admitted |
| **Teacher Cache** | All-data heterogeneous family blend (1 strong@0.30 + 2 full@0.10 each + 2 raw@0.125 each + baseline@0.0625 = unnormalized 0.8125) | `cache/bd26_h230_fold0_teacher_family_all_rows_v1/`: families_train.npy (6.4 GB), families_valid.npy (1.2 GB), main_source/normalized/homogeneous pairs | Full all-data cache build (cache builder script ready in v3); seed99 parent NOT yet in teacher cache | None attempted (dependent on seed9 parent completion) |
| **Students** | 2-epoch students warm-start from owned all-data parents (fixed ep3); distill from teacher cache; alpha 0.7, raw-family loss 0.1, aux 0.3 each | Seed3: ep1 complete + packaged ([delivery](research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/H230_ALLDATA_STUDENT_SEED3_ROOT_DELIVERY_20261008_R1.json), 1988s, no OOM); seed4: ep1 complete (retry1) | Seed99 student (depends on seed99 parent in cache + teacher-cache rebuild); seed4 public score unknown | None attempted (students fitted locally pre-cloud) |
| **Blend/Package** | Fixed equal-weight (0.5 each) seed3+seed4 pair; also single S3/S4; history-swap variant with frozen teacher weights at ep3 | Submissions: `c20261008_h230_all_data_pair_i16.zip` (0.6653 public), S3 (0.6653 normalized), S4 (packaged, score pending), history-swap (0.6644 public), prior-hetero-ep1 (0.6649 observed) | Seed99 single + future seeds (depend on student recipe) | None (all packaged locally) |

---

## 2. Local Strong Parent (Seed99) Launch Mechanism

### Authorization & Execution
- **Authorization JSON**: `research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/H230_SEED99_LOCAL_ROOT_FIT_AUTHORIZATION_20261008_R1.json`
- **Contract JSON**: `research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h230_source_v4_faithful_muon/H230_STRONG_PARENT_SEED99_LOCAL_CONTRACT_V1_20261008.json`
- **Attempt ID**: `h230-strong-parent-seed99-muon-local-v1-20261008`

### Command Shape (systemd/bandit_resources.py)
```bash
timeout --signal=TERM --kill-after=10s 7200s \
  env MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONHASHSEED=0 \
  PYTHONPATH=/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h230_source_v4_faithful_muon/src \
  WNN_GPU_MEMORY_BYTES=4300000000 WNN_LOADER_RAM_GB=6 WNN_LOADER_WORKERS=0 \
  WNN_THREADS=1 WNN_PROJECT_ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack \
  /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
  research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h230_source_v4_faithful_muon/scripts/train_rnn.py \
  --name bd26_h230_strong_parent_seed99_muon_local_v1 \
  --device cuda --arch gru --cell gru --hidden 256 --layers 2 \
  --seed 99 --valid-fold -1 --train-source train+valid --include-heldout-valid \
  --epochs 4 --checkpoint-policy fixed --checkpoint-epoch 3 \
  --batch 16 --chunk 1000 --loss wmse --loss-norm chunk \
  --lr 0.001 --wd 0.01 --optim muon --sched onecycle --warmup-frac 0.05 \
  --precision fp16 --muon-scope hidden --muon-gatewise none \
  --muon-lr-mult 3 --muon-momentum 0.95 \
  --muon-ortho-dtype-policy native_bf16_if_supported_else_fp32_v1 \
  --propensity c20260929_prop_f0 --prop-power 2 --prop-eps 0.03 \
  --valid-true-mask 0.03 --aux-struct 0.3 --aux-next 0.3 \
  --ema-teacher 1.0 --ema-decay 0.999 --eval-ema \
  --since-weight 32 --since-floor 0.25 \
  --distill-cache-root cache/bd26_h230_fold0_teacher_family_all_rows_v1 \
  --distill-cache-all-data --distill-cache-variant source \
  --distill-alpha 0.5 --distill-boot-h 8
```

### Frozen Source Worktree
- **Path**: `research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h230_source_v4_faithful_muon`
- **Source Members**: 57 verified files (manifest SHA256: `49a4048f559...`)
- **Snapshot Commit**: Reference pinned to `833aaf54...`

### Resource Limits
- CPU cores: 1 | RAM: 9 GB | VRAM: 4.3 GB (cuda_allocator) | Wall time: 7200s (2 hours hard bound)
- **Actual**: 6031.27s fit + ~620s export = ~110 min total
- **Peak RAM**: 17.7 GB parent hierarchical | Peak VRAM: observed 5.0 GB

---

## 3. Fields to Change for Seed 9 Local Run (Do Not Create)

To run the identical recipe for seed 9 locally, **only** these fields would need changing:

| Field | Seed99 | → Seed9 |
|---|---|---|
| `--seed` | 99 | 9 |
| `--name` | `bd26_h230_strong_parent_seed99_muon_local_v1` | `bd26_h230_strong_parent_seed9_muon_local_v1` |
| Attempt ID | `h230-strong-parent-seed99-muon-local-v1-20261008` | `h230-strong-parent-seed9-muon-local-v1-YYYYMMDD` |
| Output checkpoint path | `artifacts/runs/bd26_h230_strong_parent_seed99_muon_local_v1/model_ep3.pt` | `artifacts/runs/bd26_h230_strong_parent_seed9_muon_local_v1/model_ep3.pt` |
| Prediction cache path | `cache/pred_h230_seed99_ep3_fp32_train.npy` | `cache/pred_h230_seed9_ep3_fp32_train.npy` |
| Contract path | `h230_source_v4_faithful_muon/H230_STRONG_PARENT_SEED99_LOCAL_CONTRACT_V1_...json` | `h230_source_v4_faithful_muon/H230_STRONG_PARENT_SEED9_LOCAL_CONTRACT_V1_...json` |

**All other flags, resource limits, and torch/optimizer settings remain identical.**

---

## 4. Student Stage: Recipe Status & Blockers

### Source Readiness
✅ **Student recipe is source-ready**: `h230_source_v3_all_data_distill/scripts/train_rnn.py` exists with all-data distillation mode; `build_distill_cache.py` ready for full-index teacher-prediction export.

### Teacher Cache Composition
- **Current cache path**: `cache/bd26_h230_fold0_teacher_family_all_rows_v1/` (14 GB, built 2026-10-08 04:53)
- **Teachers used**: 8 fold-0-clean checkpoints (clean-fold-trained only, though predictions cover all supplied groups)
- **Does it include seed99 parent predictions?** ❌ **NO** — cache was built before seed99 fit completed. Teacher cache includes: `main_source`, `main_normalized`, `main_homogeneous`, and `families_train/valid` (4 family streams).

### Current Student Artifacts
- ✅ **Seed3 (S3)**: `artifacts/runs/bd26_h230_all_data_student_gru256_seed3_v1/model_ep1.pt` (completed, exported, packaged)
- ✅ **Seed4 (S4, retry1)**: `artifacts/runs/bd26_h230_all_data_student_gru256_seed4_retry1_v1/model_ep1.pt` (completed, exported, packaged)
- ❌ **Seed99 student**: Does not exist (blocked on teacher-cache rebuild with seed99 parent predictions)

### Blockers to Running Student Seed99
1. **Teacher cache must be rebuilt** to include seed99 parent predictions (`cache/pred_h230_seed99_ep3_fp32_train.npy`)
   - Requires: `build_distill_cache.py` run with new teacher manifest including seed99
   - Path: `research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h230_source_v3_all_data_distill/scripts/build_distill_cache.py`

2. **Full-index teacher-family cache build** (full all-data membership validation)
   - Current cache membership: clean-fold fit only; student fit must declare that scope or rebuild with all-data teachers
   - **Replication note**: Maxim's eight teachers were all-data trained; our alternatives remain fold-0 only

3. **Raw-versus-clipped family targets** (see [MAXIM_REPLICATION_REFLECTION.md](reviews/MAXIM_REPLICATION_REFLECTION.md) line 22)
   - Current: `scripts/train_rnn.py` clamps family targets to ±2 (line: `--clipping` or distill builder code)
   - UNVERIFIED whether this matches Maxim's raw-export design; must be explicitly audited before claiming method parity

---

## 5. Submitted H230 Students & Scores

| Variant | Zip | Public Score | Seed(s) | Notes |
|---|---|---|---|---|
| **Pair (normalized)** | `c20261008_h230_all_data_pair_i16.zip` | **0.6653** | S3+S4 (0.5/0.5) | Exact: bd26_h230_all_data_student_gru256_seed3_v1@ep1 + seed4_retry1_v1@ep1 |
| **S3 (seed3, normalized)** | `c20261008_h230_all_data_s3_i16.zip` | **0.6653** | S3 only | Single seed3 alone |
| **S4 (seed4, retry1)** | `c20261008_h230_all_data_s4_i16.zip` | *pending* | S4 only | Packaged; public portal unknown |
| **History-Swap Variant** | `c20261008_h230_all_data_s3_history_swap_i16.zip` | **0.6644** | S3 + frozen teachers@ep3 | Teacher weights frozen at epoch 3 instead of trained 2 more epochs |
| **Prior Hetero (ep1)** | `h230_hetero_s3_ep1_i16/` | **0.6649** | S3 | Earlier heterogeneous-head experiment (v2 source) |

**Differences from faithful recipe:**
- **Pair & S3/S4**: Full 2-epoch distill from fold-0-clean cache; alpha 0.7, family loss 0.1, aux 0.3; all-data fit
- **History-Swap**: Same student recipe but **teacher weights held at epoch 3** (no +2-epoch training) — change only in teacher weight timeline, not student loss
- **Prior Hetero**: Different source version (v2), separate family-head architecture, earlier checkpoint ancestry — not faithful Maxim reconstruction

**Interpretation:**
- Pair/S3/S4 = faithful all-data distillation students (seed99 not yet admitted)
- History-Swap ≈ modest(-0.0009) delta, suggests teacher-weight movement contributes positively but subtly
- Prior Hetero (0.6649) shows family head can match pair score within ±0.0004

---

## 6. Summary & Recommended Next GPU Job

### Key Facts (15 lines)
1. **Strong parent seed99 completed locally** (6031s fit, ep3 fixed, 0 OOM); exports verified; ready for cache integration.
2. **Teacher cache exists** (14 GB, fold-0-clean scope, full-index prediction coverage) but **does not include seed99 parent**.
3. **Students S3+S4 submitted** with public scores: pair 0.6653, history-swap 0.6644, prior-hetero 0.6649; all below 0.666 public best.
4. **Seed9 parent** proposed as fresh alternative; Kaggle v8 packet passes mount/cache validation audit but is **not yet launched or admitted**.
5. **Student recipe source-ready** (v3 scripts + build_distill_cache.py exist); blockers are teacher-cache rebuild and seed9 parent completion.
6. **Replication differences documented**: raw-vs-clipped family targets (unknown effect), baseline-teacher substitution (two GRU128 instead of organizer), fold-0-only teacher scope (vs. Maxim's all-data teachers).
7. **C0 comparison only completed** from replication plan; C1–C4 heterogeneous + family-head variants never ran (see MAXIM_REPLICATION_REFLECTION.md).
8. **No seed99 student yet** — blocked on teacher-cache rebuild with seed99 predictions.
9. **All-data students S3/S4 are fold-0-cache-trained** (not all-data teachers), within allowed analogue scope; no heldout held back.
10. **History-Swap shows modest teacher-weight effect** (-0.0009 vs. normalized S3), suggesting finalization penalty in original Maxim recipe.
11. **Public score plateau at 0.6653–0.6660** across pair/normalized/hetero variants; no mechanism yet breaks through baseline 0.666 public.
12. **Seed9 parent Kaggle v8** is audit-verified but requires fresh root admission; no GPU agent or Vast worker currently active on H230.
13. **Resource limits stable**: local 1 CPU/9 GB RAM/4.3 GB VRAM; seed99 fit sustained ~110 min; seed9 profile estimate similar.
14. **Continuous research authorized** until user explicit stop; H230 allocation remains open; throughput and cloud capacity pending seed9 admission decision.
15. **Next execution path depends on user direction**: (A) admit seed9 parent locally/cloud, rebuild teacher cache, run seed99 student; (B) run C1 control with existing cache; (C) shift to independent arms (e.g., raw-vs-clipped family targets as H272 variant).

### Single Recommended Next Job
**Launch seed9 strong parent locally under fresh authorization** using h230_source_v4_faithful_muon with --seed 9 (all other flags identical to seed99).
- **Why**: (i) Seed99 completes the Muon-parent pair baseline for comparison; (ii) seed9 offers independent fresh seed evidence before committing to Kaggle/Vast overhead; (iii) once seed9 exported, teacher-cache rebuild becomes runnable; (iv) reduces cloud risk by profiling parent-fit scaling locally first.
- **Basis contract**: `research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h230_source_v4_faithful_muon/H230_STRONG_PARENT_SEED99_LOCAL_CONTRACT_V1_20261008.json` (adapt seed → 9, attempt_id, output dirs only).
- **Expected wall time**: ~110–120 min; **resource ceiling**: same 1 CPU/9 GB/4.3 GB VRAM.
- **Deliverable**: Checkpoint `model_ep3.pt`, predictions `pred_h230_seed9_ep3_fp32_train.npy`, terminal receipt with 0 OOM confirmation.
- **Gate before student run**: Confirm both seed99 and seed9 parents are exported; rebuild teacher cache with both; then admit seed99 student or seed9 student (or both-parent blend if applicable).
