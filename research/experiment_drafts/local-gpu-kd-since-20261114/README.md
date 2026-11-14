# Local GPU research log, from 2026-11-14: cold one-stage distillation × since-weighting on a 40-minute GRU cell

**Status: running.** Owner: Vlad / Claude Code. A live log of local-GPU experiments (one RTX 3060 Laptop,
6 GB) with all ideas, triage, code and per-epoch results. Updated as cells finish.
**Results table: [RESULTS.md](RESULTS.md)** (generated from the fit logs in this folder).

## Why this exists

Iteration speed. One cold fit of the recipe below takes about 40 minutes and reaches 0.687 on full valid, so
a factorial that needs an L4 wave can be answered locally in an evening. The first question is PR #92 B03:
do since-weighting and distillation add up or substitute?

## Recipe (frozen; `recipe_args.sh` BASE + PARENT, launched by `run_cell.sh`)

GRU H256×2 (0.98M parameters), raw inputs, Muon on recurrent matrices + AdamW elsewhere, LR 1e-3, onecycle
with 5% warmup, batch 32 × chunk 1000, 4 epochs, fixed last epoch, WMSE, `--since-weight 32 --since-floor .25`,
auxiliary next/struct heads 0.3, strict IEEE FP32 (`strict_launch.py`, TF32 off). Trainer source: PR49 legacy
runtime `188c0788e`. Fit on the 10,325 train-only sequences (split `0c4de3a2…`, 282 linked sequences excluded),
evaluate all 1,873 valid sequences / 3,528,273 scored rows with `global_wp_clip2_v1`. Teacher for distillation:
our local reproduction of the PR49 `fbc1k_s5` teacher (valid 0.69142), feature-only train predictions;
target = (1−α)·label + α·teacher.

Differences from Semyon's L4 contract: Muon at LR 1e-3 instead of AdamW 3e-4, 4 epochs instead of 6, no K16
stem, auxiliary heads on. Numbers here are not directly comparable with PR #95/#96 cells; paired deltas are.

## Findings so far (seed 3, see RESULTS.md for every epoch)

- **Cold one-stage distillation works: +0.0038** (0.69047 vs 0.68671). It beats our Oct 29 two-stage chain
  (parent → distilled child, 0.68997) in one 40-minute fit and sits 0.001 below its teacher.
- **Distillation removes the last-epoch regression** that label-only runs show (0.6874 → 0.6867).
- **since32 is worth +0.0008 on this Muon recipe** without a teacher (0.68671 vs 0.68591), less than the
  +0.0029 on the AdamW contract (PR #95).
- **With a teacher, since32 adds more, not less** at epochs 1–3 (+0.0018, +0.0019, see RESULTS.md); the
  final cell of the 2×2 lands with `l14_kd50_nosince_s3`.
- Zero-fit: the three local teachers' residuals correlate 0.995, so a three-teacher mean target is not run.

## Operations

`run_queue.sh` runs `queue.txt` one cell at a time (a single fit already uses ~94% GPU, 3.0 GB VRAM); each
cell is a systemd scope with a hard memory cap and no swap. `thermal_guard.sh` freezes the fit at CPU ≥ 95 °C.
`make_results.py` regenerates RESULTS.md. Fit logs (`*.fit.log`) carry the full config, per-epoch gradient
statistics and valid scores. Weights and caches stay on the local machine.

## External ideation round 1 (Freebuff 0.2.x, DeepSeek V4.1 Flash, 2026-11-14)

Prompt: [FREEBUFF_TASK_R1.md](FREEBUFF_TASK_R1.md) (anonymized problem statement only). Output:
[FREEBUFF_IDEAS_R1.md](FREEBUFF_IDEAS_R1.md). Triage against the ledger:

| # | Idea | Status |
|---|---|---|
| 1 | Differentiable weighted-correlation loss | closed: H-018 refuted (E-023), gcorr fine-tunes tested |
| 2 | Ordinal/categorical head | closed: CE readout family H-046 and successors |
| 3 | Train on the unrounded event-clock mean | **new**; needs a zero-fit check of the quantization share of error first |
| 4 | Learned per-row noise scale (NLL) | tested: H-NLL-STRONG-GRU / H-LOSS25-FAITHFUL-NLL |
| 5 | Per-event increment heads | already in the recipe as `--aux-next` |
| 6 | Stateful TBPTT | already how the trainer chunks sequences |
| 7 | Density-ratio reweighting | refuted Sep 26 (n_eff ~ 1e-5) |
| 8 | Since-phase specialists | **new as since bands**; zero-fit diagnostic: valid WP by since decile (same as PR #93 C04) |

## External ideation round 2 (Freebuff, DeepSeek V4.1 Flash): distillation

Prompt [FREEBUFF_TASK_R2.md](FREEBUFF_TASK_R2.md), output [FREEBUFF_IDEAS_R2.md](FREEBUFF_IDEAS_R2.md).

| # | Idea | Status |
|---|---|---|
| 1 | Per-row mixing weight by label noise | trainer already has `--distill-alpha-far/-tau` (more teacher far from a label change); queued as `l14_kdfar_s3` (0.3 near, 0.8 far, tau 32) |
| 2 | Mean of three teachers | **killed zero-fit**: |y|-weighted residual correlations between the three teachers are 0.9952-0.9955 on 400 train sequences (`teacher_resid_diag.py`), so the mean target is no quieter than one teacher; cell removed from the queue |
| 3 | Cross-fitted residual gate (GBM on y - t) | open; needs the scored-row mask for train, which this cache does not carry |
| 4 | Second round with a label/teacher/student vote | open; needs a train export of the round-1 student |
| 5 | Teacher on rank-transported inputs | open; expensive, proxy-shift test first |
| 6 | Teacher fraction rising over epochs | open; needs a two-line trainer change |

Caveat on `teacher_resid_diag.py`: the train target cache has an empty mask array, so the diagnostic includes
unscored rows and its absolute WP (0.51) is not the scored-row WP. The residual correlations between
teachers do not depend on that.

## Next

Queue order is in RESULTS.md. Planned after the queue: teacher fraction rising over epochs (needs a two-line
trainer change in a separate source copy), second distillation round from the round-1 student, PR #93 C04
(since-band readouts) as a zero-fit check.
