# Literature notes: LR schedules, staged distillation and tail weighting (2026-10-09)

Root (Claude Code) research note. Maps primary sources to this project's trainer, records what we
already implement, and binds each idea to a queued or completed matched experiment. Evidence levels:
**measured** (our runs), **queued**, **idea** (not yet implemented).

## Learning-rate schedules

| Source | Main claim | Our status |
|---|---|---|
| [Smith & Topin, Super-Convergence / 1cycle](https://arxiv.org/abs/1708.07120) | One cycle with a large peak LR trains faster; large LR regularizes | Our default `onecycle` = PyTorch OneCycleLR: 5% warmup, cosine anneal to ≈0 (peak/2500), beta1 cycling 0.85↔0.95 |
| [Loshchilov & Hutter, SGDR (ICLR 2017)](https://arxiv.org/abs/1608.03983v5) | Cosine annealing with warm restarts improves anytime performance | `sgdr`; **measured** slightly worse than WSD in small screens on folds 0 and 3 |
| [Hägele et al., Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations (NeurIPS 2024)](https://arxiv.org/pdf/2405.18392) | Constant LR + short cooldown (WSD) matches cosine; 1−sqrt cooldown beats linear; SWA/schedule-free strong but not optimal | `wsd` with 1−sqrt cooldown. **Measured:** +0.0136 (fold0) and +0.0134 (fold3) over onecycle in 220-update small screens; **tie** for full-scale fold1 students (+0.00004); scratch fold1 teacher +0.0011, CI [−0.0025, +0.0047] (one seed each, inconclusive) |
| [Bergsma et al., Straight to Zero: linear decay to zero (ICLR 2025)](https://arxiv.org/pdf/2502.15938) | Linear decay-to-zero beats cosine-to-10%; benefit grows with tokens per parameter | Added `--wsd-shape linear` (main `src/wnn/optim.py`, tested) and snapshot `continuous_execution_E529/h230_source_v3_lineard2z`. **Queued:** matched from-scratch fold1 teacher |
| [Zhai et al., Scaling Vision Transformers (CVPR 2022)](https://arxiv.org/abs/2106.04560) | Trapezoidal schedule: warmup, constant, linear cooldown (the WSD precursor named in Hägele et al.) | `--sched wsd --wsd-shape linear --wsd-decay 0.2 --wsd-final 0`. **Queued:** matched from-scratch fold1 teacher |
| [Defazio et al., The Road Less Scheduled / Schedule-Free AdamW (NeurIPS 2024)](https://arxiv.org/pdf/2405.15682) | No schedule needed; interpolation + averaging replaces momentum; needs its own LR/WD sweep | **Idea** (not implemented; would need an independently written optimizer) |
| [Kalra & Barkeshli, Why Warmup the Learning Rate? (NeurIPS 2024)](https://arxiv.org/abs/2406.09405) | Warmup mainly allows a larger target LR; longer warmup adds little | Keep short warmup; not a tuning priority |
| [Ibrahim et al., Simple and Scalable Strategies to Continually Pre-train LLMs (TMLR 2024)](https://arxiv.org/pdf/2403.08763) | Re-warm then re-decay LR when continuing training; re-warming can cause forgetting | Our warm-started students re-warm (onecycle from parent) at peak 1e-4; consistent with the full-scale schedule tie |

Interpretation: schedule shape matters strongly in very short training (small screens) and little for
warm-started 2-epoch distillation fine-tunes. From-scratch teachers are the remaining place where it can
matter; single-seed teacher noise is about ±0.002 (rerun of the identical recipe: 0.66557 vs 0.66742), so
small schedule effects need multiple seeds.

## Staged / curriculum distillation

| Source | Main claim | Our status |
|---|---|---|
| [Jin et al., Knowledge Distillation via Route Constrained Optimization (ICCV 2019)](https://arxiv.org/abs/1904.09149) | Distill from anchor points on the teacher's training route (easy → hard) instead of only the converged teacher | **Queued** two-stage analogue: broad (all four families) → strong-only, and strong → broad; one epoch per stage, family-head core conversion between stages |
| [Mirzadeh et al., Improved Knowledge Distillation via Teacher Assistant (AAAI 2020)](https://arxiv.org/pdf/1902.03393) | Large teacher–student gaps hurt; intermediate teacher assistants bridge them; a stronger teacher can produce a worse student | Motivates testing teacher *sets* rather than only the strongest teacher; relevant for future larger teachers (GRU512/768) |
| [Jafari et al., Annealing Knowledge Distillation (EACL 2021)](https://arxiv.org/abs/2104.07163v1) | Feed annealed soft targets gradually | **Idea**: alpha schedule over training (trainer currently has fixed `--distill-alpha` and noise-adaptive `--distill-alpha-far`) |

## Tail emphasis and target distribution

Measured on scored validation rows (rows ≥99, mask): targets clipped to ±2, std 0.89; 20% exact zeros
(zero metric weight); |y|>0.5 rows are 58% of rows and 91% of metric weight; |y|>1 rows are 23% of rows and
55% of weight; 3.5% clipped at ±2. The official metric and our weighted MSE both weight rows by |clip(y)|.

**Queued** in snapshot `continuous_execution_E529/h230_source_v4_tailweight` (defaults reproduce v3/v2):
- `--label-weight-power 1.5`: distillation weights |clip(y)|^1.5 (tails beyond metric weighting).
- `--label-weight-floor 0.1`: weights max(|clip(y)|, 0.1) so zero-label rows keep teacher signal.

## Teacher research directions (from this session)

1. Teacher diversity via an expanded 12-teacher cache (adds s100, w320, chunk1000, te07 chosen by the
   cross-fold blend analysis) — **queued**, matched seed3 student.
2. Larger teachers (serving limit does not apply to teachers) — **idea**, Kaggle T4 or local GRU384/512.
3. Cross-fitted (out-of-fold) teacher targets instead of in-sample teacher predictions — **idea**;
   trainer supports inner cross-fit manifests.
4. Privileged (non-causal) teachers distilled into causal students — **idea**; needs leakage audit.
5. fixed_mix activation (top of every small screen on folds 0 and 3) — **queued** as a from-scratch teacher.

Related notes: [combinations and sampled queue](COMBINATIONS_AND_SAMPLED_QUEUE_R1.md),
[combination evidence](COMBINATION_EVIDENCE_20261009_R1.md), continuation log
`controller_claude_20261008/CODEX_CONTINUATION_LOG.md`.
