# Optimizer panel — 2026-10-25 R1

User: "go crazy with research. what's with custom optimizer?" and "think a lot about theory and
optimization methods". [Theory note](OPTIMIZER_THEORY_R1.md) fixes the measured regime
(noise-dominated labels, early peak, 94% clipped steps, schedule ties) and maps optimizer
families onto it.

- New code: [`src/wnn/optim_variants.py`](../../../src/wnn/optim_variants.py) — `CautiousAdamW`
  (2411.16085) and `AdEMAMix` (2409.03137), `--optim cautious_adamw|ademamix`; exactness tests
  (cautious step 1 = AdamW; AdEMAMix α=0 = AdamW; β3 schedule endpoints). Commit `7161d0eef`,
  frozen worktree `research_tmp/optimizer-screen-20261025`.
- Six fits ([plans](PLANS_R1.json), [runner](run_panel.sh)): Muon(rnn, strict FP32),
  Cautious AdamW, AdEMAMix(α4, β3 .999, full warmup) × F0/s1001, F2/s1002; controls reused
  exactly from the [i1 lead screen](../i1_lead_panel_r1/README.md) zero-block fits. Screens
  [F0](screen_f0.json)/[F2](screen_f2.json). Running after the lead screen.
- Execution note: a first queue start launched briefly while the lead unit was in a failed
  state; it was stopped before any checkpoint and the empty run directory removed.

## Interim: strict-FP32 Muon, F0 ([grade](grade_f0_muon_only_R1.json))

AdamW control 0.659530 → Muon(rnn) **0.663349**, paired **+0.003818 [+0.000562, +0.007782]**;
class 3 0.39129 → 0.39347. Matches the historical bf16-era Muon gain (~+0.004) and contradicts
the Oct-24 synthetic screen: the strict-FP32 fix did not remove Muon's real-data advantage.
F2 cell pending; one seed per cell.

## F0 panel ([grade_f0_R1.json](grade_f0_R1.json))

| Arm | Official WP | Paired Δ vs AdamW [95% CI] | Class 3 | Public-mix |
|---|---:|---|---:|---:|
| AdamW (control) | 0.659530 | — | 0.3913 | 0.6195 |
| Muon (rnn), strict FP32 | 0.663349 | **+0.003818 [+0.000562, +0.007782]** | 0.3935 | 0.6238 |
| Cautious AdamW | 0.658172 | −0.001359 [−0.003956, +0.001343] | 0.3889 | 0.6182 |
| AdEMAMix (α4, β3 .999) | 0.655013 | **−0.004517 [−0.007931, −0.000390]** | 0.3852 | 0.6148 |

AdEMAMix telemetry: clip rate rises 0.85 → 0.99 and mean clip factor falls to 0.35 by epoch 2,
i.e. the slow EMA (α=4) multiplies the effective step at the AdamW LR; this result is
"AdEMAMix at AdamW's LR", not a test of its mechanism at a matched effective step. Theory
link for Muon: orthogonalization discards per-step gradient magnitude, so its recurrent
matrices are immune to clipping's implicit batch reweighting — part of its gain may be robust
normalization; the clip panel helps separate this. F2 cells pending.
