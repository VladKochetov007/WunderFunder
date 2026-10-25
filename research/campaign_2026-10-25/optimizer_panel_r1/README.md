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

## F2 Muon and Cautious ([partial grade](grade_f2_partial_R1.json)) — Muon replicates

| Arm | Official WP | Paired Δ vs AdamW | Class 3 |
|---|---:|---|---:|
| AdamW | 0.683091 | — | 0.43985 |
| Muon (rnn), strict FP32 | 0.687427 | **+0.004336 [+0.001969, +0.007153]** | 0.44467 |
| Cautious AdamW | 0.682020 | −0.001071 [−0.002813, +0.000514] | 0.43815 |

**Muon: +0.0038 (F0) / +0.0043 (F2), mean +0.0041, both intervals exclude zero** — the largest
replicated gain this session, above the ~0.001 seed-noise floor; class 3 improves in both
cells. Cautious AdamW negative in both cells: closed. AdEMAMix F2 pending.

## Successor: Muon on the production student lineage (`wnn-muon6`, queued after clip panel)

Stage-6 warm continuation (init `c20261003_combo_s3_f0@ep3`, 6 epochs, LR 1e-4, distillation
α .9, fixed CP5) with `--optim muon --muon-scope rnn --muon-ortho-dtype fp32
--strict-muon-fp32` ([argv](stage6_muon_f0.argv), [runner](run_stage6_muon.sh)) vs the reused
strict-FP32 AdamW stage-6 control (0.6812108). Same frozen overnight source as the control:
that tree's Muon defaults to BF16 Newton–Schulz on CUDA (`auto`), so the explicit fp32 flags
are required — caught before launch; a cherry-pick of the main-branch fix was unnecessary and
discarded.
