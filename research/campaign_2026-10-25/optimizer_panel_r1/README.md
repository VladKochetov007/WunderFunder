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
