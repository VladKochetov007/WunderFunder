# L4 objective and clipping follow-up — 4 November2026

[Contract](CONTRACT_R1.json): maximum six new fits, four scratch teachers and up
to two students from one selected family. Existing PR61 and PR60 controls are
reused, and the current twelve-fit cycle remains unchanged. Actual execution is
not established until receiving profile/fit receipts exist.

The first teacher arm changes clip1 to5. A fitting-only, no-update gradient
diagnostic determines whether the second arm uses late auxiliary-loss annealing
or half peak LR. Every new teacher retains GRU256x2, native211 inputs, four
epochs/fixedep3 and the same seed pair. The optional family/student selection
rule is frozen in the contract and remains exposed-development selection.

- [Diagnostic entrypoint](../../../scripts/l4_teacher_capacity_20261104/gradient_decomposition_20261104.py),
  [gradient calculations](../../../scripts/l4_teacher_capacity_20261104/gradient_decomposition_core.py)
  and [checkpoint loader](../../../scripts/l4_teacher_capacity_20261104/gru_checkpoint_io.py)
  reuse the canonical model/input/objective code and preserve fitted parameters.
  Both seeds at ep2/ep3 see the same eight fitting-only batches/windows.
- [Auxiliary coefficient schedule](../../../src/wnn/training/auxiliary_schedule.py)
  is opt-in and default-off. It preserves the allocated auxiliary heads and
  seeded initialization; the configured weights stay unchanged for the first
  half of updates and decay tozero at the last update.
- [Prior clipping/objective audit](../l4-teacher-capacity-20261104/CLIPPING_AND_OBJECTIVE_AUDIT_R1.md)
  separates synthetic optimizer behavior from real quality and explains why a
  large head gradient does not identify which loss caused it.
- [Completed PR60 screen](../w18-l4-quality-20261104/RESULTS_20261104_R1.md)
  found no useful improvement from the tested alpha/family/innovation variants;
  those recipes are not repeated here.

All primary comparisons use official train-only fitting and complete official
validation under ADR0044. No random replacement split, teacher valid ancestry,
changed scoring rule, quantization, or numerical public forecast is admitted.
Current status: code/preparation; GPU diagnostic, profiles and new fits pending.
