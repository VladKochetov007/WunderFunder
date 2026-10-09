# Maxim-replication status addendum — 2026-10-07

This addendum updates the execution state after the explicit research resume;
the original replication reflection remains a historical snapshot and is not
rewritten. The user-reported 0.6674 submission is still ahead of our public
best of 0.6660. No new public submission or gain has been produced.

## What was and was not tested

Our completed C0 was ordinary supervised continuation. It did not use
heterogeneous teacher targets or Maxim's family head. E-511 C1 was a
homogeneous main-teacher distillation control. Its exposed fold-0 point
comparison is positive but imprecise; the interval crossing zero does not
reject the teacher-target direction and cannot estimate public transfer.

The closest heterogeneous C3 analogue did not train. The structurally similar
family-head parameter partition is not the key discrepancy. The proposed
recipe differs from Maxim's source in teacher ancestry and composition, the
replacement of one organizer-baseline source with two independently trained
raw-only GRU128 sources, and clipping family predictions to ±2 where Maxim
used raw family predictions. Initialization/RNG conventions also differ.
Those are declared treatment differences, not demonstrated explanations of a
score gap.

## Execution bottleneck

E-508-v5 verified real Kaggle inputs and completed a short T4 profile, then
correctly stopped before fitting: its projected pair fit/export time was 8,524
seconds against that packet's frozen 6,300-second ceiling. No student fit,
scientific score, or OOM occurred. E-510 cannot simply be reused: its builder
expects the v1 receipt, emits unsupported `--device cuda:0`, and embeds the
old runtime ceiling. Thus the main failure was that the relevant heterogeneous
recipe never reached a valid fit, not that an evaluated replication lost.

E-520 is currently a **plan only** for an independently trained raw-baseline
pair. It can recover a missing own-source prerequisite, but it is not itself a
replication of Maxim's heterogeneous student. Its profile needs fresh quota,
actual-device measurement, and independent admission before any fit. After
that prerequisite, the controller must own one end-to-end heterogeneous
student/control comparison with raw-versus-clipped family targets made an
explicit treatment, plus complete prediction export and independent grading.

## Allocation consequence

Keep the heterogeneous-teacher direction unresolved and eligible. Do not use
the C0 interval, E-508 runtime rejection, code-only milestones, or missing
artifacts as a negative score reward. Treat infrastructure failures as cost
and reliability evidence. Do not claim an exact replication until the actual
recipe, source ancestry, complete run and comparison are delivered. See the
[current allocation](../RESUMED_ALLOCATION_20261007.md),
[E-508-v5 profile receipt](../raw_teacher_pair/E508_V5_TERMINAL_PROFILE_RESULT_20261007.md),
and [E-520 plan](../own_baseline_E520/README.md).
