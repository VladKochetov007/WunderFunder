# Bounded LSQ scale confirmation recommendation R2

This supersedes the F3-fold recommendation in R1. It is a controller-facing
allocation proposal, not a fit admission or a resource authorization.

The F0 seed-3 10% learned-scale endpoint had a small positive WP point delta
against its matched nonlearned per-update absmax control (`+0.00019414`), with
a descriptive paired component interval crossing zero. It is one exposed fold
and one fitting seed. That result supports a bounded replication check, not a
scale-benefit claim or a sweep.

The independent F3 ancestry inventory at
[`INVENTORY_R1.md`](../inference/f3_qat_ancestry_r1/INVENTORY_R1.md), SHA-256
`9374f6ce62590604310e481f3da8ad08bdad2d56550d7321e3e8f583ae7df1d6`, finds
no exact clean-T3 CP5 F3 seed-4 parent or matching three-teacher H384 cache.
Producing that strict lineage requires the missing seed-4/5 clean-T3 teacher
fits and full train/valid exports, a blended F3 cache, and an F3 S4 combo
initializer. This is a real ancestry gap, not evidence against LSQ. The same
inventory confirms the F0 clean-T3 CP5 seed-4 parent exists (checkpoint SHA
`e6be82ff799f4ea2b62e770e9c340a8cfd8f98780d1c7dec4a76eab5a853d35c`) and
identifies its fold-matched F0 target cache. Thus an exact F3 prerequisite
should not delay a cheaper same-fold, distinct-seed F0 confirmation.

After the root-owned F0 seed-4 profile/cohort is complete and fresh resource
admission is available, prioritize one **matched 10% LSQ pair on F0 / fitting
seed 4**. Start both arms from the same clean-T3 CP5 seed-4 parent and bind
them to the same F0 seed-4 propensity ancestry, clean-T3 cache, split, and a
component-complete 10% fitting subset. Prepare a seed-4 subset manifest using
the established 10% protocol; hold its ordered members identical across the
pair and record its new manifest hash. Learn every selected row step, including
the two Q8 head blocks, in the candidate. The control recomputes nonlearned
absmax row steps per latent update. Keep FP32 AdamW `lr=1e-4`, `wd=0.01`,
OneCycle, 1,440 optimizer updates, fixed raw epoch-0 endpoint, no EMA/SWA,
clipping and sample-order protocol fixed; change only run name and the
scale-learning flag. Grade the complete fixed F0 endpoint panel with the
existing paired component-bootstrap grader.

This confirmation estimates whether the seed-3 F0 screen repeats across a
distinct fitting seed while holding fold and ancestry family fixed. It does
not provide cross-fold robustness or pristine validation. If it remains
promising, root can separately compare the value of broader F0 coverage with
the preparation cost of the strict F3 teacher/cache lineage. The old R1 F3
proposal remains useful as that broader follow-up, but it is not the next
feasible allocation. A fresh profile is required if the exact source, hardware,
shape, or recipe changes; current actual-profile evidence is not fit admission.

The decision needs no F3 teacher preparation before the F0 seed-4 pair. It
still requires root to freeze the exact seed-4 subset/commands, verify all
ancestry and data identities, and make a fresh resource admission. Do not
interpret the allocation proposal itself as permission to start either fit.

## Implementation clarification — 2026-10-13

The successor recipe now freezes the S4 subset manifest and uses the same 315
selected components and ordered group membership as the seed-3 10% LSQ pair,
while assigning the new experiment ID
`E-QAT-LSQ-10PCT-F0S4-20261013-R1`. This exact membership choice makes the
1,440-update budget above valid: 1,048 train groups plus 104 fit-valid groups
give 72 batches of 16 for one epoch. A different seed-4 sample would change
that count; do not force it to 1,440 by dropping groups or splitting
components. The prepared recipe remains unadmitted, and two actual S4
100-update profile receipts plus a fresh root resource decision are required
before fitting. The [independent S4 recipe review](QAT_LSQ_F0S4_PROFILE_RECIPE_REVIEW_R1.md)
records the exact paths, hashes, parent identity and command checks.
