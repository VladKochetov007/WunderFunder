# Modified clean-fold TL3 supervision comparison R2

**Design only: proposed, unadmitted, not preregistered, and not runnable.** R2 is a new successor packet; the R1 recipe and its `NEEDS_REVISION` review are preserved unchanged. R2 itself grants no authority for implementation, training, inference, cache generation, remote work, admission, or publication; it does not revoke separate user-authorized bounded capability preparation. Its machine-readable recipe is [`recipe.json`](recipe.json), SHA-256 `5b713c1da84d4d3064e99373b2ed512620760d857cde481b5eca38166e49d70a`.

## What the comparison asks

R1 compared a three-member, multi-stage candidate with the frozen rank-1 deep labels-only endpoint, which did not match final-model count, training budget, architecture, or initialization. The [R1 independent review](../tl3_modified_recipe_design_r1/INDEPENDENT_REVIEW_V1.json) therefore rejected that CP30 framing. R2 changes the primary contrast to two arms with the **same three final architectures, fixed 0.34/0.33/0.33 blend, seeds, initial states, fit memberships, normalization, optimizer/scheduler, epoch and update budgets, fixed checkpoints, and clean shared ancestor graph**.

Only final-member supervision weights differ. The candidate keeps the copied R1 recipe: main distillation alpha 0.7 for the 0.34 distilled member and 0.5 for each tiedlead member, plus family-loss weight 1.0 on the distilled member. The matched control sets every final-member distillation alpha to 0 and the family-loss weight to 0, while retaining the identical architecture and auxiliary family-head module. Candidate and control use the same clean fold-specific `fbc1k_s5@ep3` warm parent for the distilled member; tiedlead members are scratch-started from the same paired member seeds.

This estimates the **joint final-stage teacher-supervision bundle conditional on a shared graph and fixed parent**, not main distillation separately from family supervision, not upstream teacher/KD mechanisms, and not a KD-free pipeline. The shared `fbc1k_s5` parent is itself distilled from clean fin256 teachers and remains in both arms. The frozen rank-1 deep result is a descriptive historical anchor only; it is never the primary matched control or a substitute for an R2 pair. R2 is not an exact TL3 reproduction or a public-score estimate.

## Zero-loss and matching requirements

At control alpha 0, the objective must take the exact labels-only WMSE path; implementation must not compute `0 * teacher_loss`, which could propagate non-finite values. At family weight 0, the family loss contributes no gradient, but the same family-head module, initialization, parameter ordering, checkpoint topology, and prediction output shape remain. If the current implementation suppresses that head when its weight is zero, a narrowly reviewed adaptation is required before preregistration; the design does not assume that capability already exists. Synthetic capability checks and independent adapter review are listed in the recipe.

Before any future preregistration, compare the allowed-config diff against the four listed supervision weights only. Verify identical paired initial parameter and normalization hashes, exact batch order/update counts, same fold membership, and same provider plus pinned software/runtime within every pair. A route mismatch or any other configuration drift leaves that pair incomplete.

## Cells, completeness, and cost

The frozen 8/8/7/7 matrix contains F0/F1 seeds 1001–1008 and F2/F3 seeds 1001–1007. Every one of the 30 cells has candidate and control members at the same seeds: `S`, `S+10000`, and `S+20000` for distilled, tiedlead A, and tiedlead B. Only all 30 valid pairs yield a complete CP30 result. A pair is valid only when both arms have finite source-bound outputs on the same complete ordered evaluation membership, same provider/runtime, matching initial-state and normalization hashes, expected update/skip counts, and the fixed checkpoints. Any missing, invalid, unresolved, update-mismatched, or route-mismatched pair makes CP30 incomplete; report partial results descriptively, preserve failures, and do not replace seeds or drop pairs after seeing outcomes. Completion is not fresh confirmation because all four folds are exposed.

The primary statistic is the equal-fold mean of the within-fold paired score deltas, with 8/8/7/7 cells. Report all pairs and fold means. Keep paired seed bootstrap uncertainty separate from whole-component evaluation-unit bootstrap uncertainty; both condition on the fixed shared ancestor graph, and the component interval also conditions on fitted predictions. A `+0.001` practical delta is only a proposal carried from R1, not a frozen R2 rule; root must freeze or replace it in a dated preregistration before outcomes.

The planned work is 60 shared ancestor fits (15 graph nodes × 4 folds) plus 180 final-member fits (30 pairs × 2 arms × 3 members), 240 fits total. The control adds 90 final-member fits beyond candidate-only R1. No runtime or throughput estimate is available.

## Fold membership and exposure

Counts below distinguish evaluation membership entries/row-groups from unique connected components. Each fold excludes its **own** evaluation entries and their full component set from all fitted ancestors, normalization, caches, students, and selection; the F0 value 468 does not apply to every fold.

| Fold | Fit train | Fit valid | Evaluation entries | Unique evaluation components | Evidence status |
|---|---:|---:|---:|---:|---|
| F0 | 10,535 | 1,405 | 468 | 141 | Canonical ordered membership comparison plus accepted same-row alignment receipt |
| F1 | 10,533 | 1,416 | 457 | 149 | F1 plan’s parent-live metadata observation; refresh exact manifest before preregistration |
| F2 | — | — | — | — | Exact manifest/counts not pinned in the cited local receipts; required before preregistration |
| F3 | 10,539 | 1,368 | 505 | 145 | Exact read-only split metadata receipt; refresh at preregistration/admission |

The F0 comparison proves ordered memberships/component mapping despite distinct historical (`09263ad4…`) and canonical (`c09f7240…`) manifest identities; new work must bind the canonical identity. Before preregistration, pin all four exact manifests and data/source hashes and recompute ordered membership and component maps. These listed counts are scoped to their receipts and do not replace fresh admission checks. All folds are exposed, so this is development/robustness evidence only.

The public incumbent is a full `train+valid` fit that included the former validation partitions. R2 is a fold-specific fit-only CV analogue: each fold’s fit population excludes its own evaluation components through every learned ancestor. Neither score is an estimate of the public full-fit model or leaderboard transfer.

## Readiness limits and files

The exact historical creative `prop_f0` producer and W18 cache producer/row/preprocessing provenance remain unresolved; these limit historical attribution but do not require reopening the finite search or recovering old outputs. The modified design instead requires fresh per-fold propensity fits and replacement caches, each bound to its actual code/source, data, split, fit rows/components, normalization/preprocessing, transform, checkpoints and runtime. Those new clean outputs must be rebuilt and reviewed before preregistration. The canonical tiedlead adapter, same-topology family head and alpha-zero control path still require capability review, and F2 split metadata remains a preregistration blocker. This design packet grants no implementation authority and does not revoke separately user-authorized bounded capability preparation.

- [`recipe.json`](recipe.json) contains the candidate configuration copied from R1, the zero-supervision control overrides, all 30 pairs, count evidence, and source hashes.
- [`tl3_modified_recipe_design_r1/README.md`](../tl3_modified_recipe_design_r1/README.md) and [`tl3_modified_recipe_design_r1/recipe.json`](../tl3_modified_recipe_design_r1/recipe.json) preserve the prior candidate proposal; the pinned [`R1 review`](../tl3_modified_recipe_design_r1/INDEPENDENT_REVIEW_V1.json) explains why its former comparator was not accepted.
- [`F0_MEMBERSHIP_COMPARISON.json`](../export_preparation/F0_MEMBERSHIP_COMPARISON.json) and the [accepted F0 result review](../strong_control_evaluation_r4/records/RESULT_ACCEPTANCE_REVIEW_V1.json) bind F0 membership and its 468-row-group/141-component count.
- The [F1 plan](../../priority25_execution_r4/f1_beleriand_fallback/F1_BELERIAND_PLAN.json), [F1 observation binding](../../priority25_execution_r4/f1_beleriand_fallback/PREPARATION_BINDING.json), and [F3 split receipt](../../priority25_execution_r4/f3_beleriand_preparation_r1/F3_SPLIT_METADATA_V1.json) support the scoped F1/F3 counts above.
- [`INDEPENDENT_REVIEW_V1.json`](INDEPENDENT_REVIEW_V1.json) will hold the byte-bound independent review of this exact R2 packet; no review is claimed until that receipt is present.

No separate preparation manifest is produced; source/evidence pins are in `recipe.json`. The R2 packet intentionally does not edit the campaign index or any frozen recipe.
