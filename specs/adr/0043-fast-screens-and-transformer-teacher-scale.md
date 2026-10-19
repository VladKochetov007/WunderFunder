# ADR0043: fast idea screens and large Transformer teachers

Status: accepted user instruction, October 19, 2026.

## Decision

For future drafts replace ADR0040's mandatory 30 fits per arm with an initial
budget of **4–6 total new scientific fits per idea comparison**. A fresh two-arm
screen normally has 2–3 matched fold/seed cells, with one fit per arm per cell.
State every arm and stage, new versus reused fits and all teacher/student costs.
Profiles are engineering work and are charged compute without becoming quality
replications. Exact clean controls may be reused; different exports are not new
fits. Further fitting is targeted and separately budgeted, not an automatic matrix.

Two or three cells estimate direction, not significance or independent fold/seed
variance. Show every paired effect, source-qualified membership, existing fold
exposure and conditional uncertainty. Prefer different folds/seeds where useful.
A noisy or initially weaker direction can justify follow-up through a new
mechanism, teacher/student transfer or complete-mixture value. Keep the official
metric and ancestry requirements. Do not rewrite historical CP30 recipes/results;
amend future admission explicitly when an old experiment continues.

Prioritize a large causal Transformer teacher on the team's eight B300 GPUs,
then evaluate distillation into a deterministic GRU. The design must specify
numeric/group embeddings, temporal dynamics, context/attention semantics, Muon
and AdamW groups, schedule, memory, distributed loss and actual-shape profiling.
Use hardware capacity as an upper constraint, not evidence that larger is better.
Strict FP32/autocast off/TF32 off remains the scientific precision contract.
Optimizer intermediates and compiled attention must satisfy it too.

This is policy and design authority. Existing stopped B300 controllers remain
stopped; their recipes are not restarted. The October20 local/Kaggle start and
15-hour budget stay governed by ADR0042. No rental/billing change is authorized.
Actual remote training requires an allocated device/time budget and runnable,
source-bound recipe; the design note does not claim either exists.

## Implementation and trace

- [AGENTS.md](../../AGENTS.md) and its CLAUDE.md symlink expose the new rule.
- [Workflow](../../research/unified/methods/EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md),
  [PR template](../../research/unified/templates/EXPERIMENT_DRAFT_PR.md) and shared
  draft/B300 skills define future handoffs.
- [Plan helper](../../scripts/experiment_draft_plan.py) defaults to two cells,
  accepts explicit panels, rejects over-budget initial screens and preserves
  historical v1 CP30 recipes through an explicit `--legacy-source-commit` pin.
- [Transformer design](../../research/unified/ideas/LARGE_DYNAMICS_TRANSFORMER_B300_20261019_R2.md)
  gives the proposed scale, compute and student path. All gains remain unmeasured.

## Consequences

Spend compute on representations, scale and teacher/student endpoints rather
than fixed replication quotas. A4-fit screen and a 4-fit distillation comparison
are 8 new fits total; never present them as a single 4-fit campaign. Repeated or
adaptive screens retain selection exposure. Preserve negatives and original
evidence, and allocate more precision only when it can change a decision.
