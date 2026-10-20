---
kind: review
status: complete
created_utc: 2026-10-19
hypothesis_ids: [H-LARGE-DYNAMICS-TEACHER-20261019-R2]
experiment_ids: [E-LARGE-DYNAMICS-TEACHER-SCALE-20261019-R2]
evidence_paths:
  - research/unified/ideas/CAUSAL_AR_TRANSFORMER_GRU_DISTILLATION_B300_20261017_R1.md
  - scripts/experiment_draft_plan.py
---

# Independent specification review of the larger teacher direction

Two fresh-context native subagents used gpt-6-luna, reasoning max, read-only
assignments and no fits, server calls, source edits or recursive delegation.
`transformer_scale_scout` researched primary hardware/optimizer/architecture
sources. `teacher_design_critic` independently inspected the old proposal,
current state, B300/evidence skills and frozen plan helper. Root owns the R2
design and policy edits. This is design criticism, not independent approval of
future code, a measured model or a statistical replication.

## Public findings and disposition

| Finding | Consequence and response |
|---|---|
|4–6 total fits and 4–6 paired cells differ by a factor of two for a fresh two-arm study. | Adopt total new fits:2–3 cells, explicit arm inventory and cached controls. Correct the scout's ambiguous matched-run phrasing during synthesis. |
| Two or three cells cannot identify both seed and fold effects reliably. | Show all effects and exposure; call it a directional screen. No automatic significance statement or CP30 escalation. |
| Teacher improvements do not establish GRU improvements. | Count student/control fits separately and rank the complete student/served endpoint. Intermediate assistants are optional experiments. |
| Plan generator hardcodes 30 per arm. | Change the prospective default and support explicit panels, while retaining recipes explicitly declaring historical 30-cell plans. |
| Patch or compressed-summary tokens can leak later rows to earlier predictions. | Use only completed input-block summaries with explicit completion-time visibility; require future-perturbation/reset checks. |
| Auxiliary future observations can leak through tokenization, preprocessing or crop tails. | Isolate the target path, fit transforms on fitting components only, and mask cross-sequence/padded horizons. |
| AR state across optimizer-updated chunks is an approximation. | Keep plain head control; test sequential/scan gradients and exact reset/prefix semantics before execution. |
| Weighted DDP must normalize across all ranks AND microsteps. | Use the documented global denominator derivation, no extra accumulation divisor, and single-versus-multirank numerical checks. |
| Model-state memory is a lower-bound accounting; dense attention can dominate. | Report state and score-map terms separately; actual-shape memory/profile remains mandatory before admission. |
| Muon LR conventions and semantic matrix partition matter. | Use an explicit match_rms_adamw convention and hidden-matrix allowlist, disjoint AdamW groups and separately defined Q/K/V geometry. Root's upstream-source inspection additionally found the BF16 internal cast, requiring a pinned FP32 variant. |
| Available model width does not guarantee quality or transfer on repeated correlated data. | Retain scale/context/loss alternatives and teacher/student follow-through; no universal scaling or public-score prediction. |
| Existing local/Kaggle authority is not an eight-B300 launch reservation. | Prepare the handoff; preserve stop history and actual resource admission. |

The scout favored 0.68B as a first quality candidate with a smaller control. Root's
quality-first synthesis keeps 0.68B as a serious reference and proposes 1.65B as
the flagship paired recipe, with 3.17B eligible after informative results. This
is a resource-allocation preference rather than a correctness disagreement.

## Remaining checks

The artifact follow-up found actionable migration gaps: autocast was unchecked,
count30 could masquerade as legacy, reused endpoints were not counted and
follow-up had no allocation field. Root corrected these: require autocast=false,
an explicit full legacy source pin with v1 panel metadata, matching reused
artifact/ancestry references, separate new/reused totals and a follow-up budget
reference. The printer returns the declarations; it does not verify their
contents or authorize execution. Concurrently reported ledger/navigation gaps
were resolved by append-only records and explicit index links.

Validation after correction: `uv run --no-sync ruff format/check` passes for the
planner, focused tests and calculator; the
[migration tests](../../../tests/test_experiment_draft_plan.py) pass10 cases;
`uv run --no-sync python scripts/check_training_structure.py` reports61 files and
zero errors. The actual historical transformer-ar recipe from source
`87b9b3ea396ff3bd4685d51420405a293a411db0`, expanded with its explicit legacy
source pin, retains30 cells/arm and60 fits. The new recipe expands to two
cells/arm and four new fits, launch_ready=false. These are CPU engineering checks,
not model training, numerical DDP validation or B300 memory measurements.

No agent has reviewed a new runnable Transformer trainer, actual B300 attention
backend or a fitted student. The [R2 design](../ideas/LARGE_DYNAMICS_TRANSFORMER_B300_20261019_R2.md)
lists the numerical/causal/DDP/profile gates still required. Its source citations
are indexed in the [source register](../sources/TRANSFORMER_B300_SCALE_SOURCES_20261019_R2.md).
This review does not approve a launch or establish empirical promise beyond the
plausibility of the stated mechanisms.
