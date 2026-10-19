# Large dynamics Transformer teacher: design handoff

Status: proposed, launch_ready=false. No new trainer, B300 profile, fit or teacher
cache is supplied. This packet turns the user's scale request into an explicit
recipe and compute inventory; it does not restart the stopped server campaign.

- [Design](../../unified/ideas/LARGE_DYNAMICS_TRANSFORMER_B300_20261019_R2.md):
  numeric/group embeddings, causal attention and memory, dynamics objectives,
  0.68B/1.65B/3.17B scale, FP32 Muon, WSD, DDP and GRU transfer.
- [PR description](PR.md): eleven-section English handoff and known implementation gaps.
- [TradeFM adaptation](TRADEFM_ADAPTATION.md): user-supplied market-dynamics paper, task mismatch and proposed observable-event survival/mark learning.
- [Transition-objective recipe](transition_recipe.json) and [expanded panel](TRANSITION_PLAN.json): same0.68B teacher, auxiliary loss off/on, four total new fits; a separate initial comparison, not included free in the capacity panel. This is now the preferred first dynamics screen.
- [Recipe](recipe.json): the initial four-fit capacity screen, two explicit cells,
  starting optimizer/scheduler and readiness limitations. Paths/membership and
  actual commands remain to be source-bound by the receiving implementer.
- [Generated panel](PLAN.json): four teacher fits, two cells per arm, no launch; derived from the recipe and plan helper.
- [Calculated scenarios](CAPACITY_SCENARIOS.json): generated trunk/state/attention-map formulas and hypothetical timings, not a hardware measurement.
- [Capacity calculator](capacity.py): stdlib-only matrix/state/score-map arithmetic
  and hypothetical step-time scenarios. It reads no data, imports no torch and
  certifies neither available memory nor actual throughput. Run from repository
  root with `uv run python research/experiment_drafts/large-dynamics-teacher-b300-r2/capacity.py`.
- [Plan expansion](../../../scripts/experiment_draft_plan.py):
  `uv run python scripts/experiment_draft_plan.py research/experiment_drafts/large-dynamics-teacher-b300-r2/recipe.json`
  prints four planned fits and remaining work; it launches nothing.
- [Independent design review](../../unified/reviews/LARGE_TEACHER_DESIGN_REVIEW_20261019_R2.md)
  and [primary sources](../../unified/sources/TRANSFORMER_B300_SCALE_SOURCES_20261019_R2.md)
  preserve criticism, source scope and limits.

Reuse canonical datasets/input assembly/objectives/checkpoint/provenance modules.
New model, attention, FP32 optimizer and DDP components need focused modules and
numerical checks. Do not copy a trainer or add logic to scripts/train_rnn.py.
Published [draft PR #31](https://github.com/VladKochetov007/WunderFunder/pull/31) targets research-team. [Publication receipt](PUBLICATION.json) records the initial immutable source commit, policy/base commits and code/recipe hashes; metadata publication is not a launch.

The later user paper motivates the event-objective comparison before committing to the full scale ladder. Both panels remain unrun and require separate actual compute admission. No completed source paper result is presented as our target-score evidence.
