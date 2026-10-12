# Research notes and evidence index, 2026-10-06 .. 2026-10-12 (start here)

## October12 archive and live STOP

Start with the [complete research handoff](RESEARCH_COMPLETE_HANDOFF_20261012.md)
for the source-qualified method/evidence summary, strongest Q8 submissions,
local and public validation limits, serving work, failed attempts, and linked
artifact catalog and complete source-preserving ledger appendices. The appendices
are built by [this bounded generator](build_research_complete_handoff_appendix_20261012.py);
see its [point-in-time receipt](campaign_2026-10-12/research_archive_inventory_r1/APPENDIX_BUILD_RECEIPT_R1.json).
It is explicitly a pre-drain draft at Main snapshot
cfea172a: root must add terminal evidence for the already admitted clean-F0
seed4 continuation and Kaggle R7 before marking it final. The user STOP closes
new experiment admission; this draft does not reopen research. The point-in-time
metadata inventory is in
[research_archive_inventory_r1](campaign_2026-10-12/research_archive_inventory_r1/MAIN_STATUS_INVENTORY_R1.json)
and predates the latest Main cleanup.

October12 continuation: [all CPU optimization, serving repair, native measurements and public-fold diagnostics](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/zen4_cpu_research_20261012_r1/INDEX.md). This includes unrun ideas and negative results. The historical policies/headlines below retain their dated scope; current ADR0030 uses local/CV evidence for selection and public scoring for transfer confirmation.

Unified ledgers (append-only): [hypotheses.jsonl](hypotheses.jsonl) currently has 479 rows / 458 distinct IDs; [experiments.jsonl](experiments.jsonl) has 796 rows / 736 distinct IDs. Repeated IDs record revisions, retries, or additional events. The configured tree has 158 nodes across 11 families, and its journal has 134 events at the handoff snapshot. See [tree configuration](bandit_tree/config.json), [journal](bandit_tree/journal.jsonl), and [generated tree/index](bandit_tree/generated/INDEX.md). The generated integrity snapshot is stale against these source ledgers and should be rebuilt after the admitted drain. Live state: [specs/current-state.md](../specs/current-state.md).

## Read in this order
1. **Current best solution explained** (architecture, training, distillation, HPO, validation, plots, CIs): [`diversity_stack_20261009/solution_report/CURRENT_BEST_SOLUTION_EXPLAINED.md`](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/solution_report/CURRENT_BEST_SOLUTION_EXPLAINED.md)
2. **Chronological research log with evidence per pass (passes 1-47):** [`LOOP_LOG_20261010.md`](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/LOOP_LOG_20261010.md)
3. **Contract (teacher-ceiling objective, falsifiers):** [`CONTRACT_R3_TEACHER_CEILING_20261010.md`](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/CONTRACT_R3_TEACHER_CEILING_20261010.md)
4. **Validation protocol and noise measurements:** [`hpo/VALIDATION_PROTOCOL.md`](../hpo/VALIDATION_PROTOCOL.md), evidence report [`hpo/evidence/scratch_effects_fold1.json`](../hpo/evidence/scratch_effects_fold1.json)
5. **Santander/BNP transfer, EDA probes, serving-speed plan:** [`SANTANDER_LESSONS_AND_SERVING_SPEED_20261010.md`](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/SANTANDER_LESSONS_AND_SERVING_SPEED_20261010.md)
6. **External ideation (Grok, OpenCode, Freebuff, Haiku miners) rounds 1-4 and teammate digest:** [`external_ideation_20261010/`](campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/external_ideation_20261010/) (`SYNTHESIS_20261010.md`, `round2/SYNTHESIS_ROUND2.md`, `round3/SYNTHESIS_ROUND3.md`, `round4/*`, `TEAMMATE_RECIPE_DIGEST.md`, `BNP_TRANSCRIPT_MINING.md`, `OWN_NOTES_MINING_20261010.md`)
7. **Backlog and queues:** [`backlog/IDEA_BACKLOG.json`](campaign_2026-10-06/bandit_continuation/backlog/IDEA_BACKLOG.json), local queue [`local_queue_r1/`](campaign_2026-10-06/bandit_continuation/local_queue_r1/), Vast lane [`vast_teachers_r1/`](campaign_2026-10-06/bandit_continuation/vast_teachers_r1/), HPO [`hpo/studies/teacher_v2/`](../hpo/studies/teacher_v2/), Kaggle long packet [`kaggle_long_20261010/`](campaign_2026-10-06/bandit_continuation/kaggle_long_20261010/)
8. **Prediction bank (scores):** [`prediction_bank/`](campaign_2026-10-06/bandit_continuation/prediction_bank/) (BANK_FOLD*.jsonl, LATEST_GRADE_*.json)

## Headline findings (details and evidence in the log)
- Information ceiling: with k future rows visible fold-1 WP is 0.666 (0), 0.680 (2), 0.711 (8), 0.780 (32), 0.891 (128), 0.960 (512). Causal prediction is a long-horizon price forecast. (E-H330)
- Recipe plateau: chunk length, bootstrap, snapping, width, epochs, forecasting head all within +-0.001-0.002 of the control (v6 seeds 0.66605/0.66578). (E-H331, E-H332)
- Student distillation chain gives about +0.003 over a scratch teacher (paired CI +0.0016..+0.0049 on fold 1); student-stage knobs are flat.
- Post-hoc output tricks, value-frequency features, duplicate-row/lattice ideas: closed or retracted (E-H339).
- Teammate best public 0.6674 = 2-member distilled pair; leaderboard top 0.6717, ours 0.6664 (21st).
- In flight: long-epoch ladder (Vast, local, Kaggle), dropout/gated/EMA arms, soup merge. Results append to the log and ledgers.

## Policies in force
Historical October10 policy: select on mean score, not CI lower bound; public score decides; single-member packages upload one at a time. Later user instructions/ADR0030/0032/0033 supersede the public-selection and release-timing parts: compare complete fixed feasible endpoints on local/CV evidence, package authorized available blends, and use public for confirmation. Independent implementation remains binding.
- IDEA_LEDGER_20261011.md added to index
