# Current project state

**Snapshot date:**18 October2026. **Research status:** ranked25 execution resumed on Beleriand and DataSphere. **Current scope:** ordered experiments, initially rank1; one Beleriand GPU. Provider spending stays within the current campaign allocation.

## Ranked25 execution authority,18 October2026

The user authorizes ordered testing of ideas1–25, with one Beleriand GPU worker and discretionary DataSphere use within the existing balance. The [execution packet](../research/campaign_2026-10-18/priority25_execution_r4/README.md) owns current claims, recipes and evidence. No billing change or top-up is authorized.

**Rank1, first completed pair:** [F0/seed1001 fixed epoch5 result](../research/campaign_2026-10-18/priority25_execution_r4/receipts/RANK01_F0_S1001_RESULT.json). Wide H512×1 scores0.673575751; deep H384×2 scores0.672108767. The paired delta is+0.001466984, with a conditional component-bootstrap95% interval[−0.000411695,+0.003589820]. Both arms completed89,520 successful updates over six epochs, with zero skips/nonfinite steps. Their terminal checkpoints passed independent structural checks; predictions, sidecars, logs and official grade are archived and hash-bound. This is one previously exposed fold/seed pair of30required per arm, not CP30, convergence or public-transfer evidence. The interval crossing zero does not stop the planned cross-fold comparison.

**Current worker:** [F1/seed1001 is admitted on Beleriand for both arms](../research/campaign_2026-10-18/priority25_execution_r4/receipts/F1_BELERIAND_ADMISSION_V1.json), under `computer-vision-homework-r01-f1-sequence.service`, GPU UUID `GPU-a50d0c3b-d28f-a2a7-8b43-3e02b543fc0c`. The initial wide stage1 worker is running; the canonical loader reports10,533 train groups,1,416 fit-valid groups,457 evaluation groups and89,580 scheduled updates. The [08:06UTC observation](../research/campaign_2026-10-18/priority25_execution_r4/receipts/F1_INITIAL_PROGRESS_20261018T0805Z.json) now establishes actual optimizer work through epoch0 batch140/1493, with zero new memory events and no OOM. The separate [F1 data/source preflights](../research/campaign_2026-10-18/priority25_execution_r4/receipts/F1_PREFLIGHT_EXECUTION_V2.json) passed all six full input hashes, native split/component checks and523 frozen source hashes on the unchanged Python3.13.15/Torch2.14.0+cu130 runtime. Reviewed F1 checkpoint helpers use14,930 updates/epoch,29,860 at stage1 and89,580 at terminal; these remain expected counts until actual checkpoints arrive. The older frozen plan's UNADMITTED preparation field is superseded by the separate admission receipt; plan bytes were not changed.

DataSphere remains authorized, but the [latest provider/cell reconciliation](../research/campaign_2026-10-18/priority25_execution_r4/receipts/PROVIDER_CELL_RECONCILIATION_20261018T0740Z.json) finds no active job in the latest100, unchanged balance41,652,791 and the ongoing official infrastructure incident affecting DataSphere. Profile v2 and the distinct runtime canary both failed before a reported start with INTERNAL13; no cloud optimizer or runtime result exists. Their temporary IAM files were removed. Do not repeat unchanged probes; require recovery or a genuinely distinguishing diagnostic. Missing budget approval is not a dependency.

[Issue16](https://github.com/VladKochetov007/WunderFunder/issues/16) requests F2/seed1001 both arms from the team cluster; no worker claim is observed. Issue19 is a separate B300 experiment, also without an observed claim. Beleriand now owns F1/seed1001; DataSphere F1 preparation is not an active allocation. Preserve foreign jobs, including the other user's process on the shared selected GPU. Rank2 remains pending; continue rank1 across folds and seeds.

## Operating state

Historical campaign stops below retain their original scopes and dates; they do not override the current ranked25 execution authority.

The latest research closeout is dated 15 October 2026 at 18:39 UTC. It records
that the admitted G fit, export and grade finished, all admitted work drained,
and the research admission barrier remained closed. The latest user explicitly resumed the sequence-moment/heuristic investigation locally on main with a5GB aggregate task RAM cap. This supersedes that pause for this exact scope; unrelated campaign admission and the other agent's Kaggle ownership remain unchanged. Older plans, quotas and stop snapshots are historical authority only.
See the [pause closeout](../research/unified/operations/RESEARCH_PAUSE_20261015_R1.md).

The separately recorded October 17 research-reset scope resumes research thinking, literature, EDA and bug repair while excluding model training for that packet. The geometry-specific local fit authority above is separate and does not expand that packet's scope.
The [research reset contract](../research/campaign_2026-10-17/research_reset_r1/CONTRACT.json)
records this boundary. No fit, rental or submission is authorized by this reset.

The [research reset packet](../research/campaign_2026-10-17/research_reset_r1/README.md)
records deployed creative prediction/cache repairs, deterministic checks,
full-data target/event census, sampled full-feature EDA, source registers and
independent proposal/critique rounds. Its ideas are proposals, not new model scores.


## Current scoped geometry investigation

The [geometry record](../research/unified/validation/SEQUENCE_MOMENT_GEOMETRY_20261017_R1.md) contains128train/128valid raw-input EDA, all112visually inspected features and 33 matched tiny-GRU endpoints across two capacities. Independently verified pooled moment gain is+0.0008833, with opposing exposed-fold effects; this is not a production or public gain. The preregistered discrete occupancy/dwell and capacity interaction pilot completed from immutable source snapshots; fixed half-mixtures expose conditional value of weaker standalone components. All fits are terminal; no broader campaign or Kaggle admission is resumed. Allcode and provenance are maintained on main and integrated into research-team. One initial1.4GBmetadata-reader OOM was diagnosed and fixed; later tasks are contained within5GBaggregateRAM and one CPU inside the existing two-core parent. No Kaggle job is modified.

## Public leaderboard snapshot

The official leaderboard was read on 17 October 2026. The team was at public
WP **0.66832**, rank **22**, with **97 team submissions**; the leader was at **0.67243**. This is a dated
external team result, linked to the [official leaderboard](https://wundernn.io/connectome/leaderboard).
The page does not establish which exact ZIP or submission ID produced the
current team-best score. The result is not private-leaderboard evidence and
does not replace local validation.

## Latest measured local evidence

The [R2 research-reset packet](../research/campaign_2026-10-17/research_reset_r2/README.md)
quantifies the rare clock exceptions:0.003172% of scored valid target weight,
and0.001953% for retrospective label-reconstruction mismatch rows. It therefore
reduces their research priority without restoring the universal H-001 claim.
On467 exposed sequences, all36 causal fixed-output transforms across three saved
arrays are negative (AND-hold loses0.118–0.126WP); noncausal label-run oracle gains
~0.03 cannot be treated as recoverable causal gain. Within-run fluctuations
correlate~0.96 across arrays. Complete target/mask alignment passes, but historical
prediction-producer binding remains unverified. All25 proposal dispositions
were revised against old code and current evidence. This packet has zero fits,
optimizer steps or new inference; it is distinct from the geometry fits above.

The October17 [no-fit packet](../research/campaign_2026-10-17/research_reset_r1/README.md)
adds deployed legacy fixes and a four-group frozen-checkpoint replay (WP0.06289742
with wrong preprocessing,0.72994782restored; bitwise manual/config parity). This
is bug isolation, not a new model ranking. Its full four-column census refutes
H-001 with1,253need-eligible exceptions. The shared evidence and25proposal branches
were published on research-team; the [dated publication receipt](../research/campaign_2026-10-17/research_reset_r1/publication/README.md)
records that session's missing API authentication. A later upstream integration
put that R1 history into both main and research-team at0f479660; the R2
[publication record](../research/campaign_2026-10-17/research_reset_r2/PUBLICATION.md)
distinguishes the subsequently published follow-up from main promotion.

The last completed comparison in the current pause record is the exposed-F0
G-versus-C grade: G scored 0.68125607, a delta of **+0.00002956** over C, with
a conditional component 95% interval of **[−0.00038018, +0.00037707]**. This
single-seed comparison resolves no quality advantage. See the
[pause receipt](../research/unified/operations/RESEARCH_PAUSE_20261015_R1.md)
and [campaign frontier](../research/campaign_2026-10-13/strategy_execution_r1/FRONTIER.md).

Other late Main results remain scoped to their recorded panels. For example,
the matched short architecture screen measured H512×1 at 0.67323776 versus
H256×2 at 0.66977764 on one exposed F0 seed. That is a conditional screen, not
a converged architecture ranking or public-score forecast; see the
[student frontier](../research/campaign_2026-10-14/cpu_student_frontier_r1/FRONTIER.md).

## Code and draft status

The modular trainer is integrated in Main. [scripts/train_rnn.py](../scripts/train_rnn.py)
is the compatibility command-line entrypoint; reusable training components
live under [src/wnn/training/](../src/wnn/training/README.md). The refactor
receipt reports static structure/lint checks and CLI help validation. It does
not report unit or integration tests, numerical training parity, a new fit, or
a throughput benchmark. Read the
[refactor receipt](../research/unified/methods/MODULAR_RNN_TRAINER_REFACTOR_20261017_R1.md)
before interpreting the change as evidence about model quality or performance.

The subsequent [repository maintenance](../research/unified/operations/MAIN_MAINTENANCE_20261017.md)
adds synthetic CPU regression and integration checks, legacy class compatibility
fixes, and fewer intermediate feature concatenations. This is software evidence;
it establishes no new model score, GPU parity or measured training speedup.

The 17 October GitHub snapshot recorded 14 experiment-prefixed draft pull
requests. They contain implementation proposals and handoffs; the count is not
a count of completed fits. Historical results already present in Main must be
tied to their own receipts. See the
[repository audit](../research/unified/reviews/REPOSITORY_AUDIT_20261017.md),
[pull-request list](https://github.com/VladKochetov007/WunderFunder/pulls), and
[experiment draft workflow](../research/unified/methods/EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md).

## Navigation

- [Research objective](research-objective.md) — goal and binding constraints.
- [Research library](../RESEARCH.md) — how to find ideas, results and source
  evidence.
- [Repository audit](../research/unified/reviews/REPOSITORY_AUDIT_20261017.md),
  [model catalog](../research/unified/models/CURRENT_MODEL_CATALOG.md),
  [validation protocols](../research/unified/validation/VALIDATION_PROTOCOLS.md),
  and [storage map](../research/unified/operations/STORAGE_MAP.md) — current
  maintained indexes for the 17 October audit.
- [Decision index](adr/README.md) — current decisions and their successors.
- [Complete dated handoff](../research/RESEARCH_COMPLETE_HANDOFF_20261012.md)
  — October 12 archive, not a current state.
- [Preserved state history](current-state-history-20261017.md) — byte-exact copy
  of the previous 199,958-byte file, SHA-256
  37be5efacd617f8e712c99822c0399242949aef10fbeebc9abd20ee90f499902; its
  same-directory location preserves the original relative link bases.
