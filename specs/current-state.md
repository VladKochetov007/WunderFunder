# Current project state

**Snapshot date:**18 October2026. **Research status:** ranked25 execution resumed on Beleriand and DataSphere. **Current scope:** ordered experiments, initially rank1; one Beleriand GPU. Provider spending stays within the current campaign allocation.

## Ranked25 execution authority,18 October2026

The latest user explicitly requests testing ideas1–25 end-to-end on Beleriand and DataSphere, with at most one Beleriand GPU and job labels `computer-vision-homework-*`. This supersedes the earlier research-reset no-fit restriction for this campaign. The [execution packet](../research/campaign_2026-10-18/priority25_execution_r4/README.md) owns the current ordered tracker, provider budget and source-bound recipes. Both rank1 actual-data FP32 profiles passed (40 finite optimizer updates per shape). The first F0/seed1001 paired six-epoch pipeline is running on one Beleriand GPU. Wide epochs0–1 completed29,840 updates with zero skips/nonfinite gradients; its [stage1 archive and continuation state passed structural checks](../research/campaign_2026-10-18/priority25_execution_r4/receipts/RANK01_WIDE_F0_S1001_STAGE1_VERIFICATION.json). The deep arm has now [completed and independently verified epoch0](../research/campaign_2026-10-18/priority25_execution_r4/receipts/RANK01_DEEP_F0_S1001_EP0_VERIFICATION.json) with 14,920 successful updates, zero skips/nonfinite gradients and matching full-state/FP32/source checks; epoch1 is running. Fixed epoch5 endpoints, exports and independent quality grades remain pending. The user now explicitly authorizes discretionary DataSphere use ("сколько хочешь на датасфере делай"); the numeric-budget wait is superseded. Root allocates bounded jobs within existing balance and live quotas, with no billing changes or top-ups; the first admitted A100 profile `bt13i2k83lo8n4m8kplk` has now failed during provider preparation with INTERNAL13 and no reported start or allocated instance. The [terminal incident](../research/campaign_2026-10-18/priority25_execution_r4/receipts/DATASPHERE_PROFILE_V2_TERMINAL_INCIDENT.json) verifies empty provider/output manifests, unchanged observed balance and earlier CPU/GPU jobs with the same missing-start pattern. No cloud runtime/profile or quality result exists. The credential is removed; staged v3 is not submitted. The distinct [runtime canary](../research/campaign_2026-10-18/priority25_execution_r4/receipts/DATASPHERE_CANARY_V1_TERMINAL.json), job `bt12dffc0bhjrba9jcp6`, also failed before start with INTERNAL13 using the known-successful cu126 environment and explicit no-training probe. No files/runtime result exist; its token is removed, no paid job remains active, and observed balance is unchanged. The exact provider failure remains unresolved; no unchanged retries are scheduled. The [live official incident and effective historical controls](../research/campaign_2026-10-18/priority25_execution_r4/receipts/DATASPHERE_PROVIDER_INCIDENT_20261018.json) now corroborate an infrastructure limitation affecting DataSphere. Historical CPU/GPU probes already uploaded an entrypoint without datasets or an explicit requirements list; an equivalent CPU retry is not admitted. Public status is consistent with our failures, but does not establish their exact internal cause. The healthy Beleriand sequence and reviewed F1 preparation continue. Historical scope statements below retain their dates and do not override this resume.

## Operating state

The user additionally authorizes repository issues for the team’s cluster. [Issue16](https://github.com/VladKochetov007/WunderFunder/issues/16) is now published and requests F2/seed1001 only, both arms, with a pinned [full handoff](../research/campaign_2026-10-18/priority25_execution_r4/cluster/README.md). Authenticated Zen access succeeded despite connector/CLI limitations. No worker claim, hardware or cluster execution is yet verified. Beleriand owns F0/seed1001; DataSphere F1/seed1001 remains adapter preparation, without a fit allocation.

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
