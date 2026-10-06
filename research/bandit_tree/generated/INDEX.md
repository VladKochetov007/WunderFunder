# Research bandit tree

Generated navigational index. Original ledgers and legacy bandit remain archived; every ledger event is retained with its original line bytes and SHA-256.

Statuses are displayed as recorded or curated and have not been independently verified by this index. Validation, holdout, and fold data remain exposed. Links describe recorded relationships, not causal effects. Unknown priors and costs remain unknown; no score is inferred.

Root: **Wunder Connectome Challenge2 research** · metric `global_wp_clip2_v1` · nodes 729 · raw hypothesis rows 179 · raw experiment rows 506 · legacy arms 18 · catalog files 57769.

Config snapshot: [research/bandit_tree/generated/snapshots/config-7937734920b291d001f4c53ffd4062f7c07171a00a9556582d2e3b8cd0437c36.json](<snapshots/config-7937734920b291d001f4c53ffd4062f7c07171a00a9556582d2e3b8cd0437c36.json>).

Missing or ambiguous artifact and relationship links are called out below and in `tree.json` warnings. The recursive file inventory is in `catalog.json`; generated output is excluded.

## Curated decision tree

### Teachers and knowledge transfer (`family:teacher`)

- **Causal segment-memory Transformer teacher** (`arm:causal-transformer-teacher`) — not_tested; eligible=True.
  - **next_test:** Profile an independently implemented causal attention teacher on Kaggle; retain the completed friend recipe as a comparator, change convergence/event-context or transfer explicitly, measure teacher and full serving endpoint separately.
  - **claim_scope:** specific recipe / exposed-data exploration; no inferred public gain
  - artifact: [research/campaign_2026-10-06/branch_teacher_audit/REPORT.md](<../../campaign_2026-10-06/branch_teacher_audit/REPORT.md>)
  - artifact: [research/bandit_tree/TEACHER_AGENT_BRIEF.md](<../TEACHER_AGENT_BRIEF.md>)
  - artifact: [research/campaign_2026-10-06/branch_teacher_audit/post_fetch_b7fd5d2f/OBSERVATIONS.md](<../../campaign_2026-10-06/branch_teacher_audit/post_fetch_b7fd5d2f/OBSERVATIONS.md>)
  - **Transfer Transformer predictions into the serving GRU** (`arm:transformer-student`) — not_tested; eligible=False.
    - **next_test:** Same student recipe and paired seeds: cross-fitted attention-teacher targets versus GRU-teacher targets; fixed parent/head blend endpoint.
    - **claim_scope:** specific recipe / exposed-data exploration; no inferred public gain
    - **Temporal summary / representation transfer** (`arm:transformer-temporal-transfer`) — not_tested; eligible=False.
      - **next_test:** Prediction-only control versus a single predeclared causal temporal-summary auxiliary; no teacher representation at inference.
      - **claim_scope:** specific recipe / exposed-data exploration; no inferred public gain
    - **Quiet-row transfer from an attention teacher** (`arm:transformer-quiet-transfer`) — not_tested; eligible=False.
      - **next_test:** Use causal age/activity for training emphasis, retain all-row objective; compare global and quiet-row scores on same exposed components.
      - **claim_scope:** specific recipe / exposed-data exploration; no inferred public gain
    - **Attention-teacher transfer × origin-label reanchor** (`arm:transformer-reanchor-interaction`) — not_tested; eligible=False.
      - **next_test:** Paired factorial unchanged/teacher/reanchor/combination complete endpoints; measure interaction rather than add separate deltas.
      - **claim_scope:** A new interaction proposal, no run or forecast
  - **Origin-target privileged attention teacher** (`arm:privileged-transformer`) — not_tested; eligible=False.
    - **next_test:** Cross-fit teacher by continuous chain; future inputs allowed only offline; predict Y_t with origin weights/masks. Contrast teacher and student separately.
    - **claim_scope:** specific recipe / exposed-data exploration; no inferred public gain
    - artifact: [research/campaign_2026-10-05/continuation_night/NEXT_UNTESTED_IDEAS.md](<../../campaign_2026-10-05/continuation_night/NEXT_UNTESTED_IDEAS.md>)
    - **Transfer predictable teacher structure separately from future innovation** (`arm:learnable-privileged-transfer`) — not_tested; eligible=False.
      - **next_test:** Cross-fit a causal projection of privileged teacher targets by chain; compare ordinary distillation and projected residual/moment supervision in a fixed causal student. Keep all-row labels and origin masks; no future input at serving.
      - **claim_scope:** New proposed protocol; no fit, reproduced improvement or public forecast
      - artifact: [research/campaign_2026-10-06/branch_teacher_audit/post_fetch_b7fd5d2f/OBSERVATIONS.md](<../../campaign_2026-10-06/branch_teacher_audit/post_fetch_b7fd5d2f/OBSERVATIONS.md>)
  - **Convergence-aware causal attention and event context** (`arm:teacher-convergence-context`) — not_tested; eligible=False.
    - **next_test:** First vary optimization schedule with context fixed; then event-context versus row-context at fixed schedule. Pair the GRU and four-epoch attention controls, with measured resources and complete endpoint comparisons.
    - **claim_scope:** New proposed protocol; no fit, reproduced improvement or public forecast
    - artifact: [research/campaign_2026-10-06/branch_teacher_audit/post_fetch_b7fd5d2f/OBSERVATIONS.md](<../../campaign_2026-10-06/branch_teacher_audit/post_fetch_b7fd5d2f/OBSERVATIONS.md>)
- **Teacher lineage, cross-fitting and ensemble targets** (`arm:teacher-crossfit`) — not_tested; eligible=True.
  - **next_test:** Verify actual teacher checkpoint hashes, chain membership and cache sidecars; compare fresh modern ensemble teachers without overlap.
  - **claim_scope:** specific recipe / exposed-data exploration; no inferred public gain
- **Compress complementary teacher recipes into a GRU256 student pair** (`arm:heterogeneous-teacher-compression`) — not_tested; eligible=True.
  - **next_test:** Independently reproduce the heterogeneous-target protocol: same-architecture ensemble versus diverse teacher-family mixture in matched causal GRU256 students; audit producer epochs, normalization, chain membership and source scaling. Compare fixed student pair against our current complete endpoint.
  - **claim_scope:** Root implementation and paired comparison untested; external public gain is source-reported evidence, not our expected delta
  - artifact: [research/campaign_2026-10-06/branch_teacher_audit/RECENT_OTHER_BRANCHES.md](<../../campaign_2026-10-06/branch_teacher_audit/RECENT_OTHER_BRANCHES.md>)
  - artifact: [research/campaign_2026-10-06/branch_teacher_audit/post_fetch_833aaf54/iterations/2026-10-05-wave20/public-result.json](<../../campaign_2026-10-06/branch_teacher_audit/post_fetch_833aaf54/iterations/2026-10-05-wave20/public-result.json>)
  - artifact: [research/campaign_2026-10-06/branch_teacher_audit/post_fetch_833aaf54/manifest.json](<../../campaign_2026-10-06/branch_teacher_audit/post_fetch_833aaf54/manifest.json>)
  - artifact: [research/bandit_tree/HETEROGENEOUS_AGENT_BRIEF.md](<../HETEROGENEOUS_AGENT_BRIEF.md>)
  - **Replace a teacher family with a two-stream teacher** (`arm:two-stream-teacher-compression`) — not_tested; eligible=False.
    - **next_test:** Hold student recipe, seeds and mixture construction fixed; compare raw-only and causal two-stream teacher components. Test the compressed student pair before any additional inference teacher member.
    - **claim_scope:** Root implementation and paired comparison untested; external public gain is source-reported evidence, not our expected delta
    - artifact: [research/campaign_2026-10-06/branch_teacher_audit/RECENT_OTHER_BRANCHES.md](<../../campaign_2026-10-06/branch_teacher_audit/RECENT_OTHER_BRANCHES.md>)
    - artifact: [research/campaign_2026-10-06/branch_teacher_audit/post_fetch_833aaf54/iterations/2026-10-05-wave20/public-result.json](<../../campaign_2026-10-06/branch_teacher_audit/post_fetch_833aaf54/iterations/2026-10-05-wave20/public-result.json>)
    - artifact: [research/campaign_2026-10-06/branch_teacher_audit/post_fetch_833aaf54/manifest.json](<../../campaign_2026-10-06/branch_teacher_audit/post_fetch_833aaf54/manifest.json>)
  - **Function-preserving widening of a distilled student** (`arm:distilled-function-preserving-widening`) — not_tested; eligible=False.
    - **next_test:** Prove GRU/head widening parity before training; match narrow and wide two-epoch continuation against the same stronger teacher. Measure frozen student-pair quality, native cost and 3GB resource profile rather than compare unrelated initializations.
    - **claim_scope:** Root implementation and paired comparison untested; external public gain is source-reported evidence, not our expected delta
    - artifact: [research/campaign_2026-10-06/branch_teacher_audit/RECENT_OTHER_BRANCHES.md](<../../campaign_2026-10-06/branch_teacher_audit/RECENT_OTHER_BRANCHES.md>)
    - artifact: [research/campaign_2026-10-06/branch_teacher_audit/post_fetch_833aaf54/iterations/2026-10-05-wave20/public-result.json](<../../campaign_2026-10-06/branch_teacher_audit/post_fetch_833aaf54/iterations/2026-10-05-wave20/public-result.json>)
    - artifact: [research/campaign_2026-10-06/branch_teacher_audit/post_fetch_833aaf54/manifest.json](<../../campaign_2026-10-06/branch_teacher_audit/post_fetch_833aaf54/manifest.json>)

### Temporal state and memory (`family:temporal`)

- **Production event clock / quiet-row anatomy** (`arm:event-clock`) — diagnostic_completed; eligible=False.
  - **next_test:** Use independently reproduced AND-clock facts to predeclare a causal state update test, not output holding.
  - **claim_scope:** specific recipe / exposed-data exploration; no inferred public gain
  - artifact: [research/campaign_2026-10-05/continuation_night/event_semantics/reproduction/v2/receipt.json](<../../campaign_2026-10-05/continuation_night/event_semantics/reproduction/v2/receipt.json>)
  - artifact: [research/campaign_2026-10-06/REPORT.md](<../../campaign_2026-10-06/REPORT.md>)
  - artifact: [research/campaign_2026-10-06/STRETCH.md](<../../campaign_2026-10-06/STRETCH.md>)
  - **Event/quiet residual belief updates** (`arm:event-quiet-state`) — not_tested; eligible=True.
    - **next_test:** Independent implementation and small resource profile; compare unrestricted residual and lagged-event control, preserving all-row output.
    - **claim_scope:** specific recipe / exposed-data exploration; no inferred public gain
    - artifact: [research/campaign_2026-10-05/continuation_night/event_semantics/NEXT_TEST_CONTRACT.md](<../../campaign_2026-10-05/continuation_night/event_semantics/NEXT_TEST_CONTRACT.md>)
- **Residual diagonal / content / erase-write memory** (`arm:residual-memory`) — prepared_unrun; eligible=True.
  - **next_test:** Integrate causal sequence reset/detach and event-index contract; compare fixed additive versus delta memory with same parent.
  - **claim_scope:** specific recipe / exposed-data exploration; no inferred public gain
  - artifact: [research/campaign_2026-10-05/continuation_night/residual_memory/DESIGN.md](<../../campaign_2026-10-05/continuation_night/residual_memory/DESIGN.md>)
- **Quiet-row output holds / online corrections: tested recipes** (`arm:quiet-output-holds`) — screen_failed; eligible=False.
  - **next_test:** Preserve measured failures; any child must change the mechanism and match a complete predictor control.
  - **claim_scope:** specific recipe / exposed-data exploration; no inferred public gain
  - artifact: [research/campaign_2026-10-06/ONLINE.md](<../../campaign_2026-10-06/ONLINE.md>)
  - artifact: [research/campaign_2026-10-06/STRETCH.md](<../../campaign_2026-10-06/STRETCH.md>)

### Targets, losses and supervision (`family:objective`)

- **Fixed readout mixture on the complete GRU256 recipe** (`arm:readout-transfer`) — prepared_unrun; eligible=True.
  - **next_test:** Kaggle B16 common control and fixed mixture: profile first, paired fits and fixed complete endpoint. Tiny-panel gains do not establish public transfer.
  - **status_scope:** Latest 3GB-cap packet/draft only; prior attempts/profiles retained separately in raw ledger events and source receipts.
  - **claim_scope:** specific recipe / exposed-data exploration; no inferred public gain
  - artifact: [research/campaign_2026-10-06/cap3g_transfer_ram12/RECEIPT.json](<../../campaign_2026-10-06/cap3g_transfer_ram12/RECEIPT.json>)
- **Original-label reanchor versus unchanged / bootstrap-off** (`arm:reanchor`) — prepared_unrun; eligible=True.
  - **next_test:** Profile and run matched reanchor0.1, unchanged bootstrap and bootstrap-off on a common full recipe.
  - **status_scope:** Latest 3GB-cap packet/draft only; prior attempts/profiles retained separately in raw ledger events and source receipts.
  - **claim_scope:** specific recipe / exposed-data exploration; no inferred public gain
  - **planned_experiment_ids:** ['E-494']
  - **registration_note:** E-494 is the prepared study label; no experiment ledger row exists yet.
  - artifact: [research/campaign_2026-10-06/cap3g_transfer_ram12/RECEIPT.json](<../../campaign_2026-10-06/cap3g_transfer_ram12/RECEIPT.json>)
- **Distillation alpha × bootstrap horizon** (`arm:supervision-frontier`) — interrupted; eligible=True.
  - **next_test:** Recover fit-only calibration on Kaggle; bind measured mass vectors, then profile paired alpha/horizon and direction contrasts.
  - **claim_scope:** specific recipe / exposed-data exploration; no inferred public gain
  - **evidence state:** E-496 CPU calibration attempt — interrupted: exit -15 after 751.8seconds; no report; own cgroup OOM kills zero; parent counter 3 before and after; laptop OOM cause unestablished (source: [research/campaign_2026-10-06/STATUS_AFTER_OOM.md](<../../campaign_2026-10-06/STATUS_AFTER_OOM.md>))
  - artifact: [research/campaign_2026-10-05/continuation_night/supervision_frontier/E496_HANDOFF_PACKET.json](<../../campaign_2026-10-05/continuation_night/supervision_frontier/E496_HANDOFF_PACKET.json>)
  - artifact: [research/campaign_2026-10-06/STATUS_AFTER_OOM.md](<../../campaign_2026-10-06/STATUS_AFTER_OOM.md>)
  - **Measured denominator: chunk / fixed / shuffled mass** (`arm:denominator`) — not_tested; eligible=False.
    - **next_test:** Obtain genuine fit-pass calibration; resolve fixed/shuffled templates with unchanged order/control and actual teacher epoch hashes.
    - **claim_scope:** specific recipe / exposed-data exploration; no inferred public gain
- **Hole-only episode loss in a tiny recurrent specialist** (`arm:quiet-episode-specialist`) — insufficient_evidence; eligible=True.
  - **next_test:** Preserve two positive specialist seeds but first test a champion-preserving residual/specialist contribution; tiny absolute score is below incumbent.
  - **claim_scope:** specific recipe / exposed-data exploration; no inferred public gain
  - artifact: [research/campaign_2026-10-06/TINY_GRU.md](<../../campaign_2026-10-06/TINY_GRU.md>)
  - artifact: [research/campaign_2026-10-06/wave1_tiny_gru_s0.json](<../../campaign_2026-10-06/wave1_tiny_gru_s0.json>)
  - artifact: [research/campaign_2026-10-06/wave1_tiny_gru_s1.json](<../../campaign_2026-10-06/wave1_tiny_gru_s1.json>)
  - **All-row and mixed episode objectives: tested recipes** (`arm:quiet-episode-global`) — screen_failed; eligible=False.
    - **next_test:** A new descendant must specify what repairs the tested negative objective interaction; no blanket family rejection.
    - **claim_scope:** specific recipe / exposed-data exploration; no inferred public gain
    - artifact: [research/campaign_2026-10-06/wave1_tiny_gru_all_s0.json](<../../campaign_2026-10-06/wave1_tiny_gru_all_s0.json>)
    - artifact: [research/campaign_2026-10-06/wave1_tiny_gru_mixed_s0.json](<../../campaign_2026-10-06/wave1_tiny_gru_mixed_s0.json>)

### Data and feature representation (`family:representation`)

No curated nodes.

### Optimization and training stability (`family:optimization`)

- **Update cadence and recurrent gradient credit** (`arm:cadence`) — prepared_unrun; eligible=True.
  - **next_test:** Profile matched1000/1000,2000/1000,2000/2000 with row-matched EMA on Kaggle; choose a common batch within3GB.
  - **status_scope:** Latest 3GB-cap packet/draft only; prior attempts/profiles retained separately in raw ledger events and source receipts.
  - **claim_scope:** specific recipe / exposed-data exploration; no inferred public gain
  - artifact: [research/campaign_2026-10-06/cap3g_cadence_ram12/RECEIPT.json](<../../campaign_2026-10-06/cap3g_cadence_ram12/RECEIPT.json>)
  - **Gradient cadence × measured denominator** (`arm:cadence-denominator-interaction`) — not_tested; eligible=False.
    - **next_test:** Hold row budget, EMA horizon, teacher lineage and sampler fixed across a small factorial; inspect pooled loss and complete prediction endpoint.
    - **claim_scope:** A new interaction proposal, no run or forecast
- **Precision and optimizer-step mechanism** (`arm:optimizer-precision`) — insufficient_evidence; eligible=False.
  - **next_test:** Resolve the seed-specific AdamW update discrepancy and clipping-coverage scope before admitting a full recipe; preserve diagnostic-only results.
  - **claim_scope:** specific recipe / exposed-data exploration; no inferred public gain
  - artifact: [research/campaign_2026-10-05/continuation_night/admission/E499_ACTUAL_DIAGNOSTIC_REVIEW.md](<../../campaign_2026-10-05/continuation_night/admission/E499_ACTUAL_DIAGNOSTIC_REVIEW.md>)

### Diversity, ensembles and interactions (`family:ensemble`)

- **Matched width / optimizer / teacher capacity** (`arm:width`) — prepared_unrun; eligible=True.
  - **next_test:** Finish3GB-capped Kaggle pairedW256/W384 resource probes, B16 with commonB8 fallback, then same-optimizer fixed endpoints.
  - **status_scope:** Latest 3GB-cap packet/draft only; prior attempts/profiles retained separately in raw ledger events and source receipts.
  - **claim_scope:** specific recipe / exposed-data exploration; no inferred public gain
  - **evidence state:** prior W384 provider packet — ERROR in compatibility preflight before scientific fits (source: [research/campaign_2026-10-06/STATUS_AFTER_OOM.md](<../../campaign_2026-10-06/STATUS_AFTER_OOM.md>))
  - **evidence state:** W384 v3r1 immutable handoff packet — review GO for handoff only; no push or fit during review; admission required under current 3GB cap (source: [research/campaign_2026-10-05/continuation_night/admission/W384_V3R1_INDEPENDENT_REVIEW_GO.md](<../../campaign_2026-10-05/continuation_night/admission/W384_V3R1_INDEPENDENT_REVIEW_GO.md>))
  - **evidence state:** latest 3GB packet — prepared; not evidence of completed scientific width comparison (source: [research/campaign_2026-10-06/STATUS_AFTER_OOM.md](<../../campaign_2026-10-06/STATUS_AFTER_OOM.md>))
  - artifact: [research/campaign_2026-10-05/continuation_night/admission/W384_V3R1_INDEPENDENT_REVIEW_GO.md](<../../campaign_2026-10-05/continuation_night/admission/W384_V3R1_INDEPENDENT_REVIEW_GO.md>)
  - artifact: [research/campaign_2026-10-06/STATUS_AFTER_OOM.md](<../../campaign_2026-10-06/STATUS_AFTER_OOM.md>)
- **Serving-feasible seed mixtures and tensor soups** (`arm:seed-ensemble`) — not_tested; eligible=True.
  - **next_test:** Score exact incumbent replacement weights; separate training seeds from checkpoint replicas and verify delivered ZIP runtime.
  - **claim_scope:** specific recipe / exposed-data exploration; no inferred public gain
  - **Independent student complementarity and causal blend gates** (`arm:complementarity-gated-blend`) — not_tested; eligible=False.
    - **next_test:** Aligned common-endpoint predictions: compare frozen equal/fixed-weight blend with a predeclared causal-context gate; account for weight/gate selection on exposed slices, then frozen other-slice evaluation.
    - **claim_scope:** New proposed protocol; no fit, reproduced improvement or public forecast
    - artifact: [research/campaign_2026-10-06/branch_teacher_audit/post_fetch_b7fd5d2f/OBSERVATIONS.md](<../../campaign_2026-10-06/branch_teacher_audit/post_fetch_b7fd5d2f/OBSERVATIONS.md>)

### Serving and compression (`family:deployment`)

- **Quantization and native runtime headroom** (`arm:deployment`) — not_tested; eligible=True.
  - **next_test:** Recheck exact ZIP ABI and realistic timing before submission; use saved public results as anchor, not proxy forecast.
  - **claim_scope:** specific recipe / exposed-data exploration; no inferred public gain

### Evaluation, data exposure and diagnostics (`family:evaluation`)

No curated nodes.

### Compute, safety and recovery (`family:infrastructure`)

- **OOM, provider and monitor recovery** (`arm:resource-recovery`) — insufficient_evidence; eligible=True.
  - **next_test:** Diagnose failed TPUpreflight and aggregatehostmemory pressure; provider-only experiments afterlocalcutoff, bounded reviewed retries.
  - **claim_scope:** specific recipe / exposed-data exploration; no inferred public gain
  - **evidence state:** E-463 and provider monitor history — earlier recorded running state is not a current live-status check (source: [research/experiments.jsonl](<../../experiments.jsonl>))
  - **evidence state:** saved night W384 and TPU monitor, Oct6 19:57 Kyiv — both ERROR; no active experiment service; watchers remain active (source: [research/campaign_2026-10-06/STATUS_AFTER_OOM.md](<../../campaign_2026-10-06/STATUS_AFTER_OOM.md>))
  - **evidence state:** current session observability — systemd bus unavailable; NVIDIA query failed; live state unknown (source: [research/campaign_2026-10-06/STATUS_AFTER_OOM.md](<../../campaign_2026-10-06/STATUS_AFTER_OOM.md>))
  - artifact: [research/campaign_2026-10-06/STATUS_AFTER_OOM.md](<../../campaign_2026-10-06/STATUS_AFTER_OOM.md>)
  - artifact: [research/campaign_2026-10-06/kaggle_inbox/NOTE.md](<../../campaign_2026-10-06/kaggle_inbox/NOTE.md>)

### Unresolved classification (`family:unclassified`)

No curated nodes.

## Node and source index

Each raw ledger event appears as its own source link, including repeated IDs and status updates.

| Node ID | Title | Type / latest recorded status | Parent | Source rows | Evidence refs |
|---|---|---|---|---|---|
| root | Wunder Connectome Challenge2 research | root | — | [research/bandit_tree/config.json](<../config.json>) | — |
| family:teacher | Teachers and knowledge transfer | family | root | [research/bandit_tree/config.json](<../config.json>) | — |
| family:temporal | Temporal state and memory | family | root | [research/bandit_tree/config.json](<../config.json>) | — |
| family:objective | Targets, losses and supervision | family | root | [research/bandit_tree/config.json](<../config.json>) | — |
| family:representation | Data and feature representation | family | root | [research/bandit_tree/config.json](<../config.json>) | — |
| family:optimization | Optimization and training stability | family | root | [research/bandit_tree/config.json](<../config.json>) | — |
| family:ensemble | Diversity, ensembles and interactions | family | root | [research/bandit_tree/config.json](<../config.json>) | — |
| family:deployment | Serving and compression | family | root | [research/bandit_tree/config.json](<../config.json>) | — |
| family:evaluation | Evaluation, data exposure and diagnostics | family | root | [research/bandit_tree/config.json](<../config.json>) | — |
| family:infrastructure | Compute, safety and recovery | family | root | [research/bandit_tree/config.json](<../config.json>) | — |
| family:unclassified | Unresolved classification | family | root | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:causal-transformer-teacher | Causal segment-memory Transformer teacher | arm · not_tested | family:teacher | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:transformer-student | Transfer Transformer predictions into the serving GRU | arm · not_tested | arm:causal-transformer-teacher | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:transformer-temporal-transfer | Temporal summary / representation transfer | arm · not_tested | arm:transformer-student | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:transformer-quiet-transfer | Quiet-row transfer from an attention teacher | arm · not_tested | arm:transformer-student | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:privileged-transformer | Origin-target privileged attention teacher | arm · not_tested | arm:causal-transformer-teacher | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:teacher-crossfit | Teacher lineage, cross-fitting and ensemble targets | arm · not_tested | family:teacher | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:readout-transfer | Fixed readout mixture on the complete GRU256 recipe | arm · prepared_unrun | family:objective | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:reanchor | Original-label reanchor versus unchanged / bootstrap-off | arm · prepared_unrun | family:objective | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:supervision-frontier | Distillation alpha × bootstrap horizon | arm · interrupted | family:objective | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:denominator | Measured denominator: chunk / fixed / shuffled mass | arm · not_tested | arm:supervision-frontier | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:cadence | Update cadence and recurrent gradient credit | arm · prepared_unrun | family:optimization | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:optimizer-precision | Precision and optimizer-step mechanism | arm · insufficient_evidence | family:optimization | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:event-clock | Production event clock / quiet-row anatomy | arm · diagnostic_completed | family:temporal | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:event-quiet-state | Event/quiet residual belief updates | arm · not_tested | arm:event-clock | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:residual-memory | Residual diagonal / content / erase-write memory | arm · prepared_unrun | family:temporal | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:quiet-episode-specialist | Hole-only episode loss in a tiny recurrent specialist | arm · insufficient_evidence | family:objective | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:quiet-episode-global | All-row and mixed episode objectives: tested recipes | arm · screen_failed | arm:quiet-episode-specialist | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:quiet-output-holds | Quiet-row output holds / online corrections: tested recipes | arm · screen_failed | family:temporal | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:width | Matched width / optimizer / teacher capacity | arm · prepared_unrun | family:ensemble | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:seed-ensemble | Serving-feasible seed mixtures and tensor soups | arm · not_tested | family:ensemble | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:deployment | Quantization and native runtime headroom | arm · not_tested | family:deployment | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:resource-recovery | OOM, provider and monitor recovery | arm · insufficient_evidence | family:infrastructure | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:transformer-reanchor-interaction | Attention-teacher transfer × origin-label reanchor | arm · not_tested | arm:transformer-student | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:cadence-denominator-interaction | Gradient cadence × measured denominator | arm · not_tested | arm:cadence | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:teacher-convergence-context | Convergence-aware causal attention and event context | arm · not_tested | arm:causal-transformer-teacher | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:learnable-privileged-transfer | Transfer predictable teacher structure separately from future innovation | arm · not_tested | arm:privileged-transformer | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:complementarity-gated-blend | Independent student complementarity and causal blend gates | arm · not_tested | arm:seed-ensemble | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:heterogeneous-teacher-compression | Compress complementary teacher recipes into a GRU256 student pair | arm · not_tested | family:teacher | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:two-stream-teacher-compression | Replace a teacher family with a two-stream teacher | arm · not_tested | arm:heterogeneous-teacher-compression | [research/bandit_tree/config.json](<../config.json>) | — |
| arm:distilled-function-preserving-widening | Function-preserving widening of a distilled student | arm · not_tested | arm:heterogeneous-teacher-compression | [research/bandit_tree/config.json](<../config.json>) | — |
| hypothesis:H-001 | Targets change only on rows where i0 bid and ask price columns both update. | hypothesis · supported | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L1>) | experiment:E-001 (linked) |
| hypothesis:H-002 | Each sequence's targets are integer multiples of one of four quanta {0.12543,0.24133,0.46854,0.65687}, defining market classes with different tick regimes. | hypothesis · supported | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L2>) | experiment:E-001 (linked), experiment:E-002 (linked) |
| hypothesis:H-003 | Features are rank-Gauss transformed: bounded near +-2.326 (Phi^-1(0.99)) with sentinel -5.19934 = Phi^-1(1e-7) for missing values. | hypothesis · probing | family:representation | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L3>) | experiment:E-001 (linked) |
| hypothesis:H-004 | Training the row-clock GRU on all 10,607 sequences for more epochs lifts validation WP well above baseline. | hypothesis · refuted | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L4>) | experiment:E-006 (linked), experiment:E-007 (linked) |
| hypothesis:H-005 | A Rust per-row GRU kernel is several times faster than onnxruntime for the dense GRU 256x2. | hypothesis · refuted | family:deployment | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L5>) | experiment:E-004 (linked), experiment:E-005 (linked) |
| hypothesis:H-006 | Skipping zero input columns in the input projection cuts its cost proportional to density. | hypothesis · supported | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L6>) | experiment:E-005 (linked) |
| hypothesis:H-007 | is_scored is predictable from causal features, so propensity-weighted training aligns the loss with scored rows. | hypothesis · supported | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L7>) | experiment:E-008 (linked), experiment:E-031 (linked), experiment:E-034 (linked), experiment:E-036 (linked), experiment:E-038 (linked) |
| hypothesis:H-008 | Class-3 targets are a function of recent event-clock book states, so event-lag features fix class 3. | hypothesis · refuted | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L8>) | experiment:E-009 (linked), experiment:E-010 (linked) |
| hypothesis:H-009 | Class-3 WP is bounded by predictability of the next book-state change during a visit. | hypothesis · supported | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L9>) | experiment:E-011 (linked) |
| hypothesis:H-010 | Snapshot queue/flow features (volumes, depletion since event, trades, i1, a*) predict class-3 break direction throughout the visit. | hypothesis · refuted | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L10>) | experiment:E-012 (linked), experiment:E-013 (linked) |
| hypothesis:H-011 | Online (per-sequence, label-free) transition statistics of book states predict the next break direction early in a visit, lifting class 3. | hypothesis · refuted | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L11>) | experiment:E-014 (linked) |
| hypothesis:H-012 | Class-3 scored rows concentrate just before breaks, so near-break weighting would lift class 3. | hypothesis · refuted | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L12>) | experiment:E-016 (linked) |
| hypothesis:H-013 | i1 price moves lead i0 breaks, so i1 moves accumulated since the last i0 break predict break direction far from the break. | hypothesis · refuted | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L13>) | experiment:E-017 (linked) |
| hypothesis:H-014 | Class-3 WP is capped near 0.42-0.50 because break direction is only predictable within ~20 rows of a break. | hypothesis · supported | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L14>) | experiment:E-013 (linked), experiment:E-014 (linked), experiment:E-018 (linked), experiment:E-021 (linked) |
| hypothesis:H-015 | Targets are future level changes over a horizon on the event clock (t0 ~ future bid - current ask, t1 ~ current bid - future ask), which would make past targets computable online after the horizon. | hypothesis · probing | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L15>) | experiment:E-015 (linked) |
| hypothesis:H-016 | Class-3 targets are anchor minus current price with a slow anchor, so a label-free level reconstruction from price jumps and its deviation from an EMA equilibrium predicts targets. | hypothesis · refuted | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L16>) | experiment:E-019 (linked), experiment:E-020 (linked), experiment:E-021 (linked) |
| hypothesis:H-017 | A 64-byte-aligned weight arena in access order, zero-alloc stepping, folded-normalization sparse input and vectorizable bounded-error activations speed up the per-row GRU by >= 1.5x. | hypothesis · refuted | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L17>) | experiment:E-022 (linked) |
| hypothesis:H-018 | Training directly on the competition objective (weighted Pearson loss per TBPTT chunk) beats the mixed corr + weighted-MSE loss. | hypothesis · refuted | family:objective | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L18>) | experiment:E-023 (linked) |
| hypothesis:H-019 | Train and validation differ in market-class mix; matching the validation mix by weighted sampling improves validation WP. | hypothesis · probing | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L19>) | experiment:E-024 (linked) |
| hypothesis:H-020 | An event-clock GRU stepping only on i0 price events, alongside the row GRU, carries event-level state across long quiet visits and lifts classes 2-3. | hypothesis · refuted | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L20>) | experiment:E-056 (linked) |
| hypothesis:H-021 | Sentinel -5.19934 (missing) values distort per-column normalization; replacing them with a neutral value plus missing flags improves WP. | hypothesis · open | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L21>) | — |
| hypothesis:H-022 | Class experts (or a model trained without class-3 dominance) lift classes 0-2, where headroom exists. | hypothesis · refuted | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L22>) | experiment:E-055 (linked) |
| hypothesis:H-023 | The WP-optimal prediction is E[m\|y\|y\|x]/E[m\|y\|\|x] with m = is_scored; training with mask x \|y\| weights beats \|y\| weights alone. | hypothesis · supported | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L23>) | experiment:E-030 (linked), experiment:E-033 (linked) |
| hypothesis:H-024 | Residual decomposition by class x rows-to-break x \|y\| locates recoverable error in classes 0-2. | hypothesis · proposed | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L24>) | — |
| hypothesis:H-025 | Validation-period data is closer to test conditions; fine-tuning on devA improves devB more than the same amount of extra train data. | hypothesis · inconclusive | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L25>) | experiment:E-030 (linked) |
| hypothesis:H-026 | Auxiliary heads predicting the target 20 and 200 rows ahead (Jane Street multi-responder pattern) regularize the main head. | hypothesis · refuted | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L26>) | experiment:E-027 (linked) |
| hypothesis:H-027 | More recurrent capacity lifts WP; with limited data the gain is only realized with early stopping or more sequences. | hypothesis · probing | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L27>) | experiment:E-028 (linked) |
| hypothesis:H-028 | Class-0/1 models under-react to imminent large moves one to a few rows before a price break, where direction is observable; a sharper near-break pathway (event-time features, dual clock, or a dedicated head) recovers that gap. | hypothesis · refuted | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L28>) | experiment:E-029 (linked), experiment:E-056 (linked) |
| hypothesis:H-029 | Tempering propensity weights p^alpha trades bias for variance; alpha != 1 beats alpha = 1. | hypothesis · supported | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L29>) | experiment:E-039 (linked), experiment:E-048 (linked) |
| hypothesis:H-030 | Training from scratch on all train sequences with propensity weights exceeds devA fine-tuning gains. | hypothesis · supported | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L30>) | experiment:E-038 (linked) |
| hypothesis:H-031 | is_scored is (nearly) a deterministic function of near-future events; adding K future rows lifts propensity AUC toward 1. | hypothesis · refuted | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L31>) | experiment:E-037 (linked) |
| hypothesis:H-032 | A propensity-gated mixture (separate heads for high- and low-propensity rows) improves on single-head weighting. | hypothesis · proposed | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L32>) | — |
| hypothesis:H-033 | A propensity-weighted GRU small enough for the 1-vCPU budget (hidden 128-192) keeps most of the +0.034 gain. | hypothesis · supported | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L33>) | experiment:E-039 (linked), experiment:E-040 (linked), experiment:E-042 (linked) |
| hypothesis:H-034 | Feeding the causal propensity p(x) as an input feature (not only as a loss weight) improves WP. | hypothesis · refuted | family:representation | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L34>) | experiment:E-045 (linked) |
| hypothesis:H-035 | Mask-weighted devA fine-tuning on top of the full-train propensity model adds beyond either alone. | hypothesis · supported | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L35>) | experiment:E-045 (linked) |
| hypothesis:H-036 | A single stateful ONNX graph (features + GRU) runs within the 1-vCPU budget at hidden <= 192 once the feature block is fused into few ops. | hypothesis · supported | family:deployment | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L36>) | experiment:E-041 (linked), experiment:E-043 (linked) |
| hypothesis:H-037 | Averaging weights over the last 30% of steps (SWA) improves the h128 propensity model at zero inference cost. | hypothesis · refuted | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L37>) | experiment:E-052 (linked) |
| hypothesis:H-038 | Causal per-sequence running z-scores of key features make inputs regime-relative and lift cross-class generalization. | hypothesis · proposed | family:representation | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L38>) | — |
| hypothesis:H-039 | With propensity^2 x \|y\| weights, the weighted-MSE term alone is WP-optimal and beats the corr + 0.5 wMSE mix. | hypothesis · refuted | family:objective | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L39>) | experiment:E-051 (linked) |
| hypothesis:H-040 | The competition metric is now global (pooled) weighted Pearson; the target level (class/spread-driven, 18-20% of weighted variance) and output scale matter, which per-sequence WP ignored. | hypothesis · supported | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L40>) | experiment:E-060 (linked), experiment:E-061 (linked) |
| hypothesis:H-041 | Clip at +-2 punishes large-scale outputs: rescaling the organizer baseline by ~0.25 lifts validation WP 0.6171 -> 0.6728 (global) and 0.5896 -> 0.6316 (per-seq). | hypothesis · supported | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L41>) | experiment:E-061 (linked) |
| hypothesis:H-042 | Target law: with U = F - bid (common future price F, integer ticks) and current spread sp, t0 = U - sp and t1 = -U when F is inside [bid, ask]; above the ask t1 = -(sp+2); below the bid t0 = -(sp+2). Targets are a deterministic 1-D copula of (sp, U). | hypothesis · supported (class 3 exact); approximate for classes 0-2 | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L42>) | experiment:E-063 (linked), experiment:E-064 (linked), experiment:E-074 (linked) |
| hypothesis:H-043 | Knowing the branch (F below bid / inside spread / above ask) given class and spread gives WP 0.968; magnitude adds only 0.02. The problem reduces to 3-way classification plus analytic tilted readout. | hypothesis · supported | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L43>) | experiment:E-065 (linked) |
| hypothesis:H-044 | A rational readout p = sum_b P(b\|x) M1_b / sum_b P(b\|x) M0_b beats linear stacking on the same information (+0.015). | hypothesis · refuted as causal gain: E-066 relied on oracle spread (E-073, E-074) | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L44>) | experiment:E-066 (linked) |
| hypothesis:H-045 | F is the price at a fixed real-time horizon (rows are irregular in time); a-columns encode clocks. If found, past U become known online and give per-sequence supervision. | hypothesis · supported for class 3 in revised form: F = round(mean mid over next 4 book events), not a real-time horizon | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L45>) | experiment:E-062 (linked), experiment:E-064 (linked), experiment:E-079 (linked) |
| hypothesis:H-046 | Loss: a branch/U categorical head trained with CE (propensity-weighted) and read out analytically beats wMSE, which beats Huber and per-sequence correlation losses. | hypothesis · proposed | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L46>) | — |
| hypothesis:H-047 | The teammate's compact raw-plus-first-difference GRU trained with our causal since weighting and cross-fitted propensity mask improves the current public since/since pair. | hypothesis · rejected_for_public_incumbent_pair | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L47>) | The merged model improved a different since/base reference pair, but adding it to the exact public since/since pair reduced score at every tested weight from 0.025 to 0.50 on OOF folds 0-1; both exact and lambda-0.02 intervals were below zero even at weight 0.025. |
| hypothesis:H-048 | Stable structured recurrent state preserves useful multiscale evidence, complementary to the clean since/since pair. | hypothesis · proposed_not_fit | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L48>) | — |
| hypothesis:H-049 | Normalized content-addressed recurrent memory improves target WP over decay-matched positional memory. | hypothesis · proposed_not_fit | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L49>) | — |
| hypothesis:H-050 | A causal specialist-state mixture improves the target map over hash routing and a universal operator. | hypothesis · proposed_not_fit | family:objective | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L50>) | — |
| hypothesis:H-051 | Current a6 endpoint clock plus learned content retrieval retains useful event context beyond a periodic Walsh position control. Clock-source comparison tests a joint write/decay intervention. | hypothesis · cpu_implementation_pending_independent_grade | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L51>) | experiment:E-273 (linked) |
| hypothesis:H-052 | On the same TRAIN-code known rows, current-spread auxiliary can improve shared GRU representation over original future-proxy spread; original proxy may retain useful future-tail information. | hypothesis · cpu_cache_and_patch_pending_independent_grade | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L52>) | experiment:E-272 (linked) |
| hypothesis:H-053 | Two monotone age-like features may admit a nondegenerate common increment coordinate useful for variable-time state modeling. Physical units are not assumed. | hypothesis · blocked_by_frozen_fit_gate_failure | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L53>) | experiment:E-274 (linked) |
| hypothesis:H-054 | A GRU256 since-student distilled (alpha .5) from the fold's two GRU128 since parents, then GCLP FT, adds >= +0.0005 pooled Global WP when blended equally with the since pair on clean folds 0 and 1; head-only FT is predeclared as a second arm. | hypothesis · running | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L54>) | — |
| hypothesis:H-055 | GCLP FT whose correlation uses decayed pooled sums of earlier chunks (decay .995) follows the whole-pool Pearson gradient and beats batch-pooled gcorr FT, because batch-shared offsets/scales become visible. | hypothesis · queued | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L55>) | — |
| hypothesis:H-056 | Jointly fine-tuning members' heads on the pooled correlation of their equal-weight average beats a matched solo extra head FT, since members' effective scale inside the raw average is otherwise untrained. | hypothesis · implemented | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L56>) | — |
| hypothesis:H-057 | Convex member weights fitted on other folds' OOF rows improve the held-out fold over equal weights once members differ in type (GRU256 student vs GRU128 since). | hypothesis · tool_ready | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L57>) | — |
| hypothesis:H-058 | An LSTM128 since member (seed 5, same recipe) adds diversity beyond GRU seeds: pair+LSTM equal thirds beats the pair by >= +0.0005 on clean folds 0 and 1. Online refit part of the source is not transferable (no labels at Wunder test time). | hypothesis · queued | family:ensemble | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L58>) | — |
| hypothesis:H-059 | Distillation weight should grow with the label's distance to resolution: target = (1-a(u)) y + a(u) teacher with u = rows until the next target change (known at train time from y; inputs stay causal). Expected to beat a constant alpha because the alpha 0.7 gain is concentrated far from the next quote move (F5) where ~44-50% of the metric loss lives (F3). | hypothesis · proposed | family:teacher | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L59>) | experiment:E-320 (linked), experiment:E-323 (linked) |
| hypothesis:H-060 | At alpha 0.7 the student learns mostly from the teacher mean, so teacher quality bounds it. Adding the fold-bound alpha-0.7 GRU256 student to the two since GRU128 teachers (equal thirds) gives a better-denoised target and a better student, even though averaging that student with weaker members hurts at inference (E-326 D05_D07 -0.0009). | hypothesis · unknown_pending_evidence_audit | family:teacher | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L60>) | — |
| hypothesis:H-062 | Uniform soup of the student's fixed-epoch EMA checkpoints (ep2+ep3, ep1-3) beats the ep3 checkpoint: EMA 0.999 spans ~0.26 epoch (3,780 optimizer steps/epoch), a soup spans 1-2 epochs and flattens toward a wider optimum. | hypothesis · unknown_pending_evidence_audit | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L61>) | — |
| hypothesis:H-063 | Pre-registered from E-323 (parent better in class-3 long runs, head-FT better in classes 1-2): class 3 -> 0.75 parent/0.25 FT head, classes 0-2 -> 0.25/0.75, beats the 0.5/0.5 merged heads. | hypothesis · unknown_pending_evidence_audit | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L62>) | — |
| hypothesis:H-064 | Seed partners of the GRU256 student are 0.985-0.999 residual-correlated, so a bag gains only ~+0.0005 (holdout). A distilled LSTM student learns the same smooth teacher target (so it can be nearly as strong) with a different recurrence, giving lower residual correlation and a larger bag gain. LSTM192 is sized so GRU256 + LSTM192 can fit the runtime budget. | hypothesis · unknown_pending_evidence_audit | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L63>) | — |
| hypothesis:H-065 | Matched single-trunk parent/head combo improves the complete predictor | hypothesis · supported | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L64>) | — |
| hypothesis:H-066 | Width384 improves on the current trunk at useful cost | hypothesis · probing | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L65>) | — |
| hypothesis:H-067 | A second combo seed improves the score/runtime frontier | hypothesis · open | family:deployment | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L66>) | — |
| hypothesis:H-068 | Byte weights reduce memory traffic without material WP loss | hypothesis · probing | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L67>) | — |
| hypothesis:H-069 | Quantization-aware distillation repairs compressed recurrent accuracy | hypothesis · probing | family:deployment | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L68>) | — |
| hypothesis:H-070 | Larger state with rank128 recurrence gives useful score per CPU cost | hypothesis · open | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L69>) | — |
| hypothesis:H-071 | EMA retention per row explains part of chunk1000 behavior | hypothesis · probing | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L70>) | — |
| hypothesis:H-072 | Midpoint credit affects prediction independently of optimizer cadence | hypothesis · open | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L71>) | — |
| hypothesis:H-073 | Aligned chunk-mass correction changes useful training emphasis | hypothesis · probing | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L72>) | — |
| hypothesis:H-074 | Late Muon to AdamW improves final fit | hypothesis · open | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L73>) | — |
| hypothesis:H-075 | Module clipping prevents head gradients suppressing recurrent learning | hypothesis · open | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L74>) | — |
| hypothesis:H-076 | Order pairing reduces trajectory variance | hypothesis · open | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L75>) | — |
| hypothesis:H-077 | Current teacher does not absorb all benefits of row weighting | hypothesis · open | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L76>) | — |
| hypothesis:H-078 | Run-relative volume history improves representation inside GRU | hypothesis · open | family:representation | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L77>) | — |
| hypothesis:H-079 | Changed truncation boundaries provide useful training views | hypothesis · open | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L78>) | — |
| hypothesis:H-080 | Label-aligned thinning augmentation creates useful invariance | hypothesis · open | family:representation | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L79>) | — |
| hypothesis:H-081 | Explicit missingness avoids distortion by sentinel scale | hypothesis · open | family:representation | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L80>) | — |
| hypothesis:H-082 | Learned regime experts improve modern representations | hypothesis · open | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L81>) | — |
| hypothesis:H-083 | Offsets/amplitudes improve the strong model under cleaner calibration | hypothesis · open | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L82>) | — |
| hypothesis:H-084 | Targets benefit from different row/event histories | hypothesis · open | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L83>) | — |
| hypothesis:H-085 | Activity modulates useful EMA information | hypothesis · open | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L84>) | — |
| hypothesis:H-086 | i1 history provides conditional residual information | hypothesis · open | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L85>) | — |
| hypothesis:H-087 | Event age modulates useful retained history | hypothesis · open | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L86>) | — |
| hypothesis:H-088 | Anonymous transition structure predicts without physical labels | hypothesis · open | family:representation | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L87>) | — |
| hypothesis:H-089 | Cycle-based aggregation provides useful smoothing | hypothesis · open | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L88>) | — |
| hypothesis:H-090 | Factorized branch and magnitude improve modern readout | hypothesis · open | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L89>) | — |
| hypothesis:H-091 | Content addressing retains useful event information | hypothesis · open | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L90>) | — |
| hypothesis:H-092 | Explicit stable modes improve retained weak signals | hypothesis · open | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L91>) | — |
| hypothesis:H-093 | Activity specialists retain different causal information | hypothesis · screened_uncertain | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L92>) | — |
| hypothesis:H-094 | Prefix context changes feature interpretation | hypothesis · screened_uncertain | family:representation | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L93>) | — |
| hypothesis:H-095 | Hidden state can learn stable gain from disjoint episodes | hypothesis · probing | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L94>) | — |
| hypothesis:H-096 | Cross-fitted targets reduce student bias | hypothesis · open | arm:transformer-student | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L95>) | — |
| hypothesis:H-097 | Confirmed modern teachers produce a better causal student | hypothesis · open | arm:causal-transformer-teacher | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L96>) | — |
| hypothesis:H-098 | Short future features teach the original causal target | hypothesis · open | arm:privileged-transformer | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L97>) | — |
| hypothesis:H-099 | Useful retained context can be localized by reset lag | hypothesis · probing | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L98>) | — |
| hypothesis:H-100 | ARTBP with linear optimizer improves forecasting | hypothesis · open | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L99>) | — |
| hypothesis:H-101 | Current-spread supervision may interact with modern combo | hypothesis · open | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L100>) | — |
| hypothesis:H-102 | An LSTM128 partner complements the modern combo | hypothesis · open | family:ensemble | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L101>) | — |
| hypothesis:H-103 | Broader support yields a noncollapsed predictive increment feature | hypothesis · open | family:representation | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L102>) | — |
| hypothesis:H-104 | Recomputing stale incoming state can improve current recipe | hypothesis · open | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L103>) | — |
| hypothesis:H-105 | Numerical precision changes matter for current learner | hypothesis · probed_no_clear_gain | arm:optimizer-precision | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L104>) | — |
| hypothesis:H-106 | A soft prefix mixture sends direct prediction gradients through every specialist and may retain specialization without hard top-one routing. | hypothesis · open | family:infrastructure | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L105>) | — |
| hypothesis:H-107 | A consistency teacher averaged in observed endpoint-event time may be better aligned than a teacher averaged per optimizer update. | hypothesis · probing | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L106>) | — |
| hypothesis:H-108 | Causal endpoint-relative volume history can modulate GRU update strength even if additive volume features are weak. | hypothesis · open | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L107>) | — |
| hypothesis:H-109 | The fixed final head procedure may change specialist/adapter comparisons that currently use parent-only predictors. | hypothesis · open | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L108>) | — |
| hypothesis:H-110 | Full causal GRU continuation can exploit all eight TPUs with globally normalized weighted gradients; BF16 may change score/runtime against matched FP32. | hypothesis · open | family:infrastructure | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L109>) | — |
| hypothesis:H-111 | Train-defined paired likelihood plus a target-specific weighted-Bayes readout and anchored regression residual may improve finite-sample learning. | hypothesis · open | family:objective | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L110>) | — |
| hypothesis:H-112 | Stable diagonal memory and bounded content-addressed recurrent state may improve causal prediction beyond matched fixed-spectrum/static branches. | hypothesis · open | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L111>) | — |
| hypothesis:H-113 | Learned univariate spline edges on causal bounded latent state may improve local nonlinear prediction beyond a near-equal-parameter MLP. | hypothesis · open | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L112>) | — |
| hypothesis:H-114 | H-114 | hypothesis · 118activationfitscomplete_106officiallygraded_recurrentscalingpositive_scheduleinteractionlead | family:optimization | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L113>) | [research/campaign_2026-10-05/ACTIVATION_RESEARCH.md](<../../campaign_2026-10-05/ACTIVATION_RESEARCH.md>) |
| hypothesis:H-115 | Evidence eligibility and archive lineage | hypothesis · proposed | family:evaluation | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L114>) | — |
| hypothesis:H-116 | Refresh public transfer calibration | hypothesis · proposed | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L115>) | — |
| hypothesis:H-117 | Class-by-age headroom and mixture sensitivity | hypothesis · proposed | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L116>) | — |
| hypothesis:H-118 | Parent-preserving residual fitting | hypothesis · proposed | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L117>) | — |
| hypothesis:H-119 | Reproduce completed E415/E418 arithmetic | hypothesis · proposed | family:representation | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L118>) | — |
| hypothesis:H-120 | Causal event definitions and a6 identity | hypothesis · proposed | arm:event-clock | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L119>) | — |
| hypothesis:H-121 | Fixed GELU/LeakyReLU readout transfer | hypothesis · proposed | arm:readout-transfer | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L120>) | — |
| hypothesis:H-122 | Bounded recurrent candidate transfer | hypothesis · proposed | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L121>) | — |
| hypothesis:H-123 | E489 FP32-Muon shape-by-duration | hypothesis · proposed | arm:optimizer-precision | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L122>) | — |
| hypothesis:H-124 | Original-label reanchoring on surviving spans | hypothesis · proposed | arm:reanchor | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L123>) | — |
| hypothesis:H-125 | Clipped pooled-WP fine-tuning | hypothesis · proposed | family:infrastructure | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L124>) | — |
| hypothesis:H-126 | Distillation denominator controls | hypothesis · proposed | arm:denominator | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L125>) | — |
| hypothesis:H-127 | Label-aware selection weights | hypothesis · proposed | family:representation | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L126>) | — |
| hypothesis:H-128 | Equal-target objective weighting | hypothesis · proposed | family:objective | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L127>) | — |
| hypothesis:H-129 | Main/auxiliary/EMA gradient attribution | hypothesis · proposed | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L128>) | — |
| hypothesis:H-130 | E442 teacher overlap and cross-fitting | hypothesis · proposed | arm:teacher-crossfit | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L129>) | — |
| hypothesis:H-131 | Generation3 and bag-as-teacher | hypothesis · proposed | arm:teacher-crossfit | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L130>) | — |
| hypothesis:H-132 | Origin-aligned future8 teacher | hypothesis · proposed | arm:privileged-transformer | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L131>) | — |
| hypothesis:H-133 | Privileged conditional-moment teacher | hypothesis · proposed | arm:privileged-transformer | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L132>) | — |
| hypothesis:H-134 | Current-target direction auxiliary | hypothesis · proposed | arm:supervision-frontier | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L133>) | — |
| hypothesis:H-135 | E432 warmup regime heads/intercepts | hypothesis · proposed | family:infrastructure | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L134>) | — |
| hypothesis:H-136 | E433 target-specific histories | hypothesis · proposed | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L135>) | — |
| hypothesis:H-137 | E434 activity-by-history | hypothesis · proposed | family:infrastructure | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L136>) | — |
| hypothesis:H-138 | E434 event-age-by-history | hypothesis · proposed | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L137>) | — |
| hypothesis:H-139 | E435 i1 conditional innovations | hypothesis · proposed | family:infrastructure | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L138>) | — |
| hypothesis:H-140 | E435 anonymous transition structure | hypothesis · proposed | family:infrastructure | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L139>) | — |
| hypothesis:H-141 | E436 cycle8 then EMA | hypothesis · proposed | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L140>) | — |
| hypothesis:H-142 | E437 nonlinear t1 correction | hypothesis · proposed | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L141>) | — |
| hypothesis:H-143 | E413 history/current/no-volume | hypothesis · proposed | family:infrastructure | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L142>) | — |
| hypothesis:H-144 | E427 volume update gate | hypothesis · proposed | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L143>) | — |
| hypothesis:H-145 | E431 end-to-end branch/U readout | hypothesis · proposed | family:infrastructure | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L144>) | — |
| hypothesis:H-146 | Joint support with continuous escape | hypothesis · proposed | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L145>) | — |
| hypothesis:H-147 | Conditional sign/amplitude posterior | hypothesis · proposed | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L146>) | — |
| hypothesis:H-148 | Learned residual diagonal/LRU memory | hypothesis · proposed | arm:residual-memory | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L147>) | — |
| hypothesis:H-149 | Spline residual readout | hypothesis · proposed | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L148>) | — |
| hypothesis:H-150 | Event/quiet level-jump posterior | hypothesis · proposed | arm:event-quiet-state | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L149>) | — |
| hypothesis:H-151 | Class3 difference/long-age specialists | hypothesis · proposed | arm:event-quiet-state | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L150>) | — |
| hypothesis:H-152 | E444 content-addressed event memory | hypothesis · proposed | arm:residual-memory | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L151>) | — |
| hypothesis:H-153 | Erase/write Delta-rule memory | hypothesis · proposed | arm:residual-memory | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L152>) | — |
| hypothesis:H-154 | E444 stable two-block state | hypothesis · proposed | arm:residual-memory | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L153>) | — |
| hypothesis:H-155 | E430 cadence/mass/credit | hypothesis · proposed | arm:cadence | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L154>) | — |
| hypothesis:H-156 | Random-origin prefix views | hypothesis · proposed | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L155>) | — |
| hypothesis:H-157 | Truncation-boundary jitter | hypothesis · proposed | arm:cadence | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L156>) | — |
| hypothesis:H-158 | E438 exact thinning/view averaging | hypothesis · proposed | family:representation | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L157>) | — |
| hypothesis:H-159 | E445 causal incoming-state refresh | hypothesis · proposed | family:infrastructure | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L158>) | — |
| hypothesis:H-160 | Fixed-packet state-drift response | hypothesis · proposed | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L159>) | — |
| hypothesis:H-161 | E439 late optimizer handoff | hypothesis · proposed | family:infrastructure | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L160>) | — |
| hypothesis:H-162 | E440 ARTBP finite-window credit | hypothesis · proposed | arm:cadence | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L161>) | — |
| hypothesis:H-163 | Weight-decay0.1 modern contrast | hypothesis · proposed | family:infrastructure | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L162>) | — |
| hypothesis:H-164 | Aggressive EMA horizon | hypothesis · proposed | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L163>) | — |
| hypothesis:H-165 | Antithetic sequence ordering | hypothesis · proposed | family:infrastructure | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L164>) | — |
| hypothesis:H-166 | Class-tier auxiliary | hypothesis · proposed | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L165>) | — |
| hypothesis:H-167 | W256/W320/W384 by optimizer | hypothesis · proposed | arm:width | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L166>) | — |
| hypothesis:H-168 | E441 LSTM diversity partner | hypothesis · proposed | family:infrastructure | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L167>) | — |
| hypothesis:H-169 | MLP encoder diversity | hypothesis · proposed | family:unclassified | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L168>) | — |
| hypothesis:H-170 | Two-timescale student | hypothesis · proposed | family:temporal | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L169>) | — |
| hypothesis:H-171 | E443 sentinel neutralization | hypothesis · proposed | family:infrastructure | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L170>) | — |
| hypothesis:H-172 | E443 current-spread auxiliary | hypothesis · proposed | family:infrastructure | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L171>) | — |
| hypothesis:H-173 | Nuisance-coordinate style augmentation | hypothesis · proposed | family:representation | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L172>) | — |
| hypothesis:H-174 | E429 L2 native engine | hypothesis · proposed | arm:deployment | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L173>) | — |
| hypothesis:H-175 | INT8/QAT diversity enabler | hypothesis · proposed | arm:deployment | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L174>) | — |
| hypothesis:H-176 | Exact seed3/seed4 fixed bag | hypothesis · proposed | arm:seed-ensemble | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L175>) | — |
| hypothesis:H-177 | E448 compatible tensor soup | hypothesis · proposed | arm:seed-ensemble | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L176>) | — |
| hypothesis:H-178 | Older finals/all-valid transfer survey | hypothesis · proposed | arm:seed-ensemble | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L177>) | — |
| hypothesis:H-179 | E463/E465 actual TPU recovery | hypothesis · proposed | arm:resource-recovery | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L178>) | — |
| hypothesis:H-180 | Distillation-alpha by stopped-bootstrap horizon interaction in the complete strong recipe | hypothesis · proposed | arm:supervision-frontier | [research/hypotheses.jsonl](<../../hypotheses.jsonl#L179>) | — |
| experiment:E-001 | data forensics | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L1>) | — |
| experiment:E-002 | guard test | experiment | hypothesis:H-002 | [research/experiments.jsonl](<../../experiments.jsonl#L2>) | — |
| experiment:E-003 | baseline | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L3>) | — |
| experiment:E-004 | benchmark | experiment | hypothesis:H-005 | [research/experiments.jsonl](<../../experiments.jsonl#L4>) | — |
| experiment:E-005 | benchmark | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L5>) | — |
| experiment:E-006 | training | experiment | hypothesis:H-004 | [research/experiments.jsonl](<../../experiments.jsonl#L6>) | — |
| experiment:E-007 | training | experiment | hypothesis:H-004 | [research/experiments.jsonl](<../../experiments.jsonl#L7>) | — |
| experiment:E-008 | probe | experiment | hypothesis:H-007 | [research/experiments.jsonl](<../../experiments.jsonl#L8>) | — |
| experiment:E-009 | probe | experiment | hypothesis:H-008 | [research/experiments.jsonl](<../../experiments.jsonl#L9>) | — |
| experiment:E-010 | probe | experiment | hypothesis:H-008 | [research/experiments.jsonl](<../../experiments.jsonl#L10>) | — |
| experiment:E-011 | oracle probe (non-causal) | experiment | hypothesis:H-009 | [research/experiments.jsonl](<../../experiments.jsonl#L11>) | — |
| experiment:E-012 | probe | experiment | hypothesis:H-010 | [research/experiments.jsonl](<../../experiments.jsonl#L12>) | — |
| experiment:E-013 | probe | experiment | hypothesis:H-010 | [research/experiments.jsonl](<../../experiments.jsonl#L13>) | — |
| experiment:E-014 | probe | experiment | hypothesis:H-011 | [research/experiments.jsonl](<../../experiments.jsonl#L14>) | — |
| experiment:E-015 | probe | experiment | hypothesis:H-015 | [research/experiments.jsonl](<../../experiments.jsonl#L15>) | — |
| experiment:E-016 | probe | experiment | hypothesis:H-012 | [research/experiments.jsonl](<../../experiments.jsonl#L16>) | — |
| experiment:E-017 | probe | experiment | hypothesis:H-013 | [research/experiments.jsonl](<../../experiments.jsonl#L17>) | — |
| experiment:E-018 | oracle bound (non-causal) | experiment | hypothesis:H-014 | [research/experiments.jsonl](<../../experiments.jsonl#L18>) | — |
| experiment:E-019 | probe | experiment | hypothesis:H-016 | [research/experiments.jsonl](<../../experiments.jsonl#L19>) | — |
| experiment:E-020 | probe | experiment | hypothesis:H-016 | [research/experiments.jsonl](<../../experiments.jsonl#L20>) | — |
| experiment:E-021 | diagnostic with leaky controls | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L21>) | — |
| experiment:E-022 | benchmark | experiment | hypothesis:H-017 | [research/experiments.jsonl](<../../experiments.jsonl#L22>) | — |
| experiment:E-023 | training A/B | experiment | hypothesis:H-018 | [research/experiments.jsonl](<../../experiments.jsonl#L23>) | — |
| experiment:E-024 | data forensics | experiment | hypothesis:H-019 | [research/experiments.jsonl](<../../experiments.jsonl#L24>) | — |
| experiment:E-025 | probe | experiment | hypothesis:H-013 | [research/experiments.jsonl](<../../experiments.jsonl#L25>) | — |
| experiment:E-026 | note | experiment | hypothesis:H-020 | [research/experiments.jsonl](<../../experiments.jsonl#L26>) | — |
| experiment:E-027 | training A/B | experiment | hypothesis:H-026 | [research/experiments.jsonl](<../../experiments.jsonl#L27>) | — |
| experiment:E-028 | training A/B | experiment | hypothesis:H-027 | [research/experiments.jsonl](<../../experiments.jsonl#L28>) | — |
| experiment:E-029 | diagnostic | experiment | hypothesis:H-024 | [research/experiments.jsonl](<../../experiments.jsonl#L29>) | — |
| experiment:E-030 | training factorial | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L30>) | — |
| experiment:E-031 | model fit | experiment | hypothesis:H-007 | [research/experiments.jsonl](<../../experiments.jsonl#L31>) | — |
| experiment:E-032 | incident | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L32>) | — |
| experiment:E-033 | seed replicate | experiment | hypothesis:H-023 | [research/experiments.jsonl](<../../experiments.jsonl#L33>) | — |
| experiment:E-034 | training | experiment | hypothesis:H-007 | [research/experiments.jsonl](<../../experiments.jsonl#L34>) | — |
| experiment:E-035 | integrity scan | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L35>) | — |
| experiment:E-036 | training | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L36>) | — |
| experiment:E-037 | probe (non-causal) | experiment | hypothesis:H-031 | [research/experiments.jsonl](<../../experiments.jsonl#L37>) | — |
| experiment:E-038 | training A/B | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L38>) | — |
| experiment:E-039 | training A/B | experiment | hypothesis:H-029 | [research/experiments.jsonl](<../../experiments.jsonl#L39>) | — |
| experiment:E-040 | training A/B | experiment | hypothesis:H-033 | [research/experiments.jsonl](<../../experiments.jsonl#L40>) | — |
| experiment:E-041 | benchmark + equivalence | experiment | hypothesis:H-036 | [research/experiments.jsonl](<../../experiments.jsonl#L41>) | — |
| experiment:E-042 | training A/B | experiment | hypothesis:H-033 | [research/experiments.jsonl](<../../experiments.jsonl#L42>) | — |
| experiment:E-043 | equivalence + benchmark | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L43>) | — |
| experiment:E-044 | end-to-end submission test | experiment | hypothesis:H-036 | [research/experiments.jsonl](<../../experiments.jsonl#L44>) | — |
| experiment:E-045 | training A/B | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L45>) | — |
| experiment:E-046 | holdout grading (private metric) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L46>) | — |
| experiment:E-047 | seed replicate | experiment | hypothesis:H-033 | [research/experiments.jsonl](<../../experiments.jsonl#L47>) | — |
| experiment:E-048 | training A/B | experiment | hypothesis:H-029 | [research/experiments.jsonl](<../../experiments.jsonl#L48>) | — |
| experiment:E-049 | independent audit (subagent, read-only) | experiment | hypothesis:H-036 | [research/experiments.jsonl](<../../experiments.jsonl#L49>) | — |
| experiment:E-050 | fixes + re-verification | experiment | hypothesis:H-036 | [research/experiments.jsonl](<../../experiments.jsonl#L50>) | — |
| experiment:E-051 | training A/B | experiment | hypothesis:H-039 | [research/experiments.jsonl](<../../experiments.jsonl#L51>) | — |
| experiment:E-052 | training A/B | experiment | hypothesis:H-037 | [research/experiments.jsonl](<../../experiments.jsonl#L52>) | — |
| experiment:E-053 | fallback training | experiment | hypothesis:H-007 | [research/experiments.jsonl](<../../experiments.jsonl#L53>) | — |
| experiment:E-054 | end-to-end submission test | experiment | hypothesis:H-036 | [research/experiments.jsonl](<../../experiments.jsonl#L54>) | — |
| experiment:E-055 | oracle expert test | experiment | hypothesis:H-022 | [research/experiments.jsonl](<../../experiments.jsonl#L55>) | — |
| experiment:E-056 | training A/B | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L56>) | — |
| experiment:E-057 | training attempt (not evaluated) | experiment | hypothesis:H-038 | [research/experiments.jsonl](<../../experiments.jsonl#L57>) | — |
| experiment:E-058 | training attempt (not evaluated) | experiment | hypothesis:H-038 | [research/experiments.jsonl](<../../experiments.jsonl#L58>) | — |
| experiment:E-059 | training attempt (not evaluated) | experiment | hypothesis:H-038 | [research/experiments.jsonl](<../../experiments.jsonl#L59>) | — |
| experiment:E-060 | metric anatomy | experiment | hypothesis:H-040 | [research/experiments.jsonl](<../../experiments.jsonl#L60>) | — |
| experiment:E-061 | baseline rescoring | experiment | hypothesis:H-041 | [research/experiments.jsonl](<../../experiments.jsonl#L61>) | — |
| experiment:E-062 | label rule search (class 3) | experiment | hypothesis:H-045 | [research/experiments.jsonl](<../../experiments.jsonl#L62>) | — |
| experiment:E-063 | tick reconstruction | experiment | hypothesis:H-042 | [research/experiments.jsonl](<../../experiments.jsonl#L63>) | — |
| experiment:E-064 | target law check all classes | experiment | hypothesis:H-042 | [research/experiments.jsonl](<../../experiments.jsonl#L64>) | — |
| experiment:E-065 | oracle bounds | experiment | hypothesis:H-043 | [research/experiments.jsonl](<../../experiments.jsonl#L65>) | — |
| experiment:E-066 | readout probe | experiment | hypothesis:H-044 | [research/experiments.jsonl](<../../experiments.jsonl#L66>) | — |
| experiment:E-067 | spread decodability | experiment | hypothesis:H-042 | [research/experiments.jsonl](<../../experiments.jsonl#L67>) | — |
| experiment:E-068 | submission verification | experiment | hypothesis:H-041 | [research/experiments.jsonl](<../../experiments.jsonl#L68>) | — |
| experiment:E-069 | candidate rescoring under global WP | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L69>) | — |
| experiment:E-070 | F horizon probe (class 3) | experiment | hypothesis:H-045 | [research/experiments.jsonl](<../../experiments.jsonl#L70>) | — |
| experiment:E-071 | loss bake-off | experiment | hypothesis:H-046 | [research/experiments.jsonl](<../../experiments.jsonl#L71>) | — |
| experiment:E-072 | CE head variants | experiment | hypothesis:H-046 | [research/experiments.jsonl](<../../experiments.jsonl#L72>) | — |
| experiment:E-073 | readout with decoded spread | experiment | hypothesis:H-044 | [research/experiments.jsonl](<../../experiments.jsonl#L73>) | — |
| experiment:E-074 | law validity check classes 0-2 | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L74>) | — |
| experiment:E-075 | aux structure heads + blends | experiment | hypothesis:H-046 | [research/experiments.jsonl](<../../experiments.jsonl#L75>) | — |
| experiment:E-076 | submission verification | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L76>) | — |
| experiment:E-077 | public leaderboard | experiment | hypothesis:H-041 | [research/experiments.jsonl](<../../experiments.jsonl#L77>) | — |
| experiment:E-078 | online-learning ceiling | experiment | hypothesis:H-045 | [research/experiments.jsonl](<../../experiments.jsonl#L78>) | — |
| experiment:E-079 | label definition search (class 3) | experiment | hypothesis:H-045 | [research/experiments.jsonl](<../../experiments.jsonl#L79>) | — |
| experiment:E-080 | online recalibration with rebuilt labels (class 3) | experiment | hypothesis:H-045 | [research/experiments.jsonl](<../../experiments.jsonl#L80>) | — |
| experiment:E-081 | architecture sweep (partial) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L81>) | — |
| experiment:E-082 | class-3 next-move direction probe | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L82>) | — |
| experiment:E-083 | architecture sweep (complete) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L83>) | — |
| experiment:E-084 | paired bootstrap full valid (critic decision rule) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L84>) | — |
| experiment:E-085 | final fits and packages | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L85>) | — |
| experiment:E-086 | public leaderboard | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L86>) | — |
| experiment:E-087 | symreg explanation: scoring mask (stopped at 3 terms) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L87>) | — |
| experiment:E-088 | linear residual-lift probes on heuristic families | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L88>) | — |
| experiment:E-089 | GRU + heuristic library inputs (B1) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L89>) | — |
| experiment:E-090 | volume-code decoding (class 3) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L90>) | — |
| experiment:E-091 | a-column semantics | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L91>) | — |
| experiment:E-092 | nonlinear residual probe (GBM, all heuristics + raw) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L92>) | — |
| experiment:E-093 | GBM ensemble member (heuristics + raw, trained on 3000 train seqs x 3% rows) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L93>) | — |
| experiment:E-094 | adversarial validation train vs valid (V4) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L94>) | — |
| experiment:E-095 | sequence adjacency (V1), independent reproduction of the validation scout | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L95>) | — |
| experiment:E-096 | GBM residual probe, chain-grouped (leak re-check of E-092) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L96>) | — |
| experiment:E-097 | arm decisions, full valid, 4 chain-grouped folds (scripts/decide.py) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L97>) | — |
| experiment:E-098 | true validation masks as training weights (Opus scout M1), chain folds 0-1 | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L98>) | — |
| experiment:E-099 | propensity calibration per class, full valid (Opus scout R1) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L99>) | — |
| experiment:E-100 | symbolic explanation of class-3 direction (chain-grouped) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L100>) | — |
| experiment:E-101 | symbolic direction models per class (chain-grouped) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L101>) | — |
| experiment:E-102 | stacking of passing arms (vs ab24_eps003, chain folds) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L102>) | — |
| experiment:E-103 | label-aware selection model P(s\|x,y) (Opus scout M2 go/no-go) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L103>) | — |
| experiment:E-104 | class-mass-matched loss weights (Opus scout M3 proxy) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L104>) | — |
| experiment:E-105 | tooling: C++ symbolic-regression screening kernel | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L105>) | — |
| experiment:E-106 | visual partial dependence of direction on symbolic terms (marimo, held-out chain folds) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L106>) | — |
| experiment:E-107 | multi-horizon aux heads (2nd/4th next change) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L107>) | — |
| experiment:E-108 | tooling | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L108>) | — |
| experiment:E-109 | per-class offset calibration (visual follow-up) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L109>) | — |
| experiment:E-110 | stress-panel validation proxy (opus-shift) reproduced | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L110>) | — |
| experiment:E-111 | candidate selection on stress panel (lambda 0.05) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L111>) | — |
| experiment:E-112 | price-impact-split trade flow (opus-micro H1), train-fitted tables | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L112>) | — |
| experiment:E-113 | blend weight sweep on stress panel (ab30_eps_aux + baseline x0.25) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L113>) | — |
| experiment:E-114 | visual: per-sequence WP vs train-unlikeness (marimo shift_cell) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L114>) | — |
| experiment:E-115 | diversity search among existing models (stress panel, lam 0.05, 0.7 eps_aux + 0.3 member) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L115>) | — |
| experiment:E-116 | distilled final fit | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L116>) | — |
| experiment:E-117 | symbolic explanation of level S = t0 + t1 (chain-grouped) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L117>) | — |
| experiment:E-118 | online-norm arm properly scored (predict_run fix) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L118>) | — |
| experiment:E-119 | weight soup of 4 same-seed arms (native CPU predict) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L119>) | — |
| experiment:E-120 | raw-view member (our recipe, event features zeroed) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L120>) | — |
| experiment:E-121 | 3-member blend (eps_aux + rawonly + baseline) vs 2-member (stress panel) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L121>) | — |
| experiment:E-122 | scoring-mask explanations per class (symreg, chain-grouped) + single-feature AUCs | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L122>) | — |
| experiment:E-123 | TBPTT chunk length on eps+aux recipe (opus-arch A2) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L123>) | — |
| experiment:E-124 | valid true-mask fine-tune on K-fold GRUs (ab37_ft_f0/f1 vs ab26_tv, held-out chain folds) | experiment · supported (2 folds only) | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L124>) | — |
| experiment:E-125 | diversity member: raw-only plain unweighted MSE GRU (ab38) blended with ab30_eps_aux (stress panel) | experiment · refuted | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L125>) | — |
| experiment:E-126 | where the baseline blend gain lives: 0.7 ab30_eps_aux + 0.3 baseline x0.25 vs ab30, per market class | experiment · observation | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L126>) | — |
| experiment:E-127 | per-class blend weights (GRU vs baseline x0.25), cross-fitted on chain folds 01<->23 | experiment · refuted | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L127>) | — |
| experiment:E-128 | does the valid true-mask fine-tune (E-124) survive the 0.7/0.3 baseline blend? held-out chain folds 0,1 | experiment · supported (2 folds, point estimates) | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L128>) | — |
| experiment:E-129 | baseline blend gain (0.7 ab30 + 0.3 baseline x0.25 minus ab30) by within-class tercile of train-unlikeness (adv score) | experiment · observation (point estimates) | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L129>) | — |
| experiment:E-130 | baseline blend gain by within-class tercile of sequence volatility (mean \|y\| on scored rows, diagnostic only) (haiku r3_a idea 3) | experiment · observation (point estimates) | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L130>) | — |
| experiment:E-131 | optimal GRU blend weight per train-unlikeness tercile (test of E-129 hypothesis) | experiment · supported (weak, flat optimum) | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L131>) | — |
| experiment:E-132 | GRU level and baseline blend gain by within-class tercile of scoring rate (fraction of scored rows per sequence, diagnostic only) | experiment · observation (point estimates) | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L132>) | — |
| experiment:E-133 | why are densely scored sequences hard (E-132)? target structure by scoring-rate tercile | experiment · observation | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L133>) | — |
| experiment:E-134 | valid true-mask fine-tune gain (ab37_ft vs ab26_tv, held-out folds 0/1) by scoring-rate tercile | experiment · observation (2 folds) | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L134>) | — |
| experiment:E-135 | is the dense-scored gap (E-132) a data property? three models by scoring-rate tercile, lam 0 | experiment · observation | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L135>) | — |
| experiment:E-136 | scoring-rate gap (E-132) per market class, ab30_eps_aux lam 0 | experiment · observation | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L136>) | — |
| experiment:E-137 | causal proxy for scoring density: sequence mean of prop_v1 (row-wise causal propensity) vs true scoring rate | experiment · observation | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L137>) | — |
| experiment:E-138 | blend weight gated by causal density proxy (sequence-mean prop_v1 tercile within class), cross-fitted chain folds 01<->23 | experiment · refuted | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L138>) | — |
| experiment:E-139 | metric weight share (sum \|y\| on scored rows) by within-class scoring-rate tercile | experiment · observation | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L139>) | — |
| experiment:E-140 | LB-candidate analog with train-only models: blend[ab30_eps_aux w \| baseline x0.25] vs blend[ab20_gru128_e2 0.5 \| baseline 0.5] (analog of LB-best 0.6569) | experiment · supported | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L140>) | — |
| experiment:E-141 | valid true-mask fine-tune on top of models already trained with true valid masks (ab39_ftm vs ab29_mask, held-out chain folds 0/1) | experiment · weak / not passing | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L141>) | — |
| experiment:E-142 | final candidates on the 368 holdout sequences (never trained on; chain-leaky with dev) | experiment · weak support | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L142>) | — |
| experiment:E-143 | packages verified in python:3.11-slim-bookworm --cpus 1 (valid_subset20) | experiment · verified | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L143>) | — |
| experiment:E-144 | public leaderboard | experiment · measured | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L144>) | — |
| experiment:E-145 | fine-tune dose sweep on true-mask fold models (ab41 vs ab29_mask, held-out chain folds 0/1) | experiment · flat | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L145>) | — |
| experiment:E-146 | 2-seed fine-tuned final + baseline (3 members) vs LB incumbent; runtime | experiment · shelved | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L146>) | — |
| experiment:E-147 | WiSE-FT weight interpolation / task-vector extrapolation between ab29_mask (parent) and ab39_ftm (valid true-mask FT), held-out chain folds 0/1 | experiment · supported | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L147>) | — |
| experiment:E-148 | head-only fine-tune and LP-FT (Kumar 2202.10054) on true-mask fold models | experiment · supported (small) | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L148>) | — |
| experiment:E-149 | i1 lead probe on all 697 class-3 valid sequences (physicist H-P2): partial corr(i1 mid displacement since last i0 move, next i0 move sign \| GRU direction), chain bootstrap | experiment · weak | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L149>) | — |
| experiment:E-150 | scoring-propensity headroom (physicist H-P3): sklearn HistGradientBoosting on 114 heuristics + 112 raw + prop_v1 + class, fit chain folds 0-1 (700k rows), scored folds 2-3 (700k rows) | experiment · headroom confirmed | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L150>) | — |
| experiment:E-151 | rows-to-next-i0-event slices (new tool scripts/slice_compare.py, cache/rte_valid.npz) of the valid fine-tune, held-out chain folds 0/1, lam 0.02 | experiment · observation | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L151>) | — |
| experiment:E-152 | data learning curve, fold 0, full recipe (floor + aux-next + true masks), train sequences 25/50/100% (valid folds 1-3 always in), 2 epochs | experiment · observation | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L152>) | — |
| experiment:E-153 | long training, fold 0, full recipe, 12 epochs one-cycle, wd 0.01 vs 0.1 (grokking / epoch-wise double descent check) | experiment · observation (promising) | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L153>) | — |
| experiment:E-154 | mean-teacher EMA consistency (lambda 1, decay 0.999, EMA weights evaluated) | experiment · observation (promising, unconfirmed) | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L154>) | — |
| experiment:E-155 | prop_v2 PropensityNet (hidden 128, 8 epochs, all other chain folds) as training weights vs prop_v1 | experiment · weak positive | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L155>) | — |
| experiment:E-156 | fine-tune children soup (DiWA-style, 4 seeds from one parent), WiSE of the soup, and dropout 0.1 | experiment · refuted | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L156>) | — |
| experiment:E-157 | baseline member fine-tuned on valid (base_rnn = exact baseline wrapped as a run): head-only (ab48) and GRU cores (ab49); blend 0.65 ab39_ftm + 0.35 member, held-out chain folds | experiment · small | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L157>) | — |
| experiment:E-158 | final-level candidates on the 368 holdout sequences (chain-leaky) vs LB incumbent final_s4ft_base35 | experiment · small | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L158>) | — |
| experiment:E-159 | incident: prop_v2_dev (dev-all, 8 epochs, constant lr 2e-3) saved its last epoch | experiment · fixed | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L159>) | — |
| experiment:E-160 | brainteaser panel: metric/target algebra (research/panel/brainteaser_algebra.md) | experiment · analysis | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L160>) | — |
| experiment:E-161 | brainteaser panel: hidden-assumptions audit (research/panel/brainteaser_audit.md) | experiment · analysis | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L161>) | — |
| experiment:E-162 | brainteaser panel: information and timing (research/panel/brainteaser_timing.md) | experiment · analysis | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L162>) | — |
| experiment:E-163 | public leaderboard | experiment · measured | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L163>) | — |
| experiment:E-164 | EMA-4 confirmation (fold 1) and EMA-4 final + LP-FT package | experiment · supported | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L164>) | — |
| experiment:E-165 | public leaderboard | experiment · measured | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L165>) | — |
| experiment:E-166 | gap-closer check M1: average of two independent 2-epoch fold models (research/panel/gap_closer.md) | experiment · analysis | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L166>) | — |
| experiment:E-167 | gap-closer check M2/M3: mixing EMA-4 final with 2-epoch FT model; EMA per-epoch trend | experiment · analysis | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L167>) | — |
| experiment:E-168 | encoder x cell screen, fold 0, 2-epoch recipe vs ab46_ref_f0 (linear GRU, 0.6724) | experiment · promising | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L168>) | — |
| experiment:E-169 | training throughput: measured P0/P1 fixes with parity (research/perf-2026-09-27.md) | experiment · verified | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L169>) | — |
| experiment:E-170 | prop_v2 propensity weights on the EMA-4 recipe (ab62_p2 vs ab62_base_s3 = ab56/ab58), chain folds 0/1, final epoch | experiment · refuted (EMA recipe) | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L170>) | — |
| experiment:E-171 | LP-FT objectives on EMA-4 fold parents (ab63_*, batch 16, true masks, folds 0/1) | experiment · small, consistent | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L171>) | — |
| experiment:E-172 | final-parent candidates on holdout vs LB-0.6618 package (0.65 ab60_final_ema4_p1_lpft + 0.35 base) | experiment · candidate (low risk, small) | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L172>) | — |
| experiment:E-173 | swap the baseline member for a second EMA-4 GRU (game_changers M5), exact linear-combination WP from moments, held-out chain folds, lam 0.02 | experiment · promising (2 folds) | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L173>) | — |
| experiment:E-174 | TD targets from the EMA teacher (ideas_round4 B1b: alpha 0.5, ramp 0.5 epoch) on EMA-4, folds 0/1, final epoch | experiment · refuted | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L174>) | — |
| experiment:E-175 | native engine: conv encoder port (worktree branch worktree-agent-adf2cdf4134c35cb5, commit aa34ce2, native/CONV_PORT.md) | experiment · ready (unmerged) | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L175>) | — |
| experiment:E-176 | seed ladder + 2-GRU swap with a different-seed partner (ab62_base_s4 vs s3), EMA-4, folds 0/1, lam 0.02 | experiment · supported (2 folds, 2 partners) | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L176>) | — |
| experiment:E-177 | partner scan: 61 (fold 0) / 56 (fold 1) partners for the EMA-4 anchor, exact moment WP, paired chain bootstrap (research/panel/partner_scan.md) | experiment · supported | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L177>) | — |
| experiment:E-178 | 2-GRU final candidates (no baseline): second-seed EMA-4 final ab65 (seed 3, holdout 0.6943, gcorr-LP 0.6948) + the 0.6618 member; holdout vs LB-0.6618 package | experiment · candidate | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L178>) | — |
| experiment:E-179 | retry scout round 5: offline pre-screens in the 2-GRU regime (research/panel/retry_round5.md) | experiment · analysis | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L179>) | — |
| experiment:E-180 | running loss normalizer (brainteaser card B) on EMA-4, folds 0/1, final epoch; partner score | experiment · refuted | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L180>) | — |
| experiment:E-181 | conv1d(k=16) encoder on the EMA-4 recipe, folds 0/1, final epoch; partner score | experiment · refuted (EMA recipe) | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L181>) | — |
| experiment:E-182 | stretch-age weights (ideas_round4 B2: --since-weight 32 --since-floor 0.25) on EMA-4, folds 0/1, final epoch; partner score | experiment · supported | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L182>) | — |
| experiment:E-183 | native engine: online-norm inputs port (worktree branch worktree-agent-a66c73e4b5356739d, commit e743622, native/ONLINE_NORM_PORT.md) | experiment · ready (unmerged) | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L183>) | — |
| experiment:E-184 | since-weight EMA-4 finals (ab67 s4, s3) as a 2-GRU package; holdout vs LB-0.6618 package | experiment · candidate | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L184>) | — |
| experiment:E-185 | completed chunk-500 and online-normalization EMA-4 arms as second GRU members, OOS chain folds 0/1 | experiment · refuted (pair) | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L185>) | — |
| experiment:E-186 | causal conditional-mask residual map, cross-fitted inside OOS chain folds | experiment · refuted | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L186>) | — |
| experiment:E-187 | class-3 target-episode multiplicity diagnostic on 240 validation sequences | experiment · screen_pass | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L187>) | — |
| experiment:E-188 | training-loss versus true scored target-weight mass audit | experiment · analysis | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L188>) | — |
| experiment:E-189 | since-weight EMA-4 robustness across sequence-index octiles, OOF folds 0/1 | experiment · robustness | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L189>) | — |
| experiment:E-190 | since-weight EMA-4 robustness across label-free i0 price-event activity and market class, OOF folds 0/1 | experiment · robustness_with_caveat | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L190>) | — |
| experiment:E-191 | class-3 i1 price-transition decoder-coverage audit and known-only lead probe | experiment · refuted (known-transition displacement) | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L191>) | — |
| experiment:E-192 | since-weight EMA-4 delta by train/valid shift-score tercile (label-free discriminator score, OOF folds 0/1) | experiment · robustness_caveat | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L192>) | — |
| experiment:E-193 | cross-fitted previous-winner event-phase readout, EMA-4 since-weight OOF folds 0/1 | experiment · refuted | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L193>) | — |
| experiment:E-194 | chain-isolation audit of since-weight EMA-4 gain on OOF folds 0/1 | experiment · robustness_with_caveat | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L194>) | — |
| experiment:E-195 | fixed six-epoch EMA+since training, same-seed paired OOF folds 0/1 versus four epochs | experiment · screen_pass | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L195>) | — |
| experiment:E-196 | fixed checkpoint, pure-chain and independent-partner check of six-epoch EMA+since | experiment · promote_fixed_epoch5_final_fit | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L196>) | — |
| experiment:E-197 | EMA-6 since-weight robustness by train/valid shift and diagnostic future distance | experiment · robustness | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L197>) | — |
| experiment:E-198 | final-fit reserved holdout chain-isolation audit | experiment · validation_caveat | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L198>) | — |
| experiment:E-199 | EMA-6 mask-softening stress at lambda 0.05, same-seed OOF folds | experiment · robustness | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L199>) | — |
| experiment:E-200 | exact-clipped pooled pure-chain independent-seed pair lift from EMA-6 second member | experiment · strong_oof_pair_evidence | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L200>) | — |
| experiment:E-201 | mixed final six-epoch seed3 plus old four-epoch seed4 versus old two-seed finals | experiment · candidate_unverified | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L201>) | — |
| experiment:E-202 | one-vCPU scorer-image runtime audit of mixed two-GRU package | experiment · runtime_blocker_under_measured_conditions | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L202>) | — |
| experiment:E-203 | native shared-feature ensemble and SIMD inference optimization | experiment · runtime_candidate | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L203>) | — |
| experiment:E-204 | cheap target-capacity imbalance diagnostic on 831 pure-chain OOF sequences | experiment · cheap_gate_failed | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L204>) | — |
| experiment:E-205 | pure-chain row-group-order heterogeneity of EMA-6 single-model lift | experiment · heterogeneous_positive_overall | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L205>) | — |
| experiment:E-206 | pure-chain row-group-order heterogeneity of EMA-6 independent-seed pair lift | experiment · pair_robustness_with_heterogeneity | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L206>) | — |
| experiment:E-207 | two fixed six-epoch final fits compared with old and mixed final pairs | experiment · final_candidate_uncertain_holdout | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L207>) | — |
| experiment:E-208 | existing auxiliary next-change direction head cross-fit readout | experiment · rejected | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L208>) | — |
| experiment:E-209 | final two-six-epoch shared-native scorer-image package verification | experiment · package_locally_ready | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L209>) | — |
| experiment:E-210 | prior corrected four-epoch pair repackaged with shared-native scorer-image path | experiment · package_locally_ready | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L210>) | — |
| experiment:E-211 | causal warm-up regime correction screen | experiment · rejected | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L211>) | — |
| experiment:E-212 | capped class-3 target-episode training weight, fixed four-epoch fold-0 gate | experiment · rejected_at_fold0 | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L212>) | — |
| experiment:E-213 | causal event-activity gate for mixed independent-seed pair | experiment · rejected_below_gain_gate | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L213>) | — |
| experiment:E-214 | i1 volume-code flow shock event-level falsifier | experiment · rejected_at_event_gate | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L214>) | — |
| experiment:E-215 | i0 own-book volume-code refill/depletion analogue event-level falsifier | experiment · rejected_at_event_gate | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L215>) | — |
| experiment:E-216 | class-stratified six-epoch pair diagnostic on pure-chain OOF folds | experiment · diagnostic_no_routing | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L216>) | — |
| experiment:E-217 | repeat one-vCPU scorer-image runtime stress for three shared-native final ZIPs | experiment · runtime_risk_mixed | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L217>) | — |
| experiment:E-218 | independent second-seed replication of fixed six-epoch EMA+since pair | experiment · positive_but_below_promotion_gate | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L218>) | — |
| experiment:E-219 | mixed shared-native pair idle one-vCPU scorer-image timing | experiment · idle_runtime_pass_with_machine_caveat | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L219>) | — |
| experiment:E-220 | validation-period repeat-two four-epoch EMA+since fold-0 score and champion-aware gate | experiment · rejected_at_paired_champion_gate | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L220>) | — |
| experiment:E-221 | exploratory pure-chain single-member diagnostic of validation repetition | experiment · exploratory_uncertain | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L221>) | — |
| experiment:E-222 | metric-weighted EMA teacher consistency fold-0 screen | experiment · rejected_at_parent_and_pair_gate | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L222>) | — |
| experiment:E-223 | user-reported public leaderboard calibration and standings | experiment · measured_public_feedback_user_reported | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L223>) | — |
| experiment:E-224 | five-submission reserved-holdout versus measured-public calibration audit | experiment · calibration_warning_with_rank_consistency | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L224>) | — |
| experiment:E-225 | reserved-holdout class heterogeneity of mixed six/four versus submitted corrected pair | experiment · exploratory_class_heterogeneity | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L225>) | — |
| experiment:E-226 | one-layer EMA+since GRU fold-0 strong-member screen | experiment · rejected_at_paired_gate | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L226>) | — |
| experiment:E-227 | user-reported public feedback for mixed six/four shared package | experiment · measured_public_feedback_user_reported | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L227>) | — |
| experiment:E-228 | direct final-pair exact-mask and class-mix audit | experiment · no_candidate_promoted | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L228>) | — |
| experiment:E-229 | temporary dual-GRU v2 scorer-image benchmark | experiment · runtime_gate_failed | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L229>) | — |
| experiment:E-230 | class-3 four-event target-law score influence audit | experiment · diagnostic_no_score_gain | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L230>) | — |
| experiment:E-231 | anonymized-feature domain-geometry screen | experiment · screen_only | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L231>) | — |
| experiment:E-232 | repeat existing auxiliary next-change logit cross-fitted residual readout | experiment · negative_existing_head | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L232>) | — |
| experiment:E-233 | class-3 winner-tier training auxiliary, matched fold-0 pair gate | experiment · rejected_at_predeclared_pair_gate | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L233>) | — |
| experiment:E-234 | causal class-3 spread-constraint projection falsifier | experiment · rejected_exact_row_screen | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L234>) | — |
| experiment:E-235 | cross-fitted three-feature causal price-history residual correction | experiment · rejected_full_fold_gate | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L235>) | — |
| experiment:E-236 | one-feature past-eight-event drift residual correction | experiment · rejected_full_fold_gate | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L236>) | — |
| experiment:E-237 | scoring-mask BCE auxiliary matched fold-0 pair gate | experiment · rejected_at_predeclared_pair_gate | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L237>) | — |
| experiment:E-238 | out-of-fold direction versus magnitude oracle and event-clock residual anatomy | experiment · diagnostic | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L238>) | — |
| experiment:E-239 | event-residual ARMA, variance clustering, and spectral analysis | experiment · diagnostic_no_score_gain | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L239>) | — |
| experiment:E-240 | observable six-book-event return cycle and seasonal residual readout | experiment · rejected_readout_screen | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L240>) | — |
| experiment:E-241 | joint sign and magnitude two-head GRU with soft versus strict gate | experiment · rejected_at_predeclared_pair_gate | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L241>) | — |
| experiment:E-242 | seasonal price-event duration forecasting and residual gate | experiment · diagnostic_signal_below_pair_gate | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L242>) | — |
| experiment:E-243 | scorer-weighted residual peak length, full GARCH diagnostic, and VECM screen | experiment · diagnostic_no_promoted_model | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L243>) | — |
| experiment:E-244 | training-only current target-contrast direction auxiliary matched fold-0 pair gate | experiment · rejected_at_predeclared_pair_gate | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L244>) | — |
| experiment:E-245 | unused current-sign logit readout on pure fold 0 | experiment · diagnostic_no_pair_gain | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L245>) | — |
| experiment:E-246 | causal raw-price-code cycle-break residual screen | experiment · rejected_residual_screen | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L246>) | — |
| experiment:E-247 | classwise noncausal oracle global-score allocation | experiment · diagnostic_priority_class3_direction | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L247>) | — |
| experiment:E-248 | event-shift cross-correlation of class-3 target and pair output | experiment · diagnostic_future_information_oracle | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L248>) | — |
| experiment:E-249 | scored-row next-book-event information oracle | experiment · diagnostic_future_information_oracle | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L249>) | — |
| experiment:E-250 | opposite-fold causal next-price-event side screen | experiment · rejected_one_fold_causal_screen | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L250>) | — |
| experiment:E-251 | preregistered training-only class-3 future quote-path auxiliary | experiment · rejected_at_predeclared_pair_gate | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L251>) | — |
| experiment:E-252 | class-stratified 80 versus 100 percent validation training chains at fixed optimizer steps | experiment · rejected_at_predeclared_pair_gate | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L252>) | — |
| experiment:E-253 | class-3 Rao-Blackwell future-quote Markov kernel CPU screen | experiment · rejected_at_bias_gate | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L253>) | — |
| experiment:E-254 | pre-registration: capacity and optimizer arms in EMA-6 since regime (plan peppy-twirling-wall Phase 2) | experiment · preregistered | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L254>) | — |
| experiment:E-255 | deployment: three-level native build, all members on every CPU path, runtime governor (plan Phase 0) | experiment · implemented | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L255>) | — |
| experiment:E-256 | Phase 2 arm A: weight decay 0.1 on EMA-6 since recipe, fold 0, fixed epoch 5 (pre-registered E-254) | experiment · rejected_at_fold0_gate | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L256>) | — |
| experiment:E-257 | native: 4-row blocked AVX2+FMA gemv | experiment · implemented | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L257>) | — |
| experiment:E-258 | K-member ensemble scan, all members two-stage gcorr LP-FT from fixed epochs, 831 pure-chain sequences folds 0-1, exact clipped global WP | experiment · supported | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L258>) | — |
| experiment:E-259 | brainteaser: why fold scan (+) and final holdout (0/-) disagree on diversity ensembles | experiment · supported | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L259>) | — |
| experiment:E-260 | brainteaser checks + upload pre-registration (goal: new public best over 0.6622) | experiment · preregistered | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L260>) | — |
| experiment:E-261 | instrument: holdout proxies calibrated on 10 known public scores (scripts/lb_calibration.py) | experiment · preregistered_upload | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L261>) | — |
| experiment:E-262 | h192 EMA-6 since model as a replacement ensemble partner, fold 0 | experiment · rejected_at_fold0_gate | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L262>) | — |
| experiment:E-263 | submission candidate screen and package runtime-governor audit | experiment · screened_package_readiness | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L263>) | — |
| experiment:E-264 | MLP encoder as replacement partner in EMA-6 since ensemble, fold 0 | experiment · rejected_at_fold0_gate | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L264>) | — |
| experiment:E-265 | Phase 2 LR and batch-64 arm fold 0 resource check | experiment · resource_failed_retry_after_queue | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L265>) | — |
| experiment:E-266 | live public leaderboard and submission window check | experiment · context_verified | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L266>) | — |
| experiment:E-267 | user-reported submission results and quota screenshot | experiment · measured_public_feedback_user_reported | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L267>) | — |
| experiment:E-268 | EMA8 scout run status audit | experiment · incomplete_no_score | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L268>) | — |
| experiment:E-269 | merged teammate model added to the exact user-reported public incumbent pair; OOF folds 0-1 and merged-weight scan | experiment · rejected_for_public_pair | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L269>) | — |
| experiment:E-270 | clean fold0 frozen since/since rank8 paired hidden-state residual screen | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L270>) | — |
| experiment:E-271 | post-reboot bounded resource and scheduler audit | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L271>) | — |
| experiment:E-272 | TRAIN-only current-code geometry and tail-copula falsification | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L272>) | — |
| experiment:E-273 | all112 causal feature relations and direct event coordinate | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L273>) | — |
| experiment:E-274 | feature-only shared inverse-clock constrained falsifier | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L274>) | — |
| experiment:E-275 | independent event-memory long-sequence numeric falsifier | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L275>) | — |
| experiment:E-276 | amended event-memory independent numeric reproduction | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L276>) | — |
| experiment:E-277 | completed clean fourfold since/since reference panel | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L277>) | — |
| experiment:E-278 | synthetic GPU architecture arithmetic and cost falsifier | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L278>) | — |
| experiment:E-279 | independent event-memory fresh TRAIN feature exposure proof | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L279>) | — |
| experiment:E-280 | component-held-out input-only feature-relation falsifier | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L280>) | — |
| experiment:E-281 | input-only orderbook copula/source-family falsifier | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L281>) | — |
| experiment:E-282 | input-only causal event-rate and quiet-publication falsifier | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L282>) | — |
| experiment:E-283 | input-only joint-endpoint conditional clock-order probe | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L283>) | — |
| experiment:E-284 | independent optimized event-memory CPU numerical reproduction | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L284>) | — |
| experiment:E-285 | independent raw input clock-order reproduction | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L285>) | — |
| experiment:E-286 | independent raw event-rate relation reproduction | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L286>) | — |
| experiment:E-287 | fold0 current-known spread supervision screen (frozen, completed after resume) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L287>) | — |
| experiment:E-288 | CPU probe: cross-fitted per-class output multipliers under pooled Global WP (reference since pair, 4 clean folds) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L288>) | — |
| experiment:E-289 | fold0 distilled GRU256 since student (alpha .5, fold teachers s3+s4 @ep3), head-only vs two-stage GCLP FT | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L289>) | — |
| experiment:E-290 | output-space mix of GCLP-FT members with their own pre-FT since parents (4 members, equal) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L290>) | — |
| experiment:E-291 | N2 pooled-memory GCLP FT (decay .995) + head-only vs two-stage FT + shared-trunk parent/head-FT output mix (folds 0+1; holdout with final ab67 members) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L291>) | — |
| experiment:E-292 | cross-fitted convex weights over [head-FT s3, head-FT s4, parent s3, parent s4] (folds 0/1, each fitted on the other) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L292>) | — |
| experiment:E-293 | N1 ensemble-aware joint head FT vs matched solo extra head FT (since pair, folds 0+1) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L293>) | — |
| experiment:E-294 | holdout (368 seq) screen of shared-trunk merged-head bags with existing final trunks: since pair (SP2), + chunk-500 s5, + online-norm s5 | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L294>) | — |
| experiment:E-295 | CPU probe: causal per-sequence EMA smoothing of the SP2-style 4-output average (folds 0+1) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L295>) | — |
| experiment:E-296 | SP2 package build and verification (own Rust int16 engine, merged heads, governor off) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L296>) | — |
| experiment:E-297 | data/eval: class and chain composition of valid vs clean four-fold reference score | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L297>) | — |
| experiment:E-298 | data/eval: class mix needed to reproduce public level | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L298>) | — |
| experiment:E-299 | data/eval: nested public calibration of subset/class-reweighted proxies | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L299>) | — |
| experiment:E-300 | tool: scripts/de_public_proxy.py | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L300>) | — |
| experiment:E-301 | data/eval: implied public class mix (level match of class-reweighted pooled WP) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L301>) | — |
| experiment:E-302 | data/eval: TRAIN target laws H1-H4 (fit 240 / held-out 160 sequences, component-disjoint) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L302>) | — |
| experiment:E-303 | data/eval: current-state vs future share of metric variance; structural S/D output bound (fold 0, since s3) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L303>) | — |
| experiment:E-304 | engine | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L304>) | — |
| experiment:E-305 | fold0 runtime-feasible d256 combos with merged heads (S3+D, S4+D, SP2+D vs SP2) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L305>) | — |
| experiment:E-306 | class-routed head mixes (retrospective market class as router; upper bound of a causal detector), folds 0+1 | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L306>) | — |
| experiment:E-307 | PREREGISTERED folds 2+3 confirmation of parent+head output mixes and class-routed RSP2 | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L307>) | — |
| experiment:E-308 | holdout: SP3/SP4 seed bag (new final since seeds 5, 6) and routed RSP2/RSP4 | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L308>) | — |
| experiment:E-309 | extra heads on shared since trunks (folds 0+1): head-bag FT seeds 5/6, distilled head (alpha .5, fold teachers), train+valid gcorr head | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L309>) | — |
| experiment:E-310 | fp32 packages SP2_f32 and RSP2_f32 (routed: warm-up aux-class detector, groups stepped through warm-up, one group after row 99; governor off) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L310>) | — |
| experiment:E-311 | distilled GRU256 since student, fold 1 (scored after cutoff from the finished 05:35 job) + folds 0+1 under the routed framework | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L311>) | — |
| experiment:E-312 | creative move 1 (scout: pooled-metric between/within sequence): causal running-mean de-meaning/amplification p' = p - lam*mean_{99..t}(p), lam per class, cross-fitted over 4 clean folds | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L312>) | — |
| experiment:E-313 | creative: residual forensics of clean OOF since pair (cr_surprise subagent) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L313>) | — |
| experiment:E-314 | creative: 12 reframing cards (cr_reframer subagent), 8 probed | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L314>) | — |
| experiment:E-315 | creative: winning-Kaggle mindset scout (5 linked writeups/notebooks + order-book competitions) and 5 OOF probes | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L315>) | — |
| experiment:E-316 | S4_D package: since s4 trunk (parent+gcorr head-FT merged) + final distilled GRU256 trunk (parent+gcorr head-FT merged), equal 0.5, int16, governor off | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L316>) | — |
| experiment:E-317 | class-3 matured-label residual feedback (cr_mature): determination lag, true vs feature-reconstructed labels, future-swap leak control | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L317>) | — |
| experiment:E-318 | upload variants: routed R4D (classes 0-2 -> FT s4 + d256 merged; class 3 -> s4 merged + d256 merged), D_only, FT4_D on 4 clean folds and holdout | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L318>) | — |
| experiment:E-319 | upload set prepared (not uploaded): R4D_i16, S4D_i16, D256_i16, RSP2_f32 — all packaging gates pass | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L319>) | — |
| experiment:E-320 | distillation strength alpha 0.7 vs 0.5 for the GRU256 since student (fold 0) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L320>) | — |
| experiment:E-321 | class-3 queue-race decoding from in-run volume/trade history (cr_queue2) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L321>) | — |
| experiment:E-322 | causal magnitude correction by class x rows-since-last-move bin (24 scales), cross-fitted over 4 folds on the R4D OOF analog | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L322>) | — |
| experiment:E-323 | explanatory residual analysis of the R4D OOF analog (notebook F1-F7) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L323>) | — |
| experiment:E-324 | public leaderboard results of the 2026-10-02 upload set (user-uploaded, 00:24-00:25) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L324>) | — |
| experiment:E-325 | infra incident: campaign GPU cap (40,000 s) exhausted 01:39 | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L325>) | — |
| experiment:E-326 | CPU screen: alpha diversity pair on fold 0 (screen research/campaign_2026-10-02/packages/alpha_pair_f0.json) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L326>) | — |
| experiment:E-327 | CPU diagnostic (H-061): do distillation teachers memorise their fitted rows? fold-0 teachers, valid rows fitted (folds 1-3 eval groups) vs held out (fold 0), scored mask, per class | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L327>) | — |
| experiment:E-328 | fold-1 confirmation of distillation alpha 0.7 vs 0.5 (c20261002_d256a07_f1r; screen research/campaign_2026-10-02/packages/alpha_f1.json) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L328>) | — |
| experiment:E-329 | alpha curve, fold 0: alpha 0.85 student (c20261001_d256a085_f0; screen night_f0.json) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L329>) | — |
| experiment:E-330 | H-060 teacher ensemble (since s3+s4 + fold-bound alpha-0.7 student, thirds), alpha 0.7, fold 0 (c20261002_te07_f0) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L330>) | — |
| experiment:E-331 | H-059 noise-adaptive distillation a(u)=0.5..0.9, tau 32, fold 0 (c20261002_nad_f0; screen nad_f0.json) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L331>) | — |
| experiment:E-332 | H-062 weight soup of fixed-epoch EMA checkpoints, fold 0 (c20261002_soup_f0) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L332>) | — |
| experiment:E-333 | final alpha-0.7 students (train + dev valid, s3 and s4) on the 368-sequence holdout vs the 0.6641-public alpha-0.5 student (finals_holdout.json) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L333>) | — |
| experiment:E-334 | DTW / piecewise-constant idea round: 4 CPU scouts (readout, labels, streams, objective); research/campaign_2026-10-02/dtw/SUMMARY.md | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L334>) | — |
| experiment:E-335 | H-063 class-routed head mixing on one trunk (class 3 0.75 parent, classes 0-2 0.75 head-FT) vs 0.5/0.5 merged heads | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L335>) | — |
| experiment:E-336 | engine: q16x precision (int16 weights x int16 activation codes, _mm256_madd_epi16, exact int32 flush) for GRU/LSTM matvecs | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L336>) | — |
| experiment:E-337 | H-060 teacher-ensemble student, fold-1 confirmation (c20261002_te07_f1; screen research/campaign_2026-10-02/packages/te_f01.json) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L337>) | — |
| experiment:E-338 | scheduling: stopped-bootstrap fold-0 screen c20261002_sbd_f0 stopped after 81 s to free the GPU for the final H-060 student (barrier archived as STOP_NEW_JOBS.archived_0954_sbd_swap) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L338>) | — |
| experiment:E-339 | final teacher-ensemble student c20261002_final_te07_s3 (teachers ab67_final_since_s3/s4 + c20261002_final_d256a07_s3, alpha 0.7) on the 368-sequence holdout + package | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L339>) | — |
| experiment:E-340 | public leaderboard: alpha-0.7 distilled GRU256 students (user-uploaded 10:44-10:45) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L340>) | — |
| experiment:E-341 | holdout seed-bag check of the final students (bag_holdout.json) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L341>) | — |
| experiment:E-342 | holdout: head choice inside a 2-student pair (bag_heads_holdout.json) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L342>) | — |
| experiment:E-343 | local LB probe: all 12 public submissions re-scored on the 368-seq holdout (research/lb/lb_probe.py, registry.json; holdout preds archived in research/lb/preds/) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L343>) | — |
| experiment:E-344 | residual analysis notebook notebooks/residuals_students_20261002.py (fold 0 OOF; plots research/campaign_2026-10-02/plots/R1-R4) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L344>) | — |
| experiment:E-345 | H-064 distilled LSTM192 student as bag partner, folds 0+1 (dlstm_f01.json) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L345>) | — |
| experiment:E-346 | H-065 TCN student (C128 L12, distilled alpha 0.5), fold 0 (dtcn_f0.json) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L346>) | — |
| experiment:E-347 | H-065 TCN student (C128 L12, alpha 0.5), folds 0+1 (dtcn_f01.json) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L347>) | — |
| experiment:E-348 | Muon (per gate block on recurrent matrices, optim.py) GRU256 student, fold 0, epoch-2 screen vs control at epoch 2 (muon_f0.json) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L348>) | — |
| experiment:E-349 | Muon student: fold-1 ep2 screen and fold-0 full schedule (muon_check.json, muon_full_f0.json) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L349>) | — |
| experiment:E-350 | C1 student learning curve + seeds + width, fold 0 (scores/c1.json) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L350>) | — |
| experiment:E-351 | C3 since-weight x distillation alpha 2x2, folds 0+1, seed 3 (scores/c3.json) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L351>) | — |
| experiment:E-352 | Muon full schedule, folds 0+1 (scores/muonfull.json) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L352>) | — |
| experiment:E-353 | stopped-bootstrap target (alpha 0.5, h 8), folds 0+1, parent+headFT (scores/sbd.json) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L353>) | — |
| experiment:E-354 | ep2 screens: TBPTT chunk 1000 and chrono-64 init, folds 0+1 (scores/ep2.json) | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L354>) | — |
| experiment:E-355 | twoclock student ep2 screen fold 0; teacher-ensemble seed-4 final on holdout | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L355>) | — |
| experiment:E-356 | seed-3 Muon + chunk1000 + stopped-bootstrap full-schedule parents, folds 0/1 | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L356>) | — |
| experiment:E-357 | two-timescale student epoch-2 matched controls, folds 0/1 | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L357>) | — |
| experiment:E-358 | Codex independent EDA continuation cohorts1+2 | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L358>) | — |
| experiment:E-359 | fullschedule combo two seeds matched controls folds0+1 | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L359>) | — |
| experiment:E-360 | independent CPU strongstudent t1 causaloffset falsifier | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L360>) | — |
| experiment:E-361 | Codex puzzle continuations and corrected unapplied optimizer prototype | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L361>) | — |
| experiment:E-362 | round2 objective estimator and independent coefficient review | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L362>) | — |
| experiment:E-363 | round2 future path laws and round3 independent raw decoder audit | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L363>) | — |
| experiment:E-364 | round2 strong GRU history usage diagnostic | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L364>) | — |
| experiment:E-365 | round2 blind specification counterexamples and CPU streaming equivalence | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L365>) | — |
| experiment:E-366 | round3 hybrid exit-law falsification | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L366>) | — |
| experiment:E-367 | width320 fixed ep3 twofold confirmation | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L367>) | — |
| experiment:E-368 | round3 raw-consistent strong GRU history replay | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L368>) | — |
| experiment:E-369 | round3 input-observation coordinate identifiability witness | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L369>) | — |
| experiment:E-370 | round3 prefix physical event teacher rejection gate | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L370>) | — |
| experiment:E-371 | round3 exact gradient state capture synthetic instrument | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L371>) | — |
| experiment:E-372 | round3 independent sampled history arithmetic review | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L372>) | — |
| experiment:E-373 | round3 exact engineered suffix washout retention | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L373>) | — |
| experiment:E-374 | round3 anonymous observable stopping simulator | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L374>) | — |
| experiment:E-375 | round4 blind actual update-state specification fixtures | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L375>) | — |
| experiment:E-376 | round4 samefold teacher and propensity recorded membership audit | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L376>) | — |
| experiment:E-377 | round4 independent exact suffix washout reproduction | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L377>) | — |
| experiment:E-378 | round4 finite local state forgetting instrument | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L378>) | — |
| experiment:E-379 | round4 independent observable simulator reproduction | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L379>) | — |
| experiment:E-380 | round4 isolated sourcebound online capture prototype | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L380>) | — |
| experiment:E-381 | combo fixed fold23 confirmation gate | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L381>) | — |
| experiment:E-382 | round4 independent v1 capture rejection | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L382>) | — |
| experiment:E-383 | round4 peer innovation frozen proxy falsifier | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L383>) | — |
| experiment:E-384 | round4 independent peer innovation rejection reproduction | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L384>) | — |
| experiment:E-385 | round4 independent v2 capture typed RNG rejection | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L385>) | — |
| experiment:E-386 | independent combo fold23 gate and metadata review | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L386>) | — |
| experiment:E-387 | round5 preregistered loss geometry instrument | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L387>) | — |
| experiment:E-388 | round5 two input-only observable support tests | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L388>) | — |
| experiment:E-389 | round4 final v2 typed moment amendment and exact E385 recovery | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L389>) | — |
| experiment:E-390 | round4 isolated v3 exact RNG and moment correction | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L390>) | — |
| experiment:E-391 | round4 independent v3 private RNG normalization rejection | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L391>) | — |
| experiment:E-392 | round5 actual tiny GRU common macrostep credit control | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L392>) | — |
| experiment:E-393 | round5 independent loss geometry instrument rejection | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L393>) | — |
| experiment:E-394 | v4_builder | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L394>) | — |
| experiment:E-395 | v4_independent_builder_review | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L395>) | — |
| experiment:E-396 | completed_capture_independent_review | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L396>) | — |
| experiment:E-397 | false_timeout_2112 | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L397>) | — |
| experiment:E-398 | R5_mechanism_instrument_v2 | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L398>) | — |
| experiment:E-399 | R6_state_drift_builder | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L399>) | — |
| experiment:E-400 | R6_state_drift_blind_critic | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L400>) | — |
| experiment:E-401 | R6_state_drift_final_review | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L401>) | — |
| experiment:E-402 | R6_amp_linearity_audit | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L402>) | — |
| experiment:E-403 | final reviewed warm state-drift probe | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L403>) | — |
| experiment:E-404 | final descriptive AMP result review | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L404>) | — |
| experiment:E-405 | user-stop consolidation and existing submission packaging | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L405>) | — |
| experiment:E-406 | user-reported public results and archive identity binding | experiment | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L406>) | — |
| experiment:E-407 | private Kaggle TPU inventory and compatibility | experiment · completed_hardware_framework_probe | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L407>) | — |
| experiment:E-408 | local and private dual-T4 training compatibility | experiment · completed | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L408>) | — |
| experiment:E-409 | actual D256/W384 native precision and CPU runtime probe | experiment · completed | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L409>) | — |
| experiment:E-410 | fixed complete-predictor combo head comparison | experiment · completed | hypothesis:H-065 | [research/experiments.jsonl](<../../experiments.jsonl#L410>) | — |
| experiment:E-411 | fixed complete-predictor weight quantization degradation pilot | experiment · completed | hypothesis:H-068 | [research/experiments.jsonl](<../../experiments.jsonl#L411>) | — |
| experiment:E-412 | full-data fixed combo seed3 and final head | experiment · public_confirmed | hypothesis:H-065 | [research/experiments.jsonl](<../../experiments.jsonl#L412>) | — |
| experiment:E-413 | private dual-T4 volume-history versus equal-width current-input control | experiment · completed | hypothesis:H-078 | [research/experiments.jsonl](<../../experiments.jsonl#L413>) | — |
| experiment:E-414 | actual mixed-fit distilled normalizer calibration and matched predictive controls | experiment · calibration_completed | hypothesis:H-073 | [research/experiments.jsonl](<../../experiments.jsonl#L414>) | — |
| experiment:E-415 | fixed frozen-state conditional-cue proxy screen | experiment · completed | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L415>) | — |
| experiment:E-416 | native byte-weight predictor parity and runtime | experiment · completed | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L416>) | — |
| experiment:E-417 | current-student state reset and matched attenuation panel | experiment · completed | hypothesis:H-099 | [research/experiments.jsonl](<../../experiments.jsonl#L417>) | — |
| experiment:E-418 | literal conditional feature family screens | experiment · completed | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L418>) | — |
| experiment:E-419 | matched QAT and training-precision continuations | experiment · completed_matched_onefold_screen | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L419>) | — |
| experiment:E-420 | native factorized W384/r128 zero-update model parity and runtime | experiment · real_native_parity_and_clean_timing_completed | hypothesis:H-070 | [research/experiments.jsonl](<../../experiments.jsonl#L420>) | — |
| experiment:E-421 | modern frozen-trunk factorized categorical branch/U readout | experiment · frozen_head_screens_completed | hypothesis:H-090 | [research/experiments.jsonl](<../../experiments.jsonl#L421>) | — |
| experiment:E-422 | mature-episode sequence gain representation probe | experiment · reciprocal_gain_screen_completed | hypothesis:H-095 | [research/experiments.jsonl](<../../experiments.jsonl#L422>) | — |
| experiment:E-423 | EMA retention in rows from full schedule | experiment · completed_no_improvement_signal | hypothesis:H-071 | [research/experiments.jsonl](<../../experiments.jsonl#L423>) | — |
| experiment:E-424 | H093/H094 production-objective prefix specialists and low-rank adapters | experiment · completed_eight_fits_common_fp32_paired_screen | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L424>) | — |
| experiment:E-425 | soft-mixture prefix specialists | experiment · completed_fixed_fit_export_and_paired_score_no_supported_gain | hypothesis:H-106 | [research/experiments.jsonl](<../../experiments.jsonl#L425>) | — |
| experiment:E-426 | event-clock consistency-teacher EMA | experiment · completed_three_matched_fits_and_paired_screen_no_event_clock_gain | hypothesis:H-107 | [research/experiments.jsonl](<../../experiments.jsonl#L426>) | — |
| experiment:E-427 | causal volume-gated recurrent updates | experiment · paused_source_only_GPU_smokes_and_fits_not_started | hypothesis:H-108 | [research/experiments.jsonl](<../../experiments.jsonl#L427>) | — |
| experiment:E-428 | synthetic end-to-end mature-gain sensitivity ladder | experiment · complete | hypothesis:H-095 | [research/experiments.jsonl](<../../experiments.jsonl#L428>) | — |
| experiment:E-429 | L2-resident inference engine build and payload runtime gate | experiment · planned_cpu_engine_work | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L429>) | — |
| experiment:E-430 | chunk optimizer-cadence and EMA-horizon controls | experiment · three_arm_implementation_drafts_and_synthetic_discriminator_complete | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L430>) | — |
| experiment:E-431 | current-combo end-to-end branch and U readout | experiment · planned_implementation_required | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L431>) | — |
| experiment:E-432 | regime-conditioned readout and class-three head | experiment · planned_implementation_required | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L432>) | — |
| experiment:E-433 | target-specific row/event history | experiment · planned_implementation_required | hypothesis:H-084 | [research/experiments.jsonl](<../../experiments.jsonl#L433>) | — |
| experiment:E-434 | activity-conditioned and event-age memory interactions | experiment · planned_implementation_required | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L434>) | — |
| experiment:E-435 | cross-instrument innovation and statistical transition structure | experiment · planned_implementation_required | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L435>) | — |
| experiment:E-436 | cycle smoothing versus row EMA | experiment · planned_implementation_required | hypothesis:H-089 | [research/experiments.jsonl](<../../experiments.jsonl#L436>) | — |
| experiment:E-437 | strong-combo nonlinear t1 offset calibration | experiment · planned_cpu_analysis | hypothesis:H-083 | [research/experiments.jsonl](<../../experiments.jsonl#L437>) | — |
| experiment:E-438 | R11 label-exact thinning and training-view augmentation | experiment · planned_implementation_required | hypothesis:H-080 | [research/experiments.jsonl](<../../experiments.jsonl#L438>) | — |
| experiment:E-439 | late Muon handoff and module clipping | experiment · planned_implementation_required | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L439>) | — |
| experiment:E-440 | ARTBP finite-window credit with linear optimizer | experiment · planned_implementation_required | hypothesis:H-100 | [research/experiments.jsonl](<../../experiments.jsonl#L440>) | — |
| experiment:E-441 | LSTM architecture-diversity pair | experiment · planned_implementation_required | hypothesis:H-102 | [research/experiments.jsonl](<../../experiments.jsonl#L441>) | — |
| experiment:E-442 | teacher overlap diagnosis and generation-three student | experiment · planned_cpu_diagnosis_then_fit | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L442>) | — |
| experiment:E-443 | current-spread auxiliary and sentinel handling | experiment · planned_implementation_required | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L443>) | — |
| experiment:E-444 | content-addressed memory and structured two-block state | experiment · planned_implementation_required | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L444>) | — |
| experiment:E-445 | incoming recurrent state refresh versus equal-work sham | experiment · planned_implementation_required | hypothesis:H-104 | [research/experiments.jsonl](<../../experiments.jsonl#L445>) | — |
| experiment:E-446 | common-increment coordinate and component-overlap audit | experiment · planned_cpu_audit | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L446>) | — |
| experiment:E-450 | current combo seed4 full-data replication and fixed equal bag | experiment · public_confirmed_seed4_0.6660; fixed_equal_bag_unsubmitted | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L447>) | — |
| experiment:E-457 | fixed complete combo parent/head public-transfer endpoints | experiment · public_confirmed_fixed_parent_and_head_endpoints | hypothesis:H-065 | [research/experiments.jsonl](<../../experiments.jsonl#L448>) | — |
| experiment:E-458 | private Kaggle CPU JAX recurrent implementation parity | experiment · complete | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L449>) | — |
| experiment:E-459 | whole supplied-validation cached fixed-predictor scorecard | experiment · complete | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L450>) | — |
| experiment:E-460 | actual public i16 callback over all supplied validation | experiment · complete | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L451>) | — |
| experiment:E-461 | JAX full-objective bridge CPU implementation checks | experiment · complete | hypothesis:H-098 | [research/experiments.jsonl](<../../experiments.jsonl#L452>) | — |
| experiment:E-449 | W384 legacy fold1 replication and planned optimizer-width factorial | experiment · v7_actual_dual_probes_passed_paired_fits_stopped_on_host_RAM_guard_no_complete_checkpoints_or_scores | hypothesis:H-066 | [research/experiments.jsonl](<../../experiments.jsonl#L453>) | — |
| experiment:E-462 | Fixed exact public-combo INT8 recurrent compression and packaged runtime | experiment · original_q8_zips_withdrawn; python311_repair_compatibility_passed_runtime_held | arm:deployment | [research/experiments.jsonl](<../../experiments.jsonl#L454>) | — |
| experiment:E-463 | E-463 | experiment · TPU_recovery_private_kaggle_v1_pushed | arm:resource-recovery | [research/experiments.jsonl](<../../experiments.jsonl#L455>), [research/experiments.jsonl](<../../experiments.jsonl#L490>) | — |
| experiment:E-464 | E-464 | experiment · W384_serial_recovery_private_kaggle_v1_pushed | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L456>), [research/experiments.jsonl](<../../experiments.jsonl#L489>) | — |
| experiment:E-465 | All-eight-TPU full-fold FP32/BF16 AdamW precision continuation | experiment · actual_FP32_JAX_CPU_AdamW7460_reference_pass_TPU_fullfit_not_started | hypothesis:H-110 | [research/experiments.jsonl](<../../experiments.jsonl#L457>) | — |
| experiment:E-466 | Raw target-mechanics and strong-GRU residual diagnostic | experiment · complete_exposed_descriptive_CPU_only | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L458>) | — |
| experiment:E-467 | Linear memory, recurrent attention, and KAN-inspired spline components | experiment · complete_ten_fixed_GPU_fits_and_seven_paired_component_comparisons | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L459>) | — |
| experiment:E-468 | Paired codebook likelihood and metric-relevant readouts with anchored residual | experiment · complete_five_frozen_GPU_heads_and_all_ten_between_arm_comparisons | hypothesis:H-111 | [research/experiments.jsonl](<../../experiments.jsonl#L460>) | — |
| experiment:E-469 | Fixed two-new-seed replication of all ten recurrent/linear-memory/attention/spline arms | experiment · complete_all_twenty_fixed_seed_replication_fits | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L461>) | — |
| experiment:E-470 | Fixed8pointwiseactivation×2AdamW/Muon16GPUfits | experiment · complete_all16GPUfits_officialscore_reproduction_and30pairedcontrasts | hypothesis:H-114 | [research/experiments.jsonl](<../../experiments.jsonl#L462>) | — |
| experiment:E-472 | Twoadditionalfixedseedsforall16activationoptimizerrecipes | experiment · complete_all32replicationfits_48totalofficialscores_90conditionalcontrasts | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L463>) | — |
| experiment:E-471 | BoundedGRUcandidatecurvature×AdamW/Muon | experiment · complete_all6_commonFP32_scientific_fits_7primary_2backenddiagnostics | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L464>) | — |
| experiment:E-473 | FixedGELUcurvature_slope_scale_approximationcontrols×optimizer | experiment · complete_all10GPUfits_14officialpairedcontrasts | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L465>) | — |
| experiment:E-475 | Allsixboundedrecurrentactivationoptimizerrecipesreplicatedattwonewseeds | experiment · complete_all12replications_18totalrecurrentfits_officialscores_21contrasts | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L466>) | — |
| experiment:E-474 | LocalizedGELUsineversuslinearslopeembeddingandheadplacement | experiment · complete12fits_12officialscores_14jointcontrasts | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L467>) | — |
| experiment:E-476 | ActivationoptimizerLRshape_meanareafactorial | experiment · complete12fits_12officialscores_20paired2jointinteractioncontrasts | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L468>) | — |
| experiment:E-477 | Parallelmixedreadout_head_dropoutfactorial | experiment · complete12fits_12officialscores_14primary4secondarycontrasts | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L469>) | — |
| experiment:E-478 | Scale_positiveboundedGRUcandidateMuontoH64two-layerincumbenttopology | experiment · complete6fits_6officialscores_3pairedcontrasts | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L470>) | — |
| experiment:E-479 | Learned_per_layer_bounded_GRU_candidate_shape_x_optimizer | experiment · complete16fullfits_independently_graded_and_raw_exports_reproduced | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L471>) | — |
| experiment:E-480 | Activation_optimizer_LR_schedule_two_seed_replication | experiment · complete24fullfits_independently_graded_and_raw_exports_reproduced | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L472>) | — |
| experiment:E-481 | Bounded_candidate_sine_versus_matched_central_slope_endpoint_cubic | experiment · complete18fullfits18independentgrades30contrasts_rawexportreproduced | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L473>) | — |
| experiment:E-482 | Kaggle_T4_candidate_recipe_portability_replication | experiment · six_remote_fits_independently_graded_all10contrasts_rawexport_reproduced | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L474>) | — |
| experiment:E-483 | Isolated_recurrent_Muon_momentum_x_bounded_candidate_shape | experiment · eighteen_fullfits_independently_graded_all27contrasts_rawexport_reproduced | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L475>) | — |
| experiment:E-484 | Causal_context_conditioned_GELU_LeakyReLU_readout_mixture | experiment · all12_independently_graded_and_raw_exports_reproduced | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L476>) | — |
| experiment:E-485 | Bounded_recurrent_activation_shape_x_training_duration_H128 | experiment · probe_failed_native_update_peak_before_full_fits | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L477>) | — |
| experiment:E-486 | CPU_frozen_checkpoint_training_shape_x_inference_shape_counterfactual | experiment · cpu_cross_complete_all36cells_all54contrasts_independently_reproduced | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L478>) | — |
| experiment:E-487 | Fit_only_Muon_orthogonalizer_precision_cross_gradient_source_diagnostic | experiment · actual_CUDA_diagnostic_complete_independent_all9delta_cells_verified_BF16fails_FP32passes | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L479>) | — |
| experiment:E-488 | Native_GRU_context_mixture_gate_x_training_duration_AdamW | experiment · complete_all12_continuous_fits_24snapshots_independent_official_grades_all27global54target2000draw_arithmetic_reproduced | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L480>) | — |
| experiment:E-489 | E-489 | experiment · recovered_v2_probe_failed_clipping_coverage_no_fullfits | arm:optimizer-precision | [research/experiments.jsonl](<../../experiments.jsonl#L481>), [research/experiments.jsonl](<../../experiments.jsonl#L484>), [research/experiments.jsonl](<../../experiments.jsonl#L485>) | — |
| experiment:E-490 | continuation_contract_and_review_coverage | experiment · registered_no_scientific_fits | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L482>) | — |
| experiment:E-491 | actual_continuation_runner_cpu_fixture | experiment · complete | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L483>) | — |
| experiment:E-492 | E-492 | experiment · independent_raw_counts_reproduced | arm:event-clock | [research/experiments.jsonl](<../../experiments.jsonl#L486>), [research/experiments.jsonl](<../../experiments.jsonl#L487>), [research/experiments.jsonl](<../../experiments.jsonl#L499>), [research/experiments.jsonl](<../../experiments.jsonl#L501>), [research/experiments.jsonl](<../../experiments.jsonl#L504>), [research/experiments.jsonl](<../../experiments.jsonl#L505>) | — |
| experiment:E-493 | E-493 | experiment · paired_control_fullshape_resource_probe_admitted | arm:readout-transfer | [research/experiments.jsonl](<../../experiments.jsonl#L488>) | — |
| experiment:E-500 | W384_serial_Kaggle_packet_revision_v3r1 | experiment · private_packet_independent_GO_controller_admission_review_required_not_pushed | arm:width | [research/experiments.jsonl](<../../experiments.jsonl#L491>), [research/experiments.jsonl](<../../experiments.jsonl#L492>), [research/experiments.jsonl](<../../experiments.jsonl#L498>) | — |
| experiment:E-495 | fresh strong-recipe cadence and recurrent-credit comparison | experiment · preparation_review_go_unsealed_resource_admission_pending | arm:cadence | [research/experiments.jsonl](<../../experiments.jsonl#L493>) | — |
| experiment:E-497 | causal residual external matrix memory prototype | experiment · isolated_prototype_review_go_trainer_integration_resource_probe_pending | arm:residual-memory | [research/experiments.jsonl](<../../experiments.jsonl#L494>) | — |
| experiment:E-496 | E-496 | experiment · calibration_interrupted_no_report | arm:supervision-frontier | [research/experiments.jsonl](<../../experiments.jsonl#L495>), [research/experiments.jsonl](<../../experiments.jsonl#L503>), [research/experiments.jsonl](<../../experiments.jsonl#L506>) | — |
| experiment:E-498 | prepared_dependency_queue_inventory | experiment · prepared_not_launched | family:unclassified | [research/experiments.jsonl](<../../experiments.jsonl#L496>) | — |
| experiment:E-499 | E-499 | experiment · finished | arm:optimizer-precision | [research/experiments.jsonl](<../../experiments.jsonl#L497>), [research/experiments.jsonl](<../../experiments.jsonl#L500>), [research/experiments.jsonl](<../../experiments.jsonl#L502>) | — |
| legacy:target_stopped_bootstrap | target_stopped_bootstrap | legacy_arm · historical_needs_refresh | root | [research/bandit/arms.json](<../../bandit/arms.json>) | — |
| legacy:arch_lstm_student | arch_lstm_student | legacy_arm · historical_needs_refresh | root | [research/bandit/arms.json](<../../bandit/arms.json>) | — |
| legacy:arch_tcn_student | arch_tcn_student | legacy_arm · historical_needs_refresh | root | [research/bandit/arms.json](<../../bandit/arms.json>) | — |
| legacy:arch_two_timescale_state | arch_two_timescale_state | legacy_arm · historical_needs_refresh | root | [research/bandit/arms.json](<../../bandit/arms.json>) | — |
| legacy:teacher_ensemble | teacher_ensemble | legacy_arm · historical_needs_refresh | root | [research/bandit/arms.json](<../../bandit/arms.json>) | — |
| legacy:distill_alpha | distill_alpha | legacy_arm · historical_needs_refresh | root | [research/bandit/arms.json](<../../bandit/arms.json>) | — |
| legacy:data_all_valid | data_all_valid | legacy_arm · historical_needs_refresh | root | [research/bandit/arms.json](<../../bandit/arms.json>) | — |
| legacy:seed_bag_engine | seed_bag_engine | legacy_arm · historical_needs_refresh | root | [research/bandit/arms.json](<../../bandit/arms.json>) | — |
| legacy:input_representation | input_representation | legacy_arm · historical_needs_refresh | root | [research/bandit/arms.json](<../../bandit/arms.json>) | — |
| legacy:distill_rb_weight | distill_rb_weight | legacy_arm · historical_needs_refresh | root | [research/bandit/arms.json](<../../bandit/arms.json>) | — |
| legacy:distill_patient | distill_patient | legacy_arm · historical_needs_refresh | root | [research/bandit/arms.json](<../../bandit/arms.json>) | — |
| legacy:noise_reg_long | noise_reg_long | legacy_arm · historical_needs_refresh | root | [research/bandit/arms.json](<../../bandit/arms.json>) | — |
| legacy:muon_rnn | muon_rnn | legacy_arm · historical_needs_refresh | root | [research/bandit/arms.json](<../../bandit/arms.json>) | — |
| legacy:lstm_fbias_pair | lstm_fbias_pair | legacy_arm · historical_needs_refresh | root | [research/bandit/arms.json](<../../bandit/arms.json>) | — |
| legacy:student_chunk1000 | student_chunk1000 | legacy_arm · historical_needs_refresh | root | [research/bandit/arms.json](<../../bandit/arms.json>) | — |
| legacy:init_chrono | init_chrono | legacy_arm · historical_needs_refresh | root | [research/bandit/arms.json](<../../bandit/arms.json>) | — |
| legacy:student_width320 | student_width320 | legacy_arm · historical_needs_refresh | root | [research/bandit/arms.json](<../../bandit/arms.json>) | — |
| legacy:student_wsd6_ema | student_wsd6_ema | legacy_arm · historical_needs_refresh | root | [research/bandit/arms.json](<../../bandit/arms.json>) | — |

## Journal events

- [research/bandit_tree/journal.jsonl](<../journal.jsonl#L1>) `sha256:86c391e7152980c27e942282646d0cfd2928f5eca18f933f2c759277b49bc0d4` — `{"actor": {"model": "controller with Luna/max builder and independent auditor", "tool": "Codex"}, "compute_runs_launched": 0, "decision": "Build an additive bandit-tree navigation layer; preserve all original bytes and ledger events. Historical priors are archival, future hypotheses can deepen with evidence.", "event_type": "migration_contract", "evidence_refs": ["research/bandit_tree/ORIGINALS.json", "research/ideas/library_ideation_footprint_2026-10-06.json", "research/campaign_2026-10-06/branch_teacher_audit/REPORT.md"], "id": "TREE-20261006-001", "node_id": "root", "timestamp_utc": "2026-10-06T17:46:10.156049+00:00"}`
- [research/bandit_tree/journal.jsonl](<../journal.jsonl#L2>) `sha256:422b6bdc5e6e4db01edd270f4963a5b21c8331f6761adbf406b0e9218998e9b3` — `{"decision": "Exact raw-row, filesystem-coverage, link and original-anchor checks passed on initial generated snapshot; rebuild after final source edits. Latest quant-probes unavailable; independent teacher/student/blend brief saved. No training launched.", "event_type": "preservation_verification", "evidence_refs": ["research/bandit_tree/PRESERVATION.json"], "id": "TREE-20261006-002", "node_id": "root", "timestamp_utc": "2026-10-06T18:06:56Z"}`
- [research/bandit_tree/journal.jsonl](<../journal.jsonl#L3>) `sha256:caaa6a5a50f6a959e92500c2503370e6fd00c523d500f0f0333be8e846842100` — `{"code_merged_from_friend": false, "compute_runs_launched": 0, "decision": "User refreshed origin/quant-probes to b7fd5d2f. Archive completed singlefold attention result (+0.00005 vs GRU256; fifth-member+0.00028) without universal ceiling inference. Add convergence/context, causally learnable privileged transfer, and complementarity/gated-blend child protocols. Prior access failures retained as historical. No expected public gain inferred.", "event_type": "remote_evidence_update", "evidence_refs": ["research/campaign_2026-10-06/branch_teacher_audit/post_fetch_b7fd5d2f/manifest.json", "research/campaign_2026-10-06/branch_teacher_audit/post_fetch_b7fd5d2f/OBSERVATIONS.md"], "id": "TREE-20261006-003", "new_node_ids": ["arm:teacher-convergence-context", "arm:learnable-privileged-transfer", "arm:complementarity-gated-blend"], "node_id": "arm:causal-transformer-teacher", "timestamp_utc": "2026-10-06T18:58:28Z"}`
- [research/bandit_tree/journal.jsonl](<../journal.jsonl#L4>) `sha256:3075bc77284f6ff09fe27de0d489e7afd1107b53cd19c70901f63ea6173efd9a` — `{"actor_model": "Luna/max independent scout; root source audit", "code_merged_from_friend": false, "compute_runs_launched": 0, "decision": "Refreshed creative-iteration branch records user-reportedpublic0.6674 for heterogeneous-distilled GRU256 studentpair. Prioritize independent heterogeneous compression; retain attentionchildren. Source-specific warmstart,twoepoch alpha.7,fourfamilyauxhead .1 andunnormalized .8125 teacher scale require matchedcontrols. External result doesnotreplace rootbest0.6660 or become expectedpublicdelta.", "event_type": "external_public_evidence_and_frontier_update", "evidence_refs": ["research/campaign_2026-10-06/branch_teacher_audit/post_fetch_833aaf54/manifest.json", "research/campaign_2026-10-06/branch_teacher_audit/post_fetch_833aaf54/iterations/2026-10-05-wave20/public-result.json", "research/bandit_tree/HETEROGENEOUS_AGENT_BRIEF.md"], "id": "TREE-20261006-004", "new_node_ids": ["arm:heterogeneous-teacher-compression", "arm:two-stream-teacher-compression", "arm:distilled-function-preserving-widening"], "node_id": "arm:heterogeneous-teacher-compression", "timestamp_utc": "2026-10-06T19:04:52Z"}`
- [research/bandit_tree/journal.jsonl](<../journal.jsonl#L5>) `sha256:31ee8426dae83159dbcae9120503ec69b5f316029f0d8953657b4584bebbb981` — `{"actor": {"external_models": "Freebuff GLM, OpenCode free Muse Spark/MiMo after access check", "native_subagents": "gpt-6-luna/max", "tool": "Codex"}, "compute_runs_launched": 0, "decision": "Begin independent heterogeneous-distillation continuation plus diverse literature arms; replace old local cutoff/caps with explicit user instruction. First24h allocation; then reassess, continue until explicit stop with graceful drain. Unknown numeric beliefs remain null.", "event_type": "campaign_authorization_and_resource_update", "evidence_refs": ["research/campaign_2026-10-06/bandit_continuation/CONTRACT.md"], "id": "TREE-20261006-005", "node_id": "root", "old_resources": {"aggregate_cpu_cores_max": 5, "aggregate_ram_bytes_max": 40000000000, "aggregate_vram_bytes_max": 3000000000, "new_experiment_compute": "Kaggle GPU/TPU only; current access and resource admission required", "new_local_experiments_after_2026_10_06_1000_Kyiv": false}, "resources": {"aggregate_cpu_cores_max": 10, "aggregate_ram_bytes_max": 20000000000, "aggregate_vram_bytes_max": 5000000000, "authority": "research/campaign_2026-10-06/bandit_continuation/CONTRACT.md", "first_review_after_hours": 24, "local_admission_cutoff": null, "local_allocator_initial_bytes": 4300000000, "local_gpu_workers_max": 1, "memory_high_bytes": 18000000000, "new_experiment_compute": "Local and Kaggle; separate cloud headroom and live quota", "stop_policy": "Explicit user stop closes admission and drains healthy jobs; resource emergencies stop offending jobs"}, "timestamp_utc": "2026-10-06T19:59:28.944259+00:00"}`

## Link and artifact warnings

No unresolved configured links or artifacts were found.

## Config snapshot fields

The parsed config is retained in `tree.json` metadata and the exact config bytes are archived in the linked snapshot.
