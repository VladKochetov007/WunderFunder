> Extracted from the final October 12 handoff. Its original SHA-256 was `e033f8d52f4d4444e5f0c5fa006c2b4f56b6e82f7187e2ed9fb1b74b20e6f704`. See [archive index](../README.md) and [split manifest](../SPLIT_MANIFEST.json). Local links were rebased for this page.

# Challenge 2 research handoff — 2026-10-12

**Archive state: FINAL — 2026-10-12T16:12:41.588603+00:00.** Both admitted logical runs are terminal and all three delegated roles are finished. The clean-F0 seed4 control completed its fixed CP5 export and official grade; Kaggle R7 failed before fitting and was completely collected. Source integration HEAD before final appendix regeneration: `99455217227e531c50dee5fcedf63184f8f07770`. This archive records completed, failed, uncertain, and untested work; it authorizes no new research. Research admission is closed and the research wakeup cron is disabled.

The governing record is
[STOP_NEW_RESEARCH_AND_FINAL_ARCHIVE_20261012_R1.json](../../../../campaign_2026-10-12/STOP_NEW_RESEARCH_AND_FINAL_ARCHIVE_20261012_R1.json).
New research admission is closed. The only work permitted by that record is
draining (a) the already admitted clean-F0 seed-4 student continuation through
its fixed CP5 export and official grade and (b) already admitted Kaggle R7
terminal collection/grading. R7 subsequently ended ERROR at14:38:59Z and was completely collected: 114 files/1,363,683 bytes, all hashes and sizes independently checked. Its profile exited2 in0.291s because the frozen worker argv retained literal `{python}`; no fit, checkpoint, export, prediction or score exists. GPUbytes0. The stale pipeline running/empty-stages receipt is not evidence of a fit. See [terminal triage](../../../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261012/r7_terminal_triage_r1/TERMINAL_TRIAGE_R1.json). This is infrastructure failure, not a scientific negative. No retry under the user stop. The local seed4 logical run is now complete; its final receipt and grade are recorded below.
Final drain must distinguish successful, partial, and failed terminal
outcomes and preserve originals. Do not infer missing scores.

The final archive contains 479 hypothesis-ledger rows, 798 experiment-ledger rows, 158 configured research-tree nodes across 11 families, and 135 journal events. The ledgers are append-only and can contain repeated IDs for revisions,
retries, or additional events; row count is not the number of unique hypotheses
or experiments. The current experiment ledger includes the preregistered fixed
half-mixture screen; the journal also includes an archive bookkeeping event,
which is not a scientific result. Full source rows, source-line links, and
hashes are preserved in the [dated ledger snapshots](../ledger-snapshot/README.md).
The metadata-only document catalog is split into browseable groups in the
[evidence catalog](../evidence-catalog/README.md). The research-tree navigation is rebuilt after final appendix generation; its integrity check is a point-in-time snapshot, and mutable read-only monitoring may later change catalog freshness without changing scientific evidence. See
[the Main status inventory](../../../../campaign_2026-10-12/research_archive_inventory_r1/MAIN_STATUS_INVENTORY_R1.json),
SHA-256 `7395df8a06ff7a53767301d642e7012806516d34d226b953a33e6198e0a820de`.
It is a point-in-time, metadata-only inventory from the earlier archive
preparation snapshot at `62378c04`; it is not the post-cleanup Main
inventory. It records status/path counts and file-size classes; it omits
credential-like, provider-raw, large-JSON path details and all file contents.
It is an aid for root's explicit-file cleanup, not authority to stage or
delete. The cleanup commit b607ca0e tracked useful documentation/evidence
files and removed fourteen weight-bearing ZIPs from Git tracking without
deleting their local Main copies or rewriting history. The current source
snapshot also includes later worker/stdout and skill-link updates. Main
status has since changed; refresh inventory only as part of root's final drain.

## Research objective and decision boundary

The active goal remains a valid deterministic Challenge 2 submission that can
reach top-1 and public score above 0.700. Those objectives have not been met.
The strongest available public observation in this archive is the user-reported
0.6671 for candidate C. No numerical public forecast is supported. Existing
local folds, public rows, and submission packages have been repeatedly inspected
and are exposed development evidence. None is a fresh holdout.

Choose complete endpoints using source-qualified local evaluation across
folds and distinct fitting seeds, with the exact served package checked for
metric, precision, determinism, size, and runtime. Public feedback confirms
transfer; it does not replace local comparison, and a local delta is never
added to a public score. A component with lower standalone quality can still
contribute conditionally to a fixed mixture, but that contribution must be
measured on the declared evaluation units. A negative result closes only the
exact tested recipe or combination at that evidence level.

Project-wide binding rules and rationale are in
[the research objective](../../../../../specs/research-objective.md),
[research methods](../../../../RESEARCH_METHODS.md),
[ADR0031 cross-fold top-tier release](../../../../../specs/adr/0031-cross-fold-top-tier-submission-release.md),
[ADR0030 local-evidence selection](../../../../../specs/adr/0030-local-evidence-selects-submissions.md),
[ADR0021 producer/consumer validation](../../../../../specs/adr/0021-validate-consumers-against-producer-artifacts.md),
[ADR0023 faithful recipe reconstruction](../../../../../specs/adr/0023-faithful-method-reconstruction-before-variants.md),
and [current state](../../../../../specs/current-state.md). Teammate methods may be read
and independently implemented; teammate weights and copied source code are
not part of our submission lineage.
