> Extracted from the final October 12 handoff. Its original SHA-256 was `e033f8d52f4d4444e5f0c5fa006c2b4f56b6e82f7187e2ed9fb1b74b20e6f704`. See [archive index](../README.md) and [split manifest](../SPLIT_MANIFEST.json). Local links were rebased for this page.

## Execution, incidents, and resource controls

No new rental, top-up, or billing change was authorized in this archive.
Vast lane B's last evidence showed zero credit/no owned instance, with later
read-only access unavailable; lack of a refreshed account view is not proof of
remote teardown. Kaggle runs require phase-level receipts, actual mounted input
hashes, worker resource checks, terminal output collection and quota
reconciliation. RUNNING alone does not prove a worker phase.

The strongest current support admission target is 17 decimal GB aggregate
working memory, below the existing 18 GB high and 20 GB hard caps; swap stays
zero and aggregate CPU stays five cores. When an active worker is included,
the memory arithmetic is parent current plus the worker's unused reserved
headroom plus the new bounded task. See
[ADMISSION_MEMORY_MARGIN_20261012_R1.json](../../../../campaign_2026-10-12/ADMISSION_MEMORY_MARGIN_20261012_R1.json)
and [KILL_PREVENTION_PLAN_20261012.md](../../../../campaign_2026-10-12/KILL_PREVENTION_PLAN_20261012.md).

The first failed metadata scan treated NPY magic bytes as version/length
fields, then attempted an unbounded read and hit its 650 MiB scope. The
follow-up command also failed and its exact source/command mismatch is
preserved; it is not evidence that a second full metadata scan ran. A later
attempt staged a 1.69 GB cache into tmpfs and was OOM-killed. The source,
commands, exits and cgroup diagnoses are preserved. No model or scientific
comparison completed in those incidents.
Correct readers verify NPY magic and version, cap header length (existing
bound below 64 KiB), account for each payload plus header, and budget dtype
conversion, masks, labels, gradients, library workspaces and transients.
Large frozen artifacts stay on disk or are memory-mapped; never retry a
malformed reader by raising its cap. The fixed F0 blend scorer completed at
1,402,112 KiB under a 1.5 GB limit, with little headroom. Reuse its result;
an equivalent future analysis needs approximately 1.9 GB or a streaming/smaller
panel design and fresh aggregate admission.

Kaggle R4/R5/R6 export repairs failed before valid worker scoring due to mount,
host-admission, timeout-boundary, or source-identity issues. Their preserved
packets are execution evidence rather than negative model results. R7 terminal receipts classify a pre-fit literal-placeholder argv failure, not OOM or a model result. Earlier validators failed to exercise that exact mounted worker command. Future separately authorized preflight must materialize and smoke-check final argv and reject unresolved placeholders. The complete terminal collection is preserved; no retry is authorized. The user-issued
STOP supersedes older open allocations and “keep compute busy” instructions
for new work; healthy admitted runs drain normally.

## Completed drain and durable preservation

Local seed4: terminal fixedCP5/export/officialgrade complete. Kaggle R7: terminal pre-fit ERROR, complete collection, no scientific result or retry. All delegated roles completed; no active bandit units remain. Global admission barrier has an empty drain set, genericqueue STOP remains, and the exact research cron is removed. All historical open allocations are superseded; resume requires explicit user instruction.

Export validator R1 expected nonexistent `export.run_name`; R2 derives the run from frozen worker argv and passes on actual artifacts. Valid computation was not repeated. A preliminary NPZ shape probe decompressed the y member without inspecting/using label values; export itself remained feature-only, and official grading intentionally read labels. The validator's translated ru_maxrss reading conflicts with its configured500MB scope and is retained as unresolved measurement evidence, not a trusted peak. P1 parent-hierarchical max increased7 with no OOM/kill and no attributed cause; P2/P3/export max/OOM/kill remained0. High/max counters count events, not bytes. These caveats are preserved in the final handoff.

All useful project temporary artifacts are now on durable `/home` btrfs: 5,569 regular files /1,545,120,715 bytes, including native builds, fixtures, source scripts and logs. Every inventoried file was checksum-verified before its original was replaced by a compatibility symlink. No research payload is retained solely in tmpfs. Unrelated browser/system/auth/session caches remain untouched. Private model/data/binary payloads stay Git-ignored; the safe recovered text, source and preservation inventories are committed. See [temporary inventory](../../../../campaign_2026-10-12/research_archive_inventory_r1/TMP_PRESERVATION_INVENTORY_R1.json), [copy receipt](../../../../campaign_2026-10-12/research_archive_inventory_r1/TMP_DURABLE_COPY_RECEIPT_R1.json), [migration receipt](../../../../campaign_2026-10-12/research_archive_inventory_r1/TMP_DURABLE_MIGRATION_RECEIPT_R1.json), [native blob receipt](../../../../campaign_2026-10-12/research_archive_inventory_r1/TMP_NATIVE_BINARY_PRESERVATION_R1.json), and [recovered text manifest](../../../../campaign_2026-10-12/recovered_tmp_notes_r1/RECOVERED_NOTES_MANIFEST_R1.json).

The user-reported public best remains0.6671 for C. For planning, approximately0.667 is the incumbent reference; a forecast above0.670 is not established by the two exposed F0 controls. D/E remain frozen, unscored human-upload trials. Metadata ridge did not outperform its chronological recent-five baseline. No new fit, blend search, quantization benchmark, cohort or paid action follows this archive.
