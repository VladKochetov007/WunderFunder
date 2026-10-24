---
kind: review
status: complete
created_utc: 2026-10-24T14:16:11Z
updated_utc: 2026-10-24T14:16:11Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-24/resume4h_r1/publication_source_r1/LINK_ADAPTATION_R1.json
  - research/campaign_2026-10-24/resume4h_r1/publication_source_r1/reference_snapshots/README.md
  - research/campaign_2026-10-24/resume4h_r1/publication_source_r1/URL_QUERY_REDACTION_R2.json
  - research/campaign_2026-10-24/resume4h_r1/publication_source_r1/SYNC_RECEIPT_R1.json
---

# Publication link adaptation review — R2

## Question or claim

Do the publication-only link adaptations resolve the 62 missing dependencies while preserving the original reports, and does the final synced metadata pass URL redaction?

## Method

Read-only checks on publication worktree /home/vlad/development/wunder/research_tmp/resume4h-publication-20261024 at research-team base d32be666a10c6928a100a91a021337225520ce32. I validated the adaptation manifest against the local execution-base Git objects, mapped publication links, local snapshot files and sanitizer receipt. I did not request GitHub pages, execute code, read model or label payloads, or alter the PR worktree.

## Evidence and interpretation

The adaptation manifest resolves all 62 targets: 46 as immutable GitHub blob links pinned to execution base c3e208b387ead0ec967f9abc4b14275eb8ba94c5, and 16 as exact local snapshots. The 46 referenced Git blobs exist and match each manifest byte count and SHA-256. The 16 snapshot destinations exist and match their source byte counts and SHA-256 values.

Thirty publication Markdown files contain 166 mapped link occurrences covering those 62 unique targets. All 137 immutable URL occurrences have the expected repository, full commit and source path, with no query strings; the one section anchor matches its source heading. All 29 local snapshot occurrences resolve to the mapped snapshot file. I inverted the link replacements in each changed Markdown file: every result matches its recorded original SHA-256, and each publication file matches its recorded publication SHA-256.

The snapshot index states that its two Python files, age_readout_portable_identity.py and age_readout_portable_validate.py, are documentary source snapshots, not canonical implementations or a runnable environment. The source manifest says canonical src/ paths were not modified. The publication diff has no src/ paths and git diff --check is clean.

The adaptation hashes describe the publication files immediately before link replacement; they are not hashes of the separate source checkout’s whole unified README files. The publication branch intentionally ports selected navigation hunks onto research-team. Root’s later PR-refresh/index additions are separate integration edits; port those selectively without replacing research-team’s existing README content.

The final URL_QUERY_REDACTION_R2 receipt records 440 text files scanned, zero changed files, zero redacted values needed, and zero remaining credential query values. I independently scanned 463 publication text files: 41 raw.githubusercontent.com URL occurrences remained, with zero nonplaceholder sensitive query values. The sanitizer script and R2 receipt copies match between source and publication trees. R2 supersedes the earlier R1 receipt after final sync.

## Limits and validity

The immutable blob URLs were structurally checked against the local execution-base Git objects; no HTTP request was made. This confirms exact commit/path/blob identity, not live GitHub availability. No source snapshots were executed.

## Decision and next step

Pass for the 62-target link adaptation and current URL sanitization. No link-resolution or source-hash blocker remains in this bounded review. Root should finish the already-planned selected index additions, add this review to campaign navigation, and rerun the sanitizer after any later sync that restores metadata copies.

## Source records

- [Link adaptation manifest](publication_source_r1/LINK_ADAPTATION_R1.json) and [snapshot index](publication_source_r1/reference_snapshots/README.md).
- [URL redaction R2 receipt](publication_source_r1/URL_QUERY_REDACTION_R2.json), [sanitizer](publication_source_r1/sanitize_publication_urls.py), and [campaign sync receipt](publication_source_r1/SYNC_RECEIPT_R1.json).
