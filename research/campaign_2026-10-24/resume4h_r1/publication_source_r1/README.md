# Current-root trainer source snapshot

This folder preserves the exact three-file uncommitted trainer diff from the
current root execution checkout. The source checkout is
experiment/next-cycle-recovery-20261021 at
c3e208b387ead0ec967f9abc4b14275eb8ba94c5; its publication target is
origin/research-team at the separately recorded base SHA in
SOURCE_FILE_HASHES_R1.json. The publication worktree does not overwrite or
import current-root trainer files.

CURRENT_ROOT_TRAINER_PATCH_R1.patch is the byte-exact git diff --binary HEAD
for the three listed paths; the JSON manifest records the patch hash, each
source file SHA-256 and each source-base Git blob ID. The provenance file adds
the inherited age-portability dependency closure. The other two changes are the
current resume's identity/import and selected-membership producer-schema
repairs. This classifies the source changes without implying that they all
belong in the current campaign PR.

The source branch and publication base have diverged. The patch is relative to
c3e208b, so do not apply it blindly to research-team; root should compare
the exact base blobs and port only any required reviewed repair. No model code
was copied into this packet, and no tests or model/data imports were run.

## Campaign copy timing

The campaign was copied while root was still publishing new outputs. The
machine-readable receipt CAMPAIGN_COPY_DRIFT_R1.json records the exact
source-only and changed paths, hashes, sizes, source execution base, and
publication base observed after the copy. Treat this worktree as a prepared
snapshot, not as the final campaign source; root should synchronize and verify
the changed packet after the result write settles.

## Prepared sparse publication worktree

A no-checkout worktree was prepared at
/home/vlad/development/wunder/research_tmp/resume4h-publication-20261024 on
branch experiment/resume4h-evidence-20261024, based on
origin/research-team at d32be666a10c6928a100a91a021337225520ce32.
Sparse checkout includes only selected campaign, navigation, ledger, index,
prior-review, and current experiment-draft paths; no full repository checkout
was performed. The base index was initialized with git read-tree -mu HEAD.
At final verification both cached and tracked-worktree diffs were empty, so no
files are staged and no canonical trainer code was copied over the publication
base.

The copied packet contains the campaign snapshot, the prior ten-report Astra
review folder and its source note, and the new experiment-draft README, PR
draft, and recipe. All 25 files in the latter two groups matched their root
source hashes at verification. CAMPAIGN_COPY_DRIFT_R1.json records that the
campaign copy overlapped root's continuing writes: three source files were
added and seven existing files changed after or during the initial copy. Root
must resynchronize those exact paths, then verify the completed packet before
staging. This temporary mismatch is recorded for reproducibility rather than
silently treating the worktree as an atomic snapshot.

## Source and synchronization records

- [Execution-base source hashes](SOURCE_FILE_HASHES_R1.json) identify all three original blobs, current file hashes and the exact delta; they do not imply parity with research-team.
- [Exact trainer delta](CURRENT_ROOT_TRAINER_PATCH_R1.patch) preserves the inherited closure expansion and current-cycle producer-schema repair without changing canonical trainer files in the publication branch.
- [Initial copy drift](CAMPAIGN_COPY_DRIFT_R1.json) preserves the non-atomic first copy and every then-missing/changed path.
- [Root synchronization receipt](SYNC_RECEIPT_R1.json) records the later selected campaign copy, navigation additions and append-only ledger port. Submission metadata is included for provenance; ZIPs, labels, predictions and moments remain outside Git.

The worktree is a publication view of the recorded execution, not a newly validated execution environment. Exact reruns require the original source checkout plus its recorded patch and private input artifacts.

## Publication-copy URL credential redaction

The root research tree retains the original GitHub API metadata snapshots,
including private query values, as local/private evidence. They must not be
published from the source tree. The publication worktree copies are sanitized
separately: [the rerunnable standard-library helper](sanitize_publication_urls.py)
replaces only HTTP(S) query values whose parameter names contain `token`,
`signature`, `credential`, `secret`, or `api_key`. It preserves URL hosts and
paths, source refs, noncredential query segments, fragments, and decoded
source/scientific content. It skips `PR*_FILES_GH.json` raw dumps; the seven
unreferenced dumps listed in `SYNC_RECEIPT_R1.json` were already omitted from
this copy.

The first bounded pass scanned 440 publication text files and redacted 41
`token` query values in 17 metadata JSON files; the postscan found zero
remaining credential-like query values. No scientific payload was present or
modified. The per-file original and sanitized hashes, paths, parameter-name
counts, script hash, and scan outcome are in
[URL_QUERY_REDACTION_R1.json](URL_QUERY_REDACTION_R1.json); credential values
are never recorded. These changed publication-copy hashes are deliberate and
do not replace or rewrite frozen source hashes.

R1 records the exact helper hash used for the 41-value redaction. The helper
was then Ruff-formatted and lint-checked; this changed formatting/docstring
bytes only. A subsequent [R2 no-op receipt](URL_QUERY_REDACTION_R2.json)
records the current helper SHA-256
`ee0ded7ac9de62d9661fbb92af1b8f78c4a10fb3f19cbc19520fca740a98c19b`, a second
440-file scan, zero changed files, and zero remaining matches. Ruff format
check and Ruff lint both pass for the current helper.

Root's final synchronization may reintroduce source copies of those metadata
files. Rerun the helper against the publication worktree before publication;
it writes a new create-only `URL_QUERY_REDACTION_R<n>.json` receipt and is a
no-op for already-redacted files:

```sh
env WNN_MEM_MAX=80000000 WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_RUNTIME_MAX=30s \
  WNN_SCOPE_UNIT=wnn-publication-url-redaction-RN-20261024 \
  scripts/limited .venv/bin/python \
  research/campaign_2026-10-24/resume4h_r1/publication_source_r1/sanitize_publication_urls.py \
  --root /home/vlad/development/wunder/research_tmp/resume4h-publication-20261024/research/campaign_2026-10-24/resume4h_r1
```

## Publication reference resolution

[Link adaptation](LINK_ADAPTATION_R1.json) resolves 62 source-checkout references missing from the research-team base: 46 use full-commit GitHub URLs after byte equality with the execution base was checked; 16 changed or untracked small records are preserved in [exact reference snapshots](reference_snapshots/README.md). Two repaired Python modules are included there as documentary source snapshots; no canonical `src/` path changes. Thirty publication Markdown files have link-only adaptations. The manifest records every original/replacement destination and before/after hash, so reversing those replacements reconstructs the pre-adaptation bytes. Original root reports remain unchanged.

The [R3 query-redaction pass](URL_QUERY_REDACTION_R3.json) re-applies the credential-value redaction after final synchronization and reports zero remaining matches. Source refs and scientific values are unchanged. Publication navigation preserves the research-team base and ports selected new entries; it is deliberately not a byte copy of the diverged execution checkout's old navigation.

- [Staged snapshot](STAGED_SNAPSHOT_R1.json) lists exact staged paths, sizes and hashes before its own receipt. Generated SVG/CLI output and frozen-source whitespace is preserved; the standard whitespace check reports those warnings, with no conflict markers. No canonical source, scientific binary payload or unrelated root change is staged.
