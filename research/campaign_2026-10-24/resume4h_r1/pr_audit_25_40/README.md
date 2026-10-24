# Pull request audit: PRs 25–40

This folder records a fresh read-only review of all repository PRs numbered 25 through 40, including merged PRs #32 and #40. The exact GitHub PR JSON, captured through bounded `gh` calls, is preserved per PR; see [the findings report](REPORT.md). The report distinguishes plans, engineering profiles, partial fits, saved model outputs, and official quality grades.

- [REPORT.md](REPORT.md) — state, head SHA/date, scientific stage, precision/ancestry, available result evidence, and artifact accessibility for every PR.
- `PR##_GH.json` — exact `gh pr view` response for each PR 25–40.
- `PR##_FILES_METADATA_GH.json` — fully paginated changed-file path metadata for each PR, including additions/deletions; these lists avoid `gh pr view`'s 100-file cap and do not contain patches.
- `PR33_RAW_DELIVERY_GIT_METADATA.json` — local Git tree metadata (blob OID and byte size only) for the one tracked #33 archive.
- `PR33_ARCHIVES_DIR_GH.json`, `PR33_ARCHIVE_ARCHIVE_MANIFEST_R1.json`, and `PR33_ARCHIVE_README.md` — GitHub directory listing, trusted contents manifest, and archive README for #33's tracked raw snapshot tarball. The archive was initially not fetched; see the later recovery receipt below.
- [PR33_ARCHIVE_RECOVERY_R2.json](PR33_ARCHIVE_RECOVERY_R2.json) — canonical physical-scratch recovery receipt, generated from the archive, branch manifest, and Git object bytes. It verifies all 368 file paths/sizes/hashes without extraction or execution; no trained model weights or endpoint payloads are present. `PR33_ARCHIVE_RECOVERY_R1.json` is preserved as the earlier hand-entered receipt; R2 supersedes it for digest provenance.
- `FRESH_INVENTORY_GH.json` — a fresh all-state PR inventory; PR #40 was newest at the query.
- `ACTIONS_*_ARTIFACTS_GH.json`, `ALL_ACTION_ARTIFACTS_GH.json`, and `RELEASES_GH.json` — actual workflow-artifact and release metadata queries.
- `PR26_SHORT4H_*` — public packet directory metadata, the committed README and the pre-run publication snapshot. Final age/innovation scores are preserved in the exact PR #26/#27 comments inside their PR JSON.
- [remote_sources/README.md](remote_sources/README.md) — source-bound PR #32/#40 engineering evidence-branch metadata; root later recovered and verified the newest #32 archive as recorded in [REMOTE_ARTIFACT_RECOVERY_R1.json](../REMOTE_ARTIFACT_RECOVERY_R1.json).

The initial PR audit read only small text/API metadata. Subsequent recoveries verified the #32 engineering archive in `REMOTE_ARTIFACT_RECOVERY_R1.json` and the #33 operational snapshot in `PR33_ARCHIVE_RECOVERY_R2.json`. Root safely extracted #32 to physical scratch; #33 was streamed without extraction. Neither archive's code was executed. The #33 tarball contains two input bundles, which were streamed only for manifest hash verification, and no trained model weights, prediction arrays, checkpoints, or scientific-fit endpoint payloads. Private B300 run paths are references only; their object sizes and expiry remain unpublished.

Imported README snapshots retain the exact source bytes and their original relative links. Those links resolve in the source branch directories identified by the accompanying GitHub metadata, not in this audit folder. Use this README and the findings report for local navigation; do not treat absent local copies as missing remote source files.
