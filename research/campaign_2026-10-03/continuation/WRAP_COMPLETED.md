# Research wrapped on main

Research is paused at the user's request. All five Codex agents are closed and
no experiment or training process is running. The watcher is stopped and both
queue pause markers remain. Do not launch a run without explicit resumption.

Read [RESUME.md](RESUME.md) for the scientific synthesis, limits and pending
tasks. The October checkpoint contains 2,443 changed files, including 92 plots,
with no newly committed numerical blobs or LFS pointers. Original online-norm
and convolution agent commits are ancestors of main; exact historical patches
and changed sources are in `research/imported_agent_branches/`. Current native
implementations are retained. All local branch histories are incorporated.

`CONSOLIDATION_MANIFEST.json` hashes the checkpoint's changed files except itself.
`WRAP_VERIFICATION.json` records branch ancestry, size and process checks. No
scientific tests or model evaluations were run during wrapping. Historical report
whitespace was preserved to avoid invalidating frozen evidence hashes.

Large numerical outputs remain local, with sizes/paths in
`LOCAL_ARTIFACT_INVENTORY.json` and additional verified recovery archives described
in `CURRENT_RECOVERY.md`. Competition data and trained weights remain external
dependencies. Git alone does not transport those files. No LFS or DVC is needed.

Four existing candidates are copied under `artifacts/submission_quota_2026-10-03/`
as individual ZIPs and equivalent tar.gz payloads. Upload ZIPs individually;
tarballs are transport copies. `SUBMISSION_ARCHIVES.json` records provenance,
member identities, sizes and checksums. Candidate1 is the already scored0.6641
reference; candidates2–4 explore teacher, inference precision and chunk500
ensemble directions. Their scores are not established improvements. No new
full-data Muon/chunk1000 submission exists. Nothing was uploaded or pushed.

The pre-existing nested skill deletion and separate untracked `wunder/` project
remain untouched. They do not prevent using the canonical main branch.
