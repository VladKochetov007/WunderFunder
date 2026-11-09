# Repository update

Recorded 2026-11-09 for local review.

| Field | Value |
|---|---|
| Branch | codex/transformer-causal-input-stem-20261109 |
| Parent PR | #72, head 4021de8ec5130143c207319c73a102a453eb911e |
| Current HEAD when packet work began | 4021de8ec5130143c207319c73a102a453eb911e |
| Intended target | research-team |
| Target base | 7e4074314a62d0907c9608f9220cf72958708a2d |
| Hypothesis / experiment | H-B300-TRANSFORMER-CAUSAL-INPUT-STEM-20261109 / E-B300-TRANSFORMER-CAUSAL-INPUT-STEM-20261109 |
| Draft PR URL / merge SHA | null / null |
| Execution source SHA | null until the root-owned implementation is committed and reviewed |

The shared worktree contains ROOT-owned implementation and receiver-gate edits.
This packet update records their intended experiment contract; it does not
review or certify those code changes. ROOT owns those files and their receiving
gates. The receiver requires a full source SHA and a clean tree, so no runtime
source pin is claimed while the branch is uncommitted.

The six JSON configs were generated from the exact PR72 C0 JSON using the
standard-library JSON reader/writer. Structural differences are limited to
seed and model.input_stem_kernel; plain omits the latter. Their SHA-256 values
are recorded in recipe.json. The source PR72 config hash is
5881e2facddc48a0023f588c7e47dfd02e065e823bc94d565faf36ec3d7594c9.

Static preparation only. No local tests, model imports, training, inference,
profiles, remote jobs, commits, pushes, or PR creation were performed. No
quality or public-score result is claimed. The actual receiver source identity,
profile approvals, fit receipts, output mount capacity, fit wall time, peak
memory, and artifact sizes remain pending.
