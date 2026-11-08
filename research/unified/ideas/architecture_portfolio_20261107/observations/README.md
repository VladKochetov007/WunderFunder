# Read-only source observations

Captured7 November2026 for architectural planning. These are source responses,
not new model evaluations. [Content hashes](../SOURCE_RECEIPTS.json) identify
the acquired bytes; no Authorization header or credential is included.

| Local source | Purpose |
|---|---|
| [PR61 terminal comment](pr61_status_comment.json) |10/12 completion, teacher scores and unstarted-student failure |
| [PR61 metadata](pr61.json) | Open PR identity/source references |
| [R5 release lookup](pr61_r5_release.json) | GitHub404 for the proposed final tag, not absence of prior weights |
| [Open research PR inventory](open_research_prs.json) | Prevent proposing old mechanisms as new implementations |
| [PR63 description](pr63_info.json), [comments](pr63_comments.json) | Prepared curvature-matched Huber proposal; no comments/results |
| [PR64 description](pr64_info.json), [comments](pr64_comments.json) | Prepared causal-recency proposal; no comments/results |
| [PR65 description](pr65_info.json), [comments](pr65_comments.json) | SDPA gating and conditional scale/student plan; no comments/results |
| [PR66 description](pr66_info.json), [comments](pr66_comments.json) | Matched SwiGLU plan; no comments/results |
| [PR67 description](pr67_info.json), [comments](pr67_comments.json) | Fixed-TL3-contribution objective and chronological source manifest; no comments/results |
| [PR67 submitted-model audit](pr67_SUBMISSION_AUDIT.md), [source pin](pr67_SUBMISSION_AUDIT_SOURCE.json) |6 November portal observations and separate clean-valid comparison |
| [Historical Mamba-2 source](historical_mamba2_regressor.py), [source pin](historical_mamba2_source.json) | Reference-only CUDA adapter inspection; do not import this snapshot as a new implementation |

The local arXiv search used the bundled skill with query
`all:"limit order book" AND all:transformer`, maximum4 papers, returning
arXiv2609.16930,2610.02917,2502.15757 and2508.17086. The raw abstract output is
kept locally, untracked; [SOURCES](../SOURCES.md) links the relevant primary
papers and records which components might transfer. This search found no
reproducible competition-winning score. A first malformed CLI invocation and
an underscore query were corrected; they are not scientific attempts/results.

The authenticated-user response is local/untracked because the portfolio only
needs account identity, not a copied profile. The API read was as SemyonEpanov;
the GitHub connector404 was followed by authorized native read access. No
GitHub object was created or updated.
