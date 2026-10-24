# PR audit #1–24

This folder preserves the fresh GitHub metadata and a read-only scientific-source audit for every repository pull request numbered 1–24. It covers 22 existing PRs; #16 and #19 do not resolve as pull requests. The audit separates proposals and engineering checks from scored endpoints, records evaluation exposure and ancestry limits, and identifies where artifact manifests point without claiming that private payloads were downloaded or verified.

- [REPORT.md](REPORT.md) gives per-PR status, head SHA/local tracking comparison, newest results, artifact locations/accessibility, and likely value within the active four-hour window.
- `pr_01.json` … `pr_15.json`, `pr_17.json`, `pr_18.json`, and `pr_20.json` … `pr_24.json` are the structured `gh pr view` responses, including body, comments, commits, changed files, checks, reviews, state and head.
- [pr_16_not_found.json](pr_16_not_found.json) and [pr_19_not_found.json](pr_19_not_found.json) preserve the failed PR lookups. PR #18 refers to issue #19; that does not create a PR #19.
- [pr15_latest_results_comment_rest.json](pr15_latest_results_comment_rest.json) and [pr15_stop_comment_rest.json](pr15_stop_comment_rest.json) preserve focused REST metadata for the two decisive PR #15 comments, including `updated_at` for the edited stop comment.

GitHub reads used the prescribed `scripts/limited gh ...` wrapper with the then-current quarter-core / 250 MB per-call bound. Local remote-tracking refs were read through the same wrapper; all 22 existing PR branch refs exactly matched the fresh GitHub head SHA. No fetch, checkout, merge, training, test, import, or artifact-payload download was performed. A scan found no signed-URL query tokens in the saved metadata.
