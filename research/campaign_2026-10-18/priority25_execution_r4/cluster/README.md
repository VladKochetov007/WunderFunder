# Cluster handoff through repository issues

The user explicitly authorizes issues for the team’s powerful cluster. The first handoff accelerates the already registered rank1 comparison; it introduces no new model mechanism or completed result.

- [Prepared issue body](ISSUE_RANK01.md) specifies the frozen recipe, first three additional fold pairs, exclusion of the active Beleriand pair, source/data staging, environment checks, missing scheduler binding and return artifacts. It is an operator handoff, not a ready generic cluster launcher.
- [Publication and ownership state](HANDOFF.json) records the authenticated-issue-access dependency, actual Beleriand reservation and proposed unclaimed cluster cells. The issue URL and cluster execution remain null.

Git SSH publication works, but the current GitHub connector cannot access the repository and `gh` is not authenticated. No issue was created. Once an authorized GitHub session is available, inspect existing issues for duplicates, create this issue using the prepared body, record its URL and actual cell claims, and collect the worker’s source-bound evidence. Posting alone does not establish an automatic cluster run.

From the repository root, the exact publication command after authentication and duplicate checking is:

```sh
gh issue create --repo VladKochetov007/WunderFunder \
  --title '[Cluster] Rank 1: paired wide/deep GRU validation across remaining folds' \
  --body-file research/campaign_2026-10-18/priority25_execution_r4/cluster/ISSUE_RANK01.md
```

This command has not been executed. Do not attach an invented issue URL or describe the prepared file as an issued job.

The existing Beleriand sequence continues. Its one-GPU limit remains unchanged, and this cluster handoff does not authorize a new paid-cloud allocation. Return to the [execution packet](../README.md).
