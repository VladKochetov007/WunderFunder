# PR77 progress publisher

`publish_pr77.py` is a local-only reader for the L4 dashboard mirror and ROOT's
fit-admission receipt. It checks the six fixed source/config/GPU identities,
official train/valid membership, rank-zero diagnostics, and the native terminal
and evaluation journals. It does not connect to the L4 host or inspect GPU
state.

ROOT may start the bounded 12-hour publisher after setting `WNN_GITHUB_BASIC`
in its own process environment:

```powershell
& 'C:\Users\admin\Documents\Codex\reviews\galam-test-env-20261027\Scripts\python.exe' `
  'C:\Users\admin\Documents\Codex\reviews\WunderFunder-l4-next70-20261110\research\experiment_drafts\l4-next70-20261110\reporting\publish_pr77.py' `
  --root 'C:\Users\admin\Documents\Codex\artifacts\l4-next70-20261110' `
  --pr77 --maxseconds 43200
```

The default loop polls the local mirror every 30 seconds, publishes at most once
per source-clock half hour, and publishes one deduplicated comment when each
cell reaches terminal evidence. When all six fixed-E6 endpoints and their six
full-valid events pass, it reports the baseline plus the three predeclared
target pairings for each seed. Each output score is the unweighted mean of the
two selected target components. Validation is repeatedly exposed development
evidence; no public score or significance claim is made.

For a local, credential-free inspection, use `--dryrun`. `--once` performs one
live posting cycle and exits. Both modes read the same admission and mirror
files; only the live mode imports the existing allowlisted `ProgressGitHub`
client. The state file defaults beside the script and is persisted atomically
before each POST so a lost response is reconciled by its marker before retry.

Focused CPU tests (no GitHub API calls):

```powershell
& 'C:\Users\admin\Documents\Codex\reviews\galam-test-env-20261027\Scripts\python.exe' -m pytest `
  'C:\Users\admin\Documents\Codex\reviews\WunderFunder-l4-next70-20261110\research\experiment_drafts\l4-next70-20261110\reporting\test_publish_pr77.py' -q
```
