Legacy prediction export omitted saved preprocessing settings and could expose
partially populated teacher arrays as full caches. Restore checkpoint settings,
require full membership/provenance for training consumers, and reject selected
nonfinite predictions. The four legacy server files were deployed with verified
hashes and retained backups; frozen checkpoints/caches and historical logs remain
unchanged. The reusable main helper is tested but is not a global replacement for
all existing trainer cache paths.

The same change records the user's no-fit research phase: a 249.6M-row census of
four selected price-code columns, a 5.12M-row full-feature sample, input lag/shift
controls, 32 primary literature works, independent criticism and 25 proposal cards.
The census refutes H-001's universal event-clock claim; its original ledger row
is preserved and a correction appended. Each proposal has explicit prior work,
counterevidence and a future 30 paired-cell-per-arm design; adapters and fit
authorization remain pending. No trained candidate or leaderboard gain is promoted.

Validation: Ruff/format/structure and 6 stdlib checks pass; 66 CPU tests pass,
2 dataset-dependent tests skip. This suite includes tiny synthetic optimizer/resume
fixtures, not competition-data fits. After removing the false target-clock test
assertion, the feature suite passes 5 tests with 1 data skip. A frozen-checkpoint
CPU replay on 4 preselected groups / 11,522 scored rows changes canonical WP from
0.06289742 with wrong preprocessing to 0.72994782 with restored settings; manual and
restored inputs/predictions are bitwise equal, checkpoint SHA unchanged. This is
a bug-isolation diagnostic, not a full-panel quality comparison. Source, deployment,
EDA and replay hashes were independently reviewed. Raw unified-diff artifacts
retain their required blank context-line spaces. Generated research-tree refresh
is unavailable in this sparse checkout because an old configured document root
is absent; maintained topic/campaign navigation covers every new output.
