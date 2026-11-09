# Submission audit — 9 November 2026

Read-only signed-in Chrome observations from the official
[submission history](https://wundernn.io/connectome/submit) and
[leaderboard](https://wundernn.io/connectome/leaderboard), plus GitHub release
metadata and the private HF archive. No upload or model execution occurred.

| Current portal evidence | Value |
|---|---|
| Team | kotiki |
| Position | 45 / 5769 |
| Best public WP | 0.66832, tl3.zip, 39m12s |
| Leader | Gargantua, 0.67662 |
| Total submissions | 108 |
| Today | 0 / 5 |
| Latest submission | tabdata-w18-final-20261103-r1.zip, 4 November |
| Latest public WP / runtime | 0.66453 / 14m32s |

No new team submission has appeared since4November. The full-corpus W18 package
is already submitted and is .00379 below the older TL3 public result; reuploading
it is not a new experiment. These are displayed platform observations, not an
independent replay or forecasts for unseen data.

The [HF archive](https://huggingface.co/hft-kagglers/wnn-challenge2-model-archive/tree/main)
was checked live: head167452c72a8e994b280446d2fca93f7c08e16f66,154commits,
21.8GB, last update28days ago. It remains the12October archive; it does not
supply new PR72 weights or CPU acceptance. Its model-card public-best wording
is historical and is superseded by the current portal table above.

The [W18 release](https://github.com/VladKochetov007/WunderFunder/releases/tag/tabdata-pr54-final-w18-20261103-r1)
contains the already-submitted5,454,663-byte ZIP, SHA256
932700e4e10721829641870f18198fca06ba3f9c5f06be5541d70bf38bf880f9, with its
historical Python3.11/offline/CPU package-check receipts. Release metadata was
read; no weights were downloaded or deserialized during this preparation.

The [PR49 checkpoint release](https://github.com/VladKochetov007/WunderFunder/releases/tag/tabdata-pr49-checkpoints-20261029-r1)
is available (80,858,660-byte research archive plus manifest/SHA files). Therefore
"not on this Mac" must not be confused with "weights do not exist". Checkpoints
are not an accepted submission ZIP. Fresh TL3 dynamic-int8 has already appeared
on the portal at .66355, below old TL3; do not repeat it under a new filename.

PR72's latest report has six producer grades but0/6 independent checkpoint-byte
audits and no CPU package. PR70's tiny fixed-blend development gain is not a
public result or evidence that a large Transformer meets the serving limit.
PR74 and this PR have no trained weights at all.

No **new, unsubmitted ZIP with matching byte-bound CPU acceptance** was located
in the checked worktrees/release evidence. To make another submit actionable,
first select an unsubmitted endpoint, bind checkpoint/normalization/ancestry,
export the actual CPU package and measure that exact ZIP on Linux x86-64,
Python3.11, one CPU,16GB RAM, offline, under20MB and within the60-minute budget.
Those execution steps were not run locally. The new PR is ready for receiver
review, not a ready submission and not a claim of a new leaderboard score.

[SUBMISSION_INPUTS.json](SUBMISSION_INPUTS.json) pins the released checkpoint
archives and four relevant endpoint hashes from read-only metadata inspection.
The [PR51 exporter packet](https://github.com/VladKochetov007/WunderFunder/pull/51)
can use the released PR49 inputs without retraining, but first compare its exact
recipe with the already-submitted tl3c blend (.66355). An unexported local packet
is not evidence that the team has never submitted the same predictor elsewhere.
The PR52 release supplies the W18 endpoint missing from PR49, but neither
archive includes consumed teacher prediction banks or complete cache receipts.
