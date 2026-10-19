<!-- https://github.com/VladKochetov007/WunderFunder/pull/26#issuecomment-6256781766 updated 2026-10-19T14:30:03Z -->
<!-- tabdata-b300-20261018:E-B300-PR4-AGE-SUCCESSOR-V1:status: -->
### E-B300-PR4-AGE-SUCCESSOR-V1: status update
**Status:** partial / in progress. Scored matched pairs 0/30; active paired cells 4; pending paired cells 26; terminal false.
E-B300-PR4-AGE-SUCCESSOR-V1 is a new PR4 successor; it does not claim that the source PR was executed. Each reported value is one matched fold/seed comparison using the original official metric and epoch-5 endpoints.
The compared endpoints are the fixed epoch-5 strict-FP32 successor arms on the same fold and seed. These are exposed development folds; partial means are descriptive and do not establish significance, pristine holdout performance, or serving readiness.
Primary paired-cell mean delta WP: —. Equal-fold means below are secondary descriptive summaries.

| Matched pairs | Control WP (equal-fold mean) | Feature WP (equal-fold mean) | Paired delta (equal-fold mean) |
|---:|---:|---:|---:|
| 0 | — | — | — |

Source identity: `a266b036b02e8e012c6eb31072291dad618a169d`.
Evidence and frozen contract: https://github.com/VladKochetov007/WunderFunder/blob/experiment/tabdata-native-age-cp30/research/campaign_2026-10-18/b300_execution_r1/parallel_expansion_v1/README.md
Automated reporting only; this process does not fit, export, or grade models.


<!-- https://github.com/VladKochetov007/WunderFunder/pull/26#issuecomment-6268924635 updated 2026-10-19T14:16:32Z -->
<!-- tabdata-user-stop-request-20261019-r1 -->
### User-requested tabdata stop; remote stop not yet confirmed

The user requests preserving results and removing all owned training from the B300 node. **No new fits, profiles or scientific retries are authorized.**

**Remote stop is NOT confirmed:** earlier SSH sessions closed; the latest explicitly requested plain SSH attempt (14:12 UTC / 17:12 Moscow) reached the password prompt. After password input, the server closed the connection without providing a shell. The exact cause is unknown; there was no Permission denied response. No remote command or stop signal could be executed. Existing GPU workloads may still be running; no empty-node or freed-resource claim is made.

Latest preserved PR15 snapshot (14:10 UTC / 17:10 Moscow): 64/120 scored fits, 16/30 cells per arm. Equal-fold means: baseline 0.67824425; moments 0.67809108 (delta -0.00015317); dictionary 0.67756978 (-0.00067447); combined 0.67702747 (-0.00121678). Partial development-fold evidence, not a significance or public-transfer claim.

R01 recovery has no published validated paired scores: twelve receipt chains report a stdout filename mismatch. Raw grade JSON/checkpoints require server access before numerical recovery can be claimed. Age/innovation each report 0/30 scored pairs.

Checkpoints/predictions/logs must remain intact. With a working node shell, stop the owned `tabdata-*` units and verify their cgroups/project CUDA processes have drained; preserve foreign workloads.

[Frozen result/API snapshots and access evidence](https://github.com/VladKochetov007/WunderFunder/blob/6b2f92e2fa72cf43b37b73ead9dba9dcb7989ac8/research/campaign_2026-10-18/b300_execution_r1/stop_20261019_r1/README.md).
