# Tabdata three-epoch bounded cohort

This records a new ten-fit cohort, not completion of the original long experiments. Six PR35 Transformer input-encoding fits use GPUs2–7; four PR37 cold-label WMSE/NLL fits share GPUs0–1, two workers each. PR36 is excluded because its exact warm parents were not present. Full fitting components and full declared held-out evaluation membership are retained; all science uses strict FP32. Partial epochs remain progress rather than fixed-endpoint quality comparisons.

The node unit started at 2026-10-22 18:30:55 UTC. Its absolute four-hour deadline is 22:30:55 UTC, including preprocessing/profiles/fit/evaluation/cleanup/publication. The supervisor owns all children in one systemd cgroup, never signals foreign processes, has no restart, and reserves time for final publication. Its per-phase cutoff test includes a descendant ignoring SIGTERM. The exact deployment and service properties are in `DEPLOYMENT.json`. CPU preparation affinity was separately expanded after observing the full-data preprocessing bottleneck; the scientific recipe and deterministic preprocessing algorithm were unchanged.

The source files here match the reviewed deployment scripts. Runtime PLAN, command receipts, source/data identities, profiles, logs, checkpoints, prediction arrays, official grades and cleanup proofs remain in the node's `state/next4h-20261022-r2` directory. GitHub comments carry observed progress and results. Do not infer GPU activity or completion from this source archive; consult the dated live execution receipts.

The initial cold NLL unit test used a bare Sequential instead of the canonical model.head interface; that fixture was repaired and all four tests pass on the node. Numeric tests require the existing conftest strict-FP32 fixture, which was included before rerunning all 29 checks. No scientific source change was made to resolve either packaging/test issue.

[Continuation V2](continuation_v2/README.md) supersedes the initial preparation receipt with actual ten-fit execution, the preserved pre-GPU cgroup failure, prospective NLL command correction and the original absolute deadline.

[Separate twelve-fit NLL scaleout](scaleout_v1/README.md) records additional paired seeds, profile admission, measured lane headroom and the same original deadline; its live receipt initially precedes scientific-fit admission.

[H512 capacity screen](wide_v1/README.md) adds16 actual fits, reaching38 concurrent trainings after all real-shape profiles passed. [Private MPS probe](mps_probe_v1/README.md) records the conditional0.79x added-pair result and complete probe cleanup; no aggregate-speed or quality claim.

[Logout incident and recovery](logout_recovery_v1/README.md) supersedes earlier live activity: user-manager logout stopped the original services; corrected retries restored38 fits without renewing the four-hour wall. Original partial results and the failed first NLL recovery profiles remain preserved.
