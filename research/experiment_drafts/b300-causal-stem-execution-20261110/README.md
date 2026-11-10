# B300 receiving execution,10 November2026

The user explicitly requested loading the existing8B300 in addition to the running L4 wave. This packet executes the six scientific cells of [PR74](https://github.com/VladKochetov007/WunderFunder/pull/74); it creates no new model hypothesis or duplicate draft. The importable source remains the frozen PR74 commit8a8afec964985a6ba8b13d1c9af1ab00e14c6504. [Current authority](AUTHORITY_R2.json) binds the actual source-capsule bytes, recipes, physical GPU pairs and finite budget.

## Receiving amendment

Keep Transformer768x24/12heads/context1024, B128 per rank/world2/global256, chunk/horizon256/A4, Muon3e-4, fixed4E/3280updates and BF16/BF16x9 with FP32 master/optimizer tensors. Plain/K1/K16 are each fitted fresh at seeds203201/203202. Primary K16?K1; secondary K16?plain and K1?plain, separately per seed/target. Preserve configuredE4; other epochs are diagnostic.

The original two-pair same-seed sequential placement is amended to four physical pair streams:0/1 plain203201?K1_203202;2/3 K1_203201?K16_203202;4/5 K16_203201;6/7 plain203202. Every cell receives its own production50-update and zero-warmup peak50-update profile on its assigned pair, with both-rank model/optimizer/scheduler/RNG reload. All12 profiles and manual ROOT approval precede six fresh fits. No scientific recipe or world-size change.

## Verified receiving state

Native Linux,256CPU,~1.9TiB available RAM,8B300 with275040MiB/card, uniformNV18 interconnect. Python3.12.14/Torch2.14.0+cu130/CUDA13.0. Existing source/cache/footer identities and original receiver_data check passed; no download or environment install. All73 general tests, stemDDP, MuonDDP and compiled BF16/BF16x9 stem tests passed. The initial general-test attempt failed because three JSON test fixtures were omitted; exact same-commit blobs were supplied and only that suite rerun. First failure/logs remain preserved.

Git archive on the Windows sender represented packet JSON withCRLF while Git blob pins wereLF. AuthorityR2 binds exact already-received capsule bytes; per-cell parsed-JSON equality and line-ending conversion proof are recorded in it. Source/manifest hashes are unchanged; no recipe edit. The source receipt contains sourceSHA959fc8a335f97a83de31c12dfa6a24f5db4776b78b70bf45f7890cf09b96b5d6 and manifestSHA1cb6bf60c668f34ea3c49e6f4c3b0c774ea6cdc7e8f8c5f29e5d9f04322f1936.

The independent official split remains train10325 only ? full valid1873,3,528,273 scored rows, global_wp_clip2_v1. Fit normalizers on training only, no teacher/initializer/reference. Validation is repeatedly exposed. No new quality, statistical-significance or public-score claim follows from engineering tests.

## Operation and collection

ROOT owns one existing persistent SSH/RPC stream, GPU admission and birth-bound process groups. Detached bounded controllers preserve source/config/data/runtime/UUID identities, gradients/clipping/LR/loss, full-valid evaluation each epoch and all checkpoints. Source metrics are mirrored to the shared local dashboard http://127.0.0.1:8770/. Progress and terminal evidence update PR74 as SemyonEpanov.

The native first-GPU clock is2026-11-10T19:12:10UTC; original fit deadline00:12:10UTC and collection deadline01:12:10UTC. Budgets:100GB output,100GB disk reserve,256GiB aggregate owned RSS,8threads/rank. No clock reset, automatic extra fits, historical resumes, foreign process signals, rental/billing or competition upload. All12 native profiles passed (600 engineering updates); ROOT independently verified every original receipt against exact source, normalized/raw configs, full official membership, train-only normalization and both-rank reload. [ROOT fit admission](ROOT-fit-admission-r1.json) is `fa608f409499b67ef6c37b6d0e6a07f9c1a32392d321edbf669effd870c8ba18`. All6 fits are admitted. Four fresh fit workers started at19:48:14UTC under native controller2349499/birth174998309; two successors are queued on their independent pair streams. Checkpoints and full-valid evaluation are retained every epoch. Profiles reserved at most110.19GiB/card, peak hostRSS6.40GiB/rank; steady recent windows were~1.2s/update. A provisional full-panel ETA is2.5?3.5h, to be revised after the first complete epoch.

Native root:/data/epanov_s/tabdata-causal-stem-b300-20261110-r1. Local evidence:C:/Users/admin/Documents/Codex/artifacts/b300-causal-stem-20261110. See the [acceptance update](https://github.com/VladKochetov007/WunderFunder/pull/74#issuecomment-6776984446). Execution source/controller review and actual profile/fit receipts are appended here when they exist.

## Receiver correction and evidence

The actual Torch2.14 launcher starts each rank in a distinct session. R1?s group-only resource guard undercounted rank RSS; R1 was used for the bounded engineering profiles only. Full scientific fits use the independently reviewed R2 descendant runner, binding PID/start ticks across sessions for resource accounting and stop signals. It was exercised on the native host with a detached CPU child and an unrelated purpose-created sentinel: the child stopped on deadline and the sentinel remained alive. The first proof read an incorrect receipt filename after its stop assertions; the corrected fresh CPU proof passed. No scientific fit was repeated.

- [Independent review](receiver_critic_r1.md), [frozen R2 identity](receiver-r2-identity.json), and [ROOT twelve-profile review](ROOT-profile-review-r1.json).
- [Original twelve-profile evidence archive](native-profile-evidence-r1.tar.gz) with [file hashes](native-profile-evidence-r1-manifest.json); historical missing-fixture failure remains included in its acceptance receipts.
- [Exact received source capsule](source-r1.tar.gz) and [source manifest](source-r1.manifest.json); source is archival PR74, not this receiving-documentation commit.
- [ROOT fit controller receipt](execution-receipts/033-ROOT-six-fits-launch.json), [initial native phase/GPU proof](execution-receipts/034-fit-start-proof.json), and [native detached-child stop proof](execution-receipts/031-native-owned-tree-proof-r3.json).
- Receiver R1 helpers are frozen in receiving/; fit-only controller/ownership correction in receiving-r2/. Never launch scientific fits with the group-only R1 runner. The ROOT admission is materialized at both control/ROOT-fit-admission-r1.json and control/b300-driver-r1/ROOT-fit-admission-r1.json because the original fit result writer hashes its control-local copy.

Scientific quality endpoints are still pending. Diagnostic profile throughput and a running controller do not establish model improvement.

For exact receiver replay, extract [the original test-fixture supplement](test-fixtures-r1.tar.gz) into the source capsule before its original tests; [its manifest](test-fixtures-r1-manifest.json) binds all three exact Git blobs. These JSON-only additions do not alter importable training source.

## Live observations after launch

At19:55UTC all four first fits had266?271/3280successful updates, last windows~1.21s and GPU Util95?100% acrossall8cards (114150?114268MiB/card). Last per-update losses1.47?1.57 and preclip gradient norms1.41?3.86 were finite; LR?2.992e-4 after164 warmup steps. These describe operation, not tensor-core FLOP efficiency or generalization. See [native update/GPU readings](execution-receipts/035-live-first-updates.json).

The launch environment omitted WNN_MONITOR_HEARTBEAT_SECONDS; raw rank0.jsonl advances normally, while monitor_progress.json only flushes at phase boundaries. Scientific jobs/source/configs are kept unchanged. A separately reviewed [read-only observer](observe_six_fits_r1.py) attaches to all six admitted launcher PID/births as they appear and appends observational GPU/CPU/RAM/I/O events. Sampled step/epoch explicitly come from complete rank0 records, not an invented trainer heartbeat. At20:00UTC it observed500?506updates and~12.3?12.8GB worker-family RSS per pair. It stops appending at each native terminal marker; ROOT collects final observational hashes after close. [Attachment proof](execution-receipts/038-observer-and-live-resources.json).

## Durable reporting and canonical readout

The [PR74 progress publisher](reporting/publish_pr74.py) is running locally (PID15296), posting every30minutes and per terminal fit. It verifies frozen admission/source/normalized config/exact sparse membership and derives successful updates from rank0 logs. All4 exact valid events and fixedE4 terminal status are required for paired scores. The [first automated post](https://github.com/VladKochetov007/WunderFunder/pull/74#issuecomment-6778321915) is confirmed as SemyonEpanov. Readout publication uses [a separate one-result watcher](reporting/watch_readout_r2.py), also live, bounded to the original wave deadline.

The independently reviewed [canonical receiver](receiving-readout-r2/README_R2.md) passed all4 native Linux CPU integrity fixtures, including the symlink check skipped on Windows. [Native receipt](execution-receipts/041-readout-native-CPU-fixtures.json). The separate [waiting launcher](wait_and_readout_r1.py) is detached on B300 (PID2358942/birth175234844), waits for all6 completed fits and phase-lock release, then invokes that exact reviewed receiver once. Current [wait-state proof](execution-receipts/044-durable-readout-wait-state.json). It starts no fit/profile and changes no completed producer files. The actual receiver directory is control/readout-receiver-r2, R1 helpers are control/receiver-r1/receiving, R2 helpers control/receiver-r2, and the actual authority is control/AUTHORITY_R2.json with SHA74663454d5c8ace8f7c1cdeb6c651e3d9a93fb9b2e1c4cc756bd03fb45df3f77; earlier placeholder examples in the receiver README are a template.

Full-valid E1 scores: plain203201 .668349853; K1_203201 .668575532; K16_203201 .668877675; plain203202 .661403130. Full validation took350?365seconds per fit, plus~44?45seconds training-cohort diagnostics. This reporting is on the same exposed development data and does not replace the fixedE4 comparison. First-epoch hardware/resource and standalone graphs are preserved in the shared local artifacts (10PNG+10SVG per active fit).
