# Proposed complete-valid baseline3 runtime audit

Preparation only. No empirical audit has run. The immutable plan requires independent review, a separate root allocation, and release of CPU18/SMT58 before execution. The unchanged ZIP is `submissions/overnight_baseline3/baseline3_fixed.zip`, SHA256 `9471821e81e7c3f80a088147512183b72e0ab52bc593a1136d2becc229d0d098`.

Current review plan SHA256: `1337643db2d5b3d07932ddc47c1a2f808d57c31a41a80c85eee04e622bda1275`.
Worker SHA256: `27bbca080e70696d3f6611158d38c76bc70a0391df70e977bcf7cc3d7917f7da`.
Launcher SHA256: `b335b031b016355d4c959c4d1f8a6cc9c0d5ee1a0b9e50b019fc762cee37ce3d`.

## Coverage and feasibility

The audit visits all 1,873 row groups of complete `valid.parquet` exactly once, in file order, with 20,000 rows each. It uses all **37,460,000 complete `PredictionModel.predict` callbacks**, including warmup/state updates. The caller sets `need_prediction = step_in_seq >= 99` in advance; there must be exactly **37,274,573 emitted two-value float32 predictions** and 185,427 warmup `None` returns. It does not read dataset prediction flags or scoring masks.

The existing frozen `SequenceFile.read(targets=False, mask=False)` already requests only `seq_ix` plus the 112 raw inputs and uses `use_threads=False`. A narrow proxy independently rejects every other requested column, checks constant sequence identity within each row group, and records one read per row group. Only one sequence's feature and output buffers remain resident. Every raw feature and every emitted prediction must be finite. Per-sequence and whole-stream checksums bind output order; predictions are not saved as a scoreable cache.

The ZIP is safely extracted into a fresh temporary directory; the process runs from an empty working directory under the scorer's existing CPython3.11/glibc2.36 environment. Every package file, model weight, source reader, loader, interpreter and declared dependency version is hash/version checked. The package chooses its normal v3 branch, and its three members and equal thirds remain unchanged. No monkeypatch of CPU detection, no bulk `run()` path, no governor, and no second full engine are used.

The existing 67.71 microseconds/row trace suggests about 42.3 minutes of callback work for 37.46M rows, plus input decode and validation overhead. This is an estimate only; the proposed audit tests substantially broader inputs than cycling five sequences. The opaque native ensemble exposes no state serialization API. State evidence is therefore explicit sequence-ID/reset transitions and complete warmup callbacks, alongside the previous frozen reset/prefix/reference proof. No hidden-state hash or new full-data numerical oracle is claimed.

## Time and resource definitions

- **Callback time:** sum of timed complete per-sequence callback loops, including caller attribute assignments, packaged callback, cheap return-shape/dtype/None assertions, and output assignment. No per-row timer or expensive numerical oracle is added. The reported per-callback average divides by all 37,460,000 rows.
- **Input time:** complete Arrow reading, decompression, sequence-ID checks and conversion to contiguous float32 features. It is separate from callback time.
- **Validation/checksum time:** finite checks and streaming raw/output hashes, batched per sequence.
- **Full process time:** launcher start through worker exit, retaining dependency/source checks, startup, extraction, model initialization, all input I/O/conversion, callbacks, validation/checksums, pre/post whole-Parquet hashes, and temporary-directory cleanup. No validation or I/O cost is subtracted to claim a process deadline pass. Elapsed time through the last callback is also recorded.

Report observed time on 37.46M rows and separately project each total to 39.4M by multiplying by `39.4/37.46`. Callback qualification requires that projected callback time be **at most 50 minutes** and both declared SMT cleanliness conditions pass. Report separately whether the projected **full process time is at most 60 minutes**. Functional completion, runtime qualification, projected process-budget compliance, and the observed hard deadline are separate fields.

The launcher uses a user systemd scope with `CPUQuota=100%`, `MemoryMax=16G`, `MemorySwapMax=0`, plus CPU18 affinity and a 16GiB per-process address-space cap. Library compute pools are set to one; actual OS helper threads are permitted, counted, and sampled with their CPU ticks. Every observed thread must retain CPU18 affinity. No other CPU or GPU is used. A generated metadata-only scope probe on CPU24 verified actual `cpu.max=100000 100000`, `memory.max=17179869184`, and swap limit zero. CPU18/58 remained untouched. An initial `taskset -c24` argument error occurred before Python ran and is preserved; the correct frozen command uses separate `-c`, `18` arguments.

The single attempt has a **65-minute maximum budget**: termination at launcher elapsed 3,895 seconds, two seconds before forced kill, and three seconds reserved for final receipts. The launcher records child PID/process-group/start ticks immediately. There is no retry or restart, including after noise or failure; partial sequence receipts and logs are retained.

SMT58 has a single two-second precheck requiring busy fraction below 5%. After the trace starts, all outcomes are retained. Both callback-region aggregate sibling occupancy and whole-trace sibling occupancy must be below 5% for clean runtime qualification. Per-sequence start/end jiffies and resource samples are saved; aggregate occupancy uses summed busy ticks divided by summed ticks, not an average of ratios. This tests sustained contamination as well as reporting individual samples.

## Input identity and limits

Input SHA256 is `2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782`, 5,615,509,128 bytes, resolved read-only source `/home/grinchenkoeg/WunderFunder/datasets/valid.parquet`. Complete file hashes and size/mtime/path are checked before and after the audit. Hashing touches opaque file bytes but never decodes target or scoring-mask columns. The pre-hash warms the filesystem cache, so recorded I/O is explicitly **not a cold-cache measurement**. The input/source identity is inherited from preserved receipts until the allocated worker re-verifies it.

This is a coverage/runtime audit of an existing package. It creates no quality score, no new target-derived summary, no training evidence, no new artifact, and no submission. Previous exact engine comparisons are reused; checksum reproducibility alone is not a new numerical oracle. Host OS/shared libraries are not fully hermetically pinned. Measurements qualify this shared-host v3 path, not every scorer machine or fallback ISA. Full validation exposure does not become fresh generalization evidence because targets are absent.

`plan.json` contains the exact source/package/input hashes, column whitelist, immutable invocation template and expected outputs. `author-source-checks.json` records compilation plus generated jiffy aggregation checks. `resource-launcher-probe.json` preserves the generated launcher verification, including the pre-execution syntax failure. No empirical data was opened during this preparation.
