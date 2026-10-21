# Groupwise fast/slow Kalman feature memory profile

**Status: completed resource profile; no model-quality result.**

On 2026-10-18, the exact groupwise feature extractor and weighted-ridge path ran on the first predeclared `fold0.fit.train` component (component 2406, row group 6768, sequence 1386210565). The profile read one complete 20,000-row sequence and its fit labels, computed prefix/future and reset invariants, constructed the raw, raw-plus-clock and raw-plus-39-state designs, and fit the small ridge shapes. It created no OOF predictions, read no scoring masks, and produced no score.

## Resource result

- Worker: one CPU, 2,000,000,000-byte reservation, no swap, no GPU.
- Runtime: 5.02 seconds; worker cgroup peak 1,122,492,416 bytes (process RSS peak 1,132,536 KiB).
- Parent campaign peak: 7,675,568,128 bytes, below the 17 GB working target.
- Worker, parent-local, and parent-hierarchical `high`, `max`, `oom`, and `oom_kill` counter deltas: all zero.
- Exit status: 0. No limit event or GPU allocation was observed.

The profile used source footer and sample pins before opening Parquet metadata. Its [copied profile receipt](../../campaign_2026-10-18/online_features_r1/group_fast_slow_r1/profile_receipt_r1.json), [durable attempt](../../campaign_2026-10-18/online_features_r1/runtime/attempts/group-fast-slow-profile-20261018-r1.json), and [worker log](../../campaign_2026-10-18/online_features_r1/runtime/logs/group-fast-slow-profile-20261018-r1.log) preserve exact evidence. It supports the same 2 GB worker cap for the full cross-fit only as a measured single-sequence allocation basis; the full runner still needs fresh parent-resource admission and ongoing monitoring. It does not prove the 32-component peak or improve the model.
