# Live frontier

Updated 2026-10-06T22:33:01.967694+00:00. Contract: [CONTRACT.md](CONTRACT.md).

- Own public best remains **0.6660**; Maxim's reported **0.6674** is the source reference. No new submission or verified public gain.
- C0 ordinary continuation finished both seeds and fixed evaluation. Equal-pair local correlation **0.68115296**, warm-parent pair **0.68134198**; difference **−0.00018902**, component-bootstrap 95% interval **[−0.00536403, +0.00354846]**. This is exposed fold0, not the public champion comparator or a public forecast. [Report](local/E502_C0_EVALUATION_v2/report.md), [plot](plots/C0/paired_control.png).
- Seed3's original failed guard receipt remains preserved; independent attestation accepted its completed artifact after a parent-limit reload incident. Correct aggregate limits are now persisted: 10CPU,20GB RAM, no swap. The grader's initial target-index failure occurred before scoring; fixed union validation and a separately frozen second gate produced the report.
- Missing teacher training predictions are exporting serially under the shared GPU lock. File-cache reclaim was explicitly recorded before admission; RAM reserves were not lowered. [Plan](local/TEACHER_EXPORT_SEQUENCE_v1.json), [cache receipt](local/E503_PARENT_CACHE_RECLAIM_20261007.json).
- C1–C4 family-distillation fits remain unadmitted pending complete caches, two clean raw baseline teachers, exact recipe validation and resource profiles. The independent implementation passed 13 focused tests, including preserving random state and batch order when extending the head.
- E509 serving capacity screen measured a **1.713×** median runtime ratio for two warm-parent trunks versus the public champion's single trunk. Concurrent laptop load, one input stream and parent checkpoints limit this result. It requires serving follow-up; it does not approve or reject any candidate. [Receipt](local/E509_SERVING_CAPACITY_RESULT.json).
- Kaggle E508 actual-trainer profile is running privately; no profile result yet. GPU v3 completed a hardware-only gate, not scientific training. TPU v1 remains queued. The durable poller records quota/status without resubmitting. E510 production teacher packet stays blocked on controller review of the actual profile.
- Fifteen literature proposals H181–H195 remain registered with unknown numerical priors. Freebuff GLM5.3Flash/max is processing a fresh directed Net2Net source. OpenCode **MiMo-v2.6-flash-free** passed actual access and proposed sampled within-sequence relational distillation; an isolated external implementation is underway. Neither has a scored result. Grok4.7 is not exposed through the currently configured Freebuff/OpenCode routes; it has not been used.
- [EDA notes](EDA_NOTES.md) retain historical teacher disagreement findings and saved plots. Disagreement was not monotonically related to error and is not a calibrated hardness signal.

Next: finish teacher exports and profile-gated cloud teachers; admit matched C1–C4 in small blocks; independently validate externally implemented exploration children before Kaggle training. Preserve the exploration allocation rather than cycling on ordinary continuation.

Tree catalogs are point-in-time snapshots; journal and original evidence are append-only. Live logs may change without invalidating structural preservation.
