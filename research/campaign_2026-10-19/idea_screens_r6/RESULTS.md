# R6: two-target gradients at one frozen GRU endpoint

The two target gradients were **mostly aligned in the shared input/GRU trunk** on this fixed probe: 12 of 16 windows had positive inner products, four negative. The pooled cosine is **+0.333540**. This directly measures gradients rather than inferring conflict from scores or input covariance. It does not support a blanket claim that the two targets oppose each other throughout this model.

| Parameter block | Pooled cosine | Negative windows / 16 | Median window cosine | Window cosine range |
|---|---:|---:|---:|---:|
| Shared input + GRU | +0.333540 | 4 | +0.575297 | −0.665909 … +0.954083 |
| Shared hidden readout | +0.026854 | 3 | +0.038487 | −0.289026 … +0.273462 |

All four negative trunk windows had cosine below −0.1. Two hidden-readout windows were below −0.1. There were no undefined cosines. The trunk negatives all occurred in the first fixed window (start1024); the second window (start4096) had positive trunk cosine in all eight sequences. This is a descriptive within-probe pattern, not an identified time/settling effect or an independent replication across 16 trials.

The pooled target1 norm was1.365746 in the trunk versus0.489626 for target0; the readout norms were0.622395 and0.310247. Norm ratios vary substantially by window. Their differences depend on this checkpoint, masked WMSE and target scales; they do not establish a balancing rule or improvement from rescaling. The nearly orthogonal pooled readout gradients and local conflicts leave specific interference questions open, but PCGrad, separate trunks/heads and optimizer interventions were not tested.

## Fixed measurement

[The pre-code design](DESIGN.md) and [design review](DESIGN_REVIEW.json) fixed the R01 F0 wide checkpoint, first eight components of the existing R3 **fit** panel and two128-row windows per sequence. The model is the existing211-input / GRU512×1 FP32 endpoint with its saved normalization. It was reconstructed with canonical sourcea266b036; raw112 plus99 causal event features were assembled from the legal prefix, without reading evaluation arrays.

The forward state starts at row0 and follows the full causal prefix in chunks of at most512 rows. It is detached at each diagnostic boundary; backward covers only that128-row window. Thus these are truncated-window gradients conditional on a legal fixed-weight prefix, not the original training chunk2000 horizon, a stale-state comparison, or a full-sequence Jacobian. Targets use the unchanged canonical clipped-target pooled WMSE, and the two contributions share the same joint weight denominator. Input/GRU parameters and shared hidden readout are measured separately; the final target-specific output rows are excluded.

All16 windows completed;24 source-hashed fitting NPY files were mapped,0 evaluation sequences read,0 optimizer steps or quality grades performed. The state fingerprint was identical before/after and every parameter `.grad` field stayed empty. Autograd produced two target gradients from the same forward; scalar norms and cross-products were accumulated in FP64, without transferring gradient arrays or changing weights.

## Evidence and execution boundary

[Exact RESULT](records/probe_v3/terminal_capture/RESULT.json), [SOURCE](records/probe_v3/terminal_capture/SOURCE.json), [DONE](records/probe_v3/terminal_capture/DONE.json), [compact scalar audit](RESULT_AUDIT.json), [45-pin source manifest](PREPARATION_MANIFEST.json), [V3 implementation review](IMPLEMENTATION_REVIEW_V3.json), [root startup review](ROOT_EXEC_STARTUP_V3_REVIEW.json), [resource reconciliation](RESOURCE_RECONCILIATION.json) and [execution records](records/README.md) trace this diagnostic. The audit reconstructs window summaries and pooled dot/norm/cosine from compact cross-product records; it does not repeat backward or inspect tensor payloads.

The successful CPU unit exited0at01:11:35UTC, about41s from rounded systemd timestamps, with20.015s recorded CPU. Probe phases total34.091s. Systemd MemoryPeak478,564,352B, progress processRSS maximum822,554,624B and reported own-cgroup-current maximum524,730,368B remain separate observations with no invented explanation for their difference. Parent and populated F3 child memory events did not change. Raw receipts were saved/verified before stopping only the exact exited support unit.

Two pre-gradient startup failures are retained. V1 correctly rejected root's same-interpreter import smoke; V2 rejected TorchVersion metadata under restricted unpickling after24 sample NPY mappings. V3 changes only the narrow restricted-loader compatibility/inactive auxiliary-default gate and source-wave startup ordering, not the scientific sample/windows/objective. See [loader correction](LOADER_COMPATIBILITY_V3.md) and [preserved versions](records/README.md). The terminal empty ControlGroup caused one unscoped root-cgroup observation; it is preserved and explicitly excluded from own-unit event inference.

Idea20 now has one narrow actual-gradient diagnostic. One exposed F0 checkpoint, eight adaptively researched fit components and16 dependent windows do not estimate training-wide conflict prevalence, prove harmful optimization or establish a quality/public/CP30 gain. Ideas16/21/23, recurrent-state energy and covariance-preconditioner interventions remain untested. No follow-up training or sweep is admitted by this result.
