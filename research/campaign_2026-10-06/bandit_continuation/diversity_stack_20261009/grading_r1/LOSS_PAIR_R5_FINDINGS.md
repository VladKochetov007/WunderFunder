# D01: paired AdamW losses, evaluated October 9

WMSE wins this small matched screen. Equal blending does not improve the overall score. This is one seed, 350 complete training sequences from 83 components, two epochs / 220 updates per arm, evaluated on the **exposed** fold0 panel of 468 groups / 141 components. It is not a comparison with the full-data submission models.

| Endpoint | Target0 | Target1 | Official mean |
|---|---:|---:|---:|
| AdamW WMSE | 0.591212178 | 0.620791668 | **0.606001923** |
| AdamW weighted Huber, delta1 | 0.583089471 | 0.619388800 | 0.601239135 |
| Fixed equal raw-prediction average | 0.588875601 | **0.622084801** | 0.605480201 |

Huber minus WMSE is -0.004762788, paired component-bootstrap 95% interval [-0.008843298, -0.001161896]. Equal blend minus WMSE is -0.000521722, interval [-0.002507812, +0.001263459]. The 2,000 draws use seed271828 and resample whole evaluation components. They do not quantify training-seed or public-transfer uncertainty. The primary specification was fixed before recovered predictions were scored; no blend weights or checkpoints were searched.

The independent grader reproduced the official streamed metric with a separately centered formula to within1e-12. Fixed scored masks, original clipped-label magnitude weights and prediction/label clipping at±2 are unchanged. Training and evaluation components are disjoint and both models started from our own matched initialization. Both inference outputs were FP32.

[Primary result](LOSS_PAIR_R5_GRADE_R2.json), SHA256 `94c8bb8031be7e4776a7b80f694d852551165bb0b52f7f9df5d287c8b3b46f34`; [frozen specification](FROZEN_GRADER_R2.json); [independent artifact review](../ROOT_GPU_R5_TERMINAL_REVIEW_R1.json).

## What this says about combining models

Loss diversity alone is modest here: weighted prediction correlations are0.98563/0.98671 and residual correlations0.98468/0.98600 for targets0/1. These are posthoc diagnostics of individually clipped predictions. The equal blend improves target1 by0.00129313 but loses0.00233658 on target0. That makes target-specific weights a concrete hypothesis to test with component-cross-fitted selection and later fold/scale checks; it is not a retrospectively confirmed gain. [Saved diagnostic report](LOSS_PAIR_R5_DIAGNOSTICS_R1.json) and [reproduction script](analyze_loss_pair_disagreement.py).

The priority remains the already planned matched **Muon × WMSE/Huber** pair. Complete that optimizer-by-loss pool before allocating broad loss sweeps. Then assess individual models, fixed combinations and component-cross-fitted policies. A weaker individual remains eligible for conditional teacher or blend value. This result does not reject Huber under another optimizer, training scale or target policy.

Huber also changes gradient scale, not just outlier influence. R4 recorded median gradient norms approximately0.230 versus0.683 and clipping fractions0% versus17.27%. A future `2 × Huber` control can help separate local quadratic-loss scale from tail treatment. Register it as a distinct loss-scale interaction; do not rewrite D01 or D02. No such fit has been admitted by these notes.

## Execution, cost and retained artifacts

R4 finished both fits on two T4s, then failed prediction export on an unsupported CLI argument. R5 reused the **same checkpoints**; it did not repeat training. Root verified all149 collected files /298,070,207bytes against the terminal inventory, then all14 shared control artifacts. Nested `out/recovery/failure.json` describes the preserved R4 failure; current R5 top-level status and export receipt are complete.

R4 recorded epoch training times about57.7+23.1seconds per arm. R5 worker wall was166.84seconds; concurrent exports each took81.37seconds. These exclude provider startup and do not establish a TPU or5090 multiplier. Separate process wall for the original fits was not recorded and remains null. Kaggle provider quota increased172.9allocation-seconds from the last pre-R5 snapshot to02:06UTC; this account difference is not a per-kernel billing receipt.

Each prediction NPY is74,880,128bytes including its128-byte header. Checkpoints, full resume state, normalization, initialization, sample orders, LR/beta and gradient traces remain alongside hash-bound sidecars. They can be rescored without another fit. The primary result contains absolute prediction paths; the completed R5 kernel also supplies private reusable control outputs for native GPU/TPU experiments.

No public score or expected-public-best claim follows from this screen. Fold1 confirmation and serving relevance remain separate work.
