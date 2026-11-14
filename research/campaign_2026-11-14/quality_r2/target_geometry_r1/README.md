# Teacher/student output-space target geometry

## Frozen probe

Bounded eight-minute zero-fit analysis; no GPU, fitting, upload, network, model-parameter gradients, coefficient search, or writes outside this folder. Python runs through `scripts/limited` with `WNN_CPU_QUOTA=50%`, `WNN_MEM_MAX=2G`, and OMP/BLAS threads set to one because the Kaggle builder is active.

Read the temporal-increment KD and supervised metric-alignment proposals in [`../IDEAS_R1.md`](../IDEAS_R1.md). Compare cached train-only outputs for `l14_kd90_s3` and `tt29rep_fbc1k_s5` on 200 deterministically chosen, uniformly spaced valid sequence positions (`linspace(0, 1872, 200)`, integer indices). Verify prediction sidecars cover the official valid row groups and preserve row-group identity.

The proposed temporal loss is fixed as

`L_inc = sum[(e[s,t,k] - e[s,t-1,k])^2] / (200 * 19,999 * 2)`,

where `e = student_prediction - teacher_prediction`, using saved output values as-is. It has uniform weight on every adjacent pair and both targets, no label weighting or scoring mask, and resets at each sequence boundary. This is a descriptive operator probe; it does not select a training coefficient.

For context, also calculate (a) unweighted pointwise teacher MSE on raw cached outputs, (b) the trainer's α=.9 target form from `training/losses.py`: target `0.1*clip2(y) + 0.9*clip2(teacher)`, weighted by `abs(clip2(y))` over all valid sample rows, and (c) label WMSE toward `clip2(y)` with those same original label-magnitude weights. The target form uses label weights only here; since-based row weights are omitted from all probe losses. All loss gradients are with respect to un-clipped student output. The supervised metric loss is `-global_wp_clip2_v1`; its gradient uses the official scored mask, clip2 labels/predictions, and the scorer's `abs(clipped label)` weights.

Normalize each loss downhill vector to unit L2 norm. Report the official WP directional derivative along that vector and its cosine with the WP ascent gradient. Also report the cosine of both pointwise KD gradients with the supervised metric-loss gradient. Verify analytical WP and increment-loss gradients on a small deterministic CPU finite-difference fixture at two fixed step sizes; these are derivative checks, not candidate perturbation grades.

For each target, the score-gradient implementation is `0.5 * clip_indicator * w * ((y-mean_y)/sqrt(A*B) - rho*(q-mean_q)/B)`, with `q=clip2(p)`, `w=abs(clip2(y))*official_mask`, `A=sum(w*(y-mean_y)^2)`, and `B=sum(w*(q-mean_q)^2)`. The increment gradient is the reset-per-sequence `2*D.T@D@(p-teacher)/(200*19,999*2)`.

This is exposed-valid output-space geometry. It is not a network-parameter gradient, fit-quality comparison, fresh-CV result, or evidence of seed robustness. The sample and loss definitions are frozen; no loss weight or prediction coefficient is tuned.

## Artifacts

- [`target_geometry.py`](target_geometry.py) maps the fixed sequence sample, checks analytical gradients, and computes output-space derivatives.
- [`RESULTS_R1.json`](RESULTS_R1.json) records exact sample IDs, provenance hashes, loss definitions, finite-difference checks, and measured geometry.
- [`masked_wmse_diagnostic.py`](masked_wmse_diagnostic.py) computes the same-mask WMSE cosine without touching the original result JSON.
- [`MASKED_WMSE_DIAGNOSTIC_R1.json`](MASKED_WMSE_DIAGNOSTIC_R1.json) preserves the original all-row result reference and adds scored-coordinate alignment.

The official scorer is [`src/wnn/gmetric.py`](../../../../src/wnn/gmetric.py). Trainer loss semantics are read from [`src/wnn/training/losses.py`](../../../../src/wnn/training/losses.py) and [`src/wnn/nn.py`](../../../../src/wnn/nn.py). Root owns campaign indexing.

## Results and readout

The 200-sequence panel contains 398,203 scored rows (9.96% mask coverage). KD90 student WP is 0.6653085 on this sample (t0 0.6507084, t1 0.6799086); the independent reference matches exactly. Only 0.033% of scored student outputs are outside the official clip. These sample scores are context only.

| Loss downhill direction | Cosine with official-WP ascent | WP derivative per unit L2 output step |
|---|---:|---:|
| Uniform adjacent-increment KD | 0.002253 | +0.00000311 |
| Uniform pointwise teacher MSE | 0.006800 | +0.00000940 |
| α=.9 clipped label/teacher target, `abs(clip2(y))` weights | 0.284276 | +0.00039287 |
| Label WMSE, `abs(clip2(y))` weights | 0.327843 | +0.00045308 |

The explicitly requested cosine between pointwise KD and the supervised metric-loss gradient is +0.00680 for uniform teacher-only MSE and +0.28428 for the α=.9 label-weighted target form. Positive signs mean the loss downhill direction locally raises WP. Both analytic gradients passed the small finite-difference fixture: official-WP errors were at most 2.2e-6 at steps 0.002/0.001; increment-loss errors were below 9e-12 at steps 1e-5/2e-6.

The scored-coordinate WMSE cosine is **0.997901** (t0 0.998500, t1 0.997543); the scored-coordinate gradient norm is 0.328532 of the all-row norm. Their product reproduces the original **0.327843 all-row cosine**. Thus the lower all-row cosine is explained by support mismatch: official WP has zero gradient off-mask while WMSE includes all rows. The original all-row values remain unchanged in `RESULTS_R1.json`; the new same-mask calculation is in the supplemental JSON. This closes the card's zero-fit gate for adding a metric-alignment loss: these measurements do not justify that fit. The increment direction also has very weak pooled-WP alignment (0.002253 cosine), so it is not supported as an increment-only fit by this probe. The all-row pointwise KD cosines are likewise descriptive across mismatched supports and are not same-mask alignment evidence.

No fit is justified by either zero-fit direction; that does not falsify a network-parameter effect. This is a sampled output-space calculation, not a complete training gradient or quality experiment. Both cached models used model FP32 with BF16 Muon numerics, not fully strict all-math FP32. The separately running strict control (`sourcefix48776…`) was not used. All results describe the cached predictions as they exist.
