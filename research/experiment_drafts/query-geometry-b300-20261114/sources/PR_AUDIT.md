# Duplicate and mechanism audit — 2026-11-14

This is a bounded read-only source audit for the new query-geometry screen.
No local tests, model imports, training, inference or profiling were performed.
The design is prepared for six new Transformer fits; this note supplies no
numerical acceptance or measured gain.

## Sources checked

| Source | Inspected revision | Relation to this screen |
| --- | --- | --- |
| [PR99](https://github.com/VladKochetov007/WunderFunder/pull/99) | `320ebd6ca38cf5cd16a708dd1a088c5f06f23e48` | i1 stale updates and historical-key dropout with matched independent masks. No query rotations or gains. Its controls are not reused. |
| [PR81](https://github.com/VladKochetov007/WunderFunder/pull/81) | `650c21e19cdaef4fd9605311098eab45f5316e40` | Cards01–07 change Transformer representation/objectives/depth; card08 is a noisy GRU-state consistency fork. The latter preserves clean carried state and adds an auxiliary objective, unlike this direct train-time Q transformation. |
| [PR82](https://github.com/VladKochetov007/WunderFunder/pull/82) | `c0dc2e7b85847d45e088e6090cd1bffad5fdf549` | Card08 already proposes lognormal per-sequence scaling of raw volume/delta-volume inputs. That input-scale hypothesis is excluded here. |
| [PR83](https://github.com/VladKochetov007/WunderFunder/pull/83) | `9f01bb0eb9f104f7788ec02aa11d94d303d390de` | Published column-mask negative screen: standalone around −0.0014 and fixed pair contribution around +0.00025 in the documented settings. No independent reproduction here. This discourages another unqualified masking proposal; it does not refute all activation regularization. |
| [PR92](https://github.com/VladKochetov007/WunderFunder/pull/92) | `b107418aef553cb4aecc249ad9cf2c86553d5a1d` | B09 already proposes SAM/weight perturbation. B03 is since×distillation. Neither is repeated. |
| [PR93](https://github.com/VladKochetov007/WunderFunder/pull/93) | `68e5b27066c5bc6d34a1e34f4c8b00593a16333b` | Event clocks, class heads, residual boosting, pair encoders, TensorSketch and delayed-label ceiling. No concrete Q-angle/gain panel found. |
| [PR98](https://github.com/VladKochetov007/WunderFunder/pull/98) | **`a8bf1734626221660e650325624195350ae66305`** | Refreshed after PR99 publication: strict KD pair, teacher-pool ancestry and result follow-up. Changes since `fcde99f1` are evidence/KD target work, not new query perturbations. |
| H-173 in [the PR93 ledger](https://github.com/VladKochetov007/WunderFunder/blob/68e5b27066c5bc6d34a1e34f4c8b00593a16333b/research/hypotheses.jsonl) | same PR93 pin | Nuisance-coordinate style augmentation / MixStyle was already proposed. A renamed style-augmentation arm would not satisfy the user's nonduplication request. |

Also searched the available PR93 experiment/idea/ledger tree for stochastic
depth, DropPath/DropHead, rotary noise, phase jitter, query noise/jitter and
norm-preserving perturbations. No exact pairwise Q-rotation versus moment-matched
Q-gain experiment was found. This is evidence about the inspected team material,
not a claim of global academic novelty or an exhaustive review of every historical
commit. Existing generic augmentation/noise proposals remain prior art.

## Why this is a different experiment

The intervention is applied to **Q after the native RoPE transformation and before
the original SDPA call**, only in training. It removes no input rows, features,
keys or heads. It changes no target, loss term, optimizer, source membership,
attention mask or stored K/V tensor. The same native causal attention engine
and fixed E4 endpoint remain the receiver contract.

The code names `angular` and `radial` refer to each adjacent two-dimensional
rotary pair. In user-facing interpretation these mean **pairwise rotation** and
**pairwise gain**. For q=(a,b), Jq=(-b,a), independent symmetric sign s and a
predeclared angle θ, the intended formulas are:

    rotation: q' = cos(θ) q + s sin(θ) Jq
    gain:     q' = (cos(θ) + s sin(θ)) q

For a fixed input pair both have:

    E[q']                 = cos(θ) q
    E[||q'||²]            = ||q||²
    E[||q' - q||²]        = 2 (1 - cos(θ)) ||q||²

The rotation preserves each pair norm in exact arithmetic; finite BF16 output
need only satisfy a declared rounding tolerance. These identities are algebraic
design properties, not observed model results. At θ=0.35, the gain factors are
approximately0.5965 and1.2823; the common expected-Q contraction is about0.93937.

## Important limits and falsifiers

- Independent gains across rotary pairs generally change the direction of the
  whole64D head. The radial arm preserves direction **within a pair**, not the
  whole query. Do not call this a clean whole-vector angle-versus-magnitude split.
- Equal expected representation energy does not match logit variance, softmax
  entropy, attention mass or higher moments. Those changes can explain a score
  difference. The intervention tests these frozen geometric policies, not an
  isolated universal cause of generalization.
- Both treatment arms contract expected Q relative to clean. The gain comparator
  shares that contraction; a rotation-versus-clean result alone cannot isolate
  rotation from generic noise or effective-temperature changes.
- This is not RoPE time dilation, physical time augmentation or a proven label
  symmetry. Post-RoPE query pairs combine content and position information.
- Direct K/V mutation is forbidden. Deeper-layer representations and later
  caches may still differ indirectly after a perturbed lower-layer attention
  read. Such propagation must not be described as an unchanged whole-model cache.
- Shared signs over query rows of a chunk are one frozen stochastic law. It
  need not give identical stochastic predictions under a different chunking
  policy; the registered chunk/horizon stays fixed.
- A useful rotation−clean gain supports follow-up. Rotation−gain addresses
  whether rotation earns its place beyond the matched gain policy. Rotation
  beating gain while both lose to clean is not a deployment-quality success.
- Two seeds on the same repeatedly exposed official valid are an exploratory
  screen. Preserve both deltas, fixed endpoints, per-target scores and negative
  outcomes. Bootstrap component sensitivity cannot establish training-seed
  significance or new public/private transfer.

## Primary-paper boundary

[RoFormer / RoPE](https://arxiv.org/abs/2104.09864), checked on2026-11-14, describes
rotation matrices for position representation and relative-position dependence.
It grounds the coordinate convention; it does not propose or validate our
pairwise perturbation panel. [Scheduled DropHead](https://arxiv.org/abs/2004.13342)
is an example of a different structural attention regularizer: it removes heads,
which this screen does not do. Neither paper supplies a market-score forecast.
No third-party code or weights are copied from these sources.

The implementation must still pass independent source review, server numerical
probes and one real-shape profile per cell. Preparation is not execution.
