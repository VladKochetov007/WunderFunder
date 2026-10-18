# Meta-trained causal online adaptation for a GRU teacher

**IDs:** `H-ONLINE-META-TTA-GRU-B300-20261018-R1` /
`E-ONLINE-META-TTA-GRU-B300-20261018-R1`
**Status:** implementation proposal plus synthetic profile only. No B300
hardware profile, task-data fit, student transfer, or score has been measured.

## Hypothesis

A GRU could learn when to benefit from an input-only online update if a
self-supervised next-observed-input loss updates a small per-sequence adapter
and an outer task loss meta-trains the adapter initialization. The competition
callback does not provide targets or scoring masks, so this is a causal
alternative to label-feedback adaptation. The useful result must transfer to
a regular GRU student; a teacher-only gain is not a submission gain.

The mechanism is distinct from [causal innovation features, PR #14](https://github.com/VladKochetov007/WunderFunder/pull/14),
the [Kalman smoother teacher draft, PR #17](https://github.com/VladKochetov007/WunderFunder/pull/17),
and the [static next-input auxiliary draft, PR #18](https://github.com/VladKochetov007/WunderFunder/pull/18).
PR #14 adds fixed causal residual features, PR #17 changes the representation
with a statistical state estimate, and PR #18 trains a static predictive-state
GRU. This idea updates per-sequence fast weights online and meta-optimizes those
updates against a later target task. Its closest parent is PR #18's auxiliary
loss; the interaction is explicit.

## Frozen proposed mechanism

- Input is the canonical raw112 plus causal event99 feature set (211 columns).
- A rank-16 residual adapter transforms the recurrent state and is shared by
  the task readout and next-input decoder. The decoder predicts standardized
  raw112 at the next observed row, giving the input loss a gradient to the
  fast weights.
- One fixed SGD update follows each completed support block: LR `1e-3`,
  per-sequence gradient clip norm `1.0`, proximal coefficient `0.01`.
- Rows0–98 provide the first support; before prediction at row99, use only
  transitions `h[t] -> x[t+1]` for `t=0..97`. Then predict each 256-row block
  before adapting from that block's internal observed transitions. Exclude the
  pair that crosses into the next block. Carry fast weights only within the
  sequence and reset all state on its boundary.
- Training meta-gradients differentiate through updates within the current
  2,000-row TBPTT chunk, then detach at the chunk boundary while carrying fast
  values. The GRU, task head, adapter initializer, and decoder are
  outer-trainable; only per-sequence fast weights change in the inner step.
  This exact training loop is not yet wired into the modular trainer.
- Compare cold H512×1 supervised, static-auxiliary, and meta-TTA teachers;
  then compare cold H256×2 supervised, static-teacher-distilled, and
  adaptive-teacher-distilled students. Use30 paired fold/seed cells per arm,
  strict FP32, source-clean ancestry, and unchanged `global_wp_clip2_v1`.

The B300 packet and starter code are in the
[experiment draft](../../experiment_drafts/online_meta_tta_gru_b300/README.md).
Its code readiness is limited to the low-rank adapter and a no-data synthetic
profile. Trainer integration, real data/preprocessing bindings, clean teacher
cache construction, actual-device profile, and reporting/grade commands remain
required; `launch_ready=false`.

## Evidence and limits

The initial card was committed before independent criticism. Critique agreed
that this is distinct from static auxiliary training and required exact causal
block timing, correct adapter-to-decoder gradient flow, second-order profiling,
fold-clean meta-training and a three-way student transfer comparison. The
packet was revised accordingly. There is no local quality result and no
leaderboard forecast.

The first independent implementation review returned ITERATE, identifying six
specification gaps; exact corrections and remaining integration gates are
tracked in the [review note](../reviews/ONLINE_META_TTA_IMPLEMENTATION_REVIEW_20261018_R1.md).
The packet distinguishes the earlier weak cached-expert pseudo-label gate from
this untested model-specific fast-weight update meta-trained against task loss.
The synthetic profile is an engineering probe only: random task labels are
independent of inputs, and the synthetic stream omits the production event
builder and cross-TBPTT block buffering.

The [source record](../sources/ONLINE_META_TTA_GRU_B300_20261018_R1.md)
summarizes primary literature and transfer limits. Existing time-series TTA
results are mixed; no cited paper validates this mechanism on the competition
data. A weak early result or interval crossing zero does not automatically
reject the broader online-learning family. If later evidence supports the
direction, test calibrated update magnitude, input objectives, and the
interaction with Kalman or innovation representations as separate children.
