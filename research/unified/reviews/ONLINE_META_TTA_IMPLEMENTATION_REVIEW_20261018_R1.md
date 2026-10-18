# Independent implementation review: meta-trained online GRU adaptation

**Date:** 18 October 2026
**Hypothesis / experiment:** `H-ONLINE-META-TTA-GRU-B300-20261018-R1` /
`E-ONLINE-META-TTA-GRU-B300-20261018-R1`
**Review scope:** static review of the adapter, synthetic profile, and B300
experiment packet. Reviewer was the existing independent native subagent
`/root/online_meta_tta_critic`; its returned report did not include model
metadata.
**Initial verdict:** ITERATE.
**Evidence class:** code and recipe inspection only. No tests, GPU profile,
training, dataset reads, or quality evaluation were performed by the reviewer.

## Initial findings

The reviewer confirmed by inspection that the low-rank adapter math is
internally coherent, the next-input decoder consumes the adapted recurrent
state, the task query loss differentiates through `create_graph=True` updates,
and the 211-wide model input matches raw112 plus event99. `adapted_forward`
follows the canonical RNN normalization, encoder, recurrent, and head path.

The review required corrections in six areas:

1. The original profiler command unrolled 2,147 rows rather than the stated
   2,000-row TBPTT chunk, and its optimizer rate differed from the proposal.
2. The adapter caller's standardized-input requirement, finite-value policy,
   and autograd requirement were not explicit.
3. Training/inference integration had not defined how GRU, event, fast-weight,
   and partial-block state cross a TBPTT boundary.
4. Student distillation alpha did not specify the exact clipped teacher target
   or loss reduction.
5. Seed resampling did not explain the seven-seed F2/F3 cells; Holm family
   sizes named more teacher contrasts than the packet actually listed.
6. The initial card called the GRU encoder frozen while the current method made
   it outer-trainable.

## Author response recorded in the revised packet

- `profile_episode.py` now fixes the synthetic episode at 2,000 rows by default:
  99 support rows and 1,901 query rows, segmented as seven full 256-row blocks
  plus a final 109-row partial block. Its AdamW rate is `1e-4`, task-gradient
  clipping is `1.0`, and it records process peak RSS alongside device VRAM.
  Process RSS is explicitly not a cgroup/aggregate host-memory measurement;
  the profile remains synthetic and does not certify B300 capacity.
- The adapter method and packet require standardized raw112 inputs from the
  immutable first 112 model mean/std buffers, finite validation at data ingress,
  and enabled autograd for updates. No online scaler is introduced.
- The recipe preserves the 99/256 streaming schedule across exact 2,000-row
  graph boundaries: state values and partial blocks carry forward, graphs detach,
  and a split block is updated only after completion. This is a required
  trainer behavior and remains unwired.
- S1/S2 use `target=0.5*clip(y)+0.5*clip(teacher)` with existing
  `abs(clip(y))*main_valid` weighted-MSE reduction; S0 uses the same reduction
  with clipped labels. Bootstrap, reanchor, cache scaling, and family targets
  are disabled.
- The seed bootstrap now shares eight sampled seed identities across folds,
  renormalizes over available seeds in F2/F3, and redraws zero-weight folds.
  Holm covers exactly the three pairwise contrasts within each three-arm stage.
- The first-card wording is preserved with a dated addendum explaining that
  the GRU, task head, adapter initializer, and decoder are outer-trainable.

## Remaining gate

These edits repair the packet specification, not the missing implementation.
Before scientific fits, a separate independent closure review should verify
the edited source and exact command. The actor must still integrate the
cross-chunk state/update path into the modular trainer, bind source-qualified
fold and data manifests, exercise the actual producer and event-feature path,
profile the real two-chunk second-order update on the assigned B300 with
aggregate host/GPU memory, and wire checkpoint, prediction-cache, and grading
receipts. Until then, `launch_ready=false`; there is no score or leaderboard
impact claim.

## Static closure review R2

The same independent reviewer checked the requested profiler and distinction
corrections after revision. It verified that the synthetic raw112 tensors feed
the model's raw columns, the query loss uses the canonical clipped-label
weighted-MSE reduction, and the reported update diagnostics average across all
eight update calls. The README and PR now distinguish the proposal from PR
#14's fixed innovation features and describe the original card as preserved
with an addendum. No tests, profiles, fits, or data reads were performed.

**R2 verdict before publication:** ITERATE. The only new code/doc review point
was wording that treated eight B300 cards as available. The PR now makes the
parallel plan conditional on an actual team assignment and says card
availability/topology are unverified. The reviewer also required the corrected
packet to be committed before publication; commit and final committed-tree
check remain the next release gates. This is still a draft proposal, not a
launch authorization.

The earlier local pseudo-label gate screen is a different mechanism: it applied
a simple online gate to cached experts and had only about `+0.000048` WP versus
fixed 75/25 on one exposed 32-component panel. It does not test a model-specific
fast-weight update meta-trained against task loss. It is relevant caution, not
a direct falsification of this proposal.

## Links

- [Experiment packet](../../experiment_drafts/online_meta_tta_gru_b300/README.md)
- [Initial card](../../experiment_drafts/online_meta_tta_gru_b300/INITIAL_CARD.md)
- [Idea record](../ideas/ONLINE_META_TTA_GRU_B300_20261018_R1.md)
- [Literature source](../sources/ONLINE_META_TTA_GRU_B300_20261018_R1.md)
- [Causal pseudo-label gate result](../../campaign_2026-10-18/online_features_r1/causal_proxy_gate_r1/RESULTS_R1.md)
