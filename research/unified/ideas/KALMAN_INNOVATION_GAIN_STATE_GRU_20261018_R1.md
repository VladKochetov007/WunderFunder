---
kind: idea
status: proposed
created_utc: 2026-10-18T08:18:33Z
updated_utc: 2026-10-18T08:30:52Z
hypothesis_ids:
  - H-KALMAN-INNOVATION-GAIN-GRU-20261018-R1
experiment_ids:
  - E-KALMAN-INNOVATION-GAIN-GRU-B300-20261018-R1
evidence_paths:
  - ../../campaign_2026-10-18/online_features_r1/CLOSEOUT_20261018_R1.md
  - ../../../../research_tmp/kalman_innovation_gain_gru_b300/README.md
---

# Innovation-conditioned latent gain inside a GRU

## Question

Can a causal, task-trained latent prediction-correction operator make a GRU teacher represent state changes more usefully, and can a standard GRU student retain that information through the existing teacher-cache route?

## Mechanism

For encoded input `z_t`, the observer forms a latent measurement `o_t`, predicts a local-linear prior `m^-_t` from the previous two encoded rows, computes innovation `nu_t = o_t - m^-_t`, and estimates a bounded diagonal gain:

```text
m^-_t = o_(t-1) + sigmoid(a) * (o_(t-1) - o_(t-2))
nu_t  = o_t - m^-_t
K_t   = sigmoid(b + gain_net([o_t, nu_t]))
m_t   = m^-_t + K_t * nu_t
```

The posterior and gain-weighted innovation enter zero-initialized residual branches connected to the ordinary fused GRU input and readout. State carries across trainer chunks and resets at sequence boundaries. The model weights remain fixed at inference; this is causal state adaptation, not test-time weight learning or an online-label update.

## Why it might help

The neighboring fixed output filters hurt, the additive 39-state groupwise ridge encoding lost on a nonofficial train-row proxy, and a two-seed recurrent append screen changed sign around zero. A small fixed local-trend input replay was positive but uncertain. These results weaken fixed transforms and naive feature appends while leaving open whether a learned gain used conditionally inside a recurrent update contributes something the GRU gates do not learn naturally. They are not evidence that this observer will help.

The final competition value must be a better served standard-GRU student under the same fold, cache, target scale, coefficient, seed and training budget. A teacher-only gain is insufficient if its representation does not transfer.

## Frozen comparison proposed before the stop

At H512×2, compare scratch-trained ordinary GRU, context-only observer, and innovation-conditioned observer on paired source-qualified fold/seed cells. The context-only arm masks innovation only at the gain-network input; both observer arms still apply gain-times-innovation correction. It is therefore a narrow conditioning ablation, not a no-innovation observer. The standard-GRU student comparison would use matched existing, context-teacher-replacement and innovation-teacher-replacement caches.

The proposal specified 30 paired cells, four previously exposed folds, and fold/seed/component reporting. These folds are not fresh holdouts. The detailed run matrix, FP32 recipe and logging plan live in the local draft identified below.

## Current status and blockers

Implementation exists only in the isolated local worktree `/home/vlad/development/wunder/research_tmp/kalman_innovation_gain_gru_b300`, branch `experiment/kalman-innovation-gain-gru-b300`, commit `527d2e985`. It was not pushed; no pull request exists. No fit, student cache, B300 profile, synthetic invariant check, score, or submission exists. The local read-only implementation review found no causal/chunk-state defect by inspection but identified three material prefit blockers:

1. Synthetic prefix-invariance, chunk/full-sequence, reset and batch-isolation checks have not run.
2. The actual H512 B300 profile has not run; eight B300 cards are an intended resource target, not measured availability.
3. The per-cell clean teacher-source registry is null, so the student-cache experiment cannot be launched faithfully.

The user stopped new research admission on 2026-10-18. Preserve this as an untested local code draft. Do not treat the stale branch-side PR #22 text as publication evidence; no PR #22 was created. The larger published handoffs already on record are [#14](https://github.com/VladKochetov007/WunderFunder/pull/14), [#17](https://github.com/VladKochetov007/WunderFunder/pull/17), and [#18](https://github.com/VladKochetov007/WunderFunder/pull/18).

## Sources and evidence

- [KalmanNet-inspired source assessment](../sources/KALMANNET_INNOVATION_GAIN_GRU_20261018_R1.md): primary literature motivates the prediction-correction operator only; it does not validate transfer to this task.
- [Independent static draft review](../reviews/KALMAN_INNOVATION_GAIN_GRU_DRAFT_REVIEW_20261018_R1.md): no execution or quality result.
- [Online-feature closeout](../../campaign_2026-10-18/online_features_r1/CLOSEOUT_20261018_R1.md): neighboring measured results, exposure and submission status.
