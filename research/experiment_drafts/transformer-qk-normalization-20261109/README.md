# Affine-free QK RMS normalization on B300

Published [draft PR #75](https://github.com/VladKochetov007/WunderFunder/pull/75) to `research-team`, stacked on PR72 head
`4021de8ec5130143c207319c73a102a453eb911e`. Four scientific fits and eight
engineering profiles are prepared, all **UNRUN**. No local tests, model imports,
training, inference or GPU profiles are allowed for this preparation.

## Question and mechanism

Does an affine-free, fixed-scale normalization of attention queries and keys
improve the current standalone-label Transformer? Normalize only the head axis
of each newly projected Q and K in FP32:

`N(x) = x * rsqrt(mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-6)`

Cast back to the incoming attention dtype, then apply the existing RoPE. Store
normalized, rotated keys in the existing bounded cache; never normalize old keys
again. Values, causal masks, window length, feed-forward blocks and readout stay
as in the control. There are no new parameters, buffers or RNG draws. Omitting
`model.qk_norm` or setting `none` preserves the legacy config/state/arithmetic
path; `rms` enables the treatment. The GRU factory rejects the treatment.

The usual SDPA `1/sqrt(head_dim)` is unchanged. Ignoring epsilon and finite
precision, this produces `sqrt(head_dim) * cosine(q,k)`, so at head width 64 its
logit magnitude is bounded by 8. This is a mathematical property, **not evidence
of measured attention entropy, an observed collapse, or improved quality**.

This is a recipe-package test: norm control and attention temperature change
together. Under the default Linear initialization, projected component RMS is
approximately `1/sqrt(3)`, so unit-RMS normalization can make initial logit
standard deviation roughly three times larger. That approximation is not a
measurement of these models. Useful magnitude information can also be lost.
A positive or negative result alone does not isolate a saturation mechanism;
fixed initialization-scale or learned-temperature variants need their own
prospective comparison, not post-hoc tuning of this panel.

## Evidence and non-duplication

[Henry et al.](https://arxiv.org/abs/2010.04245) motivate normalizing headwise Q/K,
but use L2 normalization and a learned scale. [ViT-22B](https://arxiv.org/abs/2302.05442)
uses Q/K LayerNorm in a different large vision model. Neither establishes a gain
on this competition; this affine-free RMS variant reproduces neither recipe.

PR64 is attention recency, PR65 output gating, PR66 SwiGLU and PR74 causal input
filtering. This treatment changes Q/K vectors inside attention and is independently
based on PR72, not on the unmeasured PR74 stem. Existing GALAM QK-Norm mentions
are architectural antecedents, not a matched test in this canonical Transformer.

PR72's [six fixed-E4 grades](https://github.com/VladKochetov007/WunderFunder/pull/72#issuecomment-6731395515)
report plain Muon control WP **0.680881 / 0.682194**; AdamW and the longer gradient
horizon did not improve those two paired outcomes. The producer grades were
complete but independent checkpoint-byte audits were 0/6 in that report. We fit
both plain controls afresh, rather than reusing its scores as paired controls.

## Frozen panel

| Arm | Seed | Epochs / updates | GPUs |
|---|---:|---:|---:|
| plain | 203301 | 4 / 3,280 | 2 B300 |
| rms | 203301 | 4 / 3,280 | same pair |
| plain | 203302 | 4 / 3,280 | 2 B300 |
| rms | 203302 | 4 / 3,280 | same pair |

Four teacher fits, two per arm, zero student fits, zero reused controls. Total
16 model-epochs / 13,120 scientific updates. Each cell additionally needs two
50-update profiles: exact production warmup and warmup-zero peak-LR diagnostic,
eight profiles / 400 engineering updates total. No automatic queue or follow-up.

The configs copy PR72 C0 exactly except seed and the optional `model.qk_norm`:
Transformer768x24, H12, FF4, context1024; B128/rank, world2, global batch256;
chunk256, detached gradient horizon256, accumulation4; Muon LR3e-4, momentum.95,
NS5, WD.01, AdamW betas.9/.95, clip1; warmup5%=164 updates, cosine floor.1;
BF16/BF16x9 forward with FP32 master/loss/optimizer arithmetic; cache-free
`standalone_labels_wmse_v2`, `all_fit` normalization, `all_after_warmup` mask.
All four epoch checkpoints are retained diagnostically; primary is configured E4.

## Evaluation and decision

Use the unchanged PR49 chronological-schema source split SHA256
`0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`:
10,325 train sequences fit, zero valid fit, all1,873 valid sequences and3,528,273
scored rows evaluate under `global_wp_clip2_v1`. The supplied component split is
not proof of calendar chronology. Validation is repeatedly exposed development
data, not an untouched holdout, and no public-LB improvement can be inferred.

Report each seed's `rms - plain`, the descriptive mean and the 2,000-draw paired
whole-component bootstrap (seed271828). The bootstrap is conditional on fitted
predictors, not training-seed uncertainty. Mean delta at least+.0005 with both
seed deltas positive triggers a separate follow-up review only. Mixed/negative
results remain evidence for this exact fixed-scale package. An interval crossing
zero does not by itself close the normalization direction. No best-epoch/seed
selection or significance claim is permitted from this four-fit screen.

Optional secondary evaluation composes exactly90% frozen PR49 TL3 +10% each
candidate, requires all four blend endpoints and tunes no coefficients. It does
not turn these research Transformer weights into a CPU submission.

## Handoff and files

- [RUNBOOK.md](RUNBOOK.md): exact staged receiver commands.
- `run.sh`: data/check/profile/fit/evaluate/bundle/compare entrypoint; no allocator.
- `receiver_data.py`: unchanged source/split identity checks.
- `acceptance.py`: fail-closed test/profile receipt binding and manual approval.
- `profile_config.py`: separate warmup-zero engineering recipe.
- `check_qk_ddp.py`: prepared two-rank gradient equivalence with uneven and empty ranks.
- `compare_panel.py`: exact source/config/data/optimizer/checkpoint-byte comparison;
  all four cells must match the frozen recipes, paired GPUs and profile receipts.
- `recipe.json`, `configs/`: frozen panel and byte hashes.
- `PR.md`: eleven-section experiment contract.
- `SUBMISSION_STATUS.md`, `SUBMISSION_INPUTS.json`: live portal/HF and released
  checkpoint audit with exact available archive/endpoint hashes.
- `VALIDATION.json`, `INDEPENDENT_REVIEW.md`: static preparation evidence only.

Receiver scaffolding is adapted from PR74 implementation8a8afec964985a6ba8b13d1c9af1ab00e14c6504;
no trainer is copied. All numerical work uses canonical `train_sequence.py`,
`predict_sequence.py` and `score_sequence.py`. The full PR diff against
`research-team` includes the inherited PR72 runtime; review our incremental diff
from4021de8 separately. Memory, disk and wall time remain unmeasured for this
new treatment. Preserve all failures and the complete physical-disk run root.

Implementation commit: `67c351ae70cacef11f4ad485b5b98192f0d5e81c`. [Incremental code diff](https://github.com/VladKochetov007/WunderFunder/compare/4021de8ec5130143c207319c73a102a453eb911e...67c351ae70cacef11f4ad485b5b98192f0d5e81c).
[PUBLICATION.json](PUBLICATION.json) records publication and initial CI success.
No merge, model execution or new platform upload occurred.
