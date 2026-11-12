# PR82 recurrent training follow-ups for the PR80 sequence receiver

## Candidate

This ports PR84 card 01 and PR82 card 03 into the exact PR80 sequence receiver as separate, source-pinned follow-ups. Each arm has seeds 203301 and 203302. Reuse the completed same-seed K16 AdamW controls as the comparator; do not refit them. The arms isolate optimizer choice and causal row weighting rather than combining them.

| Arm | Seeds | Fits | Comparator |
|---|---|---:|---|
| GRU256x2 + K16, recurrent Muon | 203301, 203302 | 2 | Existing completed same-seed AdamW K16 controls |
| GRU256x2 + K16, since-event row weighting | 203301, 203302 | 2 | Existing completed same-seed AdamW K16 controls |

Both arms retain hidden width 256, two GRU layers, causal K16 input stem, 32×1000 rows per optimizer update, E6 fixed endpoint, full official valid each epoch, strict FP32, LR 3e-4, weight decay .01, global clip 1, five-percent warmup and cosine-v2 floor .1. The same-seed AdamW controls are the historical fixed-E6 results WP .683019498 (203301) and .682629372 (203302); ROOT's canonical readout binds their model hashes.

## Optimizer contract

For Muon, every trainable `gru.weight_ih_l*` and `gru.weight_hh_l*` is a three-block parameter, one block per GRU gate. All other trainable parameters use AdamW, including the input encoder and output head. Both groups use the same peak LR and decay. The candidate uses Muon momentum .95, five Newton-Schulz steps, epsilon 1e-7, AdamW betas (.9,.999), and AdamW epsilon 1e-8. These AdamW coefficients match the existing K16 control; the PR84 CLI also uses beta2 .999 by default. Strict FP32 parameters, gradients, optimizer slots and orthogonalization are enforced by the receiver.

For since-event weighting, compute rows since the last causal i0 price event from each raw training input sequence. Multiply the main label-loss numerator's row weights by `max(.25, 1/(1 + age/32))`; preserve the original denominator and the official evaluation metric. The weighting is disabled for KD, hints, blending, pretraining, initialization, and non-GRU recipes. The full-validation report also emits WP on scored rows with age below 32 and at least 32, so both sides of the proposed mechanism can be read. No labels or future rows enter the weight calculation.

Every profile/run identity records each named optimizer group, the `torch.backends.cudnn.rnn.fp32_precision` getter when that PyTorch API exists (and records `unavailable` on older versions), and both `NVIDIA_TF32_OVERRIDE` and `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` process values.

PR84's reported development folds were exposed and used a different H512×1 setup; its gains motivate the optimizer transfer check but do not establish the result here. The since-event card reports a prior labels-only gain; this arm checks it independently against the exact K16 recipe. This validation set is also exposed development data, not a pristine holdout.

## Data and leakage contract

Use only the supplied official chronological split SHA-256 `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`: 10,325 earlier complete train sequences for fitting and train-only normalization; zero official-valid fit members; all 1,873 later complete valid sequences / 3,528,273 rows for evaluation. No teacher or initializer is used. Keep the supplied sequence memberships intact; never order by `seq_ix`. Validation is evaluated in full after every epoch and does not enter training or normalization ancestry.

## Source, configs and commands

`MANIFEST.json` binds the committed source and exact source identity to all four candidate config SHA-256s. The exact same-seed control configs are copied under [configs/controls](configs/controls) for inspection; their already-fitted weights remain the comparison. Candidate configs are under [configs](configs). Run the small source and provenance checks listed in the manifest before staging.

For each seed, profile with the exact source, config, data split and device shape before any fit:

```text
python scripts/train_sequence.py --config CONFIG --data-root DATA_ROOT --split-manifest SPLIT_MANIFEST --output PROFILE_OUTPUT --profile-updates 30
```

Fit only after ROOT approves the matching receipt; use the same command without `--profile-updates` and add `--profile-receipt APPROVED_PROFILE_RECEIPT`. The fixed campaign `CLOCK_R1`, cluster output cap, device allocation, profile admission, native paths, and all launches remain under ROOT's controller. The qualified strict-FP32 process environment is `NVIDIA_TF32_OVERRIDE=0`, `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`, and `PYTHONDONTWRITEBYTECODE=1`; capture the effective runtime values in every profile and fit receipt.

## Readout

For each arm, report both seed-level paired E6 deltas against the existing same-seed controls, pooled WP and each target's score. Preserve all epoch full-valid scores, train-eval trace, preclip and postclip/update telemetry, gradient clipping rate, LR/groups, nonfinite counters, updates/s, rows/s, GPU memory/utilization, host RAM, disk and prediction-artifact sizes. The PR82 follow-up signal is mean paired E6 gain of at least +.0005 WP with both seeds positive; use it as a prioritization rule, not a significance claim or automatic stop.
