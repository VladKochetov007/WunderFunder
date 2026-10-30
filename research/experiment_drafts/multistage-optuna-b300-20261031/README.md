# Bounded multistage search on eight B300 GPUs

The user explicitly authorized this execution on31October2026. The objective is
an independently evaluated, diverse regression ensemble and improved students,
compared with the frozen fresh TL3 reference (official-valid WP0.6923774245652725).
The existing90%TL3/10%GALAM fixed blend scored0.6924964468441978 on development-
exposed official valid. These are historical reference scores, not new results or
public-score predictions.

[CONTRACT.json](CONTRACT.json) bounds the first wave to48 scientific fit attempts,
including fitting ancestors and final refits, and six node wall hours from the
first actual GPU profile. The last20minutes are reserved for collection. A fit
has a two-hour maximum. Short exact-identity engineering profiles are counted
separately, at most56 and180seconds each, inside the same campaign deadline.

## Scientific stages

1. Fit diverse base regressors. The first runnable controls are GRU256x2 and
   LSTM256x2; the R3 batch16/32/64 ladder measures one-GPU throughput. An engineering
   profile is not a model-quality experiment. GALAM, TCN and causal Transformer
   remain candidate families and are admitted only when their actual adapters,
   monitored recipes and device profiles pass review.
2. Compare frozen base endpoints with fresh-optimizer, lower-LR fine-tuning.
   Initializer membership and source/weights identities must be verified.
3. Compare matched label-only students and fitting-only teacher-pool distillation.
   Teacher predictions and their actual clipping/weighting arithmetic are pinned.
4. Optimize nonnegative ensemble weights on qualified fitting-only predictions
   using CPU Optuna. Freeze finalists before full official-valid reporting.

Stage1 conditional TPE starts from known controls rather than a Cartesian product.
Later stages use the observed fitting-only evidence. Missing adapters or stages
are preparation dependencies, not claims of running jobs. Every attempt, including
failed, censored and pruned configurations, remains in the durable ledger.

## Validation and data

The official chronological primary boundary stays frozen:10325 fitting train
sequences, zero fitting valid sequences, and all1873 official-valid sequences
(3,528,273 scored rows).282 train sequences connected to validation are excluded.
[ADR0044](../../../specs/adr/0044-chronological-primary-validation.md) is binding.
The original split SHA is
`0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`.

Optuna and learned weights use an additional whole-component holdout drawn only
from eligible official train. Its order is not chronological: real timestamps
are unavailable, and seq_ix is shuffled. Train lacks is_scored; its
all-after-warmup WP is explicitly an auxiliary proxy. Official valid never enters
fitting, normalizers, teacher/initializer ancestry or weight optimization. Existing
full-train TL3 cannot be an inner-pure learned-mixture ancestor; rebuilding it
costs additional fits or it stays only a frozen primary reference.

The [original split](receiving/ORIGINAL_PRIMARY_SPLIT.json),
[portable source registry](receiving/PORTABLE_DATA_R1.json) and
[metadata transport](receiving/INPUT_TRANSPORT_R1.json) bind receiving inputs.
B300 raw train and valid full-file SHA256 were independently matched to the
registry generated on the original L4 data. No raw file timestamps are altered.

## Operation and monitoring

Project directory on the owned account is
`/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1`.
Use the persistent SSH connection. NIER jobs and the existing MPS server are
unrelated and remain untouched. Detach campaign workers with exact PID/birth/PGID
ownership and bounded cleanup; stopping the campaign must never signal foreign
processes. Initial workers have four CPU threads and an8GiB host-loader envelope.

The [independent review](INDEPENDENT_REVIEW_R1.json) and
[engineering policy](PROFILE_REVIEW_POLICY_R1.json) distinguish preparation,
profile approval and scientific execution. Immutable source capsules preserve
native identity while later orchestration code evolves.

Save raw native journals and PNG/SVG at every completed evaluation and stage end:
train/control WMSE, train/control WP and per-target WP, objective terms, LR,
preclip gradients/clipping/nonfinite counts, throughput and resources. Plot the
official-valid report separately with its scored mask. Keep scheduled,
inner-selected and best-observed checkpoint roles explicit. See the standing
[stage monitoring contract](../../../specs/stage-monitoring.md).

Current status must be read from actual receiving/launch/terminal receipts. This
README alone does not assert that a GPU job started or that an experiment won.
