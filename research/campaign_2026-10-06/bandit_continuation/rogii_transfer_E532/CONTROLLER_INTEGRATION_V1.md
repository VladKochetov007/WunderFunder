# Grok ROGII transfer: controller integration

Grok 4.7 delivered eight final public cards independently. [Immutable provider proposals](GROK_4_7_PROPOSALS_V1_20261008.md) and its provenance receipt are the source, not evidence of model improvement. Freebuff fallback is generating separately; do not feed it these cards before its own initial proposals are committed. Muse access failures are provider evidence only.

## Mapping and priority

| Source card | Research record | Discriminating action / scope |
|---|---|---|
| R01 soft value-grid loss | H253 | Freeze separate per-target grids from fitting labels; matched point-loss backbone/control; kernel smoothing and expectation auxiliary loss explicit. Small finite-gradient probe before full teacher fit. |
| R02 causal window shape packet | Existing H249 extension | Add polynomial slope/curvature/residual shape to that card's candidates, with channel/window selection train-only. Avoid a duplicate arm; include mean/variance controls only when actually implemented. |
| R03 rare-jump belief | H254 | Deterministic reset-safe histogram, sticky/wide transition mixture, only completed-bin updates applied forward. Compare likelihood-only, belief readout and full refit/blend endpoints. |
| R04 reliability-gated blend | H255 interaction | Instantiate prior H205 gating on materially new H249/H254 endpoints, a causal reliability statistic and fixed-mixture control. Do not reopen the closed E514 pool unchanged. |
| R05 affine jitter/joint schedule | H256 | Shared versus grouped input nuisance corruption, matched clean model and equal update budget; no invented geometric invariance. Joint/two-stage is a separately frozen child. |
| R06 fixed causal pooling | H257 | New TCN resource/representation child of H217: completed-bin pooling/hold-last versus learned causal resampling, same input alignment/receptive field. |
| R07 auxiliary corruption | H258 | Distinct anti-shortcut regularization, causal lag/mask/noise of a declared available feature block. Clean-input endpoint plus missing-block diagnostic. |
| R08 prefix-bin calibration | H259 | Train-only prior plus earlier-row-only within-sequence bins, explicit bin key/occupancy/reset. Compare raw, fixed prior and adaptive feature channels. |

Allocation: retain current healthy H230 control and TCN/GRU cloud delivery. R02/H249 streaming reference is the cheapest new independent CPU builder; R01/H253 is a distinct objective/head actor candidate after a finite small-shape probe; R03/H254 remains a protected exploration mechanism. H255 waits for a new valid prediction cohort. Their possible teacher/student interaction value is distinct from standalone ranking. All priors/public forecasts remain null. Priority is qualitative, not a fabricated numerical bandit reward.

## Required corrections before executable recipes

1. Standard `GroupNorm(B,C,T)` reduces across time, and LayerNorm can too if time is in its normalized shape. They are not inherently causal. R06 needs channel-only per-row normalization, frozen fitting-fold statistics, or audited causal prefix updates. Offline-prefix/streaming parity is mandatory. Evaluation must not recompute batch/whole-sequence statistics from unseen suffixes.
2. R01 controls must preserve or explicitly vary the current training weighting policy. Official evaluation clipping/weights remain fixed, but permitted training weighting is not forbidden. Grid edges/kernel widths cannot use reported validation labels. Raw versus clipped training target is separate from the head treatment.
3. R03 repeatedly treating a base model's density as independent observation can double-count strongly correlated rows. Freeze likelihood temperature/update timing and compare base-density summaries without filtering. Backward smoothing is not a serving operation; any privileged training teacher requires separate ancestry and held-out evaluation of a causal student.
4. Anonymous-channel affine invariance is unknown. R05 is a robustness corruption hypothesis, not a proven label-preserving augmentation. Do not independently perturb reset/sentinel channels or named structural indicators without a declared semantic audit. Preserve original raw treatment in a matched control.
5. Current distillation caches are training targets, not automatically serving-available teacher input channels. R07 must declare an actually causal inference-time auxiliary block and account for its cost, or move that block to an explicit training-only teacher/student design. Corrupting a target cache is a different hypothesis. R08 prior/bin updates use earlier features only and reset completely.
6. A first window's lack of gain does not close every window, teacher-only usage or interaction. Grok's R02/R06 phrasing about follow-ups only after a winner is an allocation suggestion, not a universal veto. Freeze bounded descendants when concrete expected information or conditional value warrants them; zero-crossing intervals alone do not reject arms.

These are controller recipe constraints, not independent empirical validation. A recipe auditor checks the actual implementation before fits, and an independent grader owns outcomes. No copied geology code, assumed conservation law, claimed public replication or model fit appears in this integration.
