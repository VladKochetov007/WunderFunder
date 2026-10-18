# Research-team experiment catalogue

- Published future-experiment draft PRs: [#22 Adaptive TBPTT](https://github.com/VladKochetov007/WunderFunder/pull/22), [#23 current-moment WP replay](https://github.com/VladKochetov007/WunderFunder/pull/23), and [#24 four-arm architecture handoff](https://github.com/VladKochetov007/WunderFunder/pull/24). Each targets research-team; the first two need adapters, the architecture handoff needs a focused port of preserved source. No model execution or run admission follows from publication.

The active [rank1 continuation handoff](https://github.com/VladKochetov007/WunderFunder/tree/experiment/rr18-01-labels/research/experiment_drafts/rr18-01-labels) is tied to the [October 18 execution packet](../campaign_2026-10-18/priority25_execution_r4/README.md); F0/S1001 and F1/S1001 are graded at audited source e35220969 (two of 30 paired cells); F3 has retained progress. See [the source-pinned review](../unified/reviews/REPOSITORY_DELTA_20261018.md) for uncertainty and the five newer team draft PRs. The [gradient-estimator continuations](../unified/ideas/GRADIENT_ESTIMATOR_CONTINUATIONS_20261018.md) add two unrun proposals without changing the active queue. These are separate from the historical fourteen-PR catalogue below.

The [October 18 base-sync receipt](BRANCH_SYNC_20261018_R3.json) records the no-rewrite merge of current `main` into `research-team`, including append-only conflict handling. The [October 17 receipt](BRANCH_SYNC_20261017_R2.json) remains the historical synchronization record for the original fourteen branches. No B300 quality result follows from publishing or synchronizing draft code. The [unified inventory](../unified/ideas/EXPERIMENT_DRAFT_INVENTORY_20261017_R1.md) records handoff scope and evidence limits.

- [Machine catalogue](CATALOG_20261017_R1.json) contains branch/source/base SHAs, exact PR links and publication status. It is navigation, not quality evidence.
- [Workflow](../unified/methods/EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md) defines30 paired cells per scientific arm, strictFP32, source-clean evaluation and separate logs/weights archives.

| Draft | Idea | Historical source SHA |
|---|---|---|
| [#1](https://github.com/VladKochetov007/WunderFunder/pull/1) | Causal Transformer with output autoregression | `484dde614cea` |
| [#2](https://github.com/VladKochetov007/WunderFunder/pull/2) | Cold GRU width versus depth at matched compute | `89582990815b` |
| [#3](https://github.com/VladKochetov007/WunderFunder/pull/3) | GRU versus Transformer teachers for the same GRU student | `75e0497b7e0c` |
| [#4](https://github.com/VladKochetov007/WunderFunder/pull/4) | Per-field ages versus seven group clocks | `72bf30823dc9` |
| [#5](https://github.com/VladKochetov007/WunderFunder/pull/5) | Function-preserving GRU widening | `8f95cc0e1974` |
| [#6](https://github.com/VladKochetov007/WunderFunder/pull/6) | Frozen-parent output anchor | `d354a0685b40` |
| [#7](https://github.com/VladKochetov007/WunderFunder/pull/7) | Learned causal initial-state family | `930ee793814d` |
| [#8](https://github.com/VladKochetov007/WunderFunder/pull/8) | Grouped feature encoding for causal Transformer | `5d4074c34199` |
| [#9](https://github.com/VladKochetov007/WunderFunder/pull/9) | Past-only smoothing while retaining raw inputs | `78ef7f8093f5` |
| [#10](https://github.com/VladKochetov007/WunderFunder/pull/10) | Trailing geometry and circle residual feature | `4e7d09de8f7f` |
| [#11](https://github.com/VladKochetov007/WunderFunder/pull/11) | Permitted Chronos or TimesFM teacher transfer | `e42ae7fbf743` |
| [#12](https://github.com/VladKochetov007/WunderFunder/pull/12) | Expand clean heterogeneous recurrent teacher cohort | `ec1e55bba63a` |
| [#13](https://github.com/VladKochetov007/WunderFunder/pull/13) | Fixed checkpoint trajectory averaging | `cc523fc5c1c3` |
| [#14](https://github.com/VladKochetov007/WunderFunder/pull/14) | Causal innovation residuals beyond moment features | `65f6fffdc559` |

## Readiness

These are experiment handoffs. Existing GRU mechanisms reuse linked trainer/model code; new Transformer/geometry/smoothing code is a starter. Every recipe is marked launch_ready=false: receiving researchers must bind fold-clean assets, finish trainer/reporting integration and profile actual B300 shapes. The plan entry point prints a matrix and does not train. No wall time, numerical parity or LB gain is claimed. No tests were added/run; static lint/format and document review only.

TPU parity remains a local engineering branch, outside these B300 quality drafts. Preserve its originating campaign evidence without launching it here. Shared primitive files appearing in related PRs have identical contents; merge them once and retain all append-only ledger/README entries when resolving conflicts.

## Packet included by this draft

- [Per-field ages versus seven group clocks](field-age/README.md) contains this draft's English proposal, recipe, code bindings and explicit missing training/profile work. `E-DRAFT-FIELD-AGE-20261017-R1`; no new fit.

## Modular trainer integration

All14 draft branches now include shared trainer revision `eb3309a322c1e24b3cc157eaefe3ca0e0290cd42` through ordinary merges. Each packet records its integration parent and links the [component map](../../src/wnn/training/README.md) and [engineering receipt](../unified/methods/MODULAR_RNN_TRAINER_REFACTOR_20261017_R1.md). The JSON catalogue retains original publication and pre-refactor heads alongside current integrated heads. This is source maintenance with no fits or measured performance result.

## Kalman causal smoother teachers and student-cache replacement — PR #17

[Draft PR #17](https://github.com/VladKochetov007/WunderFunder/pull/17) targets `research-team` from `experiment/kalman-teacher-gru-distill-b300`. It adds matched fresh GRU teacher arms (zero-padded raw control, steady-state EMA, fixed local trend, learned local trend) at four-channel and all112 scopes, then tests replacing one same-fold teacher family in a four-family GRU student cache. FP32 training, fixed checkpointing, actual full112 B300 profile, clean per-cell teacher registry, progress/gradient logging, separated artifact archives, paired folds/seeds and uncertainty are prescribed. Source hashes for both the Kalman module and imported GRU core are verified across training, prediction export and cache building.

A separate local fixed-GRU replay on 16 selected exposed-F0 groups found +0.000338 WP for local-trend posterior replacement, with a 12-component conditional interval [−0.000019,+0.000716]; an independent CPU grader reproduced it. This motivates testing the trained raw-retaining features but is not teacher/student evidence. No B300 profile, task fit, student grade or submission result exists. See the indexed [local result](../campaign_2026-10-18/online_features_r1/kalman_filter_probe/RESULTS_R2.md), [source review](../unified/reviews/KALMAN_B300_DRAFT_REVIEW_20261018_R1.md), and draft PR for runnable code and exact commands.
