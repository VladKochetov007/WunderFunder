# Current research and complete experiment archive

Updated 1 October 2026 (Moscow). This is an index to the evidence; historical results remain in their original ledgers and journals.

## Best recorded submission

**`candV1_ft2.zip`: public WP 0.6641, reported runtime 41m42s.** This is C's four-member ensemble with two epochs of head-only fine-tuning: distilled GRU256, GRU128 seed 3, GRU128 seed 4, and chunk-500 GRU128, equal weights, native C++ fp16 inference. The public result is recorded in the [September 29–30 journal](quant-probes-2026-09-29.md); it was not independently fetched from the leaderboard during this audit.

| Candidate | Reported public WP | Reported runtime |
|---|---:|---|
| A: two GRU256 members | 0.6629 | ~35m; governor dropped a member |
| B: GRU256 + two GRU128 | 0.6639 | ~43m |
| C: distilled GRU256 + two GRU128 + chunk-500 GRU128 | 0.6640 | 48m52s |
| V1: C with two-epoch head FT | **0.6641** | 41m42s |
| V2: validation repeated 3x in parent training | 0.6611 | 41m42s |
| V3: parents trained on train only, validation in head FT | 0.6629 | 49m26s |

The 0.0001 C→V1 difference is a recorded leaderboard difference, not established statistical superiority. Runtime varied substantially between submissions. V1 SHA256: `ea494921497d9f3cf53b19b1b68285133983560a249b38ef40738b3d846d319d`.

## Everything preserved on GitHub

- [Full snapshot and restoration instructions](https://github.com/VladKochetov007/WunderFunder/tree/archive/2026-10-01/snapshot/2026-10-01): 266 run directories, including five empty failed/aborted runs, 860 checkpoints, histories, logs, source snapshots and dirty patches.
- [All ready-to-use submission ZIPs](https://github.com/VladKochetov007/WunderFunder/tree/archive/2026-10-01/snapshot/2026-10-01/submissions), including the exact best V1 package.
- [Searchable run index](https://github.com/VladKochetov007/WunderFunder/blob/archive/2026-10-01/snapshot/2026-10-01/run_index.tsv) and [readable research history](https://github.com/VladKochetov007/WunderFunder/tree/archive/2026-10-01/snapshot/2026-10-01/research-history).
- [`quant-probes`](https://github.com/VladKochetov007/WunderFunder/tree/quant-probes) contains recent model research; [`cpp-engine`](https://github.com/VladKochetov007/WunderFunder/tree/cpp-engine) contains the native submission engine.

The asset snapshot cutoff is 2026-09-30 21:08 UTC (1 October 00:08 Moscow). The existing Transformer fold-1 training was still active, so its snapshot includes intermediate checkpoints, not a completed result. The ongoing job was left untouched. Bulk caches (~94 GB) and organizer dataset bodies (~33 GB) have inventories, not uploaded copies; checkpoints and experimental outputs are included. Original server files are retained.

## Research map

The [experiment ledger](experiments.jsonl), [hypothesis ledger](hypotheses.jsonl), [individual run reports](runs/INDEX.md), and [recent journal](quant-probes-2026-09-29.md) preserve exact historical protocols and results. Ledger entries are not one-to-one with model directories: one experiment may use multiple seeds/folds, and analysis-only experiments may have no trained model.

| Direction | Evidence and current interpretation |
|---|---|
| EMA training, head fine-tuning, complementary seeds/chunk-500 member, ordinary distillation | Components of the current recipe; their marginal value depends on the ensemble and evaluation protocol. |
| More width, longer training, extra seeds and larger ensembles | Small or inconsistent marginal gains. The journal's 22-member head-FT ensemble scores 0.70277 versus C's local analogue 0.70242 on reused pure folds 0/1. |
| Privileged teachers, second-generation distillation, class-dependent blend weights | No convincing gain over ordinary distillation inside C in the tested settings. Privileged teacher accuracy does not establish causal student improvement. |
| Event attention, two-stream GRU, class heads, distributional/atom heads | No useful contribution to the tested deployed blend; see individual controls and folds in the journal. |
| Decoded price levels, discrete class-3 tables, residual corrections and raw additional-feature probes | Tested variants did not yield consistent cross-fold gains. This does not establish a general information-theoretic ceiling. |
| Level augmentation/drop, sequence V-REx, periodic inference resets, excessive validation upweighting | Negative in the tested protocols. Training-only random-origin augmentation would be a different, still untested intervention. |
| Matured-label feedback | An apparent large gain used labels before their actual maturity. The corrected quote-event-clock version was null; the invalid result remains documented. |
| Causal segment-recurrent Transformer | Existing teacher experiment; fold 1 still running at snapshot. No new GPU run launched by this iteration. |
| Runtime governor | Heterogeneous model costs are approximated as equal in current policy. A cost-model discrepancy alone does not prove a score/runtime improvement. |

## Current creative-research iteration

The [registered iteration](iterations/2026-09-30-creative/) fixes three methodological problems: separate reused-fold discovery from confirmation and training-seed variation from chain uncertainty; optimize the user's competition objective instead of imposing novelty; replace categorical role vetoes with empirical comparisons and leakage checks.

M01 re-evaluated existing matched width-256/128 head-FT runs for seeds 3/4/5, fixed epochs, on 860 pure sequences in 250 chains. Candidate-minus-control deltas were **+0.000073, −0.000339, +0.000792**; mean **+0.000175**, sample SD **0.000572**. The three-seed mean-prediction ensemble improved by **+0.000696**, a different estimand. These results do not establish a stable positive per-model width effect. See the [immutable result and input hashes](iterations/2026-09-30-creative/M01-paired-seeds.result.json).

All validation folds have prior exposure. A pure chain excludes a training neighbour; it is not a fresh holdout. Chain bootstrap intervals condition on the fitted models, do not measure optimizer variation, and do not correct adaptive selection or hidden-test shift. Read recent pooled-WP comparisons separately from older block-averaged reports; historical scorers and definitions are retained with the source snapshots. The September 27 report records a live-FAQ confirmation of pooled scoring, but this iteration could not independently recover that primary-source confirmation. Imported competition documentation and older scorers still describe block averaging; the new audit verifies its declared pooled calculation rather than an independently reverified official metric.
