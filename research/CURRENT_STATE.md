# Current research and complete experiment archive

Updated 1 October 2026 (Moscow). This is an index to the evidence; historical results remain in their original ledgers and journals.

**Publication preference:** the user has paused GitHub uploads. Subsequent research is stored locally and on beleriand only until the user explicitly resumes publication. Existing archive links below are historical; do not push new progress automatically.

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

- [Full snapshot and restoration instructions](https://github.com/VladKochetov007/WunderFunder/tree/archive/2026-10-01/snapshot/2026-10-01): 268 run directories and 865 checkpoint paths across the base snapshot and late supplement, including five empty failed/aborted runs, histories, logs, source snapshots and dirty patches.
- [All ready-to-use submission ZIPs](https://github.com/VladKochetov007/WunderFunder/tree/archive/2026-10-01/snapshot/2026-10-01/submissions), including the exact best V1 package.
- [Searchable run index](https://github.com/VladKochetov007/WunderFunder/blob/archive/2026-10-01/snapshot/2026-10-01/run_index.tsv) and [readable research history](https://github.com/VladKochetov007/WunderFunder/tree/archive/2026-10-01/snapshot/2026-10-01/research-history).
- [`quant-probes`](https://github.com/VladKochetov007/WunderFunder/tree/quant-probes) contains recent model research; [`cpp-engine`](https://github.com/VladKochetov007/WunderFunder/tree/cpp-engine) contains the native submission engine.

The base asset snapshot cutoff is 2026-09-30 21:08 UTC (1 October 00:08 Moscow). The existing Transformer fold-1 pipeline then completed at 21:23:53 UTC; a [late supplement](https://github.com/VladKochetov007/WunderFunder/tree/archive/2026-10-01/snapshot/2026-10-01/delta-late) copied at 21:25:53 UTC preserves its final parent, head-only and full-FT checkpoints and logs. Ordinary full-fold readouts were 0.675331 / 0.675766 / 0.674836, respectively; these are not pure-chain paired comparisons or evidence of a better submission. Existing jobs were left untouched. Bulk caches (~94 GB) and organizer dataset bodies (~33 GB) have inventories, not uploaded copies; checkpoints and experimental outputs are included. Original server files are retained.

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
| Causal segment-recurrent Transformer | Both completed FT variants were tested at a fixed 20% teacher weight beside C: pooled gains +0.000115 / +0.000126, both negative on fold 1. Neither passed the registered +0.0005/both-fold-positive allocation gate. No new GPU run launched. |
| Runtime governor | Heterogeneous model costs are approximated as equal in current policy. A cost-model discrepancy alone does not prove a score/runtime improvement. |

## Current creative-research iteration

The [registered iteration](iterations/2026-09-30-creative/) fixes three methodological problems: separate reused-fold discovery from confirmation and training-seed variation from chain uncertainty; optimize the user's competition objective instead of imposing novelty; replace categorical role vetoes with empirical comparisons and leakage checks.

M01 re-evaluated existing matched width-256/128 head-FT runs for seeds 3/4/5, fixed epochs, on 860 pure sequences in 250 chains. Candidate-minus-control deltas were **+0.000073, −0.000339, +0.000792**; mean **+0.000175**, sample SD **0.000572**. The three-seed mean-prediction ensemble improved by **+0.000696**, a different estimand. These results do not establish a stable positive per-model width effect. See the [immutable result and input hashes](iterations/2026-09-30-creative/M01-paired-seeds.result.json).

I1 tested one preregistered disagreement-gated teacher mixture against a fixed mixture of the same models. Its pooled improvement was only **+0.000090**, below the **+0.0004** allocation threshold; gated target-weight coverage was **0.81% / 0.80%**, below the required **5%** in each fold. The specific screen failed its gate. No threshold was retuned and no student training was launched. See the [result](iterations/2026-09-30-creative/I1-result.json) and [completed iteration report](iterations/2026-09-30-creative/report.md).

The [next quant round](iterations/2026-10-01-next/report.md) prioritizes a fixed-shape, main-only head fine-tune: current auxiliary losses remain active even when the GRU trunk is frozen. Two lower-priority feasibility studies examine actual decoded quote-event clocks and sparse volume-update coordinates before new architecture/feature training. The round also records cache cleanup, restoration mappings, and the two completed Transformer comparisons. These proposals have no new training result yet.

The [deeper three-branch audit](iterations/2026-10-01-deep/report.md) now supplies three completed training-side CPU diagnostics and detailed controlled study designs. Shared main/combined-aux gradients were positively aligned in two early windows (+.068/+.294), while global clipping coupling was measured; the generic conflict explanation is unsupported at these points. A new train-only class-3 decoder removes 35.67% of proxy event writes, but signed-key agreement/support gates prevent treating it as a certified quote clock; classes 0–2 remain entirely on fallback. The fixed sparse-volume dictionary failed its geometry gate (6.91–11.93% low-error coverage versus 80% required), despite passing simple planted sparse-data controls. No new GPU fit or validation WP query ran. Next training should use two-epoch V1-like anchors, not silently replace them with one-epoch C; ensemble-aware head FT is a separate reserve. The architecture alternative is row-GRU state rather than instantaneous projection as event-GRU input, with explicit freezing of `event_gru` during head-only FT. See the three dossiers and independent cross-reviews for limitations and costs.

The [additional deepening and statistical EDA round](iterations/2026-10-01-eda/report.md) completed four preregistered CPU diagnostics (234.4 seconds of script execution; no GPU/model fits or candidate WP comparisons). All 24 parent/head1/window combined-aux cosines were positive (0.043–0.201), with clipping inactive even in chain-deletion checks; generic conflict/clipping explanations weaken. Fixed C was reproduced exactly; member correlations were 0.9893–0.9951 and most local scale-sensitivity intervals included zero. Between-class means account for 19.7%/23.3% of target-specific covariance, a legitimate component of pooled scoring. Two independent clock fits disagreed on only 8/735,555 class3 code events (joint chain-bootstrap upper bound 0.00476%) while removing 37.85% of proxy writes: a reproducible learned binary partition, not identified physical quotes. The 60.9% interval-update rate is almost entirely explained by intervals with any interior row, so it does not establish useful missing memory. Causal next-code prediction beat adaptive frequencies by 0.78976 nats/mark (conditional95% interval 0.78148–0.79799); detector constraints and original-trained priors under shuffled controls remain major interpretation limits. Quant detector/prior controls are the next mechanistic priority; aux-off remains a cheap possible future ablation, without a conflict claim. New artifacts are local/beleriand only; GitHub publication remains paused.

All validation folds have prior exposure. A pure chain excludes a training neighbour; it is not a fresh holdout. Chain bootstrap intervals condition on the fitted models, do not measure optimizer variation, and do not correct adaptive selection or hidden-test shift. Read recent pooled-WP comparisons separately from older block-averaged reports; historical scorers and definitions are retained with the source snapshots. On 1 October the official data overview and FAQ were retrieved through web search and explicitly confirmed global pooled scoring. This resolves the previous primary-source limitation: see [current metric](../METRIC.md) and [verification record](iterations/2026-10-01-next/metric-source.json). Historical reports retain what was known at their execution time.
