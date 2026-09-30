# Event clocks: a reproducible partition is a narrower claim than decoded quotes

The previous audit established a substantial observable difference: a target-fitted lookup suppressed about36% of class3 proxy events. It also failed its signed-move unique-key agreement gate (44/237). Those conclusions stand. This round asks a new question: **do independently fitted maps agree on the binary update decisions that a recurrent state machine would actually execute?** A `(1,0)` versus `(2,0)` disagreement is important to reconstructing a price path but irrelevant to a zero/nonzero clock. Conversely `(0,0)` versus `(1,0)` disagreement changes the state trajectory. Counting both identically as one conflicting unique key was an unsuitable estimand for binary clock stability.

This is not a retrospective change to the previous gate. `clock-plan.json` registers the new sampling, binary readouts, uncertainty and resource descriptor before any CLOCK2 outcomes. The old failed gate remains in the record. All data remain exposed; even excluding the previous sample does not create fresh competition confirmation. The result can justify describing a stable *learned event partition*. It cannot establish physical quote truth or improved forecasting.

## Independent fitting groups and the precise statistical object

Two groups each contain128 disjoint training-only chains, with459 and461 full20k-row sequences respectively. All sequences in a selected chain stay in its group. The audit contains64 class3 chains and eight each from classes0–2,313 sequences total. Every chain with any validation member, the whole chain containing corrupted row-group528, and every previous architecture fit/audit chain is excluded. Train labels are read only for the two fit groups; audit reads contain112 features and no targets or scoring masks. Label-derived class metadata is used for selection/reporting, never for inference routing.

For each exact full22-feature transition, fit labels produce integer target jumps after division by the fixed class3 scale. The **modal signed jump** is selected first, with deterministic first-observed tie priority, then mapped to the Boolean rule `dask!=0 or dbid!=0`. Combining all nonzero jumps before taking a majority would define a different decoder and is explicitly prohibited. Synthetic tests include a counterexample: counts4 for zero,3 for one nonzero jump,3 for another nonzero jump. The correct signed-mode decoder says no update, despite the aggregate nonzero majority6/10.

Unknown transitions retain the existing proxy `(x0 changes AND x11 changes)`; no-change rows and reset row0 do not update. A known zero transition suppresses an update rather than acting like unknown. Both fresh maps and the historical cached map use the same fallback and reset semantics. The historical table is a descriptive comparator with incomplete original fit provenance; it is never an input to either fresh fit.

The primary numerator is the number of rows on which the two candidate update masks differ. We divide it separately by all nonreset rows, observed full22 code events, and the union of candidate updates. The first describes overall execution changes; the second describes the event partition; the third describes disagreement among actual opportunities to update. On edges known to both maps we also report binary and signed differences weighted by observed event occurrences, beside unweighted unique-key counts. Neither dictionary size nor repeated rows are independent sample sizes.

## Two kinds of uncertainty, with one resampling unit

Exactly1000 registered joint draws resample fit-A chains, fit-B chains and audit chains (within market class). Each draw reconstructs the **signed mode then binary decision** from per-chain outcome counts. If a rare transition vanishes from a resampled fit group, its lookup becomes unknown and falls back to the observed proxy. Ties and zero-total keys are recorded explicitly. Audit per-key/per-proxy-bit sufficient statistics allow exact reweighted disagreement counts without rescanning24.7 million rows per draw. Their unit-weight calculation must equal the direct row scan.

The same audit multiplicity draws also produce conditional intervals with both fresh tables held fixed. The difference between these two uncertainty summaries matters: a clock may look stable under resampling audit chains but unstable when its transition labels are refitted. Neither interval includes the original adaptive research process, missing market regimes or hidden-test shift. Two disjoint fitting groups are not two independent model-training seeds, and bootstrap intervals do not turn this exposed diagnostic into confirmation.

Raw modal purity is not probability that a physical quote is correct. Target jumps also depend on future-reference changes, stops and boundaries. A frequently observed edge can have modal purity below.95 while its modal **binary** classification is stable in every chain bootstrap. A rare edge can have purity1 from one observation and be unreliable. Therefore support is recorded in both rows and distinct chains; fixed support/purity strata and bootstrap absent/nonzero/tie probabilities are reported together. Their relation to actual quote truth is unidentified here.

The new resource descriptor requires the upper joint class3 binary-XOR/code-event interval≤1%, and both maps to suppress≥1% of proxy events in at least90% of class3 audit chains. Passing means the intervention is reproducible and nontrivial at this resolution, not that it should be trained or deployed. Failure does not reject all event-memory architectures. Zero denominators are undefined; they are never counted as successful zero disagreement or zero drop.

## What an interval audit can say about `h_t` versus `z_t`

The current DualClock event GRU receives the instantaneous projected input `z_t`, not the row-GRU state `h_t`. Current211-dimensional inputs already include99 hand-crafted event deltas/flags/recencies, so calling `z_t` purely a raw snapshot would understate its existing history. Nevertheless two prefixes can have identical terminal raw inputs, terminal deltas, last-event recencies and held price delta, while differing in the order of interior trade/volume-code changes. They then have identical current `z_t`; a recurrent `h_t` can in principle differ. This is a concrete representational distinction, not proof that the current trained GRU retains the relevant difference.

The fixed A clock is used to count observations strictly **between** consecutive event writes: rows with any i0 volume-code change, any i0 trade-code change, and any i1-feature change. These are encoded input updates, not measured order submissions, cancellations, trade counts or signed volumes. Gap bins are fixed at1,2–4,5–16,17–64,65+. Completed intervals, the prefix before the first event and the right-censored tail are reported separately. If a sequence has no events, prefix and tail descriptors refer to the same unclosed span and must not be added as disjoint durations.

Many interior updates would demonstrate that the recurrent token has a substantial observed prefix to summarize beyond the last update. It would not prove information loss in existing99 features, price-predictive content, or an advantage over a normal row GRU. A sparse or empty interior would weaken this particular justification for changing event-token input. Interval-weighted and equal-chain summaries are both retained, with conditional whole-chain intervals; undefined equal-chain ratios are excluded and counted.

No elapsed-row count is relabeled a physical timestamp. No class is inferred from future labels in the online rule. All state resets at each sequence boundary, despite whole chains being the resampling/split unit. A native implementation would still need identical four-decimal rounding, unknown-key handling, exact reset/stream behavior and a full-ensemble runtime benchmark.

## Correctness and scope

The plan was frozen before outcomes. A preexecution amendment fixes undefined-ratio presentation without changing sampling, estimands or gates (`clock-preexecution-fix.json`). The script's synthetic suite covers modal-before-binary semantics, deterministic ties, disappearing bootstrap keys, mixed-size bootstrap denominators, unknown transitions, streaming/chunk/prefix equivalence and strict interval endpoints. An initially wrong synthetic test expectation was retained in `clock-selftest-failure01.json`; it was not a data-driven recipe change.

Root executes the single bounded run (≤900 seconds,2 threads,10GiB); this branch does not launch competing remote analyses. No neural training, prediction-cache evaluation, candidate WP query or GitHub upload is part of CLOCK2. The completed results and next evidence requirements appear below; the preceding sections preserve the pre-outcome rationale.

## Observed result: stable binary partition, unstable rare signed details

The one registered run completed in **126.38 seconds**, peak RSS **1.27GiB**, processing all1,233 sequences. All chain-exclusion, input-fingerprint, streaming/prefix/chunk and sufficient-statistic checks passed. `clock-output/clock-result.json` is the raw result; fresh tables and every used-array hash are preserved beside it.

Fit A observed1,821,778 events and596 distinct transition keys; fit B observed1,820,363 events and357 keys. Their dictionary overlap contains227 keys: **43 have different signed moves, but none has a different zero/nonzero classification**. That18.94% unique-key signed discrepancy is a different statistic from binary decisions at observed audit rows. It does not undo the old gate failure; it explains why that gate was poorly targeted at this new question.

On64 class3 audit chains (232 sequences), there are4,639,768 nonreset rows and735,555 full-code events. The two deployed-with-fallback masks disagree on **eight rows**:

| Denominator | Observed disagreement | Conditional audit-chain95% interval | Joint fit+audit-chain95% interval |
| --- | ---: | ---: | ---: |
| All4,639,768 nonreset rows | 0.0001724% | 0.0000264–0.0004024% | 0.0000822–0.0007453% |
| All735,555 code events | 0.0010876% | 0.0001669–0.0025554% | 0.0004854–0.0047643% |
| Union457,059 candidate writes | 0.0017503% | 0.0002667–0.0041124% | 0.0007683–0.0076728% |

There are **zero binary disagreements on735,491 events known to both maps**, versus68 signed-pair disagreements on the same events (0.00925%). The eight binary differences all occur because exactly one map knows the edge: three on25 A-only events and five on11 B-only events. The remaining28 events are unknown to both and share the proxy fallback. Comparing A/B with the historical table gives nine/seven binary differences over the same735,555 events; these remain descriptive because the historical fit provenance is incomplete.

The conditional zero-width interval for known-both binary disagreement is an empirical bootstrap degeneracy, **not proof that unseen keys can never disagree**. The joint interval is wider for the actual fallback-inclusive clock, which is the appropriate primary implementation estimand. Audit-used fitted keys have bootstrap nonzero probability exactly0 or1 **when present** in this run's1,000 draws; binary instability comes from rare keys disappearing, not sampled signed modes switching between zero and nonzero. Signed modes can still tie or switch among different nonzero values. Audit-event-weighted bootstrap tie probabilities are approximately0.00347%/0.00399% for A/B; absent-key probabilities among originally known audit events are0.00197%/0.00265%.

Support/purity strata show why low target-jump purity was not an adequate rejection rule for binary clocks:

| Fixed A/B stratum | Audit events | Share of code events | Binary differences | Signed differences |
| --- | ---: | ---: | ---: | ---: |
| Both supported (≥20 rows,≥3 chains), purity≥.95 | 398,522 | 54.180% | 0 | 0 |
| Both supported, at least one purity<.95 | 336,661 | 45.770% | 0 | 0 |
| Both known, at least one low support | 308 | 0.0419% | 0 | 68 |
| Exactly one known | 36 | 0.0049% | 8 | undefined as a pair |
| Neither known | 28 | 0.0038% | 0 through fallback | undefined |

The large low-purity stratum is reproducible under the fitted binary policy. This does not prove that its physical quote interpretation is correct; a stable supervised partition can systematically represent a different target-related event distinction.

A knows735,516/735,555 class3 events (99.9947%); B knows735,502 (99.9928%). Neither map knows **any** event in the additional class0/1/2 audit samples (217,642/115,441/90,448 events respectively). Their zero disagreement there is entirely the identical fallback policy, not successful cross-class decoding. No rounding collisions were observed in either fit or audit:259/159 distinct rounded states in fits and16,234 in the audit each had one raw bit signature; no exact-change event collapsed to the same rounded tuple. No audited union transition key occurred with conflicting raw proxy bits. These checks constrain the observed support only; native numeric parity is still separate work.

## The clock change is broad within class3; the interval explanation is weaker than it first looks

A suppresses278,406 of735,462 proxy writes: **37.8546%**, conditional chain interval **36.8002–38.9847%**. B differs by two suppressed writes. Both suppress at least1% on **all64 chains**; the equal-chain median reduction is38.13%, with10th/90th percentiles33.77%/45.06%. Neither fresh mask adds a write outside the proxy in this audit. This is a reproducible thinning of an existing event stream.

For A's456,824 complete within-sequence event intervals:

| Gap in rows | Complete intervals | Share |
| --- | ---: | ---: |
| 1 | 178,635 | 39.10% |
| 2–4 | 169,211 | 37.04% |
| 5–16 | 60,714 | 13.29% |
| 17–64 | 36,381 | 7.96% |
| 65+ | 11,883 | 2.60% |

The mean complete interval is10.02 rows. The median *chain-specific median gap* rises from1 under the proxy to3 under A; median chain-specific90th percentile rises13→25.5, and99th percentile99.5→140.49. These are medians of chain quantiles, not pooled row quantiles. The longest observed A interval is4,683 rows. There are29,522 prefix rows and35,089 right-censored tail rows; every class3 audited sequence has at least one A event. The gap distribution is shown in `clock-gap-distribution.svg`.

Strict interior observations are plentiful but partly mechanical. Mean per closed interval is8.35 i0-volume-code updates (chain interval6.93–10.09),2.09 i0-trade-code updates (1.72–2.52), and1.77 i1-feature updates (1.36–2.31). An interior i0-volume or i0-trade update occurs in60.8961% of all intervals; i1 updates occur in14.3475% (11.5342–17.4877%). At least four interior trade-code updates occur in10.33% of intervals.

**The important counterargument:** 39.1037% of intervals have gap1 and therefore no interior row at all. Among the278,189 intervals with any interior row,278,188 contain a volume update and278,188 contain a trade-code update. Thus the striking60.8961% nonempty figure is almost exactly “the interval has an interior row.” It is not strong evidence for a special rich-history mechanism. Across4,118,333 strict interior rows, i0 volume codes change on92.58%, trade-code blocks on23.18%, and i1 blocks on19.61%. These are feed-level change frequencies, not physical flow measurements; the same aggregate nonempty totals do not imply identical volume/trade updates row by row.

## Revised evidence and next decision

The new binary-partition descriptor **passes**: joint event-XOR upper bound0.0047643% is below1%, and both maps materially thin every audited class3 chain. Unlike the previous signed-key gate, this measures the actual state-update policy. This supports a *reproducible, train-fitted binary clock with explicit fallback*. It provides no claim that reconstructed signed price paths are accurate, no validation beyond the observed encoded-state support, and no model-score evidence.

A subsequent model study has two separate treatments: replace the proxy with this partition while keeping event input `z_t`, or replace `z_t` with row-history state `h_t` while keeping the proxy. Combining both from the outset could make an attractive candidate, but would leave the mechanism unidentified; a full2×2 requires a larger explicitly allocated training budget. The cheap first study should freeze one of these paired contrasts, retain ordinary row-GRU and exact serving-ensemble controls, and not choose a “best” decoder or fallback on prediction outcomes.

Thinning can help preserve event memory, but can also delete predictive book-shape changes. The row GRU already observes those changes, and the existing99 features already summarize parts of history. The interval counts neither eliminate this redundancy concern nor prove that event memory needs a different token. A1 (`h_t` for `z_t`) remains a distinct label-free representation test; its rationale is architectural, not established by the near-tautological nonempty-interval rate.

Finally, the clock saves only event-branch updates, not row recurrence or the larger head. With H=E=128 and the measured class3 write-rate decrease278,406/4,639,768≈0.0600, arithmetic savings versus proxy DualClock are about5,922 weight MACs per class3 row, before lookup cost. This is a modest fraction of total ensemble work, and classes0–2 do not save updates through the fallback. The branch still needs native DualClock support and complete-ensemble timing. No training or deployment follows from the statistical descriptor alone.
