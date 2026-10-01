# Constrained transport diagnostic: non-identification and partial class-3 law

No target-law promotion follows from this audit. The supervised transport has useful coverage in classes 2 and 3, but does not identify physical quotes or reproduce the declared target rule on all classes. The initial unknown result remains preserved in `REPORT_v1.md` and `claim_ledger.jsonl`.

## Frozen design and information boundary

`preregistration_v2.json` fixes 96 supplied TRAIN groups and a fresh panel of 24 distinct fold-1 validation components. Every fold-0 and fold-1 evaluation component was excluded from TRAIN fitting. This panel shares no component with the v1 evaluation panel. Membership was selected from metadata hashes before evaluation labels or masks were read. No external examples, mappings, statistics or modules were used.

TRAIN targets supplied modal code spreads and scalar ask-move votes. The second target identity constrains the same moves through the TRAIN spread differences; the agreement of 247,247 available identity pairs is partly algebraic and is not independent evidence for physical quote identity. Bid movement was constrained to ask movement minus spread change. Support/confidence thresholds were frozen at 3/0.90 for spreads and 3/0.80 for moves. Unknown transitions or spreads invalidate downstream intervals.

Before any evaluation label or mask access, all four decoders were applied to features alone. Their union had 326,036 eligible four-quote-event required rows out of 477,624 (68.2621%), passing the preregistered 1,000-row and 5% gate. This is an applicability gate, not label-free identification. Evaluation target lattices selected regimes only for retrospective grading; that selection is not an inference feature.

The quote clock advances only on nonzero known decoded ask/bid moves. Quote horizons 2/4/8, publication-row horizons 50/200/1000, common mean/end rules, two independent stop rules, and rounding controls were fixed before evaluation. Publication rows are not identified physical time. Future observations were accessed only to diagnose laws.

## Four-quote-event common-mean rule

The table uses `need_prediction AND is_scored`. Target mass is the sum of the two absolute clipped targets. Exactness compares both **unclipped integer tick targets**. A failure assigns that row's whole target mass when either tick target differs; this diagnostic is not GlobalWP loss. Prediction clipping can hide some tick-law discrepancies. Unknown mass stays in the full denominator.

| Regime | Components | Eligible / full scored rows | Eligible / full target mass | Mass coverage | Exact eligible pairs | Failed eligible mass |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 3 | 484 / 10,393 | 466.62183 / 11,828.51586 | 3.9449% | 83 / 484 (17.15%) | 426.10794 (91.32%) |
| 1 | 3 | 130 / 4,074 | 193.17233 / 7,042.03119 | 2.7431% | 36 / 130 (27.69%) | 171.93529 (89.01%) |
| 2 | 7 | 2,555 / 3,367 | 3,899.89728 / 5,196.88626 | 75.0430% | 1,472 / 2,555 (57.61%) | 2,677.94496 (68.67%) |
| 3 | 11 | 31,265 / 31,725 | 42,121.82977 / 42,562.83347 | 98.9639% | 26,480 / 31,265 (84.70%) | 15,295.11202 (36.31%) |

Class 3 therefore supports a partial association with this declared law, not an exact generator reconstruction. Its scored event rows have 1,685/1,873 exact pairs, with 466.02354/1,814.69521 failed eligible mass (25.68%). Quiet rows have 24,795/29,392 exact pairs, with 14,829.08848/40,307.13456 failed eligible mass (36.79%). Reporting only event rows would hide most failures. Class 3 retains 441.00370 unknown target mass; classes 0/1 retain most mass as unknown.

## Frozen controls on class 3

| Rule | Scored mass coverage | Exact eligible pairs | Failed eligible mass fraction |
|---|---:|---:|---:|
| Quote 2, common mean | 99.18% | 72.18% | 60.72% |
| Quote 4, common mean | 98.96% | 84.70% | 36.31% |
| Quote 8, common mean | 95.83% | 77.10% | 37.49% |
| Quote 4, common end | 98.96% | 53.04% | 51.74% |
| Quote 4, independent stop then end | 98.96% | 9.85% | 80.43% |
| Quote 4, independent stop then mean | 98.96% | 0.016% | 99.97% |
| Next 50 publication rows, common mean | 99.18% | 61.80% | 66.41% |
| Next 200 publication rows, common mean | 97.29% | 68.46% | 53.79% |
| Next 1,000 publication rows, common mean | 92.86% | 49.44% | 56.13% |

Parity-one and half-up four-quote mean controls give 84.29% and 84.22% exact pairs, so those two rounding choices do not explain the main failures. The tested independent stops trigger at `-(current spread + 2)` ticks; their poor reconstruction rejects only these specific rules on this supervised coordinate system, not every path-dependent exit mechanism. Controls were not tuned or selected for a submission.

## Auxiliary clock audit on frozen TRAIN

`audit_aux_clock.py` reads only the frozen TRAIN panel and the historical TRAIN-fitted `cache/class3_moves_full.pkl`, whose hash is recorded. It faithfully reconstructs the existing auxiliary decoder; it does not assert those paths are physical quotes.

Across 43 class-3 TRAIN sequences, there are 169,385 representation-code changes but 106,856 decoded quote changes. Thus 62,529 code changes (36.9153%) leave the decoded quote unchanged. `scripts/cache_quote_path.py` advances its horizons on all 22-column code changes; its metadata calls these book events. They differ from the nonzero quote-event clock used in the reported E-079 mechanism.

| Horizon | Eligible required rows | Label difference > 0.001 | Mean absolute difference | RMS difference | Mean row span: code / quote |
|---|---:|---:|---:|---:|---:|
| 8 | 840,468 | 672,804 (80.0511%) | 0.15306 | 0.23402 | 505.47 / 618.03 |
| 24 | 813,795 | 667,591 (82.0343%) | 0.18858 | 0.30676 | 701.90 / 902.41 |

Differences are in clipped class-3 auxiliary displacement units, weighted by required publication rows. Intervals containing unknown future decoder events were excluded. This establishes a substantial auxiliary-label semantics mismatch. It does not establish why a prior training run scored poorly, nor a predictive gain from changing the clock. No cache or competition implementation was changed.

## Resources and reproduction

V2: CPU 100.9452 seconds, wall 101.9131 seconds, peak RSS 1,227,448 KiB, exit 0. Auxiliary audit: CPU 2.1059 seconds, wall 2.1609 seconds, peak RSS 1,062,572 KiB, exit 0. Three independent v2 clock/unknown/quiet-row checks and Ruff checks passed. The earlier four v1 checks and seven external replay checks also passed. Source identities, exact frozen membership and every raw denominator are in the JSON artifacts. Derived artifacts remain far below the 1 GB cap.

The durable v2 job used `wnn-provided-decode-v2-20260929.service`, a 100% one-core quota, 4 GB maximum memory, zero swap, and a 600 CPU-second hard limit. Commands were:

```bash
WNN_SLICE=wnn-campaign.slice WNN_CPU_PCT=7 WNN_THREADS=1 \
WNN_CPU_QUOTA=100% WNN_MEM_MAX=4G OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 \
ARROW_NUM_THREADS=1 scripts/limited .venv/bin/python \
research/campaign_2026-09-29/provided_decode/audit_v2.py \
--fit-sequences 96 --evaluation-sequences 24 \
--output research/campaign_2026-09-29/provided_decode/results_v2.json

WNN_SLICE=wnn-campaign.slice WNN_CPU_PCT=7 WNN_THREADS=1 \
WNN_CPU_QUOTA=100% WNN_MEM_MAX=4G OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 \
ARROW_NUM_THREADS=1 scripts/limited .venv/bin/python \
research/campaign_2026-09-29/provided_decode/audit_aux_clock.py \
--output research/campaign_2026-09-29/provided_decode/aux_clock_results.json
```

## Permissible next research

The external study supplies conceptual contracts only: price-keyed net edits separate rank transport, normalized liquidity geometry obeys declared symmetries, and publication policies can have different co-change signatures. Each competition representation requires a fresh derivation using supplied data. No external quantity, fitted transformation or symbol/time match may enter it. Column indices still do not identify depth order or best quotes.

A supplied-data transition representation could encode code changes, uncertainty and the distinction between representation and supervised quote changes. It is a candidate causal representation, not a measured gain. Before another target-law branch, independent review should explain the large quiet-row failures, assess TRAIN code-cycle consistency and withheld-transition coverage, and establish more identifiable coordinates. A shared latent conditional-target readout repeats the rationale already explored in E-066/E-073 unless it adds a verified new causal coordinate, exit law or supervision mechanism. No new model branch is recommended from these results alone.
