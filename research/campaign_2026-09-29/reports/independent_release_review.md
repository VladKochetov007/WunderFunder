# Independent release review — 2026-09-29

Verdict: PASS for the corrected official evaluator and controlled campaign fitting.
Official confirmation must call `paired_component_moments` and
`paired_component_bootstrap(..., centered=True)`; block scores remain diagnostics.
No statistical public-score target claim is approved by this review.

## Evidence

- `independent_metric_check_v5.log` / `independent_metric_check.json`: nine focused
  checks passed in 4.365609 s. Eighty random masked, clipped, partitioned arrays
  matched an independent long-double centered-formula oracle to 5.56e-16. Raw
  nonfinite rejection, malformed shapes, empty selection, zero weight, constants,
  nearconstant streaming, exact component-resample replay, raw-resample ambiguity
  refusal, official per-class reporting without block eligibility, and complete
  three-sequence NPZ/NPY output passed.
- `independent_precision_search_v2.log`: 803 nearconstant opposite-component
  center/repetition fixtures passed in 0.572664 s. The earlier fixed-anchor
  counterexample is preserved in `independent_precision_search_v1.json`; the
  replacement retains per-sequence weighted means and centered second moments.
- `independent_controls_check_v2.log` / `independent_controls_check.json`: all four
  manifests independently reproduced using undirected adjacency traversal rather
  than production union-find; evaluation groups partition 1,873 sequences,
  fitting components exclude every evaluation component, fixed checkpoint policy
  is unaffected by supplied evaluation scores. Runtime 0.458855 s. Static review
  confirmed propensity's no-evaluation loader branch and fitting-only normalizer.
- `independent_launcher_check_v4.log` / `independent_launcher_check.json`: seven
  mocked-launch/telemetry and real slice-property checks passed in 0.017823 s:
  failures count, nonfinite timeouts are rejected, telemetry is bounded, prior
  active scopes are refused before new evidence, scope stop is confirmed, named
  scope/service deadlines and two-second grace are explicit. Live aggregate slice
  was 12 CPU cores, 36 GiB hard RAM, no swap. A bounded CPU-only scope probe is
  preserved separately; no GPU job or training was started by this reviewer.

All substantive checks ran under `WNN_SLICE=wnn-campaign.slice WNN_CPU_PCT=10
WNN_THREADS=1 WNN_MEM_MAX=4G scripts/limited`. Only reviewer report artifacts were
edited. Candidate and evaluator implementations were authored by other agents.

## Reviewed implementation identities

| Path | SHA-256 |
| --- | --- |
| `src/wnn/metric.py` | `6c3a8865687f1fca9257a93385c64c62a3ef0173d8ebb972ce3aaea8587ef3cc` |
| `src/wnn/gmetric.py` | `2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa` |
| `src/wnn/evaluation.py` | `96cebf2c8d564493f5a3a866b8d879964247f885587a455ebb1449e937230d0e` |
| `utils.py` | `e8e1034d61be359ebe943015e4e6bd2f76716303eff520857b7d07eeeb6144a6` |
| `scripts/verify_submission.py` | `d2c75df6accf8f88c911844171c03a020756833843d0b6db856e0cdebbe6bfb4` |
| `scripts/train_rnn.py` | `8abcb4e8dbf381a56d1db2347e6f9bfb392683eada839dfa688930e7db3ec67a` |
| `scripts/train_propensity.py` | `ea6bf05470b8bf8afa23b00b95eb980646f6cb7935e013326815af626013f3b7` |
| `src/wnn/splits.py` | `dfe1768828bec3172eb2bf84bfcb952a01e7eaef45d4ffd0d8c42a91d8e679ab` |
| `scripts/campaign_job.py` | `cfddecf202200173a035d741b01627a7c0f70826073f884d22cbbd26b1ab78c3` |

Metric/launcher hashes before and after their checks matched. The dated official
HTML hash is `95cded300df8281c99081b8cb3d7acf1253ae347b585701309fc8cb8196b97ec`.
Its text specifies pooled selected rows, clipped targets/predictions, absolute
clipped-target weights, and the mean of two global target correlations.

## Residual limits

- VRAM is observed periodically, not enforced as an instantaneous hardware cap;
  torch's fraction does not limit other CUDA allocations. Keep the total-device
  monitor and batch-32 smoke requirements.
- Older active jobs lack the newest scope identity. Root committed to waiting for
  the currently running old job to finish before the next GPU launch.
- Bootstrap uncertainty concerns the frozen local components. It does not remove
  candidate selection, previously inspected folds, historical chain overlap, or
  public-period distribution mismatch. No inference that expected public score
  reaches 0.6660 follows from this release.
- Native implementation, frozen-backbone heads/adapters, deterministic streaming
  causality, runtime, and final ZIP identity still require their own later review.
