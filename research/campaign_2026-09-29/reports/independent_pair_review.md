# Paired-head screen independent review

Verdict: PASS for the current hidden-128 local screen, with deployment parity
still pending. No candidate GPU job or training was started by this reviewer.

Blocking findings identified during review were corrected by the author:

1. Evaluation export now writes every chunk at its sequence offset and scoring
   checks every row-group position (`campaign_pair_states.py:138`,
   `campaign_pair_head_score.py:99`).
2. Backbone bytes are compared between fit/evaluation exports and head checkpoints;
   fitting validates the component-excluded manifest and each cached row's fitting
   membership (`campaign_pair_head_fit.py:95`, `campaign_pair_head_score.py:83`).
   Backbone loading requires fixed epoch and no evaluation during fitting.
3. Unary now applies GELU separately to the two factors before adding them
   (`campaign_pair_head_fit.py:54`), giving a genuinely additive matched control.
4. Required raw values are checked before clipping, nonfinite lambda/lr is
   rejected, and generated head predictions must be finite
   (`campaign_pair_head_score.py:30`, `campaign_pair_head_score.py:39`,
   `campaign_pair_head_score.py:139`).
5. First-batch extracted predictions are compared with `train_rnn.predict` before
   the export is committed (`campaign_pair_states.py:163`).

The fit path opens only fitting artifacts, computes its normalization there,
keeps identical block orders/update counts/loss/final epoch across all heads,
and zero-initializes only the final residual projection. Both factor matrices
remain nonzero. It trains on the current outer pair's fitting states; it does
not pool hidden coordinates or fitting labels from other outer pairs. Those
states may be in sample for their backbone; disjoint outer evaluation tests
the complete algorithm, without making these in-sample states themselves OOF.

Official comparisons use Global WP and stable weighted moments with
`centered=True`, resampling the same complete components for both candidates.
All three heads are compared with the same parent average; bilinear is also
compared directly with additive and output-only controls. Softened-mask results
use weights `abs(clip(y))*[1 for scored, lambda otherwise]` on required rows.
They remain transfer diagnostics, not official scores or public expectations.

## Independent checks

`independent_pair_probe_v3.log` / `independent_pair_probe.json` completed in
0.527078 s, peak RSS 626,952 KiB. It ran under a one-core CPU quota, one torch
thread and 2 GiB hard RAM cap, with no CUDA invocation:

- Manual ordinary GRU extraction versus `RNNModel.forward`: exact float32 equality.
- All three initial residual heads emit exactly zero; additive mixed derivative
  across the two members is zero.
- Required unscored infinity and NaN lambda are rejected.
- The actual AST of the authored offset writer gives exact complete order for
  19 sequences in batches 8/8/3, chunks 4,000 / 3,343 / 20,000.
- Soft-score lambda 0 / .02 / .05 / 1 agrees with an independent long-double
  formula to 1.12e-16; lambda zero equals official Global WP.

Hashes before/after the checks matched:

| Script | SHA-256 |
| --- | --- |
| `campaign_pair_states.py` | `8989e9573253889c04f47865e4c06f9a7d763f5345f58473660d6a8fdd1a5894` |
| `campaign_pair_head_fit.py` | `c034da652f8a05b81f7ede772a22d59bd7f2840015b87a841e9b9d393897bb66` |
| `campaign_pair_head_score.py` | `72fbbaaa786e6185f86b6fc82e3b0de3493b47d33c83f591d7842b8c2c934278` |

## Resource and remaining validation limits

Actual selected fitting masks give 2.52–2.72 million fit rows/fold. Hidden-128
float16 paired states plus all other export arrays need 6.21–6.83 decimal
GB/fold (6.535 GB for fold 0), 25.992 GB across four folds. Three head prediction
files add approximately 0.899 GB. Observed free space was 66.16 GB, so these
current-width exports preserve the 20 GB headroom contract. At hidden 256 the
exports would instead require 50.59 GB; recompute before changing width.

The normalization pass uses about 134 MB for its float64 state block and square
temporary at width128; the default scoring batch transfers about 67 MB of hidden
states. Exact and soft metric calls still materialize several full-panel arrays,
so scoring is not strictly bounded by one sequence. The production 36 GiB cap
is appropriate; the full panel was not executed under this review's 2 GiB cap.

First-batch parity is against research GPU inference, not offline native inference.
Float16 state caching and eventual one-row deployment must implement the same
casting convention and pass the campaign native/prefix/reset parity checks before
promotion. Actual GPU peak/runtime were not measured by this reviewer. Screening
booleans do not implement the full confirmation gates; the independent grader
must apply component uncertainty, per-fold and per-class limits, softened-mask
checks, and public-transfer restrictions separately.
