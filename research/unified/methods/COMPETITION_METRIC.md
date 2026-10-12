# Scoring reference: Global Weighted Pearson correlation

Specification ID: `global_wp_clip2_v1`. The authoritative source is the
[Connectome data overview](https://wundernn.io/connectome/docs/data_overview),
retrieved on 2026-09-29. Its snapshot is archived in
`research/campaign_2026-09-29/sources/connectome_data_overview.html`.

## Evaluation domain

Each sequence has 20,000 rows. Steps 0–98 are warm-up. Process every row in
order and reset all model state when `seq_ix` changes. Return two finite,
float32-compatible predictions at every required step, including unscored rows.
The inference callback does not receive targets or the scoring mask.

Scoring selects `need_prediction AND is_scored` over the entire evaluated
dataset. All selected rows contribute, including sequences with constant or
one-sided targets. There is no positive/negative sequence-eligibility filter.
An empty evaluated selection is an error. Invalid required predictions are
submission errors, even if their rows would not be scored.

## Per-target global correlation

For each target, let `M` be all selected rows across all sequences. Clip both
target and prediction before calculating weights and moments:

$$
y_i=\operatorname{clip}(t_i,-2,2),\qquad
\hat y_i=\operatorname{clip}(p_i,-2,2),\qquad
w_i=|y_i|,\qquad i\in M.
$$

With $W=\sum_{i\in M}w_i$, define the weighted means

$$
\bar y_w=\frac{\sum_iw_iy_i}{W},\qquad
\bar{\hat y}_w=\frac{\sum_iw_i\hat y_i}{W}.
$$

The correlation is

$$
\rho_w=
\frac{\sum_iw_i(y_i-\bar y_w)(\hat y_i-\bar{\hat y}_w)}
{\sqrt{\sum_iw_i(y_i-\bar y_w)^2\sum_iw_i(\hat y_i-\bar{\hat y}_w)^2}}.
$$

Calculate this separately for `t0` and `t1`, then average:

$$
S=\frac{\rho_{t0,w}+\rho_{t1,w}}{2}.
$$

Both targets have equal final weight. Sequence scores are not averaged.
Higher is better; the result lies in `[-1,1]`.

## Numerical and validation conventions

Use float32-compatible inputs and float64 accumulation. Validate raw selected
values before clipping, so infinity cannot become a valid clipped value.
Unselected warm-up placeholders do not enter the moments. Zero targets have
zero weight. If total weight is less than `1e-8`, or either weighted standard
deviation is at most `1e-8`, that target correlation contributes zero. Bound
the returned correlation to `[-1,1]` to remove numerical round-off.

Stable centered moments can be merged across chunks and sequences. Six raw
weighted sums are retained for research decompositions and paired component
bootstrap; near-degenerate values require care because subtraction of raw
second moments can lose precision.

## Standalone reference

The standalone scorer is `utils.ScorerStepByStep`; its `GlobalAccumulator`
does not require the research package. A direct array calculation is:

```python
import numpy as np
from utils import weighted_pearson


def global_wp(targets, predictions, need_prediction, is_scored):
    y = np.asarray(targets, dtype=np.float32)
    p = np.asarray(predictions, dtype=np.float32)
    need = np.asarray(need_prediction, dtype=bool)
    mask = need & np.asarray(is_scored, dtype=bool)
    if y.shape != p.shape or y.shape[-1] != 2 or mask.shape != y.shape[:-1]:
        raise ValueError("misaligned targets, predictions or masks")
    if not np.isfinite(p[need]).all():
        raise ValueError("nonfinite required prediction")
    if not mask.any():
        raise ValueError("no selected rows")
    sides = [weighted_pearson(y[..., k][mask], p[..., k][mask])
             for k in range(2)]
    return float(np.mean(sides))
```

Reports include target correlations, final score, selected row count, sequence
count, clipping and metric ID. Required-output validity is checked separately
from scored-row selection.

## Historical block diagnostic

`v2_wp_block_clip2_v1` previously averaged correlations over sequences with
positive and negative selected targets. It is retained only as a named legacy
diagnostic. Historical results remain identified by their original metric;
they must not be relabeled as Global WP. Diagnostic eligibility never changes
the official global evaluation domain or submission validity.
