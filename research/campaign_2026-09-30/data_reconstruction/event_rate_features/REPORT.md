# Extra-feature event-rate probe

This is an input-only observational study. Code, candidates, component membership,
initialization, controls and gate were frozen before feature reads. Six fixed choices
were fitted on16 fresh TRAIN components and evaluated on16 different components:
318,416 rows per split after steps0–98. No targets, masks, VALID rows, prediction
arrays or external data were read. Previous E091 rate associations were disclosed.

| Field | TRAIN-selected causal rate | Fit Spearman | Evaluation Spearman | Within-component median | Components passing fixed rank gate |
|---|---|---:|---:|---:|---:|
| a0 | i1 volume refresh, half-life8 rows | .295 | .267 | .186 | 0/16 |
| a1 | i0 trade refresh, half-life8 rows | -.014 | -.020 | -.022 | 0/16 |
| a2 | i1 trade refresh, half-life8 rows | .572 | .535 | .476 | 0/16 |
| a3 | i1 trade refresh, half-life8 rows | .559 | .504 | .539 | 0/16 |
| a4 | i1 price refresh, half-life8 rows | -.159 | -.159 | -.109 | 0/16 |
| a7 | i1 volume refresh, half-life8 rows | -.465 | -.465 | -.367 | 0/16 |

All fail the fixed gate: sign-adjusted heldout Spearman >=.95 on12/16 components.
Even passing that gate would establish strong rank association, not numeric rate
identity. Matched event-row-age controls were at least competitive for a0,a2,a7;
step-index correlations were small. Removing exact rank-normal endpoints left the
a0–a4 medians unchanged; a7's median became -.315. Constants count as rho0 and all
16 components remain in the denominator.

## Quiet-row publication invariants

814 evaluation rows had no rawfloat32 p/v/dp/dv change for either instrument.
Changes below end at the current row and compare it with its preceding row:

| Field | Decrease | Unchanged | Increase |
|---|---:|---:|---:|
| a0 | 425 | 4 | 385 |
| a1 | 419 | 0 | 395 |
| a2 | 747 | 0 | 67 |
| a3 | 786 | 0 | 28 |
| a4 | 397 | 0 | 417 |
| a5 | 0 | 0 | 814 |
| a6 | 0 | 3 | 811 |
| a7 | 182 | 246 | 386 |

a0–a4 change on quiet rows, so they cannot be deterministic functions solely of
the current104 published market fields. Mostly decreasing a2/a3 are compatible
with activity measures, but their positive quiet increments contradict a pure
monotonically transformed exponential decay driven only by our observed refresh
flags. This does **not** reject event rates driven by unpublished events, repeated
messages with identical values, a different clock, or other hidden source inputs.
The joint-quiet support is small and dependent; row counts are not independent
sample sizes. a5/a6 positive quiet changes reproduce their narrower age-like
behavior; they do not identify physical units or a shared clock.

## Resources and limits

Freeze/fit/evaluate all exited0; outer durations .651/2.528/3.084 seconds. Peak RSS
was1,318,748KiB. Each foreground phase used one thread pool worker, hard2GiB
address-space and60-second CPU/wall limits. This is not a durable job or cgroup
enforcement claim. Zero GPU seconds. Exact argv, source hashes, resources,
selection, complete correlations and component counts are persisted alongside
this report. Independent reproduction is pending. No model fit, target law,
source identity or public-score improvement follows from these results.
