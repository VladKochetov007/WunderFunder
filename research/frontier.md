# Active frontier

## Target and baseline

Goal: validation WP >= 0.80 (clipped weighted Pearson, METRIC.md). Organizer
baseline 0.5896 (full). Best own model: GRU 256x2 + event features, full data 2
epochs, 0.5958 on valid600 (E-007).

## Supported claims

- Targets move only on joint i0 bid+ask price events (H-001).
- Four market classes by target quantum; class 3 (q=0.65687) is large-tick with
  ~8 book states, ~35-38% of sequences, and the worst class (0.35-0.44) (H-002).
- Class-3 WP is bounded by next-break predictability; the next state alone would
  give 0.77 (H-009).
- Training is GPU-bound; inference is FLOP-bound at 12-15 GFLOPS (profiling note).

## Debunked

- H-004 more data/epochs alone: flat.
- H-005 dense native rewrite as big speedup: FLOP-bound, 112 vs 119 us.
- H-008 event-lag state features fix class 3: event-row evaluation artifact.
- H-010 snapshot queue features predict break direction early: AUC 0.55-0.59 beyond 20 rows.

## Live mechanisms

| hypothesis | mechanism | evidence | next test |
|---|---|---|---|
| H-011 | in-context Markov transition stats | open | break AUC with online transition features |
| H-007 | scored-row propensity weighting | AUC 0.82 | weighted training A/B |
| H-006 | sparse input compute | microbench 3x | sparse projection in native model |

## Score arithmetic

WP is an equal mean over sequences. With classes 0-2 at ~0.73 average and class 3
(~38% of sequences) at 0.38, overall ~0.60. Reaching 0.80 needs class 3 near
0.70 or every class far above current levels; the class-3 break-horizon result
(H-010) makes this the binding constraint.

## TODO

- [ ] H-011 in-context Markov break features (probe, then GRU input)
- [ ] H-007 propensity-weighted loss A/B
- [ ] Auxiliary targets at multiple horizons (Jane Street 8th-place pattern)
- [ ] Online test-time adaptation: labels are future-derived only; check if past targets are reconstructible from later features
- [ ] Perf pass: no per-row allocations in PyO3 step, 64-byte aligned weight arena in access order, fused gate loop, sparse input projection
- [ ] Portable wheel build in python:3.11-slim-bookworm container; submission timing on 1 vCPU
