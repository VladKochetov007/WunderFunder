# Active frontier

## Target and baseline

Goal: validation WP >= 0.80 (clipped weighted Pearson, METRIC.md). Organizer
baseline 0.5896 (full). Best own model: GRU 384x2 + event features, 3000 train
seqs, epoch 0: 0.6096 on dev600 (E-028); CPU-budget GRU 256x2: 0.6036 (E-023).
Single seed; second epochs on 3000 seqs overfit.
Best devB764: mask-weighted devA fine-tune 0.6339 (E-036); train-only propensity^2-weighted GRU 256x2 0.6332 (E-039), 192x2 0.6317, 128x2 0.6310 (E-042, deployable candidate); plain 0.5950.

## Submission candidate

`submissions/ab9_pow2_h128/submission.zip` (1.2 MB): GRU 128x2 + causal event
features, trained on train.parquet only with scoring-propensity^2 weights
(propensity net fit on devA masks). Native Rust predictor, abi3 wheel built in
python:3.11-slim-bookworm. Holdout (official scorer) 0.6333 vs baseline 0.5899;
devB 0.6310/0.6314 (two seeds); 56 us/row incl scorer (~35 min for a
validation-sized test). Independent audit E-049: no look-ahead or leakage; crash and NaN issues fixed (E-050); open rules question on mask-derived training weights (user decision).

## Decisions pending (campaign paused 2026-09-15)

1. Rules: are training weights derived from validation is_scored allowed?
   Candidate (holdout 0.6333) uses them; fallback `submissions/ab12_h128_plain_fallback`
   (devB 0.5978) does not. Ask organizers if unclear.
2. Validation data in final training (stacking measured 0.6389 devB, E-045).
3. Hardware check (btrfs scrub, memtest) and restore train.parquet (E-032).
Goal 0.80 is not supported by any tested information source (H-014, H-016, E-021).

## Incident (E-032)

train.parquet sha256 changed during the session without a write (2 runs failed
with ZSTD corruption). Treat train-reading runs after ~02:30 as invalid until the
file is restored from the archive and its hash verified. valid.parquet is intact.

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
- H-011 in-context Markov transition statistics: no early break information.
- H-012 class-3 scored rows near breaks: they are farther away.
- H-013 i1 sign-cumsum lead-lag: AUC <= 0.53 in class 3.

## Live mechanisms

| hypothesis | mechanism | evidence | next test |
|---|---|---|---|
| H-015 | exact target definition on event clock | r 0.5-0.6 with crude levels | better level recovery; online label reconstruction |
| H-007 | scored-row propensity weighting | AUC 0.82 | weighted training A/B |
| H-006 | sparse input compute | microbench 3x | sparse projection in native model |

## Score arithmetic

WP is an equal mean over sequences. With classes 0-2 at ~0.73 average and class 3
(~38% of sequences) at 0.38, overall ~0.60. Reaching 0.80 needs class 3 near
0.70 or every class far above current levels; the class-3 break-horizon result
(H-010) makes this the binding constraint.

## Bound (H-014)

Perfect next-break knowledge within 20 rows gives class-3 WP 0.42; within 100
rows 0.615. Measured direction AUC is 0.83 at 6-20 rows and ~0.55-0.60 beyond.
With class shares ~25/17/19/38%, 0.80 overall requires classes 0-2 near 1.0 or
class-3 break direction ~500 rows ahead. Current evidence says 0.80 is not
reachable causally unless a new information source for class 3 appears.

## TODO

- [x] H-011 in-context Markov break features (refuted)
- [ ] H-015 exact target definition; if exact, online per-sequence recalibration from reconstructed past labels
- [ ] Visual analysis toolkit (fast renderer) of books, events, targets, mask
- [ ] arXiv scout: queue-reactive, microprice, OFI, propagator, cross-impact
- [ ] H-007 propensity-weighted loss A/B
- [ ] Auxiliary targets at multiple horizons (Jane Street 8th-place pattern)
- [ ] Online test-time adaptation: labels are future-derived only; check if past targets are reconstructible from later features
- [ ] Perf pass: no per-row allocations in PyO3 step, 64-byte aligned weight arena in access order, fused gate loop, sparse input projection
- [ ] Portable wheel build in python:3.11-slim-bookworm container; submission timing on 1 vCPU
