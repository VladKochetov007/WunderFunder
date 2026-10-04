# Hypothesis cards 2026-10-04

Committed before any run. Fields per campaign convention. Cards are blind to
each other's results; the controller updates status only from ledgered runs.

## H-A — Full-data combo transfer (P0, exploit)
- Family: optimization recipe (Muon + chunk1000 + stopped-bootstrap h8).
- Representation change: none (recipe transport from fold panels to full data).
- Mechanism: better conditioning of the recurrent update (Muon), longer
  truncated credit (chunk1000), unbiased stopped targets (boot h8) compound
  into a better student at equal capacity.
- Predicted observable: public delta vs 0.6641 of +0.0008..+0.0018 (slope
  0.8 on +0.0018 local; slope unidentified for this family — the result
  measures it).
- Cheapest discriminating test: train full-data combo s3 (frozen flags from
  FOLD23_CONFIRMATION.txt), package single trunk int16, upload.
- Falsifier: public delta <= 0; or class3 guard failure at packaging.
- Cost: ~1 GPU-h fit + packaging; one upload slot.
- Status: proposed; blocked only on user unpause.

## H-B — Width 384 (P0, exploit of an orphaned positive)
- Family: capacity scaling.
- Representation change: wider recurrent state, same recipe/context.
- Mechanism: capacity was binding at 256 under the d256 recipe; fold0 CI > 0.
- Predicted observable: fold1 paired delta > 0 (~+0.001 if fold0 real).
- Cheapest discriminating test: W384 fold1 from `dcurve_fold.sh --hidden 384`;
  then folds 2/3 if positive; then muon+w384 interaction (H-D).
- Falsifier: fold1 delta <= 0 or fold0 direction fails to replicate.
- Cost: ~1 GPU-h per fold.
- Status: proposed. Note: w320's null was a different context (AdamW, ch2000).

## H-C — TE07 upload as teacher arbiter (P0, free)
- Family: distillation teacher quality.
- Representation change: none (already packaged, verified ZIP exists).
- Mechanism: resolves the E-330/337 (+0.0011 two-fold) vs E-339 (holdout
  -0.0008) contradiction with one public datapoint before gen-3 GPU spend.
- Predicted observable: public score of TE07s3 vs 0.6641.
- Falsifier: either outcome resolves E5; no scientific risk.
- Cost: zero training; one upload slot.
- Status: proposed; human gate.

## H-D — Muon x width interaction (conditional on H-B)
- Family: optimization x capacity.
- Mechanism: Muon's orthogonalized updates may shift the useful-width
  frontier; untested anywhere.
- Test: single fold0 screen w384 with `--optim muon`, matched no-op controls.
- Falsifier: combo-minus-muon-style increment <= 0.
- Cost: ~1 GPU-h. Status: dormant until H-B resolves.

## H-E — Engine -15% to enable 2x GRU256 pair bag (P1)
- Family: deployment representation.
- Mechanism: runtime gate (57 min > 52) bans width-256 blends; integer-madd +
  L2-resident weight split ~ -15% puts a pair at ~48 min.
- Predicted observable: pair bag adds +0.0004..+0.0008 over best single.
- Falsifier: measured public-runtime projection > 52 min, or holdout pair
  delta <= 0.
- Cost: engine work (CPU) + pair fit (~2 GPU-h) + parity gates.
- Status: proposed for phase 2.

## H-F — Chunk-mass alignment fork (month option 4) (P1)
- Family: loss normalization on the live recipe.
- Mechanism: self-normalized chunk mass has CV 15.16%; aligning the multiplier
  to own mass (s = W/D) changes the effective per-chunk force beyond a
  shuffled control.
- Test: aligned vs permuted-multiset whole-update fork on frozen actual
  states, then one paired pilot if the local screen passes.
- Falsifier: CV < 5% on the preregistered fit-mixture panel, or aligned ==
  permuted within 5x repeat RMS.
- Cost: ~2 GPU-h fork; ~6 GPU-h pilot if alive.
- Status: instrument-ready per month_research_options.md #4.

## H-G — Common-macrostep credit actual comparison (month option 3) (P1)
- Family: gradient credit across chunk boundaries.
- Mechanism: one common 2000-row loss with connected midpoint graph vs detach,
  same forward history and update clock — isolates credit, not state age.
- Predicted observable: whole-update relative difference >= 0.01 and > 5x
  repeat RMS on frozen cases; then a benefit-gated pilot.
- Falsifier: subthreshold -> close the credit branch at this horizon.
- Cost: ~10 GPU-min smoke + <= 8 GPU-h paired pilot if admitted.
- Status: instrument proven (58/58 synthetic); actual test never run.

## H-I — Causal branch/U categorical head (structural arm, P2) (U7)
- Family: output representation (the reserved 10-20% novelty budget).
- Representation change: replace scalar regression readout with branch
  probability x magnitude structure and a rational readout
  p = sum_b P(b|x) M1_b / sum_b P(b|x) M0_b, causal inputs only.
- Mechanism: targets live on a known lattice (H-002, H-042); branch identity
  is the dominant variance component (oracle 0.968, H-043). E-066 failed on
  oracle spread; E-303 was a different code-based head — the causal CE head
  is untested.
- Predicted observable: class3-under-pub143 improvement vs the same trunk's
  wMSE head; pooled neutral-to-positive.
- Cheapest discriminating test: freeze a combo trunk, train CE head on
  branch + U quantiles with propensity weighting, paired component scoring.
- Falsifier: no class3 win vs matched-budget head control.
- Cost: ~2 GPU-h screen + ~4 confirm.
- Status: proposed; highest ceiling, highest risk.

## Explicitly frozen (do not run in phase 1)
State-drift response, AMP metadata projection, ARTBP training proposal, the
80-GPU-h month lab as scoped. ARTBP blind-critic cross-review is CPU reading
and may proceed anytime. Properly closed families are listed in
`ELEPHANTS_REVIEW_2026-10-03.md` (bilinear E-270, soup E-332, routing E-335,
DTW E-334, w320-in-context).
