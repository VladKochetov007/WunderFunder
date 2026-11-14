# Final-day research contract (2026-11-15)

**Question.** Which deterministic ZIPs, built by the Nov 15 23:59 UTC deadline, maximize expected
*private* score, and which previously scored submissions should be selected for final re-scoring?
**Mode:** competition. **Tier:** C/B: exposed full valid (train-only twins) plus public LB feedback
(transfer evidence only; private is a separate hidden dataset).

**Deliverable.** Checked ZIPs (Python 3.11 offline CPU, ≤20 MB, ≤60 min, exact reset) and a
final-selection recommendation. No automatic upload; the user uploads (≈2 slots left Nov 14 UTC, 5 on Nov 15).

**Baseline.** Original team `tl3.zip` public .66832 (team best).

## Working model from the public/local table (instrument, not proof)

| Pair (same family, different fit data) | train-only public | all-data public |
| --- | ---: | ---: |
| TL3 family: PR49 TL3C blend vs original tl3.zip | .66355 | .66832 |
| KD50 GRU: s3 single vs s3/s4 final pair | .66108 | .66510 |

Within train-only models, local valid order matches public (TL3C .692336 > KD90 .69154 > KD50 .69047;
public .66355 > .66193 > .66108). Hypothesis H-Q15-TWIN: public ≈ train-only local rank + a roughly uniform
all-data shift (~+.004). Consequence: choose the **all-data twin of the best train-only local recipe**,
which is the TL3 TLead family, not our KD students. Falsifier: an all-data candidate whose train-only twin is
locally better scores clearly below tl3.zip (> .002) on public.

## Arms

1. **H-Q15-W18SWAP** (CPU, built): all-data W18 in the TL3 student slot; twin local .692377 vs .692336.
2. **H-Q15-TLEAD-SEEDS** (GPU, running): all-data TLead seeds 7/8/9 (PR49 recipe, train+valid).
   Expected value from the measured seed-ensemble law, ensemble−mean ≈ .336(1−1/k)(1−ρ̄), with ρ̄ estimated
   from train-only TLead s5/s6 full-valid predictions (export running).
3. Packaging: 4-member ZIPs only (5 members exceed 60 min at TL3 runtime ≈13 min/member).

## Invariants

Official `global_wp_clip2_v1`; fixed coefficients chosen before public feedback; strict FP32 fits; team
asset ancestry recorded; no paid compute.

## Stop

Deadline; or GPU queue complete and packages checked.
