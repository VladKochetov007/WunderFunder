# 2026-09-28 — ensemble breadth, deployment safety, capacity in EMA regime

Plan: `~/.claude/plans/peppy-twirling-wall.md`. Public best 0.6622 (since2 gclp pair).

## Phase 0 — deployment (E-255, E-257)

- Old `solution_ensemble.py` ran **one member** unless all nine x86-64-v3 cpuinfo flags were
  present. New version: three builds (`scripts/build_native.sh` → v2 SSE, avx2+fma only, v3),
  every level runs every member, outputs identical (avx2 = v3 bitwise, v2 within 4.8e-7).
- Runtime governor: skips the cold first sequence, then projects total time from a rolling
  window and drops trailing members only if the 52-minute budget would be exceeded. Tested by
  forcing a tiny budget (dropped 4 → 1, stayed in budget).
- Cost is linear in members (≈46 µs/member, ≈4 µs fixed, under training load). Public
  calibration from the mixed ZIP (25m52s) → ≈18 µs/member on the scorer: K3 ≈ 38 min,
  K4 ≈ 49 min.
- Blocked 4-row AVX2 gemv: 5–10% faster, parity 4.8e-7.

## Phase 1 — ensemble breadth (fold analogs, pure chains, exact mask)

`scripts/ensemble_scan.py`, reference = EMA-4 pair analog (`ab62_since + ab62_base_s4`),
831 pure-chain sequences, folds 0–1. Unfine-tuned analogs (`research/artifacts/ensemble_scan_phase1.json`):

| set | Δ λ0 | folds | λ.02 | train mix |
|---|---|---|---|---|
| both EMA-6 seeds | +0.00142 ± 0.00072 | +0.00113 / +0.00172 | +0.00160 | +0.00108 |
| four (EMA-6 ×2 + EMA-4 ×2) | +0.00123 ± 0.00036 | +0.00109 / +0.00138 | +0.00137 | +0.00110 |
| four + baseline×0.1 | +0.00125 ± 0.00029 | | | |
| greedy K3 (EMA-6 s3, FT'd EMA-4, EMA-6 s4) | +0.00201 | exploratory | | |

The greedy pick of a fine-tuned member showed members must be compared at equal FT status;
`ab82_ft_*` fold analogs (two-stage gcorr LP-FT from fixed epochs, `scripts/ft_gclp.sh`) are
queued, with the all-FT scan to follow.

## Phase 2 — capacity / optimizer in EMA-6 regime (E-254 pre-registration)

| arm | Δ λ0 fold 0 | verdict |
|---|---|---|
| A wd 0.1 | −0.0001 ± 0.0015 | rejected |

## Result of Phase 1 (E-258 – E-260)

- All-FT scan (831 pure chains): diversity members carry the gain. `keep_k4` (since-2 pair +
  chunk-500 + online-norm) +0.0010 ± 0.0004, uniform over all four classes and both folds;
  replacing the pair with EMA-6 seeds gains only in classes 0–2 (class 3 flat/negative).
- Final holdout says ≈0 for every diversity variant (keep_k4 +0.00003 ± 0.0005). The disagreement
  is class mix (48 isolated holdout sequences are 67% class 3), not neighbour contamination
  (neighboured fold sequences also show the mixed analog positive).
- Runtime side channel: the mixed ZIP's public/old runtime ratio 0.71 matches both members
  running (local 0.76), not the v2 single-model fallback (0.50). Holdout λ0 has tracked the public
  sign on all three tests so far; pure-chain folds over-predicted the mixed ZIP.
- Class head is causally perfect at row 99 (368/368), enabling class-routed ensembles
  (`submission/solution_routed.py`).
- Brainteaser: every final fine-tune excluded the 368-sequence holdout (20% of labelled valid).
  Full-valid fine-tunes (`ft_gclp.sh … finalall`) restore it; unmeasurable locally by design.

## Upload candidates (all verified in the scorer image: v2 = v3, corr ≥ 0.99998 vs batch)

| zip | members | evidence |
|---|---|---|
| `final_keep_k4_all` | since s4, since s3, ch500, onorm — all FT on all valid | fold +0.0010 (dev-FT analog), holdout 0.0000; + full-valid FT |
| `final_since2_all` | since s4, since s3 — FT on all valid | isolates the full-valid FT lever vs the 0.6622 incumbent |
| `final_keep_k4` | same as first, dev-only FT | isolates the diversity lever |
| `final_routed_k4_all` | classes 0–2: EMA-6 ×2 + ch500 + onorm; class 3: keep_k4 | fold +0.0017 (selected post hoc), holdout ≈0 |
