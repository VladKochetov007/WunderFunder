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
