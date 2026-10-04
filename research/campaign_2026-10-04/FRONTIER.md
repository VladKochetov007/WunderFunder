# Frontier 2026-10-04

> Historical draft below. The user subsequently authorized implementation and
> bounded retests of earlier negatives. Current authority and execution state:
> [EXECUTION_CONTRACT.md](EXECUTION_CONTRACT.md),
> [implementation_state.json](implementation_state.json), and
> [EXPERIMENT_INVENTORY.md](EXPERIMENT_INVENTORY.md). The "dead families" and
> upload queue below are not the current admission policy. E-406 records the
> already completed TE07, q16x, reference and chunk500 public uploads.

Active workspace for the campaign. Update after every ledgered run; raw
evidence stays in `research/experiments.jsonl` and per-run reports.

## Current best
- Public: 0.6641 `D256_single_distilled` (registry, user-reported).
- Local: combo (Muon/chunk1000/boot-h8) d256 s3/s4, +0.0018 pooled over CTRL
  on two independent fold panels; gates passed; NOT shipped full-data.

## Alive mechanism families
1. Optimization recipe transport (H-A) — strongest local evidence.
2. Capacity/width (H-B, H-D) — one dangling CI>0 result.
3. Distillation teacher quality (H-C, gen-3) — one unresolved contradiction.
4. Deployment representation (H-E) — gate-limited but cheap enabler.
5. Loss normalization on live recipe (H-F) — instrument-ready.
6. Gradient credit (H-G) — instrument-ready.
7. Output representation (H-I) — structural reserve arm.

## Dead families (closed, do not reopen unchanged)
Physical spread decoding; event/row universal clocks; phase/periodicity;
carry; post-hoc calibration; class scales; output EMA; structural S/D head;
bilinear cross-member head; DTW family; weight soup; class-routed parent/FT;
w320 in its tested context; merged-GRU additive members; runtime-dependent
membership. Full list with E-references in ELEPHANTS_REVIEW_2026-10-03.md.

## Next experiments by expected information gain
1. H-A full-data combo fit + package (resolves the transfer slope).
2. H-B W384 fold1 (three-fit question on a CI>0 orphan).
3. H-C TE07 upload (free; resolves teacher contradiction; human gate).
4. H-F chunk-mass alignment fork (~2 GPU-h, touches live recipe).
5. H-G macrostep credit actual comparison (informs chunk-length physics).
6. H-E engine -15% (enables the pair-bag route to >= 0.666).

## Open contradictions
- Teacher ensemble: 2-fold +0.0011 vs holdout -0.0008 (E-330/337 vs E-339).
- Transfer slope 0.8 estimated on GRU128-since family; unidentified for
  distill/Muon models (H-A measures it).
- c3on/c3off screen ambiguous (+0.0015 pooled, p=0.054, folds disagree):
  close with the predeclared rule or run the deciding fold; do not leave
  ambiguous in the ledger.

## Bandit policy
Spend ~80% on the exploit queue above; reserve >= 10-20% of GPU budget for
the structural arm (H-I). If two consecutive failures land in one family,
reframe that family with a representation change before another scalar tune.

## Stop rules
- Phase 1 ends when H-A and H-B resolve (or budget spent).
- Campaign stops when E[public] >= 0.6660 is independently audited, or when
  alive families cannot fund a further discriminating experiment, or when
  remaining uncertainty needs the human upload gate anyway.
