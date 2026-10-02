# Upload plan (2026-10-02 09:20) — not uploaded; user decision

Both archives: our own code (own Rust int16 engine, own training scripts), one distilled GRU256 since student with
merged parent + head-FT heads, same recipe/data as `3_D256_single_distilled.zip` (public **0.6641**) except
distillation alpha 0.7 instead of 0.5. Gates (`research/campaign_2026-10-02/packages/<name>_pack.log`): ZIP ≤ 20 MB,
root `solution.py`, fixed member, no governor, v2/avx2/v3 bitwise equal, reset check, offline Python 3.11 image
bitwise equal to host, int16 vs fp32 max |Δ| 5.6e-5.

| order | archive | runtime | expected public | 80% range | P(> 0.6641) | P(≥ 0.666) |
|---|---|---|---|---|---|---|
| 1 | `submissions/c20261002_D07s3_i16/submission.zip` (sha256 `177e2642…f7ff`) | 307 kcyc/row ≈ 38 min | 0.6644 | 0.6633–0.6655 | 0.64 | 0.03 |
| 2 | `submissions/c20261002_D07s4_i16/submission.zip` (seed 4; sha256 `b266aec3…658c`) | ≈ 38 min | 0.6644 | 0.6633–0.6655 | 0.64 | 0.03 |

Forecast: anchor = public 0.6641 of the alpha-0.5 student (the public-like class-mix reweighted 4-fold OOF level
matched both single-family public scores to 0.0001). Delta alpha 0.7 − 0.5, public mix: fold 0 +0.0025, fold 1
−0.0008 (E-328), holdout (the final models themselves) s3 +0.0005, s4 +0.0005 (E-333). Holdout route 0.657 × 0.00048,
fold route shrunk to prior N(0.0003, 0.0006²) × 0.75 transfer → +0.0003; ±0.0006 model and ±0.0006 LB noise.

Why upload both: 1 tests the alpha change on the leaderboard; 2 measures seed spread of a single student (the
2-student ensemble is worth +0.0004 on the holdout but costs ~71 min — over budget until the integer engine lands).
Copies with priority prefixes: `submissions/upload_2026-10-02b/` (SHA256SUMS inside).


## Optional upload 3 (added 10:50)
`submissions/upload_2026-10-02b/3_TE07s3_teacher_ensemble.zip` (sha256 `7d5296c1…231a`, 315 kcyc/row ≈ 39 min, all gates
pass). Teacher-ensemble student: +0.0011 over alpha 0.7 on folds 0+1 (E-337) but −0.0008 on the holdout (E-339).
Expected public ≈ 0.6644 (same as 1, wider spread). Upload it only after 1, as a leaderboard test of the teacher
ensemble; if it beats 1 publicly, the next session should train more teacher-ensemble seeds.

## Public results (10:44–10:45 uploads)
1_D07s3 0.6635, 2_D07s4 0.6639 (forecast 0.6644 each). Alpha 0.7 does not beat the alpha-0.5 student (0.6641); seed
spread ~0.0004. Upload 3 (TE07s3) is NOT recommended: holdout −0.0008 vs D07s3 → expected ≈ 0.663.
