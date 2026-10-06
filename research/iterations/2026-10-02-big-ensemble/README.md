# Big-ensemble branch (2026-10-02)

Hypothesis: scorer ran midpoint (4 members) in 21m06s, so ~2.5x time headroom exists; widening the fixed ensemble with existing full-data members may add public score without training.
Members exported by export_members.py (same transform as midpoint package_student.py). Harness check_e7.py: 5 valid sequences through the real solution.py, parity vs member average, determinism.
Calibration: midpoint harness 38.5 local-min projected -> 21.1 public min (ratio 1.82).

| pkg | members (equal weights) | harness local-min | projected scorer min | parity |
|---|---|---:|---:|---|
| e7 | d256,f256_s3h,wm,ow,h128 s3,s4,c500 | 116.6 | ~64 (too slow) | 3.6e-7, deterministic |
| e6a | d256,wm,ow,h128 s3,s4,c500 | 88.0 | ~48 | |
| e6b | wm,ow,sh256,h128 s3,s4,c500 | 76.8 | ~42 | |
| e6c | d256,f256_s3h,wm,h128 x3 | 100.5 | ~55 (risky) | |

e6a = V1 members (public 0.6641) + midpoint + ordinary-warm 1L256: the clean public contrast vs V1. No local OOF for these full-data fits; public is the grader here.

## v1par (2026-10-03): V1 members at parent epoch 3 (no H2 valid head FT), raw view, int8, equal 1/4.
Purpose: public test of whether the valid-only H2 head FT transfers to test (local fold: ref .6884 vs ft128H .6892, c3 equal).
Harness 55.6 local-min (~25-30 public), parity 0.0, deterministic. Contrast: V1 0.6641.

## Valid-usage matrix on public (2026-10-03)
| | head FT on valid | no head FT |
|---|---|---|
| parent train+valid | V1 (h2, fp16+governor) 0.6641 | v1par.zip (ep3, int8) pending |
| parent train-only  | V3 (h1, fp16+governor) 0.6629 | v3par.zip (ep3, int8) pending |
Caveats: V1/V3 used fp16 + time governor, new ones int8 fixed4 (local int8-fp16 ~1e-5); V3 used 1-epoch FT (h), V1 2-epoch (h2);
find256to distils fin256 teachers trained on train+valid (indirect valid leak). v3par harness 59.3 local-min, parity 0, deterministic.

## Public results 2026-10-03
v1par 0.6630 (26m07s) | v3par 0.6606 (26m12s) | v1_dm 0.6639 (30m21s) | v1_dm5 0.6641 (39m12s).
Valid matrix: parent tv+H2 0.6641, tv no-FT 0.6630, train-only+H 0.6629, train-only no-FT 0.6606.
=> valid in parent +0.0024, valid head FT +0.0011..0.0023. demean neutral on public (-0.0002): dropped. Width (2nd d256) +0.0000.
Next: tune valid adaptation (FT epochs/lr, full-FT low lr, valid+train mix, multi-seed FT heads).

## F: k-fold valid bagging (LB pair for 2026-10-04), built from existing checkpoints
kf_h128 = ft128_f{0..3}h + ft128s4_f{0..3}h (fold parents train+3/4 valid, 1-ep head FT on the same 3/4), 8 x 1/8. Harness 81.0 local-min.
full_h128 = fin128_s3h + fin128_s4h (full-valid parents, 1-ep head FT), 2 x 1/2. Harness 12.2 local-min.
Both parity 0.0, deterministic. Public contrast isolates valid-bagging (ensemble width showed ~0 public effect).
kf_h256 = ft256_f{0..3}h (h256 s3 fold parents + 1-ep head FT), 4 x 1/4: harness 115.5 local-min (~46-48 public by recent ratios 2.4-2.55).
full_h256 = fin256_s3h single: harness 19.9 local-min. Both parity 0.0, deterministic.
Local/public ratios so far: v1par 2.13, v1_dm 2.07, e6a 2.41, v1_dm5 2.43, final_mix5 2.55 (larger packages -> higher ratio).
