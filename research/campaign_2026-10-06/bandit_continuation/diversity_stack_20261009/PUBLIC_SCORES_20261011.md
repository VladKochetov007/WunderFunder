# Public scores landed 2026-10-11 (user-pasted portal list)

| Upload | Public | Portal duration | Memory |
|---|---|---|---|
| 4_z1c_z4_pair_q8vnni.zip (Rust Q8Vnni int8, z1c+z4 pair) | **0.6668** (new best, prev 0.6664 z1c i16 single) | 27m01s | 635 MB |
| 5_z1c_q8vnni_single.zip | 0.6649 | 15m41s | 625 MB |
| earlier (same list): z2c alldata 0.6649; z5 seed9 0.6664; z4 seed99 0.6662; z3 0.6662; p2 student 0.6639; p1/p3 students TIMEOUT 61m57s (i16 two-model h384 teachers) |

## Evidence / reading
- CORRECTION: z1c i16 single = 4_z1c_alldata_s4_student_hft 0.6649; q8 single 0.6649 => int8 Q8Vnni quantisation loss ~0 publicly (identical score). Pair z1c+z4 0.6668 vs members z1c 0.6649 / z4 0.6662: +0.0019 over z1c, +0.0006 over best member z4. Prior best 0.6664 was z5 seed9 single (i16).
- Not yet tried: pair with z5 (0.6664) — 2_z5_z4_pair_q8vnni.zip is built and unuploaded; expected >= 0.6668.
- Portal duration: pair 27m, single 15.7m => ~13 min/model on the scorer. Prior estimate (loaded 68-85 quiet) was pessimistic. Budget is 60 min => room for ~3-4 model Q8 ensembles (timeouts seen at ~62 min for two h384 i16).
- Diversity gain pair vs single: 0.6668-0.6649 = +0.0019 in q8 domain (consistent with local pair gain ~+0.003).
## Actions
1. Build 3-model Q8Vnni ensemble (z1c + z4 + muonall-h384/z5 diverse member) est ~40 min portal; check tail risk with 2x variance.
2. q8 loss is ~0 (resolved). Upload 2_z5_z4_pair_q8vnni.zip next (27m-class, safe).
3. Submissions slots: user reports 0/5 today.

## User-reported 2026-10-12 result

`1ya21vpd`, `h390_s3_s4_q8_z4half_20261012.zip`: **0.6671**,36m36s,642MB. Exact frozen C SHA6013d20a7ff26ce6ef46821f80ed01da848ef4da565e2cd91656417ddd5186ea. User portal transcription, not browser verified. Other two 10-12-best-r1 packages remain unsubmitted. Single result; no component contribution or new candidate forecast established.
