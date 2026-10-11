# Public scores landed 2026-10-11 (user-pasted portal list)

| Upload | Public | Portal duration | Memory |
|---|---|---|---|
| 4_z1c_z4_pair_q8vnni.zip (Rust Q8Vnni int8, z1c+z4 pair) | **0.6668** (new best, prev 0.6664 z1c i16 single) | 27m01s | 635 MB |
| 5_z1c_q8vnni_single.zip | 0.6649 | 15m41s | 625 MB |
| earlier (same list): z2c alldata 0.6649; z5 seed9 0.6664; z4 seed99 0.6662; z3 0.6662; p2 student 0.6639; p1/p3 students TIMEOUT 61m57s (i16 two-model h384 teachers) |

## Evidence / reading
- Q8Vnni pair beats its own best member by only +0.0004 public (0.6668 vs 0.6664); single z1c q8vnni (0.6649) is 0.0015 below the same model in i16 form (z1c 0.6664 earlier? check: z1c i16 single listed 0.6664 in prior list) -> int8 quantisation may cost ~0.001 on single; pair partly masks it. Needs a paired comparison: z1c i16 vs q8 on identical local fold-1 predictions (quantisation loss measurement is open).
- Portal duration: pair 27m, single 15.7m => ~13 min/model on the scorer. Prior estimate (loaded 68-85 quiet) was pessimistic. Budget is 60 min => room for ~3-4 model Q8 ensembles (timeouts seen at ~62 min for two h384 i16).
- Diversity gain pair vs single: 0.6668-0.6649 = +0.0019 in q8 domain (consistent with local pair gain ~+0.003).
## Actions
1. Build 3-model Q8Vnni ensemble (z1c + z4 + muonall-h384/z5 diverse member) est ~40 min portal; check tail risk with 2x variance.
2. Measure q8-vs-fp32 loss locally per member; consider per-layer scale calibration if >0.0005.
3. Submissions slots: user reports 0/5 today.
