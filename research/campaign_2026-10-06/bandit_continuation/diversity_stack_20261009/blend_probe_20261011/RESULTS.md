# Causal blend/stack policy probe (2026-10-11)

CPU-only, RSS about 1.2 GB. Metric: official `global_wp_clip2_v1`, computed on flat scored rows (single-model A check: 0.66923 equals the banked 0.6692). Bootstrap: `paired_component_bootstrap` (centered moments, 1000 resamples, seed 7) over components.
Scheme "crossfit": fold-1 components split into two disjoint halves (seed 0); policy fit on one half, scored on the other, both directions pooled into one held-out prediction per row (every row predicted by parameters that never saw its component). "f0->f1": fit on all of fold 0 (the same recipe stems trained on fold 0), scored on all of fold 1. Fit-fold WP is in-sample and is NOT evidence.
Exposure caveat: pairs were chosen with fold-1 half-A info (P1 and P2) or fold-0 best single (P3); fold 0 and 1 are historically exposed. EWMA state starts at row 0 over all rows; class estimate (logistic on first-99-row diff statistics, 97-98% accurate) is fixed from row 99; scored rows are >= 99 so "static for rows<99" is moot.

Pairs: P1 = h230_s3_ep3 + vt_gru512 (single 0.6692 / 0.6580, static 0.6691); P2 = h230_s3_ep3 + vt_gru384_fixmix (0.6692 / 0.6682, static 0.6722); P3 = h230_s3_onecycle + c20261003_combo_s3 (fold 1: 0.6688/0.6674, static 0.6682; fold 0: 0.6817/0.6809, static 0.6814).

Held-out WP and delta vs static 0.5/0.5 [component-bootstrap 95% CI]. dB = delta vs the better single member chosen on the fit data.

| Policy | Fit data | P1 held-out (dStatic [CI]) | P2 held-out (dStatic [CI]) | P3 fold1 crossfit | P3 f0->f1 (dStatic [CI]; dB) |
|---|---|---|---|---|---|
| static 0.5/0.5 | none | 0.66914 | 0.67220 | 0.66820 | 0.66820 |
| 1 static weight grid (0.05) | half / fold0 | +0.00004 [-.0019,+.0016] | -0.00054 [-.0011,-.00002] | +0.00062 [+.0002,+.0011] | +0.00062 [+.0002,+.0011]; dB 0 (picked w=1, i.e. single A) |
| 2 per-target weights | same | -0.00043 [-.0024,+.0011] | -0.00056 [-.0012,+.00001] | +0.00062 [+.0002,+.0011] | +0.00062 (same) |
| 3a ridge [p1,p2] | same | -0.00017 [-.0014,+.0008] | -0.00016 [-.0005,+.0001] | +0.00061 [-.0005,+.0016] | +0.00043 [+.0001,+.0008]; dB -0.00018 |
| 3b ridge +[d,abs d] | same | -0.00125 [-.0030,+.0001] | -0.00099 [-.0019,+.0001] | +0.00072 [-.0005,+.0018] | +0.00044 [+.00004,+.0009] |
| 3c ridge +causal feats, EWMA | same | -0.00404 [-.0062,-.0022] | -0.00283 [-.0048,-.0005] | +0.00016 [-.0019,+.0022] | -0.00077 [-.0021,+.0008] |
| 4 causal gate (best hl by fit: 500/50, 100/500, 100) | same | -0.00037 [-.0024,+.0012] | -0.00066 [-.0018,+.0004] | +0.00054 [+.0001,+.0010] | +0.00055 [+.0002,+.0009]; dB -0.00007 |
| 5 table class x age x vol quartile | same | -0.00115 [-.0031,+.0003] | -0.00246 [-.0041,-.0010] | +0.00024 [-.0004,+.0010] | +0.00014 [-.0002,+.0005] |
| 6 early class routing (4 cells) | same | -0.00123 [-.0036,+.0006] | -0.00109 [-.0018,-.0004] | +0.00045 [+.00002,+.0009] | +0.00033 [0,+.0007] |
| best single chosen on fit | same | +0.00009 | -0.00786 | +0.00062 | +0.00062 |

Other direction (P3 fold1->fold0 and fold-0 crossfit): all policies within +-0.0003 of static except 3c (-0.0011 to -0.0028, CI excludes 0 on fold-0 crossfit) and table 5 (-0.0005). dB vs best single is <= 0.0000 for every policy in every setting except P2 where both members are near-equal and static wins by 0.007 over the fit-chosen single (single selection is the fragile choice, not the blend).

Triple (h230_s3_ep3 + vt_gru512 + vt_chunk500, fold-1 crossfit): static 1/3 = 0.67087 (+0.0016 vs A, CI [-0.0022,+0.0058]); simplex grid 0.66959 (-0.0013 vs static3, CI [-0.0026,-0.0002]); per-target simplex 0.66902 (-0.0019, CI excludes 0); ridge 0.66997 (-0.0009 [-0.0023,+0.0001]). Fitted weights are unstable across halves ((0.3,0.3,0.4) vs (0.7,0.1,0.2)).

## Findings
- No fitted or adaptive policy beats static equal-weight averaging by a margin outside noise. Positive deltas (+0.0005 to +0.0006) occur only when one member is clearly weaker and the grid collapses toward the single better member; they equal "best single", not a gain from adaptation (dB about 0).
- Causal gating (EWMA disagreement, |p|, age), regime tables and class routing add nothing over a static weight (|delta| < 0.0007 vs the static grid in all runs); the richer the policy, the worse held-out (in-sample fit improves by up to +0.003 and then loses 0.001 to 0.004 on held-out: overfit to component structure).
- Fit-fold vs held-out gap is the headline: 3c fit +0.0035 over static in-sample, held-out -0.003 to -0.004.
- Two-member and three-member static averaging of strong members remains the best policy; with near-equal members (P2) uniform weights beat any fitted weights.

## Recommendation
Implement none of the adaptive policies in the Rust path. Keep static equal-weight averaging (0.5/0.5, or 1/3 each for three). If one member is clearly weaker (gap > 0.005 WP), a static shared weight on the stronger member (grid 0.05, fit on a different fold) is the only supported change; its measured gain is <= +0.0006 and CI-borderline. The most expensive option (gate, ~4 mults + sigmoid per row-target, under 2 us) would be feasible to serve, but there is no evidence to justify it. Gate parameters if ever needed (hl by fit): see `res_*_crossfit.json` "params".

## Files
`build_flat.py`, `ewma_feats.py`, `warm_feats.py`, `lib.py`, `policies.py`, `run_probe.py` (tasks P1 P2 P3f1 P3f0 P3x), `run_triple.py`, and `res_*.json` (full numbers, per-direction held-out WP, fit-fold WP, fitted parameters). Intermediate flat arrays live in the session scratchpad, not committed.
