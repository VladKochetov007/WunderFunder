# dtw/readout — prediction path vs the piecewise-constant target (agent final message, saved by controller)

Panel: pooled 4-fold OOF of the distilled GRU256 family (head-FT `pred_c20261001_ft_d256_s3_f{k}h`, parent
`pred_c20261001_d256_s3_f{k}`), 1,873 valid sequences, scored rows. Base head-FT pooled 0.6906 / public-mix 0.6634;
FT+parent mix 0.6912 / 0.6641. Scripts `scripts/cr_dtw_readout_{panel,core,q1..q7}.py`; raw `q1..q7` json here.

1. Every target change falls on an a6-endpoint (E) row and every E row is a change (6,500,185 of 6,500,185).
2. Between-segment share of prediction variance: class 3 72%, other classes 93–97%. Non-causal oracle (segment mean):
   +0.0250 pooled, +0.0285 public mix, +0.078 class 3 — ceiling for any within-segment denoising.
3. Causal piecewise readouts all lose on all 4 folds: hold −0.146, running mean −0.056, reset EMA(0.5) −0.0018,
   jump-only −0.090, cross-fitted stackers −0.001…−0.004.
4. Why: in-segment predictions keep gaining real information about the next move. Broadcasting the last in-segment
   prediction to the whole segment gives 0.828 (+0.138). Class 3: WP 0.02–0.34 at ≥31 rows before the move (50% of
   class-3 weight), ~0.75 just before it.
5. No lag: a perfect post-event response is worth ≤ +0.0013; a 1-row delay costs −0.010. DTW warping paths saturate
   at the band edges and flip sign by class.
6. Parent and head-FT are not shifted in time (DTW offset 0.015 rows): averaging does not blur jumps.
7. The model is efficient w.r.t. its own inputs: GBM of its own next-8-row revision R² 0.03/0.015; best readout using
   it +0.00006.
8. Training-side survivor: optional-stopping lambda-return distillation target (bootstrap from the teacher's next-row
   prediction, true label at the last row of each segment). Explains E-174's failure (bootstrapped from the segment's
   last row, not a stopping time → 17% under-dispersed class-3 target); a toy student reproduces that (−0.010) and the
   sign of the alpha 0.5→0.7 gain. In the toy it only ties H-059 / pure teacher (+0.0026 vs alpha 0.7 +0.0022).
9. Recommendation: no GPU for readouts, smoothing, DTW losses or DTW targets; lambda-return (lambda 0.5, alpha 1.0,
   fold 0) only if H-059 or alpha 0.85 shows more teacher weight pays (neither did: E-329, E-331).
