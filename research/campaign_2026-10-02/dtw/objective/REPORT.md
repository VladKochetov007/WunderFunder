# dtw/objective — training objective and output parametrisation (agent final message, saved by controller)

Setup: OOF panel, all 1873 valid sequences; P = mean of d256 parent and head-FT (alpha 0.5 student), TE = OOF since
s3+s4 teacher; base pooled 0.6912, public-like 0.6641. Population bias of a training target T = loss of a student
learning E[T|F] (cross-fitted |y|-weighted projection on [1, P, TE] per class) vs T = y. Run-coherent noise =
Σ_runs (Σ_run w(T−TE))² with training weights. Scripts `scripts/cr_dtw_objective_{panel,scored,stopped,coherent,strata,leak}.py`.

Derivations: (1) on a constant-y run, Σw(p−y)² = W(p̄−y)² + Σw(p−p̄)²; reweighting the second term by λ moves the
causal optimum away from m_t = E[y|F_t] unless λ = 1. (2) Survival lemma: with τ the first row of the next run
(observed at τ via the a6 endpoint), min(t+h, τ) is a stopping time; τ−1 (E-174's TD row) and u = τ−t (H-059's input)
are not. Optional stopping gives E[T_h|F_t] = m_t for a calibrated martingale teacher (also under the |y| tilt,
constant on a run); targets indexed by τ−1 or weighted by u condition on informative survival.

Findings:
1. Output smoothing/holding loses even with a reset at each a6 event: reset EMA β 0.3 −0.0006, hold −0.14 (4/4 folds).
2. Targets averaged/aligned/bootstrapped to the run end are biased: whole-run teacher mean −0.042, run-end TD −0.0071,
   DTW-aligned teacher −0.0038, forward mean −0.0046.
3. Stopped bootstrap (teacher at t+h while the run continues, else y) is unbiased: h=8 bias +0.0001.
4. Its run-coherent noise is 9x lower than alpha 0.7 (100x lower mixed at alpha 0.7); that noise lives in class-3
   runs of 64–1024 rows — where half the metric loss and the whole alpha 0.7 gain sit.
5. H-059 is biased (≈ −0.0003, +5–7% class-3 scale); its benefit is the same run-noise cut.
6. DTW-family losses rejected: boundaries exact, no lag → warp tolerance only blurs neighbouring runs' labels.
7. Hold-and-update, hazard heads, two-timescale state: no room. Cross-fitted teachers unnecessary (in-sample
   teachers absorb ~1% of the label residual; consistent with controller E-327).
8. Recommendation: one GPU run, alpha 0.7 + `--distill-boot-h 8`, fold 0 (`sbd_fold.sh 0 0.7 8` after applying
   `stopped_bootstrap.diff`). Predicted +0.001 vs alpha 0.7 (+0.0005…+0.003); falsified below +0.0005 or if it does
   not beat alpha 0.85 (alpha 0.85 lost to 0.7, E-329).
