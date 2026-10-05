# Method report — three models (blind ideation, Connectome Challenge 2)

Grounding docs used: `BRIEF.md`, `METRIC.md` (`global_wp_clip2_v1`), `data_overview.md`, historical idea digest (`oct5_ideas.md`). Incumbent: GRU trunk + chunk2000/EMA credit + fixed-head procedure, public 0.6660. Decomposition says head (+0.0013) > seed (+0.0005) > nonlinear head (+0.0001), so all three proposals attack the **readout/objective layer**, not the trunk.

Metric consequences used throughout:

- The point-optimal prediction under weighted Pearson is the conditional mean of `clip(t, ±2)` under the row-weighting induced by `|y|` (Gneiting 2011: the mean is the optimal point forecast for squared loss *under the loss's own weighting*). It is **not** `E[sign|x]·E[mag|x]` (Cauchy–Schwarz gap; large when sign and magnitude are negatively dependent, e.g. post-move mean reversion), and it is **not** any per-subgroup average: correlation is affine-invariant globally but not within subgroups, so per-class heads/calibration can improve locally and de-correlate globally.
- `w = |clip(y)|` is unknown at inference but predictable: activity×EMA gates (statphys), volume28 (+0.00096, H-078 open), W384 (+0.0020, H-066 open).
- t1 (excl. 0.7120) beats t0 (excl. 0.6876) by +0.024; hft shows 91.5% joint jump agreement (modal 66.2% vs 43.1% shuffled) ⇒ shared latent event process; `puzzle_a`: 0 target changes on 97,928 unchanged-i0 transitions ⇒ targets ≈ frozen between i0 moves.

---

## Model 1 — Hazard-mixture event model (non-recurrent, structured event distribution)

**Mechanism grounding.** Targets behave like marked point processes: joint jump agreement (hft), freeze-between-events (puzzle_a, olympiad hold costs 0.137), and the fractal horizon split (row16 +0.037 vs event16 −0.055) say timing features should live at fixed row horizons, marks at event horizons. No recurrence anywhere; all state is O(1) running moments.

**Causal inputs / state.** Per row, update only: EMA/log-return moments of i0 mid (p0–p21 levels with learned per-level weights), running sums of `dv0–3`/`dp0–3` over last k trades, book imbalance at fixed horizons {4, 16, 64, 256}, activity count EMA (the statphys gate), rows0–99 prefix regime intercept (the hft gap). State = dict of EMAs + last-event index; reset per `seq_ix` is trivial. 112 raw features enter only through these causal summaries.

**Heads / readout.**
- Hazard head: discrete-time hazards `h_τ(x)` for a "target change" within horizon τ, τ ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024} (Cox 1972 discrete-time survival).
- Mark head: conditional signed clipped move `m_τ(x) = E[clip(t) | event at τ, x]` per target.
- Readout: `ŷ = Σ_τ h̃_τ · m_τ` where `h̃_τ` is the increment of the implied survival curve. This is a mixture-of-horizons conditional mean — the structured analogue of "weighted conditional mean", explicitly **not** sign×magnitude.

**Exact objective.** Pseudo-labels from train targets: event at row r iff `|clip(t)_r − clip(t)_{r−1}| > q90(Δ)` (quantile from train only). Loss = `Σ_r Σ_τ BCE(h_τ, 1{Δt ≤ τ}) + λ · Σ_{events} |m_τ − clip(t)|` with `λ` tuned on valid Global WP, nothing else.

**Mathematical assumptions.** (i) Target changes are approximately event-driven with observable hazard; (ii) hazard is conditionally Markov in the feature summaries; (iii) mark distribution depends on horizon (fractal's event16 −0.055 says horizon matching matters, so allow per-τ mark heads).

**Falsifiers.** (a) Δt distribution has no concentrated/discrete structure vs a geometric null (entropy ratio ≈ 1) ⇒ collapse to regression, reject; (b) constant-carry (freeze-between-events) readout already within ε of incumbent on unchanged-i0 rows and per-row variation hurts ⇒ timing stage useless; (c) within-sequence shuffle of event times: readout correlation must collapse to ≈0, else leakage via levels not timing.

**Controls.** Same features → plain ridge on `clip(t)` (no event structure): isolates the mixture readout. Geometric (constant-rate) hazard vs learned: isolates timing information. Mark head vs constant mark per τ.

**Cheap test.** (1) 10-minute EDA: histogram/entropy of Δt over t0, t1 on ~500 train sequences (decides falsifier (a) before any training). (2) If it passes: logistic-regression hazards + ridge marks on 500 seqs (~10M rows), CPU only, <1 h on 12 cores; score on valid subset with the official scorer.

**Cost.** EDA ≈ 0.2 CPU-h; full cheap test ≈ 2–4 CPU-h, no GPU. Full version (small MLP heads) ≈ 1 GPU-h training, CPU-only inference well under 60 min / 20 MB.

---

## Model 2 — Joint bivariate distributional head (joint-target approach)

**Mechanism grounding.** 91.5% jump agreement means (t0, t1) share a latent factor; the excluded per-target gap (+0.024) plus the `|y|` weighting explain why t1 is "easier": sparse large-|y| rows dominate the weighted moments, dense small-|y| rows (t0) are noise-dominated. Training the two targets jointly with a learned dependence structure lets the strong target (t1) regularize the weak one (t0) through shared parameters — something independent heads cannot do and the incumbent's independent ep0-head does not do.

**Causal inputs / state.** Any causal trunk; specify the incumbent-class GRU (256 units) plus the E424 diagonal-modulation adapter (largest adapter point, +0.000768, never tested *with* a head procedure). Same state rules as incumbent.

**Heads / readout.** Trunk → 4 outputs per target: `μ_k`, `log σ_k` (k ∈ {t0, t1}) + one shared `ρ̂` (tanh-squashed), passed through per-target **monotone spline transforms** (rational-quadratic, Durkan et al. 2019) before the likelihood. Readout is the model mean `μ_k` per target; t0's head additionally reads t1's head features through a learned gate (justified by 91.5% agreement).

**Exact objective.** Weighted bivariate Gaussian NLL with metric-matched weights:
`L = Σ_r |clip(t_r)| · −log N₂(T(μ(x_r), θ), Σ(σ, ρ̂); T(t_r))`, where `T` are the monotone splines (applied to targets and means alike) and `Σ` is the 2×2 covariance. This trains the metric's `|y|` weighting *and* the cross-target covariance end-to-end.

**Previous negative configuration vs new nuance.** Dead family E-287/288/292/293/295/309/332 (≤6e-05) was **post-hoc** per-target affine rescaling of finished predictions. Here the transform is **trained jointly inside the likelihood** and is monotone-nonlinear, so gradient flow reshapes the trunk; that is a different mechanism, and I claim no revival of post-hoc calibration.

**Mathematical assumptions.** (i) Residuals after `T` are approximately bivariate Gaussian with stable `ρ̂`; (ii) `|y|` in the loss is a good surrogate for the metric's effective gradient weighting; (iii) the t0←t1 feature gate is sparse (mostly closed).

**Falsifiers.** (a) Spline knots collapse to linear (`T ≈ identity`) ⇒ the model is the incumbent, reject as no-new-mechanism; (b) paired component bootstrap on valid: gated t0 head ≤ ungated t0 head ⇒ joint claim dies; (c) leave-one-component-out: if the t0 gain concentrates in one or two components (puzzle_b pattern: rg54/1385), it's a component effect, not a joint effect — report separately, do not pool.

**Controls.** Independent two heads, same trunk (= incumbent structure, isolates "joint"); single joint head with no spline (isolates "transform"); frozen-ρ̂ (ρ=0) ablation (isolates "covariance").

**Cheap test.** One-epoch fresh trunk on 1k train sequences or reuse of any existing trunk checkpoint; train {joint+spline, independent, joint-no-spline} heads only. 3 × ~20 min head training on one GPU (≤5 GB), scoring on valid CPU-side minutes.

**Cost.** ≈ 1.5 GPU-h cheap test; full version ≈ 6–10 GPU-h for trunk+head with 2 seeds. Inference unchanged from incumbent (fits 60 min / 20 MB).

---

## Model 3 — Metric-emulating heteroscedastic head (counterexample to head factorization)

**Mechanism grounding.** This is the class3-weakness explanation *without* an additive class metric: Global WP is computed once over all rows. A per-class head (or per-class calibration) changes each subgroup's mean/scale; that reshapes the prediction's **between-group** covariance structure, and the global correlation can fall even when every group's local fit improves — exactly the observed pattern (signal/econometrics class `.6569` −0.057 vs +0.036/+0.039/+0.051 elsewhere). The correct response is not class heads but a single global readout whose *training objective* already encodes the metric's `|y|` weighting.

**Causal inputs / state.** Incumbent trunk, unchanged (head-only; E424/E425 parents all lacked the head procedure — this plugs directly into the H-109 queue).

**Heads / readout.** Three outputs per target from the same trunk: `m(x)` (mean), `s(x)` (scale, heteroscedastic NLL à la Nix & Weigend / Kendall & Gal 2017), and `ê(x)` = predicted `E[|clip(t)| | x]`. Final prediction is `m(x)` alone — deliberately **not** factorized into sign and magnitude, and **not** per-class.

**Exact objective.** `L = Σ_r ê(x_r) · [ (m − clip(t))² / (2 s²) + ½ log s² ]` with `ê` trained self-consistently (auxiliary MSE head on `|clip(t)|`, updated jointly). Because `ê` up-weights high-|y| rows in the *gradient*, the fixed point of `m` approximates the `|y|`-weighted conditional mean the metric actually scores — rather than the unweighted mean the incumbent's MSE head estimates.

**Previous negative configuration vs new nuance.** Distinct from post-hoc scale/soup/calibration (dead, ≤6e-05: those altered *predictions* after training) and distinct from H-073 normalizer alignment (that aligned *moment normalization* across chunk cadences). The new knob is **loss re-weighting inside training**; the closest open prior is H-073's untested predictive normalizer fits, which this generalizes.

**Mathematical assumptions.** (i) `|y|` is predictable from x (supported by volume28/W384/gates); (ii) `ê ≈ |y|` makes weighted-MSE-optimal ≈ metric-optimal; (iii) conditional heteroscedasticity is monotone in `|t|`.

**Falsifiers.** (a) `ê(x)` is flat (quantile regression of |y| on trunk features, LR test) ⇒ void; (b) paired bootstrap on valid: weighted-loss head ≤ unweighted head on official Global WP ⇒ reject; (c) pre-registered kill: explicit factorized head `E[sign|x]·E[mag|x]` evaluated on high-weight rows — if it statistically matches `m(x)`, the counterexample is falsified and we say so.

**Controls.** Same trunk, unweighted MSE head; weighted but homoscedastic (`s` const); the strawman itself (per-class heads) run once to demonstrate the local-gain/global-loss effect on the official scorer — this is a diagnostic, not a submission candidate.

**Cheap test.** Head-only training on cached trunk activations (no trunk retraining): 4 configs × ~10 min GPU; official-scorer eval on valid per config. Also the zero-training EDA first: compare `|t|` distributions on scored vs unscored valid rows — if the scored subset is not `|y|`-enriched, the whole weighting argument weakens and we know before training.

**Cost.** EDA ≈ 0.2 CPU-h; cheap test ≈ 1 GPU-h; full version = one incumbent-style head retrain ≈ 1–2 GPU-h per seed.

---

## Priority and kill order

1. **Model 3 first** — cheapest, plugs into H-109's existing head queue, and directly tests the biggest elephant (head gain, mechanism unknown). Its falsifier (a) is a pure EDA.
2. **Model 1's Δt EDA second** (10 min, decides the event hypothesis for everything downstream, including H-084/H-088).
3. **Model 2 third** — moderate GPU cost, highest structural novelty, depends on trunk availability.

No blend grids, no public-score fitting, no score promises; every model ships with a pre-registered kill condition and same-generation contrasts (teacher/seed/epochs/head held fixed within each comparison).

References: Cox (1972), *JRSS-B* 59:187–220, doi:10.1093/biomet/59.2.187 · Gneiting (2011), *JASA* 106:746–762, doi:10.1198/jasa.2011.r10138 · Nix & Weigend (1994), *ICNN* · Kendall & Gal (2017), NeurIPS, arXiv:1703.04977 · Durkan et al. (2019), NeurIPS, arXiv:1906.04032.
