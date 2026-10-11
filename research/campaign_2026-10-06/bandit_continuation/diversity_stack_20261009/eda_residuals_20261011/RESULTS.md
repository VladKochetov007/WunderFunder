# Residual / surprise EDA, fold 1 (2026-10-11)

Notebook: `residual_eda.py` (marimo app, runs as a script; ~40 s, 1 core, peak RSS ~2.7 GB). Numbers: `results.json`. Plots: `plots/`.
Data: fold-1 valid, 457 sequences / 149 components, 812,167 scored rows (mask and step>=99). All of these data are historically exposed; this is diagnosis, not a held-out claim.
Models (all clipped to +-2, official `global_wp_clip2_v1` weights |clip(y)|): student `h230_f1_s3_ep3` ep2 WP 0.6692 (t0 .6637, t1 .6748); teacher `vt_f1_ctrl_s3` 0.6656; diverse `vt_f1_gru384_fixmix` 0.6682. The student reproduces the bank grade (0.6692).
Method: each row's additive share of the weighted-Pearson numerator ("WP share"), its weight share, its share of residual SS (w*(y-p)^2), and the local weighted correlation inside each slice.
`vt_f1_lead*` bank members (WP ~0.96) are non-causal oracles and were excluded.
Caveat on |y| slices: conditioning on the outcome (slope .02 at small |y|, 1.38 at large) is a selection artifact and is not used below.

## Top 5 surprises

### 1. Market class 3 is the metric: half the weight, 69% of the error, and a ceiling shared by every member
- Class 3: weight 50.4%, WP-numerator share 28.4%, residual-SS share 69.1%, local corr 0.442. Classes 0/1/2: 0.814/0.775/0.740.
- Class is a component attribute (purity 1.0). 69 of 149 components are class 3: median component corr 0.46 (p10 0.28, p90 0.61), versus a class-0 median of 0.81.
- Class 3 has the lowest trade_rate (.068 vs .13/.32/.15), 34% zero labels, and t0/t1 label corr -0.87.
- It is not an event-clock problem: fresh-book 0.470 vs stale-book 0.416 within class 3.
- It is not a member-choice problem: across 40 bank members spanning many recipes, class-3 corr only ranges 0.433-0.446. Classwise-stitched best members give WP 0.6711, the mean of all 40 gives 0.6713, and the best single gives 0.6702. A ridge on all 114 causal clocks plus the prediction gives -3% held-out residual SS reduction (-7 to -8% inside class 3).
- **H1 (loss weighting / data selection).** Class-3 rows carry 2.4x more error per unit of numerator and are information-limited for the current inputs and objective. A class-3-conditional objective (sequence-level upweight, or a class-3-only fine-tune of the head) should lift class-3 corr beyond the 0.446 envelope.
  - Cheapest falsifier: fine-tune only the head of the student on class-3 components of the fit split, with fixed membership. If class-3 corr on fold 1 does not exceed 0.456 (the bank envelope +0.01), class 3 is information-bound and loss weighting is dead for it.
  - Alternative (new information, not loss): check whether instrument-1 or cross-component channels carry class-3 signal, using a partial-correlation scan of the residual against the i1 features (this is the part the ridge on 114 clocks could not use linearly).

### 2. Error is a per-run level offset in long label runs, and the label changes only on book events
- 94.4% of residual SS is between label runs (per-run mean residual); residual autocorr 0.987 at lag 1, 0.62 at lag 64.
- Local corr decays monotonically with rows since the label last changed: 0.816 (run start), .800, .787, .757, .704, **.599 (16+ rows)**. Runs of 16+ rows hold 56% of weight and 72% of residual SS. Class 3 x long run: 37% of weight, 57% of residual SS, corr 0.434.
- Label changes: 99.98% of scored run starts have since_book == 0, yet only 51% of book-event rows change the label. In heur rows a change without an event happens 9 times in 187k events. Event rows are 20% of rows and carry only 9.2% of the residual SS.
- Post-hoc causal fixes all hurt: causal EMA of the prediction (hl 1..16) 0.6657 down to 0.594; momentum "advance" p+a*dp 0.6689 down to 0.642; causal segment-mean since the last event row 0.607. The NON-causal oracle that averages the prediction over each true label run gives 0.7000 (+0.031), so most of the loss is knowing when and how the label steps, not within-run jitter.
- **H2 (auxiliary target / architecture).** An event-gated latent (update on book-event rows, slow path for trades) or an auxiliary head predicting "label changes at this event" plus the post-event level should lift long-run correlation.
  - Cheapest falsifier: a binary event-change probe, a logistic fit of (label changes | book-event row) on the student's own features/hidden state, evaluated on held-out sequences. If AUC is under 0.60, the change decision is not predictable and H2 is dead. If it is predictable, the auxiliary target is promising.
  - Caveat: heur_valid covers only 12% of grid steps (90% of label changes happen at steps absent from heur), so the "events only" statement is verified on heur rows. Re-verify on dense train.parquet before building on it.

### 3. The label leads the features by about one row, and it cannot be recovered by extrapolation
- corr(y_t, pred_{t+k}): k=-6 .622, k=-1 .6692 (equal to k=0 .6692), **k=+1 .6785**, k=+2 .6766, k=+3 .6746, then decays (k=24 .627).
- A zero-lag causal predictor is about 0.009 below its own k=+1 ceiling. Note k=-1 equals k=0, so the prediction is "sticky" for about a row.
- **H3 (auxiliary targets).** Multi-lag auxiliary heads on y_{t-1}..y_{t-3} (hindsight-corrected labels, which are available in training) shape the state toward the label's actual clock.
  - Cheapest falsifier: a 1-epoch small-data screen with the aux heads (loss weight ~0.1, fixed fit membership) versus the same-seed control. A fold-1 WP gain under +0.001 kills it (the control seed spread on fold 1 is ~0.001; see the bank seeds).

### 4. Members share 96-99.9% of residual; seeds and distillation give almost no diversity, architecture does
- Residual corr: student|teacher 0.975, student|gru384_fixmix 0.960. Disagreement is only weakly informative: residual SS per unit weight is 0.67 at the lowest disagreement decile, 1.09 by decile 4, then flat.
- Blends: student+teacher 0.6690 (-0.0002), student+gru 0.6722 (+0.0030), mean of 3 0.6713.
- Across 40 bank members, pairwise blend gain vs student ranges only 0.00005-0.0046 and tracks (1 - residual corr) with r = -0.92 against rho (rho 0.949-0.9996).
- **H4 (member specialisation / decorrelation).** A member trained to decorrelate its residual from the student's (or one trained on the student's residual) gains more than another seed.
  - Cheapest falsifier: add a residual-correlation penalty arm. If the blend gain per unit of lowered rho is under 0.0005 per 0.01, diversity engineering is not worth GPU time. The bank-pair regression above is a free preliminary check (it supports the mechanism with r -0.92 on the existing pairs).

### 5. Two targets share one factor, but the model matches only the common mode; the difference mode is under-dispersed and shrink/scale tweaks do not help
- t0/t1 labels correlate at -0.83, predictions only -0.61, residuals -0.75.
- Mode variance (w-weighted): label difference mode u=(y0-y1)/2 1.474, common mode v=(y0+y1)/2 0.164. Prediction variance: u 0.520 (35% of the label's), v 0.171 (matches the label's). Mode corr: u 0.634, v 0.599.
- Post-hoc rescaling does not help: shrinking v gives 0.6692 (lam=1.0) vs 0.6672 (.75) / 0.6581 (.5); a 2D grid on (u scale, v scale) peaks at (1.0, 1.0) within noise (0.6693 at u.8/v.75).
- **H5 (loss reparameterisation).** Train on (u, v) with variance-normalised separate loss terms (u dominates the label variance, so v currently takes ~45% of the model's prediction variance for ~10% of label variance).
  - Cheapest falsifier: a pure-loss change (same net, same seed, 1 epoch): if WP is not above control by +0.001, drop it. Post-hoc scaling above already shows there is no free lunch in calibration.

## Nulls worth keeping
- Row age: 96% of weight is at step>=1000; ages 99-299/300-999 have corr .67/.69 vs .67, so there is no warm-up effect.
- Volatility and clocks: the quiet regimes are the worst (absd_p0_hl64 Q1 corr .526 with 23% of weight and 34% of the error; rate_book_hl64 Q1 .48), but this is mostly class 3. The calibration slope is uniform at 1.10-1.15 across all clock and volatility slices, so per-slice rescaling is pointless (Pearson is scale invariant anyway).
- Concentration: 24/149 components carry 50% of the numerator and 29 carry 50% of the weight (80%: 64 vs 69), so there is no extreme component concentration. 101/457 sequences hold 50% of the numerator.
- t0 vs t1: .6637 vs .6748; no structural difference.
- Sign accuracy (weighted) is 84.1%/84.4%. Only 3% of rows hit the clip at |y|=2 and they carry 8.7% of the weight, so clipping is not the story.

## Reproduction
`WNN_THREADS=1 WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_MEM_MAX=3000000000 scripts/limited .venv/bin/python research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/eda_residuals_20261011/residual_eda.py`

Caveats: the lead/lag, class and run-position findings use the single student/seed on fold 1, and the bank checks are in-sample over exposed data. Nothing here is a public-score forecast.
