# Research contract — new public best (2026-09-28)

1. **Question.** Which verified package most likely scores above the public best 0.6622, and can a
   proxy calibrated on past public scores rank candidates better than the conflicting
   pure-fold (+) and holdout (≈0) proxies?
2. **Deliverable.** A scorer-verified ZIP (`PredictionModel.predict(DataPoint)`, 1 vCPU, ≤60 min,
   ≤20 MB) plus a ranked upload order with stated expected deltas. The user uploads; only the
   public score confirms success.
3. **Baseline.** `final_since2_gclp` public 0.6622 (members `ab67_final_since_s{4,3}_gcorr`).
   Leader 0.6677.
4. **Metric / invariants.** Global pooled clipped |y|-weighted Pearson over scored rows, mean of
   t0/t1. Invariants: causal per sequence, state reset at seq_ix change, no cross-sequence state,
   no test labels, training data = train + valid only, v2/avx2/v3 outputs identical within 1e-6,
   runtime governor keeps ≤52 min.
5. **Falsifiers.** A candidate is disqualified if (a) its scorer-image output differs from batch
   (corr < 0.9999), (b) its calibrated-proxy delta vs the incumbent is ≤ 0 with the calibration
   ranking the 10 known public scores correctly at Spearman ≥ 0.8, or (c) an independent critic
   finds a leak or rule violation.
6. **Allowed.** Cached predictions, valid labels, 10 known public scores, CPU ≤ 12 cores, RAM ≤
   30 GB, GPU (shared with research arms). No leaderboard probing packages, no external data.
7. **Budget.** Tonight: CPU analyses + one critic agent; GPU arms continue in background.
8. **Tier.** C (proxy metrics; public LB is a noisy sample, ~±0.0006 per paired comparison; private
   test is unseen). Human gate: the user uploads and reports the public score.
9. **Stop.** Success = user-reported public > 0.6622. Abort a candidate if it fails 5a–c. Plateau
   = two consecutive uploads within ±0.0003 of the incumbent → reframe (new mechanism family).
