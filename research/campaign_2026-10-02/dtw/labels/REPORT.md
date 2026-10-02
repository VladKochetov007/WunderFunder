# dtw/labels — event-clock label paths and DTW analogs (controller summary; agent stopped at the usage limit)

The agent's session ended (provider usage limit) before its report; results below are read from its saved outputs
(`knn_c3_L16_B2.json`, `proto_c3_L16_B2_M32.json`, `struct.json`, `drift.json`; scripts `scripts/cr_dtw_labels_*.py`).
- DTW-kNN on class-3 pre-segment event-clock paths (L=16 events, band 2, K=64, 245,301 query segments, TRAIN
  library): cross-fitted blend weight λ = 0 on every fold → +0.0000 pooled / public mix; corr(prediction, residual)
  −0.002 / −0.024 (t0/t1).
- DBA/k-medoid prototypes (M=32) of the same paths: λ = 0 everywhere → +0.0000.
- Elapsed-time drift readout (by class x elapsed bucket): −0.00017 pooled, −0.00019 mix, 3/4 folds negative.
- Anatomy (`drift.json`): metric weight by segment length; class 3 concentrates in 64–255-row segments.
Verdict: analog/motif forecasting over event-clock paths carries no information beyond the GRU.
