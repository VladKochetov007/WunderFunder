# Review coverage and experiment inventory

Historical negatives are controls, not automatic vetoes. Each entry needs code, correctness checks, and an executable sealed job before fitting.

| ID | Priority | Experiment | First contrast | Prior evidence |
|---|---|---|---|---|
| H-065 | P0 | combo_complete | Four fold-matched head fits then historic-pool final seed3 | E-359,E-381 |
| H-066 | P1 | width384 | Actual checkpoint CPU timing; fold1 replication and combo interaction | E-350,E-367 |
| H-067 | P1 | pair_bag | Seed4 fit and frozen equal pair | E-333,E-342 |
| H-068 | P1 | weight_int8 | Weight-only per-row INT8; full-sequence parity and scoring | E-336,E-406 |
| H-069 | P1 | qat_student | One matched distillation fine-tune | E-336 |
| H-070 | P1 | factorized_recurrence | W384 rank128 against budget-matched GRU | E-350 |
| H-071 | P2 | ema_rows | Combo decay sqrt(.999) versus .999 | E-354,E-356 |
| H-072 | P2 | cadence_credit | Common2000-row macrostep: connected versus midpoint-detached | E-392 |
| H-073 | P2 | distilled_normalizer | Historical/aligned/shuffled-multiplier comparison | E-362 |
| H-074 | P2 | muon_handoff | 75percent attempt switch matched full schedule | E-352 |
| H-075 | P2 | module_clipping | Reviewed module clip versus global clip | opt2 |
| H-076 | P2 | antithetic_order | Corrected reversed training order with same normalization and retained rows | opt2_review |
| H-077 | P2 | efficiency_weighting | Since weighting on/off on modern combo | E-351 |
| H-078 | P3 | volume28 | 28-feature GRU-input fit | E-321 |
| H-079 | P3 | random_origins | Random boundaries with causal prefix and row-budget matching | RESEARCH_RESET Card2 |
| H-080 | P3 | thinning | R11 augmentation with recomputed causal features | E-314 |
| H-081 | P3 | sentinel | Missing-value indicator and safe scaling | H-021 |
| H-082 | P4 | regime_heads | Cross-fitted head then end-to-end specialist fit | E-335 |
| H-083 | P4 | regime_offsets | Within-outerfold component crossfit | E-360 |
| H-084 | P4 | target_clocks | Separate t0 row-EMA and t1 event-EMA adapters | signal/econometrics EDA |
| H-085 | P4 | activity_memory | EMA times causal rate/variance adapter | linear/statphys EDA |
| H-086 | P4 | i1_innovation | Causal lagged i1 residual inputs on current GRU | E-383,E-384 |
| H-087 | P4 | event_age | Age times EMA adapter | statphys EDA |
| H-088 | P4 | observable_transitions | Causal transition feature head on strong states | hft EDA |
| H-089 | P4 | cycle_smoothing | Cycle8 then EMA versus row EMA | linear EDA |
| H-090 | P5 | factorized_branch | Frozen trunk screen then end-to-end fit | E-071,E-072 |
| H-091 | P5 | content_memory | Existing H049/H051 unified candidate and position control | H-049,H-051 |
| H-092 | P5 | structured_state | Repair literal parity then multiscale/single-scale fit | H-048,E-278 |
| H-093 | P5 | specialist_state | Specialist and universal/hash router comparison | H-050,H-032 |
| H-094 | P5 | conditional_adapter | Low-rank conditional adapter | campaign skill |
| H-095 | P5 | sequence_gain | Frozen-state recoverability then training-only gain auxiliary | E-314,Card3 |
| H-096 | P5 | crossfit_teachers | Same-population teacher lineage contrast then student | E-327,E-355,E-406 |
| H-097 | P5 | generation3 | Combo-containing teacher and matched student | E-337,E-355,E-406 |
| H-098 | P5 | origin_privileged | Cross-fitted future8 teacher preserving origin label/weights | privileged time-series literature |
| H-099 | P6 | reset_lag | Current student WP across fixed reset lengths | Card4 |
| H-100 | P6 | artbp_linear | Exact-oracle check then SGD/momentum paired predictive pilot | E-405 |
| H-101 | P6 | current_spread_aux | Bounded current-combo replication | E-287 |
| H-102 | P6 | lstm_diversity | LSTM128 matched fit and fixed bag | E-345,H-058 |
| H-103 | P6 | support_coordinate | Prospective support panel with original threshold | E-274 |
| H-104 | P6 | state_refresh | Bounded refresh versus equal-work sham predictive pilot | E-403 |
| H-105 | P6 | amp_stability | Matched FP16/BF16/FP32 bounded training contrast | E-404 |
