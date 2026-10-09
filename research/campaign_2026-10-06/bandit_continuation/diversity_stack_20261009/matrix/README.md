# Experiment matrix

Recorded runs: **258**. Verified cloud fits: **6**. Planned mechanism rows: **29**. Reviewed endpoint rows: **20**.

Open [the searchable table](experiment_matrix.html), or use [all run dimensions](runs.csv), [verified cloud fits](cloud_fits.csv), [planned experiments](planned.csv), and [reviewed results](reviewed.csv).

Blank cells mean unrecorded. Metadata presence does not establish successful training or valid held-out performance. Historical unreviewed scores are deliberately not ranked together. The full hypothesis/experiment ledgers remain authoritative.

| Priority | Experiment | Optimizer | Loss | Scheduler | Architecture / samples | Status |
|---|---|---|---|---|---|---|
| 1 | D01 | AdamW | wmse vs huber delta1 | OneCycle | GRU256x2; 350 complete training sequences | completed_graded_R5_export_recovery; 350_sequence_exposed_screen; equal_blend_not_better_than_WMSE |
| 2 | D02 | AdamW vs Muon | wmse vs huber delta1 | OneCycle | GRU256x2; same complete component sample | completed11endpoint_grade; equalMuonAdamW_WMSE_blend_loses_specific_exposed_screen |
| 3 | D03 | matched control |  | matched control | GELU vs fixed_mix readout; same GRU | selected_fixedmix_WMSE_and_Huber_in_sealed_sampler_R3; upload_reconciliation |
| 3 | D04 | matched control |  | matched control | GRU256→384→512→768 teacher ladder | planned_not_admitted |
| 3 | D05 | matched control |  | matched control | GRU vs causal TCN; declared context/receptive field | planned_not_admitted |
| 3 | D06 | matched control |  | matched control | GRU vs LSTM | planned_not_admitted |
| 3 | D07 | matched control |  | matched control | Causal slow/fast state teacher or correction | planned_not_admitted |
| 3 | D08 | matched control |  | matched control | Uniform vs fixed complete-component bag/bootstrap | planned_not_admitted |
| 3 | D09 | matched control |  | matched control | Complete sequences; input-only strata; no replacement | planned_not_admitted |
| 4 | D10 | matched control |  | matched control | Complete components; frozen OOF difficulty weights | planned_not_admitted |
| 2 | D11 | matched control |  | OneCycle vs warmup+cosine | Same GRU and component order | planned_not_admitted |
| 2 | D12 | matched control |  | OneCycle vs WSD | Same GRU and component order | selected_in_sealed_sampler_R3; upload_reconciliation |
| 3 | D13 | matched control |  | OneCycle vs plateau/noise response | Same GRU and component order | planned_not_admitted |
| 3 | D14 | matched control | clipped vs raw residual target | matched control | GRU; official clipped-label magnitude weights unchanged | planned_not_admitted |
| 3 | D15 | matched control | mix vs wmse_gcorr / gcorr_pool | matched control | GRU; fixed pooled weighting/decay semantics | planned_not_admitted |
| 2 | D16 |  |  |  | Existing ancestry-valid frozen prediction bank | existing_pool_evaluated; equal_mean_beats_ridge_on_exposed_fold0 |
| 3 | D17 |  |  |  | Current-output or causal-prefix gate | planned_not_admitted |
| 2 | D18 | matched control | supervised + declared distill | matched control | Student128→256→384; heterogeneous vs anchor-only targets | planned_not_admitted |
| 3 | D19 | matched control | mean vs common/contrast heads | matched control | Same student trunk; redundant-head control | planned_not_admitted |
| 3 | D20 | matched control | MSE vs metric-gradient correction | matched control | One-stage causal correction; fixed small step | planned_not_admitted |
| 3 | D21 | matched control | target-specific family targets | matched control | Heterogeneous recurrent teachers and two students | planned_not_admitted |
| 2 | D22 | paired original |  | original recipe | Fold1/full membership after subset screens | baseline_pair_graded; all8exports_complete; fourfamily_fold1_cache_running |
| 3 | D23 |  |  |  | Selected student or complete ensemble endpoint | planned_not_admitted |
| 2 | SAMPLED_R3_HUBER_D4 | AdamW | huber | onecycle | 350 complete train sequences /83components; fixed exposedfold0 468eval groups /141components | sealed_reviewed_source_admitted; upload_reconciliation_no_fit_yet |
| 2 | SAMPLED_R3_HUBER_D2 | AdamW | huber | onecycle | 350 complete train sequences /83components; fixed exposedfold0 468eval groups /141components | sealed_reviewed_source_admitted; upload_reconciliation_no_fit_yet |
| 2 | SAMPLED_R3_FIXEDMIX_WMSE | AdamW | wmse | onecycle | 350 complete train sequences /83components; fixed exposedfold0 468eval groups /141components | sealed_reviewed_source_admitted; upload_reconciliation_no_fit_yet |
| 2 | SAMPLED_R3_FIXEDMIX_HUBER_D1 | AdamW | huber | onecycle | 350 complete train sequences /83components; fixed exposedfold0 468eval groups /141components | sealed_reviewed_source_admitted; upload_reconciliation_no_fit_yet |
| 2 | SAMPLED_R3_WSD_WMSE | AdamW | wmse | wsd | 350 complete train sequences /83components; fixed exposedfold0 468eval groups /141components | sealed_reviewed_source_admitted; upload_reconciliation_no_fit_yet |
| 2 | SAMPLED_R3_SGDR_WMSE | AdamW | wmse | sgdr | 350 complete train sequences /83components; fixed exposedfold0 468eval groups /141components | sealed_reviewed_source_admitted; upload_reconciliation_no_fit_yet |
