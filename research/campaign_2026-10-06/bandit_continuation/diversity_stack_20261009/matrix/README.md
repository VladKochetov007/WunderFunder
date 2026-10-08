# Experiment matrix

Recorded runs: **254**. Planned mechanism rows: **23**. Reviewed endpoint rows: **6**.

Open [the searchable table](experiment_matrix.html), or use [all run dimensions](runs.csv), [planned experiments](planned.csv), and [reviewed results](reviewed.csv).

Blank cells mean unrecorded. Metadata presence does not establish successful training or valid held-out performance. Historical unreviewed scores are deliberately not ranked together. The full hypothesis/experiment ledgers remain authoritative.

| Priority | Experiment | Optimizer | Loss | Scheduler | Architecture / samples | Status |
|---|---|---|---|---|---|---|
| 1 | D01 | AdamW | wmse vs huber delta1 | OneCycle | GRU256x2; 350 complete training sequences | planned_not_admitted |
| 2 | D02 | AdamW vs Muon | wmse vs huber delta1 | OneCycle | GRU256x2; same complete component sample | planned_not_admitted |
| 3 | D03 | matched control |  | matched control | GELU vs fixed_mix readout; same GRU | planned_not_admitted |
| 3 | D04 | matched control |  | matched control | GRU256→384→512→768 teacher ladder | planned_not_admitted |
| 3 | D05 | matched control |  | matched control | GRU vs causal TCN; declared context/receptive field | planned_not_admitted |
| 3 | D06 | matched control |  | matched control | GRU vs LSTM | planned_not_admitted |
| 3 | D07 | matched control |  | matched control | Causal slow/fast state teacher or correction | planned_not_admitted |
| 3 | D08 | matched control |  | matched control | Uniform vs fixed complete-component bag/bootstrap | planned_not_admitted |
| 3 | D09 | matched control |  | matched control | Complete sequences; input-only strata; no replacement | planned_not_admitted |
| 4 | D10 | matched control |  | matched control | Complete components; frozen OOF difficulty weights | planned_not_admitted |
| 2 | D11 | matched control |  | OneCycle vs warmup+cosine | Same GRU and component order | planned_not_admitted |
| 2 | D12 | matched control |  | OneCycle vs WSD | Same GRU and component order | planned_not_admitted |
| 3 | D13 | matched control |  | OneCycle vs plateau/noise response | Same GRU and component order | planned_not_admitted |
| 3 | D14 | matched control | clipped vs raw residual target | matched control | GRU; official clipped-label magnitude weights unchanged | planned_not_admitted |
| 3 | D15 | matched control | mix vs wmse_gcorr / gcorr_pool | matched control | GRU; fixed pooled weighting/decay semantics | planned_not_admitted |
| 2 | D16 |  |  |  | Existing ancestry-valid frozen prediction bank | planned_not_admitted |
| 3 | D17 |  |  |  | Current-output or causal-prefix gate | planned_not_admitted |
| 2 | D18 | matched control | supervised + declared distill | matched control | Student128→256→384; heterogeneous vs anchor-only targets | planned_not_admitted |
| 3 | D19 | matched control | mean vs common/contrast heads | matched control | Same student trunk; redundant-head control | planned_not_admitted |
| 3 | D20 | matched control | MSE vs metric-gradient correction | matched control | One-stage causal correction; fixed small step | planned_not_admitted |
| 3 | D21 | matched control | target-specific family targets | matched control | Heterogeneous recurrent teachers and two students | planned_not_admitted |
| 2 | D22 | paired original |  | original recipe | Fold1/full membership after subset screens | planned_not_admitted |
| 3 | D23 |  |  |  | Selected student or complete ensemble endpoint | planned_not_admitted |
