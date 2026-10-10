# Own-notes mining: untested modelling ideas from our data-structure findings (2026-10-10)

Scope: read-only mining of our notes and ledgers. No training, no Python, no data reads beyond the notes and the JSONL ledgers. Sources were `research/findings-2026-09-26.md`, `reflection-2026-09-26.md`, `game_changers.md`, `brief-2026-09-26.md`, `2026-09-28-rb-class3.md`, `2026-09-28-next-iteration.md`, `hypotheses.jsonl` (365 rows), `experiments.jsonl`, the data-reconstruction reports under `research/campaign_2026-09-30/data_reconstruction/`, the spread-aux handoff, `diversity_stack_20261009/SANTANDER_LESSONS_AND_SERVING_SPEED_20261010.md`, and `writeups/santander_transcript_english_translation.md`. Experiment IDs are E-xxx, hypothesis IDs are H-xxx.

Tiers: T0 = no fit, notes or arithmetic only. T1 = CPU probe, under 1 hour. T2 = one GPU fold arm (fold 1, seed 3, matched to the current parent). T3 = full recipe with seeds and packaging.

## 1. Coverage check, so nothing below is re-proposed

Several items that look new are already registered or already measured. They are listed here so the controller can skip them.

| Item | Where it already lives | Status | Consequence for this note |
|---|---|---|---|
| Current-code spread as an auxiliary target | H-052 (cpu pending), H-101 (open), H-172 / E-443 (planned, not run), spread_aux handoff | Registered, not run | Not proposed again |
| Analytic class-3 readout through a single latent F (t0 = F - ask, t1 = bid - F) | H-284 (proposed) | Registered | Not proposed again. Note: t0 + t1 = -spread for both-nonpositive pairs is the same identity, so H-284 and the spread auxiliary are one mechanism |
| Causal exact a6-endpoint event marker vs raw price-code flag | full112_relations/REPORT.md (exact on 67,839 nonzero moves, zero false hits on 45,649 zero events), E-302 H1 (targets change only on endpoint rows), H-120 | Measured as a feature, model comparison pending | Only the propensity use in idea 10 is new, and it depends on H-120 |
| Input compression 112 -> ~80 channels | SANTANDER note, R3-3b (queued) | Queued | Not proposed again |
| Value-frequency features as inputs (R3-7), column-shared encoder (R3-8), block permutation importance (R3-9) | SANTANDER note, queued | Queued. Post-hoc frequency rescale closed (-0.0001) | Idea 6 is a different encoding (target statistics, not frequency) |
| Raw 4/16-event i0 volume-code shocks | E-215 (gate AUC +0.03 and residual r +0.08 failed) | Refuted | Idea 2 is a different construction (signed accounting, not a shock proxy) |
| Markov-kernel class-3 label denoising | research/2026-09-28-rb-class3.md | Rejected (bias 23-25%) | Not proposed |
| Future quote-path auxiliary | 2026-09-28-next-iteration.md item 1, E-251 | Rejected | Not proposed |
| Sparse exact linear relations among columns | SANTANDER note, scripts/exact_relation_finder.py | None found | Idea 4 looks for an exact null space in the price block, not a sparse equation |
| Per-class / per-density blend weights | E-127, E-138 | Refuted | Idea 10 is a volatility gate, a different proxy, low odds |
| Class-mass or segment weights | E-104, E-107, E-102 | Refuted | Not proposed |
| Mask as deterministic function of future rows | H-031 | Refuted | Idea 10 does not assume determinism |
| Signed trade flow from price impact (MO0/MO1) | E-112 (delta -0.002) | Refuted | Idea 8 uses a different signing rule and is low odds |
| Online per-sequence z-score | H-038, E-118 | Refuted | Idea 1 is a lattice-based unit decode, not a z-score |
| Event-clock GRU | H-020 | Refuted | Not proposed |

## 2. Observations that drive the ideas

- Train vs valid is separable at AUC 0.989 to 0.993 within every class (E-094). The top drivers are volume-code statistics (classes 0, 1 and 2) and a6 std and a5 mean (class 3). Our train-unlike gain is the baseline blend effect (E-129: 0.0007 to 0.0013).
- A 63-column group of the raw features has a shared value vocabulary (median cross-sequence reuse 0.97), and 100 columns have 5 or 6 decimals (SANTANDER note, scripts/santander_eda_probe.py). That contradicts H-327 only in part. H-327 says no column is a small alphabet. Both can hold if the lattice has many levels per column.
- The features look like a monotone quantile transform of a raw lattice. The lower endpoint -5.199337482452393 equals float32(Phi^-1(1e-7)) (full112 report), and fit_feature_maps.json stores per-column leader tables for i0_A, i0_B, i1_A and i1_B.
- The 22-field i0 price block has an exact degeneracy in its rank-correlation spectrum (input_price_geometry/results.json, fit group 9555: four eigenvalues near 1e-17 and effective rank 1.40). The centred linear Gram matrix was never checked.
- Class 3: F = round(mean of the next four book-event mids) matches 97.5% of rows (E-079). The residual 2.5% was not checked for half-tick ties.
- E-095: target identical across 92% of chunk links, and 90 links cross train and valid. Rules forbid carrying state across sequences at test time, so the link is not a serving feature (see the exclusion note at the end).
- E-090: volume codes need 8 PCA dimensions per side for 90% variance. ICA finds one heavy-tailed bid factor and one ask factor. That analysis used the transformed codes, not raw levels.

## 3. Ideas (12 at most)

### Idea 1. Decode the raw lattice behind the quantile-transformed columns, then normalise each sequence by its lot quantum (reconstruction, T1 then T2)

- **Source:** E-094 (drivers of the shift), H-003 and H-021/H-081 (sentinel and rank-Gauss, still open), full112_relations/REPORT.md (normal-quantile fingerprint), SANTANDER note (63 lattice columns, 100 columns with 5 or 6 decimals), fit_feature_maps.json.
- **Observation:** If a column is a quantile transform of an integer or tick-valued raw count, the empirical CDF identity holds: Phi(z_k) equals the cumulative frequency of the k-th distinct value in the fit data, up to 1/N. Once raw levels are recovered, the step between adjacent raw levels is a per-column quantum. Train and valid probably differ in that quantum (instrument or regime), which would explain the 0.99 separability.
- **Proposed change:** (a) Per lattice column, rank the distinct values, check the CDF identity against cumulative counts, and read off raw level indices. (b) Test whether the raw step is constant within the column. (c) For each sequence, estimate the quantum and convert the lattice columns to integer counts, then feed these alongside the current inputs or in place of them. This is a unit decode, not a z-score (H-038 and E-118 were refuted).
- **Why not covered:** Nothing in the ledger decodes raw units. E-090 and the R3-3b compression work on transformed values. H-081 proposes a sentinel flag, not unit recovery.
- **Falsifier (T1):** GO only if, on at least 40 of the 63 lattice columns, at least 80% of adjacent raw gaps are equal to the modal gap within 1e-6 and the CDF identity holds within 3/N. NO-GO if fewer than 20 columns pass, which would mean H-327 ("no small alphabet") is the right reading. Downstream gate (T2): after lot normalisation, within-class adversarial AUC falls below 0.95 in at least 3 of 4 classes, and fold-1 pooled WP changes by at least +0.002 with a chain-bootstrap 95% CI above zero. Stress panel at lambda 0.05 must not fall by more than 0.001.
- **Cost tier:** T1 for the decode (CPU, one thread). T2 for the arm.

### Idea 2. Exact raw-space depth accounting: adds, cancels and trades at the touch (reconstruction, T1 after idea 1)

- **Source:** brainteaser next-iteration item 3 (queue removal/refill, not run), E-215 (shock proxy refuted, with the explicit note that train-decoded depletion is not ruled out), E-112 (price-impact split, refuted), volume-block structure in full112_relations.
- **Observation:** With raw levels recovered (idea 1), the bid or ask depth change at the touch between two book-event rows must equal adds minus cancels minus trades. The trade term can come from the trade-volume columns (i0_dv0-3 with the decoded last-trade price). Cancels then appear as the residual. Cancel-driven depletion is the usual short-horizon signal, and E-215 tested only raw displacement, which mixes the three terms.
- **Proposed change:** Build per-side causal features: trade volume at the touch, cancel residual, and the cancel share of depletion over the last 4 and 16 book events. Use them as inputs or as a gated residual, not as a standalone predictor.
- **Why not covered:** E-215 is a raw code displacement, not accounting. The depletion idea in the next-iteration note has not been run.
- **Falsifier (T1, before any GPU):** Use the E-215 protocol with the same pure-chain folds. GO only if the target-side direction AUC gains at least +0.03 and the held-out partial residual correlation is at least +0.08 in both folds. Also GO only if the cancel residual is non-negative on at least 95% of quiet rows, which checks the accounting. NO-GO on either failure.
- **Cost tier:** T1, then T2 if it passes.

### Idea 3. Decode the class-3 rounding exactly (training-label exactness, T1)

- **Source:** E-079 (round of the mean of the next 4 book-event mids, 97.5% exact), E-080 (rebuilt labels), E-162 (class-3 winner equals the first move sign, 99%), 2026-09-28-rb-class3.md (label-transform failure for a different reason).
- **Observation:** Each event mid is a half-integer when the spread is odd, so the mean of four mids lies on a quarter-tick grid. The 2.5% mismatch is probably tie handling (half-up versus half-even) on exact half-tick means, not a different future. If so, the label is an exact function of four observable future mids and can be decoded without error.
- **Proposed change:** Recompute the class-3 label with each tie convention and check which one matches every row. Keep the corrected label for training only. Serving is unchanged.
- **Why not covered:** E-079 reports the rule and its accuracy, but not the tie structure of the 2.5% mismatches. The rejected rb-class3 arm changed the label distribution, not its rounding.
- **Falsifier (T1):** GO if at least 95% of the mismatches are exact half-tick ties and one tie convention matches all of them. NO-GO if more than 10% of mismatches are not ties. Expected gain is small (at most about +0.0005 pooled), so the T2 gate is the existing pure-chain exact-mask pair gate of +0.0005 on fold 0.
- **Cost tier:** T1 (CPU, under 30 minutes). T2 only if the GO holds.

### Idea 4. Exact linear identities of the 22-field price block (reconstruction, T1)

- **Source:** input_price_geometry/results.json (four eigenvalues near 1e-17 in the rank spectrum, effective rank 1.40), full112 report (first field fixes most of the 11-field block on supported values, but not exactly), data_reconstruction/REPORT.md (1,026 accepted codes, zero collisions).
- **Observation:** The rank spectrum is computed on ranks, so ties and discrete levels can create a degeneracy without linear structure. A linear degeneracy of the centred raw Gram matrix is a different test. If the 22 fields are a thermometer or one-hot code over a fixed grid, the centred raw block has exact null vectors with constant coefficients, and the bid and ask indices would be exact integers.
- **Proposed change:** Compute the null space of the centred raw Gram matrix per class on fit groups and check whether it is constant across classes. If the code is one-hot or thermometer, decode the bid and ask indices directly and build exact level and spread features. Use these as an input check, not as a replacement for the GRU inputs.
- **Why not covered:** The rank spectrum has been measured, but no linear null space and no decoded integer indices appear in the ledger. The first-field statement is a near-identity, not an exact one.
- **Falsifier (T1):** GO only if the raw null space has dimension at least 3 with coefficients that are constant (up to 1e-6) across at least 3 of 4 classes, and the decoded indices reproduce the 1,026 accepted codes on at least 99% of held-out rows. NO-GO otherwise, which would mean the rank degeneracy is a tie artefact.
- **Cost tier:** T1 (CPU).

### Idea 5. Out-of-fold target encoding of the decoded book state (T2)

- **Source:** SANTANDER note (target-statistic encodings were the generic winners in the BNP talk, writeups/semenov.txt), data_reconstruction (1,026 codes with class and spread), E-088/E-089 (heuristic inputs, refuted, but not target encodings).
- **Observation:** The decoded state (class, spread, unknown) is a small categorical with many rows per state. A train-only empirical mean of t0 and t1 per state is a strong prior that the GRU must rebuild from raw fields.
- **Proposed change:** Add a per-row prior feature equal to the out-of-fold train mean of t0 and t1 for the current decoded state, cross-fitted by chain group, with additive smoothing of 50 rows. Use it as an input to the GRU.
- **Why not covered:** E-088 and E-089 were heuristic and raw features, not target-derived encodings. The reflection lists none.
- **Falsifier:** GO only if the arm gives at least +0.002 pooled WP on fold 1 at lambda 0 and lambda 0.02 with a chain-bootstrap CI above zero, and the same sign on fold 0. Leakage check: the encoding for any chain must be built from other chain groups only. NO-GO if the gain is within the seed band of about 0.0014 (E-084 scale).
- **Cost tier:** T1 to build, T2 for the arm.

### Idea 6. Domain importance weights on the training sequences (T2)

- **Source:** E-094 (adversarial AUC 0.99), E-129 and E-130 (blend gain grows with train-unlikeness), E-110 (stress panel), reflection open question 1 ("importance weights for train sequences"), H-019 (probing: match validation class mix by sampling).
- **Observation:** A per-sequence density ratio between valid and train, estimated from the adversarial classifier, is a different reweighting from class-mix matching (H-019). It up-weights train sequences that look like valid, and the reflection already flags this as open.
- **Proposed change:** Estimate w = P(valid | stats) / P(train | stats) per train sequence with the existing adversarial model, clip to [0.2, 5], and renormalise. Use w as a sequence-level loss multiplier. Keep the propensity weights unchanged.
- **Why not covered:** H-019 matches class mix, not the density ratio. E-110 used the stress panel only as validation. No training-side domain weights appear in the ledger.
- **Falsifier:** GO only if the stress panel at lambda 0.05 improves by at least +0.0015 and full-valid WP does not fall by more than 0.0005. The effective sample size of the weights must stay at or above 30% of the train sequences. NO-GO if the ESS gate fails.
- **Cost tier:** T2 (one fold, one seed, then T3 if it passes).

### Idea 7. Cold-start deficit by row index, then a burn-in-aware loss mask (T1 then T2)

- **Source:** E-049 (reset and warm-up audit passes), H-099 (probing: useful context localised by reset lag), H-156 (random-origin prefix views, proposed), E-095 (chunks are continuous, so every chunk start is a cold reset).
- **Observation:** The model is reset at row 0 of each 20k-row chunk, while the stream is continuous across chunks. The early rows of each chunk may carry an accuracy deficit that a burn-in loss mask could remove.
- **Proposed change:** Measure pure-chain WP by row-index bucket (0-99, 100-499, 500 and later). If the early buckets lag, train with the loss on early rows down-weighted, or with a burn-in prefix whose outputs are not in the loss. Predictions still come from the causal prefix, so this is serving-legal.
- **Why not covered:** H-099 proposes the measurement and is still probing. No bucketed WP exists in the ledger.
- **Falsifier (T1):** NO-GO if the 0-99 bucket WP gap to the 500+ bucket is below 0.02 for the class-3 pure folds. GO only if the gap is at least 0.02 and a T2 burn-in arm gains at least +0.0015 on fold 1.
- **Cost tier:** T1, then T2.

### Idea 8. Exact trade sign from the last-trade price field (signed flow, T1, low odds)

- **Source:** findings (dp0-3 is the last trade price relative to the quotes), E-100 and E-101 (trade-volume flow in every class), E-112 (price-impact split refuted).
- **Observation:** If the last-trade price field takes only the bid, ask or inside values, the trade sign is exact at the touch. Signed volume accumulators then give buyer-minus-seller flow with no impact model.
- **Proposed change:** Build an exact sign from dp against the decoded touch, then signed trade-volume EMAs at two half-lives. Use them as candidate inputs.
- **Why not covered:** E-112 used a price-impact classification, which is a different signing rule. The exact touch-based sign was not measured.
- **Falsifier (T1):** NO-GO if fewer than 90% of trades get an exact sign. GO only if signed flow adds partial residual correlation of at least +0.05 in both pure folds (the E-215 scale, relaxed because the trade signal is more direct).
- **Cost tier:** T1, low odds.

### Idea 9. Exact clock-event features in the scoring propensity (T1, depends on H-120)

- **Source:** full112 report (exact a6 endpoint marker), E-087 (mask depends on a3, i1 activity, a5 clock and book state), E-103 (label-aware gain +0.0022, AUC 0.896 to 0.898), H-031 (refuted for future rows).
- **Observation:** The propensity AUC is about 0.93 pooled (findings O4; per-class 0.89 to 0.90 in E-099). The mask terms in E-087 were fitted without the exact endpoint marker or the event ages built from it.
- **Proposed change:** Add the exact a6-endpoint marker, its age, and the counts of endpoint events in the last 4 and 16 rows to the propensity model. Keep the power-2 weights and the floor unchanged.
- **Why not covered:** The endpoint marker was measured as a clock identity, not as a propensity input. H-120 covers the feature comparison, not the mask.
- **Falsifier (T1):** GO only if pure-chain AUC of the propensity rises by at least +0.01 in both folds, and the weighted WP at lambda 0.02 gains at least +0.0005 with the same sign on both folds. NO-GO if the AUC gain is below 0.005.
- **Cost tier:** T1.

### Idea 10. Blend weight gated by a causal volatility proxy (T1, low odds)

- **Source:** findings "open tests" (the blend weight gated by a causal volatility proxy), E-130 (gain largest in low-volatility sequences), E-131 (optimal GRU share per train-unlikeness tercile).
- **Observation:** Gain by volatility tercile is 0.0017, 0.0009 and 0.0004 (E-130). A causal volatility proxy from the a5 and a6 clocks could select the blend weight, a different axis from the density gate of E-138.
- **Proposed change:** Blend weight w(v) for two or three volatility bins from the causal proxy, cross-fitted by chain pairs as in E-127.
- **Why not covered:** E-138 used sequence density, not volatility, and E-127 used class. The volatility gate was listed as open and never run.
- **Falsifier:** NO-GO if the cross-fitted gain is below +0.0003 in either direction. GO only if both cross-fit directions are positive.
- **Cost tier:** T1 (CPU, predictions already cached).

### Idea 11. Lattice-valued columns as a count vocabulary with a shared encoder (T2, folded into the queued R3-8)

- **Source:** SANTANDER note R3-8 (unpivot and a column-shared encoder, queued), H-327 (112 columns form update blocks with 3.18x label-change rate in the block 0-21 rows).
- **Observation:** Columns that share a lattice can share an encoder. Idea 1 gives the level index, which is a natural vocabulary key.
- **Proposed change:** Use an embedding of the decoded level index for the lattice columns, plus the shared encoder of R3-8. This is a modification of R3-8, not a new run.
- **Why not covered:** R3-8 is queued with a mean-pooling encoder. The level-index embedding is new but depends on idea 1.
- **Falsifier:** Same as R3-8 (fold 1 gain of at least +0.003 over the default mean). Only run if idea 1 passes.
- **Cost tier:** T2, after idea 1.

### Idea 12. Class identity from the first 500 rows as a training-time curriculum key (T1, low odds)

- **Source:** H-328 (class decodable from the first 500 rows at 99.4% balanced accuracy, measured), H-022 (class experts refuted), H-083 and H-166 (open class-conditioned calibration and auxiliary).
- **Observation:** Class is causal and early-available. A curriculum that stages classes (for example class 3 last, or in proportion to its 46% weight) is not an expert mixture and does not need a router.
- **Proposed change:** Order the mini-batches by the warm-up class posterior during training only. Serving is unchanged.
- **Why not covered:** H-022 tested class experts, not ordering. E-104 tested class-mass weights, which is a loss change.
- **Falsifier (T1 screen with the existing trainer):** NO-GO unless the curriculum arm reaches at least +0.002 on fold 1 at the same epochs. Low odds, so treat as a cheap ablation only.
- **Cost tier:** T2, low odds.

## 4. Requested reconstruction ideas, summarised

The brief asked for at least three reconstruction ideas. Ideas 1 to 4 are reconstruction ideas, and idea 3 (exact class-3 rounding) is the label-side one. The main points:

- The spread from the current code is exact on the accepted 1,026 codes. Its use as a target is already registered (H-052, H-101, H-172, H-284), so it is not repeated here.
- The absolute bid, ask and mid levels are not identified. Only relative quantities are. Idea 1 is the only route to raw levels, and it depends on the quantile identity holding. If that fails, idea 1 gives a NO-GO and ideas 2 and 11 are blocked.
- Depth and trade flow are reconstructable only after idea 1. Idea 2 is the first exact trade-flow decomposition the ledger does not already contain, and it can be falsified cheaply on CPU.
- Exact decoding of the price block (idea 4) would give integer bid and ask indices. Its payoff is in serving-legal features, not in labels.

## 5. Excluded on rule grounds

- Linking a test chunk to a labelled train or valid neighbour through the chain (E-095: target identical across 92% of links) would leak labels. Rules forbid carrying state across sequences at test time. This note does not propose it. The permitted version is training-only (idea 7 and H-156).

## 6. Suggested order

1. Idea 3 (T1, rounding ties), Idea 4 (T1, null space), Idea 9 (T1, propensity), Idea 10 (T1, cached predictions). All are CPU and under an hour.
2. Idea 1 (T1 decode). It gates ideas 2 and 11.
3. Idea 6 and Idea 5 (T2 arms) once the CPU gates pass.
4. Idea 2 and Idea 7 after idea 1 and the bucketed WP measurement.
