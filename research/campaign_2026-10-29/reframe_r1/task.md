# Reframe task (fresh context): Wunder Connectome Challenge 2 — sequence regression

Objective: maximize official metric `global_wp_clip2_v1` (pooled weighted Pearson-like score on 2 targets t0/t1, labels clipped to ±2, weights = |clipped label|, fixed scored mask). Causal, deterministic online inference: model sees one row at a time per sequence (~1000-row sequences, first 99 rows warm-up unscored), must reset state per sequence. Submission: ZIP ≤ 20 MB, CPU-only Python 3.11, time limit tight (~100 s per 64 sequences OK). Current public best ≈ 0.668 (blend of 3 members: two Transformer-ish "FTT/tiedlead" members + one GRU student, fixed coefficients). Local folds ~0.66–0.71 depending on fold.

Data: anonymized order-book-like features per row (~200 numeric columns, some sentinels), sequences fall into 4 "classes" with very different predictability: per-class WP ≈ 0.80, 0.74–0.77, 0.71–0.73, and class 3 only 0.39–0.44. Targets change only at certain book events.

Decisive observations (measured):
1. Distillation from an ensemble of GRU teachers into a GRU student is the production recipe; student-slot replacement moves public score only ≈ +0.0006.
2. Muon optimizer beats AdamW by +0.0035 cold (prevents GRU update-gate saturation, 45%→3%), but only +0.0003 on the warm distilled student (parents already unsaturated).
3. Loss-family changes (NLL, Pearson, Huber, KD variants) cannot move the metric: |y|-weighted MSE is Bayes-aligned with pooled WP. Clipping training loss gradient at 1.0 helps.
4. Hand-crafted microstructure features, class-3 upweighting, extra "lead" inputs: all negative.
5. Blend law (measured on 56 model pairs and k-member ensembles): ensemble − mean member WP ≈ 0.336·(1−1/k)·(1−ρ̄), ρ = |y|-weighted prediction correlation. Same-recipe seeds ρ≈0.995–0.998; different optimizers ρ≈0.98.
6. TabM-style shared-GRU ensembles: members become diverse only by becoming weaker; net negative.
7. Public seed noise ≈ 0.0008; local paired SE ≈ 0.0004 per fold.
8. Teammates: large Transformers memorize (val R² 0.14 vs train 0.995); compact causal Transformers learn; LSTM-with-attention "GALAM" ~0.672 on full valid, plain GRU256 ~0.678, TL3-recipe GRU teacher 0.688 (train-only fit, full valid eval).

Constraints: strict FP32 training; local GPU 5 GB; Kaggle T4s; no new data sources beyond supplied train/valid and public pretrained models.

Task: propose 5 DISTINCT mechanisms (different families) that could plausibly add ≥ +0.003 to the served endpoint. At least one must come from a distant field (state the structural correspondence, e.g., from speech, control, physics, survival analysis, numerics), and at least one must be a measurement/diagnostic that would change what we do next. For each: mechanism, why it could help here (hypothesis), ceiling estimate and how to compute it from existing predictions/checkpoints without new training, transfer path to the served ZIP, cheapest informative test (≤ 4 fits or zero-fit), falsifier, cost, family label. Do not propose: optimizer swaps, loss-distance changes, class-3 reweighting, hand features, shared-weight ensembles. Be concrete and terse (≤ 900 words). Do not read or modify any repository.
