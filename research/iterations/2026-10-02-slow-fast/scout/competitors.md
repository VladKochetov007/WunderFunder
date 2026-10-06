# Public material on Wunder Alpha Connectome and earlier Wunder challenges (scouted 2026-10-02; untrusted, unverified)

## dozken/wunder (public 0.6569, #11) — https://github.com/dozken/wunder (challenge-3/README.md, RESUME.md)
- GRU 192x2, 64-128 input projection, first-difference inputs, TBPTT 1000-row chunks, AdamW cosine, EMA, SWA; 384x2 teacher only +0.003.
- Per-chunk Pearson loss subtracts the slow component the metric scores; moved to sequence-level Pearson.
- Data rank-transformed/clipped (mass at ±2.32); t0/t1 corr −0.74; lag-1 autocorr 0.96, ~0 by lag 500.
- Mask ~10% of rows, sticky runs (P(scored|prev scored)=0.77); a3/a2/a4 carry most mask signal; mask GRU AUC 0.94.
- Metric switched 2026-09-15 from per-sequence to pooled; rankings flipped. State resets every 4k rows helped them a lot (score decays along the sequence for their model; not for ours).
- A 2-GRU ensemble timed out at 69m59s => hard limit may be ~70 min. Local-to-public offset ~−0.03. Scaling predictions ×1.5 cost 0.004 (clip).
- Captured official docs: challenge-3/wunder_docs (metric spec "v2_wp_block_clip2_v1": mean over blocks of two-target WP; a block counts only if it has both signs).

## rahulb0802/alphac (public 0.6581) — https://github.com/rahulb0802/alphac (NOTES.md, reports/overnight-2026-10-02.md)
- GRU 2x128, dropout 0.3, wd 0.1, EMA 0.9995, 10-epoch OneCycle; MLP head +0.001.
- Mask weighting (0.3 + P(scored)) +0.0015; mask classifier AUC 0.90 on unseen sequences.
- No help: 2x192, aux targets, reweighting toward valid, longer TBPTT, post-processing. Seed members corr 0.985-0.994.
- Oracle per-sequence calibration +0.047 (in-sample; no causal version).
- Heaviest 10% sequences carry 27% weight, score 0.57 vs 0.66.
- Ablation: removing i0 volumes −0.058, i0 prices −0.054, i1 −0.038; a2/a3 predict mask, little target info.
- Claim: "test looks like train, valid is the easy calm outlier"; their train-holdout matched public within 0.002.

## Earlier Wunder challenges
- RNN Challenge 2025 (next-state LOB forecasting, R²): finals winner 0.3964; public repos (msrishav, KlishevDA 11th): GRU ensembles, mixed widths; deeper GRU, LSTM, TCN, attention failed.
- Predictorium (challenge 2): 1000-step LOB, weighted Pearson clip ±6; only mid-table repos.
