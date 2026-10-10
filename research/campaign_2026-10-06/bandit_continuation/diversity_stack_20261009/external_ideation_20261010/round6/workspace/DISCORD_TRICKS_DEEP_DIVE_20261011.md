# Discord tricks (sibling market-stream challenge, writeups/discord_tricks.txt) mapped to our evidence
Caveat: sibling task = 32 raw features, R2 metric, 517 sequences, Transformer/GRU; priors for transfer are weaker than for our own ablations. Before 2026-10-11 these were only read and passed to ideators (round 5); no dedicated deep dive. This is that deep dive.

| Trick (who) | Claimed gain | Our status / evidence (fold-1 WP, ctrl 0.6656-0.6662) | Verdict / next |
|---|---|---|---|
| SWA of last 3-5 ckpts (+0.002-0.0025) | big | tr_swa 0.66576 vs tr_base 0.66559; HPO top trials all use SWA | tested, ~+0.0002, no |
| post-hoc EMA (+0.005 w/ triple ensemble) | big | ema/ema4/ema6/mu3_ema all <= control | tested, no |
| chrono init GRU/LSTM "huge boost" | big | chrono 0.66481, chrono_orth 0.66045, chrono8k 0.66348 | tested, negative |
| Highway head vs linear | big | NOT tested | implement (--head highway) |
| 2-stage boosting on GRU/LSTM (2nd predicts error of 1st) | yes | GBM-on-features failed (E-093, H-347); GRU residual stage NOT tested | implement residual-stage distill |
| MAE/next-step pretraining on whole dataset (+leak of val inputs) | #4 LB | aux-next/fcast auxiliaries only (fcast_w0.1 0.66532, null); separate pretrain stage NOT tested | implement pretrain-then-finetune (inputs only) |
| Frozen GRU next-step feature extractor fed to main model ("very significant") | big | NOT tested | implement |
| Per-sequence adaptive ensemble weights by MSE | yes | blend_probe_20261011: gates/ridge/lookup all <= static; ridge held-out -0.003 | tested, no |
| Weak diverse member (bidir LSTM over 60 steps) | helped ensemble | lstm256 0.66581 (== GRU); fixmix 0.66817 (h384) -> pair gain exists via diversity | partly: diversity pairs +0.003 |
| Data aug: median-variance rescale + random stretch/compress, 5x data | yes | NOT tested (thin-rows is different) | implement causal time-warp aug |
| Train on entire dataset + early stop epoch from val | strong | we already train all-data finals (z1c/z4/z5) | done |
| Difficulty-stratified split | stabilises | not needed (component-disjoint folds) | skip |
| PCA-cluster plane-distance feature | small | class features in heuristics (heur_*), flow_v2 74 feats neutral | skip |
| Sigmoid/tanh/SiLU/GELU of raw inputs to diversify members | diversity | NOT tested | cheap variant: --input-act |
| Anti-curriculum / harder sequences upweighted | maybe | NOT tested | implement sampler |
| Muon/AdamW/Ranger optimizers | slight | Muon is our base; muon scope variants all null | done |
| torch.compile for speed | speed | not used; training speed not priority | maybe later |
| FiLM ensemble of 4 int8 models | +0.003 | static pair +0.003 local; FiLM/gates no better (blend probe) | static avg |
| Ensembling dominates (+0.003-0.0035 for 4) | yes | confirmed (pair +0.003 fold-1) | main lever, serving-bound |
## Net
4 of the 6 "big" claims are already negative/neutral here; ensembling confirmed. Untested and plausible: highway head, pretrain-then-finetune, frozen next-step extractor, residual stage, time-warp aug, anti-curriculum, input activation diversity. Implementer agent assigned; experiments run as backlog arms (fold 1, 3 seeds before any lead).
