# LR and data-scaling literature for rescaling GRU training (N = 10.6k to 12.5k sequences)

Scope: read-only literature check, 2026-10-11. Sourced claims cite the section or abstract I actually read. Items marked **[inference]** are my derivations or extrapolations, not claims from a source. Several sources were read only at abstract level; those are flagged.

## 1. AdamW learning rate vs batch size

- **Sourced.** Malladi et al. 2022, "On the SDEs and Scaling Rules for Adaptive Gradient Algorithms" (NeurIPS 2022), [arXiv:2205.10287](https://arxiv.org/abs/2205.10287). Section 5, Definitions 5.1 and 5.2: when batch B becomes kB, set eta' = eta * sqrt(k), eps' = eps / sqrt(k), and 1 - beta' = k(1 - beta) (for Adam, both beta1 and beta2). Under this rule the SDE approximation is unchanged. The derivation assumes gradient noise dominates the mean gradient (Section 4.2) and a warm-start phase (Remark 5.4). The paper does not discuss weight decay for this rule.
- **Sourced.** Linear LR scaling for SGD with warmup: Goyal et al. 2017, [arXiv:1706.02677](https://arxiv.org/abs/1706.02677) (abstract-level read). Linear scaling is the SGD rule, not the Adam rule.
- **Sourced.** Recurrent nets: naive batch-size LR scaling fails for large-batch LSTMs (Puri et al. 2018, [arXiv:1808.01371](https://arxiv.org/abs/1808.01371)). LEGW (You et al. 2019, [arXiv:1901.08256](https://arxiv.org/abs/1901.08256)) proposes linear-epoch warmup for LSTMs. No paper I found measures GRU LR vs batch.
- **[inference]** Our batch is fixed at 32 sequences, so neither rule changes LR for the 1.18x data increase. The sqrt rule only matters if we change batch size. Malladi is about batch, not dataset size.

## 2. Muon LR scaling, update-RMS matching, width transfer

- **Sourced.** Moonlight, "Muon is Scalable for LLM Training," [arXiv:2502.16982](https://arxiv.org/html/2502.16982v1), Section 2.2, Eq. (4): W_t = W_{t-1} - eta_t (0.2 * O_t * sqrt(max(A,B)) + lambda W_{t-1}). The 0.2 factor targets an update RMS of about 0.2 (AdamW's typical range is 0.2 to 0.4); Appendix A reports 0.2 and 0.4 performed similarly. Weight decay is applied outside the scaled bracket, decoupled as in AdamW. This makes Muon's LR and weight decay directly comparable to AdamW's.
- **Sourced.** Keller Jordan's original Muon uses factor sqrt(max(1, out/in)). PyTorch's `torch.optim.Muon` exposes "original" (default), "match_rms_adamw" (the Moonshot 0.2 sqrt(max) rule), and "spectral_unclamped" options, per the PyTorch docs (found via search; not fetched). NVIDIA Emerging Optimizers and Optax implement similar variants. These differ by a shape-dependent factor, so an LR tuned under one convention does not transfer to another.
- **Sourced.** Muon width transfer is approximate. "Spectral Condition for muP under Width-Depth Scaling" ([arXiv:2603.00541](https://arxiv.org/pdf/2603.00541)) reports approximate LR transfer for Muon+AdamW. "Practical Efficiency of Muon for Pretraining" ([arXiv:2505.02222](https://arxiv.org/pdf/2505.02222)) argues the optimum shifts by O(1/n) at finite width. "Understanding the Mechanisms of Fast Hyperparameter Transfer" ([arXiv:2512.22768](https://arxiv.org/html/2512.22768v1)) finds Muon's loss reduction is less concentrated in a width-stable subspace than Adam's. These are search-snippet-level summaries.
- **[inference]** Width transfer is irrelevant for our N change at fixed hidden size. It matters only if we move hidden 256 to 384 with a fixed Muon LR.

## 3. Weight decay scaling

- **Sourced.** Wang and Aitchison, "How to set AdamW's weight decay as you scale model and dataset size" (ICML 2025), [arXiv:2405.13698](https://arxiv.org/html/2405.13698v2). Section 3.1, Eq. (6): EMA timescale in iterations tau_iter = 1/(eta * lambda). Section 4.1, Eq. (10): tau_epoch = tau_iter / M, where M = N/B iterations per epoch. Eq. (11): lambda = 1/(eta * M * tau_epoch). The claim is that the optimal tau_epoch is roughly constant across dataset size, so optimal lambda falls as N grows at fixed eta and B. Section 4.2: under muP, eta = eta_base/s and lambda = s * lambda_base keeps tau_iter fixed in model width. Conditions: fixed batch, cosine schedule, no decay on normalization layers. The theory is proven for scale-invariant networks (Section 3.2, Appendix A); the authors needed architectural modifications in experiments (Appendix A.3). Guideline from the paper: the EMA timescale should not be much smaller than one epoch nor much larger than total epochs.
- **Sourced.** Bergsma et al., "Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training" ([arXiv:2505.13738](https://arxiv.org/abs/2505.13738)), abstract-level read. Optimal timescale tau = B/(eta * lambda * D) follows a power law in D/N; optimal lambda scales linearly with B at fixed N, D. Their EMA horizon is expressed as a fraction of training.
- **[inference]** Applying Wang and Aitchison Eq. (11) at fixed B = 32 and fixed epochs: M is proportional to N, so lambda should scale as 1/N. Going from 10.6k to 12.5k sequences (x1.18): lambda_new = lambda_old / 1.18 (about 0.85x). For our 0.01, that gives about 0.0085. This is a single-source extrapolation to a non-scale-invariant GRU with Muon, so it needs measurement.

## 4. Dataset size: epochs, steps, warmup

- **Sourced.** Muennighoff et al., "Scaling Data-Constrained Language Models" (NeurIPS 2023), [arXiv:2305.16264](https://arxiv.org/abs/2305.16264), abstract: up to about 4 epochs of repeated data gives loss nearly identical to unique data; beyond that, the value of added compute decays toward zero. Fitted forms were read only at abstract level.
- **[inference]** Our 3 epochs sits below the ~4-epoch repetition threshold, and the scale-up keeps epoch count fixed. So repetition does not change, and the cost of more data is just more steps: about 18% more total steps (3 epochs x 12.5k / 32 steps instead of 10.6k). Keep 3 epochs. Scheduler warmup and decay should be defined as fractions of total steps so they stretch with N. Warmup in absolute steps should be scaled by 1.18.
- **Sourced.** Wang and Aitchison (above) notes that for a fixed EMA window in epochs, the optimal lambda falls with dataset size. Their analysis holds epochs and batch fixed, which is our setting.

## 5. EMA / SWA windows

- **Sourced.** Wang and Aitchison Section 4.1 (above): the EMA window should lie between about 1 epoch and the total epoch count.
- **Sourced.** Power Lines (above) defines the averaging window as a fraction of training.
- **[inference]** Define EMA and SWA windows in epochs, not steps. For 3 epochs, a 0.5 to 2 epoch window is inside the paper's guideline. The step count grows about 18%, but the epoch window stays fixed.

## 6. Practical notes for small recurrent nets and noisy regression

- **Sourced.** No paper I found studies GRU LR vs batch on noisy financial-style regression. The nearest sources are the LSTM large-batch papers above and a small GRU batch-size study in a GNSS time-series setting (search snippet only).
- **[inference]** Keep batch fixed and do not apply the sqrt or linear rules, because the justification is batch-driven. TBPTT chunk length and sequence count interact with gradient noise, so keep chunk length fixed across the rescale.

## Summary table

| Rule | Optimizer | Formula | Implication, N 10.6k to 12.5k (x1.18), 3 epochs fixed | Confidence |
|---|---|---|---|---|
| Malladi sqrt batch rule | AdamW (Adam family) | eta' = eta sqrt(k), 1-beta' = k(1-beta) | None, because batch fixed at 32 | High for batch change; not applicable here |
| Goyal linear rule | SGD | eta' = eta k | None | High (SGD only) |
| Moonlight RMS match | Muon | 0.2 sqrt(max(A,B)) O, decoupled lambda | Keep the same convention as the fold-1 runs; LR unchanged | High for formula; convention must match |
| Keller sqrt(max(1, out/in)) | Muon | eta * sqrt(max(1, out/in)) | Same as above; not interchangeable with Moonlight | High for formula |
| muP width transfer, approximate | Muon+AdamW | LR roughly invariant under width scaling | Not relevant at fixed hidden size | Medium |
| Wang and Aitchison timescale | AdamW (Muon: inference) | tau_epoch = 1/(eta lambda M); lambda ~ 1/N at fixed B, epochs | lambda: 0.01 to about 0.0085 | Medium (single source, GRU untested) |
| Power Lines | AdamW | tau = B/(eta lambda D), power law in D/N | Same direction as above | Medium (abstract only) |
| Muennighoff repetition | Any | ~4 epochs of repeated data is near-unique | Keep 3 epochs; step count up ~18% | Medium (abstract only) |
| EMA window in epochs | Any | 1 epoch to total epochs | Keep window in epochs; steps up ~18% | Medium |
| LEGW / warmup scaling | LSTM | linear-in-epoch warmup | Scale warmup steps by 1.18 or keep fractional | Low-medium |

## Proposed measurement protocol

Run on the fold-1 setup, since fold-1 exposure rules still apply to any held-out claim. Use fixed evaluation membership and at least 3 seeds per arm.

1. **Size-response arms.** Train on 85% and 100% of fold-1 training sequences, fixed 3 epochs, batch 32. For each size, sweep lambda in {0.5x, 1x, 2x} and LR in {0.7x, 1x, 1.4x}. Check whether the best lambda at 85% is about 1.18x the best at 100%, as the 1/N rule predicts.
2. **EMA window arms.** At 100% size, EMA windows of 0.5, 1, and 2 epochs, defined in epochs (not steps).
3. **Warmup arms.** Warmup as fraction of total steps vs absolute steps; compare at both sizes.
4. **Muon convention arms.** Moonlight 0.2 sqrt(max(A,B)) vs Keller sqrt(max(1, out/in)) at a matched LR grid. Confirm that the best LR under one convention is not the best under the other before any transfer claim.
5. **Epoch arm.** 3 vs 4 epochs at full size, to check the repetition threshold for this recipe.

Success criterion: the optimum from the size-response arms moves in the direction predicted by Section 3 and is consistent across seeds. If it does not move, keep fold-1 hyperparameters at full size and report the null result. Public score and fold-1 exposure rules remain authoritative.
