# Independent GALAM trace and ensemble audit

**As of:** 2026-10-28 14:32:30 UTC
**Audit:** read-only; no source edits, fits, GPU work, or remote actions.

## Scope and evidence

I read the canonical worktree's `AGENTS.md` and ADR-0044 before tracing the R9 training code. The canonical source is commit `50053909e365549b9146002d6c7d9118b81e9f97`, training-source SHA256 `f9685716d64ecb5ceef2ec821781fa4078ff21661201edfcdac8292867990230`. The worktree was clean during inspection. The authorized protocol fits only retained earlier-train members; the full later official valid (1,873 sequences, 3,528,273 `is_scored` rows) is used for checkpoint selection and stopping, so its scores are exposed development evidence. Do not treat train-eval versus valid as a gap: their masks are `all_after_warmup` and `is_scored` respectively.

The local live snapshot is `tabdata-online-training-metrics-20261028T1432Z.json`, SHA256 `118a2c66262bc240d67441471af19a430d29d2ecb93156bef25ac72a573bf79f`, observed at 14:32:30 UTC. The older receiving proof at `launch/receiving/results-20261028T1310Z/` records complete valid coverage and SHA-checked checkpoint pointers **as of 13:10**; its README explicitly says it did not independently rescore saved prediction arrays. It is not a byte proof of the later 14:32 pointers.

The implementation is a schematic adaptation, not a verified exact reconstruction of GALAM. `src/wnn/galam/config.py:27-40` lists its width/head counts, block counts, normalization convention, fusion details, time scale, and Connectome output adaptation among unresolved details.

## Current fit evidence

| Cell | Snapshot state | Checks | Best full-valid WP | Last WP |
|---|---|---:|---:|---:|
| g0 plain GRU256x2 | early stopped | 10 | 0.677856 (epoch 8) | 0.669716 (epoch 10) |
| g1 GALAM LSTM64 | running | 3 | 0.669442 (epoch 3) | 0.669442 |
| g2 GALAM LSTM128 | running | 3 | 0.672369 (epoch 3) | 0.672369 |
| g3 GALAM depth2 | running | 3 | 0.670843 (epoch 3) | 0.670843 |
| g4 GALAM GRU72 | running | 4 | 0.671358 (epoch 3) | 0.670439 (epoch 4) |
| g5 GALAM history256 | running | 4 | 0.670726 (epoch 3) | 0.670538 (epoch 4) |
| g6 GALAM TBPTT256 | running | 4 | 0.663927 (epoch 4) | 0.663927 |
| g7 GALAM hybrid Muon | running | 4 | 0.669924 (epoch 3) | 0.669612 (epoch 4) |

Every GALAM arm improved over epochs 1-3; g6 also improved at epoch 4. g4, g5, and g7 had a small epoch-4 decline. All GALAM runs had only three or four checks and had not reached the declared minimum stopping epoch of five. That is evidence of remaining optimization time, not a forecast of final score. The g0 comparison is intentionally unmatched: event211 input, Muon, LR 1e-3, batch 32, chunk 1000 versus GALAM raw112, AdamW, LR 1e-4, batch 384, chunk 64. It cannot rank the architectures.

The cleanest within-panel contrasts are g1/g5 (only `attention_history` changes from 64 to 256) and g1/g6 (only `chunk_rows` changes from 64 to 256). At epoch 3, g5 was +0.001284 WP over g1. This is a promising early history-length signal on the exposed valid. g6 also changes optimizer-update cadence: with `accumulate_chunks=1`, its 4x longer chunk yields about 4x fewer updates per sequence epoch. It is therefore a combined gradient-horizon/update-frequency intervention, not a pure context comparison.

## Confirmed implementation behavior and risks

**BatchNorm train/eval mismatch is the clearest issue to falsify.** `PerTimeBatchNorm1d` wraps default `nn.BatchNorm1d` without an explicit running-stat policy (`src/wnn/galam/mixer.py:10-24`). GALAM loops over time in Python and calls its input norm once per step; the LSTM receives a length-one sequence (`src/wnn/galam/model.py:103-124`). With 20,000 rows, each such BatchNorm buffer updates about 20,000 times per full sequence batch. PyTorch's default momentum 0.1 gives an e-fold of roughly 10 updates, so eval running moments are dominated by the final few dozen time positions in the final fitting batches. Training normalizes with the current batch's per-time statistics, while evaluation calls `model.eval()` and uses those running moments (`src/wnn/training/sequence_monitor_evaluation.py:27-61`). Evaluation correctly asserts those buffers do not change. This behavior is confirmed; its effect on WP is not. It is not future-label leakage.

Causal state and TBPTT boundaries look correct. Attention attends only to cached past keys plus the current key, trimmed to `history` (`src/wnn/galam/attention.py:52-80`); the convolution mixer carries a left-only bounded tail (`src/wnn/galam/mixer.py:27-52`). State is retained across chunks and detached at each chunk boundary (`src/wnn/training/sequence_loop_window.py:42-62`). Thus g5 carries up to 256 values but backpropagates only 64 rows; g6 backpropagates 256 rows while attention still sees 64 keys. I found no confirmed causal-window or detach defect.

The optimizer loss is a deliberate surrogate. Each accumulation window clips targets to +/-2, sums squared raw-prediction error weighted by absolute clipped target, and divides by that window's total label mass (`src/wnn/training/sequence_objectives.py:17-66`). The official metric clips both predictions and targets, computes weighted Pearson separately for each of two outputs over the full cohort, then averages the two scores (`src/wnn/gmetric.py:23-40,159-183`). This keeps target weighting broadly aligned but is not equivalent: per-target correlation gets equal final weight while pooled MSE gradients follow each target's window label mass. Logged prediction clipping is small in this snapshot (0.01%-0.11% on valid), so clipping mismatch is secondary. Frequent norm clipping is observable: 9-12 of each arm's last 12 sampled updates clipped at threshold 1.0. That makes the threshold worth measuring in a future controlled recipe; it does not prove the threshold is harmful.

Attention and fusion diagnostics show plausible leads, not defects. The last sampled window for history256 has local/global mean absolute lags about 96/82 with entropy 4.52/4.56; history64 arms are around lags 23-33 and entropy 3.2-3.7. The longer arm is using a broader context. GALAM LSTM128 has 61.0% fusion gate values outside [0.05,0.95] (mean 0.212), much higher than the other GALAM arms' 0.02-14.4%; its current WP is nevertheless the best GALAM score. Check channel-specific gate and branch contribution trends before calling this saturation harmful. Sampled LSTM input/forget/output gate saturation is low (roughly 0.5-2.1%); candidate tanh saturation is roughly 1.7-5.9%.

Sampled hidden/cell-state gradient RMS is finite and nonzero: LSTM64 `1.42e-5/7.84e-6`; LSTM128 `1.54e-5/8.62e-6`; depth2 `2.05e-5/1.24e-5`; history256 `1.70e-5/1.26e-5`; TBPTT256 `4.40e-6/2.47e-6`; hybrid Muon `1.67e-5/1.11e-5`. These are one sampled window per asynchronous run, not a gradient-decay curve. TBPTT256's lower RMS is confounded by its 4x longer chunk and correspondingly larger per-window loss denominator. The `diagnostics/missing_gradient_output_count` values of 64 (or 256 for TBPTT256) are expected: the sampler also hooks the recurrent sequence output, but GALAM intentionally discards that return as `_, (hidden, cell) = self.lstm(...)`; they are not missing hidden/cell gradients. The diagnostic implementation is in `sequence_monitor_diagnostics.py:141-182` and `sequence_monitor_gates.py:12-70`.

**Throughput has a credible Python/launch-overhead hypothesis but no attribution yet.** `GALAM.forward_chunk` dispatches `stream_step` in a Python loop for every row (`model.py:103-108`). At 20,000 rows this is 20,000 recurrent/attention/mixer dispatches per sequence batch. In the 14:32 telemetry, last-12 mean update-window times are 0.93-3.64s, while optimizer times are 0.8-20ms; latest GALAM CPU samples are about one logical core. The update-window timer starts after batch acquisition (`sequence_loop.py:80-126`), so it excludes at least some input/prefetch wait. This points to a profiling question, not a measured bottleneck. The 13:10 proof's 27-second GPU-busy average is likewise a short sample, not whole-run utilization or FLOPs.

## Suitability, ensemble use, and fastest falsifiers

GALAM is a plausible causal sequence-regression candidate: it combines a recurrent local path, bounded causal attention in local/global memories, a causal convolution mixer, and gated fusion (`galam/model.py:13-52`, `galam/fusion.py:11-48`). It has not yet shown that this complexity beats the simpler system: the full-valid scores are early, exposed for selection, single-seed, and differently optimized. Its actual original architectural details remain unverified. Treat it as a candidate contributor to an ensemble, not a proven better standalone model.

Before judging an ensemble, retain/export aligned predictions from terminal best checkpoints and measure pairwise residual correlation plus a predeclared fixed equal-weight blend. Do not tune membership or weights on this already-used valid. Learned blend weights require fitting-only predictions with real chronological evidence; do not create random folds or infer order from `seq_ix`. The reviewed 13:10 proof package contains scalar curves/checkpoint digests, not prediction arrays, so it cannot establish complementarity.

1. For the BatchNorm question, clone a frozen best checkpoint and compare its original eval buffers with a no-gradient recalibration pass using only fixed fitting members; record per-time input/activation moments. Report the full-valid comparison as an exposed diagnostic, not an independent confirmation.
2. Read the complete monitor diagnostic history, not just each cell's last sample. Track hidden/cell state-gradient RMS across windows and per-channel fusion gates, relative-bias spread, attention lag/entropy, and clipping rate. This can falsify persistent saturation or gradient collapse without changing the running fits.
3. Profile one representative actual-source training window and one evaluation batch with CPU/CUDA timelines separating batch materialization/prefetch/H2D, per-time Python dispatch, forward, backward, and optimizer. Do not infer attribution from utilization alone.
4. Once terminal predictions exist, compute fixed-blend and residual-correlation diagnostics on precisely aligned scored rows. Keep any result marked exposed because the same official valid controlled stopping and checkpoint selection.

No code defect beyond the confirmed BatchNorm running-stat convention risk is established by these traces, and no eventual score gain is claimed.
