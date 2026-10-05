# E485 proposal: activation shape across training duration

**Design only.** Compare a fixed bounded recurrent-candidate sine shape with the native GRU candidate as training proceeds. This asks whether an early shape effect persists or changes with additional optimization; it is not a convergence or public-score claim.

## Matrix and fixed recipe

Run six trajectories: baseline `u=tanh(v)` and sine `u+0.1 sin(pi*u)`, each at initialization seeds `271836`, `271837`, and `271838`. Use `RNNModel(42, hidden=128, layers=2, dropout=0, n_out=2)`, retaining the exact reset-after GRU equation, parameter layout, and state keys from E478. The model has **236,802** trainable parameters. For each seed initialize once, clone identical parameters into both shape arms, and use fresh optimizer state for each. Apply the candidate formula only at the GRU candidate nonlinearity; keep sigmoid gates, embedding/head GELUs, and all other operations unchanged.

Train each trajectory for four complete epochs (474 updates per epoch; 1,896 per trajectory) on the frozen 24 fit components. Batch size is 4, chunk length 256, with recurrent state carried and detached over the 79 chunks and reset between batches. For zero-based epoch `e=0..3`, use the shared paired order `default_rng(seed + e).permutation(sorted_fit_ids)`; both shapes receive the same order. Keep weighted-MSE/mask semantics, FP32 parameters and inputs, full IEEE precision, global gradient clip 1.0, constant LR 0.001, and weight decay 0.01. Muon applies only to the four GRU recurrent matrices with three gate blocks and beta1/momentum 0.9; every other parameter uses AdamW betas `(0.9, 0.999)`. No schedule, dropout, EMA, distillation, or head fine-tuning.

Save checkpoints after epochs 1, 2, and 4: **18 snapshots from six continuous fits**, not 18 independent fits. Preserve fit-only normalization, fit row order, optimizer/update and clipping receipts, source hashes, and the exact model/data identities. The 16 historically exposed components may be scored only as paired diagnostics after all six trajectories finish; no checkpoint or recipe selection is allowed from them.

## Registered analysis

For each seed, compute GlobalWP differences with paired resampling of numeric-sorted components and 2,000 shared bootstrap draws. Keep seed-specific intervals; do not pool seeds.

- Shape effect: sine minus baseline at epochs 1, 2, and 4 (9 contrasts).
- Duration effect: epoch 2 minus epoch 1 and epoch 4 minus epoch 1 for each shape (12 contrasts).
- Shape-by-duration interaction: `(sine_4 - sine_1) - (baseline_4 - baseline_1)` (3 contrasts).

This is 24 contrasts total. Report all snapshots and target-wise scores. Component intervals condition on each trajectory and do not represent initialization or unseen-market uncertainty.

## Cost, admission, and interpretation

The matrix costs 11,376 optimizer updates and 11.52 million fit-row positions, plus the diagnostic passes. First require fit-only CPU checks for all six baseline/sine one-update pairs and an actual-shape GPU probe covering parity, finite/clipped/changed updates, and memory. One GPU worker only; probe/worker limits: 2 CPUs, 6 GiB host RAM, 2 GiB allocator, and stop at 3 GB observed VRAM. Parent limits remain 12 CPUs, 30 decimal GB RAM, and no swap. Admit at most 7,200 GPU seconds from the existing local budget, after E483/E484 priority and root review; stop new admissions at 07:00 UTC. A measured probe sets timeout; do not infer duration from smaller-width runs. No provider or ledger changes are part of this proposal.

If the sine-minus-baseline effect shrinks from epoch 1 to epoch 4, this supports an early-optimization interaction under this optimizer and schedule, not equal asymptotic performance. If it persists, four epochs still do not establish convergence. If trajectories diverge or outcomes vary by seed, report that heterogeneity without naming a winner. This H128 screen is neither the H256 incumbent recipe nor an unseen-component validation. Obtain an independent blind design critique before implementing source or running any fit.
