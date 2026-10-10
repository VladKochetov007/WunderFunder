I'll read the current pipeline, the round-3 notes, and the six-epoch learning curve, then propose eight long-run regimes from that evidence.The six-epoch curve is still rising while the learning rate is falling, and it breaks only in the annealing tail. End of epoch 2 is about 0.663 at roughly 5e-4, end of epoch 3 is 0.6671 at roughly 2.7e-4, end of epoch 4 is 0.662 at roughly 8e-5, and end of epoch 5 is 0.649 at the floor near 4e-7. Both targets fall together, and after epoch 3 the within-epoch swing collapses, so that drop is a real trajectory, not snapshot noise. Extra epochs on this one-cycle setting are extra steps into a sharper fit of the same 12k sequences.

Decoupled weight decay scales as learning rate times decay times step count. One epoch is about 7,460 steps. At learning rate 1e-3, weight decay 0.1 for 15 epochs is on the order of 10 nats of shrinkage and will underfit. A useful long run has a decay budget near 1–1.5 nats on the AdamW groups. Weight decay 0.01 at learning rate 1e-3, which you have now, is already about right for 3–4 epochs and too strong if you only stretch the same one-cycle to 20.

Shared settings unless an experiment says otherwise: width-256 2-layer GRU, Muon on the GRU matrices (5 Newton–Schulz steps) and AdamW on the input layer, head, and biases, weighted MSE with the metric weights, structure and next-change auxiliaries at 0.3, propensity weights, batch 32, chunk 1000, state carried and detached, fp16. Score fold 1. The probability below is my subjective chance that the pre-registered checkpoint (not the best snapshot of the run) lands at or above 0.670. Seed and fold noise is about 0.002, so these probabilities sit well below a coin flip even for the runs I like.

## 1. Branch the peak and refuse the tail

Start from `model_ep3.pt` weights only, with a fresh optimizer state.

Muon and AdamW both at a constant 3e-4 for 9 epochs, then a linear drop to 1e-5 over 3 epochs. AdamW weight decay 0.04 (about 0.9 nats). Muon weight decay 0. Step-wise EMA 0.999, started immediately. Same loss, batch, and chunk as the run that peaked.

The score was still improving at this learning rate. Epochs 4–5 are where the one-cycle schedule leaves that band. This asks whether more steps in the epoch-3 basin help when the learning rate stays at the level that made the peak. It is also the cleanest reading of “their longer runs worked”: a long flat phase, scored by an average, never enters the tail that hurt you.

Pre-registered checkpoint: EMA at the end of continuation epoch 9. **P(≥ 0.670) ≈ 0.18.**

Stop if two consecutive epoch-end EMAs sit below 0.663. The basin is already memorising and a flat learning rate does not save it. Also stop if the raw weights fall more than 0.008 below the EMA: averaging is then hiding a dying model, which is worth recording and not worth the decay phase.

## 2. From-scratch stable phase, parked at 4e-4

Warm up linearly from 0 to 4e-4 over half an epoch, hold 4e-4 for 12 epochs, then a square-root decay to 1e-5 over 1.5 epochs (14 epochs total). Muon multiplier 1. AdamW weight decay 0.03 on the non-Muon groups only (about 1.1 nats); Muon weight decay 0. EMA 0.999 from the end of epoch 2.

Each epoch, shift the BPTT cut by a fresh uniform offset in 0…999, covering every row exactly once. The forward pass still sees the whole sequence. Only the detach points move.

Epochs 1–3 of the baseline, while the learning rate travelled from about 1e-3 down through 5e-4 to 3e-4, contain the whole gain. This run simply stays in that band long enough for a slow creep, and the moving cut means a replayed sequence is not the same computation graph. EMA is part of the method: at a flat 4e-4 the instantaneous weights should wander, and the average is the model.

Pre-registered checkpoint: EMA at the end of the stable phase. **P ≈ 0.20.**

Stop if the EMA never reaches 0.662 by epoch 8, or if it falls for three consecutive epochs after having reached 0.664. That is the same cliff, shifted later.

## 3. Time-locked dropout on the encoder stream

Same schedule, optimizer, decay, EMA, and batch as experiment 2, extended to 18 epochs.

Draw one Bernoulli mask per sequence with keep probability 0.9. Hold it for all 20,000 rows. Apply it to the 256-d encoder output that enters the GRU, and apply the same mask to the encoder skip that the head concatenates with the GRU output. Redraw every epoch. No per-row dropout and no extra inter-layer dropout.

Per-row dropout is washed out by a GRU on highly repeated book rows, which is consistent with dropout moving you by less than 0.002. A mask that lasts for the whole sequence cannot be averaged away inside the sequence. The next epoch is the first time the network sees the dropped channels, so extra epochs buy new observations of the same books. The skip must be masked too, because the head can otherwise ignore the GRU and read the held columns directly.

Pre-registered checkpoint: EMA at epoch 14. **P ≈ 0.16.**

Stop if epoch-5 WP is below 0.645 (the skip was carrying the fit and 10% is too much). Also stop if train loss matches the baseline within 5% and the validation curve repeats the baseline peak-then-drop: the mask is then not on the path that learns.

## 4. Four cosine restarts, then average the cycle ends

16 epochs as four cycles of 4. Each cycle is a cosine from 8e-4 down to 1e-5 with a 10% warmup. Muon on the GRU, AdamW elsewhere, weight decay 0.02, multiplier 1. Do not reset weights between cycles. Average the four checkpoints taken at each cycle’s end, equal weights. That average is the served object.

The baseline falls into one basin during a single decay and stays there. A restart reheats the same weights into a different solution on a corpus small enough to memorize. The search space already contains SGDR, but only inside a 2–5 epoch cap, so it has never seen three real reheats. Snapshot averaging is also the one way a long run turns seed diversity into a single network.

Pre-registered checkpoint: the four-snapshot average. **P ≈ 0.15.**

Stop after cycle 3 if each cycle’s best epoch-end is at least 0.005 below the previous cycle’s peak. The restarts are reheating a dead basin. If the average trails the best single cycle by more than 0.004, the cycles did not diversify; keep the best cycle and do not start cycle 4.

## 5. AdamW on the recurrent matrices, same long stable phase

Identical schedule to experiment 2, but AdamW on every parameter, including both GRU matrices. Learning rate 8e-4, weight decay 0.015 on all 2-D weights (about 1.3 nats over the stable phase), EMA 0.999. No Muon.

Both targets fell together, which is what a shared recurrent representation does when its basis keeps being rewritten. Muon applies an orthogonalized matrix step every update. Over many replays that continues to rotate a representation that has already fit the forecast. AdamW’s curvature-scaled steps on `weight_hh` are the version of “train longer” that a non-Muon recipe would actually have run. Comparing Muon and AdamW under the original one-cycle tail will mostly show three optimizers cliffing in parallel.

Pre-registered checkpoint: EMA at the end of the stable phase. **P ≈ 0.12.**

Stop if epoch-4 WP is below 0.650. That is under-training, and the informative follow-up is a higher AdamW learning rate, not more epochs of this one. Also stop on the same three-epoch EMA decline as experiment 2.

## 6. Down-weight idle rows after the plateau

Experiment 2’s schedule for 18 epochs, with AdamW weight decay lowered to 0.02.

Through epoch 3, the loss is unchanged. From epoch 4 on, multiply the row weight by 0.25 when no event flag fires and the labels are unchanged from the previous row. Force the multiplier to 1 on any row with an event flag, any label change, and a halo of ±4 rows around a label change. The forward pass and the carried state still see every row. Log the mean multiplier on epoch 4.

A large share of the gradient on this book is “repeat the previous output” on clocks that did not move, replayed identically every epoch. The label is constant for about 9.5 rows and is set by the future path, so the rows that identify it are the updates and the boundaries. Idle rows mostly duplicate a level the GRU already holds in its state. Later epochs then spend their extra steps on the rows that can still move the fit.

Pre-registered checkpoint: EMA at epoch 14. **P ≈ 0.11.**

Stop if the mean multiplier is above 0.7: the fast clocks dominate, the reweight is a no-op, and the curve will track experiment 2. Also stop if full-metric WP is falling while correlation on event rows is rising. The metric’s mass is then in the plateaus, and this loss is aimed at the wrong rows.

## 7. Sharpness-aware updates on the head only

Muon on the GRU, one ordinary step, learning rate 4e-4, no SAM through the recurrence. SAM with ρ = 0.05 around AdamW on the input projection, the head, and the biases, at 3e-4, weight decay 0.04. Same warmup–stable–decay shape as experiment 2, 16 epochs. EMA of the weights after the outer SAM step.

A sharp head on a shared representation hurts both targets at once, and a finite set of repeated sequences is the setting where a flat head generalizes better than a minimized one. Keeping SAM off the GRU holds the run near the cost of