The gap that can still move 0.666 toward 0.700 is the sign on long class-3 plateaus. True sign with this model’s magnitude scores 0.843; the model’s own sign scores 0.685. That sign is right just after a change (~91%) and wrong often after age 512 (~74%), and class 3 keeps 73% of its weight in runs older than 16 rows. Every within-sequence family already lands on the same errors (correlation 0.95–0.99), extra width and extra epochs make things worse, and every causal edit of the output is closed. The bets below either read structure across columns and sequences that those fits never saw, or change the training process that teaches the small GRU its long-horizon sign.

## 1. Shared scenario library across sequences (data)

Probability ~10%. Many of the 12,000 sequences may be noisy replays of a much smaller library of latent trajectories, so the sign of a long plateau is borrowed from other sequences rather than filtered out of this one. A clock-normalized fingerprint of the opening rows retrieves training sequences in the same scenario, and their realized label paths become a row-wise sign target. That target is distilled into the existing width-256 GRU, which stays the only served model.

**Falsifier (1 day, no fit).** Fingerprint rows 0–500 only. Within class, take the 15 nearest training sequences. Go if their bucketed label paths correlate ≥ 0.15 with class-3 rows of age > 16, and a within-class shuffle falls to ≤ 0.03. No-go if the unshuffled correlation is ≤ 0.03.

**Expected effect.** Unknown. Upside +0.02 to +0.04 pooled if that correlation is real, because the errors sit on the sign channel.

**Fear.** The fingerprint only rediscovers the four market classes, which are already decodable at 99.4%, and within a class the futures are independent.

## 2. Slow columns are sample-and-hold copies of fast shocks (data)

Probability ~12%. The 12% and 15% blocks may be partial sums of the 31% fast-clock shocks since their last write, so their level wanders with the number of missed ticks and slowly rotates a GRU sign during a long hold. The part of a slow increment that the fast increments already explain is clock bookkeeping; the regression residual is the only new measurement. The student trains on fast innovations plus those residuals.

**Falsifier.** By sequence, regress each slow-block increment on the fast-block increments since the previous slow write. Go if median out-of-sequence R² ≥ 0.25. No-go if median R² ≤ 0.05.

**Expected effect.** +0.01 to +0.02 pooled if the label lives in the residual or in the cleaned fast innovation. About 0 if every block is an independent measurement that merely shares a time index.

**Fear.** High R² from a shared drift that is orthogonal to both targets, so cleaning the columns deletes nothing the metric uses.

## 3. Joint increments lie on a lattice (data)

Probability ~8%. Each column can look continuous while the joint increment of a block, on its own update rows, takes a few dozen discrete values. A piecewise-constant label is the trace of such a mode, and a GRU fit to analog levels drifts off the mode as a hold gets older. Decoding the lattice id causally and giving it to the same GRU supplies the mode the raw columns only imply.

**Falsifier.** On true update rows only, fit 64 clusters to one block’s standardized joint delta. Go if at least 90% of updates fall in clusters of radius ≤ 0.15 and the label-change rate differs by ≥ 3× between the top and bottom decile of clusters. No-go if within-cluster sum of squares is above half the total, or the change-rate ratio is < 1.3.

**Expected effect.** +0.015 to +0.03 if both the lattice and the change-rate split hold.

**Fear.** The clusters capture tick size or sparsity, and the label depends on the continuous residue inside each cluster.

## 4. Split the target across the four-row revelation lag (training)

Probability ~10%. The first row of a run scores −0.139 and the new level only becomes visible after about four rows, so those rows still show the previous level. Weighted MSE asks for the new level immediately, and that gradient teaches the state to treat lagging features as evidence for the new sign. On the lag window, supervise the previous level; supervise the new level only after the fast clock has had time to write.

**Falsifier.** One 3-epoch rerun of the current GRU, same width and budget. Go if fold-1 pooled WP ≥ 0.676 or class-3 WP ≥ 0.453. No-go if both gains are < 0.002.

**Expected effect.** +0.005 to +0.015 pooled. Smaller than the class-3 oracle, because most class-3 weight is not in those four rows.

**Fear.** Sign is already 90–92% just after a change, so the state recovers on its own, and dropping supervision on the boundary rows removes the only pressure that teaches a flip.

## 5. Frozen-label continuations from a clock-faithful simulator (training)

Probability ~8%. Class 3 is sample-starved (half the data costs 0.009) and its weight sits in long holds, while epoch 6 already overfits the short runs the optimizer sees over and over. Fit a feature simulator to each block’s real update clock, condition it on the true label held fixed, and extend real runs by thousands of rows. The student trains on those tails so slow-clock drift stops being read as a new sign; real change rows keep ordinary supervision.

**Falsifier.** Skip the learned simulator. Inside real runs, permute the order of the 12% and 15% updates, keep their values and the true label, and finetune one epoch from the current checkpoint. Go if class-3 sign accuracy at age > 256 rises by ≥ 2.0 points and pooled WP stays ≥ 0.667. No-go if the sign gain is < 0.5 point or WP falls below 0.664.

**Expected effect.** +0.01 to +0.02 on the class-3 mass if age decay is drift integration. Negative if the same intervention suppresses true late changes.

**Fear.** Late sign flips are net informative, which the smoothing results already show. A simulator that freezes the label will train those flips out.

## 6. Midpoint of an age-young and an age-old GRU (training)

Probability ~7%. Averaging predictions inside true runs gains +0.025 on class 3 and loses the pooled score, so young runs and old runs want different hidden dynamics, while a class-3-only model collapses (−0.14). Finetune two copies from the same checkpoint, one with loss only at run age < 8 and one only at age > 64, then serve the parameter midpoint as a single GRU. The early-run copy protects reactivity; the late-run copy protects the plateau sign; one weight vector has to carry both.

**Falsifier.** Two 1-epoch masked finetunes and one midpoint evaluation. Go if the midpoint reaches pooled WP ≥ 0.674, or class-3 WP ≥ 0.452 with pooled WP ≥ 0.666. No-go if the midpoint is below 0.659.

**Expected effect.** Unknown. Upside about +0.01 pooled when the two finetunes stay in one linear basin.

**Fear.** Masking the early rows starves the late model of the state that actually carries the level, so both poles move to a bad basin and the midpoint is worse than either.

## 7. Initial state chosen for the epoch-3 cliff (training)

Probability ~9%. The useful fit ends near epoch 3 and epoch 6 costs 0.017, so behavior past age 512 has to be present before the extra passes overfit the short runs. A hypernetwork reads only the opening causal window and sets the GRU’s initial state. Its outer loss is class-3 sign past age 256 after a fixed 3-epoch inner fit. The served artifact is still the small GRU; the hypernetwork is a training device.

**Falsifier.** Freeze the current GRU. Learn only a linear map from the mean of rows 0–32 to the initial hidden state, selected on class-3 age > 256 sign, then run the usual 3 epochs. Go if that sign accuracy rises by ≥ 2.0 points and pooled WP ≥ 0.674. No-go if the sign gain is < 0.5 point.

**Expected effect.** Unknown. Upside +0.01 to +0.02 if long-horizon sign is an initialization property that SGD on pooled weighted MSE never reaches before the overfit cliff.

**Fear.** The map from the opening window only rediscovers market class, and class conditioning of the output has already failed to move the score.

## 8. Evolve the gate biases, leave the features alone (training)

Probability ~6%. The fall from ~91% sign accuracy to ~74% is what an update gate does when integrated always-on columns leak in over hundreds of rows. Mean run length is 9.5, so almost all of the weighted-MSE gradient is spent on short runs and the biases that govern age 512 are barely identified. Evolve only the reset and update biases. Fitness is class-3 sign accuracy past age 256, with pooled WP required to stay within 0.005 of 0.669.

**Falsifier.** Population of 16, half an epoch each, non-gate weights frozen. Go if the best individual gains ≥ 3.0 sign points at age > 256 and scores pooled WP ≥ 0.664. No-go if every gain of 1 sign point costs more than 0.01 WP, or the best sign gain is < 1 point.

**Expected effect.** Unknown, and capped near +0.01 unless the age decay is almost entirely a gate-bias effect.

**Fear.** The decay is missing information about the true late change, so every bias setting that holds the sign also kills the informative flips and fails the WP constraint.

## Where I would stake the campaign

I would stake it on idea 1, the shared scenario library.

It is the only probe that can supply a sign for a long class-3 plateau from outside the sequence. That is the region that holds the weight, and it is the region where this model falls from ~91% to ~74%. Ideas 4–8 all retrain the same within-sequence map that four architectures already agree on to correlation 0.95–0.99. Idea 5 is the one I most expect to backfire, because frozen continuations punish the late flips that every smoothing test says are real. Ideas 2 and 3 are worth running the same day — they are cheap regressions — but a redundant clock or a lattice still has to be read causally inside one sequence, which is the setting that already looks information-limited.

The stake is falsifiable without a GPU fit, and the control is built in: neighbors have to beat a within-class shuffle on class-3 rows older than 16, which is the only place a library would pay for the metric. If that correlation is ≤ 0.03, the idea is dead and the campaign should not spend a distillation run on it. If it is ≥ 0.15, the sign channel is worth enough (true sign scores 0.843 with today’s magnitudes) that a mediocre distillation has a real path through +0.01.
