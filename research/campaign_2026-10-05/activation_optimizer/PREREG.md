# E470: fixed activation × optimizer matrix

All8 activations × bothAdamW/MuonAdamW =16 fits, seed271828. Same exposed
24-fit/16-test feature-component cohort and immutable E467 numerical recipe:
20,000rows/sequence, B4/chunk256, 474updates, width32, oneepoch, target-clipped
per-target mass-weighted MSE, initial zero residual head, official clipped
GlobalWP. 8,235parameters each; no arm removal or winner-based retuning.

Activation changes the input embedding and residual-head nonlinearity together.
GRU internal sigmoid/tanh gates stay unchanged. Paper-inspired pointwise
GELUSine(.1) and normalized GELUSinc(.5), additional centered-cosine residual,
and channel-splitGELU/LReLU are parameter-free. No batch/time statistics.
Muon uses the unchanged repository implementation on three-block GRU gate
matrices; everything elseAdamW. LR.001/wd.01/betas(.9,.999), constant schedule.
This compares fixed recipes, not optimizer families after equal-budget tuning.

Fit-only actualGPU B4T256 one-update probe per16arms must pass before full
fits. Root independently matched tanh/AdamW initialization, forward prediction
and actualAdamW step exactly against E467. Separate implementation critic
checks all activations, parameter partitions and causal/scoring boundaries.

One localGPUworker,2CPU/6GiBworker RAM/2GiBallocator/3GBobservedGPUstop; parent
12CPU/30decimalGB hardRAM/27GBsoft/no swap. Probe timeout180s; fullfit1200s.
Existing80device-GPU-hour ceiling and current24GPU-job-hour block remain
unchanged. Projected0.05–0.15GPUh; retain actual cost/peaks/failures. Wait for
E469worker teardown. Failed probe closes this attempt; no automatic retry.

Compare each activation againsttanh withinoptimizer and compareMuon/AdamW
withinactivation; bootstrap2000components conditional on the fixed model.
No fresh-validation/public-score claim and no conversion of pilot gains into
an expected0.7000. Save all weights, predictions and source/cohort hashes;
images undereda/ and marimo. Subsequent activationschedules/dropout/placement
controls are separate preregistered work, not implicit changes to this matrix.
