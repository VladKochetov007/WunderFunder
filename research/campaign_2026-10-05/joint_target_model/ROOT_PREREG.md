# E468: paired likelihood/readout plus an anchored residual

The independent critic found a real clipping mismatch: raw targets were used
to assign labels to a clipped codebook. This affected 6.35% of fitting rows.
The author corrected all label assignments and added an outlier test that
would fail under the old assignment. Root reviewed that correction, exact
checkpoint provenance, five control arms, and finite-update/resource guards.

Run all five arms on the existing frozen fold0 paired-state fitting export:
2,719,410 selected rows, 1,405 groups, two fixed historical GRU128 parents.
The 32 joint/16 marginal atoms are constructed from fitting targets only.
One fixed pass, blocks of 8192, AdamW lr .001/decay .01, seed271828, no tuning.
There are 332 batches per arm. Keep all five final epoch-zero checkpoints.

The target-specific weighted-Bayes, ordinary-mean, shuffled-pair and independent
readouts have exact zero-gate/zero-residual parent initialization. The regression
control receives the same auxiliary joint CE and parameter count. The common
MSE term is a clipped, shared-margin absolute-label-weighted **surrogate**;
the official score is the unchanged mean of two globally pooled correlations.
Finite loss/components/gradients/weights are checked at each update. Codebook
quantization, finite training and objective interference may all hurt; that is
an experiment result, not grounds for omitting an arm.

Then score all five once on the already exported component-separated fold0:
468 groups/141 components/808,863 officially selected rows. Verify fit/eval
manifest, both source checkpoint SHAs, export SHAs, row order and official mask.
Use the unchanged official scorer and 2000 component bootstrap replicates,
seed271828. Compare both against the unchanged historical parent pair and
against each other, retaining all results. This is an exposed historical fold,
not a score on the strongest 0.6660 public model or fresh independent validation.

The separate E2E wrapper is implemented and CPU-preflighted, but it is not part
of this frozen-state experiment; a changed trunk requires raw-sequence scoring,
not old exported hidden states. Do not infer E2E results from a frozen head.

All six CPU tests, an exact real-data parent-parity preflight and an evaluation
provenance preflight passed. The independent review is retained. Actual GPU
B8192/[8192,2,128]/128-wide-head smoke includes one AdamW step per arm and
completed exit0/teardown with 429,916,160-byte peak observed VRAM,
1,097,777,152-byte worker RAM and 3.77 seconds charged. No OOM increment.

One local GPU worker, two CPU quota, 6GiB RAM, 2GiB allocator, 3GB observed-device
stop, no swap. Fit cap900s; subsequent score cap900s. Sealed code/data contracts
and old guarded runner enforce the existing 80-device-GPU-hour campaign ceiling.
No new compute budget, competition upload, public forecast or automatic
follow-on training is inferred from this comparison.
