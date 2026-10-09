# Native GPU FP16 benchmark contract R1

This candidate measures the project's native PyTorch/cuDNN CUDA FP16 path on
the same exposed fold-0 screen as the completed GPU D01/R5 FP32 control. It is
not admitted for upload or execution until root binds the exact source closure
and R5 controls in a native root-admission record.

The control is the reviewed completed R5 training contract
`5b63ed39aa6046e9f02878e7dcb4b48982240bffa45ec09b7466dfb9eeac63f7`, with
root terminal review
`60e9770de432a6b39d3ae41c818d60b2d47b5298f630b33ee3483a0e7fecc7f9`. The
native worker consumes those R5 artifacts and independently rechecks their
producer contract, root review, all 14 shared-role hashes, actual profile
fixtures, initialization, normalization, sample order, and post-fit schedule.
The producer consumer checks also run against the actual R5 checkpoints, update
traces, NPY prediction headers, finite prediction chunks, and ordered
membership sidecars.

The only planned precision change is CUDA FP16 autocast in the existing
`train_rnn.py` trainer. Parameters remain FP32 master weights and AdamW's
floating accumulators remain FP32. CUDA TF32 stays disabled. The one predeclared
GradScaler initial scale is 1024; ordinary dynamic growth/backoff remains
enabled. There is no scaler sweep or post-hoc scaler selection. A standalone
actual-T4 profile must complete 20/20 finite updates with no scaler skips
before the worker may consider either loss fit. Every full arm records 220
attempted updates, completed updates, skips, and scale range; skipped updates
never count as throughput, and any skip invalidates the fit artifact for this
matched comparison.

The fixed recipe remains the R5 recipe: seed 20261009; 350 whole fit sequences;
28 retained fit-validation sequences unused for fitting; fixed exposed
evaluation membership of 468 sequences; 211 features (112 raw plus 99 causal
event features); two-layer GRU, hidden width 256, linear input projection,
GELU, and two outputs. Both WMSE and Huber delta 1 arms use AdamW, OneCycleLR,
maximum learning rate 0.001, weight decay 0.01, beta2 0.999, epsilon 1e-8,
global clip 1, per-chunk loss normalization, batch size 32, and chunk length
2,000 for two epochs. Inputs, target clipping, target weights, recurrent reset
per sequence, carried state within the sequence, update clock, and optimizer
settings stay fixed. The final GPU fit batch contains the 30 remaining real
sequences with drop-last disabled; no synthetic sequence is added. Only
autocast precision and its declared scaler state differ from R5.

Before any conditional fit, the worker runs the actual CUDA H256/B32/C2000
profile and a separate lifecycle check using R5's real first-batch fixture and
initial tensors. The lifecycle check records device/software identity, finite
outputs and states, repeatability after sequence reset, state save/reload,
prefix and carried-prefix agreement, CUDA-event first-forward time, warmed
forward timings, and device memory. The profile records its own first-update
time (including lazy CUDA/cuDNN setup), median repeated update-event time,
attempted/completed/skipped updates, gradients, loss finiteness, and sampled
host/device resource peaks.

Conditional fit headroom is measured after the standalone profile, from live
samples: at least 4 GB host `MemAvailable`; at least 4 GB cgroup headroom when
the cgroup is bounded, or an explicitly observed unbounded cgroup plus the host
reserve; at least 4 GB working-disk free; and at least 2 GB free on each of the
two actual GPUs. Missing measurements defer the fit while preserving a passed
profile receipt. Existing worker cgroup, output, disk, and wall guards remain
active. This is a cloud-device policy, not a local 20 GB assumption.

The two same-shaped loss arms run concurrently on the two admitted T4 devices,
as in D01. Each fit's time and aggregate pair wall time are separate. Cold
source unpack, cache verification, CUDA/cuDNN lazy setup, standalone profile,
warmed training updates, full fit/checkpoint export, prediction export, and
whole worker wall time are recorded separately. The worker retains raw final
weights, epoch checkpoints, full FP32-master/AdamW/scaler continuation state,
per-update gradient/LR/beta1/clip/AMP-skip traces, exact sample/schedule
artifacts, predictions, membership sidecars, resource samples, and device and
software inventory.

The exposed fold-0 screen is not public transfer evidence. The result requires
the frozen independent grader on the same 468 IDs and official
`global_wp_clip2_v1` metric. Report observed curves only; do not infer FLOPS or
interpolate a time-to-score from sparse endpoints. Root owns admission,
provider actions, independent review, and scientific synthesis.
