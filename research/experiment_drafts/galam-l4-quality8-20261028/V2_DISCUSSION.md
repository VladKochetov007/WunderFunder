# Proposed amendment: common20-epoch cap and a hybrid optimizer arm

28 October2026, user consultation following preparation commit f3e7f3cd.
Proposal only: the eight checked JSON recipes and their hashes remain unchanged.
Zero new profiles/fits. No server access or mutation in this discussion.

Use a common maximum20epochs rather than spending GPU7 on a10-epoch horizon
comparison. Keep all epoch checkpoints, resume-capable last checkpoint, explicit
best-by-valid reference and train-eval/full-valid plots. A best-by-valid score is
validation-selected development evidence, not independent test performance.
Also report common completed epoch5/10/20 endpoints where available. A20-epoch
cosine schedule changes the LR trajectory from epoch1; an early checkpoint is
not equivalent to a separate5-epoch cosine run.

The overfit monitor should use full official valid WP and both target metrics,
plus explicitly labelled fixed-cohort train-eval metrics and loss. A persistent
train improvement / validation degradation is an alert, not proof of its cause.
Possible early stopping is a separate, predeclared policy: e.g. no stop before
epoch5, patience4 full-validation checks, useful-gain threshold declared before
fitting. A single noisy epoch or train/val gap must not trigger it. Scheduler
recovery can occur later. This detector/patience/best-pointer behavior is proposed,
not already implemented by the current epoch monitor. Numerical failure bounds
remain distinct from scientific stopping.

Proposed GPU7: baseline architecture with Muon on explicitly enumerated hidden
attention/FFN matrices; AdamW on all remaining parameters, including recurrent
weights, normalization, biases, convolution and readout. Use explicit scaling/LR
groups, exact parameter coverage and restart/numerical checks. This is a hybrid
optimizer experiment; no gate-only or universal Muon quality claim.

Current sequence_config.py rejects GALAM+Muon, and sequence_optimizer.py uses
Transformer-specific parameter-name groups. A config flag is insufficient.
The prospective adapter must not broaden historical Transformer group semantics.
[PyTorch Muon documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html)
describes hidden2D parameter updates and AdamW for other groups. The precise
GALAM partition above is our conservative adaptation, not the author's recipe.

Batch16/32/64/128 belong first in actual-device engineering profiles. Choose a
common measured-throughput batch with legal BatchNorm tails and memory headroom,
then freeze recipes and LR/update budgets. A larger batch changes both BatchNorm
statistics and optimization granularity; no linear speedup or quality preservation
is assumed. A separate full-batch ablation remains an alternative to the hybrid
optimizer arm if actual profiling gives a stronger quality question.

The full source footer has10607sequences of20000rows before component exclusions.
With batch16, chunk64, accumulate1, the loop plans ceil(10607/16)*ceil(20000/64)
=207519update windows/epoch, or4150380over20epochs. Batch128 plans25979/epoch;
step cost also changes. These are planned windows before actual membership and
skips, not observed successful optimizer updates.

Illustrative training-only times at batch16:10ms/window gives11.53hours for20
epochs;100ms/window gives115.29hours. Neither is measured or a predictive range.
Add full validation, fixed train-eval diagnostics, data waits and checkpoint time.
The model presently calls stream_step in a Python loop over time, so small
parameter count does not imply fast epochs. Representative warmed actual-device
profiles, plus full-validation timing and eight-reader contention, must determine
the ETA. Panel wall time is approximately the slowest lane plus shared setup,
not eight fit times summed or divided by eight again.

Eight full20-epoch caps imply160fit-epochs, versus45in the first proposal; this is
a workload count with unequal arm costs, not wall time. Early stopping may reduce
it but supplies no guaranteed schedule. No default4-hour allocation is restored.
Official chronological boundary and train-only ancestry remain mandatory.
