# Source review and receiver acceptance limits

Date: 2026-10-29. Scope: the new changes above PR46 commit
`8482de5c71ce8bdd9974780438d518ea50bfeca8`. Independent reviewers read source;
no reviewer executed tests, imported a model or used local/remote accelerators.

## Block graph

Reviewed the explicit causal position mask, inclusion of the current key,
history eviction, provided timestamp bias, multi-layer attention and recurrent
state, convolution carry including kernel1, three-branch fusion, and identical
checkpoint parameter names. Ordinal timestamps share one bias calculation across
batch rows; supplied timestamps retain individual bias calculations. This
changes floating-point reduction order and is covered by the receiver comparison.

The BatchNorm implementation retains per-time batch moments, biased forward
variance and unbiased running variance. The composed EMA/cumulative updates cover
momentum0/0.1/1/None and exact time-step counters. Eval mode with running statistics
can flatten batch/time; training never does. The original wrapper's singleton
batch rejection remains before dispatch. Legacy diagnostic observers force only
their sampled chunk through the step path. No quality recipe is silently changed.

Receiver acceptance is deliberately stricter than a forward smoke check: all
state tensors, input/parameter gradients, parameter updates, AdamW state, reload,
causal prefixes, unusual partitions/history/depth, and BN edge cases. Each backend
backpropagates before the other graph is constructed, reducing artificial memory
pressure from retaining both graphs. Positive receipt fixtures precede negative
mutation tests, so rejection tests do not start from an already-invalid receipt.

No numerical equivalence or speedup has been observed. Near-zero gradients can
amplify numerical differences in AdamW; a failed receiver check must be diagnosed,
not automatically accepted with a wider tolerance. Larger block attention
activations require an actual memory measurement. Complete epoch speed also
depends on data loading, monitoring, validation and checkpoint I/O.

## Frozen BatchNorm diagnostic

Reviewed original epoch checkpoint/index, configuration, training source-file
inventory, exact source split/membership and data binding through the canonical
loader. Calibration reads train features without labels or masks. The known BN
dependency graph is acyclic: local input BN and each global mixer's two BNs can
be calibrated in the declared layerwise order. Float64 merged moments include
between-batch/time variation; averaging within-batch variances would not suffice.

All parameters and non-BN buffers are hashed before/after. The output uses an
inference-only schema and fresh paths outside input fit/data/source trees. Both
scores use the same complete valid membership, warmup99, official mask and pooled
clip2 metric. Checkpoint and captured index bytes/admission are rechecked at the
end. The index hash argument is its canonical producer field, not the file hash.

Review fixes included rejection of untracked receiver source, exactly one
explicitly selected visible GPU, process-wide TF32 override rejection, actual
runtime flags/GPU identity, prediction-bank disk headroom, and an outer Linux
timeout in commands. These are source-reviewed safeguards, not executed gates.

Remaining scientific limits: the checkpoint and validation are development
exposed; source overlap detection is approximate; train pooled moments may not
match later valid distribution. Recalibration changes the inference hypothesis
and is not a claim of equivalent BatchNorm training or improved hidden score.

## Publication boundary

Only Ruff formatting/lint, AST parsing and size checks, the source-only training
structure checker, JSON/config inspection, shell syntax and Git whitespace were
run locally. See [VALIDATION.json](VALIDATION.json). GPU commands must execute in
a separate clean reviewed checkout. This PR does not stop, restart, signal, merge,
reconfigure, resume or replace any current experiment or checkpoint.
