# Source and novelty audit

Canonical data/model/training source is pinned to PR75
`8626000a4b965763a70011a0291a891aa6535d7e`. The copied manifest binds every
canonical Python file; receiver admission rejects changed or untracked runtime
Python. The native control recipe is retained as bytes and differs in each cell
only by its declared fresh seed plus an experiment-specific adapter descriptor.

Canonical `src/wnn/data.py` defines i0 as52 columns (22p,22v,4dp,4dv), then i1
with the same layout, then8 a-columns. Feature indices are identifiers, not depth
ranks. The bilinear branch works on normalized current-row i0/i1; the full native
112-dimensional projection remains present. Extra input matrices use the pinned
optimizer's native input-module ownership, not a new Muon grouping rule.

The two attention arms retain QKV/RoPE/cache/local positions. They use the same
explicit QK and probability-times-V path; only softmax versus alpha1.5 entmax
changes. Native efficient-softmax parity is a receiver engineering check, not an
extra scientific fit. Sparse probability support does not imply sparse execution.

Latest team PR heads were read through GitHub CLI; the snapshot is
`pr-audit-20261112.json`. PR81 covers ten prior mechanisms; PR88 records partial
execution of that packet, not new baselines for this one. PR90 covers signed/additive
attention and conditional/static FFN routing. These are distinct from normalized
entmax probability selection and the i0×i1 input interaction here. No new exact
experiment duplicate was identified in that audit; historical results are not
relabelled as fresh controls. Research-team and main are source baselines, not
proof that all open PR mechanisms have been merged.

Primary normalizer references:
- https://aclanthology.org/P19-1146/ — sparse alpha-entmax and its gradient.
- https://github.com/deep-spin/entmax/blob/master/entmax/activations.py — reference
  implementation for algorithm review; any adapted code must retain its license.

The explicit QK/normalizer section disables outer autocast in both arms to
retain FP32 score arithmetic. An input `.float()` alone would not prevent an
autocast-eligible matmul from changing dtype; see the [PyTorch2.14 AMP
reference](https://docs.pytorch.org/docs/2.14/amp.html). BF16/BF16x9 remains
the model recipe; this paired attention-normalizer island is not a global
precision amendment. Numerical parity is still a receiver gate.

No local model computation was performed for the audit. No claimed speedup,
optimal hyperparameters, new quality score or B300 memory measurement follows.
