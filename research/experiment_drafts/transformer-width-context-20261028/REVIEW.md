# Independent source and design review — 28 October 2026

The next comparison prioritizes standalone Transformer quality. GRU distillation
in PR47 remains a separate future serving route; no student or KD fit is part of
this panel. Review was by a separate role using source inspection only. No model
imports, tests, training, inference or GPU profiles were executed by the reviewer.

## Frozen comparison and overlap

Six fresh five-epoch fits: baseline768×16/window1024, wide1024×16/window1024,
and baseline768×16/window4096, each at seeds1001/1002. Heads12/16 preserve
head_dim64. Data membership, row budget, batch, update count, LR schedule,
precision and fixed FINAL5 endpoint are shared. Width and context are changed
separately. This is data/update-budget matching, not equal-compute comparison.

PR44 already covers dropout/weight-decay/LR on the older development split;
this packet does not restart or duplicate those configurations. R4 StageD already
proposes context1024/2048/4096 for the1.54B model. The contribution here is a
bounded executable compact-model screen on the full source-separated protocol,
not a claim that longer context is a new hypothesis.

All six fits are assigned to this panel. The two baseline cells in PR47 have not
been run and cannot be counted as available measured controls. Importing a future
baseline into PR47 would require an explicit provenance/adoption adapter; this
packet does not implement or promise that reuse. Avoid scheduling both sets of
baseline jobs independently without resolving ownership first.

## Model semantics that constrain interpretation

`context_rows` is the causal attention window **per layer**. At16layers, the
theoretical stacked receptive field is `1 + 16*(window - 1)`:16,369 rows for
window1024 and65,521 for window4096, bounded in practice by sequence history.
It is incorrect to describe either model's total raw-input history as exactly
1024 or4096 rows.

The canonical loop detaches carried key/value tensors after every512-row chunk.
Increasing the window exposes older forward-memory states; it does not create
4096-step gradients or full-sequence backpropagation. Memory also crosses
optimizer updates, so cached representations can come from earlier parameter
versions. These streaming semantics are shared, but the amount of retained stale
memory grows with the window. A positive context result supports this particular
memory recipe, not a pure long-gradient claim.

MuonLR1e-4 is held fixed, not claimed optimal for both widths. Five epochs is a
bounded allocation, not proof of adequate optimization of the wider model.
Earlier compact development curves peaking before FINAL5 make it especially
important not to interpret a negative fixed-final result as disproving the
architecture or its longer-training descendants.

## Analytical inventory, not measured hardware readiness

Counts follow the committed linear/GELU Transformer source, with no hint head.
The parameter expression is `layers*(12*hidden^2 + 13*hidden) + 117*hidden + 2`.
Steady persistent FP32 KV storage per rank is
`layers*2*batch*hidden*(window-1)*4` bytes.

| Variant | Analytical parameters | Persistent KV at batch128/rank |
|---|---:|---:|
| Baseline | 113,495,810 | 11.99GiB |
| Wide | 201,659,394 | 15.98GiB |
| Longer window | 113,495,810 | 47.99GiB |

These calculations exclude activation graphs, transient concatenations/clones,
dtype conversions, optimizer/master weights, compilation workspaces and NCCL.
They are not peak VRAM measurements. Every shape needs a fresh receiver profile;
PR43 profiles cannot admit these runs. A memory failure requires a declared
recipe revision and new evidence, not a silent batch/precision change.

## Evaluation and review boundaries

The shared split excludes every train component linked to any valid component
under its feature-only graph. All official valid groups are scored with the
canonical pooled metric. Approximate linkage does not prove absence of every
unknown dependency, and source order is not verified calendar order. Full valid
has already been exposed to team research.

Each width/context arm must be compared with the same-seed baseline; keep all
six endpoints, both targets, both paired deltas and actual runtime/memory costs.
Two seeds on one exposed split do not establish significance. Component-bootstrap
intervals condition on these fitted predictions and do not cover training
variance. Two arm comparisons are exploratory; a favorable interval alone does
not authorize a significance claim, automatic follow-up or submission.

The study registry uses an explicit allowlist and binds the prepared root to its
study, exact cells, config, source and packet files. Wrapper scripts select the
study before imports; the Transformer panel rejects the KD cache action. Shared
helper changes create a new packet version, so earlier prepared roots require
their original checkout rather than resuming under the updated packet.

Receiver correctness, real-data coverage, six profiles and all scientific quality
results remain pending. Source review alone cannot certify runtime execution.
