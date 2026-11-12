# Causal routing experiments for B300

Two Transformer mechanisms are implemented as packet-local model adapters. The
canonical PR75 dataset, training loop, WMSE objective, optimizer, schedule,
checkpointing, monitoring and scorer remain unchanged. Ten fresh fits are prepared;
none has run. Local tests, model imports, training, inference and profiling are
prohibited for this preparation.

| Study | Question | Arms, each at two seeds |
|---|---|---|
| SR-A | Does signed interaction between two attention maps improve the teacher? | Plain Transformer; additive two-map control; differential two-map treatment |
| SR-F | Does row-dependent routing between FFN partitions improve the teacher? | The same plain anchors; static gate control; conditional gate treatment |

Both studies use the **same two newly fitted plain anchors**. Thus there are ten
unique fits, not twelve; no previous result is substituted for a fresh control.
Each study has six endpoints. Seeds are 205201 and 205202 on the pinned official
split. This is a development comparison, not two independent folds.

## Why these mechanisms

The [team PR audit](sources/PR_AUDIT.md) records current heads, comments and evidence
limits. GRU K16 already has a consistent reported gain in #79; K64, width and
dropout are already covered by #80. Transformer stem results in #74 are mixed.
QK normalization in #75 has no verified final result at its latest reported
deadline. PR81 already covers ten other mechanisms. Those studies are not rerun.

The latest review also covers #82–86: ten new GRU screens, reported Muon gains,
negative TabM/NorMuon/view-mask screens, the train-only distillation track, and
leave-regime-out validation. SR-F differs from #82's class-conditioned output
head: its gates operate inside the Transformer FFNs. None of these reported
results supplies a fresh held-out score for this packet.

SR-A changes the interaction between attention maps, rather than their recency
bias, QK normalization or output gate. SR-F is a continuation of conditional
computation ideas: earlier prefix/sequence-specialist mixtures did not establish
a gain. Here the intervention is inside every FFN, with one shared trunk and no
sequence or target specialists. Novelty is scoped to the reviewed repository
inventory, not a claim of research novelty.

## Files and source contract

- [Mechanisms and falsifiers](MECHANISMS.md): exact operators, controls, limitations
  and primary literature.
- [Recipe](recipe.json) and `configs/`: the ten-cell inventory and immutable native
  recipes. Only the seed differs from `sources/native_control.json`.
- [B300 commands](B300_COMMANDS.md): prepare, receiver checks, profile, review,
  fit, strict endpoint export, full validation and comparison.
- `runtime/signal_routes/`: model adapters and thin composition of canonical
  training components. There is no copied training loop or global monkey-patch.
- [MIDAS handoff](midas/README.md): actual metadata-only route preview, bounded
  evidence exporter and future import commands. Preview is not job execution.
- `sources/runtime_source_files.json`: byte-preserved PR81 manifest of 206
  canonical Python files at the PR75 pin; its historical schema name is retained.
- [Static review](STATIC_REVIEW.json): source/config/structure checks, separated
  from pending numerical checks on B300.
- [PR description](PR.md): research contract and decision rules.

The runtime is pinned to PR75 commit
`8626000a4b965763a70011a0291a891aa6535d7e`. The new branch starts from
`research-team` at `7e4074314a62d0907c9608f9220cf72958708a2d`; it does not merge or
change another open experiment PR. Every new fit binds both the canonical source
map and all current packet runtime files. Keep existing jobs on their original
checkouts.

## Measurement boundary

Static checks can establish syntax and metadata consistency, not numerical parity,
GPU fit, speed or quality. Actual receiver probes, per-cell profiles, fits and full
validation remain pending. The official validation set has been repeatedly exposed;
two paired seeds do not establish significance or leaderboard transfer. A useful
teacher result still needs a separately measured serving or distillation path.
