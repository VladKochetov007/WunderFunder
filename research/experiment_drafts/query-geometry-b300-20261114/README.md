# Query geometry perturbations for B300 Transformers

**Implementation and MIDAS preparation only. Zero numerical probes, profiles,
fits or inference runs have been executed by this task.** Publication evidence
will be recorded in `publication.json`. H/E-B300-QUERY-GEOMETRY-20261114.
Preparation owner: Ivanhacker13/Codex; receiver ownership and allocation are pending.

This screen tests two training regularizers at the **post-RoPE query**. Unlike
PR99, neither deletes observations nor hides attention keys. Original inputs,
causal masks, KV reads/writes, loss, optimizer and evaluation remain native.
Only a receiver's isolated copy of PR75 receives the model hook; no shared
`src/`, `scripts/` or existing training jobs are changed.

## Hypotheses and fixed comparisons

1. **Pairwise rotation:** brittle content/position alignment in attention may
   improve when training queries are rotated around their native directions.
   Each rotary pair is rotated by **+0.35 or −0.35 radians** (about ±20°),
   preserving its norm in real arithmetic.
2. **Pairwise gain:** brittle reliance on coordinate amplitudes may improve when
   each pair is scaled by **0.596475 or 1.282271**. This retains the direction of
   each pair and keeps its sign. It can change the direction of the full head.

The two treatments match conditional mean, expected squared pair norm and
expected squared displacement. The pairwise-rotation versus pairwise-gain
comparison distinguishes these geometries without simply giving one more
perturbation energy. Neither benefit is assumed; destroying useful alignment
or confidence is the primary failure mode.

| Arm / filename prefix | Model option | New fits |
| --- | --- | ---: |
| `plain` | `query_perturbation=none` | 2 |
| `angular` | `query_perturbation=angular` | 2 |
| `radial` | `query_perturbation=radial` | 2 |

Seeds are **209401/209402**, six fresh Transformer fits total. No old controls,
PR99 fits, pretrained weights, teacher-target cache or student fits are reused.
The initial inventory does not authorize GPU spending or an automatic next wave.

For each two-coordinate pair q=(a,b), define Jq=(−b,a), c=cos(0.35), u=sin(0.35),
and a uniform random sign s. The exact transformations are:

```text
rotation: q' = c q + s u Jq
gain:     q' = (c + s u) q

E[q' | q]              = c q
E[||q'||² | q]         = ||q||²
E[||q' - q||² | q]     = 2(1-c)||q||²
```

Each layer/call samples independent signs for batch item, head and rotary pair;
one sign is shared across query rows within that call. Training chunks have
fixed256 rows, so chunk size is part of the treatment. Different chunk sizes
are not expected to reproduce the same training noise. Randomness uses the
native device RNG captured by checkpoint serialization. Math is FP32, followed
by conversion to the incoming query dtype. BF16 roundoff is checked separately.

The two treatment means shrink Q/logits by c≈0.939373. Gains relative to plain
can therefore involve this shared contraction, not only noise. No fourth
deterministic-contraction control is silently added; that is a possible separately
allocated follow-up. Rotation−gain controls the shared mean but **does not match
logit variance, covariance or softmax entropy**. This is not RoPE time dilation,
not a label symmetry and not a claimed reproduction of a paper's regularizer.

The helper never mutates K/V. Deeper layers and later caches can still change
through altered lower-layer representations. At evaluation the hook is bypassed
and consumes no perturbation RNG. Parameter count and checkpoint tensor shapes
are unchanged. A large Transformer remains a teacher candidate; this packet is
not a CPU ZIP or evidence of distillation benefit.

## Frozen native recipe

Base: PR75 `8626000a4b965763a70011a0291a891aa6535d7e`, original plain recipe.
24 layers, width768, heads12, context1024; chunk/gradient horizon256;
batch128/rank ×2B300, four accumulated chunks; BF16/BF16x9 with FP32 master
parameters; Muon+AdamW; LR0.0003, decay0.01, warmup5%, cosine floor0.1, clip1.
Four full-data epochs / **fixed E4,3280 updates**; retain all epoch snapshots.
Clean label WMSE, original fitting-only normalization, no KD, no quantization.
All options remain explicit in [configs](configs/); no performance tuning is added.

Fit10325 train-source sequences, excluding282 train sequences linked to evaluation
components. Zero valid-source fitting. Evaluate all1873 official-valid sequences,
3528273 scored rows, `global_wp_clip2_v1` via independent export and scorer.
SplitSHA256 `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`.
The official valid is repeatedly exposed development data. There is no fresh
holdout or demonstrated future-regime representativeness in this screen.

## Decisions and uncertainty

Primary quality contrasts: angular−plain and radial−plain on **uncorrupted**
evaluation. The third contrast, angular−radial, tests the geometry explanation.
Report both seed deltas and their equal-weight mean. A candidate earns a
separate confirmation allocation at meanWP≥+0.001 and both seed deltas positive.
Winning against another perturbation while losing against plain is insufficient.
Rotation-specific evidence additionally needs positive angular−radial mean.

The native component bootstrap uses common draws across contrasts/seeds and
reports per-seed intervals; the two-seed mean is descriptive with no interval.
Native `p_le_0` is a sign fraction, not a calibrated p-value. No formal
significance or multiplicity-adjusted discovery is claimed. Intervals condition
on fitted predictors and omit seed-population and distribution-shift uncertainty.
Mixed seeds/small gains are inconclusive, not automatic grounds to scale.
Preserve negative endpoints; a zero-crossing interval alone does not close an idea.
Follow-up requires its own budget, source-qualified folds and later student/CPU
evidence. Local deltas must never be added to a historical public score.

## Implementation and handoff index

- [PR.md](PR.md), [recipe.json](recipe.json), [configs](configs/): review contract and six frozen cells.
- [GPU_COMMANDS.md](GPU_COMMANDS.md): explicit receiver steps and resource accounting.
- [RUNTIME_DESIGN.md](RUNTIME_DESIGN.md), `runtime.patch`, `payload/`: focused
  component and config/model/attention hooks in an isolated PR75 source tree.
- `prepare_runtime.py`, `runtime_manifest.json`: extraction and exact source inventory;
  `receiver.py`, `run.sh`: source/data/probe/profile/approval/fit/export/score gates;
  `readout.py`: complete fixed-E4 provenance and all three paired contrasts.
- `query_invariants.py`, `query_probe.py`: **unexecuted B300 checks** using explicit
  signs, moment/determinant identities, BF16 conversion, gradients, active treatment,
  unchanged one-layer K/V, causal streaming eval, same initialization and RNG replay.
- [sources/PR_AUDIT.md](sources/PR_AUDIT.md): pinned duplicate search and hypothesis criticism.
- [sources/LITERATURE.md](sources/LITERATURE.md): RoPE source and limits of the analogy.
- [sources/INDEPENDENT_REVIEW.md](sources/INDEPENDENT_REVIEW.md): separate source and
  producer/consumer review; code review is not numerical acceptance.
- [midas/README.md](midas/README.md): actual metadata-only preview, invocation hashes
  and prepared endpoint exporter/importer. Preview returned `READY_LOCAL_HANDOFF`
  with `launchable_remotely=false`; no model agent or GPU launch was performed.

GPU kernel compatibility, measured resource use, six endpoints, public score and
student transfer are unknown until the corresponding receiver evidence exists.
