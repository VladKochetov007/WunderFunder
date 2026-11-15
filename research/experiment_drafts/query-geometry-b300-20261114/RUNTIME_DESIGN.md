# Isolated query-geometry runtime

Prepared from native PR75 source
`8626000a4b965763a70011a0291a891aa6535d7e`. No local model imports, tests,
inference, profiling or training were performed. Numerical acceptance on B300
remains mandatory. This component is not the input/history-dropout runtime of PR99.

## Intervention and algebra

For each post-RoPE query pair q=(a,b), define Jq=(-b,a), c=cos(0.35),
u=sin(0.35), and a uniform Rademacher sign s. The angle is fixed in source,
not selected from validation scores.

| Mode | Query pair supplied to native attention |
| --- | --- |
| `none` | q |
| `angular` | cq + suJq |
| `radial` | (c + su)q |

Since q is orthogonal to Jq and their lengths agree, angular perturbation
preserves the pair norm in real arithmetic. Radial multipliers are positive,
approximately0.5965 and1.2823. Both arms have:

- E[q' | q] = cq;
- E[||q'||² | q] = ||q||²;
- E[||q'−q||² | q] = 2(1−c)||q||².

Angular noise covariance follows Jq; radial noise covariance follows q. This is
the intended contrast. Matching these moments does not match every logit moment,
softmax behavior or expected loss. In particular c is about0.9394: both treatments
shrink the expected query relative to the plain control. Their gains over plain
cannot separately establish a benefit of stochastic regularization rather than
this mean change. Angular−radial controls that shared mean shift but still compares
different noise covariance. A deterministic cq arm would be a separate follow-up.

The operation changes content-and-position query directions. It is neither a
new timestamp coordinate system nor a proved positional or label invariance.
Labels, fit membership, masks and the official metric are unchanged.

## Sampling, precision and state

`perturb_query` accepts query shape `[B,H,T,D]` with even D. It samples signs
with shape `[B,H,1,D/2]`, independently per batch item, head and rotary pair.
One sign is shared across the current call's query rows. Signs are resampled
for every layer invocation and later chunk; no sequence-level sign cache exists.
The default draw uses native device RNG. PR75's checkpoint source already
serializes per-rank CUDA RNG state; the receiver must still check warmed compiled
replay. Native RNG serialization alone does not prove a cold-process stochastic
resume. The frozen panel has activation checkpointing disabled and native block
compilation enabled; no new recomputation/resume guarantee is made.

The helper computes rotations and scale multiplication in FP32, then casts back
to the incoming FP32 or BF16 query dtype. Exact moment and norm equations refer
to real arithmetic. BF16 rounding/fusion differences require numerical tolerances
in the server probe; parameter, optimizer and checkpoint precision follow the
native BF16/BF16x9 recipe. No quantization or kernel optimization is introduced.

An optional explicit `signs` argument exists for an independent receiver oracle:
it must contain only±1 with the declared broadcast shape on the query device.
Native training never supplies it. `none` returns the original tensor without
sampling. Evaluation bypasses the helper entirely.

The helper reads only Q and signs. It cannot mutate K/V or the stored cache.
The block's key rotation, native rolling causal mask, V values, cache retention
and absolute RoPE positions remain unchanged. Later layers can receive changed
hidden representations, and their subsequent K/V can therefore differ indirectly.
No input, key or value is deliberately deleted; the original context is retained.

## Exact source hooks

| File | Isolated adaptation | Preserved behavior |
| --- | --- | --- |
| `sequence_config.py` | Whitelist/validate `model.query_perturbation` | All other configuration fields, precision injection and objective checks |
| `sequence_models.py` | Pass registered mode through factory/Transformer adapter | Parameter/buffer names, initializer order, GRU path, native compilation |
| `sequence_attention.py` | After query RoPE, call helper only while training and active | Key rotation, native SDPA/backend, FFN, residuals, causal mask and KV writes |
| Added `sequence_query_perturbation.py` | Focused formulas, sign sampling and boundary checks | No learned parameter, buffer, optimizer state or training orchestrator |

The native training loop, DDP scaling, normalizer, input assembly, target/loss
denominators, optimizer, scheduler, monitoring, fixed-epoch selection, exports
and official scorer are not modified. No PR99 input/history perturbation mode
is copied. No monkeypatch or global runtime mutation is needed.

`runtime.patch` applies only to the isolated receiver extraction of the source
pin. Payload files are added under its `src/wnn/training/`; the native source hash
includes them. `runtime_manifest.json` is produced by the receiver preparer and
binds every effective source file. Source identity does not prove numerical
equivalence, gain, throughput, admissible memory or successful execution.

## Static preparation and remaining evidence

The added module and patched native source were parsed as Python text. Ruff
format/lint checks cover the added module; no imports or model execution are
part of those checks. The parent packet supplies the receiver probe, profiles,
fit/export/scoring commands and independent review. Required server evidence
includes formula/sign tests, initialization and evaluation neutrality,
training-path sensitivity, causal/cache behavior, finite gradients, compiled
RNG replay and actual full-shape memory/time before a scientific fit.
