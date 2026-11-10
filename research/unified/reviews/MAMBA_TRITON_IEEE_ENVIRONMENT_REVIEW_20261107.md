# Independent review of the installed Triton IEEE environment correction

Reviewer: PR67 receiving execution agent, 7 November 2026. This review reads
ROOT's actual installed-source captures, not an independently downloaded version.
No source, environment, cache, GPU or model was changed by this reviewer.

Disposition: setting `TRITON_F32_DEFAULT=ieee` **before fresh child Python/Triton
imports and JIT** is source-supported for restoring the declared strict-FP32
Mamba cohort under the observed installed Triton version. This is an explicit
receiving environment correction. It changes neither pinned Mamba source nor
declared tensor dtype; prior TF32-default oracle evidence remains historical.

The captured genuine SSD kernels call `tl.dot` without `allow_tf32` or explicit
`input_precision`. Installed Nvidia compiler options default that precision to
`tf32`. ROOT's installed `language/core.py:2038` shows the public dot frontend
selecting `knobs.language.fp32_default` first when `input_precision` is omitted,
before its TF32 fallback. Installed `knobs.py:419` binds that knob to
`TRITON_F32_DEFAULT`. These primary installed-source facts support the proposed
override for the inspected Triton 3.4 environment. Torch TF32 switches alone did
not establish this custom kernel precision.

ROOT's concrete correction is acceptable with its declared receiving gates:

- Set the variable in every relevant child environment before any Triton import
  or compilation, and use a fresh physical-disk Triton cache.
- Rerun the genuine upstream/full-chunk/reset/causal/FP32-state/backward oracle
  on both actual Mamba GPUs 4 and 5. Do not relabel the older default-TF32 run.
- Retain executed-kernel PTX with positive FP32 arithmetic evidence and no
  TF32 dot operands, plus source/compiler/backend/package/environment and cache
  identities. Bind those receipts before Mamba profiles or fitting.
- Preserve the same environment for profiling and fitting. The current L4
  worker environment copies outer environment variables, so this override is
  inherited; the operator must record the actual child setting explicitly.

The initial GRU profiles use no custom Triton path and therefore do not require
Mamba's new native oracle as a dependency. Their source/config/runtime approval
checks remain separate. No IEEE bitwise CPU/GPU equivalence, native oracle pass,
profile result or scientific score is established by this static review.

Evidence under
`C:/Users/admin/Documents/Codex/artifacts/architecture-execution-20261107/l4-rpc/results`:

| Capture | SHA256 |
|---|---|
| `020-precision-dependencies.json` | `241f96963a82ecf70aa36eab4ce0c603e81a4ca29e5bd903ecc137f4de654397` |
| `022-triton-default.json` | `6353b3f6d9d24b3baa4e854606ff0a562c1fb391057c53b46dae596d123177f3` |
