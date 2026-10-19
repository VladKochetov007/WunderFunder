# R6 loader compatibility V3

The first corrected-startup R6 attempt stopped in restricted checkpoint
deserialization: `weights_only=True` rejected the checkpoint's
`torch.torch_version.TorchVersion` metadata global (captured failure
`252ecb7…`). No model was constructed and no forward, backward, or update ran.
The V2 startup order had already hashed and mapped the 24 selected sample NPY
files before reaching the checkpoint loader.

V3 keeps restricted loading with `weights_only=True` and temporarily allows
only `torch.torch_version.TorchVersion` through
`torch.serialization.safe_globals`. It does not disable restricted loading,
add any other safe global, or fall back to unrestricted deserialization. The
loader remains downstream of the CLI's full checkpoint SHA-256 verification.
The bounded static pickle-opcode receipt lists the four globals used by this
checkpoint (`collections.OrderedDict`, PyTorch's tensor rebuild helper and
`FloatStorage`, and `TorchVersion`) and no `STACK_GLOBAL` opcodes. Execution
now loads the checkpoint before mapping the sample arrays, so a future
checkpoint-compatibility failure is reported before sample NPY access.

The exact checkpoint provenance-args observation records `aux_weight=0.5`
with all auxiliary objective weights disabled and `aux_next_k` empty. The
canonical training loss applies `aux_weight` only inside the auxiliary-label
loop; the fixed two-output state has no such label heads. The argument gate
therefore permits this exact default only while every auxiliary objective
remains disabled. A nondefault weight, active auxiliary objective, or
nonempty `aux_next_k` remains rejected.

The rationale was checked against canonical revision `a266b036`:
`wnn/training/losses.py` applies `aux_weight` only in the loop over auxiliary
labels (SHA-256 `94e4efaaf71ab8487879e50bd4ffca7d90fb5921aff8482429a0abfa6ce9654a`),
and `wnn/training/setup_model.py` derives added outputs from requested shifts
(SHA-256 `52f8bd9a173550d9423d0d3528525454cf6f2ff48f468bda1e03d3378b2eb9bb`).
These files were inspected as evidence; they are not additional runtime
imports. The runtime import closure and fixed two-output checkpoint checks
remain unchanged.

The R6 scientific design is unchanged: the same accepted R3 sample and
full-SHA-bound R01 F0 checkpoint, model, normalization, 8 fit components,
two 128-row gradient windows per sequence, shared pooled WMSE, parameter
blocks, CPU/resource guards, and no-update diagnostic remain fixed. The
synthetic test serializes a tensor with a `TorchVersion`, verifies that the
narrow allowlist loads it, and verifies that an unrelated `SimpleNamespace`
global is still rejected. This is a compatibility test only; no real
checkpoint, sample array, or gradient job was run by this revision.

Exact V2 preparation manifest, test receipt, loader, model-probe test and
pre-V3 execution source are preserved byte-for-byte under
[`records/loader_freeze_v2/`](records/loader_freeze_v2/).
