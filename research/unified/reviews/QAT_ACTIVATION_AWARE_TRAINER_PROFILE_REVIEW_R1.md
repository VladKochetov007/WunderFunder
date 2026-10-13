# Activation-aware trainer hook and profile readiness review R1

**Decision:** GO for the prepared bounded, actual-shape compiled backend
engineering profile after root confirms live GPU/host-memory admission. This is
not fit clearance, a quality result, or serving/export approval. The next
scientific step still needs a separately admitted full-causal 100-update
profile before any longer activation-QAT fit.

## Findings

No unresolved blocker remains for the declared first-chunk profile in the
reviewed successor. The initial `99987feda784c1e157adaae5f6e57b4daa66996c`
revision had two concrete issues: it imported the private scan module on every
trainer startup, and it computed cached-weight identity after the optimizer had
changed parameter versions. Successor
`5f6812b098fdf4c88e6ccb496a0343e159781eef` makes the scan imports opt-in,
checks identity before the optimizer step, and requires cache invalidation,
latent-version advancement and changed represented-weight fingerprints in the
profile acceptance predicate. The wrapper now also rejects a paired run when
the two mode receipts disagree on the checkpoint, split/subset membership,
normalization or sampler RNG identity. It declares that pin-memory is removed
for batch-object reuse while preserving explicit fused-Adam, sync-free and
propensity-cache flags.

The compiled implementation matches the stated recurrent scope: PyTorch GRU
gate order `[reset, update, new]`; reset multiplies the recurrent candidate
affine term; update combines candidate and prior state; and the returned state
remains FP32. Dynamic symmetric A8 uses per-vector `maxabs / 127`, ties-to-even
rounding and zero codes/scale for all-zero vectors. It is applied to the
post-encoder input before every layer's `weight_ih` matvec and the previous
hidden vector before every `weight_hh` matvec. The raw projection/native I16
path and head activation blocks are excluded. The explicit FP32 and A8 modes
share the same scan implementation and eager weight-QAT replacement path.

`forward_chunk` computes represented QAT replacements outside the compiled
call. The accepted profile requires evidence that each first-chunk weight
cache matched the pre-update latent tensors, each optimizer update cleared the
cache and advanced latent versions, and the next forward used changed
represented weights. The two-update-per-mode repeated batch is a bounded cache
and compiler smoke; it does not exercise a nonzero carried state at a chunk
boundary, full-sequence update trajectory, resume/export consumer, or native
inference callback.

## Recipe and implementation comparison

| Dimension | Frozen profile contract and review |
|---|---|
| Parent and membership | The wrapper selects the frozen 10% uniform-W4 fit recipe, verifies the clean-T3 F0 S3 CP5 checkpoint SHA-256 `e30577ffb39d061bc40043f68a90285965bd2cf790f6ba5947ba78c60b776eda`, and keeps the original split and component subset. Both modes reload that same checkpoint into separate processes. |
| Data and order | Both modes use the production `SequenceFile`, `build_inputs`, cached targets, propensity cache and first batch. The same B16 batch is reused within each arm for one warmup and one measured update; only rows 0:1000 from sequence origin are forwarded, while the objective labels/cache rows retain their full 20,000-row context. The wrapper requires the logged initial sampler RNG state and fit identities to match across modes. |
| Objective and updates | The unchanged trainer computes frozen-recipe WMSE distillation plus `aux_struct=0.3` and `aux_next=0.3`, then performs fresh AdamW updates with the frozen LR, decay, OneCycle and clipping flags. There are two optimizer attempts per mode, four total. No evaluation or endpoint checkpoint is produced. |
| Execution deviation | To reuse the exact first host batch, the wrapper removes `--fast` and `--pin-memory`; the frozen command also supplies `--fused-adam`, `--sync-free` and `--prop-cache` explicitly, so those remain enabled. The profile therefore does not measure pinned-loader transfer throughput. Its purpose is compiled scan compatibility, update/cache behavior and resource peaks. |
| Backend treatment | Both arms use `torch.compile(fullgraph=True)` on the same higher-order scan. `scan_fp32` disables A8; `scan_a8` applies the stated dynamic A8 boundaries. QAT is fixed recurrent-W4; learned scales are rejected. The profile is not a comparison of native integer accumulation and does not test head A8 or native I16 projection. |

The project pins `torch>=2.14,<2.15`; the review did not import Torch or claim
that the private `torch._higher_order_ops.scan` and Inductor APIs work in the
installed runtime. Confirm that through the exact-shape GPU profile. The
finite-domain contract rejects nonfinite FP32 inputs and initial/carry state
before entering the compiled path, and checks finite losses/gradients in the
trainer. Native nonfinite-vector behavior is not reproduced; nonfinite values
are outside this profile's supported domain. The public helper also checks
model parameters and normalization before recurrence. Full-profile receipts
should be reviewed for finite final parameters and compiler counters in
addition to the wrapper's current acceptance checks.

The backend flag is included in `resume_contract` options, and the continuation
contract hashes the trainer and all `src/wnn/*.py` files. A same-mode resume
reconstructs the opt-in backend before restoring state; changing backend mode
or source cannot silently reuse a different continuation contract. The frozen
profile has no EMA, SWA or EMA-evaluation mode. Those secondary modules are not
validated as activation-A8 teachers here, and any later fit that enables them
must either propagate the backend deliberately or reject that combination.

## Scope and checks

Reviewed the exact source diff from parent `2a5c9c89` through the initial
revision above and the committed successor. Source hashes at successor
`5f6812b098fdf4c88e6ccb496a0343e159781eef`:

- `scripts/train_rnn.py`: `7f69848bb460914af41f5f06bf6b76cb9c84f9aeed57bfdc7abf78dbb2cd3008`
- `src/wnn/activation_qat.py`: `9ccdfb864222ef77c139cd64ca1adf420e6939dea652fd37e59a93c484a397b7`
- `scripts/profile_activation_qat.py`: `9d5680059f6d6104e0f2ca82a2ab93941cc44ad45c77b77de951c74b6944c901`
- Frozen recipe `RUN_RECIPE_R1.json`: `c08c6a7c46d1d117521153b0cd233410695cf738d7784d32d26de80f7a92e585`
- Profile plan `ACTIVATION_AWARE_ACTUAL_PROFILE_PLAN_R1.json`: `78358d3b4a31ffb0d791280fad4b6e447172ec2dfd54e75bd6d3fd7175ce13c5`

The bounded check in this review parsed the trainer, profile wrapper and A8
module with Python's standard-library AST parser under the campaign wrapper;
all three parsed successfully. No Torch test/import, compiler, GPU run, model
load, native rebuild, fit or prediction evaluation was performed. Earlier
focused CPU/native tests recorded by the actor were against the pre-wrapper
backend and are not independent evidence for this successor. The exact actual
profile recipe/command and root-owned execution boundary are in the linked
profile plan; any compiler, memory or runtime failure is backend-execution
evidence, not a negative model-quality result.
