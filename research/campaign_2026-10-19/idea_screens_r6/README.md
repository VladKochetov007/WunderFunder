# R6 frozen-GRU gradient diagnostic

R6 checks idea 20 with a bounded, no-update CPU diagnostic of per-target
gradients at the fixed R01 wide-F0 FP32 endpoint. It uses the first eight
components in the accepted R3 fit order, each with two causal 128-row windows
after replaying the complete prefix. The 211-column inputs retain canonical
event features and the checkpoint's saved normalization. There is no optimizer,
quality score, fitted variant, or new GPU job. The [reviewed design](DESIGN.md)
and [design review](DESIGN_REVIEW.json) fix the scientific scope.

**V3 completed and was independently audited.** [Results](RESULTS.md):
shared input/GRU pooled cosine+0.333540 with4negative/16windows; shared hidden
readout+0.026854 with3negative/16. The [scalar audit](RESULT_AUDIT.json)
reconstructs pooled norms/dot/cos to2.22e-16, without another backward pass.
This is local heterogeneous gradient geometry, not training-wide harm,
PCGrad benefit or a quality/public comparison. Weights stayed unchanged.

The successful unit `computer-vision-homework-eda19-r6-v3.service`, invocation
`b4c2ac3dc5454fbb8c72180cd2b31fe3`, ran01:10:54–01:11:35UTC and exited0.
[Raw receipts](records/probe_v3/terminal_capture/README.md) were saved and
verified before exact-unit stop. [Resource reconciliation](RESOURCE_RECONCILIATION.json)
records0new parent/F3child events and separates the memory observations.
No arrays/model/gradient payloads moved to the Mac. The staged prelaunch
README hash remains in ROOT_ADMISSION; this post-completion navigation update
does not modify the45 manifest-pinned scientific files.

The exact V3 preparation manifest is [PREPARATION_MANIFEST.json](PREPARATION_MANIFEST.json),
SHA-256 `c291687c93689463133f5e4109e7adef116dc79bca8aec08b7ec30d29a28505d`.
It binds 45 files, including the 25-module canonical WNN import closure, the
accepted R3 sample and membership identities, and the full-SHA-bound R01
checkpoint. The [V3 implementation review](IMPLEMENTATION_REVIEW_V3.json)
and [V3 root-startup review](ROOT_EXEC_STARTUP_V3_REVIEW.json) pass for the
frozen code and admission path. The [focused test receipt](TEST_RECEIPT.json)
records five passing synthetic tests and Ruff checks; the [loader addendum](LOADER_COMPATIBILITY_V3.md)
documents the narrow compatibility correction.

Three distinct launch attempts are preserved and must not be confused with
scientific outcomes:

- **V1** failed before sample-array access or model work because root's import
  smoke loaded `wnn` and then invoked the CLI in that same interpreter; the
  CLI's source guard correctly rejected the preloaded package. The separate
  import smoke passed with readers denied, no data reads, and no CUDA
  initialization.
- **V2** reached restricted checkpoint loading, where `weights_only=True`
  rejected the checkpoint's `TorchVersion` metadata before model
  construction. Before that failure, V2 hashed and mapped the 24 selected
  sample NPYs; the operator did not transfer their array payloads. It ran no
  forward, backward, or gradient window.
- **V3** keeps CPU `weights_only=True` and allowlists only
  `torch.torch_version.TorchVersion`. It loads the full-SHA-verified
  checkpoint before mapping sample arrays. The observed `aux_weight=0.5`
  default is accepted only when all auxiliary objectives are disabled and
  `aux_next_k` is empty. These are loader/startup fixes; the model, inputs,
  normalization, windows, objective, and no-update design remain fixed.

The [execution-record index](records/README.md) links the V1/V2 failure
captures, byte-preserved V1/V2 sources, V3 startup receipts, metadata-only
checkpoint observations, and F3 maintenance snapshot. The [initial design
note](DESIGN_V1.md) records the earlier 112-column assumption; capability
inspection corrected it to the fixed 211-column model before implementation.
The [continuation](CONTINUATION.md) keeps remaining untested mechanisms and
existing F3 maintenance separate from this completed diagnostic.

[F3 maintenance at01:18:37UTC](records/F3_HEALTH_20261019T0115Z/README.md) preserves the original healthy deep worker and recordedep4batch1400/1488, with0newmemoryevents. It does not alter this completed R6 diagnostic or establish F3terminal/quality.
