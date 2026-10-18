# Tiedlead capability draft

Status: bounded engineering capability, unadmitted and unfit. This isolated candidate adds an explicit `--arch tiedlead` path to the canonical trainer. It does not change the default `gru` path. The behavior is based on a reviewed static source audit; it does not establish how a historical run actually executed.

The opt-in profile assembles 263 features in a fixed order: the canonical 112 raw values, the canonical 99 event features, then 52 i1 values gathered from the most recent `ev_i0px` event. Row zero is treated as an initial event. The gather is causal and is built over the complete sequence before chunked model calls.

The model keeps the ordinary normalized input projection and adds two zero-initialized residual branches before GELU. A single shared module `Linear(104,128) → GELU → Linear(128,64) → GELU` processes the i0, i1, and lead branches. Those branches reconstruct raw scale as `xn * std + mean`; the ordinary projection consumes normalized `xn`. The i0 and i1 branches concatenate each 52-wide raw block with its within-call difference. The lead branch concatenates current i1 with its difference from the gathered lead. The first row of every `encode()` call has zero delta. Chunked calls carry the GRU state while resetting these deltas at each chunk boundary.

The profile binds 263 inputs, a 256-wide two-layer GRU, zero dropout, and FP32. It rejects input transforms or model modifiers that would change this fixed feature/model path. The fitted mean and standard deviation use the ordinary canonical normalization routine over the assembled features. Training provenance records the profile and hashes for the training and feature-only prediction/evaluation implementation.

The focused tests cover event gathering and future independence, raw-scale versus normalized branches, shared encoder calls and zero residuals, delta resets with carried GRU state, normalization, checkpoint save/load, canonical prediction export, provenance bindings, and unchanged legacy defaults. The test fixture uses tiny synthetic tensors on CPU only.

Remaining work is separate from this capability. The R2 same-topology control still needs the family head present at family weight zero and a true label-only objective branch at distillation alpha zero. This change does not implement those objective/topology controls, does not implement the native competition ZIP/ONNX path, and does not prove historical runtime identity. No real dataset, training run, GPU, remote machine, or fit result is included here.

The byte-bound source inventory and local validation receipt are in `SOURCE_MANIFEST_V1.json` and `VALIDATION_V1.json`. The isolated worktree is `/Users/nb/Documents/wunder/WunderFunder-tl3-tiedlead-capability-20261018` on branch `experiment/tl3-tiedlead-capability-20261018`, based at `2875425f220fa6b56c8f47d0ec157c1d3450bc99`.
