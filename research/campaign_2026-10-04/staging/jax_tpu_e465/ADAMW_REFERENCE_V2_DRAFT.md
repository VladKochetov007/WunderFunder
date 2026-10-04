# E465 dynamic-beta AdamW reference checks: v2 design

This is a new source-only diagnostic packet. The E465 AdamW/rnn, OneCycle, and host-contract sources remain unchanged. The v1 checker, sealed spec, failed runner record, log, and quarantine are preserved as originally run.

## Evidence to preserve

`e465-adamw-referencecheck-v1` failed after all 7,460 updates. Its host recurrence was Python binary64 while PyTorch parameters and moment tensors were FP32. The observed maxima were parameter `3.844495279015092e-6`, first moment `2.5429616745675787e-7`, and second moment `7.396730402953189e-6`; the first and second moment limits were `2e-6`. This is a real cross-precision failure. Do not relabel it as a pass or replace its attempt/log/quarantine.

## v2a: formula-only binary64 discriminator

`check_e465_adamw_reference_v2.py` compares the existing pure-host recurrence against `torch.optim.AdamW` with an FP64 scalar parameter and FP64 state, over all used OneCycle indices 0–7459. It uses a fixed `1e-10` absolute limit for the parameter and each moment. Before importing Torch it reads and hashes the immutable staged JAX `rnn.py` and host helper, verifies the source expressions for current-beta bias correction, moments, normalization, and decoupled decay, and records function-text/AST hashes. It imports no JAX and loads no model/data. The result is strictly formula-level evidence: it neither erases nor substitutes for the v1 FP32 rounding failure, and it is not evidence of JAX execution parity.

## v2b: actual FP32 JAX/Torch discriminator

`check_e465_adamw_fp32_parity.py` uses the actual staged `wnn_jax.rnn.adamw_update` in a compiled `jax.lax.scan` and compares every one of 7,460 updates against Torch FP32 AdamW. Both receive identical precomputed FP32 gradient values and the same OneCycle LR/beta arrays. Two trainable scalar parameters exercise per-step parameter and moment values; frozen `mean` and `std` buffers verify buffer preservation and exclusion from optimizer state. It checks the JAX update counter and raw maximum absolute errors by parameters, first moments, and second moments.

The FP32 acceptance limits are predeclared separately from formula-v2: parameter `5e-6`, first moment `1e-6`, second moment `1e-5`. They reflect an FP32-vs-FP32 comparison over a long recurrence and are fixed before execution. They are intentionally not the stricter binary64 limit. If the FP32 fixture fails, preserve the failure and investigate the JAX/Torch operation-order difference; do not raise the limits after seeing results. Passing this scalar fixture is still not full model/training parity.

## Execution boundary

Both specs are CPU-only drafts at 1 CPU and 2 GiB, no GPU allocation, and 180 seconds. v2a needs only the existing Torch CPU environment; v2b needs the private CPU JAX stack plus Torch. Do not run locally or upload before root reviews and seals source closure. These checks are not full E465 fit admission: the training worker still needs the separate E463 TPU real-data prerequisite, its own all-eight-device full-shape profile, and root admission.

## Failure artifacts

Both numerical checkers write a JSON report with `status: "failed"` and the measured maximum errors before raising an assertion when a numeric limit fails. This preserves a machine-readable counterexample alongside the nonzero runner result. The actual FP32 checker sets `JAX_PLATFORMS=cpu` before importing the staged `wnn_jax.rnn` module. Its immediately prior pre-review draft is preserved at `pre_root_review_fp32_import_v1/`.
