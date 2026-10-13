# Activation-aware synthetic backend failure — 2026-10-13

**Status:** diagnosed backend/trace failure; no model-quality result. Experiment E-QAT-A8-BACKEND-20261013-R1 / H-QAT-A8-20261013. The immutable root attempt used source revision `60daafba7b6bb9fc55b62ecf3f656b48eb90726e` from `artifacts/qat_renewable_r1/activation_source_r1`; no source in that root worktree was edited.

## First admitted probe

Root admitted only the B2/T32/F112/H64/L2 synthetic scan smoke, both activation modes, with a four-decimal-GB worker reservation, one CPU and a 300-second timeout. It started at 2026-10-13 18:41:28 UTC and terminated at 18:41:33 with worker exit status 1 after 5.055 seconds. The controller recorded 810,852,352 peak worker host bytes and 8,744,087,552 peak parent bytes. The admission snapshot projected 12,471,625,728 committed bytes against the 17,000,000,000-byte target. All attempt OOM, OOM-kill and memory-max/high event deltas were zero.

The controller's VRAM sampler recorded 79,691,776 bytes at both its initial and peak samples. The failure occurred during Dynamo tracing before a compiled forward or backward ran, so that brief observation is only a startup sample and must not be interpreted as a compile or workload VRAM peak. The controller log is empty: the profile script writes its caught exception and traceback into `ACTIVATION_SYNTHETIC_GPU_R1.json`, and prints its JSON timing record only on successful completion. The durable controller attempt is `runtime/attempts/qat-a8-synthetic-compiler-r1.json`; its log is `runtime/logs/qat-a8-synthetic-compiler-r1.log`.

The first exception in the profile receipt is `torch._dynamo.exc.Unsupported`, “Attempted to wrap RNN, GRU, or LSTM.” Dynamo failed while tracing `isinstance(model.gru, nn.GRU)` at `activation_qat_forward` line 131. No mode was entered (`modes` is empty), no optimizer step occurred, and no compile timing or activation-QAT effect was measured. This is a compiler/backend integration failure, not a quality failure.

## Corrective change and bounded follow-up

Code repair commit `a0e3a4f6` moves plain-GRU topology checks and static dimensions into `ActivationQATModel.__init__`, then passes those primitive values into the compiled forward. Both trainer and synthetic-profile callers now assemble a complete parameter mapping outside the compiled body and pass that mapping as tensor inputs with an explicit completeness flag. The scan body therefore does not inspect the `nn.GRU` module or enumerate model parameters while Dynamo traces it. This also keeps represented QAT tensors in the actual differentiable parameter map. The modified backend, profile helper and test pass scoped Ruff lint; backend/test formatting passes. The focused Torch tests and compile probe were not rerun.

The exact failure receipt remains in the main project checkout and is intentionally not rewritten. The successor has only static lint/format checks at this point; no Torch test or compile retry has run. Root must review the successor, issue a fresh attempt ID and admit the repeated small two-mode synthetic smoke before considering any actual-shape profile. A backend pass on this synthetic shape still would not establish actual-data performance, sequence-length resource fit, training quality or an A8 gain.
