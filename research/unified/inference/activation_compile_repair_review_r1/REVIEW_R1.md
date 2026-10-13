# Independent activation-QAT compiler repair review R1

**Verdict: GO for a freshly admitted rerun of the same B2/T32/F112/H64/L2 synthetic compiler probe only.** This is a static code review, not an execution admission. It does not clear the actual-data B16/C1000 profile, a fit, or any quality claim; those remain under root's resource gate.

## Scope and evidence

I compared frozen backend revision `60daafba7b6bb9fc55b62ecf3f656b48eb90726e` with repair `a0e3a4f699cef424027ee9a2360bbe682e313aea`, now integrated in MAIN as `ba385b09`. The `activation_qat.py` and `train_rnn.py` blobs in the integrated commit match the repair commit exactly. The first failed synthetic attempt is recorded in [the immutable failure receipt](../../../campaign_2026-10-13/qat_renewable_r1/ACTIVATION_SYNTHETIC_BACKEND_FAILURE_R1.md): Dynamo raised `Unsupported: Attempted to wrap RNN, GRU, or LSTM` while tracing `isinstance(model.gru, nn.GRU)`, before either mode entered execution. The failed attempt is a backend integration failure, not a model-quality observation.

The review was read-only against source. I did not import Torch, execute tests, compile, or use a GPU. A bounded AST parse passed for the four changed Python files, and `git diff --check 60daaf..ba385` passed. Root separately reports eight focused CPU/native tests passing on integrated MAIN commit `ba385b09` in 7.96 seconds with `PYTHONPATH=src WNN_Q8_TEST_LIBRARY=<MAIN>/artifacts/native_qat_r1 .venv/bin/python -m pytest -q tests/test_activation_qat.py`; its hash-bound receipt is pending. That result covers the forward/backward/native fixture but does not exercise Dynamo, so it does not substitute for the proposed probe.

## Findings

The repair removes the diagnosed module inspection from the compiled forward. `ActivationQATModel.__init__` checks the GRU type/encoder and stores primitive topology values. Its `forward` passes that tuple to `activation_qat_forward`; the helper's `gru_metadata is None` branch is therefore bypassed on this wrapped path, so it does not read `model.gru` or inspect its module attributes while Dynamo traces. The scan still uses the same reset/update/new equations and FP32 carried state; the change substitutes static layer/hidden/dropout metadata and leaves recurrence semantics unchanged.

Both compiled call paths now build a complete named-parameter dictionary outside the compiled call and overwrite selected entries with `qat_weight_replacements`. This preserves the existing represented-weight forward and its autograd path: the map stores tensor references and replacement results without cloning or detaching them. `parameter_map_is_complete=True` makes the scan consume that map; the GRU and head affine terms use its entries, including previously direct head biases/final-layer parameters. QAT cache construction/invalidation and optimizer sequencing are not changed by this repair. The internal callers construct the complete map from `model.named_parameters()` before applying replacements.

The trainer remains opt-in: `--activation-backend` defaults to `none`, validates only supported FP32 dense-GRU configurations when selected, and imports `activation_qat` inside the selected path. The default fused-GRU path does not import the scan implementation. `activation_qat` depends on `training_qat`, but `training_qat`, `nn`, and package initialization do not import `activation_qat`; the inspected dependency direction has no cycle.

One call contract remains explicit: compiled internal callers must pass `validate_finite=False`, because the eager finite-parameter guard enumerates `model.parameters()`. The trainer validates model tensors during backend setup and validates each input chunk and carried state before the compiled call; both the trainer and synthetic objective pass `False`. The complete-map flag is likewise an internal trust contract; these callers construct the map from all named parameters. Neither residual blocks the exact synthetic rerun.

## Next evidence boundary

If the bounded synthetic attempt succeeds, it establishes only that these two compiled modes can complete forward and backward at the small synthetic shape. It will not establish full-shape memory/runtime, actual-batch semantics, optimizer/cache behavior across updates, or quality. Root should preserve the failure receipt and bind the retry to integrated source `ba385b09`; any actual-data profile needs a separate measured admission.
