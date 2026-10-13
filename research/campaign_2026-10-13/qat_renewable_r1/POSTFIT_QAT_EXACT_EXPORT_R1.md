# Post-fit QAT exact export receipt — 2026-10-13

The deployment preparer now carries exact codes and scales from a fitted inference checkpoint into the archive whose preset matches that checkpoint. Before this change, `load_checkpoint_member()` returned only effective reconstructed float weights, so the preparer silently quantized those values again. Re-encoding could replace the trained row scales and codes.

`checkpoint_quantization()` validates the saved QAT mode, policy version and scheme, effective-float representation, selected matrix shapes/bit widths, encoded tensor names/dtypes/shapes, and exact reconstruction of every selected matrix. A QAT checkpoint with missing or mismatched metadata is rejected. The separate no-finetune control is accepted only when its zero-update provenance, source hash, preset, and training rule agree. The manifest records the checkpoint quantization summary and exact code/scale hashes, then labels each endpoint: matching archives use exact checkpoint tensors; other quantized cases are PTQ from the effective checkpoint; the FP32 case of any quantized checkpoint is explicitly an effective-reconstructed diagnostic, not latent weights.

For a fitted `recurrent_head` checkpoint, `packed_q8` uses its exact checkpoint representation. For fitted W4 presets, the corresponding W4 archive uses exact checkpoint codes/scales, while packedQ8 remains a native-Q8 transport of the effective checkpoint. Float checkpoints retain their prior PTQ path.

## Verification

- Focused checkpoint-schema regression: **6 passed**. It covers recurrent-head Q8 plus both W4 policies, bit-exact scale/code archive roundtrip, reconstruction equality, rejection of missing or incompatible QAT metadata, unchanged float-checkpoint PTQ semantics, and no-finetune control provenance.
- Ruff passed for the evaluator and focused test files; `git diff --check` passed.
- Actual checkpoint smoke: `qat_deploy_f0_s3_10pct_uniform_w4_r1/model_ep0.pt`, checkpoint SHA-256 `64e4ce4d815d69d5a144b8e3823f34ceadfd54c568ad3f7e2172a76bdcf2f917`. The `recurrent_w4` archive restored all 12 encoded code/scale fields bit-for-bit and every effective model tensor exactly. Its temporary archives were removed after verification.
- Both commands used the project limiter with one CPU thread and a 1.5 GB memory cap. The real-checkpoint smoke exercises checkpoint loading and Python archive export only; it is not native inference, a full-fold quality result, a runtime result, or evidence of a QAT gain.

Focused command:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=180s \
scripts/limited /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
-m pytest -q tests/test_evaluate_qat_deployment.py
```

Real-checkpoint smoke used the same limiter and Python, with a short inline script that loaded the checkpoint using `load_checkpoint_member()`, called `checkpoint_quantization()` and `export_case_archives()`, then compared all encoded bytes and restored tensors with `numpy.testing.assert_array_equal`.

The patch was built from detached MAIN commit `3b72120cce5ccb864ca2104b414a6bc492014d32`. No run artifact, split, evaluator output, grader, or native library was changed. Full-fold native evaluation remains owned by the controller and is not established by this receipt.
