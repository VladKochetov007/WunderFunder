# Learned row-scale QAT prototype receipt — 2026-10-13

This code-only prototype adds opt-in `--qat-learn-scales` for the two W4 policies. It registers per-block, per-row FP32 scale parameters initialized to the existing fixed absmax values, uses the LSQ-inspired normalized step derivative, retains identity STE for latent weights, persists scale parameters and optimizer slots for continuation, and exports exact effective weights with original code/scale tensors. The deployment checkpoint validator now requires the learned-scale flag, training rule, and versioned semantics to agree.

The implementation source is [the learned row-scale specification](LSQ_ROW_SCALE_SPEC_R1.md). The matched-arm invariant is exact initial represented weights and unchanged weight STE; the method is explicitly not a full LSQ reproduction. Invalid nonfinite scales fail. Zero rows stay zero-step until they become live, when absmax initialization is applied. Negative-step repair uses the same min-subnormal floor as the fixed quantizer; no vector/no-decay exception exists, and the actual optimizer group/`--wd` semantics are recorded and checked against inference-checkpoint arguments.

No fit was run. The bounded CPU command

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=180s PYTHONPATH=/home/vlad/development/wunder/wnn_qat_lsq_proto_r1/src /home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python -m pytest -q tests/test_training_qat.py tests/test_evaluate_qat_deployment.py tests/test_trainer_resume_integration.py::test_main_learned_scale_qat_resume_and_ema_export
```

passed all 23 tests in 7.62 seconds. Coverage includes fixed-versus-learned initialization identity, scale-gradient normalization, min-subnormal negative-step repair, zero/nonfinite handling, optimizer checkpoint continuation, inference parameter filtering, EMA/SWA cloned parameters, exact deployment archive roundtrip and actual trainer continuation/export through FastEMA. No quality, scale-learning benefit, or runtime claim is made. The independent reviewer should inspect resume contract coverage, optimizer-switch/freeze behavior, and the actual SWA continuation/export path before considering a matched fit.
