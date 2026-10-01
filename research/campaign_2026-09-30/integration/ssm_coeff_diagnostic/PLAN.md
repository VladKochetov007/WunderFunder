# SSM CPU/CUDA coefficient attribution, v1

The original model, failed GPU artifact, independent grader, and mixed tolerances remain unchanged. This is a diagnostic, not a new model or a deployment pass. Native/live SSM sources are not edited.

## Preserved failure

Frozen SSM source `227a5b8f...`; saved multi replay `cc01557a...`. Independent literal replay failed two second-block scalar coordinates22 and61. Their absolute errors are8.72910e-5 and2.21506e-5, compared with unchanged bounds7.95983e-5 and2.17705e-5. Readout passed. Single initialization passed every comparison.

## CPU sensitivity completed

`cpu_sensitivity_v1.json` was produced by durable service `wnn-ssm-coeff-cpu-v1.service`, invocation `873af15c77c54ce8854179b53f9772ef`, nested campaign scope with actual1CPU/2GiB/no-swap limits, CUDA hidden, exit0.3.8895seconds,575580KiB peak. No competition data and no optimizer updates.

The explicit-coefficient literal function is **bitwise identical** to the independently written recurrence for both frozen arms. A one-ULP downward perturbation of all first-block scalar transitions moves failing coordinate22 fromCPU-.297991544 to-.297883451, leaving2.08020e-5 difference fromGPU-.297904253. One-ULP first-block scale changes are much smaller. This shows plausible sensitivity to transition arithmetic; it does **not** show that any GPU transition actually differs in that direction, or establish the cause of coordinate61. Every perturbation and both directions are preserved; none is a proposed model change.

## Root-owned GPU capture: UNLAUNCHED

Exact proposed command:

```sh
.venv/bin/python research/campaign_2026-09-30/integration/ssm_coeff_diagnostic/gpu_capture.py --output research/campaign_2026-09-30/integration/ssm_coeff_diagnostic/gpu_coeff_capture_v1.pt
```

Run as one durable root-owned synthetic GPU job, suggested ID`c20260930_gpu_ssm_coeff_capture_v1`, with existing aggregate/5GB monitor and source binder. No new input RNG, fitting, gradients, optimizer, or EMA updates. The runner strictly verifies original checkpoint/replay hashes and every final weight tensor for both multi and single. It requires actual backend flags to match the saved failure and makes no precision/TF32 changes.

It captures exact FP32 CPU and GPU scalarrho[64], real rotationa/b[32], normalized injectionscale[96], and half-life/logrho/rho/phasecos/sin intermediates. It additionally saves each GPU block's actual input, normalized drive, FFT state timeline, final reset prediction/state, coefficient bit differences, and rerun-versus-original comparisons. These distinguish coefficient differences from projection/nonlinearity and FFT contributions. Exclusive PT/JSON outputs preserve evidence; expected artifact size under25MB. Frozen model weights must remain bitwise unchanged after capture.

## Subsequent CPU attribution

After root captures coefficients, suggested CPU command under nested1CPU/2GiB/300s campaign scope:

```sh
.venv/bin/python research/campaign_2026-09-30/integration/ssm_coeff_diagnostic/cpu_diagnose.py --capture research/campaign_2026-09-30/integration/ssm_coeff_diagnostic/gpu_coeff_capture_v1.pt --output research/campaign_2026-09-30/integration/ssm_coeff_diagnostic/cpu_captured_replay_v1.json
```

The CPU reader checks every source/input identity and captured CPU coefficient before testing:

- CPU maps with actual GPU coefficients: direct deployment-style coefficient substitution.
- Fixed GPU drives with CPU coefficients: isolate transition arithmetic on the GPU's actual inputs.
- Fixed GPU drives with GPU coefficients: isolate FFT versus literal recurrence.
- Fixed baseline CPU drives with GPU coefficients: transition sensitivity without nonlinear cascade changes.

All comparisons retain `atol2e-5,rtol2e-4`; no pass is a quality claim. Independent critic must reproduce attribution. An inference coefficient serialization/cache grammar will be proposed **only after actual captured differences explain the failure**, and would have to preserve training gradients, invalidate on FT/optimizer changes, and bind exact final checkpoint parameters. No serialization or cache implementation is included in this diagnostic.
