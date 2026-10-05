# E471 precision-policy diagnosis and v2 probe review

## Diagnosis

The v1 CUDA parity failure is consistent with different cuDNN TF32 policies,
not evidence that the custom GRU equations or tolerances should be changed.
On the fixed fit-only B4×256 batch, runtime defaults (`cuda.matmul.allow_tf32`
false, `cudnn.allow_tf32` true, cuDNN enabled) reproduced the original mismatch:
459/2048 output values and 36/128 final-state values exceeded the unchanged
`atol=2e-5, rtol=2e-4` thresholds; maximum absolute output/state differences
were `1.21236e-4` and `1.16259e-4`. Input and parameter gradients passed their
original thresholds, but the full parity gate correctly failed.

The diagnostic then ran both candidate and reference under one explicit policy.
With TF32 disabled for both CUDA matmul and cuDNN, cuDNN enabled, and
`float32_matmul_precision="highest"`, every output, final-state, loss, input
gradient, and parameter-gradient check passed the original tolerances. Maximum
absolute differences were `1.44542e-6` (output), `2.20537e-6` (state),
`1.76806e-9` (input gradient), and `3.27826e-7` (parameter gradient). Disabling
cuDNN also passed, but is unnecessary and slower. This identifies a shared
full-FP32 backend policy as the actionable fix; it does not establish bitwise
equivalence between cuDNN and the scripted cell.

The v1 diagnostic ran as `e471-precision-policy-diagnostic-v1`, exit 0, with
verified teardown, no worker OOM, 9.21 seconds charged, 0.380 GB peak observed
VRAM, and 1.075 GB peak RAM. Its sealed spec binds the probe source SHA
`5a41d096c4c0290346c7afbeba7e5930ba70aaee6baee2f7f56246384eeaf067`; the
working copy was subsequently edited by adding an explicit precision setter,
two policy entries, and a type annotation change (post-run SHA
`c35f1b7ff65d4405c3056b2d6eb8af79f7f8afd4d1d577646b692fde5fa19b8a`). The
post-run version is preserved under `versions/`; the exact sealed source was
reconstructed and hash-verified, then restored as the working copy. See
`PRECISION_SOURCE_RECONCILIATION.md` for the byte-identity receipt. The later
edited source was not run and gets no credit for these results.

## Independent v2 probe decision

**PASS for the bounded v2 GPU update probe only.** The separately sealed
`e471-recurrent-real-gpu-probe-fp32-v2` binds all 14 source entries without
hash or size mismatch. It exited 0 with verified teardown, no worker OOM,
11.62 seconds charged, 0.384 GB peak observed VRAM, and 1.802 GB peak RAM.
Its fixed fit-only probe reports alpha-zero parity passing the original v1
tolerances, with the exact maxima above. All six alpha × optimizer arms made
one finite parameter-changing update. No test rows were loaded or scored.

The receipt also states the common policy explicitly (`cuda.matmul.allow_tf32`
false, `cudnn.allow_tf32` false, cuDNN enabled, matmul precision highest) and
records its wrapper SHA. This is a deliberate policy change from historical
E470 cuDNN TF32 behavior. Results under it must be described as a common-FP32
recurrent experiment, not as exact numerical continuation of E470's historical
backend. The probe supports proceeding to the preregistered fixed recurrent
fit under this same sealed policy; it is not itself a full-fit or scientific
result and does not authorize changing source, tolerances, or the fit matrix.

Evidence: `recurrent_activation/precision_gpu_v1/result.json`,
`recurrent_activation/e471_precision_v1.sealed.json`,
`runner/attempts/e471-precision-policy-diagnostic-v1.json`,
`runner/logs/e471-precision-policy-diagnostic-v1.log`,
`recurrent_activation/PRECISION_SOURCE_RECONCILIATION.md`,
`recurrent_activation/gpu_probe_fp32_v2/result.json`,
`recurrent_activation/e471_gpu_probe_fp32_v2.sealed.json`, and
`runner/attempts/e471-recurrent-real-gpu-probe-fp32-v2.json`.
