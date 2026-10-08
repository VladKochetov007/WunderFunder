# E465 V6 receipt and precision-gate source proposal

This version consumes the completed E463 V6 output bundle and the same immutable E465 input manifest. It is source-only: it has not been staged to a provider, profiled, or fitted.

The preflight gate requires the accepted V6 version-1 delivery receipt, a successful terminal receipt with the complete output inventory collected and hash-verified, the exact V6 source/archive and 12-file required asset closure, the pinned parent and raw parquet identities, and the 16 source-qualified VALID fit sequences. It checks those sequence pairs against the fixed 10,535 train / 1,405 VALID-fit / 468 evaluation fold. It does not load or score validation predictions.

`--phase profile` runs the existing two-batch, all-eight-device full-shape profile on real data: two global batches of 32 sequences, 20,000 rows per sequence, 1,000-row chunks, and 40 updates per precision. The worker checks finite losses and pre-clip gradient norms for every update, finite final parameters/optimizer/EMA/recurrent state, and records paired FP32/BF16 deltas for the same host batches. Those deltas are diagnostic evidence; this gate does not set a parity threshold or admit a fit.

A fit requires the exact profile plus a separate `wnn.e465-fit-authorization.v1` receipt written by root and bound to the profile bytes, precision-gate bytes, V6 terminal receipt, input manifest, source hashes, and eight-hour whole-worker bound. The full-fit recipe and outputs are unchanged from the preceding E465 source. The authorization file is not included here, so this packet cannot start a fit by itself.

Example invocation after root has assembled a terminal bundle and immutable inputs:

```sh
python scripts/run_e465_tpu.py \
  --phase profile \
  --project-root /kaggle/input/<exact-project-mount> \
  --input-manifest /kaggle/input/<exact-input-mount>/e465-input-manifest.json \
  --v6-bundle /kaggle/input/<exact-v6-bundle-mount> \
  --output-dir /kaggle/working/e465-profile
```

The input paths and actual V6 bundle must be bound by the packet manifest before any provider launch. A fit invocation additionally supplies `--profile-report` and `--fit-authorization`; both are rejected in profile mode.

## Source scope

The copied bridge, distributed objective, host contract, OneCycle schedule, recurrent model, and existing focused tests are byte-identical to the prior staged E465 source. The worker changes are limited to V6 terminal/input receipt validation, finite profile-state checks, paired precision diagnostics, profile binding, and a root-bound fit authorization gate. The runner validates these gates before importing the JAX worker. No objective, model, target, split, seed, optimizer, schedule, batch, chunk, epoch, or prediction rule changed.

The receipt validator itself uses only the standard library. The focused suite imports JAX for a worker-profile validation regression, but loads no data, Parquet files, or NPZ contents. The existing distributed-objective and recurrent-state tests require two forced CPU devices. No TPU runtime, credentials, or provider APIs are needed by these tests.
