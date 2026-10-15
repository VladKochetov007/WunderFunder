# E463 actual-data TPU bridge V8

This source-only packet repairs the V7 host-admission error and completes the actual-data bridge handoff. It is prepared for root review; **provider_push_authorized=false** and no Kaggle provider call was made.

## What changed

V7 compared registry["dataset_ref"] with vladkochetov07/wunder-funder, but the pinned portable_inputs.json has no such field. The preserved V7 terminal receipt shows the resulting host_admission failure before JAX or a TPU worker started. The V8 builder removes only that unsupported field comparison. It still binds the whole registry SHA-256, version and fold split SHA-256, per-split file SHA-256, original byte count, and original Parquet footer length and digest.

Kaggle resolves the attached raw dataset from kernel-metadata.json as vladkochetov07/wunder-funder. The actual V7 input_mounts.json receipt records its mount at /kaggle/input/datasets/vladkochetov07/wunder-funder. V8 requires that exact owner/slug directory, rejects an ambiguous equal-size parquet pair elsewhere, checks the mounted file sizes and footer identities, then streams a full SHA-256 over each mounted parquet before staging read-only links.

The original E463 bridge recipe remains fixed: eight TPUv5e devices, B16 split across two sequences per device, 2,000-row update chunk, the pinned epoch-3 parent, fresh FP32 AdamW, one OneCycle update, the existing global pooled four-term objective, stopped incoming EMA targets, gradient clip 1.0, and the existing valid-fit-only source-qualified membership. The packet still runs the unequal-mass distributed reduction diagnostic and four synthetic full-shape profile updates. It now saves and reloads student, EMA, and optimizer state, then exports one complete 20,000-row fit-VALID sequence from the reloaded updated EMA in 2,000-row chunks. That export is an execution artifact, not a score or held-out claim.

The temporary project tree consists of symlinks to mounted source, caches, and parquet files. V8 removes that tree when the worker exits and again after a parent-side worker timeout, preserving the mounted payloads and avoiding a large or private-data download. The manifest allows at most 100 MB of output collection.

## Provenance and limitations

build_v8_packet.py generates packet/probe.py, Kaggle metadata, a deterministic bundle, and packet_manifest.json from the pinned V7 source. It reads the canonical [V7 probe](../../../campaign_2026-10-06/cloud_preflight/tpu_packet_v7_inventory_repair_20261008/packet/probe.py) and [packet manifest](../../../campaign_2026-10-06/cloud_preflight/tpu_packet_v7_inventory_repair_20261008/packet_manifest.json), the original [host-admission terminal receipt](../../../campaign_2026-10-06/cloud_preflight/tpu_v7_terminal_20261008_r1/attempt_001/outputs/cloud_preflight_tpu_v7/terminal.json), its [actual mount inventory](../../../campaign_2026-10-06/cloud_preflight/tpu_v7_terminal_20261008_r1/attempt_001/outputs/cloud_preflight_tpu_v7/input_mounts.json), the pinned [portable registry](../../../campaign_2026-10-04/portable_inputs.json), and [fold-0 asset manifest](../../../campaign_2026-10-06/cloud_preflight/tpu_packet_v7_inventory_repair_20261008/reference/fold0_assets_manifest.json) in their original folders. Their exact hashes are recorded in packet_manifest.json. The V8 folder contains no checkpoint weights, cached predictions, parquet payloads, or credentials.

The V7 terminal fixture is an incident record for the matching V7 host-admission error. It does not prove that V7 reached TPU execution. The resource plan therefore leaves TPU runtime, RSS, HBM use, and phase durations explicitly unmeasured. The worker has a 1,500-second child deadline inside a 1,650-second parent deadline; it requires at least 20% host memory headroom and caps measured TPU HBM peak at 80% of device capacity. The actual-data update and export remain unexecuted until root admits a provider run.

All selected fit-VALID groups were already exposed and are included only for backend/recipe compatibility. No metric calculation, held-out claim, model-quality result, or public-score forecast is produced.

## Offline checks

The tests bind packet/source/registry/mount hashes, exercise the actual portable registry and actual V7 mount receipt, read byte counts and Parquet footers from the supplied local parquet files, and reject mutated SHA, byte, footer, split, and mount identities. A tiny Parquet-shaped fixture checks full-file and footer hash failures without staging large files. The test suite does not initialize JAX, require a TPU, or call Kaggle.

## Root-owned provider commands

Run only after root review and fresh Kaggle quota/resource admission. From the repository root, the exact dispatch command is:

    kaggle kernels push --path research/campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8/packet --timeout 1800

Then inspect the kernel status:

    kaggle kernels status vladkochetov07/e463-tpu-realdata-reload-v8-20261013

After terminal status, collect the complete output inventory into the designated attempt directory:

    kaggle kernels output vladkochetov07/e463-tpu-realdata-reload-v8-20261013 --path research/campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8/collected/attempt_001

Check the collected byte count against the 100 MB allowance and reconcile every entry in packet_manifest.json, including terminal/complete and worker receipts, input identity and source materialization, TPU inventory/HBM records, distributed reduction and synthetic profile records, both real-data update receipts, student/EMA/AdamW checkpoints, full-sequence prediction NPZ, cleanup receipt, worker log, and the final parent/device-worker phase receipts. Preserve partial checkpoint/progress artifacts after an internal timeout; classify a provider or resource failure as execution evidence, not a scientific negative.

## October15 terminal collection

[Bounded collector](collect_terminal_r1.py) downloaded17.50MB into physical `artifacts/kaggle_tpu_v8_terminal_20261015_r1`; [terminal manifest](TERMINAL_COLLECTION_R1.json) binds every payload hash/length and exact kernel. [Remote inventory](REMOTE_INVENTORY_R1.json) and [provider log](PROVIDER_LOG_R1.txt) retain listing and session evidence without signed URLs. Root rehashed every collected payload successfully.

The actual-data receipt reports one globalB16/C2000E463update across8TPUv5e devices, model/optimizer/EMA bitwise checkpoint reload, finite loss/gradient, causal future-perturbation and zero-state checks, and an exported20k-row prediction on source-qualified fitting group(valid,254). Synthetic full-shape updates are separately labeled. This is compatibility/resource evidence; no full fit, heldout grade or public forecast. Independent interpretation and native compatibility remain pending before largerTPUallocation.

[Native consumer diagnosis](NATIVE_COMPAT_CONSUMER_DIAGNOSIS_R1.json) preserves the initial rank mismatch: the producer emits[1,20000,23], whereas the first [nativecheck](check_native_compat_r1.py) expected[20000,23] and stopped before inference. The corrected [R2check](check_native_compat_r2.py) and [actual result](NATIVE_COMPATIBILITY_R2.json) replay all20krows: finite/native resetbitwise pass, native-vsTPU maxabs.0077728/mean.001134 fails the declared1e-4 numericaltolerance. No substantialTPUfit follows this unresolved comparison. [PyTorch reference diagnostic](check_pytorch_reference_r3.py) checks exact productionfeaturehash and the same updatedEMA forward to localize the difference.

## Exact-input forward diagnosis R4

[Actual20kresult](FEATURE_FORWARD_DIAGNOSIS_R4.json) and [exactexecutedsource](check_feature_forward_r4.py) rebuild producer NumPy features from the same verified valid254/seq267134787 raw row. The NumPy input SHA exactly matches the TPU producer. The only NumPy/Torchfeature difference is since_i1px, max4.77e-7; rerunning FP32PyTorch on the exactNumPyinputs changes main2outputs by only6.56e-7 and leaves maxTPUgap.00777236. Native/PyTorch agree within1.43e-6. This rules out the featurehash discrepancy as the cause of the observed gap, not every possible model/export/backend mismatch. [Sourceprovenance](R4_EXECUTED_SOURCE_PROVENANCE_R1.json) verifies the result's executedsourcehash; the historicalrawsource retains its recorded import/format issues rather than silentlychanging its pin.

The frozen JAXforward uses jnp.matmul without an explicit precision argument. [JAXprecisiondocumentation](https://docs.jax.dev/en/latest/201/precision.html) and [defaultmatmulprecision](https://docs.jax.dev/en/latest/_autosummary/jax.default_matmul_precision.html) explain why FP32storage alone does not guarantee fullFP32multiply onTPU. Reducedmultiplyprecision is a testable next hypothesis, not yet a measured cause. A bounded CPUoperand-rounding reconstruction is assigned before any newTPUjob; no modelquality/fullTPUfit conclusion.
