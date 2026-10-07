# Independent E-463 TPU v2 packet review

Reviewer: native Luna subagent, read-only. Result: **PASS; no packet blocker
found in the reviewed scope.** The reviewer did not run tests or Kaggle
commands.

- V1's flat-mount assumption is corrected. V2 resolves the source manifest by
  frozen SHA-256, verifies the archive digest, and uses the observed nested raw
  dataset path.
- V2 safely rejects traversal, symlinks, duplicate/unlisted files and any
  mismatch against the 17-file manifest. The local archive has the exact
  manifest member set and all recorded file hashes/sizes.
- The checkpoint, JAX export, sidecar and model-source hashes remain fixed;
  sidecar ancestry and dimensions are still checked.
- The TPU backend, eight-device requirement, full-shape update and per-device
  HBM thresholds remain enforced. The only optimizer step uses synthetic data
  and does not constitute a scientific fit or score.
- Packet hashes match `tpu_prereg_v2.json`; Kaggle metadata remains private,
  offline, TPU-enabled and GPU-disabled.

The reviewer checked `tpu_probe_v2.py` and the materialized `tpu_packet_v2/`
against v1's terminal failure and E-508's observed Kaggle input layout. Local
tests are recorded separately in the test command receipt.
