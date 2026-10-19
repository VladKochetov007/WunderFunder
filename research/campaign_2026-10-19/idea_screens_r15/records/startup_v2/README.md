# R15 startup V3 transport record

This packet uses the corrected V3 source manifest and independent review. The source README launch-time bytes are preserved in `PREPARED_README.md`.

`LOCAL_IMPORT_SMOKE.json` records a fresh-process B-shaped smoke. All 25 actual owned-prefix modules matched the manifest, all NumPy/Parquet/Torch read sentinels stayed untouched, CUDA remained hidden and uninitialized, Torch thread counts were one, TF32/autocast were disabled, and no sample/model payload was opened. The first local harness attempt is preserved in `LOCAL_IMPORT_SMOKE.stderr.txt`; it exposed an environment-assertion expression error in the harness after import/config checks, not a source failure. The corrected fresh-process run is `LOCAL_IMPORT_SMOKE_V2.stdout.txt` and passed.

`ASSEMBLY_SPEC.json` binds the fixed R15 V1 identity, exact 25-module map, six canonical pins, three-file R7 allowlist, resource envelope, and systemd `CUDA_VISIBLE_DEVICES`/OMP/OPENBLAS/MKL environment. `ASSEMBLY_RECEIPT.json` records the exact 29-file transport. `REPLAY.json` binds the private scripts to the pinned reviewed R15 V2 assembler with only startup packet/private output paths changed. The fresh V3 CLI configures Torch thread counts and disables TF32 before the executor/data mapping. No remote action was performed here.


## Executed R15 V1 record

The preceding section documents local packet assembly. Root performed the separately authorized bounded launch; the exact unit completed with exit status 0 and was stopped after raw capture validation. See the [terminal capture](../terminal_capture/README.md), [resource reconciliation](../RESOURCE_RECONCILIATION.json), [metadata review](../RESOURCE_METADATA_REVIEW.json), [capture validation](../ROOT_CAPTURE_VALIDATION.json), and [stop receipt](../STOP_RECEIPT.json).

The root packet also retained [ROOT_CFG_SUMMARY.json](ROOT_CFG_SUMMARY.json), a keys-only summary, and [ROOT_STARTUP_REVIEW.json](ROOT_STARTUP_REVIEW.json), which records the local review harness correction and the V3 packet review. The launch-time source README bytes remain preserved in [PREPARED_README.md](PREPARED_README.md); V1/V2 source snapshots and the unexecuted `startup_v1` packet remain preserved separately.
