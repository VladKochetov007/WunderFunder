# E489 v2 probe import repair review

This versioned review preserves the E489 v1 probe's actual CLI-import failure
as the reason v1 is not runnable. It does not admit a retry or any full fits.

## Reviewed identities and checks

- `gpu_probe_fp32_duration_v2.py`: SHA-256
  `7ee1bf79dac158d7aaa6530f55173126ad5e8f8df1c4e50832d83e9b0ca19b20`
- `V2_CLI_IMPORT_RECEIPT.json`: SHA-256
  `bdafd2adfb9be1b556580a9def04c5196619bde7cb7413b6c4264003b75d7e80`
- Bound trainer: `af2ad34d8bac617fe5f53e28a2120c5d59a303888b54937e35e27edd1b925cdc`
- Bound tests: `fa5cd3da2afd9860b0a34d2a929ae1526747650d1f9dcd71cfbb34e553d66ec6`

V2 adds the missing `activation_placement` import directory before importing
`fit_activation_placement`, updates the probe identity string, and includes
the v2 filename in the probe's source-hash receipt. The rest of the numerical
probe implementation is unchanged from the reviewed v1 bytes. I reproduced
`--help` from `/tmp` through `scripts/limited` at 1 CPU / 2 GiB, then ran Ruff
and `py_compile` under the same hard limits; all passed. No data or CUDA was
initialized and the probe callable was not invoked.

**Source disposition: GO for a newly sealed v2 probe packet after the
controller binds this exact v2 source and output identity.** The v1 execution
failed during import before GPU/data work and remains an incident. This review
does not authorize a retry; root recorded that the admission cutoff passed
before the v1 attempt completed. Any future probe still requires the original
scientific, precision, timing, and memory gates, unchanged.
