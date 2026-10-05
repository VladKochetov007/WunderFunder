# E477 scorer hardening addendum

The unsealed scorer now validates the sealed spec with `validate_spec`, requires the recorded systemd unit result to be `success` with `ExecMainStatus=0`, and checks that each checkpoint and loaded model retains the identity normalization buffers (`mean == 0`, `std == 1`). The fit-only feature normalizer is applied externally to causal inputs; the checkpoint buffers must therefore remain identity. Bootstrap moment arrays are sorted by the numeric component IDs before both helper-based paired contrasts and custom interaction resampling, so identical seeds now map to the same component draws across all contrast families.

Checks performed without scoring heldout data or using a GPU:

- `py_compile` passed.
- Read-only `verify_sealed_run` passed for the existing successful 12-arm fit attempt (`Result=success`, `ExecMainStatus=0`), all source/input/artifact hashes, and 12-arm manifest.
- A bounded synthetic check built four-component centered moments, confirmed the pair bootstrap ran with finite values, and independently reproduced the custom mixture-by-dropout interaction interval with the same sorted component order and 200 draws (seed 991).
- The earlier full-mask synthetic check and scorer/actor forward-parity checks remain documented in `GRADER_REVIEW_V2.md`; this addendum leaves the selected-label and prediction path unchanged.

Current scorer SHA-256: `0e566cf0608c02e77c24767bbea2c9ea25d9e7bfd54b3625fe68a32b366476d5`.
