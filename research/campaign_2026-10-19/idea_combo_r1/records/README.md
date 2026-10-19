# COMBO R1 operational records

These records document the single COMBO R1 launch, raw-byte terminal capture, and exact stop. Scientific score review is recorded separately in the campaign result materials.

The run used `computer-vision-homework-idea-combo-r1-v1.service`, invocation `66d47211356441deba6a6ef9b712e844`, preparation manifest `ec4bd56f08eaebeb13e8f359b15128f1d0a67a329532b3bf9ec664fb72480bee`, and sample SHA-256 `b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64`. The native unit exited successfully; capture validation passed before the exact unit stop.

- [Launch intent](SOLE_LAUNCH_INTENT.json), [launch transport receipt](LAUNCH_TRANSPORT.json), and raw launch stdout/stderr preserve the sole launch command and invocation.
- [Capture validation](ROOT_CAPTURE_VALIDATION.json) records PASS before stop. The raw JSON envelope and byte-exact extracted files are in [terminal_capture](terminal_capture/README.md).
- [Stop receipt](STOP_RECEIPT.json), raw stop stdout, and stderr preserve the exact unit stop. Raw stdout SHA is distinct from the enclosing JSON receipt SHA.
- [Resource reconciliation](RESOURCE_RECONCILIATION.json) keeps systemd, helper, parent cgroup, and archive metadata in separate scopes. [Independent metadata review](RESOURCE_METADATA_REVIEW.json) records raw-file/identity checks and the unresolved memory-scope difference.
- [Startup packet notes](startup_v1/README.md) identify the final reviewed local bundle and superseded local-only assembly.

No model or array payload was opened or copied locally for this operations record. The R15 reference archive was stat-only at COMBO admission, then the bounded COMBO producer reported its archive hash and six restored/validated arrays. The new COMBO archive is represented by producer metadata and capture-time stat size only.

The [completed scientific scalar audit](../RESULT_AUDIT.json) and its [stdlib implementation](../audit_scalar.py) use compact captured sufficient statistics only. Source-only checks and stdout bindings are recorded in [the test receipt](../TEST_RECEIPT.json); `preparation_v1/` and `preparation_v2/` retain local pre-execution check logs. These are not additional remote attempts or fits.
