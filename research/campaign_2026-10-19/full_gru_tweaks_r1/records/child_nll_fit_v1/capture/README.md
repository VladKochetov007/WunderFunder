# NLL terminal verification capture

This folder indexes the bounded terminal-verifier observation for `fg19_child_nll_f0_s1001_fit_v1`. The support unit was `computer-vision-homework-fg19-child-nll-verify-v1.service`, invocation `91948e17c5e74cefa32ab2a546a1b46c`. Its single capture was persisted before parsing at 2026-10-21 01:53:51 UTC. The raw response is 54,641 bytes with SHA-256 `62688aa0f92ac8933f11745a32ecb8b2f461869c71b366a9ca7fa7def3819be7`; transport reports no retry and empty stderr.

The extracted `VERIFICATION.json` reports `PASS` for bounded integrity and provenance checks. It records six fixed epochs, 44,760 successful updates, 4,476 batches, selected epoch 5, next epoch 6, zero AMP-skipped updates and zero nonfinite updates. Strict FP32 finite model/optimizer checks and selected-epoch/training-state tensor identity passed. The captured quality score is null: no predictive quality evaluation is claimed. The selected `model_ep5.pt` size and SHA below are copied from accepted verifier metadata; this documentation does not read or hash the model payload.

The support unit exited successfully with `MainPID=0`, `ExecMainCode=1`, and `ExecMainStatus=0`. Its post-exit cgroup was unavailable (`ControlGroup` empty, `MemoryCurrent` not set), so `MemoryPeak` (2,465,792 bytes) is reported only as the systemd service property and is not treated as full process/cgroup RAM. Parent-campaign memory/events and selected-GPU readings are separate point-in-time observations; two unrelated GPU owners were observed and left untouched.

The durable root acceptance was written before the exact stop. The stop targeted only the accepted NLL native unit and its one successful verifier. It did not touch failed or foreign units. The native exit and verifier result are integrity/provenance evidence, not a quality result.

The structured, source-linked record is [OBSERVATION.json](OBSERVATION.json). Supporting files and authority:

- Capture transport and raw response: [TRANSPORT.json](../terminal_verification_v1/capture/TRANSPORT.json), [STDOUT.raw.json](../terminal_verification_v1/capture/STDOUT.raw.json).
- Exact extracted receipts: [VERIFICATION.json](../terminal_verification_v1/capture/VERIFICATION.json), [VERIFIER_ADMISSION.json](../terminal_verification_v1/capture/VERIFIER_ADMISSION.json), [VERIFIER_LAUNCH.json](../terminal_verification_v1/capture/VERIFIER_LAUNCH.json), and empty [unit.log](../terminal_verification_v1/capture/unit.log).
- Root capture binding: [ROOT_CAPTURE_BINDING.json](../terminal_verification_v1/capture/ROOT_CAPTURE_BINDING.json). Its status is `PASS_EXACT_FIVE_LITERAL_BINDINGS_ONLY`; it binds the exact reader, admission, launch, and extraction pins.
- Admission and launch context: [ROOT_ADMISSION_MANIFEST.json](../terminal_verification_v1/ROOT_ADMISSION_MANIFEST.json), [ACTUAL_LAUNCH_METADATA_REVIEW.json](../terminal_verification_v1/ACTUAL_LAUNCH_METADATA_REVIEW.json), and [LAUNCH_ACCEPTANCE.json](../terminal_verification_v1/LAUNCH_ACCEPTANCE.json).
- Durable accepted result: [TERMINAL_ACCEPTANCE.json](../TERMINAL_ACCEPTANCE.json).
- Later exact stop and its root binding: [exact-stop README](../exact_stop/README.md), [EXACT_STOP.json](../exact_stop/EXACT_STOP.json), [raw stop response](../exact_stop/STDOUT.raw.json), and [ROOT_STOP_BINDING.json](../exact_stop/ROOT_STOP_BINDING.json).
- Native exit/resource context: [TERMINAL_EVIDENCE.json](../TERMINAL_EVIDENCE.json), [native resource reconciliation](../NATIVE_RESOURCE_RECONCILIATION.json).
- Scoped Path authority: [HUMAN_PATH_COMPAT_EXCEPTION_AUTHORITY_V1.json](../../../preparation/child_control_terminal_verifier_v3/HUMAN_PATH_COMPAT_EXCEPTION_AUTHORITY_V1.json). It covers only the reviewed exact dictionary-path `PosixPath` decoder in training-state metadata, with `weights_only=True`; it does not grant general Path compatibility.
- Review lineage: [NLL checker root preparation](../../../preparation/child_nll_terminal_check_v1/ROOT_PREPARATION.json), [NLL V2 final-body source review](../../../preparation/child_nll_terminal_verifier_v2/FINAL_BODY_SOURCE_REVIEW_V2.json), and [capture-reader source review](../../../preparation/child_nll_terminal_result_capture_v1/SOURCE_REVIEW_V1.json).

No model, training-state, dictionary, cache-array, dataset, trainer, or source-tree payload was read or hashed to write this index.
