# V1 failed-unit capture

This folder preserves the bounded verifier V1 failure and its raw evidence. The failure occurred in the restricted Torch checkpoint loader on a `TorchVersion` metadata global; no checkpoint acceptance was produced by V1.

- [Capture validation](CAPTURE_VALIDATION.json) binds captured files.
- [Terminal evidence](TERMINAL_EVIDENCE.json), [verifier admission](VERIFIER_ADMISSION.json), [verifier launch](VERIFIER_LAUNCH.json), [unit log](unit.log), raw [stdout](STDOUT.raw.json), [stderr](STDERR.raw), and [transport](TRANSPORT.json) document the consumer attempt.
- [Failure summary](../FAILURE.json) records that this was not negative training evidence. The successful, narrowly corrected run is [V2](../../terminal_verification_v2/README.md).
