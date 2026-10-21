# AllThree verify-v2 result reader V3

[The V3 template](RESULT_CAPTURE_TEMPLATE.py) is a separate local reader candidate derived from the frozen V2 template. It changes only successful-exit handling for `VERIFICATION.json`: the support runner calls the checker and prints its JSON result, while systemd appends stdout and stderr to `unit.log`. The reader captures `unit.log` as before and records an absent `VERIFICATION.json` as `{ "bytes": 0, "present": false, "absence": "not_found" }` after confirming the output directory still has its original device and inode. It does not synthesize a verification file or parse the log.

Admission and launch receipt schema/identity checks, the exact support invocation check, cgroup and selected-GPU observations, ownership checks, the seven private-code pins, the three required captures (`VERIFIER_ADMISSION.json`, `VERIFIER_LAUNCH.json`, and `unit.log`), and the byte caps remain as in V2. Any missing required file, permission error, symlink, changed directory, or non-`FileNotFoundError` capture failure still fails closed. The optional absence entry contributes zero bytes to the aggregate cap.

[The focused static proof](test_result_capture_v3_contract.py) parses the V2/V3 readers and V5 producer as ASTs. It proves the exact normalized V2-to-V3 source delta, checks that only a missing `VERIFICATION.json` is recorded as absent, and binds the V5 runner's printed result and stdout/stderr `unit.log` destination. It does not import the reader, producer, checker, or helpers.

Five support pins in the template remain unbound: admission bytes/SHA-256, launch bytes/SHA-256, and support invocation ID. This packet does not authorize or claim a support attempt, B query, native acceptance, model/state integrity, or quality result. Root must review this exact template before separately binding actual receipt pins for a read-only capture. Preserve the failed V2 capture and do not rerun the support unit.

See [the V3 source/AST binding](V3_SOURCE_AST_BINDING.json) and [local checks](LOCAL_CHECKS.md) for exact pins and scope.
