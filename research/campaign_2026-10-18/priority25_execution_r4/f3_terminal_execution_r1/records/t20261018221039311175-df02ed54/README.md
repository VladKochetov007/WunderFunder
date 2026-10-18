# Wide terminal collector invocation

**Status:** captured successfully; only this exact exited collector unit was stopped, after durable capture. This operation preserves the original wide F3 terminal checkpoint. It makes no grade or quality claim.

The native wide attempt `computer-vision-homework-r01-wide-f3s1001-ep25-v1` finished with exit status 0. The terminal collection receipt reports source-snapshot and producer-model hash matches and records the same-arm stage-1 resume state. The independent terminal verifier and final limitations are indexed in the [maintenance note](WIDE_TERMINAL_MAINTENANCE_20261018T2210Z.md).

The local launch intent bound wrapper manifest `57a07a08dff9405149b9927979fc60ba5612c68a7be1a16a2225ba963c26d121`, independent review `c47d4f7b41895f3c680185c269f5aac0af2e63b5d19de76975ffba195b50d899`, and plan `b303146cf9db05e0a3bddf8f245d3c9b880d055ce7e97ebe5dd13510b5f4a2d8`. The remote admission repeats those wrapper/plan pins and binds support manifest/review `7e4fce0d7232a487a769e12d611050bc608f9cc7fcea300bc2063525abfa38dc` / `de709897397d7cd775d2aba84fb71f8f7d91504802814f5bcfc26e48e203763b`, remote terminal helper SHA `d7c75321065813b9a6a62333937d6591dab72d0b4eea38c5ce05b16517e75ac2`, and native attempt record SHA `f1092fe8ba5fafd7d0020ff35fa77682f0b4f8aa35167b054a661e2008f96257` (exit 0).

## Evidence index

- [LOCAL_LAUNCH_INTENT.json](LOCAL_LAUNCH_INTENT.json) — exact arm, operation, invocation and reviewed packet/plan hashes reserved before the remote launch.
- [REMOTE_LAUNCH_RESPONSE.json](REMOTE_LAUNCH_RESPONSE.json) and [REMOTE_LAUNCH_V1.json](REMOTE_LAUNCH_V1.json) — launch response and exact remote launch record/unit.
- [REMOTE_ADMISSION_V1.json](REMOTE_ADMISSION_V1.json) — validated plan, native attempt, stage-1 pair, wrapper/support pins, live sequence and resource admission.
- [REMOTE_CAPTURE_TRANSPORT_1.json](REMOTE_CAPTURE_TRANSPORT_1.json) — exact raw stdout bytes from remote capture, including base64 raw files and their byte/hash declarations; SHA-256 `481b05d69fec269204f0493e36b49bd1931a3f72dcdcb4fe3d313bad3a9d6bdb`.
- [REMOTE_CAPTURE_TRANSPORT_1.stderr.txt](REMOTE_CAPTURE_TRANSPORT_1.stderr.txt) — raw stderr from that transport (empty).
- [REMOTE_CAPTURE_RESPONSE.json](REMOTE_CAPTURE_RESPONSE.json) — parsed capture response and exact unit status.
- [REMOTE_HELPER_RECEIPT.json](REMOTE_HELPER_RECEIPT.json) and [REMOTE_RANK01_WIDE_F3_S1001_TERMINAL_COLLECTION.json](REMOTE_RANK01_WIDE_F3_S1001_TERMINAL_COLLECTION.json) — saved raw canonical helper receipt; byte-identical views of the collected payload.
- [LOCAL_EXECUTION_V1.json](LOCAL_EXECUTION_V1.json) — local raw-byte validation, capture-client recovery path and unit/helper status.
- [LOCAL_CAPTURE_CLIENT_FAILURE.stderr.txt](LOCAL_CAPTURE_CLIENT_FAILURE.stderr.txt) — original local schema-check traceback. It failed before issuing SSH capture; no client or response was patched.
- [REMOTE_STOP_RESPONSE.json](REMOTE_STOP_RESPONSE.json) and [REMOTE_STOP_RESPONSE.stderr.txt](REMOTE_STOP_RESPONSE.stderr.txt) — exact post-capture stop response and stderr.
- [LOCAL_STOP_V1.json](LOCAL_STOP_V1.json) — local record that capture was durable before requesting stop of only the named unit.
- Canonical receipt copy: [RANK01_WIDE_F3_S1001_TERMINAL_COLLECTION.json](../../../receipts/RANK01_WIDE_F3_S1001_TERMINAL_COLLECTION.json), 62,809 bytes, SHA-256 `45f7fcefb005eda4f95a31b2440688a160974d9a6d47f29e53e741ed2b9ea055`.

The recovered operation used the existing remote `terminal_remote_v1.py capture` with arm `wide`, operation `collector`, this exact invocation, and the manifest/review/helper hashes above; the corresponding existing `stop` command used the same arguments after capture became durable. Full command shape and the paired verifier invocation are in the [maintenance note](WIDE_TERMINAL_MAINTENANCE_20261018T2210Z.md).
