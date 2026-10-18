# Wide terminal verifier invocation

**Status:** verifier passed terminal structural checks; only this exact exited verifier unit was stopped, after durable capture. This is checkpoint/source/data identity evidence, not an F3 grade or quality result.

The verifier reported `status=passed`, `next_epoch=6`, checked same-arm stage-1 ancestry and the peer-pair contract, and verified the expected finite FP32 model/Adam tensors and optimizer counters. The measured counts, pins, resource limits and limitations are in the [wide terminal maintenance note](../t20261018221039311175-df02ed54/WIDE_TERMINAL_MAINTENANCE_20261018T2210Z.md).

The launch intent bound wrapper manifest `57a07a08dff9405149b9927979fc60ba5612c68a7be1a16a2225ba963c26d121`, independent review `c47d4f7b41895f3c680185c269f5aac0af2e63b5d19de76975ffba195b50d899`, and plan `b303146cf9db05e0a3bddf8f245d3c9b880d055ce7e97ebe5dd13510b5f4a2d8`. The remote admission repeats those wrapper/plan pins and binds support manifest/review `7e4fce0d7232a487a769e12d611050bc608f9cc7fcea300bc2063525abfa38dc` / `de709897397d7cd775d2aba84fb71f8f7d91504802814f5bcfc26e48e203763b`, remote terminal helper SHA `d7c75321065813b9a6a62333937d6591dab72d0b4eea38c5ce05b16517e75ac2`, and native attempt record SHA `f1092fe8ba5fafd7d0020ff35fa77682f0b4f8aa35167b054a661e2008f96257` (exit 0). The verifier also binds the captured collection receipt SHA `45f7fcefb005eda4f95a31b2440688a160974d9a6d47f29e53e741ed2b9ea055`.

## Evidence index

- [LOCAL_LAUNCH_INTENT.json](LOCAL_LAUNCH_INTENT.json) — exact arm, operation, invocation and reviewed packet/plan hashes reserved before remote launch.
- [REMOTE_LAUNCH_RESPONSE.json](REMOTE_LAUNCH_RESPONSE.json) and [REMOTE_LAUNCH_V1.json](REMOTE_LAUNCH_V1.json) — launch response and exact remote launch record/unit.
- [REMOTE_ADMISSION_V1.json](REMOTE_ADMISSION_V1.json) — validated plan, native attempt, stage-1 pair, wrapper/support pins, collected receipt, live sequence and resource admission.
- [REMOTE_CAPTURE_TRANSPORT_1.json](REMOTE_CAPTURE_TRANSPORT_1.json) — exact raw stdout bytes from remote capture, including base64 raw files and their byte/hash declarations; SHA-256 `fa9a34ebbefd9acf81d2e5a3801a56adc6fce9f90d19257c76bb94ac15cad0ac`.
- [REMOTE_CAPTURE_TRANSPORT_1.stderr.txt](REMOTE_CAPTURE_TRANSPORT_1.stderr.txt) — raw stderr from that transport (empty).
- [REMOTE_CAPTURE_RESPONSE.json](REMOTE_CAPTURE_RESPONSE.json) — parsed capture response and exact unit status.
- [REMOTE_HELPER_RECEIPT.json](REMOTE_HELPER_RECEIPT.json) and [REMOTE_RANK01_WIDE_F3_S1001_TERMINAL_VERIFICATION.json](REMOTE_RANK01_WIDE_F3_S1001_TERMINAL_VERIFICATION.json) — saved raw canonical verifier receipt; byte-identical views of the verifier payload.
- [LOCAL_EXECUTION_V1.json](LOCAL_EXECUTION_V1.json) — local raw-byte validation, capture recovery path and unit/helper status.
- [REMOTE_STOP_RESPONSE.json](REMOTE_STOP_RESPONSE.json) and [REMOTE_STOP_RESPONSE.stderr.txt](REMOTE_STOP_RESPONSE.stderr.txt) — exact post-capture stop response and stderr.
- [LOCAL_STOP_V1.json](LOCAL_STOP_V1.json) — local record that capture was durable before requesting stop of only the named unit.
- Canonical receipt copy: [RANK01_WIDE_F3_S1001_TERMINAL_VERIFICATION.json](../../../receipts/RANK01_WIDE_F3_S1001_TERMINAL_VERIFICATION.json), 65,132 bytes, SHA-256 `8444e28d22396937855b1465ab79b47b8cce58576f8f5cbf2509e58ccc1f8c50`.

The capture used the existing remote `terminal_remote_v1.py capture` with arm `wide`, operation `verifier`, this exact invocation, and the manifest/review/helper hashes above; the corresponding existing `stop` command used the same arguments after capture became durable. The command shape is indexed in the [maintenance note](../t20261018221039311175-df02ed54/WIDE_TERMINAL_MAINTENANCE_20261018T2210Z.md). No new remote operation or client repair was performed for this documentation update.
