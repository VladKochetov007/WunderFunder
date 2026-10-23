# AllThree terminal verifier V3 supplement

V3 is the local, versioned response to the preserved V2 preflight failure. It changes only `check_fit_receipts`: the five embedded receipts keep their fixed byte-count/SHA checks; `ADMISSION.json` and `LAUNCH.json` are compared with the existing native evidence files; the other three original fit receipts are classified as root-only payloads and are never read from or staged into the native evidence directory.

The packet is prepared locally. Its exact-body source review is pending; the static payload check below is not that review and does not authorize staging, admission, or execution. The V2 attempt and follow-up snapshot are historical evidence with their own limits, not a reason to retry without fresh gates.

## V3 packet files

- [V3 worker body](REMOTE_SUBMISSION_V3.py) is 707,678 bytes, SHA-256 `12f357f2127abded88b0e89d5ba08323310bcc5337a980f89a2ce90b4ec57151`. It is an exact V2-derived body with only `check_fit_receipts` replaced.
- [V3 binding](V3_BINDING.json) is 18,346 bytes, SHA-256 `1d3677bc4efcaa25cfc7dd57c1ad5ba248dbd1140906e293ab4f4f162010a170`. It carries the preserved failed-preflight and read-only outcome pins, inherited root preparation/native evidence, and all fifteen unchanged payload identities.
- [Static root payload check](ROOT_PAYLOAD_CHECK.json) is 9,991 bytes, SHA-256 `f18279e7fbb834efb784e32f8381df7a837f237f87b1a3dfd5b4da4c0271f6f3`. It records a static exact-body/payload-binding pass for fifteen payload joins and seven private code files, including the strict native identity binding. It does not independently source-review the V3 worker or establish execution readiness.
- [Focused local fixture](test_allthree_terminal_verifier_v3_receipt_sources.py) is 8,135 bytes, SHA-256 `23e93e11775db25fc983ee1819edc4e76b4cccb443f8aa1f7f67c2ee00309be3`. It parses worker source only and checks the inverse-byte/AST relation and exact native versus root-only receipt classes.
- [Local checks](LOCAL_CHECKS.json) is 2,740 bytes, SHA-256 `a2351b86b18173cf13ae63b4d241335c97e9ba69a845b97dbffffbe1f09318dc`; it records three focused tests on the final fixture revision and passing Ruff checks, with no worker or checker import/execution.
- [Packet README](README.md) describes the narrow repair and its local-only status.
- Independent exact-body review is pending. Link its final artifact here when it is created; until then, no V3 source-review pass is claimed.

## Preserved preflight and evaluation discovery

The [V2 preflight failure record](../../records/child_allthree_fit_v1/terminal_verification_v1/README.md) documents one failed V2 SSH preflight before `D.mkdir` or systemd submission, followed by one read-only snapshot. That snapshot found the exact V1 verifier support unit absent and the private directory absent, so it records zero actual verifier CPU units. It is time-bounded evidence, not a future admission gate; no blind retry is implied.

The [first-cell evaluation discovery](../first_cell_evaluation_v1/README.md) links a static, source-qualified note (16,085 bytes, SHA-256 `bb10d31ba96593a2770b4fe3cda8e1c38f4bef6399c8b7cd5d73d84bf537a982`). It pins fourteen source files from commit `68ff682977b822f38351aac0f621146f5f13b440` and creates no prediction, evaluation, score, quality result, or fit. A later evaluation design must first reconcile the 468-group evaluation-only export with the scorer's full-valid input shape/order, and bind the current split, fixed `model_ep5`, representation state, ordered membership, output dtype/shape/hash, sidecar, and reviewed checkpoint-loading boundary. No evaluation is authorized by this discovery.
