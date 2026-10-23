# AllThree terminal verifier V4 supplement

V4 is preserved as a rejected local candidate. Its exact body and prior files remain untouched.

- [V4 body](REMOTE_SUBMISSION_V4.py) is 710,826 bytes, SHA-256 `c186e7b4473bc11681f61c55a741f17b2fdab828c7906090ef10574a8aa2df90`; [render binding](V4_RENDER_BINDING.json) records create-only local preparation.
- [Exact-body source review](FINAL_BODY_SOURCE_REVIEW_V4.json) is `REJECT_SOURCE_ONLY_NO_B_ATTEMPT`.
- [Producer/consumer review](../child_allthree_terminal_result_capture_v2/PRODUCER_PAIR_REVIEW_V4.json) also rejects the candidate: its durable admission object lacks `unit`, then the producer reply guard requires `durable_admission.get("unit") == UNIT` after submission.
- V4 was not submitted or executed. The one-line repair is versioned separately in the [V5 supplement](../child_allthree_terminal_verifier_v5/SUPPLEMENT_INDEX.md); V5 uses the same verify-v2 unit/private directory because V4 had zero B actions.

Do not use this body as a submission candidate. Its reviews and failure finding are retained as historical evidence.
