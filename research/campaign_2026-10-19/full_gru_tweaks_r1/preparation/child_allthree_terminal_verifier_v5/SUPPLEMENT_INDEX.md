# AllThree terminal verifier V5 supplement

V5 is the reviewed one-key repair to the preserved V4 producer/consumer mismatch.

- [V5 body](REMOTE_SUBMISSION_V5.py) is 710,848 bytes, SHA-256 `4d555fd69954d94415c135e467eadd081aa6fd915686197e5c7e628e935b9374`; [binding](V5_BINDING.json) preserves the exact V4 candidate and rejection review, proves the one-line inverse-byte relation, and lists all 15 unchanged payload pins.
- [Exact-body source review](FINAL_BODY_SOURCE_REVIEW_V5.json) is `PASS_SOURCE_ONLY`; [root payload check](ROOT_PAYLOAD_CHECK.json) is `PASS_STATIC_EXACT_BODY_PAYLOAD_BINDING_ONLY`.
- The [V2 result reader](../child_allthree_terminal_result_capture_v2/RESULT_CAPTURE_TEMPLATE.py) is paired with this producer by [review V5](../child_allthree_terminal_result_capture_v2/PRODUCER_PAIR_REVIEW_V5.json), status `PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR`. The rejected [V4 review](../child_allthree_terminal_result_capture_v2/PRODUCER_PAIR_REVIEW_V4.json) remains available.
- V5 uses checker V2 from the [checker supplement](../child_allthree_terminal_check_v2/SUPPLEMENT_INDEX.md), the verify-v2 support unit, and its private output directory. The original verify-v1 failure is preserved and is not retried or stopped.
- The actual CPU support launch is navigated from [terminal verification V3](../../records/child_allthree_fit_v1/terminal_verification_v3/ACTUAL_INDEX.md). Root owns validation of actual receipts, checker outcome, acceptance, and exact stop; launch alone does not imply those outcomes.
- [Local checks](LOCAL_CHECKS.md) and the focused [V5 static contract fixture](test_child_allthree_terminal_verifier_v5_contract.py) cover the single insertion, identity contract, durable-reply guard, and unchanged payload pins.

This supplement is navigation only. It does not edit or replace the frozen body, binding, root payload check, reviews, or README.
