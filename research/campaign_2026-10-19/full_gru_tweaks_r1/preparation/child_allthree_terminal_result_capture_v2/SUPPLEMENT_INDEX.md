# AllThree result-capture V2 supplement

This index supplements the frozen [reader README](README.md), [reader template](RESULT_CAPTURE_TEMPLATE.py), and [source review](SOURCE_REVIEW_V2.json). The reader consumes durable admission and launch receipts; the root's four-key stdout envelope is a separate contract.

- [V2 result-capture source review](SOURCE_REVIEW_V2.json) is `PASS_CONDITIONAL_SOURCE_ONLY`. [AST literal-adaptation proof](AST_LITERAL_ADAPTATION_V2.json) records the narrow reader versioning.
- [V4 producer/consumer review](PRODUCER_PAIR_REVIEW_V4.json) is preserved as `REJECT_SOURCE_ONLY_NO_B_ATTEMPT`: V4 omitted `unit` from durable admission while its own reply guard required it.
- [V5 producer/consumer review](PRODUCER_PAIR_REVIEW_V5.json) is `PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR` and binds the one-key producer repair with the unchanged V2 reader contract.
- The rejected V4 candidate and repaired V5 candidate remain in their own [V4 supplement](../child_allthree_terminal_verifier_v4/SUPPLEMENT_INDEX.md) and [V5 supplement](../child_allthree_terminal_verifier_v5/SUPPLEMENT_INDEX.md).
- The actual V5 CPU support record is linked from [terminal verification V3](../../records/child_allthree_fit_v1/terminal_verification_v3/ACTUAL_INDEX.md). This index does not claim checker success, acceptance, or exact stop.

No historical source review, reader bytes, or receipt schema was rewritten to add this navigation file.
