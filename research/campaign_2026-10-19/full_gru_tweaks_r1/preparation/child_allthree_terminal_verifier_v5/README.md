# AllThree terminal verifier V5 candidate

This packet repairs one source-level mismatch found in the frozen V4 candidate. V4 wrote `VERIFIER_ADMISSION.json` without a `unit` field, then required `durable_admission.get("unit") == UNIT` before printing the outer reply. The independent V4 producer/consumer review records that rejection in `child_allthree_terminal_result_capture_v2/PRODUCER_PAIR_REVIEW_V4.json`; V4 and its review remain unchanged.

V5 is the exact V4 body with one dictionary entry, `"unit": UNIT,`, added to the producer admission object. The V2 support unit, private output directory, V2 checker, V1 receipt schemas, launch contents, outer four-key JSON reply, all 15 embedded payload pins, and the other V4 body bytes are preserved. [V5_BINDING.json](V5_BINDING.json) pins the V4 rejection and candidate, records the inverse-byte proof, and lists every unchanged payload pin.

The focused fixture parses source only. It derives producer admission and launch identities, compares those fields with the separately pinned V2 result-capture consumer, and evaluates the exact durable-reply guard expression with synthetic receipt metadata. It checks success for the exact unit and failure for missing/wrong units and altered supporting identities. It does not import or execute either verifier body, query B, or inspect training payloads.

This is a local candidate only. It has not been independently reviewed, staged, admitted, or executed. Root review and separate outer-consumer approval remain prerequisites for any later action. The failed verify-v1 invocation remains preserved and untouched.
