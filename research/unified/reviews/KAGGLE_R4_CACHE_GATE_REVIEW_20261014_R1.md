---
kind: review
status: complete
created_utc: 2026-10-14T00:48:37Z
updated_utc: 2026-10-14T00:48:37Z
hypothesis_ids: [H-STUDENT-LADDER-20261014]
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-14/cpu_student_frontier_r1/kaggle_profile_r4/README.md
  - research/campaign_2026-10-14/cpu_student_frontier_r1/kaggle_profile_r4/ACTUAL_PRODUCER_GATE_CHECK_R4.json
---

# Independent review of the R4 Kaggle cache identity gate

## Question or claim

Does R4 repair the R3 regular-file rejection while retaining exact cache identity checks and preserving the frozen preflight and scientific job specifications?

## Mechanism and method

I inspected the sealed R4 packet, candidate specifications, actual-producer gate harness and receipts, plus the shipped runner's identity-check implementation. The packet is bound to archive SHA-256 `5e7191304e851d3ec9f66d8d995c8d7cb94b7e5b06ea5ec0dde926e99afc5fa9` and bootstrap SHA-256 `f1a2e579b756e6583d409dc2c84febc9c17a5e8a3e6d5e3f4bf6863cc197fd8a`. The current bootstrap and runner source hashes match their bundle descriptors.

The harness compares each candidate specification to R3 after removing only the deferred-input field, verifies that exactly the three materialized cache paths are removed from that list, and byte-compares the preflight specification. The runner then verifies non-deferred input descriptors using streaming SHA-256 reads in 8 MiB blocks; remaining deferred paths retain their symlink and size gate.

## Evidence and interpretation

The actual-producer receipt reproduces the old R3 rejection and passes the corrected gate for all three candidate specifications. It records exact matches for the 1,685,600,128-byte train cache, 224,800,128-byte valid cache, and 323,404-byte metadata file, including their expected full-file hashes. Every worker `Popen` is intercepted after the identity gates, so no worker starts. The inventory is synthetic only to reach those file checks.

This is a bounded PASS for offline file-gate compatibility and source/spec preservation. It supports root's separate fresh-quota and dispatch review; it does not establish Kaggle runtime, GPU resource use, profile completion, fit behavior, or scientific quality.

## Limits and validity

The gate receipt uses actual immutable producer bytes, but mocked hardware inventory and intentionally intercepted subprocess creation. I did not perform provider actions or rerun the large actual-file hashing harness. The campaign cache/teacher provenance remains the frozen producer lineage; this review changes no target or training semantics.

## Decision and next step

R4 is technically ready for root's fresh resource/quota admission and private dispatch decision. This review is not a provider-write authorization. Preserve the R3 terminal failure as execution evidence and label any R4 outcome separately.

## Source records

- [R4 packet and actual-producer gate receipt](../../campaign_2026-10-14/cpu_student_frontier_r1/kaggle_profile_r4/README.md)
- [Machine gate-check receipt](../../campaign_2026-10-14/cpu_student_frontier_r1/kaggle_profile_r4/ACTUAL_PRODUCER_GATE_CHECK_R4.json)
- [Gate-check harness](../../campaign_2026-10-14/cpu_student_frontier_r1/kaggle_profile_r4/check_runner_gate.py)
- [Runner streaming identity verification](../../../scripts/kaggle_common.py)
