# Export-admission packet supplement index

This index is a navigation layer for the local V1 export-admission packet and its root-owned capture artifacts. It does not establish a successful export admission or authorize a new attempt. The actual V2 preflight failure and its single read-only outcome capture are preserved under [the records index](../../records/first_cell_evaluation_v1/README.md).

- [Packet overview](README.md) describes the bounded export contract and prerequisite gates.
- [Admission body template](ADMISSION_BODY_TEMPLATE_V1.py), [remote submission template](REMOTE_SUBMISSION_TEMPLATE.py), [renderer](render_admission_remote_v1.py), [source binding](ADMISSION_SOURCE_BINDING_V1.json), [remote-submission review](REMOTE_SUBMISSION_SOURCE_REVIEW_V1.json), [local checks](LOCAL_CHECKS.md), and [admission fixture](test_admission_packet_v1.py) cover the admission-side packet and its local contract.
- The [root-capture index](ROOT_CAPTURE_INDEX.md) maps the versioned reader, acceptance and stop templates, root capture drivers, orchestration plan, reviews, proofs, tests, and helper-extraction provenance.
