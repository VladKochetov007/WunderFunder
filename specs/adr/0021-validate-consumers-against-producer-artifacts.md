# ADR0021: validate consumers against actual producer artifacts

Status: accepted, 2026-10-08 Kyiv. Complements recipe and evidence audits; does not add a general framework or a new approval gate.

## Evidence

E540 V7 completed a finite FP32 control, then its verifier selected the epoch-counter mapping and omitted its sibling GradStats mapping. H232 V7B completed four epochs, then its verifier compared train row-group integers with valid row-group integers without source qualification. A pending H232 export packet would have rejected correct predictions because it used a tenfold incorrect data-byte count and treated array bytes as complete NPY file bytes. These were delivery defects, not negative scientific results. Original receipts and completed checkpoints remain preserved.

## Decision

Before admitting a new consumer or fixing one of these interfaces, exercise it on a small actual producer artifact when available. Bind the artifact hash and extraction path or JSON pointer. A synthetic fixture remains useful for adversarial mutations but cannot alone establish compatibility with the producer. For a new binary format, create a small file with the real production writer and verify header, payload, dtype, shape and completion behavior separately. Derive byte counts from shape and dtype; include format headers when checking file size.

Group identities are qualified by source before comparison. Component exclusion and ordered group fingerprints remain additional checks; accepting aliases from distinct sources does not relax actual overlap rejection. Match counter baselines to the same cgroup boundary/file and interpret timestamps with an explicit timezone. Append corrections to mutable monitoring ledgers rather than rewrite their original observations.

Keep fixes narrow and give corrected packets new identities. Preserve healthy running recipes. Reuse completed clean checkpoints for export-only recovery when source, ancestry, membership and checkpoint identity pass; refitting is not a default response to a failed postfit verifier. Continue scientific allocation separately from execution failure classification.

## Receipts

- [E540 actual producer fixture](../../tests/fixtures/E540_V7_ACTUAL_UPDATE_DIAGNOSTICS_ROW_0.origin.json) and [root V8 review](../../research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/E540_V8_ROOT_INDEPENDENT_SOURCE_REVIEW_20261008_R1.json): actual sibling mapping, strict finite FP32 update contract retained,17 focused checks.
- [H232 root export review](../../research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/H232_EP3_EXPORT_ROOT_INDEPENDENT_REVIEW_20261008_R1.json): exact own checkpoint/source/payload hashes, qualified membership and real small NPY regression,10 focused checks.
- [H218 root export review](../../research/campaign_2026-10-06/bandit_continuation/attention_training_H218/H218_FOLD0_EXPORT_ROOT_INDEPENDENT_REVIEW_20261008_R1.json): actual42-file source lock and468 excluded feature-only groups; checkpoint binding remains terminal-dependent.
