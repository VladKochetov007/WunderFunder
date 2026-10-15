---
kind: result
status: complete
created_utc: 2026-10-15T13:56:33Z
updated_utc: 2026-10-15T13:56:33Z
hypothesis_ids: [H-087, H-138]
experiment_ids: [E-PER-FIELD-CLOCK-RESOLUTION-F0-20261015-R1]
evidence_paths: [PER_FIELD_CLOCK_REDUNDANCY_SCREEN_R1.json, per_field_clock_probe_r1.py, CPU_TIMING_FIXTURE_R1.json]
---

# Per-field clock input-resolution screen

## Question or claim

Do exact per-field change ages add deterministic clock resolution beyond seven raw-field-group ages and the three clocks in the production 211-column input? This screen tests representation redundancy only; it cannot establish prediction quality.

## Mechanism and method

Using the existing raw-feature transform and actual `_flags`-based `StreamingEventFeatures`, the script streamed each of eight complete, distinct-component F0 fitting sequences. It reset each field clock to zero on raw inequality and otherwise incremented it to a 10,000-row cap. Group clocks reset when any member in the corresponding seven `nn.token_groups(112)` raw ranges changed. It compared field-reset patterns with existing `since` clocks at offsets 74–76 and checked batch/stream and sequence-reset parity.

The script verified the fixture manifest SHA-256 `941b656d8408be6f5c8f5816c1e32f3c8b24e59040bb4ed84bb895078a9f9f05`, then verified `features.bin` size 71,680,000 and SHA-256 `e8e8e1268cea54be12c3cf93e1be79b18cb097a1584851bcc4f5dc65885dd337` before mapping it read-only. It read no request-vector payload, targets, scoring mask, teachers, model weights, parquet or predictions. The selected component, sequence, source and row-group provenance is preserved per slot in the [JSON receipt](PER_FIELD_CLOCK_REDUNDANCY_SCREEN_R1.json).

## Evidence and interpretation

All eight components had 19–48 distinct per-field reset patterns, compared with seven distinct group clocks. Individual ages differed from their group's age on 7.3%–26.7% of field-row cells, depending on group. Four fields matched an existing trade-clock reset pattern across all eight sequences; no field matched either other existing clock across all eight. There were no constant fields; fields 106 and 107 changed on every compared row. The median field had 919 distinct age values (maximum 2,348). No age reached the cap. Only 0.368% of field-rows had age equal to sequence position. Existing event-feature batch/stream and reset parity passed for all eight sequences.

The result confirms extra deterministic input resolution on this fitting panel. A per-field age is a deterministic encoding of observable history, not new causal information. It has no target, mask, prediction or metric evidence and must not be described as predictive value.

## Limits and validity

This was a features-only read of eight exposed fold-0 fitting components selected for the CPU fixture, not CV or held-out evidence. The panel establishes neither general-population prevalence nor score gain. The age cap did not activate, so its value was not tested. The bounded one-CPU/500MB/no-swap command took 16.95s and peaked at 95,277,056 bytes; its task-cgroup counters report zero OOM events, and `torch` was not imported. At launch, the aggregate gate estimated 7.68GB conservative committed memory (parent current plus unused active-worker reservation plus this 500MB bound), below the17GB target, and 2.1/5 reserved CPU cores. `nn.py` and `features.py` source hashes, the exact script hash, and cgroup measurements are in the JSON receipt.

## Decision and next step

The redundancy falsifier did not trigger: per-field and group clocks are not equivalent on this panel. Keep the per-field-age child eligible, without promoting it. The next quality discriminator is a same-parent, same-budget endpoint comparison with G (group ages) versus F (individual ages), using F−G as the primary equal-width contrast and the existing 211-input endpoint as a secondary reference. Grade paired source-qualified components with unchanged `global_wp_clip2_v1`; retain uncertainty and fitting-fold exposure. This result alone does not authorize or perform a fit.

## Source records

- [Frozen fixture manifest](CPU_TIMING_FIXTURE_R1.json) and ignored local payload `artifacts/vast_a_cpu_probe_r1/features.bin`; provenance binds fold-0 train row groups, eight distinct components and the source/footer identity.
- [Runnable screen](per_field_clock_probe_r1.py), [measured JSON receipt](PER_FIELD_CLOCK_REDUNDANCY_SCREEN_R1.json), [`StreamingEventFeatures`](../../../src/wnn/features.py), and raw-group ranges in [`token_groups`](../../../src/wnn/nn.py).
- E-415's earlier [`event_age` proxy report](../../campaign_2026-10-04/E415_conditional_proxy_report.md) is the relevant prior attempt: it fit four observable event-age cues in fixed frozen-state residual heads on 6,158 sampled rows from exposed fold 0; the event-age candidate scored `−0.030966` versus the untouched GRU, despite `+0.006945` versus the equal-width capacity arm. It did not test one clock per raw field or an end-to-end input transform, so its negative does not test this representation child.
