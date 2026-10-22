---
kind: idea
status: proposed
hypothesis_ids: [H-PER-FIELD-AGE-20261022-R1]
experiment_ids: []
---

# Per-field age as a causal representation

Distinguish identical unchanged values by the number of rows since their last change, resetting each field at sequence boundaries. This is an unresolved representation, with earlier implementation/proposal work in the [event-state reframe](EVENT_STATE_AND_METRIC_OBJECTIVE_REFRAME_20261015_R1.md); it is not a new result or a failed mechanism inferred from older pooled clock proxies.

[GRU-D](https://arxiv.org/abs/1606.01865) motivates using observation timing. Here unchanged values do not establish missingness, and anonymous values do not establish a physical clock. The useful test would compare target/complete-endpoint contribution against the same parent, not merely demonstrate that ages vary. Preserve source-qualified component membership and causal reset.

The [online amendment](ONLINE_RELATIONSHIPS_20261022_R1.md) retains this separate card but authorizes **zero fits** for it. Status: untested under this amendment; no gain or additional allocation claimed.
