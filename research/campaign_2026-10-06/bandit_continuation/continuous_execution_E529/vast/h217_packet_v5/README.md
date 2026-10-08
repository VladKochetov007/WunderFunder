# H217 packet V5 — full-feature TCN capacity ladder

Status: `pending_root_review`. This packet does not authorize a provider create, push, profile, fit, export, or grade.

Two scratch arms share seed 7, sealed fold0, and the default causal 211-wide feature path (`--raw-only` omitted):

| Arm | Hidden | Layers | Mix |
|---|---:|---:|---:|
| `tcn_full128` | 128 | 12 | 1 |
| `tcn_full256` | 256 | 12 | 1 |

The sealed project tree is a byte copy of `h217_packet_v4/project`. Trainer and model files are unchanged. The historical raw-only TCN128x12 delivery is recorded as an ungraded reference and is not a paired factor. All-data training is outside this packet.

## Exact commands

Build, from the repository root:

```sh
python3 research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/vast/build_h217_v5_packet.py --output research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/vast/h217_packet_v5.tar
```

Focused tests, from the repository root:

```sh
python3 -m pytest -q research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/vast/test_h217_v5_worker.py
```

## Pending root review

`fit_authorized`, `provider_create_authorized`, `push_authorized`, `grade_authorized`, and `profile_authorized` are false.

Before any create or fit, root verifies the byte-identical V4 project, the two-TCN full-feature recipe, startup source and input hashes, fold0 membership (11940 fitted groups, 468 excluded), the strict 230-update profile gate, and phase caps of 900, 7200, and 1800 seconds inside the 21600-second cohort. Each fit is admitted from that arm's measured profile. The historical raw-only fit of 823.97 seconds does not admit either arm.
