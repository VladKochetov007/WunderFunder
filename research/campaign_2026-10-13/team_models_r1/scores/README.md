# Original archive replay results

Four complete original callbacks are now available. All use1,873 supplied-valid sequences and the official metric; final-fit/unknown overlap prevents clean-CV or public-transfer claims.

| Endpoint | Diagnostic WP |
|---|---:|
| [C_original_r1](C_original_r1/README.md) | 0.712560467 |
| [E_original_r1](E_original_r1/README.md) | 0.712065318 |
| [pair_ftt2_original_r1](pair_ftt2_original_r1/README.md) | 0.712026560 |
| [xteam_h4_original_r1](xteam_h4_original_r1/README.md) | 0.712654817 |

Original xteam_h4 exceeds E by 0.000589499 and pair_ftt2 by 0.000628257 on these diagnostic final-weight predictions. These differences do not predict public gains. The new seed6 release is a distinct endpoint.

Still running at capture: [e6a_original_r1](e6a_original_r1/README.md), [p_c3zero_original_r1](p_c3zero_original_r1/README.md), [p_c3only_original_r1](p_c3only_original_r1/README.md), [pair_ftt2_rust_r1](pair_ftt2_rust_r1/README.md).

[Immutable runtime snapshot](../runtime_snapshot_r1/README.md) retains all current logs/progress. Live operational output is ignored so jobs can continue without dirtying Git. Completed pending results and controller terminal receipts must be collected into versioned evidence; ignoring their staging location does not discard them. [Frozen worker](../score_archive.py), [controller](../run_replays.py), [contract](../CONTRACT.json) and [admission](../FULL_REPLAY_ADMISSION_R1.json) remain unchanged.

[Complete FP32 fourteen-member/nine-mixture diagnostic grade](member_bank_fp32_r1/README.md) uses a different arithmetic endpoint and is recorded separately.
