# Full original archive replays

[Contract](../CONTRACT.json), [admission](../FULL_REPLAY_ADMISSION_R1.json), [frozen worker](../score_archive.py) and [controller](../run_replays.py) define eight complete1,873-sequence diagnostic runs. Four one-CPU workers run concurrently. Seven use original callbacks; one uses the same pair_ftt2 weights through Rust. Terminal `RESULT.json` and `*_ATTEMPT.json` appear after completion; progress logs do not certify completion. Large FP32 predictions remain ignored here.

The [notebook](../team_model_comparison.py) displays saved terminal evidence. Final models have fitted or unknown validation overlap: runs establish exact behavior, conversion differences and diagnostic diversity, not clean CV or public forecasts. Full-file/group hashes, official masks and moments accompany grades. Root owns collection and comparative analysis.
