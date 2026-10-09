#!/usr/bin/env bash
set -euo pipefail
sha256sum --check --status research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/E520_PROFILE_V6_PUSH_SHA256SUMS_20261007_R1.txt
exec timeout --signal=TERM --kill-after=5s 100s kaggle kernels push -p research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/packet_v6 -t 1800
