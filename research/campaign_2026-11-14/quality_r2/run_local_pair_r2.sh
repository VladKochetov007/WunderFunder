#!/usr/bin/env bash
# Two declared matched local fits, serially; future final refits have a separate owner.
set -euo pipefail
O=/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-11-14/quality_r2
test ! -e "$O/STOP"
/bin/bash "$O/run_strict_fit.sh" q14s_local_kd90_s3 3 .9
test ! -e "$O/STOP"
/bin/bash "$O/run_strict_fit.sh" q14s_local_kd100_s3 3 1.0
