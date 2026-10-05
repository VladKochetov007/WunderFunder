#!/usr/bin/env bash
set -euo pipefail
cd /home/vlad/development/wunder/wnn_connectome_starterpack
export OPENCODE_CONFIG_CONTENT='{"permission":{"edit":"deny","bash":"deny","task":"deny","external_directory":"deny","read":"allow","glob":"allow","grep":"allow","webfetch":"allow"}}'
exec /home/vlad/.opencode/bin/opencode run --pure --agent plan \
  --model opencode/ling-3.1-flash-free --variant high --format json \
  --dir research/campaign_2026-10-05/external_reviews/context_packet \
  --title "Connectome blind method ideator: Ling high" \
  < research/campaign_2026-10-05/external_reviews/opencode_ideator_prompt.txt
