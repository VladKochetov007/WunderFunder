#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack
OUT=$ROOT/research/campaign_2026-11-14/quality_r2/original_fbc_r1
exec timeout 3500s podman run --rm --name wnn-q14-original-fbc --network none --cpus 1 --memory 4000000000 --memory-swap 4000000000 --pids-limit 64 --cap-drop ALL --security-opt no-new-privileges --security-opt label=disable --read-only --userns keep-id \
-v /home/vlad/development/wunder/research_tmp/pr51_wt:/work:ro \
-v "$ROOT/datasets/valid.parquet:/inputs/valid.parquet:ro" \
-v "$ROOT/submissions/pr49_tl3c_checkpoints_20261029/extracted/shared/chronological_primary.json:/inputs/split.json:ro" \
-v "$OUT:/out:rw" \
-v /home/vlad/development/wunder/research_tmp/q14_original_fbc_scratch:/tmp:rw \
tl3c-cpu-receiver python /work/research/experiment_drafts/tl3c-submission-20261029/validate_candidate.py \
--zip /out/original_fbc_only.zip --receipt /out/receipt.json --data /inputs/valid.parquet \
--data-sha256 2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782 \
--split /inputs/split.json --output-dir /out/full_valid --level v3 --full-valid
