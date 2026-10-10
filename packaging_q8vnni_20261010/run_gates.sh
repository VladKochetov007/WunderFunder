#!/usr/bin/env bash
# Offline python3.11 gates for both packages; one pinned core, 1 GB, no network, non-root, repo read-only.
# Run from the worktree root.
set -euo pipefail
W=$(pwd)
MAIN=/home/vlad/development/wunder/wnn_connectome_starterpack
OUT=$W/packaging_q8vnni_20261010/gates
mkdir -p "$OUT"
run() { # pkg level
  docker run --rm --network none --cpuset-cpus 4 --cpus 1 --memory 1g --memory-swap 1g --pids-limit 64 --user 1000:1000 \
    -e CUDA_VISIBLE_DEVICES= \
    -v "$MAIN":/workspace:ro -v "$W":/wt:ro -v "$OUT":/out \
    wnn-python311-submission-check python /wt/packaging_q8vnni_20261010/gate_replay.py \
    --zip /wt/packaging_q8vnni_20261010/packages/pkg_$1/submission.zip --level $2 \
    --split /workspace/research/campaign_2026-09-29/artifacts/splits/fold1.json --data /workspace/datasets/valid.parquet \
    --sequences 30 --out /out/$1_$2.json > "$OUT/$1_$2.log" 2>&1
}
run i16 v3
run q8vnni v3
run q8vnni avx2
run q8vnni v2
run i16 avx2
run i16 v2
