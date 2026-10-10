#!/usr/bin/env bash
# Gates for the upload_2026-10-10b candidates plus i16 references. Run from the worktree root.
set -uo pipefail
W=$(pwd)
MAIN=/home/vlad/development/wunder/wnn_connectome_starterpack
OUT=$W/packaging_q8vnni_20261010/gates_upload
mkdir -p "$OUT"
run() { # tag zip(path inside container) level
  docker run --rm --network none --cpuset-cpus 4 --cpus 1 --memory 1g --memory-swap 1g --pids-limit 64 --user 1000:1000 \
    -e CUDA_VISIBLE_DEVICES= \
    -v "$MAIN":/workspace:ro -v "$W":/wt:ro -v "$OUT":/out \
    wnn-python311-submission-check python /wt/packaging_q8vnni_20261010/gate_replay.py \
    --zip "$2" --level "$3" \
    --split /workspace/research/campaign_2026-09-29/artifacts/splits/fold1.json --data /workspace/datasets/valid.parquet \
    --sequences 30 --out /out/$1_$3.json > "$OUT/$1_$3.log" 2>&1
}
U=/wt/submissions/upload_2026-10-10b
run z5_i16 $U/z5_i16_single_control.zip v3
run z5_q8vnni $U/z5_q8vnni_single/submission.zip v3
run pair_i16 /workspace/submissions/c20261008_h230_all_data_pair_i16.zip v3
run pair_q8vnni $U/z5_z4_pair_q8vnni/submission.zip v3
run z5_q8vnni $U/z5_q8vnni_single/submission.zip avx2
run z5_q8vnni $U/z5_q8vnni_single/submission.zip v2
run pair_q8vnni $U/z5_z4_pair_q8vnni/submission.zip avx2
run pair_q8vnni $U/z5_z4_pair_q8vnni/submission.zip v2
run z5_i16 $U/z5_i16_single_control.zip avx2
run pair_i16 /workspace/submissions/c20261008_h230_all_data_pair_i16.zip avx2
echo done
