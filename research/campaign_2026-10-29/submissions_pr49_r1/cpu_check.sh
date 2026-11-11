#!/usr/bin/env bash
# PR51 receiver check via rootless podman (Docker daemon absent). Usage: cpu_check.sh <validate_candidate args>
set -euo pipefail
timeout 3600s podman run --rm -e PIN="${CPUSET:?}" --network none --cpus 1 --memory 16000000000 --memory-swap 16000000000 \
  --pids-limit 64 --cap-drop ALL --security-opt no-new-privileges --security-opt label=disable --read-only --userns keep-id \
  -v /home/vlad/development/wunder/research_tmp/pr51_wt:/work:ro -v /home/vlad/development/wunder/wnn_connectome_starterpack/datasets/valid.parquet:/inputs/valid.parquet:ro -v /home/vlad/development/wunder/wnn_connectome_starterpack/submissions/pr49_tl3c_checkpoints_20261029/extracted/shared/chronological_primary.json:/inputs/split.json:ro \
  -v /home/vlad/development/wunder/wnn_connectome_starterpack/submissions/tl3c_pr49_20261029:/out:rw -v /home/vlad/development/wunder/wnn_connectome_starterpack/submissions/tl3c_pr49_20261029/scratch:/tmp:rw tl3c-cpu-receiver \
  python -c 'import os,sys,runpy; os.sched_setaffinity(0,{int(os.environ["PIN"])}); sys.argv=sys.argv[1:]; sys.path.insert(0,os.path.dirname(sys.argv[0])); runpy.run_path(sys.argv[0],run_name="__main__")' /work/research/experiment_drafts/tl3c-submission-20261029/validate_candidate.py \
  --data /inputs/valid.parquet --data-sha256 2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782 --split /inputs/split.json "$@"
