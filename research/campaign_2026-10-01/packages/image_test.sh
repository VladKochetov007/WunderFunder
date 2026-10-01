#!/usr/bin/env bash
# Offline scorer-image test: python:3.11-slim-bookworm, no network, 1 CPU, 16 GiB, read-only inputs.
#   image_test.sh ZIP OUT_NPZ
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
zip=$(realpath "$1"); out=$(realpath -m "$2"); mkdir -p "$(dirname "$out")"
docker run --rm --network none --cpus 1 --memory 16g \
  -v "$root/research/campaign_2026-09-30/runtime/python311-site:/deps:ro" \
  -v "$root/scripts:/repo/scripts:ro" -v "$root/utils.py:/repo/utils.py:ro" \
  -v "$root/cache/valid_subset20.parquet:/data/valid_subset20.parquet:ro" \
  -v "$zip:/sub/submission.zip:ro" -v "$(dirname "$out"):/out" \
  -e PYTHONPATH=/deps -e PYTHONDONTWRITEBYTECODE=1 python:3.11-slim-bookworm \
  bash -c "python --version && python /repo/scripts/verify_submission.py /sub/submission.zip /data/valid_subset20.parquet /out/$(basename "$out") && chown $(id -u):$(id -g) /out/$(basename "$out")"
