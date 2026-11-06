#!/usr/bin/env bash
set -euo pipefail

stage=${1:?stage: check, profile, fit, resume, evaluate, cache, cache-parity, bundle, scale-*, or student-*}
cell=${2:-}
packet=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$packet/../../.." && pwd)
packet_rel=research/experiment_drafts/transformer-sdpa-gate-20261106
cd -- "$repo"

: "${WNN_EXPECTED_COMMIT:?pin the reviewed full source commit SHA}"
: "${WNN_DATA_ROOT:?canonical source-qualified data root}"
: "${WNN_SPLIT_F0:?canonical F0 split manifest}"
: "${WNN_SPLIT_F2:?canonical F2 split manifest}"
: "${WNN_RUN_ROOT:?dedicated physical-disk output root}"
: "${CUDA_VISIBLE_DEVICES:?exact allocated GPU IDs or UUIDs}"
[[ "$WNN_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo 'WNN_EXPECTED_COMMIT must be a full lowercase Git SHA' >&2
  exit 2
}
[[ $(uname -s) == Linux ]] || { echo 'Receiving Linux GPU host only' >&2; exit 2; }
[[ $(git rev-parse --show-toplevel) == "$repo" ]] || { echo 'Unexpected repository root' >&2; exit 2; }
[[ $(git rev-parse HEAD) == "$WNN_EXPECTED_COMMIT" ]] || {
  echo 'Source commit mismatch' >&2
  exit 2
}
[[ -z $(git status --porcelain -- src scripts tests "$packet_rel" pyproject.toml uv.lock) ]] || {
  echo 'Runtime, tests, packet, or dependency files are dirty' >&2
  exit 2
}
[[ -d "$WNN_DATA_ROOT" && -s "$WNN_SPLIT_F0" && -s "$WNN_SPLIT_F2" ]] || {
  echo 'Data root and both canonical split manifests must exist' >&2
  exit 2
}
[[ "$WNN_RUN_ROOT" == /* ]] || { echo 'WNN_RUN_ROOT must be absolute' >&2; exit 2; }
case "$WNN_RUN_ROOT" in
  /tmp|/tmp/*|/dev/shm|/dev/shm/*)
    echo 'Use dedicated physical-disk storage, not temporary or RAM-backed storage' >&2
    exit 2
    ;;
esac
mkdir -p -- "$WNN_RUN_ROOT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
for device in "${devices[@]}"; do
  [[ -n "$device" ]] || { echo 'CUDA_VISIBLE_DEVICES contains an empty device' >&2; exit 2; }
done
[[ $(printf '%s\n' "${devices[@]}" | sort -u | wc -l | tr -d ' ') == ${#devices[@]} ]] || {
  echo 'CUDA_VISIBLE_DEVICES contains a duplicate device' >&2
  exit 2
}

require_devices() {
  local count=$1
  [[ ${#devices[@]} == "$count" ]] || {
    echo "This stage requires exactly $count allocated visible GPU(s)" >&2
    exit 2
  }
}

require_absent() {
  [[ ! -e "$1" ]] || { echo "Refusing to reuse existing output: $1" >&2; exit 2; }
}

split_for_cell() {
  case "$1" in
    *_f0) split=$WNN_SPLIT_F0 ;;
    *_f2) split=$WNN_SPLIT_F2 ;;
    *) echo "Unknown fold in cell: $1" >&2; exit 2 ;;
  esac
}

check_receiver() {
  require_devices 8
  local output="$WNN_RUN_ROOT/acceptance"
  local gate_ddp="$WNN_RUN_ROOT/gate_ddp.json"
  require_absent "$output"
  require_absent "$gate_ddp"
  mkdir -- "$output"
  nvidia-smi -q > "$output/nvidia-smi.txt"
  nvidia-smi topo -m > "$output/topology.txt"
  env -u CUBLAS_EMULATE_SINGLE_PRECISION uv run --no-sync python -m pytest -q \
    tests/test_sequence_sdpa_gate.py \
    tests/test_sequence_attention_backends.py \
    tests/test_b300_sequence_models.py \
    tests/test_b300_sequence_contracts.py \
    tests/test_sequence_muon.py --junitxml="$output/pytest.xml" \
    2>&1 | tee "$output/pytest.log"
  env -u CUBLAS_EMULATE_SINGLE_PRECISION uv run --no-sync torchrun \
    --standalone --nnodes=1 --nproc-per-node=2 \
    scripts/check_sequence_ddp.py --output-receipt "$output/sequence-ddp.json" \
    2>&1 | tee "$output/sequence-ddp.log"
  env -u CUBLAS_EMULATE_SINGLE_PRECISION uv run --no-sync torchrun \
    --standalone --nnodes=1 --nproc-per-node=2 \
    scripts/check_sequence_muon_ddp.py --attention-gate headwise \
    --output-receipt "$gate_ddp" \
    2>&1 | tee "$output/gate-muon-ddp.log"
}

source "$packet/teacher_stages.sh"
source "$packet/student_stages.sh"

case "$stage" in
  check)
    [[ -z "$cell" ]] || { echo 'check takes no cell argument' >&2; exit 2; }
    export PYTHONPATH="$repo/src"
    check_receiver
    ;;
  profile|fit|resume|evaluate|cache|cache-parity|bundle)
    [[ -n "$cell" ]] || { echo "$stage requires a teacher cell" >&2; exit 2; }
    export PYTHONPATH="$repo/src"
    teacher_stage initial "$stage" "$cell"
    ;;
  scale-profile|scale-fit|scale-resume|scale-evaluate|scale-cache|scale-cache-parity|scale-bundle)
    [[ -n "$cell" ]] || { echo "$stage requires a scale cell" >&2; exit 2; }
    export PYTHONPATH="$repo/src"
    teacher_stage scale "${stage#scale-}" "$cell"
    ;;
  student-profile|student-fit|student-resume|student-evaluate|student-bundle)
    [[ -n "$cell" ]] || { echo "$stage requires a student cell" >&2; exit 2; }
    export PYTHONPATH="$repo/src"
    student_stage "${stage#student-}" "$cell"
    ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
