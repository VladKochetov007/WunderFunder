#!/usr/bin/env bash
set -euo pipefail

# One explicit stage of one cell. No queue, background process, or automatic fit.
stage=${1:?stage: check, profile-one, profile-eight, profile-peak, admit, fit, resume, evaluate, cache, bundle}
cell=${2:?cell: t1280_f0, t1792_f0, t1280_f2, t1792_f2}
case "$cell" in t1280_f0|t1792_f0|t1280_f2|t1792_f2) ;; *) exit 2 ;; esac
packet=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$packet/../../../.."
: "${WNN_EXPECTED_COMMIT:?pin the reviewed PR head SHA}"
: "${WNN_RUN_ROOT:?dedicated physical disk run root}"
: "${WNN_DATA_ROOT:?canonical data root}"
: "${CUDA_VISIBLE_DEVICES:?the exact eight allocated GPU IDs or UUIDs}"
IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#devices[@]} == 8 ]] || { echo 'Bind exactly eight allocated GPUs' >&2; exit 2; }
[[ $(uname -s) == Linux ]] || { echo 'Receiving Linux GPU host only' >&2; exit 2; }
[[ $(git rev-parse HEAD) == "$WNN_EXPECTED_COMMIT" ]] || { echo 'Source commit mismatch' >&2; exit 2; }
[[ -z $(git status --porcelain -- src scripts tests "$packet" pyproject.toml uv.lock) ]] || {
  echo 'Runtime, checks, packet or dependency files are dirty' >&2; exit 2;
}
case "$cell" in
  *_f0) split=${WNN_SPLIT_F0:?canonical F0 manifest} ;;
  *_f2) split=${WNN_SPLIT_F2:?canonical F2 manifest} ;;
esac
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
run="$WNN_RUN_ROOT/$cell"
config="$packet/configs/$cell.json"
mkdir -p -- "$run"
train=(scripts/train_sequence.py --config "$config" --data-root "$WNN_DATA_ROOT" --split-manifest "$split")
predict=(scripts/predict_sequence.py --checkpoint "$run/fit/model.pt" --data-root "$WNN_DATA_ROOT" --split-manifest "$split")
eight=(uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=8)
case "$stage" in
  check)
    checks="$WNN_RUN_ROOT/acceptance"
    mkdir -- "$checks"
    nvidia-smi -q > "$checks/nvidia-smi.txt"
    nvidia-smi topo -m > "$checks/topology.txt"
    PYTEST_ADDOPTS='' uv run --no-sync python -m pytest -q \
      tests/test_b300_sequence_models.py tests/test_b300_sequence_contracts.py \
      tests/test_large_teacher_runtime_guards.py tests/test_sequence_full_fit_data.py \
      tests/test_sequence_recovery_cuda.py tests/test_sequence_muon.py --junitxml="$checks/pytest.xml" \
      2>&1 | tee "$checks/pytest.log"
    uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=2 \
      scripts/check_sequence_ddp.py --output-receipt "$checks/ddp.json" \
      2>&1 | tee "$checks/ddp.log"
    uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=2 \
      scripts/check_sequence_muon_ddp.py --output-receipt "$checks/muon-ddp.json" \
      2>&1 | tee "$checks/muon-ddp.log"
    uv run --no-sync python "$packet/admission.py" checks --directory "$checks"
    ;;
  profile-one)
    uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
      scripts/train_sequence.py --config "$packet/configs/${cell}_profile_one.json" \
      --data-root "$WNN_DATA_ROOT" --split-manifest "$split" \
      --output "$run/profile-one" --profile-updates 12
    ;;
  profile-eight)
    [[ -f "$run/profile-one/profile_receipt.json" ]]
    "${eight[@]}" "${train[@]}" --output "$run/profile-eight" --profile-updates 12
    ;;
  profile-peak)
    [[ -f "$run/profile-eight/profile_receipt.json" ]]
    "${eight[@]}" scripts/train_sequence.py --config "$packet/configs/${cell}_profile_peak.json" \
      --data-root "$WNN_DATA_ROOT" --split-manifest "$split" \
      --output "$run/profile-peak" --profile-updates 12
    ;;
  admit)
    : "${WNN_REVIEWER:?name of receiving operator reviewing profiles}"
    : "${WNN_HOST_RAM_BUDGET_GIB:?actual allocated host RAM, including all eight ranks}"
    uv run --no-sync python "$packet/admission.py" profile --directory "$run" \
      --checks "$WNN_RUN_ROOT/acceptance/acceptance.json" --config "$config" \
      --reviewer "$WNN_REVIEWER" --host-ram-gib "$WNN_HOST_RAM_BUDGET_GIB"
    ;;
  fit)
    "${eight[@]}" "${train[@]}" --output "$run/fit" --profile-receipt "$run/approved_profile.json"
    ;;
  resume)
    "${eight[@]}" "${train[@]}" --output "$run/fit" --profile-receipt "$run/approved_profile.json" \
      --resume "$run/fit/training_state.pt"
    ;;
  evaluate)
    "${eight[@]}" "${predict[@]}" --membership evaluation --output "$run/evaluation"
    uv run --no-sync python scripts/score_sequence.py --data-root "$WNN_DATA_ROOT" \
      --split-manifest "$split" --predictions "$run/evaluation" --output "$run/fit/score.json"
    ;;
  cache)
    "${eight[@]}" "${predict[@]}" --membership fit --output "$run/teacher-cache"
    ;;
  bundle)
    args=()
    for rank in {0..7}; do args+=(--evidence "rank$rank.jsonl"); done
    uv run --no-sync python scripts/research_result_bundle.py --root "$run/fit" \
      --output-dir "$run/bundle" --evidence run_metadata.json --evidence metrics.jsonl \
      --evidence terminal_status.json --evidence score.json "${args[@]}" \
      --weight model.pt --weight training_state.pt
    ;;
  *) echo 'Unknown stage' >&2; exit 2 ;;
esac
