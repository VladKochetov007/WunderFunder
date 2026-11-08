#!/usr/bin/env bash
set -euo pipefail

action=${1:?choose data-check, reference-export/import, check, profile-*, fit, resume, evaluate, compare, bundle}
cell=${2:-}
packet=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$packet/../../.."
[[ $(uname -s) == Linux ]] || { echo 'Receiver Linux host only' >&2; exit 2; }
[[ $(git rev-parse --show-toplevel) == "$PWD" ]] || exit 2
: "${WNN_EXPECTED_COMMIT:?pin the reviewed full commit SHA}"
: "${WNN_DATA_ROOT:?bind the canonical PR49 data/cache/run root}"
: "${WNN_SPLIT:?bind the exact frozen PR49 chronological primary manifest}"
: "${WNN_RUN_ROOT:?use a fresh persistent physical-disk output root}"
: "${WNN_REFERENCE_ROOT:?bind a new reference-cache directory}"
[[ $(git rev-parse HEAD) == "$WNN_EXPECTED_COMMIT" ]] || { echo 'Source pin mismatch' >&2; exit 2; }
[[ -z $(git status --porcelain -- src scripts tests "$packet" pyproject.toml uv.lock) ]] || {
  echo 'Runtime/packet/checks/dependencies are dirty' >&2; exit 2;
}
[[ "$WNN_RUN_ROOT" == /* && "$WNN_REFERENCE_ROOT" == /* ]] || exit 2
case "$WNN_RUN_ROOT/" in /tmp/*|/dev/shm/*) echo 'Persistent disk required' >&2; exit 2 ;; esac
case "$WNN_REFERENCE_ROOT/" in /tmp/*|/dev/shm/*) echo 'Persistent disk required' >&2; exit 2 ;; esac
mkdir -p -- "$WNN_RUN_ROOT"
case "$(findmnt --noheadings --output FSTYPE --target "$WNN_RUN_ROOT")" in
  tmpfs|ramfs) echo 'Output root must be on physical disk' >&2; exit 2 ;;
esac
export TMPDIR="$WNN_RUN_ROOT/scratch"
export TORCHINDUCTOR_CACHE_DIR="$WNN_RUN_ROOT/compiler/inductor"
export TRITON_CACHE_DIR="$WNN_RUN_ROOT/compiler/triton"
mkdir -p -- "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUBLAS_EMULATE_SINGLE_PRECISION=1

case "$action" in
  data-check)
    mkdir -p -- "$WNN_RUN_ROOT"
    findmnt --target "$WNN_RUN_ROOT" > "$WNN_RUN_ROOT/output-mount.txt"
    df -hT "$WNN_RUN_ROOT" > "$WNN_RUN_ROOT/output-capacity.txt"
    uv run --no-sync python "$packet/receiver_data.py" --data-root "$WNN_DATA_ROOT" \
      --split-manifest "$WNN_SPLIT" --output "$WNN_RUN_ROOT/data-check.json"
    exit 0 ;;
  reference-export)
    : "${WNN_PR49_SOURCE:?clean exact c42baf7b PR49 source checkout}"
    : "${CUDA_VISIBLE_DEVICES:?bind exactly one allocated GPU for native replay}"
    [[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || { echo 'Bind one replay GPU' >&2; exit 2; }
    source=${3:?train or valid}
    env -u CUBLAS_EMULATE_SINGLE_PRECISION uv run --no-sync python "$packet/reference_export.py" \
      --legacy-source-root "$WNN_PR49_SOURCE" --data-root "$WNN_DATA_ROOT" \
      --split-manifest "$WNN_SPLIT" --member "$cell" --source "$source"
    exit 0 ;;
  reference-import)
    uv run --no-sync python "$packet/reference_import.py" --data-root "$WNN_DATA_ROOT" \
      --split-manifest "$WNN_SPLIT" --output-root "$WNN_REFERENCE_ROOT"
    exit 0 ;;
  baseline-score)
    uv run --no-sync python "$packet/reference_score.py" --data-root "$WNN_DATA_ROOT" \
      --split-manifest "$WNN_SPLIT" --reference-cache "$WNN_REFERENCE_ROOT/evaluation" \
      --output "$WNN_RUN_ROOT/baseline-score.json"
    exit 0 ;;
  compare)
    uv run --no-sync python "$packet/compare_panel.py" --run-root "$WNN_RUN_ROOT" \
      --output "$WNN_RUN_ROOT/comparison.json"
    exit 0 ;;
esac

: "${CUDA_VISIBLE_DEVICES:?bind exactly eight allocated B300 IDs}"
IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#devices[@]} == 8 ]] || { echo 'Eight allocated B300s required' >&2; exit 2; }
if [[ "$action" == check ]]; then
  checks="$WNN_RUN_ROOT/acceptance"
  [[ -f "$WNN_RUN_ROOT/data-check.json" && ! -e "$checks" ]] || exit 2
  mkdir -- "$checks"
  nvidia-smi -q > "$checks/nvidia-smi.txt"
  nvidia-smi topo -m > "$checks/topology.txt"
  env -u CUBLAS_EMULATE_SINGLE_PRECISION PYTEST_ADDOPTS='' uv run --no-sync python -m pytest -q \
    tests/test_sequence_blend_objective.py tests/test_sequence_reference_cache.py \
    tests/test_chronological_primary_splits.py tests/test_b300_sequence_models.py \
    tests/test_b300_sequence_contracts.py tests/test_sequence_full_fit_data.py \
    tests/test_sequence_recovery_cuda.py tests/test_sequence_muon.py \
    --junitxml="$checks/pytest.xml" 2>&1 | tee "$checks/pytest.log"
  gpu_test=tests/test_sequence_blend_gpu.py::test_compiled_composite_objective_matches_eager_finite_outputs_and_gradients
  CUBLAS_EMULATE_SINGLE_PRECISION=0 PYTEST_ADDOPTS='' uv run --no-sync python -m pytest -q \
    "$gpu_test[bf16]" --junitxml="$checks/bf16.xml" 2>&1 | tee "$checks/bf16.log"
  CUBLAS_EMULATE_SINGLE_PRECISION=1 PYTEST_ADDOPTS='' uv run --no-sync python -m pytest -q \
    "$gpu_test[bf16_bf16x9]" --junitxml="$checks/bf16x9.xml" 2>&1 | tee "$checks/bf16x9.log"
  env -u CUBLAS_EMULATE_SINGLE_PRECISION uv run --no-sync torchrun --standalone --nnodes=1 \
    --nproc-per-node=2 scripts/check_sequence_ddp.py --output-receipt "$checks/ddp.json" \
    2>&1 | tee "$checks/ddp.log"
  env -u CUBLAS_EMULATE_SINGLE_PRECISION uv run --no-sync torchrun --standalone --nnodes=1 \
    --nproc-per-node=2 scripts/check_sequence_ddp.py --blend-objective-alpha 0.1 \
    --output-receipt "$checks/complement-ddp.json" 2>&1 | tee "$checks/complement-ddp.log"
  env -u CUBLAS_EMULATE_SINGLE_PRECISION uv run --no-sync torchrun --standalone --nnodes=1 \
    --nproc-per-node=2 scripts/check_sequence_muon_ddp.py --output-receipt "$checks/muon-ddp.json" \
    2>&1 | tee "$checks/muon-ddp.log"
  uv run --no-sync python "$packet/acceptance_receipt.py" --mode write \
    --root "$checks" --commit "$WNN_EXPECTED_COMMIT" --split "$WNN_SPLIT"
  exit 0
fi

case "$cell" in
  labels_s101001|labels_s101002) reference_args=() ;;
  complement_s101001|complement_s101002) reference_args=(--reference-cache "$WNN_REFERENCE_ROOT/fit") ;;
  *) echo 'Unknown scientific cell' >&2; exit 2 ;;
esac
run="$WNN_RUN_ROOT/$cell"
config="$packet/configs/$cell.json"
train=(scripts/train_sequence.py --config "$config" --data-root "$WNN_DATA_ROOT" \
  --split-manifest "$WNN_SPLIT" "${reference_args[@]}")

case "$action" in
  profile-one|profile-eight|profile-peak)
    [[ -f "$WNN_RUN_ROOT/acceptance/COMPLETE.json" && -f "$WNN_RUN_ROOT/baseline-score.json" ]] || {
      echo 'Complete receiver checks and baseline replay first' >&2; exit 2;
    }
    uv run --no-sync python "$packet/acceptance_receipt.py" --mode verify \
      --root "$WNN_RUN_ROOT/acceptance" --commit "$WNN_EXPECTED_COMMIT" --split "$WNN_SPLIT"
    world=8
    if [[ "$action" != profile-eight ]]; then
      mkdir -p -- "$run/profile-configs"
      profile_config="$run/profile-configs/$action.json"
      uv run --no-sync python "$packet/profile_config.py" --config "$config" \
        --mode "$action" --output "$profile_config"
      train=(scripts/train_sequence.py --config "$profile_config" --data-root "$WNN_DATA_ROOT" \
        --split-manifest "$WNN_SPLIT" "${reference_args[@]}")
    fi
    if [[ "$action" == profile-one ]]; then world=1; export CUDA_VISIBLE_DEVICES="${devices[0]}"; fi
    uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node="$world" \
      "${train[@]}" --output "$run/$action" --profile-updates 12
    ;;
  fit|resume)
    uv run --no-sync python "$packet/acceptance_receipt.py" --mode verify \
      --root "$WNN_RUN_ROOT/acceptance" --commit "$WNN_EXPECTED_COMMIT" --split "$WNN_SPLIT"
    [[ -f "$run/profile-one/profile_receipt.json" && -f "$run/profile-eight/profile_receipt.json" \
      && -f "$run/profile-peak/profile_receipt.json" && -f "$run/approved_profile.json" ]] || {
      echo 'Need all three profiles and reviewer-approved eight-rank production receipt' >&2; exit 2;
    }
    resume_args=()
    if [[ "$action" == resume ]]; then resume_args=(--resume "$run/fit/training_state.pt"); fi
    uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=8 \
      "${train[@]}" --output "$run/fit" --profile-receipt "$run/approved_profile.json" \
      "${resume_args[@]}"
    ;;
  evaluate)
    [[ -f "$run/fit/terminal_status.json" ]]
    uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=8 \
      scripts/predict_sequence.py --checkpoint "$run/fit/model.pt" --data-root "$WNN_DATA_ROOT" \
      --split-manifest "$WNN_SPLIT" --membership evaluation --output "$run/evaluation" \
      --reference-cache "$WNN_REFERENCE_ROOT/evaluation" --blend-alpha 0.1
    uv run --no-sync python scripts/score_sequence.py --data-root "$WNN_DATA_ROOT" \
      --split-manifest "$WNN_SPLIT" --predictions "$run/evaluation" --output "$run/fit/score.json"
    ;;
  bundle)
    [[ -f "$run/fit/score.json" ]]
    evidence=()
    for name in run_metadata.json metrics.jsonl terminal_status.json score.json \
      rank0.jsonl rank1.jsonl rank2.jsonl rank3.jsonl rank4.jsonl rank5.jsonl rank6.jsonl rank7.jsonl; do
      evidence+=(--evidence "fit/$name")
    done
    uv run --no-sync python scripts/research_result_bundle.py --root "$run" \
      --output-dir "$run/bundle" "${evidence[@]}" --evidence evaluation/manifest.json \
      --evidence profile-one/profile_receipt.json --evidence profile-eight/profile_receipt.json \
      --evidence profile-peak/profile_receipt.json --evidence approved_profile.json \
      --weight fit/model.pt --weight fit/training_state.pt
    ;;
  *) echo 'Unknown stage' >&2; exit 2 ;;
esac
