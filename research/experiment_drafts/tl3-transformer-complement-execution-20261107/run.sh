#!/usr/bin/env bash
set -euo pipefail

action=${1:?data-check, reference-export/import, baseline-score, check, profile-*, fit, resume, evaluate, compare, bundle}
cell=${2:-}
packet=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$packet/../../.."
legacy="$PWD/research/experiment_drafts/tl3-transformer-complement-20261106"
[[ $(uname -s) == Linux ]] || exit 2
: "${WNN_PYTHON:?absolute provisioned Python executable}"
: "${WNN_EXPECTED_COMMIT:?reviewed receiving source commit}"
: "${WNN_SOURCE_MANIFEST:?Git-blob source capsule manifest}"
: "${WNN_SOURCE_MANIFEST_SHA256:?trusted sender-side manifest digest}"
: "${WNN_DATA_ROOT:?preserved canonical PR49 data root}"
: "${WNN_SPLIT:?original chronological primary split}"
: "${WNN_RUN_ROOT:?fresh physical-disk run root}"
: "${WNN_REFERENCE_ROOT:?fresh checked TL3 reference root}"
[[ "$WNN_PYTHON" == /* && "$WNN_RUN_ROOT" == /* && "$WNN_REFERENCE_ROOT" == /* ]] || exit 2
case "$WNN_RUN_ROOT/ $WNN_REFERENCE_ROOT/" in *'/tmp/'*|*'/dev/shm/'*) exit 2 ;; esac
mkdir -p -- "$WNN_RUN_ROOT"
case "$(findmnt --noheadings --output FSTYPE --target "$WNN_RUN_ROOT")" in tmpfs|ramfs) exit 2 ;; esac
export TMPDIR="$WNN_RUN_ROOT/scratch"
export TORCHINDUCTOR_CACHE_DIR="$WNN_RUN_ROOT/compiler/inductor"
export TRITON_CACHE_DIR="$WNN_RUN_ROOT/compiler/triton"
mkdir -p -- "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUBLAS_EMULATE_SINGLE_PRECISION=1
"$WNN_PYTHON" -m wnn.training.sequence_source_capsule --root "$PWD" \
  --manifest "$WNN_SOURCE_MANIFEST" --manifest-sha256 "$WNN_SOURCE_MANIFEST_SHA256" \
  --commit "$WNN_EXPECTED_COMMIT"
launch=("$WNN_PYTHON" -m torch.distributed.run --standalone --nnodes=1)
observe=("$WNN_PYTHON" scripts/observe_sequence.py --interval 10 \
  --max-seconds "${WNN_OBSERVER_MAX_SECONDS:-604800}")

case "$action" in
  data-check)
    findmnt --target "$WNN_RUN_ROOT" > "$WNN_RUN_ROOT/output-mount.txt"
    df -hT "$WNN_RUN_ROOT" > "$WNN_RUN_ROOT/output-capacity.txt"
    "$WNN_PYTHON" "$legacy/receiver_data.py" --data-root "$WNN_DATA_ROOT" \
      --split-manifest "$WNN_SPLIT" --output "$WNN_RUN_ROOT/data-check.json"; exit ;;
  reference-export)
    : "${WNN_PR49_SOURCE:?exact c42 Git-blob capsule}"
    : "${WNN_PR49_SOURCE_MANIFEST:?legacy capsule inventory}"
    : "${WNN_PR49_SOURCE_MANIFEST_SHA256:?trusted legacy inventory digest}"
    : "${CUDA_VISIBLE_DEVICES:?one allocated replay GPU}"
    [[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || exit 2
    env -u CUBLAS_EMULATE_SINGLE_PRECISION "$WNN_PYTHON" "$legacy/reference_export.py" \
      --legacy-source-root "$WNN_PR49_SOURCE" --legacy-source-manifest "$WNN_PR49_SOURCE_MANIFEST" \
      --legacy-source-manifest-sha256 "$WNN_PR49_SOURCE_MANIFEST_SHA256" \
      --data-root "$WNN_DATA_ROOT" --split-manifest "$WNN_SPLIT" \
      --member "$cell" --source "${3:?train or valid}"; exit ;;
  reference-import)
    "$WNN_PYTHON" "$legacy/reference_import.py" --data-root "$WNN_DATA_ROOT" \
      --split-manifest "$WNN_SPLIT" --output-root "$WNN_REFERENCE_ROOT"; exit ;;
  baseline-score)
    "$WNN_PYTHON" "$legacy/reference_score.py" --data-root "$WNN_DATA_ROOT" \
      --split-manifest "$WNN_SPLIT" --reference-cache "$WNN_REFERENCE_ROOT/evaluation" \
      --output "$WNN_RUN_ROOT/baseline-score.json"; exit ;;
  compare)
    "$WNN_PYTHON" "$legacy/compare_panel.py" --run-root "$WNN_RUN_ROOT" \
      --config-root "$packet/configs" --allow-homogeneous-devices \
      --output "$WNN_RUN_ROOT/comparison.json"; exit ;;
esac
: "${CUDA_VISIBLE_DEVICES:?bind two allocated B300 IDs for this cell}"
IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#devices[@]} == 2 && "${devices[0]}" != "${devices[1]}" ]] || exit 2
checks="$WNN_RUN_ROOT/acceptance"
if [[ "$action" == check ]]; then
  [[ -f "$WNN_RUN_ROOT/data-check.json" && ! -e "$checks" ]] || exit 2
  mkdir -- "$checks"
  nvidia-smi -q > "$checks/nvidia-smi.txt"
  nvidia-smi topo -m > "$checks/topology.txt"
  env -u CUBLAS_EMULATE_SINGLE_PRECISION PYTEST_ADDOPTS='' "$WNN_PYTHON" -m pytest -q \
    tests/test_sequence_blend_objective.py tests/test_sequence_reference_cache.py \
    tests/test_chronological_primary_splits.py tests/test_b300_sequence_models.py \
    tests/test_b300_sequence_contracts.py tests/test_sequence_full_fit_data.py \
    tests/test_sequence_recovery_cuda.py tests/test_sequence_muon.py tests/test_pr67_receiving.py \
    tests/test_sequence_observer.py \
    --junitxml="$checks/pytest.xml" 2>&1 | tee "$checks/pytest.log"
  gpu_test=tests/test_sequence_blend_gpu.py::test_compiled_composite_objective_matches_eager_finite_outputs_and_gradients
  for precision in bf16 bf16_bf16x9; do
    emulate=0; name=bf16; if [[ "$precision" == bf16_bf16x9 ]]; then emulate=1; name=bf16x9; fi
    CUBLAS_EMULATE_SINGLE_PRECISION="$emulate" PYTEST_ADDOPTS='' "$WNN_PYTHON" -m pytest -q \
      "$gpu_test[$precision]" --junitxml="$checks/$name.xml" 2>&1 | tee "$checks/$name.log"
  done
  for kind in ddp complement-ddp muon-ddp; do
    script=scripts/check_sequence_ddp.py; extra=()
    [[ "$kind" != complement-ddp ]] || extra=(--blend-objective-alpha 0.1)
    [[ "$kind" != muon-ddp ]] || script=scripts/check_sequence_muon_ddp.py
    env -u CUBLAS_EMULATE_SINGLE_PRECISION "${launch[@]}" --nproc-per-node=2 "$script" \
      "${extra[@]}" --output-receipt "$checks/$kind.json" 2>&1 | tee "$checks/$kind.log"
  done
  "$WNN_PYTHON" "$legacy/acceptance_receipt.py" --mode write --root "$checks" \
    --commit "$WNN_EXPECTED_COMMIT" --split "$WNN_SPLIT"; exit
fi
case "$cell" in
  labels_s101001|labels_s101002) reference_args=() ;;
  complement_s101001|complement_s101002) reference_args=(--reference-cache "$WNN_REFERENCE_ROOT/fit") ;;
  *) exit 2 ;;
esac
run="$WNN_RUN_ROOT/$cell"
config="$packet/configs/$cell.json"
"$WNN_PYTHON" "$legacy/acceptance_receipt.py" --mode verify --root "$checks" \
  --commit "$WNN_EXPECTED_COMMIT" --split "$WNN_SPLIT"
[[ -f "$WNN_RUN_ROOT/baseline-score.json" ]] || exit 2
train=(scripts/train_sequence.py --config "$config" --data-root "$WNN_DATA_ROOT" \
  --split-manifest "$WNN_SPLIT" --monitor-reference-root "$WNN_REFERENCE_ROOT" "${reference_args[@]}")
case "$action" in
  profile-one|profile-two|profile-peak)
    world=2
    if [[ "$action" != profile-two ]]; then
      mkdir -p -- "$run/profile-configs"
      config="$run/profile-configs/$action.json"
      "$WNN_PYTHON" "$legacy/profile_config.py" --config "$packet/configs/$cell.json" \
        --mode "$action" --output "$config"
      train[2]="$config"
    fi
    if [[ "$action" == profile-one ]]; then world=1; export CUDA_VISIBLE_DEVICES="${devices[0]}"; fi
    "${observe[@]}" --output "$run/$action" -- "${launch[@]}" \
      --nproc-per-node="$world" "${train[@]}" --output "$run/$action" --profile-updates 12 ;;
  fit|resume)
    for name in profile-one profile-two profile-peak; do [[ -f "$run/$name/profile_receipt.json" ]] || exit 2; done
    [[ -f "$run/approved_profile.json" ]] || exit 2
    extra=(); [[ "$action" != resume ]] || extra=(--resume "$run/fit/training_state.pt")
    "${observe[@]}" --output "$run/fit" -- "${launch[@]}" --nproc-per-node=2 "${train[@]}" --output "$run/fit" \
      --profile-receipt "$run/approved_profile.json" "${extra[@]}" ;;
  evaluate)
    [[ -f "$run/fit/terminal_status.json" ]] || exit 2
    "${launch[@]}" --nproc-per-node=2 scripts/predict_sequence.py --checkpoint "$run/fit/model.pt" \
      --data-root "$WNN_DATA_ROOT" --split-manifest "$WNN_SPLIT" --membership evaluation \
      --output "$run/evaluation" --reference-cache "$WNN_REFERENCE_ROOT/evaluation" --blend-alpha 0.1
    "$WNN_PYTHON" scripts/score_sequence.py --data-root "$WNN_DATA_ROOT" \
      --split-manifest "$WNN_SPLIT" --predictions "$run/evaluation" --output "$run/fit/score.json" ;;
  bundle)
    evidence=()
    for name in run_metadata.json metrics.jsonl terminal_status.json score.json rank0.jsonl rank1.jsonl \
      monitor_events.jsonl monitor_evaluations.jsonl monitor_progress.json monitor_observer.jsonl; do
      evidence+=(--evidence "fit/$name")
    done
    "$WNN_PYTHON" scripts/research_result_bundle.py --root "$run" --output-dir "$run/bundle" \
      "${evidence[@]}" --evidence evaluation/manifest.json --evidence approved_profile.json \
      --evidence profile-one/profile_receipt.json --evidence profile-two/profile_receipt.json \
      --evidence profile-peak/profile_receipt.json --weight fit/model.pt --weight fit/training_state.pt ;;
  *) exit 2 ;;
esac
