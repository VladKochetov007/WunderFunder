#!/usr/bin/env bash
set -euo pipefail

action=${1:?action: check, teacher-*, or student-*}
cell=${2:-}
packet=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$packet/../../.."
[[ $(uname -s) == Linux ]] || { echo 'Receiving Linux B300 host only' >&2; exit 2; }
[[ $(git rev-parse --show-toplevel) == "$PWD" ]] || {
  echo 'Run from the pinned repository root' >&2
  exit 2
}
: "${WNN_EXPECTED_COMMIT:?pin the reviewed full source commit SHA}"
: "${WNN_DATA_ROOT:?bind the canonical data root}"
: "${WNN_SPLIT_F0:?bind the canonical exposed F0 manifest}"
: "${WNN_SPLIT_F2:?bind the canonical exposed F2 manifest}"
: "${WNN_RUN_ROOT:?bind a dedicated physical-disk output root}"
[[ $(git rev-parse HEAD) == "$WNN_EXPECTED_COMMIT" ]] || {
  echo 'Source commit mismatch' >&2
  exit 2
}
[[ -z $(git status --porcelain -- src scripts tests "$packet" pyproject.toml uv.lock) ]] || {
  echo 'Runtime, checks, packet or dependencies are dirty' >&2
  exit 2
}
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUBLAS_EMULATE_SINGLE_PRECISION=1

write_profile_config() {
  local source="$1" destination="$2" mode="$3" world="$4"
  mkdir -p -- "$(dirname -- "$destination")"
  uv run --no-sync python - "$source" "$destination" "$mode" "$world" <<'PY'
import json
import sys
from pathlib import Path

source, destination, mode, world = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
config = json.loads(source.read_text(encoding="utf-8"))
if mode not in {"one", "peak"}:
    raise ValueError("profile config mode must be one or peak")
config["required_world_size"] = world
if mode == "peak":
    config["warmup_fraction"] = 0.0
if destination.exists():
    if json.loads(destination.read_text(encoding="utf-8")) != config:
        raise ValueError(f"profile config differs from existing file: {destination}")
else:
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, allow_nan=False)
        stream.write("\n")
PY
}

if [[ "$action" == "check" ]]; then
  : "${CUDA_VISIBLE_DEVICES:?bind the exact allocated eight B300 IDs or UUIDs}"
  IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
  [[ ${#devices[@]} == 8 ]] || { echo 'Receiver checks require eight allocated B300s' >&2; exit 2; }
  checks="$WNN_RUN_ROOT/acceptance"
  mkdir -p -- "$WNN_RUN_ROOT"
  [[ ! -e "$checks" ]] || { echo 'Acceptance output already exists; use a fresh run root' >&2; exit 2; }
  mkdir -- "$checks"
  findmnt --target "$WNN_RUN_ROOT" > "$checks/output-mount.txt"
  df -hT "$WNN_RUN_ROOT" > "$checks/output-capacity.txt"
  lsblk --output NAME,TYPE,FSTYPE,SIZE,MOUNTPOINTS > "$checks/block-devices.txt"
  nvidia-smi -q > "$checks/nvidia-smi.txt"
  nvidia-smi topo -m > "$checks/topology.txt"
  env -u CUBLAS_EMULATE_SINGLE_PRECISION PYTEST_ADDOPTS='' \
    uv run --no-sync python -m pytest -q \
      tests/test_sequence_swiglu.py tests/test_b300_sequence_models.py \
      tests/test_b300_sequence_contracts.py tests/test_sequence_full_fit_data.py \
      tests/test_sequence_recovery_cuda.py tests/test_sequence_muon.py \
      --junitxml="$checks/pytest.xml" 2>&1 | tee "$checks/pytest.log"
  env -u CUBLAS_EMULATE_SINGLE_PRECISION \
    uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=2 \
      scripts/check_sequence_ddp.py --output-receipt "$checks/ddp.json" \
      2>&1 | tee "$checks/ddp.log"
  for activation in gelu swiglu; do
    env -u CUBLAS_EMULATE_SINGLE_PRECISION \
      uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=2 \
        scripts/check_sequence_muon_ddp.py --ff-activation "$activation" \
        --output-receipt "$checks/muon-${activation}-ddp.json" \
        2>&1 | tee "$checks/muon-${activation}-ddp.log"
  done
  exit 0
fi

case "$action" in
  teacher-*) kind=teacher; stage=${action#teacher-} ;;
  student-*) kind=student; stage=${action#student-} ;;
  *) echo 'Unknown action' >&2; exit 2 ;;
esac
: "${CUDA_VISIBLE_DEVICES:?bind the allocated B300 IDs or UUIDs}"
IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"

if [[ "$kind" == teacher ]]; then
  case "$cell" in
    gelu_f0_s101001) config="$packet/configs/teacher/gelu_f0_s101001.json"; fold=0; seed=101001; activation=gelu; width=3072 ;;
    swiglu_f0_s101001) config="$packet/configs/teacher/swiglu_f0_s101001.json"; fold=0; seed=101001; activation=swiglu; width=2048 ;;
    gelu_f2_s101002) config="$packet/configs/teacher/gelu_f2_s101002.json"; fold=2; seed=101002; activation=gelu; width=3072 ;;
    swiglu_f2_s101002) config="$packet/configs/teacher/swiglu_f2_s101002.json"; fold=2; seed=101002; activation=swiglu; width=2048 ;;
    *) echo 'Unknown teacher cell' >&2; exit 2 ;;
  esac
  run="$WNN_RUN_ROOT/teachers/$cell"
  if [[ "$fold" == 0 ]]; then split="$WNN_SPLIT_F0"; else split="$WNN_SPLIT_F2"; fi
  [[ ${#devices[@]} == 8 ]] || { echo 'Teacher profile/fit requires eight allocated B300s' >&2; exit 2; }
  train=(scripts/train_sequence.py --config "$config" --data-root "$WNN_DATA_ROOT" --split-manifest "$split")
  predict=(scripts/predict_sequence.py --checkpoint "$run/fit/model.pt" --data-root "$WNN_DATA_ROOT" --split-manifest "$split")
  case "$stage" in
    profile-one)
      profile_config="$run/profile-configs/${cell}_one.json"
      write_profile_config "$config" "$profile_config" one 1
      env CUDA_VISIBLE_DEVICES="${devices[0]}" \
        uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
          scripts/train_sequence.py --config "$profile_config" --data-root "$WNN_DATA_ROOT" \
          --split-manifest "$split" --output "$run/profile-one" --profile-updates 12
      ;;
    profile-eight)
      [[ -f "$run/profile-one/profile_receipt.json" ]]
      uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=8 \
        "${train[@]}" --output "$run/profile-eight" --profile-updates 12
      ;;
    profile-peak)
      [[ -f "$run/profile-one/profile_receipt.json" && -f "$run/profile-eight/profile_receipt.json" ]]
      profile_config="$run/profile-configs/${cell}_peak.json"
      write_profile_config "$config" "$profile_config" peak 8
      uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=8 \
        scripts/train_sequence.py --config "$profile_config" --data-root "$WNN_DATA_ROOT" \
        --split-manifest "$split" --output "$run/profile-peak" --profile-updates 12
      ;;
    fit|resume)
      [[ -f "$run/profile-one/profile_receipt.json" && -f "$run/profile-eight/profile_receipt.json" \
        && -f "$run/profile-peak/profile_receipt.json" && -f "$run/approved_profile.json" ]] || {
        echo 'Fit requires all three profiles and a reviewer-approved eight-rank receipt' >&2
        exit 2
      }
      if [[ "$stage" == resume ]]; then
        uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=8 \
          "${train[@]}" --output "$run/fit" --profile-receipt "$run/approved_profile.json" \
          --resume "$run/fit/training_state.pt"
      else
        uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=8 \
          "${train[@]}" --output "$run/fit" --profile-receipt "$run/approved_profile.json"
      fi
      ;;
    evaluate)
      [[ -f "$run/fit/terminal_status.json" ]]
      uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=8 \
        "${predict[@]}" --membership evaluation --output "$run/evaluation"
      uv run --no-sync python scripts/score_sequence.py --data-root "$WNN_DATA_ROOT" \
        --split-manifest "$split" --predictions "$run/evaluation" --output "$run/fit/score.json"
      ;;
    cache-eight)
      [[ -f "$run/fit/score.json" ]]
      uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=8 \
        "${predict[@]}" --membership fit --output "$run/cache-eight"
      ;;
    cache-one)
      [[ -f "$run/fit/score.json" ]]
      env CUDA_VISIBLE_DEVICES="${devices[0]}" \
        uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
          "${predict[@]}" --membership fit --output "$run/cache-one"
      ;;
    cache-parity)
      [[ -f "$run/cache-one/manifest.json" && -f "$run/cache-eight/manifest.json" ]]
      uv run --no-sync python "$packet/../b300-transformer-distillation-runtime/high_compute_r4/cache_parity.py" \
        --one "$run/cache-one" \
        --eight "$run/cache-eight" --data-root "$WNN_DATA_ROOT" --split-manifest "$split" \
        --output "$run/cache-parity.json"
      ;;
    bundle)
      [[ -f "$run/fit/score.json" && -f "$run/cache-parity.json" ]]
      args=()
      for file in run_metadata.json metrics.jsonl terminal_status.json score.json rank0.jsonl \
        rank1.jsonl rank2.jsonl rank3.jsonl rank4.jsonl rank5.jsonl rank6.jsonl rank7.jsonl; do
        args+=(--evidence "fit/$file")
      done
      uv run --no-sync python scripts/research_result_bundle.py --root "$run" \
        --output-dir "$run/bundle" "${args[@]}" \
        --evidence approved_profile.json \
        --evidence profile-one/profile_receipt.json \
        --evidence profile-eight/profile_receipt.json \
        --evidence profile-peak/profile_receipt.json \
        --evidence cache-one/manifest.json --evidence cache-eight/manifest.json \
        --evidence cache-parity.json --weight fit/model.pt --weight fit/training_state.pt
      ;;
    *) echo 'Unknown teacher stage' >&2; exit 2 ;;
  esac
else
  case "$cell" in
    student_gelu_f0_s1001) config="$packet/configs/student/f0_s1001.json"; fold=0; seed=1001; teacher_cell=gelu_f0_s101001; activation=gelu; width=3072; teacher_seed=101001 ;;
    student_swiglu_f0_s1001) config="$packet/configs/student/f0_s1001.json"; fold=0; seed=1001; teacher_cell=swiglu_f0_s101001; activation=swiglu; width=2048; teacher_seed=101001 ;;
    student_gelu_f2_s1002) config="$packet/configs/student/f2_s1002.json"; fold=2; seed=1002; teacher_cell=gelu_f2_s101002; activation=gelu; width=3072; teacher_seed=101002 ;;
    student_swiglu_f2_s1002) config="$packet/configs/student/f2_s1002.json"; fold=2; seed=1002; teacher_cell=swiglu_f2_s101002; activation=swiglu; width=2048; teacher_seed=101002 ;;
    *) echo 'Unknown student cell' >&2; exit 2 ;;
  esac
  [[ ${#devices[@]} == 1 ]] || { echo 'Student fit requires one allocated B300' >&2; exit 2; }
  case "$stage" in profile-one|profile-peak|fit|resume|evaluate|bundle) ;; *) echo 'Unknown student stage' >&2; exit 2 ;; esac
  run="$WNN_RUN_ROOT/students/$cell"
  teacher_run="$WNN_RUN_ROOT/teachers/$teacher_cell"
  teacher_cache="$teacher_run/cache-eight"
  teacher_one="$teacher_run/cache-one"
  parity="$teacher_run/cache-parity.json"
  if [[ "$fold" == 0 ]]; then split="$WNN_SPLIT_F0"; else split="$WNN_SPLIT_F2"; fi
  mkdir -p -- "$run/cache-bindings"
  uv run --no-sync python "$packet/validate_teacher_cache.py" --data-root "$WNN_DATA_ROOT" \
    --split-manifest "$split" --one "$teacher_one" \
    --eight "$teacher_cache" --parity "$parity" --checkpoint "$teacher_run/fit/model.pt" \
    --activation "$activation" --width "$width" --fold "$fold" --seed "$teacher_seed" \
    --expected-config "$packet/configs/teacher/${teacher_cell}.json" \
    --source-root "$PWD" --entrypoint "$PWD/scripts/train_sequence.py" \
    --binding-output "$run/cache-bindings/$stage.json"
  train=(scripts/train_sequence.py --config "$config" --data-root "$WNN_DATA_ROOT" \
    --split-manifest "$split" --teacher-cache "$teacher_cache")
  predict=(scripts/predict_sequence.py --checkpoint "$run/fit/model.pt" \
    --data-root "$WNN_DATA_ROOT" --split-manifest "$split")
  case "$stage" in
    profile-one)
      uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
        "${train[@]}" --output "$run/profile-one" --profile-updates 12
      ;;
    profile-peak)
      [[ -f "$run/profile-one/profile_receipt.json" ]]
      profile_config="$run/profile-configs/${cell}_peak.json"
      write_profile_config "$config" "$profile_config" peak 1
      uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
        scripts/train_sequence.py --config "$profile_config" --data-root "$WNN_DATA_ROOT" \
        --split-manifest "$split" --teacher-cache "$teacher_cache" \
        --output "$run/profile-peak" --profile-updates 12
      ;;
    fit|resume)
      [[ -f "$run/profile-one/profile_receipt.json" && -f "$run/profile-peak/profile_receipt.json" \
        && -f "$run/approved_profile.json" ]] || {
        echo 'Student fit requires both profiles and a reviewer-approved profile receipt' >&2
        exit 2
      }
      if [[ "$stage" == resume ]]; then
        uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
          "${train[@]}" --output "$run/fit" --profile-receipt "$run/approved_profile.json" \
          --resume "$run/fit/training_state.pt"
      else
        uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
          "${train[@]}" --output "$run/fit" --profile-receipt "$run/approved_profile.json"
      fi
      ;;
    evaluate)
      [[ -f "$run/fit/terminal_status.json" ]]
      env CUDA_VISIBLE_DEVICES="${devices[0]}" \
        uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
          "${predict[@]}" --membership evaluation --output "$run/evaluation"
      uv run --no-sync python scripts/score_sequence.py --data-root "$WNN_DATA_ROOT" \
        --split-manifest "$split" --predictions "$run/evaluation" --output "$run/fit/score.json"
      ;;
    bundle)
      [[ -f "$run/fit/score.json" ]]
      args=()
      for file in run_metadata.json metrics.jsonl terminal_status.json score.json rank0.jsonl; do
        args+=(--evidence "fit/$file")
      done
      for profile_stage in profile-one profile-peak; do
        args+=(--evidence "$profile_stage/profile_receipt.json")
      done
      for binding_stage in profile-one profile-peak fit resume evaluate bundle; do
        if [[ -f "$run/cache-bindings/$binding_stage.json" ]]; then
          args+=(--evidence "cache-bindings/$binding_stage.json")
        fi
      done
      uv run --no-sync python scripts/research_result_bundle.py --root "$run" \
        --output-dir "$run/bundle" "${args[@]}" \
        --evidence approved_profile.json --weight fit/model.pt --weight fit/training_state.pt
      ;;
    *) echo 'Unknown student stage' >&2; exit 2 ;;
  esac
fi
