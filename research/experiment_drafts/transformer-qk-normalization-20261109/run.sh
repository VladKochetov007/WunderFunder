#!/usr/bin/env bash
set -euo pipefail
action=${1:?choose data-check, check, profile, profile-peak, fit, resume, evaluate, evaluate-blend, compare, compare-blend, bundle, panel-bundle}
cell=${2:-}
packet=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$packet/../../.."
[[ $(uname -s) == Linux ]] || { echo 'Receiving Linux host only' >&2; exit 2; }
: "${WNN_EXPECTED_COMMIT:?pin the reviewed full commit}"
: "${WNN_DATA_ROOT:?preserved canonical PR49 data root}"
: "${WNN_SPLIT:?exact original chronological split}"
: "${WNN_RUN_ROOT:?fresh persistent output root}"
[[ "$WNN_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ && $(git rev-parse HEAD) == "$WNN_EXPECTED_COMMIT" ]] || exit 2
[[ -z $(git status --porcelain -- src scripts tests "$packet" pyproject.toml uv.lock) ]] || {
  echo 'Source, checks, packet or dependencies are dirty' >&2; exit 2;
}
[[ "$WNN_RUN_ROOT" == /* && "$WNN_DATA_ROOT" == /* && "$WNN_SPLIT" == /* ]] || exit 2
case "$(realpath -m -- "$WNN_RUN_ROOT")/" in /tmp/*|/dev/shm/*) exit 2 ;; esac
mkdir -p -- "$WNN_RUN_ROOT"
case "$(findmnt --noheadings --output FSTYPE --target "$WNN_RUN_ROOT")" in tmpfs|ramfs) exit 2 ;; esac
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUBLAS_EMULATE_SINGLE_PRECISION=1
export TMPDIR="$WNN_RUN_ROOT/scratch"
export TORCHINDUCTOR_CACHE_DIR="$WNN_RUN_ROOT/compiler/inductor"
export TRITON_CACHE_DIR="$WNN_RUN_ROOT/compiler/triton"
mkdir -p -- "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

case "$action" in
  data-check)
    findmnt --target "$WNN_RUN_ROOT" > "$WNN_RUN_ROOT/output-mount.txt"
    df -hT "$WNN_RUN_ROOT" > "$WNN_RUN_ROOT/output-capacity.txt"
    uv run --no-sync python "$packet/receiver_data.py" --data-root "$WNN_DATA_ROOT" \
      --split-manifest "$WNN_SPLIT" --output "$WNN_RUN_ROOT/data-check.json"
    exit 0 ;;
  panel-bundle)
    [[ -f "$WNN_RUN_ROOT/comparison.json" && ! -e "$WNN_RUN_ROOT/panel-evidence.tar" ]] || exit 2
    panel_evidence=(comparison.json data-check.json acceptance/COMPLETE.json
      acceptance/pytest.xml acceptance/bf16.xml acceptance/bf16_bf16x9.xml
      acceptance/qk-ddp.json acceptance/muon-ddp.json output-mount.txt output-capacity.txt)
    for item in plain_s203301 rms_s203301 plain_s203302 rms_s203302; do
      panel_evidence+=("runs/$item/bundle/manifest.json")
    done
    if [[ -f "$WNN_RUN_ROOT/comparison-with-blend.json" ]]; then
      panel_evidence+=(comparison-with-blend.json)
    fi
    tar --create --file "$WNN_RUN_ROOT/panel-evidence.tar" --directory "$WNN_RUN_ROOT" "${panel_evidence[@]}"
    sha256sum "$WNN_RUN_ROOT/panel-evidence.tar" > "$WNN_RUN_ROOT/panel-evidence.sha256"
    exit 0 ;;
  compare|compare-blend)
    compare_args=()
    if [[ "$action" == compare-blend ]]; then compare_args=(--with-blend); fi
    uv run --no-sync python "$packet/compare_panel.py" --run-root "$WNN_RUN_ROOT" \
      --split-manifest "$WNN_SPLIT" --data-root "$WNN_DATA_ROOT" "${compare_args[@]}"
    exit 0 ;;
esac
: "${CUDA_VISIBLE_DEVICES:?bind exactly two allocated B300 GPU IDs}"
IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#devices[@]} == 2 && "${devices[0]}" != "${devices[1]}" ]] || exit 2

if [[ "$action" == check ]]; then
  checks="$WNN_RUN_ROOT/acceptance"
  [[ -f "$WNN_RUN_ROOT/data-check.json" && ! -e "$checks" ]] || exit 2
  mkdir -- "$checks"
  nvidia-smi -q > "$checks/nvidia-smi.txt"
  nvidia-smi topo -m > "$checks/topology.txt"
  env -u CUBLAS_EMULATE_SINGLE_PRECISION PYTEST_ADDOPTS='' timeout --kill-after=60s 1800s \
    uv run --no-sync python -m pytest -q tests/test_sequence_qk_norm.py \
    tests/test_b300_sequence_models.py tests/test_b300_sequence_contracts.py \
    tests/test_sequence_standalone_labels.py tests/test_sequence_precision.py \
    tests/test_sequence_loop_sampling.py tests/test_sequence_recovery_cuda.py \
    tests/test_sequence_muon.py --junitxml="$checks/pytest.xml" 2>&1 | tee "$checks/pytest.log"
  for mode in bf16 bf16_bf16x9; do
    emulation=0
    if [[ "$mode" == bf16_bf16x9 ]]; then emulation=1; fi
    CUBLAS_EMULATE_SINGLE_PRECISION="$emulation" PYTEST_ADDOPTS='' timeout --kill-after=60s 1800s \
      uv run --no-sync python -m pytest -q \
      "tests/test_sequence_qk_norm_gpu.py::test_compiled_qk_norm[$mode]" \
      --junitxml="$checks/$mode.xml" 2>&1 | tee "$checks/$mode.log"
  done
  env -u CUBLAS_EMULATE_SINGLE_PRECISION timeout --kill-after=60s 900s \
    uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=2 \
    "$packet/check_qk_ddp.py" --output "$checks/qk-ddp.json" 2>&1 | tee "$checks/qk-ddp.log"
  env -u CUBLAS_EMULATE_SINGLE_PRECISION timeout --kill-after=60s 900s \
    uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=2 \
    scripts/check_sequence_muon_ddp.py --output-receipt "$checks/muon-ddp.json" \
    2>&1 | tee "$checks/muon-ddp.log"
  uv run --no-sync python "$packet/acceptance.py" --mode write --root "$WNN_RUN_ROOT" \
    --commit "$WNN_EXPECTED_COMMIT" --split "$WNN_SPLIT" --data-root "$WNN_DATA_ROOT"
  exit 0
fi
case "$cell" in plain_s203301|rms_s203301|plain_s203302|rms_s203302) ;;
  *) echo 'Unknown frozen cell' >&2; exit 2 ;;
esac
uv run --no-sync python "$packet/acceptance.py" --mode verify --root "$WNN_RUN_ROOT" \
  --commit "$WNN_EXPECTED_COMMIT" --split "$WNN_SPLIT" --data-root "$WNN_DATA_ROOT"
run="$WNN_RUN_ROOT/runs/$cell"
config="$packet/configs/$cell.json"
train=(scripts/train_sequence.py --config "$config" --data-root "$WNN_DATA_ROOT" --split-manifest "$WNN_SPLIT")
launch=(uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=2)
case "$action" in
  profile|profile-peak)
    if [[ "$action" == profile-peak ]]; then
      mkdir -p -- "$run/profile-configs"
      config="$run/profile-configs/peak.json"
      uv run --no-sync python "$packet/profile_config.py" --config "$packet/configs/$cell.json" --output "$config"
      train=(scripts/train_sequence.py --config "$config" --data-root "$WNN_DATA_ROOT" --split-manifest "$WNN_SPLIT")
    fi
    timeout --kill-after=60s 1800s "${launch[@]}" "${train[@]}" --output "$run/$action" --profile-updates 50 ;;
  fit|resume)
    uv run --no-sync python "$packet/acceptance.py" --mode profiles --root "$WNN_RUN_ROOT" \
      --commit "$WNN_EXPECTED_COMMIT" --split "$WNN_SPLIT" --data-root "$WNN_DATA_ROOT" --cell "$cell"
    resume_args=()
    if [[ "$action" == resume ]]; then resume_args=(--resume "$run/fit/training_state.pt"); fi
    timeout --kill-after=60s 28800s "${launch[@]}" "${train[@]}" --output "$run/fit" \
      --profile-receipt "$run/approved_profile.json" "${resume_args[@]}" ;;
  evaluate|evaluate-blend)
    extra=(); suffix=''
    if [[ "$action" == evaluate-blend ]]; then
      : "${WNN_REFERENCE_ROOT:?exact PR49 reference cache root with evaluation/manifest.json}"
      extra=(--reference-cache "$WNN_REFERENCE_ROOT/evaluation" --blend-alpha 0.1)
      suffix='-blend'
    fi
    [[ -f "$run/fit/terminal_status.json" ]] || exit 2
    timeout --kill-after=60s 7200s "${launch[@]}" scripts/predict_sequence.py \
      --checkpoint "$run/fit/model.pt" --data-root "$WNN_DATA_ROOT" --split-manifest "$WNN_SPLIT" \
      --membership evaluation --output "$run/evaluation$suffix" "${extra[@]}"
    uv run --no-sync python scripts/score_sequence.py --data-root "$WNN_DATA_ROOT" \
      --split-manifest "$WNN_SPLIT" --predictions "$run/evaluation$suffix" --output "$run/fit/score$suffix.json" ;;
  bundle)
    evidence=()
    for name in run_metadata.json metrics.jsonl terminal_status.json score.json rank0.jsonl rank1.jsonl \
      monitor_events.jsonl monitor_evaluations.jsonl monitor_progress.json epoch_index.json; do
      evidence+=(--evidence "fit/$name")
    done
    if [[ -f "$run/fit/score-blend.json" ]]; then
      evidence+=(--evidence fit/score-blend.json --evidence evaluation-blend/manifest.json)
    fi
    uv run --no-sync python scripts/research_result_bundle.py --root "$run" --output-dir "$run/bundle" \
      "${evidence[@]}" --evidence evaluation/manifest.json --evidence profile/profile_receipt.json \
      --evidence profile-peak/profile_receipt.json --evidence approved_profile.json \
      --weight fit/model.pt --weight fit/training_state.pt ;;
  *) echo 'Unknown stage' >&2; exit 2 ;;
esac
