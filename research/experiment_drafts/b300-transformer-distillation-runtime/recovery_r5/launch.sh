#!/usr/bin/env bash
# Explicit receiver actions; each invocation ends before the next operator decision.
set -euo pipefail

handoff=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$handoff/../../../.."
[[ $(uname -s) == Linux ]] || { echo 'Run this handoff on the receiving Linux host' >&2; exit 2; }
: "${WNN_EXPECTED_COMMIT:?set the reviewed full commit SHA}"
: "${WNN_CAMPAIGN_ROOT:?set the new R5 campaign root}"
: "${UV_PROJECT_ENVIRONMENT:?set the existing team B300 environment}"
[[ $WNN_EXPECTED_COMMIT =~ ^[0-9a-f]{40}$ ]] || { echo 'A full commit SHA is required' >&2; exit 2; }
[[ $(git rev-parse HEAD) == "$WNN_EXPECTED_COMMIT" ]] || { echo 'Source commit mismatch' >&2; exit 2; }
[[ -z $(git status --porcelain -- src scripts tests "$handoff/.." pyproject.toml uv.lock) ]] || {
  echo 'Runtime, tests, packet or dependencies are dirty' >&2; exit 2;
}
[[ $UV_PROJECT_ENVIRONMENT == /* && -x $UV_PROJECT_ENVIRONMENT/bin/python ]] || {
  echo 'Use an existing absolute B300 environment; this handoff installs nothing' >&2; exit 2;
}
[[ $WNN_CAMPAIGN_ROOT == /* ]] || { echo 'Campaign root must be absolute' >&2; exit 2; }
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_EMULATE_SINGLE_PRECISION=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

run_python() { uv run --no-sync python "$@"; }
run_cell() { run_python "$handoff/run.py" "$1" --campaign-root "$WNN_CAMPAIGN_ROOT" --cell "$2"; }
fail() { echo "$*" >&2; exit 2; }
pair_paths() {
  case "$1" in 1e-4|3e-4) ;; *) fail 'LR must be 1e-4 or 3e-4' ;; esac
  pair_lr=$1
  selection="$WNN_CAMPAIGN_ROOT/selection-lr-$pair_lr.json"
  retention="$WNN_CAMPAIGN_ROOT/retention-lr-$pair_lr.json"
}
verify_pair() {
  local plan="$WNN_CAMPAIGN_ROOT/retention-lr-$1.json"
  [[ -f $plan && ! -L $plan ]] || fail 'Previous LR pair has no regular retention plan'
  local digest
  digest=$(sha256sum -- "$plan" | cut -d ' ' -f 1)
  run_python "$handoff/retention_apply.py" --verify --plan "$plan" --plan-sha256 "$digest"
}
sweep_cell() {
  local cell=$1 fold=$2 outer=$3
  local fit="$WNN_CAMPAIGN_ROOT/runs/$cell/fit"
  local index="$fit/epoch_indexes/epoch-000030.json"
  local output="$WNN_CAMPAIGN_ROOT/runs/$cell/epoch_evaluation"
  local digest
  digest=$(run_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["index_sha256"])' "$index")
  local resume=()
  [[ ! -e $output && ! -L $output ]] || resume=(--resume)
  run_python "$handoff/epoch_sweep.py" --fit-dir "$fit" \
    --config "$WNN_CAMPAIGN_ROOT/cells/$cell/config.json" --data-root "$WNN_DATA_ROOT" \
    --outer-split "$outer" --inner-split "$WNN_INPUTS_ROOT/inner-f$fold.json" \
    --epoch-index "$index" --epoch-index-sha256 "$digest" --output "$output" "${resume[@]}"
}

action=${1:-}
if [[ $action != prepare ]]; then
  [[ -d $WNN_CAMPAIGN_ROOT && ! -L $WNN_CAMPAIGN_ROOT ]] || fail 'Prepare a fresh campaign first'
  # One explicit action owns this campaign. No background jobs or automatic retries.
  exec 9>"$WNN_CAMPAIGN_ROOT/.launch.lock"
  flock -n 9 || fail 'Another launch action owns this campaign'
fi
case "$action" in
  prepare)
    [[ $# == 1 ]] || fail 'Usage: launch.sh prepare'
    bash "$handoff/prepare.sh"
    cat "$WNN_CAMPAIGN_ROOT/split_audit.json"
    run_python "$handoff/run.py" list --campaign-root "$WNN_CAMPAIGN_ROOT"
    ;;
  check)
    [[ $# == 1 ]] || fail 'Usage: launch.sh check'
    run_python "$handoff/run.py" check --campaign-root "$WNN_CAMPAIGN_ROOT"
    ;;
  fit-cell)
    [[ $# == 2 ]] || fail 'Usage: launch.sh fit-cell CELL'
    cell=$2
    case "$cell" in
      R5_lr_1e-4_f0|R5_lr_1e-4_f2) ;;
      R5_lr_3e-4_f0|R5_lr_3e-4_f2) verify_pair 1e-4 ;;
      *) fail 'Unknown cell; only the four frozen R5 fits are allowed' ;;
    esac
    [[ ! -e $WNN_CAMPAIGN_ROOT/runs/$cell/fit && ! -L $WNN_CAMPAIGN_ROOT/runs/$cell/fit ]] || {
      fail 'Fit output exists; use the documented explicit resume procedure'
    }
    : "${WNN_REVIEWER:?set the receiver name}"
    : "${WNN_HOST_RAM_BUDGET_GIB:?set the measured host allocation}"
    for step in profile-one profile-eight profile-peak admit fit; do run_cell "$step" "$cell"; done
    ;;
  finish-pair)
    [[ $# == 2 ]] || fail 'Usage: launch.sh finish-pair LR'
    pair_paths "$2"
    : "${WNN_DATA_ROOT:?set the bound data root}"
    : "${WNN_SPLIT_F0:?set the canonical F0 manifest}"
    : "${WNN_SPLIT_F2:?set the canonical F2 manifest}"
    : "${WNN_INPUTS_ROOT:?set the prepared inner-split root}"
    for target in "$selection" "$retention" \
      "$WNN_CAMPAIGN_ROOT/midas-evidence/R5_lr_${pair_lr}_f0-pre-retention" \
      "$WNN_CAMPAIGN_ROOT/midas-evidence/R5_lr_${pair_lr}_f2-pre-retention"; do
      [[ ! -e $target && ! -L $target ]] || fail 'Finalization artifact exists; use explicit recovery commands in LAUNCH_AND_STORAGE.md'
    done
    sweep_cell "R5_lr_${pair_lr}_f0" 0 "$WNN_SPLIT_F0"
    sweep_cell "R5_lr_${pair_lr}_f2" 2 "$WNN_SPLIT_F2"
    run_python "$handoff/epoch_selection.py" \
      --f0-output "$WNN_CAMPAIGN_ROOT/runs/R5_lr_${pair_lr}_f0/epoch_evaluation" \
      --f2-output "$WNN_CAMPAIGN_ROOT/runs/R5_lr_${pair_lr}_f2/epoch_evaluation" --output "$selection"
    for fold in 0 2; do
      run_python "$handoff/export_midas.py" --campaign-root "$WNN_CAMPAIGN_ROOT" \
        --cell "R5_lr_${pair_lr}_f$fold" \
        --output "$WNN_CAMPAIGN_ROOT/midas-evidence/R5_lr_${pair_lr}_f$fold-pre-retention"
    done
    run_python "$handoff/retention_plan.py" \
      --f0-output "$WNN_CAMPAIGN_ROOT/runs/R5_lr_${pair_lr}_f0/epoch_evaluation" \
      --f2-output "$WNN_CAMPAIGN_ROOT/runs/R5_lr_${pair_lr}_f2/epoch_evaluation" \
      --selection "$selection" --output "$retention"
    sha256sum -- "$retention"
    echo 'Review the plan and preserve the evidence bundle before explicit cleanup-pair.'
    ;;
  cleanup-pair)
    [[ $# == 3 ]] || fail 'Usage: launch.sh cleanup-pair LR REVIEWED_RAW_PLAN_SHA256'
    pair_paths "$2"
    [[ $3 =~ ^[0-9a-f]{64}$ ]] || fail 'Pass the reviewed raw plan SHA256'
    run_python "$handoff/retention_apply.py" --apply --plan "$retention" --plan-sha256 "$3"
    ;;
  verify-pair)
    [[ $# == 2 ]] || fail 'Usage: launch.sh verify-pair LR'
    pair_paths "$2"
    verify_pair "$pair_lr"
    ;;
  *) fail 'Actions: prepare | check | fit-cell CELL | finish-pair LR | cleanup-pair LR SHA256 | verify-pair LR' ;;
esac
