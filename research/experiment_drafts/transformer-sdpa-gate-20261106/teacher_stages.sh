# Teacher profile, fit, evaluation, cache, parity, and bundle stages.
teacher_stage() {
  local family=$1
  local action=$2
  local name=$3
  local config run output fit checkpoint cache prediction score
  if [[ $family == initial ]]; then
    case "$name" in
      none_f0|none_f2|constant_half_f0|constant_half_f2|headwise_f0|headwise_f2) ;;
      *) echo "Unknown initial teacher cell: $name" >&2; exit 2 ;;
    esac
    config="$packet/configs/teacher_${name}.json"
    run="$WNN_RUN_ROOT/teachers/$name"
  else
    case "$name" in
      scale_none_f0|scale_none_f2|scale_headwise_f0|scale_headwise_f2) ;;
      *) echo "Unknown scale teacher cell: $name" >&2; exit 2 ;;
    esac
    config="$packet/configs/$name.json"
    run="$WNN_RUN_ROOT/scale-teachers/$name"
  fi
  split_for_cell "$name"
  [[ -s "$config" ]] || { echo "Missing config: $config" >&2; exit 2; }
  mkdir -p -- "$run"
  fit="$run/fit"
  checkpoint="$fit/model.pt"
  cache="$run/teacher-cache"
  prediction="$run/evaluation"
  score="$fit/score.json"
  local train=(scripts/train_sequence.py --config "$config" --data-root "$WNN_DATA_ROOT" --split-manifest "$split")
  local predict=(scripts/predict_sequence.py --checkpoint "$checkpoint" --data-root "$WNN_DATA_ROOT" --split-manifest "$split")
  local eight=(uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=8)
  export CUBLAS_EMULATE_SINGLE_PRECISION=1
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  case "$action" in
    profile)
      require_devices 8
      output="$run/profile"
      require_absent "$output"
      "${eight[@]}" "${train[@]}" --output "$output" --profile-updates 12
      ;;
    fit)
      require_devices 8
      [[ -s "$run/profile/approved_profile.json" ]] || {
        echo 'Reviewer-created approved_profile.json required before fitting' >&2
        exit 2
      }
      require_absent "$fit"
      "${eight[@]}" "${train[@]}" --output "$fit" \
        --profile-receipt "$run/profile/approved_profile.json"
      ;;
    resume)
      require_devices 8
      [[ -s "$run/profile/approved_profile.json" && -s "$fit/training_state.pt" ]] || {
        echo 'Approved profile and resumable fit state are required' >&2
        exit 2
      }
      "${eight[@]}" "${train[@]}" --output "$fit" \
        --profile-receipt "$run/profile/approved_profile.json" --resume "$fit/training_state.pt"
      ;;
    evaluate)
      require_devices 8
      [[ -s "$checkpoint" && -s "$fit/terminal_status.json" ]] || {
        echo 'Completed fixed-final teacher checkpoint required' >&2
        exit 2
      }
      require_absent "$prediction"
      require_absent "$score"
      "${eight[@]}" "${predict[@]}" --membership evaluation --output "$prediction"
      uv run --no-sync python scripts/score_sequence.py --data-root "$WNN_DATA_ROOT" \
        --split-manifest "$split" --predictions "$prediction" --output "$score"
      ;;
    cache)
      require_devices 8
      [[ -s "$checkpoint" && -s "$fit/terminal_status.json" ]] || {
        echo 'Completed fixed-final teacher checkpoint required' >&2
        exit 2
      }
      require_absent "$cache"
      "${eight[@]}" "${predict[@]}" --membership fit --output "$cache"
      ;;
    cache-parity)
      require_devices 8
      [[ -s "$cache/manifest.json" && -s "$checkpoint" ]] || {
        echo 'Completed eight-rank teacher cache required first' >&2
        exit 2
      }
      local one="$run/teacher-cache-one"
      local parity="$run/cache-parity.json"
      require_absent "$one"
      require_absent "$parity"
      uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
        "${predict[@]}" --membership fit --output "$one"
      uv run --no-sync python \
        research/experiment_drafts/b300-transformer-distillation-runtime/high_compute_r4/cache_parity.py \
        --one "$one" --eight "$cache" --data-root "$WNN_DATA_ROOT" \
        --split-manifest "$split" --output "$parity"
      ;;
    bundle)
      [[ -s "$score" && -s "$fit/terminal_status.json" && -s "$checkpoint" && -s "$fit/training_state.pt" ]] || {
        echo 'Fit, fixed evaluation score, and resumable checkpoint are required' >&2
        exit 2
      }
      local bundle="$run/bundle"
      require_absent "$bundle"
      local evidence=(--evidence run_metadata.json --evidence metrics.jsonl --evidence terminal_status.json --evidence score.json)
      for rank in {0..7}; do evidence+=(--evidence "rank$rank.jsonl"); done
      uv run --no-sync python scripts/research_result_bundle.py --root "$fit" \
        --output-dir "$bundle" "${evidence[@]}" \
        --weight model.pt --weight training_state.pt
      ;;
    *) echo "Unknown teacher action: $action" >&2; exit 2 ;;
  esac
}
