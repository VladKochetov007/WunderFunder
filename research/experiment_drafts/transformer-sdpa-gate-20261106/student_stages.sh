# Matched small-Transformer KD profile, fit, evaluation, and bundle stages.
student_stage() {
  local action=$1
  local name=$2
  case "$name" in
    none_f0|headwise_f0|none_f2|headwise_f2) ;;
    *) echo "Unknown student cell: $name" >&2; exit 2 ;;
  esac
  split_for_cell "$name"
  local fold=${name##*_f}
  local teacher=${name%_f*}
  local config="$packet/configs/student_f${fold}.json"
  local teacher_run="$WNN_RUN_ROOT/teachers/${teacher}_f${fold}"
  local cache="$teacher_run/teacher-cache"
  local teacher_checkpoint="$teacher_run/fit/model.pt"
  local parity="$teacher_run/cache-parity.json"
  local run="$WNN_RUN_ROOT/students/$name"
  local fit="$run/fit"
  local checkpoint="$fit/model.pt"
  local evaluation="$run/evaluation"
  local score="$fit/score.json"
  [[ -s "$config" && -s "$teacher_checkpoint" && -s "$cache/manifest.json" && -s "$parity" ]] || {
    echo 'Student config and parity-checked matching teacher cache are required' >&2
    exit 2
  }
  mkdir -p -- "$run"
  env CUBLAS_EMULATE_SINGLE_PRECISION=1 uv run --no-sync python - \
    "$cache" "$teacher_checkpoint" "$split" "$WNN_DATA_ROOT" \
    "$parity" "$teacher" "$fold" "$packet/configs/teacher_${name}.json" \
    "$run/teacher_binding.json" <<'PY'
import json
import sys
from pathlib import Path
from wnn.splits import file_sha256, load_campaign_split
from wnn.training.sequence_cache import TeacherCache
from wnn.training.sequence_config import load_config
from wnn.training.sequence_provenance import canonical_sha256, source_identity

cache_root = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[2])
split_path = Path(sys.argv[3])
data_root = Path(sys.argv[4])
parity_path = Path(sys.argv[5])
arm = sys.argv[6]
fold = int(sys.argv[7])
split = load_campaign_split(split_path, data_root)
members = [("train", group) for group in split.train_groups] + [
    ("valid", group) for group in split.valid_groups
]
cache = TeacherCache(cache_root, split, members)
manifest_path = cache_root / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
producer = manifest["producer"]
config = producer["config"]
model = config["model"]
parity = json.loads(parity_path.read_text(encoding="utf-8"))
checkpoint_sha = file_sha256(checkpoint_path)
expected_config = load_config(Path(sys.argv[8]))
expected_source = source_identity(Path.cwd(), Path.cwd() / "scripts/train_sequence.py")
identity = producer.get("provenance", {}).get("identity", {})
expected_gate = "headwise" if arm == "headwise" else "none"
expected_seed = 101001 if fold == 0 else 101002
checks = {
    "full teacher config": config == expected_config and identity.get("config_sha256") == canonical_sha256(expected_config),
    "exact teacher source": identity.get("source_sha256") == expected_source["sha256"],
    "parity pass": parity.get("schema") == "wnn.sequence-cache-parity.v1" and parity.get("passed") is True,
    "cache manifest binding": parity.get("manifest_sha256", {}).get("eight") == file_sha256(manifest_path),
    "checkpoint binding": parity.get("checkpoint_sha256") == checkpoint_sha == producer.get("checkpoint_sha256"),
    "split binding": parity.get("split_sha256") == split.sha256 == manifest.get("split_sha256"),
    "fit cache": manifest.get("complete") is True and manifest.get("membership") == "fit",
    "eight-rank cache": manifest.get("world_size") == 8 and cache.metadata.get("world_size") == 8,
    "two-output cache": manifest.get("hint_dim") == 0 and cache.metadata.get("hint_dim") == 0,
    "teacher arm": model.get("attention_gate", "none") == expected_gate,
    "teacher fold and seed": config.get("expected_fold") == fold and config.get("seed") == expected_seed,
    "teacher shape and precision": (model.get("hidden"), model.get("layers"), model.get("heads"), config.get("precision_mode")) == (768, 24, 12, "bf16_bf16x9"),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("teacher cache provenance mismatch: " + ", ".join(failed))
binding = {
    "schema": "wnn.sdpa-gate-teacher-binding.v1",
    "teacher_checkpoint_sha256": checkpoint_sha,
    "teacher_config_sha256": canonical_sha256(expected_config),
    "teacher_source_sha256": expected_source["sha256"],
    "split_sha256": split.sha256,
    "cache_manifest_sha256": file_sha256(manifest_path),
    "parity_receipt_sha256": file_sha256(parity_path),
    "parity_receipt": parity,
    "arm": arm,
    "fold": fold,
}
binding_path = Path(sys.argv[9])
if binding_path.exists():
    if json.loads(binding_path.read_text(encoding="utf-8")) != binding:
        raise SystemExit("existing student binding differs; preserve it and use a new run root")
else:
    with binding_path.open("x", encoding="utf-8") as stream:
        json.dump(binding, stream, indent=2, allow_nan=False)
        stream.write("\n")
PY
  mkdir -p -- "$run"
  local train=(scripts/train_sequence.py --config "$config" --data-root "$WNN_DATA_ROOT" --split-manifest "$split" --teacher-cache "$cache")
  local one=(uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1)
  export CUBLAS_EMULATE_SINGLE_PRECISION=1
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  case "$action" in
    profile)
      require_devices 1
      local output="$run/profile"
      require_absent "$output"
      "${one[@]}" "${train[@]}" --output "$output" --profile-updates 12
      ;;
    fit)
      require_devices 1
      [[ -s "$run/profile/approved_profile.json" ]] || {
        echo 'Reviewer-created approved_profile.json required before student fitting' >&2
        exit 2
      }
      require_absent "$fit"
      "${one[@]}" "${train[@]}" --output "$fit" \
        --profile-receipt "$run/profile/approved_profile.json"
      ;;
    resume)
      require_devices 1
      [[ -s "$run/profile/approved_profile.json" && -s "$fit/training_state.pt" ]] || {
        echo 'Approved student profile and resumable fit state are required' >&2
        exit 2
      }
      "${one[@]}" "${train[@]}" --output "$fit" \
        --profile-receipt "$run/profile/approved_profile.json" --resume "$fit/training_state.pt"
      ;;
    evaluate)
      require_devices 1
      [[ -s "$checkpoint" && -s "$fit/terminal_status.json" ]] || {
        echo 'Completed fixed-final student checkpoint required' >&2
        exit 2
      }
      require_absent "$evaluation"
      require_absent "$score"
      uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
        scripts/predict_sequence.py --checkpoint "$checkpoint" --data-root "$WNN_DATA_ROOT" \
        --split-manifest "$split" --membership evaluation --output "$evaluation"
      uv run --no-sync python scripts/score_sequence.py --data-root "$WNN_DATA_ROOT" \
        --split-manifest "$split" --predictions "$evaluation" --output "$score"
      ;;
    bundle)
      [[ -s "$score" && -s "$fit/terminal_status.json" && -s "$checkpoint" && -s "$fit/training_state.pt" ]] || {
        echo 'Student fit and fixed evaluation score are required' >&2
        exit 2
      }
      local bundle="$run/bundle"
      require_absent "$bundle"
      uv run --no-sync python scripts/research_result_bundle.py --root "$run" \
        --output-dir "$bundle" --evidence fit/run_metadata.json --evidence fit/metrics.jsonl \
        --evidence fit/terminal_status.json --evidence fit/score.json --evidence fit/rank0.jsonl \
        --evidence teacher_binding.json --weight fit/model.pt --weight fit/training_state.pt
      ;;
    *) echo "Unknown student action: $action" >&2; exit 2 ;;
  esac
}
