#!/usr/bin/env bash
set -uo pipefail
umask 077

# Frozen valid-prediction export for the already completed H391 FIT_R2 run.
# Root owns fresh GPU/credit admission and the decision to invoke this script.
PROJECT_ROOT=/root/wnn
SOURCE_ROOT=/root/wnn/h391_f1_h384_source_r1
SOURCE_MANIFEST="$SOURCE_ROOT/SOURCE_MANIFEST_H391_R1.json"
PREDICTOR="$SOURCE_ROOT/scripts/predict_run.py"
PYTHON_BIN=/opt/conda/bin/python
RUN_NAME=vt_f1_h391_dsq_h384_t3_a9_e6_s3_r2
FIT_DIR=/root/run/h391_fit_r2
RUN_DIR=/root/run/h391_export_r1
PORTABLE_MANIFEST=/root/wnn/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/multifold_r1/portable_inputs_fold1.json
SPLIT=/root/wnn/research/campaign_2026-09-29/artifacts/splits/fold1.json
CHECKPOINT="$PROJECT_ROOT/artifacts/runs/$RUN_NAME/model_ep5.pt"
PROVENANCE="$PROJECT_ROOT/artifacts/runs/$RUN_NAME/provenance.json"
FIT_EXIT="$FIT_DIR/trainer_exit_code.txt"
FIT_LOG="$FIT_DIR/trainer.out"
FIT_IDENTITY="$FIT_DIR/identity.txt"
VALID_PARQUET="$PROJECT_ROOT/datasets/valid.parquet"
VALID_TARGETS="$PROJECT_ROOT/cache/valid_targets.npz"
PREDICTION="$PROJECT_ROOT/cache/pred_${RUN_NAME}_valid.npy"
PARTIAL="${PREDICTION}.partial"
METADATA="${PREDICTION%.npy}.json"
SEQSUMS="$PROJECT_ROOT/cache/seqsums_${RUN_NAME}.npz"

SOURCE_MANIFEST_SHA256=744bb3d8b40f4816387f449bbd6c284cc419642e3be548580954e4067f4f7c3d
PREDICTOR_SHA256=46279845540e680940c4444124164102d2499e0468aa7767fc5ba64f90854018
CHECKPOINT_SHA256=0dde2adaf43f36c50862a4fed59e8572660a42dab9ead8cf14cb14f8cb173326
CHECKPOINT_BYTES=8695318
PROVENANCE_SHA256=e9c97557fa9be149153f15073365e31e445798f1820ddb28eae865663442493b
TRAINING_STATE_SHA256=9b58a7843b5843004bf4e759be6cb3f6ebc4c0bac177f4bcd9f95fead15f8181
HISTORY_SHA256=0b12832f3032d53e6b9c48112fab783a2274a8fed309236c721921cfae750638
PORTABLE_MANIFEST_SHA256=551905f3363c65608e2d5a0f49e8eeca362f574ab6d98d443e31cee0790fb5fd
SPLIT_SHA256=31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae
VALID_PARQUET_SHA256=2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782
VALID_TARGETS_SHA256=dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955
FIT_TERMINAL_AUDIT_SHA256=3a8a8b8d3e425703e15fb881dc0eb1804d6c441c7e213f447cde3f517d5c6d46

EXIT_RECEIPT="$RUN_DIR/exit_code.txt"
END_RECEIPT="$RUN_DIR/ended_utc.txt"

fail() {
    printf 'H391_EXPORT_R1_PREFLIGHT_FAILED: %s\n' "$*" >&2
    exit 70
}

finish() {
    local rc=$?
    trap - EXIT
    if [[ -d "$RUN_DIR" && ! -e "$EXIT_RECEIPT" && ! -L "$EXIT_RECEIPT" ]]; then
        local tmp="$EXIT_RECEIPT.tmp.$$"
        printf '%s\n' "$rc" > "$tmp" && mv "$tmp" "$EXIT_RECEIPT" || true
        date -u +'%Y-%m-%dT%H:%M:%SZ' > "$END_RECEIPT" 2>/dev/null || true
    fi
    exit "$rc"
}
trap finish EXIT

hash_is() {
    local expected="$1" path="$2" actual
    [[ -f "$path" && ! -L "$path" ]] || fail "missing or symlinked file: $path"
    actual="$(sha256sum "$path" | awk '{print $1}')" || fail "cannot hash: $path"
    [[ "$actual" == "$expected" ]] || fail "hash mismatch: $path expected=$expected actual=$actual"
}

[[ -d "$RUN_DIR" && -f "$RUN_DIR/start_intent.txt" ]] || fail 'must be started through H391_EXPORT_START_R1.sh'
[[ ! -e "$EXIT_RECEIPT" && ! -L "$EXIT_RECEIPT" ]] || fail "exit receipt already exists: $EXIT_RECEIPT"
[[ -x "$PYTHON_BIN" ]] || fail "missing interpreter: $PYTHON_BIN"
[[ "$(command -v timeout 2>/dev/null)" == /usr/bin/timeout ]] || fail 'GNU timeout must be /usr/bin/timeout'
for tool in sha256sum awk date stat nvidia-smi free df tr; do
    command -v "$tool" >/dev/null || fail "required command is unavailable: $tool"
done

hash_is "$SOURCE_MANIFEST_SHA256" "$SOURCE_MANIFEST"
hash_is "$PREDICTOR_SHA256" "$PREDICTOR"
hash_is "$CHECKPOINT_SHA256" "$CHECKPOINT"
[[ "$(stat -c %s "$CHECKPOINT")" == "$CHECKPOINT_BYTES" ]] || fail 'fixed epoch-5 checkpoint byte count changed'
hash_is "$PROVENANCE_SHA256" "$PROVENANCE"
hash_is "$TRAINING_STATE_SHA256" "$PROJECT_ROOT/artifacts/runs/$RUN_NAME/training_state.pt"
hash_is "$HISTORY_SHA256" "$PROJECT_ROOT/artifacts/runs/$RUN_NAME/history.json"
hash_is "$PORTABLE_MANIFEST_SHA256" "$PORTABLE_MANIFEST"
hash_is "$SPLIT_SHA256" "$SPLIT"
hash_is "$VALID_PARQUET_SHA256" "$VALID_PARQUET"
hash_is "$VALID_TARGETS_SHA256" "$VALID_TARGETS"

# Verify every manifest-listed Python source file, not just the entry point.
"$PYTHON_BIN" - "$SOURCE_ROOT" "$SOURCE_MANIFEST" "$SOURCE_MANIFEST_SHA256" "$PREDICTOR_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
manifest_sha, predictor_sha = sys.argv[3:5]
manifest_bytes = manifest_path.read_bytes()
if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha:
    raise SystemExit("source manifest digest changed during preflight")
manifest = json.loads(manifest_bytes)
if manifest.get("schema") != "wnn.h391-f1-h384-independent-source.v1":
    raise SystemExit("unexpected H391 source manifest schema")
if manifest.get("source_root") != str(root):
    raise SystemExit("source manifest is bound to a different source root")
files = manifest.get("files")
if not isinstance(files, dict) or len(files) != 58:
    raise SystemExit("unexpected frozen source manifest file inventory")
if files.get("scripts/predict_run.py") != predictor_sha:
    raise SystemExit("predictor digest is not bound by the frozen manifest")
for rel, expected in files.items():
    path = root / rel
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing or symlinked frozen source file: {rel}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"frozen source digest mismatch: {rel}")
for subtree in (root / "scripts", root / "src"):
    for path in subtree.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel not in files:
            raise SystemExit(f"unmanifested Python source file: {rel}")
print(f"source_manifest_verified files={len(files)} sha256={manifest_sha}")
PY
source_rc=$?
[[ "$source_rc" -eq 0 ]] || fail "frozen source-tree verification failed rc=$source_rc"

[[ -f "$FIT_EXIT" && ! -L "$FIT_EXIT" ]] || fail "missing completed fit exit receipt: $FIT_EXIT"
[[ "$(tr -d '\r\n' < "$FIT_EXIT")" == 0 ]] || fail 'H391 FIT_R2 did not exit successfully'
[[ -f "$FIT_LOG" && ! -L "$FIT_LOG" ]] || fail "missing H391 training log: $FIT_LOG"
[[ -f "$FIT_IDENTITY" && ! -L "$FIT_IDENTITY" ]] || fail "missing H391 fit identity: $FIT_IDENTITY"
for identity_line in 'experiment_id=H391' 'fit_id=H391_FIT_R2' 'physical_gpu=0' 'checkpoint_epoch=5'; do
    awk -v wanted="$identity_line" '$0 == wanted { found = 1 } END { exit !found }' "$FIT_IDENTITY" ||
        fail "fit identity is missing $identity_line"
done

# Check the producer provenance and the actual six-epoch trainer log against the fixed CP.
"$PYTHON_BIN" - "$PROVENANCE" "$FIT_LOG" "$RUN_NAME" "$CHECKPOINT_SHA256" "$SPLIT_SHA256" <<'PY'
import json
import sys
from pathlib import Path

provenance_path, log_path = map(Path, sys.argv[1:3])
run_name, checkpoint_sha, split_sha = sys.argv[3:6]
provenance = json.loads(provenance_path.read_text())
args = provenance.get("args", {})
if provenance.get("seed") != 3 or args.get("seed") != 3:
    raise SystemExit("producer provenance seed mismatch")
if args.get("name") != run_name:
    raise SystemExit("producer provenance run name mismatch")
if args.get("checkpoint_policy") != "fixed" or args.get("checkpoint_epoch") != 5:
    raise SystemExit("producer provenance does not select fixed epoch 5")
if args.get("epochs") != 6 or provenance.get("selected_epoch") != 5:
    raise SystemExit("producer provenance epoch selection mismatch")
if provenance.get("split_manifest_sha256") != split_sha:
    raise SystemExit("producer provenance split identity mismatch")
if provenance.get("checkpoint_sha256", {}).get("model_ep5.pt") != checkpoint_sha:
    raise SystemExit("producer checkpoint map does not match fixed checkpoint SHA")
epochs = []
selected_fixed = False
for line in log_path.read_text(errors="strict").splitlines():
    if line == "SELECTED_FIXED 5":
        selected_fixed = True
    if line.startswith("FIT "):
        epochs.append(json.loads(line[4:]))
if not selected_fixed or [row.get("epoch") for row in epochs] != list(range(6)):
    raise SystemExit("training log is incomplete or lacks SELECTED_FIXED 5")
last = epochs[-1]
updates = last.get("update_diagnostics", {}).get("cumulative", {})
if last.get("evaluation") != "withheld":
    raise SystemExit("training log does not preserve the withheld evaluation mode")
expected = {"completed_batches": 4476, "optimizer_attempts": 44760,
            "optimizer_completed_updates": 44745, "amp_skipped_updates": 15}
if any(updates.get(key) != value for key, value in expected.items()):
    raise SystemExit("training log terminal update counts differ from the fit receipt")
print("producer_gate_verified run=" + run_name + " epoch=5 epochs=6 fit_exit=0")
PY
producer_rc=$?
[[ "$producer_rc" -eq 0 ]] || fail "producer run/epoch/checkpoint gate failed rc=$producer_rc"

# predict_run.py opens its .partial in truncating mode and removes a stale seqsums
# file; fail closed on every output/side-effect path before invoking it.
for output_path in "$PREDICTION" "$PARTIAL" "$METADATA" "$SEQSUMS"; do
    [[ ! -e "$output_path" && ! -L "$output_path" ]] || fail "prediction output collision: $output_path"
done

{
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    printf 'run=%s\nepoch=5\nphysical_gpu=0\nsource_manifest_sha256=%s\npredictor_sha256=%s\ncheckpoint_sha256=%s\nprovenance_sha256=%s\n' \
        "$RUN_NAME" "$SOURCE_MANIFEST_SHA256" "$PREDICTOR_SHA256" "$CHECKPOINT_SHA256" "$PROVENANCE_SHA256"
    free -b
    df -B1 "$PROJECT_ROOT"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
} > "$RUN_DIR/before.txt" 2>&1 || fail 'could not write pre-export resource snapshot'

WNN_PROJECT_ROOT="$PROJECT_ROOT" \
WNN_PORTABLE_DATA_MANIFEST="$PORTABLE_MANIFEST" \
PYTHONPATH="$SOURCE_ROOT/src" \
CUDA_VISIBLE_DEVICES=0 \
WNN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /usr/bin/timeout --signal=TERM --kill-after=30s 1800s \
    "$PYTHON_BIN" "$PREDICTOR" "$RUN_NAME" valid --epoch 5 \
    > "$RUN_DIR/predictor.out" 2>&1 &
PREDICTOR_TIMEOUT_PID=$!
printf '%s\n' "$PREDICTOR_TIMEOUT_PID" > "$RUN_DIR/predictor_timeout.pid" || fail 'cannot record timeout PID'
wait "$PREDICTOR_TIMEOUT_PID"
predictor_rc=$?
printf '%s\n' "$predictor_rc" > "$RUN_DIR/timeout_exit_code.txt" || fail 'cannot record predictor exit code'
[[ "$predictor_rc" -eq 0 ]] || fail "predictor/timeout exited rc=$predictor_rc"

[[ -f "$PREDICTION" && ! -L "$PREDICTION" ]] || fail 'prediction NPY missing or symlinked'
[[ -f "$METADATA" && ! -L "$METADATA" ]] || fail 'prediction sidecar missing or symlinked'
[[ ! -e "$PARTIAL" && ! -L "$PARTIAL" ]] || fail 'prediction left a partial output; preserve for diagnosis'

# Verify NPY shape/dtype/finite values and sidecar binding without loading the
# ~300 MB prediction array into RAM. This does not score the in-sample valid set.
"$PYTHON_BIN" - "$PREDICTION" "$METADATA" "$RUN_NAME" "$CHECKPOINT_SHA256" "$SPLIT_SHA256" \
    > "$RUN_DIR/output_validation.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

npy_path, sidecar_path = map(Path, sys.argv[1:3])
run_name, checkpoint_sha, split_sha = sys.argv[3:6]
expected_shape = (1873, 20000, 2)
expected_bytes = 299680128
if npy_path.stat().st_size != expected_bytes:
    raise SystemExit(f"unexpected prediction byte count: {npy_path.stat().st_size}")
arr = np.load(npy_path, mmap_mode="r", allow_pickle=False)
if arr.shape != expected_shape or arr.dtype != np.dtype("float32"):
    raise SystemExit("prediction NPY shape/dtype mismatch")
for first in range(0, expected_shape[0], 16):
    if not np.isfinite(arr[first:first + 16]).all():
        raise SystemExit(f"non-finite prediction values in rows {first} onward")
del arr
prediction_sha = hashlib.sha256()
with npy_path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        prediction_sha.update(chunk)
metadata = json.loads(sidecar_path.read_text())
checks = {
    "run": run_name,
    "split": "valid",
    "epoch": 5,
    "checkpoint_sha256": checkpoint_sha,
    "prediction_sha256": prediction_sha.hexdigest(),
    "split_manifest_sha256": split_sha,
    "inference_precision": "fp16",
}
if any(metadata.get(key) != value for key, value in checks.items()):
    raise SystemExit("prediction metadata does not bind the expected run/checkpoint/output")
if len(metadata.get("row_groups", [])) != expected_shape[0]:
    raise SystemExit("prediction metadata row membership count mismatch")
print(json.dumps({"status": "valid_prediction_artifact", "shape": list(expected_shape),
                  "dtype": "float32", "bytes": expected_bytes,
                  "prediction_sha256": prediction_sha.hexdigest(),
                  "sidecar_sha256": hashlib.sha256(sidecar_path.read_bytes()).hexdigest(),
                  "checkpoint_sha256": checkpoint_sha, "epoch": 5}, sort_keys=True))
PY
validation_rc=$?
[[ "$validation_rc" -eq 0 ]] || fail "prediction output validation failed rc=$validation_rc"

{
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
    free -b
    df -B1 "$PROJECT_ROOT"
} > "$RUN_DIR/after.txt" 2>&1 || fail 'could not write post-export resource snapshot'
printf 'H391_EXPORT_R1_COMPLETE run=%s epoch=5 output=%s\n' "$RUN_NAME" "$PREDICTION"
