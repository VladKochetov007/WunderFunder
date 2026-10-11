#!/usr/bin/env bash
set -uo pipefail
umask 077

PROJECT_ROOT=/root/wnn
SOURCE_ROOT=/root/wnn/h391_f1_h384_source_r1
SPLIT=/root/wnn/research/campaign_2026-09-29/artifacts/splits/fold1.json
PORTABLE_MANIFEST=/root/wnn/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/multifold_r1/portable_inputs_fold1.json
CACHE_ROOT=/root/wnn/cache/dc_98411230
RUN_NAME=vt_f1_h391_dsq_h384_t3_a9_e6_s3_profile_r4
RUN_DIR=/root/run/h391_profile_r4
OUT_DIR="${PROJECT_ROOT}/artifacts/runs/${RUN_NAME}"
OOD_LOG=/root/run/logs/pipeline_swn25_r2.log
TRAINER="${SOURCE_ROOT}/scripts/train_rnn.py"
TRAINER_SHA256=8c041bf2a49b43ef5a014a5f7cee59302605d464f5b82fe8c7dc12267c386a40
PARENT_DIR="${PROJECT_ROOT}/artifacts/runs/vt_f1_muonall_h384_e3_s4"
PARENT_CHECKPOINT_SHA256=752f462036a8e340db00de98dcb141795f3a0ae54cfffce3e95ce7ceed7f5e9e
PARENT_PROVENANCE_SHA256=e11ca9b3bfb9fc302f199514eb075960933e8fd27dfa49d58a31b76d5f5011f6
SPLIT_SHA256=31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae
PORTABLE_MANIFEST_SHA256=551905f3363c65608e2d5a0f49e8eeca362f574ab6d98d443e31cee0790fb5fd
CACHE_METADATA_SHA256=a2f82d8dbc7bbbe2c7cf36e18e3b7be0b41f887aa14a742b43f616b060856779

fail() {
    printf 'H391_PROFILE_R4_PREFLIGHT_FAILED: %s\n' "$*" >&2
    exit 70
}

hash_is() {
    local expected="$1"
    local path="$2"
    local actual
    [[ -f "$path" ]] || fail "missing file: $path"
    actual="$(sha256sum "$path" | awk '{print $1}')" || fail "cannot hash: $path"
    [[ "$actual" == "$expected" ]] || fail "hash mismatch: $path expected=$expected actual=$actual"
}

[[ ! -e "$RUN_DIR" ]] || fail "run directory already exists: $RUN_DIR"
[[ ! -e "$OUT_DIR" ]] || fail "profile output already exists: $OUT_DIR"
[[ -f "$OOD_LOG" ]] || fail "missing co-resident OOD log: $OOD_LOG"

TIMEOUT_BIN="$(command -v timeout)" || fail 'timeout is unavailable'
PYTHON_BIN=/opt/conda/bin/python
[[ -x "$PYTHON_BIN" ]] || fail "missing interpreter: $PYTHON_BIN"
for tool in bash sha256sum awk date nvidia-smi tail sleep free df mkdir; do
    command -v "$tool" >/dev/null || fail "required command is unavailable: $tool"
done
[[ "$TIMEOUT_BIN" == /usr/bin/timeout ]] || fail "unexpected timeout path: $TIMEOUT_BIN"

hash_is "$TRAINER_SHA256" "$TRAINER"
hash_is "$SPLIT_SHA256" "$SPLIT"
hash_is "$PORTABLE_MANIFEST_SHA256" "$PORTABLE_MANIFEST"
hash_is "$CACHE_METADATA_SHA256" "$CACHE_ROOT/metadata.json"
hash_is "$PARENT_CHECKPOINT_SHA256" "$PARENT_DIR/model_ep2.pt"
hash_is "$PARENT_PROVENANCE_SHA256" "$PARENT_DIR/provenance.json"

mkdir -m 700 "$RUN_DIR" || fail "cannot create run directory: $RUN_DIR"
printf '%s\n' 'experiment_id=H391' 'profile_id=H391_PROFILE_R4' \
    'physical_gpu=0' 'visible_cuda_device=0' 'trainer_batch_limit=30' \
    'warmup_complete_batches=5' 'measured_complete_batches=25' \
    'profile_runtime_limit_seconds=120' 'fit_authorized=false' \
    "trainer_sha256=${TRAINER_SHA256}" \
    "cache_metadata_sha256=${CACHE_METADATA_SHA256}" \
    "parent_checkpoint_sha256=${PARENT_CHECKPOINT_SHA256}" \
    > "$RUN_DIR/identity.txt"

TRAINER_ARGS=(
    --name "$RUN_NAME" --seed 3
    --device cuda --arch gru --cell gru --hidden 384 --layers 2 --dropout 0
    --train-seqs 10607 --valid-seqs 600 --train-source train+valid
    --valid-fold 1 --n-valid-folds 4 --valid-train-fraction 1 --valid-repeat 1
    --eval-source vfold --split-manifest "$SPLIT" --no-eval-during-fit
    --init vt_f1_muonall_h384_e3_s4@ep2
    --epochs 6 --checkpoint-policy fixed --checkpoint-epoch 5
    --batch 16 --chunk 2000 --loss wmse --loss-norm chunk
    --lr 0.0001 --wd 0.01 --optim adamw --sched onecycle --precision fp16
    --distill-cache-root "$CACHE_ROOT" --distill-cache-variant homogeneous --distill-alpha 0.9
    --label-weight-floor 0 --label-weight-power 1
    --distill-loss mse --distill-channel-weights 1,1 --distill-family-weight 0
    --propensity c20260929_prop_f1 --prop-power 2 --prop-eps 0.03
    --valid-true-mask 0.03 --aux-struct 0.3 --aux-next 0.3 --since-weight 0
    --fast --profile-warmup 5 --profile-steps 25
)
[[ "${#TRAINER_ARGS[@]}" -eq 96 ]] || fail "unexpected trainer argv length: ${#TRAINER_ARGS[@]}"
printf '%q\n' "${TRAINER_ARGS[@]}" > "$RUN_DIR/trainer_argv.shell"
printf '%s\n' "trainer_argv_token_count=${#TRAINER_ARGS[@]}" >> "$RUN_DIR/identity.txt"

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/start_utc.txt"
{
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    free -b
    df -B1 "$PROJECT_ROOT"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
    tail -n 12 "$OOD_LOG"
} > "$RUN_DIR/before.txt" 2>&1 || fail 'could not collect pre-profile resource snapshot'

WNN_PROJECT_ROOT="$PROJECT_ROOT" \
WNN_PORTABLE_DATA_MANIFEST="$PORTABLE_MANIFEST" \
PYTHONPATH="$SOURCE_ROOT/src" \
CUDA_VISIBLE_DEVICES=0 \
WNN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$TIMEOUT_BIN" --signal=TERM --kill-after=5s 120s \
    "$PYTHON_BIN" "$TRAINER" "${TRAINER_ARGS[@]}" \
    > "$RUN_DIR/trainer.out" 2>&1 &
TRAINER_PID=$!
printf '%s\n' "$TRAINER_PID" > "$RUN_DIR/trainer.pid"

(
    while kill -0 "$TRAINER_PID" 2>/dev/null; do
        {
            date -u +'%Y-%m-%dT%H:%M:%SZ'
            nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
            awk '/MemAvailable/ {print "MemAvailable_kB=" $2}' /proc/meminfo
            tail -n 2 "$OOD_LOG"
        } >> "$RUN_DIR/during_samples.txt" 2>&1
        sleep 2
    done
) &
SAMPLER_PID=$!
printf '%s\n' "$SAMPLER_PID" > "$RUN_DIR/sampler.pid"

wait "$TRAINER_PID"
rc=$?
printf '%s\n' "$rc" > "$RUN_DIR/trainer_exit_code.txt"
kill "$SAMPLER_PID" 2>/dev/null || true
wait "$SAMPLER_PID" 2>/dev/null || true
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/end_utc.txt"
{
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    free -b
    df -B1 "$PROJECT_ROOT"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
    tail -n 12 "$OOD_LOG"
    [[ ! -e "$OUT_DIR/model_ep5.pt" ]] || printf 'UNEXPECTED_PROFILE_CHECKPOINT=%s\n' "$OUT_DIR/model_ep5.pt"
    [[ ! -e "$OUT_DIR/training_state.pt" ]] || printf 'UNEXPECTED_PROFILE_STATE=%s\n' "$OUT_DIR/training_state.pt"
} > "$RUN_DIR/after.txt" 2>&1
exit "$rc"
