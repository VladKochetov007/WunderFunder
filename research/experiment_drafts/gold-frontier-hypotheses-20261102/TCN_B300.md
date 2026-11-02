# D: causal TCN versus GRU, B300 receiver

Implementation uses the existing `wnn.nn.TCNModel` / `CausalConvBlock`; the
sequence adapter packs its per-layer left histories into a tensor for the
existing detach/reset and prediction interface. Six layers have dilations
1,2,4,8,16,32 and receptive field 64 rows, not the GRU's unrestricted state.
The prediction head and normalization are the existing TCN path. Forward
semantics must be checked on the receiver for full sequence versus differently
chunked inference before interpreting a training result. This has not been run.

Frozen starting recipe: GRU256x2 versus TCN256x6, raw112, labels only, AdamW
3e-4, decay .01, 5% warmup, six epochs, batch 1, 256-row chunks, accumulate 4,
FP32 with TF32 off, one B300 per fit. Same fold/seed and row/update counts; parameter
counts, wall time, memory and FLOPs are measured and reported, not presumed equal.
Two control and two TCN fits are planned. No KD or Muon is active in this teacher
comparison. Muon is restricted to Transformer in this runtime; switching optimizer
would be a different comparison. Two separate student-transfer controls and two KD
fits, clean target caches and their cost are additional.

Start only after the receiving host runs reviewed causal/streaming, numerical,
resume, fold-ancestry and shape acceptance checks against this exact source. These
commands do not install dependencies or alter running jobs. Use the approved team
environment and physical scratch; do not skip profile admission. `WNN_SPLIT_F0`
and `WNN_SPLIT_F2` must be clean source-qualified manifests, not PR54 full-fit data.

```bash
cd "$WNN_REPO"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
: "${WNN_DATA_ROOT:?}" "${WNN_SPLIT_F0:?}" "${WNN_SPLIT_F2:?}" "${WNN_RUN_ROOT:?}"
PACKET=research/experiment_drafts/gold-frontier-hypotheses-20261102

# Run once per pair of CELL and matching SPLIT: f0_s1001/F0, f2_s1002/F2.
CELL=f0_s1001
SPLIT="$WNN_SPLIT_F0"
for ARCH in gru tcn; do
  CONFIG="$PACKET/configs/d_${ARCH}_${CELL}.json"
  uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    scripts/train_sequence.py --config "$CONFIG" --data-root "$WNN_DATA_ROOT" \
    --split-manifest "$SPLIT" --output "$WNN_RUN_ROOT/d_${ARCH}_${CELL}_profile" \
    --profile-updates 8
done
```

Review each `profile_receipt.json`, finite updates, actual GPU memory and
headroom. Write an approved copy as `approved_profile.json` only after receiver
engineering checks pass; retain all other identity fields. Each config/source/data/
GPU tuple needs its own approved receipt. Then:

```bash
for ARCH in gru tcn; do
  CONFIG="$PACKET/configs/d_${ARCH}_${CELL}.json"
  RUN="$WNN_RUN_ROOT/d_${ARCH}_${CELL}"
  uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    scripts/train_sequence.py --config "$CONFIG" --data-root "$WNN_DATA_ROOT" \
    --split-manifest "$SPLIT" --output "$RUN" \
    --profile-receipt "$WNN_RUN_ROOT/d_${ARCH}_${CELL}_profile/approved_profile.json"
  uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    scripts/predict_sequence.py --checkpoint "$RUN/model.pt" --data-root "$WNN_DATA_ROOT" \
    --split-manifest "$SPLIT" --membership evaluation --output "${RUN}_eval"
  uv run --no-sync python scripts/score_sequence.py --data-root "$WNN_DATA_ROOT" \
    --split-manifest "$SPLIT" --predictions "${RUN}_eval" --output "$RUN/score.json"
done
```

Repeat with `CELL=f2_s1002` and `SPLIT="$WNN_SPLIT_F2"`. Export a teacher fit
cache only after selecting the teacher with the predeclared inner criteria, using
`predict_sequence.py --membership fit`; the separately budgeted KD students and
source-qualified caches remain an additional comparison. Full valid is exposed.
A TCN teacher score does not imply KD transfer or a ready CPU submission.

## Separately budgeted student transfer

After both teacher runs for one cell complete and their clean ancestry is
verified, export fitting-only caches. Then compare the same student trained
against GRU and TCN teachers. This is **four additional student fits** for two
cells, plus cache export and individual profiles, not included in the four
teacher fits. Both teachers were fixed-final; do not select using outer scores.

```bash
KD_PACKET=research/experiment_drafts/b300-transformer-distillation-runtime
for TEACHER in gru tcn; do
  TEACHER_RUN="$WNN_RUN_ROOT/d_${TEACHER}_${CELL}"
  CACHE="$WNN_RUN_ROOT/d_${TEACHER}_${CELL}_fit_cache"
  STUDENT="$WNN_RUN_ROOT/d_student_from_${TEACHER}_${CELL}"
  CONFIG="$KD_PACKET/configs/gru256_kd_${CELL}.json"
  uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    scripts/predict_sequence.py --checkpoint "$TEACHER_RUN/model.pt" \
    --data-root "$WNN_DATA_ROOT" --split-manifest "$SPLIT" --membership fit --output "$CACHE"
  uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    scripts/train_sequence.py --config "$CONFIG" --data-root "$WNN_DATA_ROOT" \
    --split-manifest "$SPLIT" --teacher-cache "$CACHE" \
    --output "${STUDENT}_profile" --profile-updates 8
done
```

Review and copy each profile receipt to its own `approved_profile.json` as
above, preserving identity. Then run:

```bash
for TEACHER in gru tcn; do
  CACHE="$WNN_RUN_ROOT/d_${TEACHER}_${CELL}_fit_cache"
  STUDENT="$WNN_RUN_ROOT/d_student_from_${TEACHER}_${CELL}"
  CONFIG="$KD_PACKET/configs/gru256_kd_${CELL}.json"
  uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    scripts/train_sequence.py --config "$CONFIG" --data-root "$WNN_DATA_ROOT" \
    --split-manifest "$SPLIT" --teacher-cache "$CACHE" --output "$STUDENT" \
    --profile-receipt "${STUDENT}_profile/approved_profile.json"
  uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    scripts/predict_sequence.py --checkpoint "$STUDENT/model.pt" --data-root "$WNN_DATA_ROOT" \
    --split-manifest "$SPLIT" --membership evaluation --output "${STUDENT}_eval"
  uv run --no-sync python scripts/score_sequence.py --data-root "$WNN_DATA_ROOT" \
    --split-manifest "$SPLIT" --predictions "${STUDENT}_eval" --output "$STUDENT/score.json"
done
```

These are output-KD students, with the same configuration, seed and data for
each teacher; hint loss is zero. The existing KD recipes use three epochs, not
the six-epoch teacher schedule. Exact cache acceptance, GPU profiles and KD
transfer have not been run. Check teacher-cache size, ancestry, and all results;
this still does not measure a complete CPU submission blend.
