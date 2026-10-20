# Post-Fit Package and Replay Commands

Preparation only. None of these commands has been run from this manifest. They
do not assert checkpoint availability, model quality, or measured serving speed.
Use the existing physical scratch disk for all outputs. Do not stage outputs in
`/tmp` or `/dev/shm`.

## Common paths and resource wrapper

Run from the integrated repository after the active fit releases its resources:

```bash
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack
SCRATCH=/home/vlad/development/wunder/research_tmp
NATIVE_BUILT=$SCRATCH/overnight-ema-native-20261020-artifacts/native-built
RUNS=$ROOT/artifacts/runs
OUT=$SCRATCH/overnight-ema-candidate-packages-20261020
CHECKS=$OUT/checks
mkdir -p "$OUT" "$CHECKS"
cd "$ROOT"

limited() {
  WNN_MEM_MAX=2000000000 WNN_CPU_QUOTA=100% WNN_THREADS=1 \
    WNN_TMP_ROOT="$SCRATCH" "$ROOT/scripts/limited" "$@"
}
```

Every build, Python export, ABI check, replay, and Docker gate below runs through
`limited`: one CPU, 2,000,000,000 bytes of memory, zero swap, one numerical
thread, and physical-disk scratch. The Python 3.11 container repeats the 2 GB
cap with `--memory=2g --memory-swap=2g`.

## Add the required AVX2 member

The frozen native source tree used for v2/v3 is
`32a1ddd91af86913cda40e113029d7bb283b806f`. Check that
`git rev-parse HEAD:native` still returns this tree. Build AVX2 as the first
post-fit native preparation step:

```bash
limited /usr/bin/time -v env \
  WNN_BUILD_CPUS=1 WNN_BUILD_MEM=2000000000 \
  WNN_USE_EXISTING_BUILD_IMAGE=1 WNN_BUILD_LEVELS=avx2 \
  "$ROOT/scripts/build_native.sh" "$NATIVE_BUILT"
```

This creates `native-built/avx2/wnn_native.abi3.so` and reruns the static ABI
gate over v2/avx2/v3. The packager requires all three directories. The current
`solution_multi.py` dispatch uses v2 or v3; do not copy another binary under an
incorrect AVX2 label.

## Export only the selected candidate

The two candidate checkpoints are
`overnight_nll_f0_s3_nll_e2_r1@ep1` and
`overnight_nll_f0_s3_wmse_e2_r1@ep1`. The optional fixed blend is
`oldz4_h230_student_f0_s3_r3@ep1:0.75` with
`overnight_nll_f0_s3_nll_e2_r1@ep1:0.25`. Select the model and precision from
the fit evidence first; the commands below package only the explicitly chosen
endpoint. There is no six-package candidate/precision loop.

For an ordinary 211-input H256×2 member, the exporter should record
`input_width: 211` and `representation_mode: none`. The optional blend has the
same 211-input representation. The exporter loads the checkpoint on CPU,
requires provenance, retains the first two output rows, and rejects unsupported
feature modes. Confirm that the exact selected run directory, checkpoint, and
provenance sidecar exist before export. `solution_multi.py` reads package
precision metadata and supports `q16x` and `q8vnni`.

Set `NAME`, `PRECISION`, and `RUN_REFS` to one quality-selected endpoint, then
run this once for that endpoint:

```bash
NAME=nll-h256-e2-q16x
PRECISION=q16x
RUN_REFS=('overnight_nll_f0_s3_nll_e2_r1@ep1:1.0')
limited env PYTHONPATH="$ROOT/src" "$ROOT/.venv/bin/python" \
  "$ROOT/scripts/package_submission.py" "$NAME" "${RUN_REFS[@]}" \
  --runs-dir "$RUNS" --native-from "$NATIVE_BUILT" \
  --solution "$ROOT/submission/solution_multi.py" \
  --precision "$PRECISION" --require-provenance --normalized-coefficients \
  --output-zip "$OUT/$NAME.zip" --receipt "$OUT/$NAME.build.json" \
  --evidence-note "Fixed checkpoint export; packaging is not a quality or speed claim."
```

For the WMSE member, set `RUN_REFS=('overnight_nll_f0_s3_wmse_e2_r1@ep1:1.0')`.
For the optional blend, set
`RUN_REFS=('oldz4_h230_student_f0_s3_r3@ep1:0.75' 'overnight_nll_f0_s3_nll_e2_r1@ep1:0.25')`.
Use a distinct `NAME` and `PRECISION` only when evidence calls for another
package. The packager is create-only, checks the 20,000,000-byte ZIP limit, and
records checkpoint, tensor, native-library, and ZIP hashes.

## Static ABI and Python 3.11 callback gate

After the AVX2 build and selected package, inspect their actual bytes:

```bash
limited "$ROOT/.venv/bin/python" "$ROOT/scripts/check_submission_abi.py" "$NATIVE_BUILT"
limited "$ROOT/.venv/bin/python" "$ROOT/scripts/check_submission_abi.py" "$OUT/$NAME.zip"
```

These checks inspect ELF/ABI metadata; they do not run the callback. For the
Python 3.11 callback gate, use the already-present
`$ROOT/cache/valid_subset20.parquet`; do not make another data copy. For each
native level v2 and v3, save a bounded reference replay from the actual ZIP and
compare it in the offline Python 3.11 image:

```bash
PARITY_DATA=$ROOT/cache/valid_subset20.parquet
ZIP=$OUT/$NAME.zip
for LEVEL in v2 v3; do
  REFERENCE=$CHECKS/$NAME.$LEVEL.reference.npz
  limited env WNN_NATIVE_LEVEL="$LEVEL" PYTHONPATH="$ROOT/src" \
    "$ROOT/.venv/bin/python" "$ROOT/scripts/verify_submission.py" \
    "$ZIP" "$PARITY_DATA" "$REFERENCE"
  limited docker run --rm --network none --cpus=1 --memory=2g --memory-swap=2g \
    --pids-limit=128 \
    -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void \
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
    -e PYTHONPATH= -e LD_LIBRARY_PATH= -e TMPDIR=/checks \
    -v "$ROOT:/workspace:ro" -v "$OUT:/artifacts:ro" \
    -v "$CHECKS:/checks:rw" -w /workspace \
    wnn-python311-submission-check:latest \
    python scripts/check_submission_python311.py \
      --zip "/artifacts/$NAME.zip" \
      --data /workspace/cache/valid_subset20.parquet \
      --level "$LEVEL" \
      --reference "/checks/$NAME.$LEVEL.reference.npz" \
      --output "/checks/$NAME.$LEVEL.python311.json"
done
```

Use fresh `NAME`/output paths; both verifier artifacts and callback receipts are
create-only. The offline gate requires Python 3.11, clean library/Python paths,
two full sequences, finite `(2,)` float32 callback values, agreement within
`1e-6` with the reference, a reset-prefix replay, and the requested native level.
This is packaging/runtime compatibility evidence, not a quality comparison.

Do not use `scripts/benchmark_submission_callbacks.py` here: it expects
`model.model` and the legacy receipt schema and only covers q8, so it does not
match `solution_multi.py` packages. `verify_submission.py` calls the actual
submission callback and provides the needed row-time measurement.

## One source-qualified official-metric replay

Use the fit's frozen manifest at
`research/campaign_2026-09-29/artifacts/splits/fold0.json`. The verifier calls
`load_campaign_split(SPLIT, ROOT)`, checks all manifest sources, requires the exact
`datasets/valid.parquet`, validates each selected row group's `seq_ix`, and
replays `evaluation_groups` in manifest order. Predictions, targets, and mask
are streamed through bounded temporary memmaps. The receipt contains the
ordered row groups, source path, split SHA-256, sequence IDs, rounded log fields,
and the full unrounded `GlobalAccumulator.result()` value. This fold is
evaluation exposure for its fitted models; it is not a fresh holdout.

After one fixed candidate and native level are selected, run this once for that
served endpoint:

```bash
SPLIT=$ROOT/research/campaign_2026-09-29/artifacts/splits/fold0.json
LEVEL=v3
limited env WNN_NATIVE_LEVEL="$LEVEL" PYTHONPATH="$ROOT/src" \
  "$ROOT/.venv/bin/python" "$ROOT/scripts/verify_submission.py" \
  "$OUT/$NAME.zip" "$ROOT/datasets/valid.parquet" \
  "$OUT/$NAME.$LEVEL.fold0.npz" \
  --campaign-split-manifest "$SPLIT" \
  --receipt "$OUT/$NAME.$LEVEL.fold0.json"
```

This is the callback replay and official accumulator for the exact selected
source-qualified fold. Do not repeat it with the full `valid.parquet` or run
`scripts/native_validation_score.py` on the same ZIP as a duplicate scorer.
The subset20 Python 3.11 gate above is separate packaging compatibility only.
The verifier writes through memmaps; for 468 groups the prediction, target, and
mask payload is approximately 159 MB before NPZ framing. Keep it on physical
scratch. The earlier estimate of 669 MB applies to every validation row group,
which this selected-fold replay does not read.

## Analytic payload and runtime costs

Assuming standard `RNNModel`, linear 211-to-H encoder, two GRU layers,
`encoder_width = recurrent_width = readout_width = H`, one hidden readout layer,
and an exported final head truncated to two targets:

| Export | Float32 values, including `mean`/`std` | Raw tensor bytes |
|---|---:|---:|
| H256×2, 211 inputs | 976,040 | 3,904,160 B (3.72 MiB) |
| H230×2, 211 inputs | 793,234 | 3,172,936 B (3.03 MiB) |
| H230 + H256 blend | 1,769,274 | 7,077,096 B (6.75 MiB) |

The built v2/v3 libraries add 2,413,408 bytes; the required AVX2 library adds
about another 1.2 MB at the current build scale. Exact ZIP size is known after
export. Check checkpoint arguments and `members.json` before relying on these
counts. Loading `.pt` checkpoints uses Torch and can require substantially more
temporary memory than the exported model. Approximate dense MACs per row are
972,032 for H256×2 and 789,590 for H230×2. The H230/H256 blend performs about
1,761,622 MACs per row (1.81× H256 alone); its 75%/25% weights do not skip either
predictor. q16x/q8vnni speed and quality remain measured questions.
