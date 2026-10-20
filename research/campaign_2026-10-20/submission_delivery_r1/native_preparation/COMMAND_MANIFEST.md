# Post-Fit Package and Replay Commands

Preparation only. These commands were not run. No checkpoints were loaded, no
Torch process was started, and no dataset was read for this manifest.

## Inputs and native levels

Run from the integrated repository after the GPU fits release memory:

```bash
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack
SCRATCH=/home/vlad/development/wunder/research_tmp
NATIVE_BUILT=$SCRATCH/overnight-ema-native-20261020-artifacts/native-built
RUNS=$ROOT/artifacts/runs
OUT=$SCRATCH/overnight-ema-candidate-packages-20261020
CHECKS=$OUT/checks
mkdir -p "$OUT" "$CHECKS"
cd "$ROOT"
```

The packager requires all three directories `v2`, `avx2`, and `v3`. The frozen
native source tree used for v2/v3 is
`32a1ddd91af86913cda40e113029d7bb283b806f`. Confirm that
`git rev-parse HEAD:native` still returns this tree before building AVX2. Then
run this as the first post-fit preparation step:

```bash
WNN_MEM_MAX=2000000000 WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_TMP_ROOT="$SCRATCH" WNN_BUILD_CPUS=1 WNN_BUILD_MEM=2000000000 \
WNN_USE_EXISTING_BUILD_IMAGE=1 WNN_BUILD_LEVELS=avx2 \
scripts/limited /usr/bin/time -v scripts/build_native.sh "$NATIVE_BUILT"
```

This adds `native-built/avx2/wnn_native.abi3.so` and reruns the static ABI gate
over the v2/avx2/v3 directory. The ordinary `solution_multi.py` dispatch loads
v2 or v3; the packager still requires the AVX2 member in every ZIP. Do not fill
that path with a differently labeled binary.

## Fixed exports

`package_submission.py` loads `model_ep1.pt`, requires a provenance sidecar when
`--require-provenance` is set, exports float32 model tensors, and truncates the
final head to its first two output rows. The intended NLL and WMSE checkpoints
are `overnight_nll_f0_s3_nll_e2_r1@ep1` and
`overnight_nll_f0_s3_wmse_e2_r1@ep1`; the optional blend uses
`oldz4_h230_student_f0_s3_r3@ep1` at 0.75 and the new NLL checkpoint at 0.25.
The exporter must report `input_width: 211` and `representation_mode: none` for
each member. It will fail if a named checkpoint or provenance file is absent or
if the model is outside its native contract.

Use the new `solution_multi.py` from the integrated repository. It reads the
package's `precision.txt` and passes `q16x` or `q8vnni` into the native
predictors. Separate ZIPs keep weights and coefficients fixed while varying
precision:

```bash
package_one() {
  local name="$1" precision="$2"
  shift 2
  PYTHONPATH=src scripts/limited .venv/bin/python scripts/package_submission.py \
    "$name" "$@" \
    --runs-dir "$RUNS" \
    --native-from "$NATIVE_BUILT" \
    --solution "$ROOT/submission/solution_multi.py" \
    --precision "$precision" \
    --require-provenance --normalized-coefficients \
    --output-zip "$OUT/$name.zip" \
    --receipt "$OUT/$name.build.json" \
    --evidence-note "Fixed checkpoint export; no quality or serving claim from packaging."
}

for precision in q16x q8vnni; do
  package_one "nll-h256-e2-$precision" "$precision" \
    'overnight_nll_f0_s3_nll_e2_r1@ep1:1.0'
  package_one "wmse-h256-e2-$precision" "$precision" \
    'overnight_nll_f0_s3_wmse_e2_r1@ep1:1.0'
  package_one "oldz4-75-newnll-25-$precision" "$precision" \
    'oldz4_h230_student_f0_s3_r3@ep1:0.75' \
    'overnight_nll_f0_s3_nll_e2_r1@ep1:0.25'
done
```

The run directory spelling above is based on the parent task IDs and visible
run-directory names; it was not checked by opening checkpoint files. Confirm
all three exact run directories and `model_ep1.pt` files at handoff. The packager
refuses to overwrite ZIPs or receipts. It also enforces a 20,000,000-byte ZIP
limit and records hashes for source checkpoints, exported tensors, native
libraries, and the final ZIP.

## Static ABI and Python 3.11 callback gates

After the AVX2 build and each package:

```bash
python3 scripts/check_submission_abi.py "$NATIVE_BUILT"
python3 scripts/check_submission_abi.py "$OUT/nll-h256-e2-q16x.zip"
```

The first command should report all three libraries; the second inspects the
actual ZIP bytes. This is a static ELF check only. The package's copied
`avx2` binary is part of the ZIP contract even though `solution_multi.py`
currently dispatches directly to v2/v3.

For the Python 3.11 callback gate, create a reference prediction file from the
same ZIP and precision on a fixed 20-sequence subset, then import and replay
the ZIP in the existing offline image. Repeat for each candidate/precision and
for both levels. The reference replay is a packaging control; it is not an
independent quality measurement.

```bash
PARITY_DATA=$ROOT/cache/valid_subset20.parquet
NAME=nll-h256-e2-q16x
ZIP=$OUT/$NAME.zip
for level in v2 v3; do
  REFERENCE=$OUT/$NAME.$level.reference.npz
  WNN_NATIVE_LEVEL="$level" PYTHONPATH=src \
    WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2000000000 \
    WNN_TMP_ROOT="$SCRATCH" \
    scripts/limited .venv/bin/python scripts/verify_submission.py \
      "$ZIP" "$PARITY_DATA" "$REFERENCE"
  docker run --rm --network none --cpus=1 --memory=1g --memory-swap=1g \
    --pids-limit=128 \
    -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void \
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
    -e PYTHONPATH= -e LD_LIBRARY_PATH= -e TMPDIR=/checks \
    -v "$ROOT:/workspace:ro" \
    -v "$OUT:/artifacts:ro" -v "$CHECKS:/checks:rw" \
    -w /workspace wnn-python311-submission-check:latest \
    python scripts/check_submission_python311.py \
      --zip "/artifacts/$NAME.zip" \
      --data "/workspace/cache/valid_subset20.parquet" \
      --level "$level" \
      --reference "/artifacts/$NAME.$level.reference.npz" \
      --output "/checks/$NAME.$level.python311.json"
done
```

`check_submission_python311.py` requires Python 3.11, no `PYTHONPATH` or
`LD_LIBRARY_PATH` override in the container, two complete 20,000-row groups,
finite `(2,)` float32 callback outputs, and agreement within `1e-6` against the
reference. It also replays the first sequence prefix after the second to check
reset behavior and checks which level was imported. The Docker caps set swap to
zero (`--memory-swap` equals `--memory`). If the named subset is unavailable,
use another fixed subset of at least two full sequences and generate the
reference from exactly that same file.

## Native local official-metric replay

`verify_submission.py` replays every row of its input through the ZIP callback,
checks the warmup/finite-output contract, streams the repository's official
`global_wp_clip2_v1` accumulator, and stores predictions, targets, and masks.
Use the subset above only for a small runtime screen. For quality evidence, use
the source-qualified validation data authorized for the checkpoint, and label
any overlap with fitting membership. A full `valid.parquet` replay is not a
fresh holdout when its sequences were used for fitting or selection.

The same command with the full validation file (run once per fixed candidate,
precision, and selected ISA level) is:

```bash
NAME=nll-h256-e2-q16x
LEVEL=v3
WNN_NATIVE_LEVEL="$LEVEL" PYTHONPATH=src \
  WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2000000000 \
  WNN_TMP_ROOT="$SCRATCH" \
  scripts/limited .venv/bin/python scripts/verify_submission.py \
    "$OUT/$NAME.zip" "$ROOT/datasets/valid.parquet" \
    "$OUT/$NAME.$LEVEL.full-predictions.npz"
```

The output reports `global_wp_clip2_v1`, per-target scores, load time, and
microseconds per row. About 39.4 million rows take roughly 47 minutes at
71 microseconds per row; actual speed must be measured on the final package.
The prediction/target/mask arrays total about 669 MB; `verify_submission.py`
writes them with uncompressed `np.savez`. Keep outputs under physical scratch.

After a final ZIP is fixed and hash-pinned, the existing scorer-wrapper replay
is:

```bash
ZIP_SHA256=<sha256-of-final-zip>
PYTHONPATH=src WNN_NATIVE_LEVEL=v3 WNN_CPU_QUOTA=100% WNN_THREADS=1 \
  WNN_MEM_MAX=2000000000 WNN_TMP_ROOT="$SCRATCH" \
  scripts/limited .venv/bin/python scripts/native_validation_score.py \
    --submission "$OUT/final.zip" \
    --data "$ROOT/datasets/valid.parquet" \
    --expected-sha256 "$ZIP_SHA256" \
    --output "$OUT/final.official-score.json"
```

## Analytic payload and runtime costs

Assuming the standard `RNNModel`, linear 211-to-H encoder, two GRU layers,
`encoder_width = recurrent_width = readout_width = H`, one hidden readout layer,
and the exporter retaining only the first two final-head rows:

| Export | Float32 values, including `mean`/`std` | Raw tensor bytes |
|---|---:|---:|
| H256×2, 211 inputs | 976,040 | 3,904,160 B (3.72 MiB) |
| H230×2, 211 inputs | 793,234 | 3,172,936 B (3.03 MiB) |
| H230 + H256 blend | 1,769,274 | 7,077,096 B (6.75 MiB) |

The built v2/v3 libraries add 2,413,408 bytes. The required AVX2 library adds
about another 1.2 MB at the current build scale; this plus both models is
comfortably below the packager's 20 MB ZIP limit before metadata, but the exact
ZIP size is known only after export. Check actual checkpoint arguments and
`members.json` before relying on these counts. Checkpoint `.pt` loading uses
Torch and can use substantially more temporary memory than the exported model
because it loads the full training checkpoint before selecting model tensors.

The approximate dense multiply-accumulate work per row is 972,032 for H256×2
and 789,590 for H230×2 under the same assumptions. A two-member H230/H256 blend
therefore performs about 1,761,622 MACs per row, approximately 1.81× the NLL
model alone. The 75%/25% coefficients do not skip either predictor. q16x/q8vnni
runtime and prediction quality remain measured questions; this manifest makes
no speed or score claim.
