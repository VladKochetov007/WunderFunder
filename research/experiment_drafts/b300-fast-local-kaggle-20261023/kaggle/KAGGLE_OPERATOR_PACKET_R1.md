# CPU grade operator packet

This is a no-launch packet template for the admitted full-panel grade. The
`build_grade_bundle.py` finalization phase verifies the root admission, patches
the four transport constants in [`kernel_driver.py`](kernel_driver.py), and
stages upload folders. The current files contain no candidate artifacts and
make no provider changes.

## Bundle layout

Build one private dataset upload directory with only:

- `grade_bundle.wnn`: gzip tar with `bundle_manifest.json` as the first member,
  followed by regular files declared in that manifest. Do not add directory,
  link, alias, undeclared, or credential-like archive members. The manifest
  entries use the grader schema `{source_path,bundle_path,sha256,bytes}`.
- `GRADE_ADMISSION_R1.json`: separate root-issued read-only input, intentionally
  outside the bundle manifest to avoid a hash cycle.
- `dataset-metadata.json`: private dataset metadata with the final ID.

Using one opaque `.wnn` file lets the dataset upload use `--dir-mode skip`
without dropping nested `src/`, `scripts/`, `research/`, or `grader/` files.
The previous B16 upload diagnosis found that this mode skipped a manifest when
the staging code copied only the archive and metadata. Here, the grade manifest
is inside the root-pinned archive and is extracted by the driver. The grader
adapter then verifies the manifest digest against the root admission. Its CLI is
run exactly as follows (paths are built from the extracted root):

```text
python grader/age_readout_grade.py \
  --bundle-root <extracted-root> \
  --bundle-manifest <extracted-root>/bundle_manifest.json \
  --admission /kaggle/input/<dataset-slug>/GRADE_ADMISSION_R1.json \
  --admission-sha256 <root-pinned-admission-sha256> \
  --output <fresh-path-under-/kaggle/working/disk_scratch>
```

The grader adapter is still being finalized. The builder derives and packages
its exact adapter and dependency hashes; root's admission binds those hashes.
Root's final admission must also bind the four grader dependencies, all candidate
and parent predictions/sidecars, the local cache-verification attestation and
its control chain, target NPZ, both split/subset manifests, parent milestone,
the 19 producer-source files, each parent provenance file, recipe, and source
maps/checkpoints. The adapter contract is fail-closed; this template does not
substitute for admission.

## Kernel behavior

[`kernel_driver.py`](kernel_driver.py) checks the frozen archive digest, requires
the manifest first, rejects unsafe or undeclared archive entries, and extracts
only regular files under `/kaggle/working/disk_scratch`. It makes the extracted
bundle read-only and runs the grader directly from that root. It sets temporary
and common cache directories under `disk_scratch`, disables GPU visibility,
uses two-CPU affinity and two-thread library settings, and monitors process-tree
RSS against 4,000,000,000 bytes. On success it removes only the extracted bundle
and the run's temporary/cache trees, leaving the small report, logs, and driver
receipt for Kaggle output collection. On failure, scratch and logs are retained.
The grader gets the remaining time within a
1,800-second total job cap, with a 60-second finalization reserve. Stdout,
stderr, the grader report, and a small driver receipt stay under the same disk
root. Nonzero exits and watchdog stops keep their logs and receipt.

The kernel metadata template sets `enable_gpu=false`, `enable_tpu=false`,
`enable_internet=false`, `is_private=true`, and an empty CPU `machine_shape`.
Its only dataset source is the private grade bundle; no raw Parquet or 32 GB
feature-cache mount is needed. The finalizer copies the pinned driver and final
metadata into a clean kernel directory; do not push this operations directory.

## Provider commands after root admission

These commands are documented only; none has been run. `datasets create` is
private by default and `dataset-metadata.json` also marks the dataset private.
Use a fresh upload folder containing only the three files above, then verify the
remote file listing and archive size before preparing the kernel.

```sh
kaggle datasets create --path <fresh-dataset-upload-dir> --dir-mode skip
kaggle kernels push -p <root-admitted-kernel-dir>
kaggle kernels status vladkochetov07/wnn-age-readout-grade-20261023-r1-runner
```

`kernels push` uploads and starts the notebook; this CLI version has no
`--no-run` flag. Root therefore launches only after final admission. For
collection, use a new disk-backed destination and the existing one-shot helper;
it polls status and downloads only after a terminal status:

```sh
scripts/limited python scripts/kaggle_collect_kernel_output.py \
  --kernel vladkochetov07/wnn-age-readout-grade-20261023-r1-runner \
  --dest /home/vlad/development/wunder/research_disk/b300-fast-20261023/kaggle_ops/collected/<fresh-run-id> \
  --max-bytes 1200000000 --min-free-bytes 20000000000
```

The helper invokes `kaggle kernels status` once per call. On terminal state it
runs `kaggle kernels output <owner/kernel> -p <dest>/outputs.partial`, hashes
downloaded files into `MANIFEST.json`, renames the staging directory, and writes
`COLLECTED.json`. Use a fresh destination: its retry path removes an existing
partial directory and final output directory. The direct CLI equivalents are
`kaggle kernels status <owner/kernel>` and
`kaggle kernels output <owner/kernel> -p <fresh-destination>`.

## Readiness cost

No provider upload, launch, reservation, or data download occurred while making
this packet. Final bounded checks plus initial builder checks took 0.68 measured
seconds: Ruff formatting (0.05s across two passes), Ruff check (0.09s across two
passes), Python compile (0.25s across two passes), builder CLI help (0.07s), and
metadata-only builder preflight (0.22s). One earlier lint attempt took 0.03s and
found style issues that were fixed, for 0.71 measured seconds total. The preflight stopped because all
six candidate output directories are still empty; it did not hash candidate
arrays or cache payloads and created no package directory. Brief read-only
source/metadata inspections and the earlier format-correction attempt were not
independently timed. Archive construction, upload/mount, CPU grading, and
collection costs remain unmeasured. On success, scratch cleanup is intended to
leave a small output; the 1.2 GB collection ceiling also accommodates most
bounded failure bundles plus logs. The remote grade has zero fits and zero new
prediction exports.
