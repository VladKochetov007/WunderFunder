# Kaggle read-only readiness and CPU grade handoff

**No kernel was uploaded or launched.** This receipt records live quota, the latest visible WNN kernel metadata, exact statuses for five recent kernels, and current dataset versions as of October 23, 2026. The machine-readable snapshot is [KAGGLE_READINESS_R1.json](KAGGLE_READINESS_R1.json).

## Live access and quota

The installed Kaggle CLI is 2.2.4. Read-only quota and data metadata calls succeeded through the existing account profile at 2026-10-23 08:35 UTC. GPU quota has 1,635.575 seconds unreserved (27m15.575s), and TPU quota has 72,000 seconds (20h), both refreshing at 2026-10-24 00:00 UTC. Provider reserved time is zero for both. The API does not return a CPU quota or an account-wide active/queued kernel roster. The five newest visible WNN runners from Oct 20–21 return `ERROR`; this does not establish the state of every account kernel or a CPU-only session. The quota API object and CLI agree on GPU used/total (106,364.425 / 108,000 seconds) once the API `timedelta` day field is included.

The local runner interface accepts a private CPU notebook through `kernel-metadata.json` with `enable_gpu=false`, `enable_internet=false`, an empty `machine_shape`, and dataset sources. The current CLI's `kaggle kernels push -p FOLDER [-t TIMEOUT]` uploads and starts the kernel. Its installed help has no `--no-run` option, so it was not invoked. Kaggle's [notebook documentation](https://www.kaggle.com/docs/notebooks) currently lists four CPU cores, 30 GB RAM, a 12-hour CPU session ceiling, and up to 20 GB of saved output under `/kaggle/working`. Account-specific CPU slot availability is not exposed by the queried API; the root-selected 1,800-second worker limit is a job hard cap, not a provider quota.

## Recommended CPU grading path

Root selected local fitting for the six residual-age heads and a separate Kaggle CPU grade after the six outputs and grade contract are frozen. This avoids fitting on Kaggle and avoids uploading the full 32,009,440,768-byte F0/F1 input-feature and parent-cache payloads.

The expected grade input is six candidate prediction arrays plus two fold parent prediction arrays (592,001,024 bytes total) and the frozen `cache/valid_targets.npz` (337,186,198 bytes; SHA256 `dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955`). Together these payloads are 929,187,222 bytes. Six head checkpoints and provenance/source maps, two parent sidecars, old cache sidecars plus the root cache-verification receipt, the frozen recipe/admission, and grader code/pins must fit within the remaining 70,812,778 bytes of the 1 GB bundle gate. The exact bundle size and upload time are not measured until root freezes the files. Do not inspect or compress candidate arrays before the independent grader binds their hashes.

The target NPZ is an explicit transport copy, not the original local path. Its frozen source identity includes the source path, SHA256, size, and source mtime. The remote adapter must validate the content pin and record the actual mounted path and transport mtime separately; it must not require copied-file mtime to equal the source mtime or fabricate that metadata. No raw Kaggle dataset mount is needed for this grade path. The account's public `vladkochetov07/wunder-funder` version 1 contains `valid.parquet` (5,615,509,128 bytes) and `valid_mask.parquet` (91,972,846 bytes), but that 5.7 GB route is unnecessary. The dataset file-list summary differs from the per-file sum by 1,186,464,470 bytes. No online parent-cache dataset exists in the current owner inventory.

The independent remote grader should verify all eight prediction payload hashes, NPY headers, exact order and source-qualified membership before opening labels. The root must first provide a local full-payload hash and coverage receipt binding both old cache payloads to their JSON sidecars. The remote report must label that evidence as a provenance attestation; the 32 GB cache arrays themselves do not need to be transferred. Do not include or score in-sample fit-parent prediction arrays.

| Phase | Current cost evidence |
|---|---|
| Stage data | Predictions plus target NPZ total 929,187,222 bytes; remaining metadata/source/checkpoint allowance is 70,812,778 bytes under the 1 GB gate. Upload/mount time is unmeasured. |
| Fit | 0 fits on Kaggle. |
| Export | 0 new prediction exports on Kaggle. |
| Check arrays | Hash and validate 592,001,024 bytes before label access; runtime is unmeasured. |
| Grade | Use the frozen independent grader on both full folds and both targets; runtime is unmeasured. Root's 1,800-second cap is the worker limit. |
| Collect | Return the small grade receipt, logs, and manifests; exact size/time are pending. |

For the eventual kernel, mount inputs read-only under `/kaggle/input`. Put writable outputs and all scratch at `/kaggle/working/disk_scratch`; set `TMPDIR`, `TMP`, `TEMP`, `XDG_CACHE_HOME`, `TORCHINDUCTOR_CACHE_DIR`, `NUMBA_CACHE_DIR`, and `POLARS_TEMP_DIR` there. Restrict affinity to two CPUs, set math/data-library threads to two, and stop before task RSS reaches 4,000,000,000 bytes. Never use `/tmp` or `/dev/shm`. Keep only the grade receipt/logs in final notebook output; the 20 GB output ceiling is ample for the <1 GB inputs, but it does not authorize caching or uploading the 32 GB feature arrays.

## Source and cache inventory

The current public dataset and private WNN assets were inspected by metadata only. The existing F1 parent model is a small local artifact: `model_ep3.pt` (3,981,215 bytes), provenance (163,160 bytes), propensity model (165,623 bytes) and provenance (21,648 bytes). The full `src/wnn` tree is 3,975,841 bytes. The two full parent caches are on physical Btrfs under `/home/vlad/development/wunder/research_tmp`; F1 alone totals 15,985,120,384 bytes. These were considered for a CPU cache-build path before root selected independent grading instead.

The old online-readout panel is terminal at 7/12 exported heads after 7,800.451 seconds, with five exports missing and no quality grade. It is preserved as historical evidence and is not a current Kaggle job or a completed new screen.

## Kaggle CPU fitting contingency

If local heat prevents the six selected head fits, CPU execution on Kaggle is a plausible but unadmitted fallback. Current Kaggle docs list 4 CPU cores, 30 GB RAM, a 12-hour CPU session, and 20 GB of saved `/kaggle/working` output; they separately describe additional scratch disk. Current dataset docs list 200 GB per dataset and 200 GB total private storage. A 32 GB four-sequence-cache dataset is below the per-dataset limit and can be mounted read-only under `/kaggle/input`; it must not be copied into the 20 GB saved-output area. [Notebook specifications](https://www.kaggle.com/docs/notebooks), [dataset specifications](https://www.kaggle.com/docs/datasets).

The saved Oct 23 metadata snapshot enumerated 21 owner datasets with no next page and six relevant private assets totaling 20,417,772 bytes, but it does not record the complete private-byte total. Therefore, account-wide remaining private storage is not fully verified. CPU session availability was also not exposed by the quota API; no new poll or upload was made.

Reusing all four existing 32 GB sequence arrays would avoid feature-cache rebuild time, but their recorded paths are local absolute paths. The Kaggle runner would need a content-bound read-only cache-root mapping and an explicit operational amendment so provenance retains both the original path and mounted transport path. In this path, the cache-build cost is already paid locally and must not be charged again. Root's local-derived estimate for three F1 heads is roughly 1,500 seconds; a ~3,000-second two-fold fit estimate assumes similar F0 cost, remains unmeasured on Kaggle, and excludes staging, grading, and collection. A separate raw-data rebuild route would add cache-generation time and needs its own estimate/admission. Neither path is currently admitted or a quality result.

## Commands and limits

Correction to the readiness snapshot: its earlier Kaggle-helper search was too narrow. The repository already has generic bundle, kernel bootstrap, and output collection tools in `scripts/kaggle_bundle.py`, `scripts/kaggle_kernel.py`, and `scripts/kaggle_collect_kernel_output.py`. I reviewed their actual interfaces. The bundle/bootstrap pair serves the existing GPU worker, so this CPU grade uses the small dedicated `kernel_driver.py`; the collection helper is reusable. This correction preserves the earlier snapshot as a dated record.

The exact quota calls, metadata/status results, dataset contents, runner-interface notes, and limits are in [KAGGLE_READINESS_R1.json](KAGGLE_READINESS_R1.json). Authenticated calls ran through `scripts/limited` at 25% CPU, 400 MiB RAM, zero swap, and 90 seconds, with temporary/cache paths under `/home/vlad/development/wunder/research_disk/b300-fast-20261023`. The initial local-only `--version` and `--help` checks occurred before that wrapper was applied and completed in under one second; all provider API calls were bounded. No candidate prediction value, Parquet content, or cache payload was opened, hashed, compressed, uploaded, or downloaded.

## No-launch packet builder

After all six candidate directories are complete and root freezes the grading admission, `build_grade_bundle.py` prepares the exact source map, sidecars, candidate outputs, cache-verification attestation/control chain, target NPZ, and grader dependency closure. It does not load NPY/NPZ arrays or contact Kaggle. The prepare phase checks the complete file inventory before hashing, enforces a one-gigabyte expanded-size gate, hashes in bounded chunks, and writes a deterministic manifest-first archive. Root then pins the manifest and issues the separate admission. Finalization verifies both and stages the private dataset and CPU-kernel directories under this physical-disk folder.

```sh
WNN_TMP_ROOT=/home/vlad/development/wunder/research_disk/b300-fast-20261023/scratch \
WNN_CPU_QUOTA=25% WNN_MEM_MAX=400M WNN_THREADS=2 WNN_RUNTIME_MAX=90s \
scripts/limited python /home/vlad/development/wunder/research_disk/b300-fast-20261023/kaggle_ops/build_grade_bundle.py \
  prepare --repo-root "$PWD"

WNN_TMP_ROOT=/home/vlad/development/wunder/research_disk/b300-fast-20261023/scratch \
WNN_CPU_QUOTA=25% WNN_MEM_MAX=400M WNN_THREADS=2 WNN_RUNTIME_MAX=90s \
scripts/limited python /home/vlad/development/wunder/research_disk/b300-fast-20261023/kaggle_ops/build_grade_bundle.py \
  finalize --repo-root "$PWD" \
  --prepared-dir /home/vlad/development/wunder/research_disk/b300-fast-20261023/kaggle_ops/prepared_bundle_r1 \
  --admission /path/to/root/GRADE_ADMISSION_R1.json
```

The finalized private dataset folder contains `grade_bundle.wnn`, `GRADE_ADMISSION_R1.json` as a separate read-only dataset input, and `dataset-metadata.json`. The manifest remains inside the archive to avoid a manifest/admission hash cycle. The kernel packet pins the archive and admission digests; extraction, temporary files, and caches use `/kaggle/working/disk_scratch`. Finalization is local-only. The documented dataset-create and kernel-push commands remain separate provider mutations and have not been run.

## Portable CPU profile amendment

The [root amendment](../KAGGLE_CPU_DELIVERY_AMENDMENT_R1.json) supersedes the local-fit route above because local thermal admission failed. [Suffix exporter](export_suffix_cache.py) and [profile packet builder](prepare_suffix_profile_bundle.py) provide exact column transport and manifest-first archive preparation. The [prepared bundle receipt](../PROFILE_BUNDLE_PREPARE_R2.json) and [file map](../PROFILE_BUNDLE_MANIFEST_R2.json) record the completed disk-only stage; fitting targets and weights are present only in the fit suffixes. No evaluator-label cache is included. The profile remains separate from scientific admission.

[Profile driver](profile_kernel_driver.py) binds the exact root admission, verifies and safely extracts the opaque archive, rebuilds only F0 and cleans generated large files on either success or failure. [Finalizer R2](prepare_suffix_profile_bundle_r2.py) fixes the original text-marker ambiguity by replacing only the AST-identified admission constant; prepared archive bytes stay unchanged. [Launch packet](../PROFILE_LAUNCH_PACKET_R2.json) pins the rendered kernel and metadata.
