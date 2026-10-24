# Portable CPU profile operator packet R2

This packet records the source-only CPU profile handoff for the age-readout
screen. Root owns Kaggle admission, upload, execution, and result collection.

## Frozen local inputs

- Profile driver template: `profile_kernel_driver.py`, SHA256
  `c24750358eec2f2215fa8892cc131a7dfadabc6ca1648b21febe836df411b959`.
- Root profile admission: `PROFILE_F0_CPU_R2_ADMISSION.json`, SHA256
  `15b0f04dbda3bc1b8c1db1254fcf84cae7b6e3b1343b40e7f056a47bb6f5c890`.
- Prepared opaque archive: `age_actor_bundle.wnn`, 1,143,552,000 bytes;
  SHA256 `971871550f50be91c5afe1134e1012a3ba3b9c94e2157b4f8b6ed17a62442ef8`.
- Bundle manifest SHA256
  `ea241506eadd0a34f437c699bab3b7784a69b818b710e3d016c9b3599774f99e`.
- Frozen actor source map SHA256
  `939e8a00a003d9299a397cac80753df67a56b65b34ed3bd0e8dce6726b109523`.
- Successful source-only finalizer output kernel SHA256
  `098e0a93a29a75485c1e40cdeef57a94212d246fe1cad806cd01a129a7ef6da5`.

`PROFILE_DRIVER_FIXTURE_R2.json` records the bounded synthetic driver and
finalizer checks. They used a 25% CPU quota, one thread, a 400 MB memory cap,
zero swap, and physical disk scratch. The success and timeout paths each
removed the extracted bundle and generated cache/temp/run/resume trees while
retaining the logs and driver receipt. No real cache values, labels, model
fits, Kaggle API calls, uploads, or kernel launches were part of these checks.
Root's separate launch receipt records the active profile kernel as
`vladkochetov07/wnn-age-readout-f0-cpu-profile-20261023-r2` and the private
input dataset as `vladkochetov07/wnn-age-cpu-profile-20261023-r2`. Kaggle
normalized the kernel slug from the requested title-derived ID; use this
observed canonical kernel reference for status and collection, and do not
push the kernel again.

## Output collection

The input archive is mounted below `/kaggle/input`; it is not in the kernel's
working output. The driver removes the extracted archive tree on success and
failure. It also clears partial extraction, cache, temp, run, and resume trees.
The retained output is limited to the profile JSON receipt, driver receipt,
stdout, and stderr. Therefore the existing one-shot collector can download the
small kernel output set without copying the input archive.

After the root-owned kernel reaches a terminal state, collect to a fresh
physical-disk directory with the existing helper. The tighter limits below
allow for incidental notebook metadata while keeping the download bounded:

```sh
WNN_TMP_ROOT=/home/vlad/development/wunder/research_disk/b300-fast-20261023/scratch \
WNN_CPU_QUOTA=25% WNN_MEM_MAX=400M WNN_THREADS=1 \
scripts/limited .venv/bin/python scripts/kaggle_collect_kernel_output.py \
  --kernel vladkochetov07/wnn-age-readout-f0-cpu-profile-20261023-r2 \
  --dest /home/vlad/development/wunder/research_disk/b300-fast-20261023/kaggle_ops/profile_outputs_r2 \
  --max-bytes 100000000 \
  --min-free-bytes 1000000000
```

The helper checks terminal status, downloads to `outputs.partial`, enforces the
byte cap, hashes the files, and writes its collection marker. Kaggle's output
command returns the kernel output set as a whole; the input archive is absent
from that set because it stays under `/kaggle/input` and the driver deletes
its working extraction on every terminal path.
