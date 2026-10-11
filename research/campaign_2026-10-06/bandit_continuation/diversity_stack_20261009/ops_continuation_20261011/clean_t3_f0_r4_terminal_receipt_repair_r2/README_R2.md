# R4 terminal receipt closure R2

This identified repair closes the generic terminal receipt for the already
downloaded `attempt_001`. It does not contact Kaggle, redownload files, retry the
kernel, or authorize fitting. The frozen R1 collector remains unchanged. R1
successfully downloaded the terminal logs and outputs, then failed while
building its generic receipt because the R4 target omitted the required
`root_admission_sha256` field.

The R2 helper binds the exact terminal-download admission
`3eee6881ec10d8d4b18b5769733982aa9baf47fa28e5a10cbed8698a310e546e`, the
existing accepted push receipt, the frozen R1 collector bytes, the reviewed R2
generic collector, the predownload 108-name inventory and the same existing
attempt. It refuses to run if either R2 output receipt already exists. It then
checks the actual provider inventory and response log using the reviewed
generic consumer before writing two exclusive-create receipts.

The helper was first run without arguments for a read-only validation pass and
then once with `--write`. The exact bounded execution used MAIN's limited runner
at 50% CPU, one thread, 700,000,000-byte memory maximum and 120 seconds:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=700000000 WNN_RUNTIME_MAX=120s \
/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited \
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
/home/vlad/development/wunder/wnn_r4_terminal_collector_20261012/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_r4_terminal_receipt_repair_r2/close_r4_terminal_attempt_r2.py --write
```

The actual attempt passes terminal file verification: 108 provider files match
the downloaded files, the CLI response log matches, and both recorded download
commands exited zero. The R4 pipeline itself ended with
`ValueError: Worker reservation exceeds host operational budget` after its
paired-profile stage. No teacher fit or export stage started. The paired T4
guard passed, with 9,252,970,496 bytes summed worker peak RAM and 3,497,000,960
bytes peak VRAM per worker. The seed-3 profile completed 200/200 optimizer
updates; seed 4 completed 199/200, with one AMP skip and one nonfinite-gradient
step. This is startup/resource evidence, not a model-quality result.

The original attempt and its provider files are retained under the MAIN
collection directory. R2 adds only `TERMINAL_COLLECTION_RECEIPT_R2.json` and
`R4_TERMINAL_DIAGNOSTICS_R2.json`; the corresponding hashes and paths are
recorded in `RECEIPT_REPAIR_RUN_R2.json`.
