# H512 fold-0 CP2 retained prediction export

This packet prepares one full-valid prediction export for the fixed, exposed-CV
comparison `0.75 * oldz4_h230_student_f0_s3_r3@ep1 + 0.25 * E529 H512 model_ep2`.
It does not authorize a launch or perform a score, fit, package build, or upload.
Root must create `ROOT_EXPORT_RELEASE_R1.json` after its own fresh GPU-lock,
resource, checkpoint, and output-collision admission. The helper refuses to run
in export mode without that exact release and the pinned source/contract hashes.

The selected H512 checkpoint is an intermediate prefix from a four-epoch run
that timed out during epoch index 3. Its epoch-2 checkpoint has already been
used in the fold-0 snapshot-curve comparison, so its 468 evaluation groups are
exposed. It has no optimizer state and must not be described as a completed
four-epoch endpoint. The full-valid prediction preserves the canonical 1,873
row-group order; a later fixed blend comparison may use only the 468 row-group
IDs recorded by the fold-0 manifest and their 141 components.

The producer is the exact, hash-pinned C337 `predict_run.py` snapshot used by
the earlier snapshot-curve consumer. The wrapper verifies the complete listed
source snapshot, imports the pinned trainer and predictor, and rebinds only
their `ROOT` values to the main project before touching data. Fit-time and
inference source trees differ; prior actual checkpoint consumption supports
compatibility, not source-tree identity. The bounded preflight checks the
actual valid parquet footer, the target archive's `row_group` order, existing
student prediction sidecar, source/producer receipts, and all output paths. It
calls the real predictor's `--help` parser only and verifies CUDA remains
uninitialized.

Run the no-launch preflight from the main checkout after this packet is
integrated:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s \
  scripts/limited /usr/bin/env \
  WNN_PROJECT_ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack \
  WNN_LOADER_WORKERS=0 WNN_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0 \
  PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python \
  research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/e529_muon512_f0_cp2_export_r1/run_e529_retained_export_r1.py \
  --preflight-only
```

Root's future release must use the declared schema and status, set
`launch_authorized` to `true`, and copy every producer, source, fold, output,
blend, and resource pin from `EXPORT_CONTRACT_R1.json`. The prepared export argv
uses a 900-second TERM/KILL-bounded worker, one CPU/GPU, zero loader workers,
one thread, no swap, a 6.8 GB reservation, 4.3 GB allocator budget, and 5 GB
observed VRAM ceiling. The prior 404-second profile covered three checkpoint
predictions plus scoring; it is not an isolated-export timing estimate.

After a root-authorized export, the helper checks the complete NPY header,
canonical row order, finiteness, exact checkpoint/split sidecar, and computes
the NPY/sidecar hashes. It does not score the single model or the fixed blend.
