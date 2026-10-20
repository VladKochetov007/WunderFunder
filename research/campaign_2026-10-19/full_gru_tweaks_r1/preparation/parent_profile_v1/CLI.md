# Parent profile admission

`REMOTE_SUBMISSION_TEMPLATE.py` is the unbound source template and exits before side effects while cache receipts are null. `REMOTE_SUBMISSION.py` is the rendered, cache-receipt-bound body. It does not launch merely because the metadata binding is complete: root still owns the fresh environment gates, independent review, and the only profile launch.

The rendered body binds the accepted cache CPU V2 unit `computer-vision-homework-fg19-teacher-cache-cpu-build-v2.service`, invocation `a358e89cd6fe42118e426a954ee897f0`, acceptance SHA-256 `88b75986538ecf4c7255705c173d5ac16946dd5ca44e9c4c69c331a381b5bec7`, exact-stop SHA-256 `286601bdf82dd19c94226ae06d5bef4841b59061c66e115c6fccb71370788ea4`, and metadata SHA-256 `03d9cf1fbd20b8bb3ad7b7d0c4513293fd1d5128e816256e4bf3cdfbfa4b43e3`. Root has staged the two receipt JSON files at the corresponding `B/research/full_gru_tweaks_r1/cache_cpu_v2/` paths. Both output arrays are bound by captured stat identity only.

The template checks the published 167-file source-68 manifest, the four staged teacher-cache metadata JSONs, current registered F0 data identities, the pinned F0 split JSON, cache metadata bytes and output-array stats, parent/worker headroom, selected-GPU ownership/free reserve, physical scratch, and absence of the new profile destinations/unit. It does not read model/data/cache arrays or hash array payloads.

Membership checks compare cache `fit_groups.train` and `fit_groups.valid` to the canonical split independently by namespace. Numeric group IDs may overlap across train and valid namespaces; only fit-valid versus outer-valid must be disjoint. Each output `row_groups` list must have unique membership equal to its namespace’s fit set. The canonical trainer checks its exact row order against the target-index arrays in the profile unit before profile updates; preflight does not claim or verify this order.

- Source: `/home/grinchenkoeg/computer-vision-homework-20261018/source-fg19-cache-68ff6829`, commit `68ff682977b822f38351aac0f621146f5f13b440`, 167 files, manifest SHA-256 `a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621`.
- Project root: `/home/grinchenkoeg/computer-vision-homework-20261018/data`; interpreter `/home/grinchenkoeg/WunderFunder/.venv/bin/python`; `PYTHONPATH` is the source `src` directory.
- Cache: `/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_t3_fit_homogeneous_cache_v1`.
- Profile run: `fg19_parent_h256x2_f0_s1001_profile_r2_v1`; run output `/home/grinchenkoeg/computer-vision-homework-20261018/data/artifacts/runs/fg19_parent_h256x2_f0_s1001_profile_r2_v1`.
- Evidence/log directory: `/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_parent_profile_v1`; native unit `computer-vision-homework-fg19-parent-profile-v1.service`.

The exact 43-option canonical trainer argv is:

```bash
scripts/train_rnn.py \
  --name fg19_parent_h256x2_f0_s1001_profile_r2_v1 --device cuda --seed 1001 \
  --split-manifest /home/grinchenkoeg/computer-vision-homework-20261018/splits/fold0.json \
  --train-source train+valid --valid-fold 0 --train-seqs 10535 \
  --valid-repeat 1 --valid-train-fraction 1 --arch gru --cell gru --enc linear \
  --hidden 256 --layers 2 --head-layers 1 --dropout 0 --aux-struct 0.3 --aux-next 0.3 \
  --batch 16 --chunk 2000 --epochs 4 --checkpoint-policy fixed --checkpoint-epoch 3 \
  --no-eval-during-fit --loss wmse --loss-norm chunk \
  --distill-cache-root /home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_t3_fit_homogeneous_cache_v1 \
  --distill-cache-variant homogeneous --distill-cache-scale 1 --distill-alpha 0.9 \
  --distill-family-weight 0 --lr 0.0001 --wd 0.01 --beta2 0.999 --sched onecycle \
  --warmup-frac 0.05 --clip 1 --precision fp32 --optim adamw \
  --profile-steps 4 --profile-warmup 2 --max-batches 6 --grad-stats
```

The environment binds `WNN_PROJECT_ROOT` to `B/data`, sets `WNN_LOADER_WORKERS=0` and `WNN_LOADER_RAM_GB=1`, one CPU thread, `WNN_GPU_MEMORY_BYTES=4294967296`, selected UUID `GPU-ec74220d-78d7-cc47-69fc-0de1fb580944`, `NVIDIA_TF32_OVERRIDE=0`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The admission requires 5,294,967,296 free GPU bytes and leaves foreign GPU processes untouched. Worker cap: one CPU, `MemoryHigh=7,500,000,000`, `MemoryMax=8,000,000,000`, swap 0, `RuntimeMaxSec=600`. Parent limits remain two CPUs, 18/20 GB high/max, swap 0, with 17 GB admission headroom. Physical `B/research_tmp` must have at least 25 GiB free.

The generic profile uses the canonical two-batch fit normalizer (batch 16, 32 fitting sequences, transformed rows sampled at stride `[::7]`), then two warmup and four measured batches. Six batches at 20,000 rows with 2,000-row update windows produce 60 in-memory AdamW updates. The runner returns before evaluation or checkpoint saving; it does not grade quality or read held-out labels. It emits `PROFILE` and `PROFILE_UPDATE_EVIDENCE`; generic mode does not report CUDA peak allocation or process RSS. If the post-submit `systemctl show` times out, the timeout is recorded in `LAUNCH.json` as `timeout_unknown`; never retry the profile submission.
