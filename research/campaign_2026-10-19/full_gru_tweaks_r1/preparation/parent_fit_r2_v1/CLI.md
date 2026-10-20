# Full parent fit R2 admission

The rendered body binds parent-profile acceptance SHA-256 `426792c35c9685682de266db056414cf2a6a0d0a550dfab07c4fb528be6fbebf`, exact-stop SHA-256 `a401b954f63c761776699221403fa9cf11ba6e6d29cc69a8f7f1effab3f25257`, and invocation `dc7f34afbdfb4fed8bdfbb66d20d558b`. It embeds both raw receipts and rechecks B-side profile evidence hashes before any fit output is created. The template remains null and fail-closed. Root still owns final review, fresh preflight, and the only launch.

- Run: `fg19_parent_h256x2_f0_s1001_fit_r2_v1`
- Output/evidence: `/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_parent_fit_r2_v1`
- Run artifacts: `/home/grinchenkoeg/computer-vision-homework-20261018/data/artifacts/runs/fg19_parent_h256x2_f0_s1001_fit_r2_v1`
- Unit: `computer-vision-homework-fg19-parent-fit-r2-v1.service`
- Source: `/home/grinchenkoeg/computer-vision-homework-20261018/source-fg19-cache-68ff6829` at commit `68ff682977b822f38351aac0f621146f5f13b440`
- Cache: `/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_t3_fit_homogeneous_cache_v1`

The exact canonical trainer argv materialized by `_fit_args()` is:

```bash
scripts/train_rnn.py \
  --name fg19_parent_h256x2_f0_s1001_fit_r2_v1 --device cuda --seed 1001 \
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
  --grad-stats --save-training-state
```

The initial run has no `--resume-state`: it starts fresh from seed-1001 initialization and writes canonical epoch-boundary recovery state. A restart from an actual state file would be a separate exact-resume admission. The expected count is 29,840 updates (746 batches/epoch × 10 chunk updates/batch × 4 epochs), labeled expected rather than measured.

The process environment binds `WNN_PROJECT_ROOT` to `B/data`, `PYTHONPATH` to source `src`, one CUDA device/thread, `NVIDIA_TF32_OVERRIDE=0`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Limits are one CPU, `MemoryHigh=7,500,000,000`, `MemoryMax=8,000,000,000`, swap 0, 4 GiB allocator cap, 1 GB GPU free reserve, zero loader workers, 25 GiB physical scratch minimum, and a 24-hour runtime cap. Parent limits/headroom remain those in the reused reviewed gates.
