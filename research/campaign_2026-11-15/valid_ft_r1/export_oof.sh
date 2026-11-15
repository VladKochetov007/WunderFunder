#!/usr/bin/env bash
cd /home/vlad/development/wunder/research_tmp/chrono_root
export WNN_GPU_MEMORY_BYTES=450000000 WNN_PROJECT_ROOT=$PWD NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 WNN_THREADS=3 TMPDIR=/home/vlad/development/wunder/research_tmp
for r in q15v_V_f0 q15v_V_f1 q15v_V6_f0 q15v_V6_f1 q15v_Vfbc_f0 q15v_Vfbc_f1 q15v_Vkd_f0 q15v_Vkd_f1; do
  [ -e cache/pred_${r}_valid.npy ] && continue
  /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python -u /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-11-14/quality_r2/teacher_pool_r3/export_small_batch.py $r valid --epoch 1 --inference-precision fp32 > /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-11-15/valid_ft_r1/$r.valid.log 2>&1 || echo FAIL $r
  echo DONE $r
done
