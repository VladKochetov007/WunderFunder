#!/usr/bin/env bash
# Local F0/s1001 view-mask p=0.15 (control: tabm29_f0_s1001_muon_local_r1).
set -euo pipefail
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack; cd $ROOT
D=$ROOT/research/campaign_2026-10-29/view_mask_r1
ARGS="--device cuda --arch gru --hidden 512 --layers 1 --head-layers 1 --dropout 0 --batch 8 --chunk 2000 --epochs 3 --checkpoint-policy fixed --checkpoint-epoch 2 --no-eval-during-fit --loss wmse --loss-norm chunk --optim muon --sched onecycle --warmup-frac 0.05 --lr 0.001 --wd 0.01 --beta2 0.999 --clip 1 --precision fp32 --grad-stats --deterministic --batch-order seeded --train-source train+valid --train-seqs 10607 --valid-repeat 1 --research-features i1_held_i0book --research-feature-control all --seed 1001 --split-manifest research/campaign_2026-09-29/artifacts/splits/fold0.json --fit-subset-manifest research/campaign_2026-10-20/plateau_review_r1/fit_f0_10pct_s1001.json --fit-subset-experiment-id E-PLATEAU-I1-20261020-R1 --muon-scope rnn --strict-fp32-kernels"
export NVIDIA_TF32_OVERRIDE=0 PYTHONPATH=$ROOT/src WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=300% WNN_THREADS=3 WNN_MEM_MAX=12000000000
for arm in "p15:--view-mask-p 0.15"; do name=vmask29_f0_s1001_${arm%%:*}_local_r1; flags=${arm#*:}
  [ -f artifacts/runs/$name/model_ep2.pt ] || scripts/limited .venv/bin/python -u scripts/train_rnn.py $ARGS $flags --name $name > $D/$name.fit.log 2>&1
  [ -f cache/pred_${name}_valid.npy ] || scripts/limited .venv/bin/python scripts/predict_run.py $name valid --epoch 2 --campaign-split-manifest research/campaign_2026-09-29/artifacts/splits/fold0.json --campaign-full-valid-feature-only --device cuda --inference-precision fp32 --output-name $name > $D/$name.export.log 2>&1
done
echo LOCAL_DONE
