# Shared train-only student arguments (PR49 _base protocol + production parent knobs).
SPLIT=/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-29/teacher_track_r1/chronological_primary.json
BASE="--hidden 256 --layers 2 --dropout 0 --train-source train --train-seqs 10325 --eval-source dev --full-valid --split-manifest $SPLIT --precision fp32 --checkpoint-policy fixed --loss-norm chunk --sched onecycle --warmup-frac 0.05 --wd 0.01 --clip 1 --beta2 0.999 --grad-stats --save-training-state --train-eval-seqs 0 --loss wmse --optim muon --muon-scope rnn --since-weight 32 --since-floor .25 --batch 32 --chunk 1000"
PARENT="--epochs 4 --lr .001 --aux-next 0.3 --aux-struct 0.3 --aux-weight 0.5"
CHILD="--epochs 6 --lr .0001 --batch 16 --chunk 2000 --aux-next 0.3 --aux-struct 0.3 --aux-weight 0.5 --inherit-init-preprocessing"
