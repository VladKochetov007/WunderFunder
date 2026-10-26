#!/usr/bin/env bash
# Serialize GPU work (5 GB VRAM cap): muon6 (+its auto-grade) -> parent-numerics export/grade -> finals.
while systemctl --user is-active -q wnn-muon6; do sleep 30; done
/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-26/kaggle_parent_numerics_r1/export_grade.sh || echo "kpn grade failed" >> /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-26/final_muon_students_r1/pipeline.log
exec /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-26/final_muon_students_r1/run_final_students.sh
