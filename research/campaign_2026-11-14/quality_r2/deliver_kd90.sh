#!/usr/bin/env bash
# Complete only the declared confirmation -> two refits -> native packages path.
set -euo pipefail
R=/home/vlad/development/wunder/wnn_connectome_starterpack
O=$R/research/campaign_2026-11-14/quality_r2
Z=$R/submissions/quality_20261114_r2
T=/home/vlad/development/wunder/research_tmp/q14_packages
P=$R/research/campaign_2026-10-26/packaging_r1/package_tool.py
cd "$R"
while systemctl --user is-active -q wnn-q14-fit-kd90-s4; do sleep 20; done
"$R/.venv/bin/python" "$O/decide_kd90.py"
mkdir -p "$Z" "$T"
for seed in 3 4; do
  test ! -e "$O/STOP"
  name=q14_final_kd90_s$seed
  unit=wnn-q14-fit-final-s$seed
  systemd-run --user --unit="$unit" --slice=wnn-campaign.slice \
    -p CPUQuota=200% -p MemoryMax=16G -p MemorySwapMax=0 -p RuntimeMaxSec=5400 \
    --working-directory="$R" /bin/bash "$O/run_final_fit.sh" "$seed"
  systemd-run --user --unit="wnn-q14-thermal-final-s$seed" --slice=wnn-campaign.slice \
    -p CPUQuota=5% -p MemoryMax=64M /bin/bash "$O/thermal_guard.sh"
  while systemctl --user is-active -q "$unit"; do sleep 20; done
  test "$(systemctl --user show "$unit" -p Result --value)" = success
  test -f "/home/vlad/development/wunder/research_tmp/chrono_root/artifacts/runs/$name/model_ep3.pt"
  ln -s "/home/vlad/development/wunder/research_tmp/chrono_root/artifacts/runs/$name" "artifacts/runs/$name"
  WNN_CPU_QUOTA=100% WNN_MEM_MAX=4G scripts/limited .venv/bin/python "$P" export "$name" 3 "$Z/kd90_s$seed.npz"
  if [ "$seed" = 3 ]; then
    .venv/bin/python "$P" build "$Z/final_kd90_s3_single.zip" "$name@ep3=$Z/kd90_s3.npz=1.0"
    WNN_CPU_QUOTA=100% WNN_MEM_MAX=4G scripts/limited .venv-py311-profile/bin/python "$P" replay \
      "$Z/final_kd90_s3_single.zip" "$T/replay_final_kd90_s3_single"
  fi
done
.venv/bin/python "$P" build "$Z/final_kd90_s3s4_pair.zip" \
  "q14_final_kd90_s3@ep3=$Z/kd90_s3.npz=0.5" "q14_final_kd90_s4@ep3=$Z/kd90_s4.npz=0.5"
WNN_CPU_QUOTA=100% WNN_MEM_MAX=4G scripts/limited .venv-py311-profile/bin/python "$P" replay \
  "$Z/final_kd90_s3s4_pair.zip" "$T/replay_final_kd90_s3s4_pair"
sha256sum "$Z"/*.zip > "$O/SHA256SUMS"
echo NATIVE_PACKAGES_READY_REQUIRES_ROOT_PARITY_AND_HANDOFF
