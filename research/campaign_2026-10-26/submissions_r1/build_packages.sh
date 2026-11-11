#!/usr/bin/env bash
# Build and natively replay the pre-declared packages (SELECTION_RULE_R1.json) for available Muon students.
set -euo pipefail
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack; cd $ROOT
S=$ROOT/research/campaign_2026-10-26/submissions_r1
P=$ROOT/research/campaign_2026-10-26/packaging_r1/package_tool.py
Z=$ROOT/submissions/muon_20261026; T=/home/vlad/development/wunder/research_tmp/pkg_20261026
mkdir -p $Z $T
export OMP_NUM_THREADS=1 PYTHONPATH=$ROOT/src
for s in 3 4; do
  r=vt_all_h390m_s${s}_r1
  [ -f artifacts/runs/$r/model_ep5.pt ] && [ ! -f $Z/h390m_s$s.npz ] && .venv/bin/python $P export $r 5 $Z/h390m_s$s.npz | tee -a $S/build.log
done
build() { local name=$1; shift; [ -f $Z/$name.zip ] && return 0; python3 $P build $Z/$name.zip "$@" | tee -a $S/build.log
  .venv-py311-profile/bin/python $P replay $Z/$name.zip $T/replay_$name | tee -a $S/build.log
  cp $T/replay_$name/REPLAY_RECEIPT.json $S/${name}_REPLAY_RECEIPT.json
  .venv/bin/python $P score $T/replay_$name/predictions.npy | sed "s/^/$name panel /" | tee -a $S/build.log; }
TL5="fttall_tlead_s5@ep3=base/weights_tl5.npz=0.33"; TL6="fttall_tlead_s6@ep3=base/weights_tl6.npz=0.33"
if [ -f $Z/h390m_s3.npz ]; then
  build muon_single_gru_s3 "vt_all_h390m_s3_r1@ep5=$Z/h390m_s3.npz=1.0"
  build muon_tl3_s3 "vt_all_h390m_s3_r1@ep5=$Z/h390m_s3.npz=0.34" "$TL5" "$TL6"
fi
if [ -f $Z/h390m_s3.npz ] && [ -f $Z/h390m_s4.npz ]; then
  build muon_gru_pair_s3s4 "vt_all_h390m_s3_r1@ep5=$Z/h390m_s3.npz=0.5" "vt_all_h390m_s4_r1@ep5=$Z/h390m_s4.npz=0.5"
  build muon_tl3_s3s4 "vt_all_h390m_s3_r1@ep5=$Z/h390m_s3.npz=0.17" "vt_all_h390m_s4_r1@ep5=$Z/h390m_s4.npz=0.17" "$TL5" "$TL6"
fi
sha256sum $Z/*.zip > $S/SHA256SUMS
