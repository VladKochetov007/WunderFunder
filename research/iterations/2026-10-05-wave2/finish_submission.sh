#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave2"
for ((attempt=0; attempt<245; attempt++)); do
    if ! systemctl --user is-active --quiet wunder-wave2-full-tstream3; then break; fi
    sleep 30
done
test -s artifacts/runs/w2fts_s3ft/model_ep1.pt
export WNN_CPU_PCT=3 WNN_MEM_MAX=12G OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
scripts/limited .venv/bin/python "$D/export_twostream.py" artifacts/runs/w2fts_s3ft/model_ep1.pt "$D/tstream_final.npz" --export-only > "$D/export-final.log" 2>&1
~/venv311/bin/python "$D/native_audit.py" "$D/tstream_final.npz" "$D/tstream_final.probe.npz" research/iterations/2026-10-02-big-ensemble/run_vlad_pair_bw "$D/native-final.json" > "$D/native-final.log" 2>&1
.venv/bin/python "$D/build_submission.py" > "$D/build-submission.log" 2>&1
~/venv311/bin/python "$D/audit_submission.py" "$D/blend_twostream" > "$D/audit-submission.log" 2>&1
echo "DONE $(date -Is) $D/blend_twostream.zip"
