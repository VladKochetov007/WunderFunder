#!/usr/bin/env bash
# Optimization-dynamics diagnostic (research/campaign_2026-10-02/opt/OPT_REPORT.txt), clean fold 0/1, ~15 GPU min:
#  A. epoch curves of the distilled students (GRU256 a0.5 / a0.7, LSTM192 a0.5) from their saved ep0-3 snapshots
#     (skipped when diag/epoch_curves_cpu.json exists)
#  B. default-path parity of opt/train_rnn_optim.diff: original vs patched train_rnn, 30 deterministic steps
#  C. pre-clip gradient norms / clip rate at the trained a0.7 ep3 point (lr 0, 100 steps)
#  D. 400-step smokes on the real schedule (a0.7 recipe, same seed and batches): AdamW, Muon, chrono init
# Parts fail independently; outputs go to research/campaign_2026-10-02/opt/diag/.
#   opt_diag.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
export WNN_VRAM_GB=${WNN_VRAM_GB:-4.6} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
out=research/campaign_2026-10-02/opt/diag
mkdir -p "$out"
py=.venv/bin/python

# A runs only if the CPU pass (same script, --device cpu) has not produced the curves already
[[ -f $out/epoch_curves_cpu.json ]] || $py research/campaign_2026-10-02/opt/epoch_curve.py --output "$out/epoch_curves.json" \
  || echo "OPT_DIAG epoch_curve failed"

patched=$out/train_rnn_patched.py
if ! patch -s -o "$patched" scripts/train_rnn.py research/campaign_2026-10-02/opt/train_rnn_optim.diff; then
  echo "OPT_DIAG patch no longer applies; skipping B-D"
  exit 0
fi
sed -i "s|^ROOT = Path(__file__).resolve().parents\[1\]|ROOT = Path(\"$PWD\")|" "$patched"
grep -q "^ROOT = Path(\"$PWD\")" "$patched" || { echo "OPT_DIAG cannot pin ROOT in the patched copy"; exit 0; }

recipe=(--train-source train+valid --full-valid --loss wmse --aux-struct 0.3 --aux-next 0.3
  --propensity c20260929_prop_f0 --prop-power 2 --prop-eps 0.03 --valid-true-mask 0.03 --hidden 256 --layers 2
  --batch 32 --chunk 2000 --wd 0.01 --seed 3 --ema-teacher 1.0 --ema-decay 0.999 --eval-ema --since-weight 32
  --since-floor 0.25 --distill c20260929_since_s3_f0,c20260929_since_s4_f0 --distill-alpha 0.7 --fast
  --split-manifest research/campaign_2026-09-29/artifacts/splits/fold0.json --checkpoint-policy fixed
  --no-eval-during-fit --epochs 1 --checkpoint-epoch 0)

# B: parity of the default path (both write a per-step loss trace and the final weights)
for arm in orig patched; do
  script=scripts/train_rnn.py
  [[ $arm == patched ]] && script=$patched
  $py "$script" --name "c20261002_optdiag_par_$arm" "${recipe[@]}" --lr 0.001 --max-batches 3 --deterministic \
    --parity-dump "$out/parity_$arm.pt" > "$out/parity_$arm.log" 2>&1 || echo "OPT_DIAG parity $arm failed"
done
$py - "$out" <<'EOF' || echo "OPT_DIAG parity compare failed"
import json, sys, torch
out = sys.argv[1]
a, b = (torch.load(f"{out}/parity_{k}.pt", map_location="cpu", weights_only=False) for k in ("orig", "patched"))
res = {"steps": len(a["losses"]), "losses_equal": bool(torch.equal(a["losses"], b["losses"])),
       "weights_equal": all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"]),
       "ema_equal": all(torch.equal(a["ema"][k], b["ema"][k]) for k in a["ema"]),
       "max_abs_weight_diff": max(float((a["model"][k].float() - b["model"][k].float()).abs().max()) for k in a["model"])}
print("OPT_DIAG parity", json.dumps(res))
open(f"{out}/parity.json", "w").write(json.dumps(res, indent=1) + "\n")
EOF

# C: gradient norms at the end of the real a0.7 run (lr 0: weights and EMA stay at ep3)
$py "$patched" --name c20261002_optdiag_late "${recipe[@]}" --init c20261001_d256a07_s3_f0@ep3 --lr 0 \
  --max-batches 10 --grad-stats --parity-dump "$out/late_ep3.pt" > "$out/late_ep3.log" 2>&1 \
  || echo "OPT_DIAG late grad stats failed"
grep GRAD_STATS "$out/late_ep3.log"

# D: first 400 steps of the real 1-epoch-schedule recipe, paired batches
for arm in adamw muon chrono; do
  extra=()
  [[ $arm == muon ]] && extra=(--optim muon)
  [[ $arm == chrono ]] && extra=(--rnn-init chrono)
  $py "$patched" --name "c20261002_optdiag_smoke_$arm" "${recipe[@]}" --lr 0.001 --max-batches 40 --grad-stats \
    "${extra[@]}" --parity-dump "$out/smoke_$arm.pt" > "$out/smoke_$arm.log" 2>&1 || echo "OPT_DIAG smoke $arm failed"
  grep GRAD_STATS "$out/smoke_$arm.log"
done
$py - "$out" <<'EOF' || echo "OPT_DIAG smoke compare failed"
import json, sys, torch
out = sys.argv[1]
res = {}
base = torch.load(f"{out}/smoke_adamw.pt", map_location="cpu", weights_only=False)["losses"].double()
for arm in ("adamw", "muon", "chrono"):
    try:
        loss = torch.load(f"{out}/smoke_{arm}.pt", map_location="cpu", weights_only=False)["losses"].double()
    except FileNotFoundError:
        continue
    d = (loss - base)[200:]
    res[arm] = {"steps": len(loss), "mean_loss_0_100": float(loss[:100].mean()), "mean_loss_300_400": float(loss[300:].mean()),
                "paired_diff_200_400": float(d.mean()), "paired_diff_sd_of_mean_iid": float(d.std() / len(d) ** 0.5)}
print("OPT_DIAG smoke", json.dumps(res))
open(f"{out}/smoke.json", "w").write(json.dumps(res, indent=1) + "\n")
EOF
rm -f "$out"/parity_*.pt
echo "OPT_DIAG_DONE"
