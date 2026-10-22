#!/usr/bin/env bash
# Run the frozen export/grade chain only after this exact two-fit worker succeeds.
set -euo pipefail

root=/home/vlad/development/wunder/wnn_connectome_starterpack
packet=$root/research/campaign_2026-10-20/submission_delivery_r1/nll_adaptation
source_root=/home/vlad/development/wunder/research_tmp/overnight-nll-adaptation-20261020
group=/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/wnn.slice/wnn-campaign.slice
fit=wnn-overnight-nll-fit-20261020.service
export_unit=wnn-overnight-nll-export-20261020.service

[[ $(systemctl --user show "$fit" -p InvocationID --value) == d2a82026557c4f5b915a361b8f8e375a ]]
while [[ $(systemctl --user show "$fit" -p MainPID --value) != 0 ]]; do
  [[ ! -e "$packet/STOP_POST_FIT" ]] || exit 0
  sleep 15
done
[[ ! -e "$packet/STOP_POST_FIT" ]] || exit 0
[[ $(systemctl --user show "$fit" -p Result --value) == success ]]
[[ $(systemctl --user show "$fit" -p ExecMainStatus --value) == 0 ]]

python3 - "$packet" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

packet = Path(sys.argv[1])
report = json.loads((packet / "fit_plan_report.json").read_text())
expected = ["wmse_two_epoch_fit", "nll_two_epoch_fit"]
if report["status"] != "complete" or [s["name"] for s in report["steps"]] != expected:
    raise RuntimeError("The admitted pair did not complete in its frozen order")
if any(s["returncode"] != 0 for s in report["steps"]):
    raise RuntimeError("A fit failed; no prediction exports admitted")
admission = json.loads((packet / "post_fit_admission.json").read_text())
for name, expected_hash in admission["execution_pins"].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected_hash:
        raise RuntimeError(f"Post-fit source or plan changed: {name}")
source = admission["producer_worktree"]
subprocess.run(
    ["git", "-C", source, "diff", "--exit-code", admission["producer_revision"],
     "--", "src/wnn", "scripts", "utils.py"], check=True,
)
extra = subprocess.check_output(
    ["git", "-C", source, "ls-files", "--others", "--exclude-standard", "--",
     "src/wnn", "scripts", "utils.py"], text=True,
).splitlines()
if any(name.endswith(".py") for name in extra):
    raise RuntimeError("Untracked Python source in the frozen producer execution tree")
properties = subprocess.check_output(
    ["systemctl", "--user", "show", admission["fit_unit"], "-p",
     "ExecMainStartTimestampMonotonic", "-p", "ExecMainExitTimestampMonotonic"], text=True,
)
times = dict(line.split("=", 1) for line in properties.splitlines())
fit_seconds = (int(times["ExecMainExitTimestampMonotonic"])
               - int(times["ExecMainStartTimestampMonotonic"])) / 1e6
spent = admission["prior_profile_seconds"] + fit_seconds
if fit_seconds <= 0 or spent + admission["post_fit_cap_seconds"] > 54000:
    raise RuntimeError("Post-fit work does not fit the remaining 15-hour allocation")
with (packet / "post_fit_start.json").open("x") as stream:
    json.dump({"timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "fit_worker_seconds": fit_seconds, "spent_seconds_before_export": spent,
               "remaining_seconds_before_export": 54000 - spent,
               "fit_success_and_execution_pins_verified": True}, stream, indent=2)
    stream.write("\n")
PY

# Release only inactive file cache left charged by the completed fit.
available=$(awk '$1=="inactive_file" {print $2}' "$group/memory.stat")
if (( available > 0 )); then
  printf '%s swappiness=0\n' "$available" > "$group/memory.reclaim" 2>/dev/null || true
fi

systemd-run --user --unit="$export_unit" --slice=wnn-campaign.slice \
  --working-directory="$root" \
  -p CPUQuota=100% -p MemoryHigh=3400000000 -p MemoryMax=4000000000 \
  -p MemorySwapMax=0 -p RuntimeMaxSec=2700 -p TimeoutStopSec=60 \
  -p RemainAfterExit=yes -p Nice=10 -p IOSchedulingClass=idle \
  -p "StandardOutput=append:$packet/exports.log" -p StandardError=inherit \
  --setenv PYTHONUNBUFFERED=1 --setenv "WNN_PROJECT_ROOT=$root" \
  --setenv "PYTHONPATH=$source_root/src" --setenv "TMPDIR=$source_root" \
  --setenv WNN_THREADS=1 --setenv WNN_LOADER_WORKERS=0 \
  --setenv WNN_LOADER_RAM_GB=3.5 \
  --setenv WNN_GPU_MEMORY_BYTES=4500000000 \
  --setenv OMP_NUM_THREADS=1 --setenv MKL_NUM_THREADS=1 --setenv OPENBLAS_NUM_THREADS=1 \
  /usr/bin/flock --nonblock \
  "$root/research/campaign_2026-10-06/bandit_continuation/runtime/gpu.lock" \
  /usr/bin/time -v "$root/.venv/bin/python" "$root/scripts/kaggle_sequence.py" \
  --plan "$packet/post_fit_plan.json"

while [[ $(systemctl --user show "$export_unit" -p MainPID --value) != 0 ]]; do
  current=$(cat "$group/memory.current")
  if (( current > 4400000000 )); then
    available=$(awk '$1=="inactive_file" {print $2}' "$group/memory.stat")
    requested=$((current - 4000000000))
    (( requested > available )) && requested=$available
    if (( requested > 0 )); then
      printf '%s swappiness=0\n' "$requested" > "$group/memory.reclaim" 2>/dev/null || true
    fi
  fi
  {
    date -u +%FT%TZ
    printf 'parent_memory_bytes=%s\n' "$current"
    systemctl --user show "$export_unit" -p MemoryCurrent -p MemorySwapCurrent
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
  } >> "$packet/export_resource_samples.log"
  sleep 5
done
[[ $(systemctl --user show "$export_unit" -p Result --value) == success ]]
[[ $(systemctl --user show "$export_unit" -p ExecMainStatus --value) == 0 ]]
