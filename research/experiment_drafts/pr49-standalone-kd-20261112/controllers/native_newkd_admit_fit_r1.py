"""Admit four successful actual-device profiles and launch fixed-endpoint fits."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

root = Path("/data/lc-epanov_s/tabdata-final-quality-20261112-r1")
packet = root / "kd-intake-r2/pr49-standalone-kd-r2"
manifest = json.loads((packet / "PANEL_MANIFEST_R2.json").read_bytes())
complete = json.loads((root / "control/profile-newkd/complete.json").read_bytes())
assert len(complete["results"]) == 4
assert all(
    row["exit_code"] == 0 and not row["timed_out"] and not row["resource_errors"]
    for row in complete["results"]
)
preflight = json.loads((root / "control/NEW_KD_CANONICAL_PREFLIGHT_R1.json").read_bytes())
assert preflight["returncode"] == 0
clock = json.loads((root / "control/CLOCK_R1.json").read_bytes())
assert time.time() < clock["fit_deadline_epoch"]
approved = root / "approved-profiles"
approved.mkdir(exist_ok=True)
rows = []
for entry in manifest["cells"]:
    path = root / "runs" / entry["cell"] / "profile/profile_receipt.json"
    receipt = json.loads(path.read_bytes())
    identity = receipt["identity"]
    assert receipt["status"] == "profile_complete" and receipt["updates"] == 30
    assert receipt["error_count"] == 0 and receipt["errors"] == []
    assert receipt["engineering_reload"]["status"] == "passed"
    assert all(
        row["model_optimizer_scheduler_rng_equal"]
        for row in receipt["engineering_reload"]["by_rank"]
    )
    assert (
        identity["source_sha256"]
        == "ae7ae9386060475363afad63cdca140c740e80289fe908e82211dc3bb2f61242"
    )
    assert (
        identity["data"]["split_sha256"]
        == "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
    )
    assert identity["world_size"] == 1
    assert (
        identity["data"]["teacher"]["checkpoint_sha256"]
        == "759165147d99b33034b74f4af63a30e8a202dafc78fd5e5699d05b6cfe9bef90"
    )
    assert (
        identity["data"]["teacher"]["manifest_sha256"]
        == "138ef43c6ef03755a3b6195ca83f2c346f7dca4e3fe365f9e8e151ad2c71792d"
    )
    assert len(identity["data"]["fit_members"]) == 10325
    assert len(identity["data"]["evaluation_members"]) == 1873
    assert all(member[0] == "train" for member in identity["data"]["fit_members"])
    assert receipt["peak_reserved_bytes"] < 20 * 1024**3
    receipt["approved"] = True
    receipt["approval_note"] = (
        "Root reviewed actual canonical-config check, four concurrent 30-step L4 profiles, exact source/split, no errors and model/optimizer/scheduler/RNG roundtrip. Fixed six-epoch endpoint; raw full-valid report-only; unchanged wave clock."
    )
    receipt["profile_evidence_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with (approved / (entry["cell"] + ".json")).open("x") as handle:
        json.dump(receipt, handle, indent=2)
    rows.append(
        {
            "cell": entry["cell"],
            "elapsed_seconds": receipt["elapsed_seconds"],
            "reserved_bytes": receipt["peak_reserved_bytes"],
            "profile_sha256": receipt["profile_evidence_sha256"],
        }
    )
python = "/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python"
command = [python, str(packet / "l4_standalone_kd_wave_r1.py"), "fit-newkd"]
env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
with (root / "control/newkd-fullfits-r1.log").open("xb") as log:
    process = subprocess.Popen(
        command,
        cwd=str(packet),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
owner = {
    "pid": process.pid,
    "pgid": os.getpgid(process.pid),
    "birth": int(
        Path("/proc/" + str(process.pid) + "/stat").read_text().split(") ")[1].split()[19]
    ),
    "command": command,
    "at_epoch": time.time(),
    "profiles": rows,
    "fit_deadline_epoch": clock["fit_deadline_epoch"],
}
with (root / "control/NEW_KD_FULLFIT_LAUNCH_R1.json").open("x") as handle:
    json.dump(owner, handle, indent=2)
print(json.dumps(owner))
