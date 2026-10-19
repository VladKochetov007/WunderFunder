from __future__ import annotations

import base64
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

B = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
DEST = B / "support-source/eda19-r10-v1"
OUT = B / "research_tmp/nf19-r10-v1"
UNIT = "computer-vision-homework-eda19-r10-v1.service"


def properties(unit: str) -> dict[str, str]:
    """Read exact systemd properties without acting on the unit."""
    result = subprocess.run(
        ["systemctl", "--user", "show", unit], capture_output=True, text=True, timeout=10
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def encoded(path: Path) -> dict[str, object]:
    """Transport only explicitly selected small receipts as original bytes."""
    payload = path.read_bytes()
    assert len(payload) < 4_000_000, path
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "base64": base64.b64encode(payload).decode(),
    }


def cgroup_snapshot(path: Path) -> dict[str, object]:
    """Record the scoped campaign hierarchy; never fall back to the host root."""
    assert path.name == "wnn-campaign.slice" and path.is_dir()
    return {
        "path": str(path),
        "memory_events": (path / "memory.events").read_text(),
        "memory_events_local": (path / "memory.events.local").read_text(),
        "memory_current": (path / "memory.current").read_text().strip(),
        "populated_descendants": {
            str(item.parent.relative_to(path)): item.read_text().split()
            for item in path.rglob("cgroup.procs")
            if item.read_text().strip()
        },
        "direct_pids": (path / "cgroup.procs").read_text().split(),
    }


launch = json.loads((DEST / "ROOT_LAUNCH.json").read_bytes())
unit = properties(UNIT)
assert launch["unit"] == UNIT and launch["systemd_run_returncode"] == 0
assert unit["InvocationID"] == launch["invocation_id"]
assert unit["SubState"] == "exited" and unit["Result"] == "success"
assert unit["ExecMainStatus"] == "0" and unit["ExecMainCode"] == "1"
assert unit["CPUQuotaPerSecUSec"] == "500ms"
assert unit["MemoryHigh"] == "1400000000" and unit["MemoryMax"] == "1500000000"
assert unit["MemorySwapMax"] == "0" and unit["RuntimeMaxUSec"] == "10min"
parent = properties("wnn-campaign.slice")
assert parent["ControlGroup"].endswith("/wnn-campaign.slice")
campaign = Path("/sys/fs/cgroup") / parent["ControlGroup"].lstrip("/")
files = {
    "ROOT_ADMISSION.json": DEST / "ROOT_ADMISSION.json",
    "ROOT_LAUNCH.json": DEST / "ROOT_LAUNCH.json",
    "ROOT_IMPORT_SMOKE.json": DEST / "ROOT_IMPORT_SMOKE.json",
    "PROBE_LOG.txt": DEST / "probe.log",
    **{name: OUT / name for name in ["START.json", "SOURCE.json", "RESULT.json", "DONE.json"]},
}
for optional in ["PROGRESS.jsonl", "FAILURE.json"]:
    if (OUT / optional).exists():
        files[optional] = OUT / optional
assert not (OUT / "FAILURE.json").exists()
receipt = {
    "schema": "nf19.r10_capture.v1",
    "utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "unit": unit,
    "parent": parent,
    "campaign": cgroup_snapshot(campaign),
    "own_cgroup": None,
    "own_cgroup_empty_terminal": unit["ControlGroup"] == "",
    "files": {name: encoded(path) for name, path in files.items()},
    "output_inventory": {path.name: path.stat().st_size for path in OUT.iterdir()},
}
if unit["ControlGroup"]:
    own = Path("/sys/fs/cgroup") / unit["ControlGroup"].lstrip("/")
    assert UNIT in own.parts and own.is_relative_to(campaign)
    receipt["own_cgroup"] = {
        "path": str(own),
        "memory_events": (own / "memory.events").read_text() if own.exists() else None,
        "pids": (own / "cgroup.procs").read_text().split() if own.exists() else None,
    }
print(json.dumps(receipt))
