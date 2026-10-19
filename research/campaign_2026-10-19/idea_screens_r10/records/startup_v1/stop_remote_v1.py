from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path

UNIT = "computer-vision-homework-eda19-r10-v1.service"
DEST = Path("/home/grinchenkoeg/computer-vision-homework-20261018/support-source/eda19-r10-v1")


def properties() -> dict[str, str]:
    """Observe only this exact support unit."""
    result = subprocess.run(
        ["systemctl", "--user", "show", UNIT], capture_output=True, text=True, timeout=10
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


launch = json.loads((DEST / "ROOT_LAUNCH.json").read_bytes())
before = properties()
assert launch["unit"] == UNIT and launch["systemd_run_returncode"] == 0
assert before["InvocationID"] == launch["invocation_id"]
assert before["SubState"] == "exited" and before["Result"] == "success"
assert before["ExecMainStatus"] == "0" and before["ExecMainCode"] == "1"
assert before["CPUQuotaPerSecUSec"] == "500ms"
assert before["MemoryHigh"] == "1400000000" and before["MemoryMax"] == "1500000000"
assert before["MemorySwapMax"] == "0" and before["RuntimeMaxUSec"] == "10min"
result = subprocess.run(
    ["systemctl", "--user", "stop", UNIT], capture_output=True, text=True, timeout=20
)
after = properties()
assert result.returncode == 0 and after["ActiveState"] == "inactive"
assert after["SubState"] == "dead"
print(
    json.dumps(
        {
            "schema": "nf19.r10_exact_stop.v1",
            "utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "unit": UNIT,
            "before": before,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "after": after,
        }
    )
)
