import hashlib
import json
import os
import time
from pathlib import Path

plan = Path("/data/epanov_s/tabdata/runs/pr32-production-admission-20261024-r4-batch/plan.json")
supervisor = Path(
    "/data/epanov_s/tabdata/runtime/pr32-operator-20261024/run_teacher_cohort_batch.py"
)
for path, expected in [
    (plan, "102123dd970481a1a87f1ca69d1ce59906dd8e4e9af06a5cefb90c00c24d01da"),
    (supervisor, "6f1cbebf80e436d09162a50cd99571e60d7a97c831c19ad547bc9d1b76d99cac"),
]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Frozen launch input changed")
try:
    fd = os.open(
        "/data/epanov_s/tabdata/runs/pr32-production-admission-20261024-r4-batch/LAUNCHED_ONCE.json",
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
except FileExistsError:
    raise SystemExit(0) from None
with os.fdopen(fd, "w") as f:
    json.dump(
        {
            "pid": os.getpid(),
            "created_unix": time.time(),
            "plan_sha256": "102123dd970481a1a87f1ca69d1ce59906dd8e4e9af06a5cefb90c00c24d01da",
            "operator_sha256": "6f1cbebf80e436d09162a50cd99571e60d7a97c831c19ad547bc9d1b76d99cac",
        },
        f,
    )
    f.flush()
    os.fsync(f.fileno())
python = "/data/epanov_s/tabdata/.venv/bin/python"
os.execv(python, [python, str(supervisor), "--plan", str(plan)])
