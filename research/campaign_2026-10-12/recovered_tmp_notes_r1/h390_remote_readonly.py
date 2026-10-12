import importlib.util
import json
import sys
from pathlib import Path

path = Path.cwd() / "research/campaign_2026-10-06/bandit_continuation/vast_teachers_r1/controller.py"
spec = importlib.util.spec_from_file_location("h390_lane_controller", path)
assert spec is not None and spec.loader is not None
controller = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = controller
spec.loader.exec_module(controller)
for command in sys.argv[1:]:
    result = controller.ssh(command, timeout=60)
    print(json.dumps({"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}))
