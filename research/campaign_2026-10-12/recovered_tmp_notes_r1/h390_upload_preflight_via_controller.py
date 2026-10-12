import importlib.util
import sys
from pathlib import Path

path = Path.cwd() / "research/campaign_2026-10-06/bandit_continuation/vast_teachers_r1/controller.py"
spec = importlib.util.spec_from_file_location("h390_lane_controller", path)
assert spec is not None and spec.loader is not None
controller = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = controller
spec.loader.exec_module(controller)
script = "/tmp/h390_r2_preflight.py"
controller.scp([script], "/root/.h390_r2_preflight_b2bee94b.py", upload=True, timeout=90)
print(f"uploaded {Path(script).stat().st_size} bytes to lane-b instance {controller.receipt()['instance_id']}")
