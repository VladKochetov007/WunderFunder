import importlib.util
import sys
from pathlib import Path

path = Path.cwd() / "research/campaign_2026-10-06/bandit_continuation/vast_teachers_r1/controller.py"
spec = importlib.util.spec_from_file_location("h390_lane_controller", path)
assert spec is not None and spec.loader is not None
controller = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = controller
spec.loader.exec_module(controller)
archive = "/tmp/h390_all_data_homogeneous_r2_b2bee94b.tar.gz"
controller.scp([archive], "/root/.h390_all_data_homogeneous_r2_b2bee94b.tar.gz", upload=True, timeout=90)
print(f"uploaded {Path(archive).stat().st_size} bytes to lane-b instance {controller.receipt()['instance_id']}")
