"""Bounded regression checks for seed4 package verifier integration."""
from __future__ import annotations
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
source = ROOT / "research/campaign_2026-10-04/staging/e450/verify_seed4_package.py"
spec = importlib.util.spec_from_file_location("seed4_parity_review", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
shape = (20, 20_000, 2)
mask = np.broadcast_to((np.arange(20_000) >= 99)[None, :, None], shape)
assert mask.sum() == 20 * 19_901 * 2
assert not mask[:, :99, :].any() and mask[:, 99:, :].all()
row = {"pred": np.zeros(shape, np.float32), "y": np.zeros(shape, np.float32), "mask": np.broadcast_to(np.arange(20_000) >= 99, shape[:2])}
report = module.compare_callbacks({name: row for name in ("i16_v3", "i16_v2", "f32_v3")})
assert report["passed"] and report["rows"] == 400_000
with tempfile.TemporaryDirectory() as directory:
    result_path = Path(directory) / "result.json"
    log_path = Path(directory) / "child.log"
    script = "import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({'reset_bitwise':True,'past_only_bitwise':True}))"
    result = module.child_json(["-c", script, str(result_path)], log_path, os.environ.copy(), timeout=5, result_path=result_path)
    assert result["reset_bitwise"] and result["past_only_bitwise"]
    assert not log_path.read_text()
output = ROOT / "research/campaign_2026-10-04/checks/e450_verifier_review_v1.json"
output.write_text(json.dumps({"status":"passed", "scope":"Synthetic full-size mask and callback-array parity integration; actual file-only child JSON result; no model/package predictions or timing measured"}, indent=2)+"\n")
print(output)
