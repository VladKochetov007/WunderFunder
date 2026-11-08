from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


def test_dashboard_preserves_metrics_without_mixing_units() -> None:
    """Execute the real JavaScript predicates against independent sentinel families."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable for dashboard JavaScript checks")
    script = Path(__file__).with_name("dashboard_metric_groups_invariant.cjs")
    result = subprocess.run(
        [node, str(script)], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "passed"
    assert receipt["checked_keys"] >= 29
