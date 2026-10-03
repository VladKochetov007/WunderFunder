"""Exact fixed-panel baseline domains from frozen arrays, not full-test inference."""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
ACTOR = HERE.parent / "matched_usage"
sys.path.insert(0, str(ROOT))
from utils import GlobalAccumulator  # noqa: E402

started = time.monotonic()
samples = json.loads((ACTOR / "samples.json").read_text())["samples"]
accumulators = {
    name: GlobalAccumulator()
    for name in ("cpu_all_panel_scored", "cpu_after511_panel_scored", "cache_after511_panel_scored")
}
max_failures, rms_failures, correlation_failures = [], [], []
for i, _sample in enumerate(samples):
    with np.load(ACTOR / f"seq_{i:02d}.npz") as z:
        y, mask = z["targets"], z["mask"]
        reference, cache = z["reference"], z["cached"]
        subset = mask & (np.arange(len(mask)) >= 512)
        accumulators["cpu_all_panel_scored"].add(y, reference, mask)
        accumulators["cpu_after511_panel_scored"].add(y, reference, subset)
        accumulators["cache_after511_panel_scored"].add(y, cache, subset)
        difference = reference - cache
        if np.abs(difference).max() > 0.02:
            max_failures.append(i)
        if np.sqrt(np.mean(difference.astype(float) ** 2)) > 0.005:
            rms_failures.append(i)
        if min(np.corrcoef(reference[:, k], cache[:, k])[0, 1] for k in range(2)) < 0.999:
            correlation_failures.append(i)
result = {
    "scope": "exact baseline scores for selected24components only; altered arms only available at queries; no model forward",
    "domains": {name: accumulator.result() for name, accumulator in accumulators.items()},
    "drift_failures": {
        "max_abs": max_failures,
        "rms": rms_failures,
        "correlation": correlation_failures,
    },
    "cost": {
        "wall_seconds": time.monotonic() - started,
        "cpu_seconds": time.process_time(),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    },
}
(HERE / "domain_diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
