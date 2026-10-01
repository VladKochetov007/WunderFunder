from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from wnn.evaluation import (
    _score_component_moments,
    paired_component_bootstrap,
    paired_component_moments,
)


def main() -> None:
    """Find precision failures when resampling a single shifted component."""
    started = time.perf_counter()
    tried = 0
    for center in np.linspace(0.1, 1.9, 73).astype(np.float32):
        lower = np.nextafter(center, np.float32(0))
        for repeats in (1, 2, 3, 5, 7, 10, 16, 25, 64, 100, 128):
            one = np.tile(np.array([[lower, lower], [center, center]]), (repeats, 1))
            y = np.stack([one, -one])
            reference = np.ones_like(y)
            mask = np.ones(y.shape[:-1], dtype=bool)
            a, b = paired_component_moments(y, y, reference, mask)
            score = _score_component_moments(a, np.array([2, 0]))
            tried += 1
            if center - lower > 2e-8 and abs(score - 1) > 1e-6:
                report = {
                    "status": "counterexample",
                    "center": float(center),
                    "lower": float(lower),
                    "repeats": repeats,
                    "single_component_score": score,
                    "expected_single_component_score": 1.0,
                    "cases_tried": tried,
                    "bootstrap": paired_component_bootstrap(
                        a, b, np.array([0, 1]), n_boot=50, centered=True
                    ),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report, indent=2))
                return
    report = {
        "status": "no_counterexample",
        "tried": tried,
        "elapsed_seconds": time.perf_counter() - started,
        "evaluation_sha256": hashlib.sha256(
            (ROOT / "src/wnn/evaluation.py").read_bytes()
        ).hexdigest(),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
