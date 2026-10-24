"""Recompute completed diagnostic metrics independently from saved arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wnn.gmetric import global_wp


def hash_file(path: Path) -> str:
    """Hash an artifact without retaining its bytes in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(result_path: Path) -> dict:
    """Check stored prediction hashes and independently reproduce both metric masks."""
    result = json.loads(result_path.read_text())
    if result["status"] != "complete":
        raise ValueError("only full completed diagnostics are independently verified")
    checks = {}
    for name, phase in result["phases"].items():
        arrays = {}
        for stem, artifact in phase["artifacts"].items():
            path = Path(artifact["path"])
            if hash_file(path) != artifact["sha256"]:
                raise ValueError(f"artifact hash mismatch: {stem}")
            arrays[stem] = np.load(path, mmap_mode="r", allow_pickle=False)
        y = arrays["targets"]
        p = arrays["predictions"]
        m = arrays["mask"]
        wp = global_wp(y, p, m)
        expected_wp = phase["metrics"]["wp"]
        if abs(wp["wp"] - expected_wp) > 1e-10:
            raise ValueError(f"official WP mismatch: {name}")
        # WMSE uses clipped labels and raw predictions, not metric-clipped predictions.
        target = np.clip(y[:, 99:, :], -2, 2).astype(np.float64)
        weight = np.abs(target)
        error = p[:, 99:, :].astype(np.float64) - target
        wmse = float(np.sum(weight * error**2) / np.sum(weight))
        if abs(wmse - phase["weighted_mse_all_after_warmup"]) > 1e-10:
            raise ValueError(f"weighted MSE mismatch: {name}")
        checks[name] = {
            "wp": wp["wp"],
            "wmse": wmse,
            "sequences": len(y),
            "selected_rows": wp["selected_rows"],
        }
    receipt = {
        "status": "verified",
        "result_sha256": hash_file(result_path),
        "cell": result["cell"],
        "completed_epochs": result["completed_epochs"],
        "independent_array_recomputation": checks,
    }
    path = result_path.with_name("INDEPENDENT_METRIC_CHECK.json")
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    """Verify all complete checkpoints without any GPU work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    for path in args.root.glob("evaluations/*/inode-*/RESULT.json"):
        if json.loads(path.read_text())["status"] == "complete":
            print(json.dumps(verify(path)))


if __name__ == "__main__":
    main()
