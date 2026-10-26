"""Independently recompute recorded metrics from immutable prediction arrays."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def file_hash(path: Path) -> str:
    """Hash one artifact in bounded reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_result(path: Path, source: Path) -> None:
    """Verify hashes and reproduce official WP and same-mask WMSE without GPU use."""
    import numpy as np

    sys.path.insert(0, str(source / "src"))
    from wnn.gmetric import global_wp

    result = json.loads(path.read_text())
    expected_hash = file_hash(path)
    receipt = path.with_name("VERIFIED_METRICS.json")
    if receipt.exists() and json.loads(receipt.read_text()).get("result_sha256") == expected_hash:
        return
    if result["status"] != "complete":
        raise ValueError("cannot verify an incomplete result")
    checks = {}
    for phase in ["train_probe", "heldout"]:
        values = result[phase]
        arrays = {}
        for name, artifact in values["artifacts"].items():
            payload = (path.parent / artifact["path"]).resolve(strict=True)
            if (
                not payload.is_relative_to(path.parent.resolve())
                or file_hash(payload) != artifact["sha256"]
            ):
                raise ValueError(f"array containment/hash mismatch: {phase}/{name}")
            arrays[name] = np.load(payload, mmap_mode="r", allow_pickle=False)
        y, p = arrays["targets"], arrays["predictions"]
        wp = global_wp(y, p, arrays["selected_mask"])["wp"]
        target = np.clip(y[:, 99:, :], -2, 2).astype(np.float64)
        weight = np.abs(target)
        error = p[:, 99:, :].astype(np.float64) - target
        wmse = float(np.sum(weight * error**2) / np.sum(weight))
        expected = values["metrics"]
        key = "official_wp" if phase == "heldout" else "diagnostic_wp_all_after_warmup"
        if (
            abs(wp - expected[key]) > 1e-10
            or abs(wmse - expected["weighted_mse_all_after_warmup"]) > 1e-10
        ):
            raise ValueError(f"independent metric mismatch: {phase}")
        checks[phase] = {"wp": wp, "wmse": wmse, "sequences": len(y)}
    receipt.write_text(
        json.dumps(
            {
                "status": "verified",
                "result_sha256": expected_hash,
                "verifier_sha256": file_hash(Path(__file__)),
                "metrics": checks,
            },
            indent=2,
        )
        + "\n"
    )
