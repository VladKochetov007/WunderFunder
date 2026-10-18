"""Prepare a fixed 75/25 FP32 reference from hash-pinned archived predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    """Hash an immutable input or completed output."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Write the fixed full-fold reference and its first-eight-group panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    root = args.base / "submission-f0-r1"
    source = args.base / "result_snapshots/rr18_r01_f0_s1001_e6/predictions"
    pins = {
        "wide": "21af6180c3290a99f450d701daca03581762ea18cfc70ffcc9453929435ee8f0",
        "deep": "50ea5e088b60da8b31ee1796771c4c13f2940131293a51793d63ceadf324be3d",
    }
    for arm, expected in pins.items():
        if digest(source / f"{arm}.npy") != expected:
            raise ValueError(f"{arm} identity differs")
    output = root / "mix75_reference.npy"
    if output.exists():
        raise FileExistsError(output)
    wide, deep = [np.load(source / f"{arm}.npy", mmap_mode="r") for arm in pins]
    mixed = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=wide.shape)
    for group in range(len(wide)):
        mixed[group] = np.float32(0.75) * wide[group] + np.float32(0.25) * deep[group]
    mixed.flush()
    np.save(root / "mix75_panel_reference.npy", mixed[:8])
    report = {
        "source_pins": pins,
        "wide_coefficient": 0.75,
        "deep_coefficient": 0.25,
        "files": {p.name: digest(p) for p in (output, root / "mix75_panel_reference.npy")},
    }
    (root / "REFERENCE.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
