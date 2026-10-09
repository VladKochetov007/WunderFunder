"""Describe recovered loss diversity after the preregistered primary grade.

These diagnostics are explicitly posthoc on the exposed panel. They do not
select blend weights or replace the frozen primary comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    """Hash an artifact without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_corr(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    """Compute a centered weighted Pearson correlation of diagnostic vectors."""
    mass = weight.sum()
    xc = x - np.dot(x, weight) / mass
    yc = y - np.dot(y, weight) / mass
    return float(
        np.dot(xc * weight, yc) / np.sqrt(np.dot(xc * weight, xc) * np.dot(yc * weight, yc))
    )


def analyze(root: Path, grade_path: Path, expected_grade_sha: str, output: Path) -> None:
    """Measure target-specific disagreement using exact already-graded artifacts."""
    if output.exists():
        raise FileExistsError(output)
    if sha256(grade_path) != expected_grade_sha:
        raise ValueError("primary grade identity changed")
    grade = json.loads(grade_path.read_text())
    spec_path = grade_path.parent / "FROZEN_GRADER_R2.json"
    if sha256(spec_path) != grade["spec_sha256"]:
        raise ValueError("primary specification identity changed")
    spec = json.loads(spec_path.read_text())
    targets_path = root / spec["targets"]["path"]
    if sha256(targets_path) != spec["targets"]["sha256"]:
        raise ValueError("target identity changed")
    arrays = {}
    groups = None
    for label in ("wmse", "huber_d1"):
        binding = grade["bindings"][label]
        path = Path(binding["path"])
        if sha256(path) != binding["sha256"]:
            raise ValueError(f"prediction changed: {label}")
        sidecar = path.with_suffix(".json")
        if sha256(sidecar) != binding["sidecar_sha256"]:
            raise ValueError(f"sidecar changed: {label}")
        current_groups = json.loads(sidecar.read_text())["row_groups"]
        if groups is not None and current_groups != groups:
            raise ValueError("prediction groups differ")
        groups = current_groups
        arrays[label] = np.load(path, mmap_mode="r", allow_pickle=False)
    with np.load(targets_path, allow_pickle=False) as cache:
        positions = {int(group): index for index, group in enumerate(cache["row_group"])}
        indices = [positions[int(group)] for group in groups]
        targets = np.asarray(cache["y"][indices], dtype=np.float32)
        mask = np.asarray(cache["mask"][indices], dtype=bool)
    mask &= (np.arange(20_000) >= 99)[None, :]
    diagnostics = {}
    for target in range(2):
        y = np.clip(targets[:, :, target][mask].astype(np.float64), -2, 2)
        weight = np.abs(y)
        a = np.clip(arrays["wmse"][:, :, target][mask].astype(np.float64), -2, 2)
        b = np.clip(arrays["huber_d1"][:, :, target][mask].astype(np.float64), -2, 2)
        diagnostics[f"t{target}"] = {
            "prediction_weighted_correlation": weighted_corr(a, b, weight),
            "residual_weighted_correlation": weighted_corr(a - y, b - y, weight),
            "weighted_disagreement_rms": float(np.sqrt(np.average((a - b) ** 2, weights=weight))),
            "weighted_disagreement_mae": float(np.average(np.abs(a - b), weights=weight)),
            "wmse_member_wp": weighted_corr(y, a, weight),
            "huber_member_wp": weighted_corr(y, b, weight),
            "equal_blend_minus_wmse": grade["scores"]["equal_blend"][f"t{target}"]
            - grade["scores"]["wmse"][f"t{target}"],
        }
        for label, value in (("wmse", a), ("huber_d1", b)):
            if abs(weighted_corr(y, value, weight) - grade["scores"][label][f"t{target}"]) > 1e-12:
                raise ValueError("diagnostic target metric differs from primary grade")
    result = {
        "schema": "wnn.loss-pair-disagreement-diagnostics.r1",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_sha256": sha256(Path(__file__)),
        "primary_grade_sha256": expected_grade_sha,
        "selection": "posthoc diagnostics after primary grade; no weight search or new endpoint",
        "exposure": "same exposed fold0, one training seed, 350 fit sequences",
        "diagnostic_weights": "original clipped-label magnitude on official fixed scored mask",
        "prediction_representation": "each member independently clipped to +-2 for correlation",
        "target_diagnostics": diagnostics,
        "allocation": "Complete the predeclared Muon cross before broad loss sweeps. Consider target-specific mixing and loss-scale controls as distinct prospective hypotheses; these diagnostics do not establish their generalization.",
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(diagnostics))


def main() -> None:
    """Require an independently pinned primary result before diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--grade", type=Path, required=True)
    parser.add_argument("--expected-grade-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.grade, args.expected_grade_sha256, args.output)


if __name__ == "__main__":
    main()
