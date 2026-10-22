"""Exact file and metric helpers for the online relationship screen grader."""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import Any  # NumPy's parsed header does not expose a precise static type.

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from grade_prediction_bank import score  # noqa: E402

from wnn.evaluation import (  # noqa: E402
    paired_component_bootstrap,
    paired_component_moments,
)

ROWS = 20_000
SCORED_FROM_ROW = 99
TIME_BINS = ((99, 500), (500, 2_000), (2_000, 10_000), (10_000, ROWS))
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 271828


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def npy_coverage(path: Path) -> dict[str, Any]:
    """Parse an NPY header and prove that its file has exactly the declared payload."""
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version in ((2, 0), (3, 0)):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"{path}: unsupported NPY version {version}")
        header_bytes = stream.tell()
    if fortran_order:
        raise ValueError(f"{path}: expected C-order NPY data")
    dtype = np.dtype(dtype)
    payload_bytes = math.prod(shape) * dtype.itemsize
    file_bytes = path.stat().st_size
    if file_bytes != header_bytes + payload_bytes:
        raise ValueError(f"{path}: NPY file length differs from header payload coverage")
    return {
        "shape": list(shape),
        "dtype": dtype.str,
        "header_bytes": header_bytes,
        "payload_bytes": payload_bytes,
        "file_bytes": file_bytes,
    }


def prediction_memmap(
    path: Path,
    expected_shape: tuple[int, int, int],
    expected_sha256: str,
    *,
    expected_dtype: str = "<f4",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Verify byte coverage, fp32 shape and file hash before opening a prediction map."""
    coverage = npy_coverage(path)
    if coverage["shape"] != list(expected_shape) or coverage["dtype"] != expected_dtype:
        raise ValueError(
            f"{path}: expected {expected_dtype} {expected_shape}, got "
            f"{coverage['dtype']} {tuple(coverage['shape'])}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{path}: prediction SHA256 differs from its frozen sidecar")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != expected_shape or array.dtype.str != expected_dtype:
        raise ValueError(f"{path}: NPY loader disagrees with parsed header")
    return array, {**coverage, "sha256": actual_sha256}


def official_scores(
    targets: np.ndarray, mask: np.ndarray, prediction: np.ndarray
) -> dict[str, dict[str, float | int | str]]:
    """Return official overall and fixed-bin global WP/T0/T1 scores."""
    scored = np.asarray(mask, dtype=bool) & (np.arange(ROWS)[None, :] >= SCORED_FROM_ROW)
    result = {"overall": score(targets, scored, prediction)}
    for start, stop in TIME_BINS:
        bin_mask = scored.copy()
        bin_mask[:, :start] = False
        bin_mask[:, stop:] = False
        result[f"rows_{start}_{stop}"] = score(targets, bin_mask, prediction)
    return result


def paired_endpoint_moments(
    targets: np.ndarray, prediction: np.ndarray, scored_mask: np.ndarray
) -> np.ndarray:
    """Compute canonical centered per-sequence moments for one fixed endpoint."""
    candidate, _ = paired_component_moments(targets, prediction, prediction, scored_mask)
    return candidate


def paired_component_contrast(
    targets: np.ndarray,
    mask: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    components: np.ndarray,
    candidate_scores: dict[str, dict[str, float | int | str]],
    reference_scores: dict[str, dict[str, float | int | str]],
    candidate_moments: np.ndarray | None = None,
    reference_moments: np.ndarray | None = None,
) -> dict[str, Any]:
    """Report fixed-model target/bin deltas and an overall official-WP component interval."""
    if candidate_moments is None or reference_moments is None:
        candidate_moments, reference_moments = paired_component_moments(
            targets, candidate, reference, mask
        )
    interval = paired_component_bootstrap(
        candidate_moments,
        reference_moments,
        components,
        n_boot=BOOTSTRAP_DRAWS,
        seed=BOOTSTRAP_SEED,
        centered=True,
    )
    deltas: dict[str, dict[str, float]] = {}
    for band, candidate_result in candidate_scores.items():
        reference_result = reference_scores[band]
        deltas[band] = {
            metric: float(candidate_result[metric] - reference_result[metric])
            for metric in ("wp", "t0", "t1")
        }
    return {
        "point_deltas": deltas,
        "overall_official_wp_paired_component_bootstrap": interval,
        "uncertainty_scope": "conditional on these fixed fitted endpoints; no training-seed variance",
    }
