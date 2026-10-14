from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from wnn.quantization import quantize_rows, quantize_rows_q4

CHECKPOINTS = {
    "clean_f0_control_h256_l2_cp5": (
        "artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/model_ep5.pt",
        "e30577ffb39d061bc40043f68a90285965bd2cf790f6ba5947ba78c60b776eda",
    ),
    "clean_f0_muon_teacher_h384_l2_cp2": (
        "artifacts/runs/clean_t3_f0_muonall_h384_e3_s3_r5/model_ep2.pt",
        "201221c6b4140dda70b078db2a9d7a7246ec40b813f054570d195cecefc61582",
    ),
}


def sha256_file(path: Path) -> str:
    """Hash a bounded, existing checkpoint without an extra payload copy."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_matrix(matrix: np.ndarray) -> dict:
    """Measure best rank-approximation energy and exact row quantizer distortion."""
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("expected a finite two-dimensional matrix")
    singular = np.linalg.svd(values.astype(np.float64), compute_uv=False)
    energy = singular**2
    total = float(energy.sum())
    if total <= 0:
        raise ValueError("zero matrix has no relative energy diagnostic")
    cumulative = np.cumsum(energy) / total
    rows, columns = values.shape
    rank_records = []
    for fraction in (0.125, 0.25, 0.5, 0.75):
        rank = max(1, int(min(rows, columns) * fraction))
        rank_records.append(
            {
                "rank": rank,
                "relative_frobenius_error": float(np.sqrt(energy[rank:].sum() / total)),
                "factor_matrix_elements": rank * (rows + columns),
                "dense_matrix_elements": rows * columns,
            }
        )
    quantization = {}
    for bits, encode in ((8, quantize_rows), (4, quantize_rows_q4)):
        codes, scales = encode(values)
        represented = codes.astype(np.float32) * scales[:, None]
        error = represented.astype(np.float64) - values.astype(np.float64)
        quantization[f"q{bits}"] = {
            "relative_frobenius_error": float(np.sqrt(np.square(error).sum() / total)),
            "maximum_absolute_error": float(np.max(np.abs(error))),
            "row_scale_min": float(scales.min()),
            "row_scale_max": float(scales.max()),
        }
    return {
        "shape": [rows, columns],
        "spectral_norm": float(singular[0]),
        "frobenius_norm": float(np.sqrt(total)),
        "stable_rank": total / float(energy[0]),
        "rank_for_95_percent_energy": int(np.searchsorted(cumulative, 0.95)) + 1,
        "rank_for_99_percent_energy": int(np.searchsorted(cumulative, 0.99)) + 1,
        "low_rank": rank_records,
        "row_quantization": quantization,
    }


def main() -> None:
    """Inspect fixed historical weights; do not fit, export, or grade a candidate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    started = time.monotonic()
    results = {}
    for name, (relative, expected_sha) in CHECKPOINTS.items():
        path = args.root / relative
        if path.stat().st_size > 64_000_000 or sha256_file(path) != expected_sha:
            raise ValueError(f"checkpoint does not match trusted identity: {relative}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        weights = checkpoint["model"]
        matrices = {}
        for key, tensor in weights.items():
            if tensor.ndim != 2 or not key.endswith("weight") and "weight_" not in key:
                continue
            values = tensor.detach().numpy()
            matrices[key] = describe_matrix(values)
            if key.startswith("rnn.weight_"):
                for gate, block in zip(
                    ("reset", "update", "candidate"), np.split(values, 3), strict=True
                ):
                    matrices[f"{key}/{gate}"] = describe_matrix(block)
        results[name] = {
            "checkpoint": relative,
            "checkpoint_sha256": expected_sha,
            "matrices": matrices,
        }
    receipt = {
        "recorded_utc": datetime.now(UTC).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
        "quantization_source_sha256": sha256_file(args.root / "src/wnn/quantization.py"),
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "results": results,
        "interpretation": "Unweighted matrix geometry only. No activation/state distribution, fitting, quality, causal perturbation, runtime, or target-host capacity measurement. Checkpoints differ in training recipe and width; not a controlled optimizer comparison.",
    }
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
