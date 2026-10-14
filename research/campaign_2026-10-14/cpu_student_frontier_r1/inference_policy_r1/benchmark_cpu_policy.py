"""Run the official original xteam_h4 complete-sequence callback for one policy."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import platform
import resource
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pyarrow.parquet as pq

MAIN_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
SCORE_PATH = MAIN_ROOT / "research/campaign_2026-10-13/team_models_r1/score_archive.py"
DEFAULT_DATA = MAIN_ROOT / "datasets/valid.parquet"
POLICIES = ("q8vnni", "q8vnni256", "q8vnni_blocked", "q8vnni_fused")


def digest(path: Path) -> str:
    """Hash a file with bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_score_adapter() -> ModuleType:
    """Load the official campaign validator and complete-row callback replay."""
    sys.path.insert(0, str(MAIN_ROOT))
    spec = importlib.util.spec_from_file_location("cpu_policy_score_archive", SCORE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load score adapter: {SCORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_solution(package: Path, policy: str) -> tuple[ModuleType, Any, Path]:
    """Import one package in an isolated process and assert its selected native policy."""
    if "wnn_native" in sys.modules:
        raise RuntimeError("run each package policy in a fresh Python process")
    precision_path = package / "precision.txt"
    selected = precision_path.read_text().strip()
    if selected != policy:
        raise ValueError(f"package selects {selected!r}; expected {policy!r}")
    os.environ.pop("WNN_PRECISION", None)
    os.environ.pop("WNN_NATIVE_LEVEL", None)
    module_name = f"cpu_policy_solution_{policy}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, package / "solution.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load solution from {package}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.PredictionModel()
    actual = model.model.precision
    if actual != policy:
        raise AssertionError(f"selected-policy assertion failed: requested {policy}, got {actual}")
    native = sys.modules.get("wnn_native")
    if native is None or not getattr(native, "__file__", None):
        raise RuntimeError("callback did not load a native extension")
    return module, model, Path(native.__file__).resolve()


def read_sequence(
    score: ModuleType, data_path: Path, group: int
) -> tuple[int, np.ndarray, np.ndarray]:
    """Read one whole source row group, validate its causal mask, and retain raw features."""
    table = pq.ParquetFile(data_path).read_row_group(group, use_threads=False)
    sequence, need = score.validate_sequence(table)
    features = np.column_stack([table[name].to_numpy() for name in score.FEATURE_COLUMNS]).astype(
        np.float32, copy=False
    )
    features = np.ascontiguousarray(features)
    if features.shape != (20_000, 112) or not np.isfinite(features).all():
        raise ValueError(f"invalid raw features for row group {group}: {features.shape}")
    if need.shape != (20_000,) or np.any(need[:99]) or not np.all(need[99:]):
        raise ValueError(f"invalid causal prediction mask for row group {group}")
    return int(sequence), features, np.ascontiguousarray(need, dtype=np.bool_)


def prediction_delta(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    """Report exact equality and a bounded numerical summary for one prediction pair."""
    delta = left.astype(np.float64, copy=False) - right.astype(np.float64, copy=False)
    absolute = np.abs(delta)
    return {
        "bitwise_equal": bool(np.array_equal(left, right)),
        "different_elements": int(np.count_nonzero(delta)),
        "max_abs": float(absolute.max(initial=0.0)),
        "mean_abs": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
    }


def cpu_identity(requested_cpu: int | None) -> dict[str, object]:
    """Pin this callback process to one allowed logical CPU and record feature flags."""
    affinity_before = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    selected = requested_cpu if requested_cpu is not None else min(affinity_before, default=None)
    if selected is not None:
        if selected not in affinity_before:
            raise ValueError(f"CPU {selected} is outside current affinity")
        os.sched_setaffinity(0, {selected})
    records: list[dict[str, object]] = []
    current: dict[str, object] = {}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines() + [""]:
            if not line.strip():
                if current:
                    records.append(current)
                    current = {}
                continue
            key, sep, value = line.partition(":")
            if sep and key.strip() in ("processor", "model name", "flags"):
                current[key.strip()] = value.strip()
    except OSError:
        records = []
    selected_record = next(
        (item for item in records if item.get("processor") == str(selected)),
        records[0] if records else {},
    )
    return {
        "affinity_before": affinity_before,
        "affinity_after": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else [],
        "selected_cpu": selected,
        "model_name": selected_record.get("model name", platform.processor()),
        "cpu_flags": selected_record.get("flags"),
        "platform": platform.platform(),
        "python": sys.version,
    }


def parse_args() -> argparse.Namespace:
    """Parse one fixed policy package and a short list of complete sequence groups."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--groups", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Measure and save complete original callbacks with explicit policy/reset assertions."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    identity = cpu_identity(args.cpu)
    args.out.mkdir(parents=True)
    score = load_score_adapter()
    solution, model, native_path = load_solution(args.package, args.policy)
    entries = []
    for group in args.groups:
        sequence, features, need = read_sequence(score, args.data, group)
        started = time.perf_counter()
        prediction = score.replay(model, sequence, features, need)
        callback_seconds = time.perf_counter() - started
        fresh_model = solution.PredictionModel()
        if fresh_model.model.precision != args.policy:
            raise AssertionError("fresh callback selected a different policy")
        reset_prediction = score.replay(fresh_model, sequence, features, need)
        reset_equal = bool(np.array_equal(prediction, reset_prediction))
        if not reset_equal:
            raise AssertionError(f"fresh reset replay differs for row group {group}")
        comparison = None
        if args.reference_dir is not None:
            reference_path = args.reference_dir / f"group_{group}.npy"
            reference = np.load(reference_path, mmap_mode="r", allow_pickle=False)
            comparison = prediction_delta(prediction, reference)
        prediction_path = args.out / f"group_{group}.npy"
        np.save(prediction_path, prediction, allow_pickle=False)
        entries.append(
            {
                "row_group": group,
                "sequence_id": sequence,
                "input_sha256": hashlib.sha256(memoryview(features).cast("B")).hexdigest(),
                "need_sha256": hashlib.sha256(memoryview(need).cast("B")).hexdigest(),
                "prediction_sha256": digest(prediction_path),
                "prediction_shape": list(prediction.shape),
                "prediction_dtype": str(prediction.dtype),
                "callback_seconds": callback_seconds,
                "seconds_per_row": callback_seconds / len(features),
                "fresh_reset_bitwise": reset_equal,
                "comparison_to_reference": comparison,
            }
        )
        del model
        model = fresh_model
    receipt = {
        "policy_requested": args.policy,
        "policy_observed": model.model.precision,
        "package_directory": str(args.package.resolve()),
        "archive": str(args.archive.resolve()) if args.archive else None,
        "archive_sha256": digest(args.archive) if args.archive else None,
        "solution_sha256": digest(args.package / "solution.py"),
        "precision_sha256": digest(args.package / "precision.txt"),
        "native_library": str(native_path),
        "native_library_sha256": digest(native_path),
        "selected_native_level": getattr(solution, "_LEVEL", None),
        "host": identity,
        "data_path": str(args.data.resolve()),
        "data_sha256": digest(args.data),
        "groups": entries,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scope": "complete supplied-validation callback runtime/parity probe; no quality grade",
    }
    (args.out / "RESULT.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    logging.info("wrote %s", args.out / "RESULT.json")


if __name__ == "__main__":
    main()
