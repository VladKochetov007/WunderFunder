"""Hash-gated CPU CLI for the fixed target-free R10 support screen."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
import platform
import posixpath
import sys
import time
from pathlib import Path

import numpy as np

from .geometry import (
    BOUNDARY_TOLERANCE,
    PAIR_COLUMNS,
    ROW_STRIDE,
    WINDOW_ROWS,
    classify_window,
    query_rows,
)
from .panel_io import (
    MEMBERSHIP_SHA256,
    SAMPLE_SHA256,
    InputBinding,
    ValidatedPanel,
    input_file_receipts,
    load_and_validate_metadata,
    map_fit_arrays,
    sha256_file,
    verify_fit_inputs,
)

DESIGN_REVIEW_SHA256 = "129b30e6dc41e0af08803d333bc64cf429922692efca24c04e7a1971d2ce002d"
SOURCE_SNAPSHOT = "a266b036"
MEMORY_HIGH_BYTES = 1_400_000_000
MEMORY_EXTRA_BUDGET_BYTES = 256_000_000
TIMEOUT_SECONDS = 600
CLASS_NAMES = ("box_outside", "box_inside_hull_outside", "hull_inside")
RUNTIME_PY = {
    "idea_screens_r10": "idea_screens_r10/__init__.py",
    "idea_screens_r10.geometry": "idea_screens_r10/geometry.py",
    "idea_screens_r10.panel_io": "idea_screens_r10/panel_io.py",
    "idea_screens_r10.run_probe": "idea_screens_r10/run_probe.py",
}


def smoke_imports(bundle_root: Path, source_root: Path) -> dict[str, object]:
    """Import the complete R10 runtime closure without loading sample arrays."""
    before = set(sys.modules)
    package_root = bundle_root.resolve()
    imported: dict[str, str] = {}
    for name, relative in RUNTIME_PY.items():
        module = importlib.import_module(name)
        expected_path = (package_root / relative).resolve()
        actual_path = Path(str(module.__file__)).resolve()
        if actual_path != expected_path:
            raise RuntimeError(f"R10 import resolved outside the staged source: {name}")
        imported[name] = str(actual_path)
    forbidden = ("wnn", "pyarrow", "torch")
    newly_loaded = sorted(
        name
        for name in set(sys.modules) - before
        if any(name == root or name.startswith(root + ".") for root in forbidden)
    )
    if newly_loaded:
        raise RuntimeError(f"R10 import loaded forbidden runtime modules: {newly_loaded}")
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    return {
        "runtime_import_paths": imported,
        "new_forbidden_imports": newly_loaded,
        "array_reads": 0,
    }


def verify_preparation_manifest(
    package_dir: Path, bundle_root: Path, source_root: Path, expected_sha256: str
) -> tuple[dict[str, object], dict[str, str]]:
    """Verify every staged R10 byte and the semantic-only canonical feature source."""
    manifest_path = package_dir / "PREPARATION_MANIFEST.json"
    if sha256_file(manifest_path) != expected_sha256:
        raise ValueError("R10 preparation manifest SHA-256 differs from the CLI pin")
    manifest = json.loads(manifest_path.read_bytes())
    if not isinstance(manifest, dict) or manifest.get("schema") != "nf19.r10_preparation.v1":
        raise ValueError("unexpected R10 preparation manifest schema")
    if manifest.get("runtime_py") != RUNTIME_PY:
        raise ValueError("R10 runtime module map differs from the frozen four-module closure")
    files = manifest.get("files")
    if not isinstance(files, dict) or "canonical:wnn/data.py" not in files:
        raise ValueError("R10 manifest lacks its canonical semantic source pin")
    verified: dict[str, str] = {}
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("R10 manifest file pins must be string pairs")
        if relative.startswith("canonical:"):
            path = source_root / "src" / relative.removeprefix("canonical:")
        else:
            path = bundle_root / relative
            if not path.resolve().is_relative_to(bundle_root.resolve()):
                raise ValueError(f"R10 manifest path escapes its staged bundle: {relative}")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"R10 source pin differs: {relative}")
        verified[relative] = expected
    return manifest, verified


def parse_args() -> argparse.Namespace:
    """Parse fixed sample, output, and preparation identities."""
    parser = argparse.ArgumentParser(description="Fixed target-free R10 support screen")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sample-sha256", required=True)
    parser.add_argument("--expected-preparation-manifest-sha256", required=True)
    return parser.parse_args()


def _run(args: argparse.Namespace) -> None:
    """Validate identities first, then stream only fixed fit x/mask arrays."""
    sample_dir, output = args.sample.resolve(), args.output.resolve()
    if args.expected_sample_sha256 != SAMPLE_SHA256:
        raise ValueError("CLI sample identity differs from the accepted R3 panel")
    if sample_dir.parent.name != "research_tmp":
        raise ValueError("R10 sample must be under physical B/research_tmp")
    base, scratch = sample_dir.parent.parent, sample_dir.parent
    if output.parent != scratch:
        raise ValueError("R10 output must be a direct sibling of the sample under research_tmp")
    source_root = base / f"source-{SOURCE_SNAPSHOT}"
    if not source_root.is_dir():
        raise FileNotFoundError(f"canonical source snapshot is missing: {source_root}")
    package_dir = Path(__file__).resolve().parent
    bundle_root = package_dir.parent
    _, pins = verify_preparation_manifest(
        package_dir, bundle_root, source_root, args.expected_preparation_manifest_sha256
    )
    _require_runtime_environment()
    _check_physical_output(output, scratch)
    output.mkdir()
    started = time.monotonic()
    deadline = started + TIMEOUT_SECONDS
    _write_json(
        output / "START.json",
        {
            "schema": "nf19.r10_support.start.v1",
            "created_utc": dt.datetime.now(dt.UTC).isoformat(),
            "design_sha256": pins.get("idea_screens_r10/DESIGN.md"),
            "design_review_sha256": DESIGN_REVIEW_SHA256,
            "preparation_manifest_sha256": args.expected_preparation_manifest_sha256,
            "sample_sha256": SAMPLE_SHA256,
            "membership_sha256": MEMBERSHIP_SHA256,
            "resource_cap": {
                "cpu_cores": 0.5,
                "memory_high_bytes": MEMORY_HIGH_BYTES,
                "memory_max_bytes": 1_500_000_000,
                "timeout_s": TIMEOUT_SECONDS,
            },
        },
    )
    (output / "PROGRESS.jsonl").open("xb").close()
    try:
        memory_before = _guard_memory("before_sample_metadata_and_arrays")
        panel = load_and_validate_metadata(sample_dir, args.expected_sample_sha256)
        bindings = verify_fit_inputs(sample_dir, panel)
        if time.monotonic() > deadline:
            raise TimeoutError("R10 exceeded its 600-second bound while hashing fit inputs")
        memory_after_hash = _guard_memory("after_fit_input_hashes_before_mapping")
        source = _source_receipt(
            panel,
            bindings,
            pins,
            args.expected_preparation_manifest_sha256,
            memory_before,
            memory_after_hash,
            started,
        )
        _write_json(output / "SOURCE.json", source)
        _append_json_line(
            output / "PROGRESS.jsonl",
            {
                "event": "fit_input_hashes_verified",
                "files": 128,
                "elapsed_s": time.monotonic() - started,
            },
        )
        per_component: list[dict[str, object]] = []
        mapped_input_count = 0
        for sequence_index, binding in enumerate(bindings):
            if time.monotonic() > deadline:
                raise TimeoutError("R10 exceeded its 600-second bound during fixed queries")
            fit = map_fit_arrays(binding)
            coordinates = np.asarray(fit.x[:, (0, 52)], dtype=np.float64)
            if not np.isfinite(coordinates).all():
                raise ValueError("fixed R10 support coordinates contain nonfinite values")
            mapped_input_count += 2
            rows = query_rows(fit.scored)
            if rows.size != 493:
                raise ValueError("R10 fixed query count changed from 493 per fit sequence")
            stats = _sequence_summary(fit.record, coordinates, rows, deadline)
            per_component.append(stats)
            _append_json_line(
                output / "PROGRESS.jsonl",
                {
                    "event": "component_complete",
                    "component": stats["component"],
                    "query_count": stats["query_count"],
                    "completed": sequence_index + 1,
                    "elapsed_s": time.monotonic() - started,
                },
            )
            del coordinates, fit
            _guard_memory(f"after_component_{sequence_index + 1}")
        if mapped_input_count != 128:
            raise RuntimeError(f"R10 mapped {mapped_input_count} input NPYs, expected 128")
        result = _result_payload(per_component, panel.fit_component_order, started)
        result_sha = _write_json(output / "RESULT.json", result)
        _write_json(
            output / "DONE.json",
            {
                "schema": "nf19.r10_support.done.v1",
                "status": "complete",
                "result_sha256": result_sha,
                "result_bytes": (output / "RESULT.json").stat().st_size,
                "elapsed_s": time.monotonic() - started,
                "query_count": result["query_count"],
                "input_npy_files_hashed": 128,
                "input_npy_files_mapped": mapped_input_count,
            },
        )
    except Exception as exc:
        _write_json(
            output / "FAILURE.json",
            {
                "schema": "nf19.r10_support.failure.v1",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_s": time.monotonic() - started,
            },
        )
        raise


def _source_receipt(
    panel: ValidatedPanel,
    bindings: list[InputBinding],
    pins: dict[str, str],
    preparation_manifest_sha256: str,
    memory_before: dict[str, int | str | None],
    memory_after_hash: dict[str, int | str | None],
    started: float,
) -> dict[str, object]:
    return {
        "schema": "nf19.r10_support.source.v1",
        "source_snapshot": SOURCE_SNAPSHOT,
        "source_pins": pins,
        "preparation_manifest_sha256": preparation_manifest_sha256,
        "sample_sha256": SAMPLE_SHA256,
        "membership_sha256": MEMBERSHIP_SHA256,
        "fit_component_order": list(panel.fit_component_order),
        "eval_component_order_metadata_only": list(panel.eval_component_order),
        "fit_sequences": len(panel.fit_records),
        "eval_sequences_metadata_only": 32,
        "input_npy_file_count_hashed": 128,
        "input_npy_file_count_expected_to_map": 128,
        "input_npy_files": input_file_receipts(bindings),
        "query_spec": {
            "columns": list(PAIR_COLUMNS),
            "window_rows": WINDOW_ROWS,
            "stride": ROW_STRIDE,
            "minimum_t": WINDOW_ROWS,
            "queries_per_sequence": 493,
            "total_queries": 31_552,
        },
        "target_values_accessed": False,
        "y_files_opened": 0,
        "evaluation_arrays_read": 0,
        "official_scorer_calls": 0,
        "model_loads": 0,
        "training_updates": 0,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "memory_before_inputs": memory_before,
        "memory_after_input_hashes": memory_after_hash,
        "elapsed_before_geometry_s": time.monotonic() - started,
    }


def _sequence_summary(
    record: dict[str, object], coordinates: np.ndarray, rows: np.ndarray, deadline: float
) -> dict[str, object]:
    counts = {name: 0 for name in CLASS_NAMES}
    hull_kinds = {"point": 0, "line": 0, "area": 0}
    vertex_sum = 0
    area_ratio_sum = 0.0
    area_ratio_count = 0
    for index, row_value in enumerate(rows):
        if index % 32 == 0 and time.monotonic() > deadline:
            raise TimeoutError("R10 exceeded its 600-second bound during a sequence")
        row = int(row_value)
        geometry = classify_window(coordinates[row - WINDOW_ROWS : row], coordinates[row])
        counts[geometry.category] += 1
        hull_kinds[geometry.hull_kind] += 1
        vertex_sum += geometry.hull_vertices
        if geometry.hull_box_area_ratio is not None:
            area_ratio_sum += geometry.hull_box_area_ratio
            area_ratio_count += 1
    query_count = int(rows.size)
    if sum(counts.values()) != query_count:
        raise RuntimeError("R10 geometry classes are not exhaustive")
    return {
        "component": int(record["component"]),
        "source": str(record["source"]),
        "group": int(record["group"]),
        "seq_ix": int(record["seq_ix"]),
        "query_count": query_count,
        "class_counts": counts,
        "class_rates": {key: value / query_count for key, value in counts.items()},
        "hull_dimension_counts": hull_kinds,
        "mean_hull_vertices": vertex_sum / query_count,
        "nondegenerate_box_count": area_ratio_count,
        "mean_hull_box_area_ratio": area_ratio_sum / area_ratio_count if area_ratio_count else None,
    }


def _result_payload(
    per_component: list[dict[str, object]], fit_order: tuple[int, ...], started: float
) -> dict[str, object]:
    by_component = {int(item["component"]): item for item in per_component}
    if tuple(by_component) != fit_order or len(per_component) != 64:
        raise ValueError("R10 component output order differs from the fixed fit panel")
    if sum(int(item["query_count"]) for item in per_component) != 31_552:
        raise ValueError("R10 fixed-panel query count differs from the frozen 31,552 rows")
    all_summary = _group_summary(per_component)
    first_half = _group_summary([by_component[value] for value in fit_order[:32]])
    second_half = _group_summary([by_component[value] for value in fit_order[32:]])
    return {
        "schema": "nf19.r10_support.result.v1",
        "status": "complete",
        "design_review_sha256": DESIGN_REVIEW_SHA256,
        "sample_sha256": SAMPLE_SHA256,
        "membership_sha256": MEMBERSHIP_SHA256,
        "fit_component_order": list(fit_order),
        "query_count": sum(int(item["query_count"]) for item in per_component),
        "queries_are_serially_dependent": True,
        "target_values_accessed": False,
        "evaluation_arrays_read": 0,
        "official_scorer_calls": 0,
        "model_loads": 0,
        "training_updates": 0,
        "classes": list(CLASS_NAMES),
        "all_64_equal_component_and_pooled_summary": all_summary,
        "fixed_first32_component_summary": first_half,
        "fixed_last32_component_summary": second_half,
        "per_component": per_component,
        "boundary_tolerance_normalized": BOUNDARY_TOLERANCE,
        "geometry_method": "monotone-chain hull with fixed normalized float64 cross-product tolerance",
        "interpretation_limit": "Input geometry prevalence only; no target utility, prediction, transfer or outside-panel claim.",
        "elapsed_total_s": time.monotonic() - started,
    }


def _group_summary(items: list[dict[str, object]]) -> dict[str, object]:
    total = {name: 0 for name in CLASS_NAMES}
    per_component_rates: dict[str, list[float]] = {name: [] for name in CLASS_NAMES}
    query_count = 0
    for item in items:
        component_count = int(item["query_count"])
        query_count += component_count
        counts = item["class_counts"]
        rates = item["class_rates"]
        assert isinstance(counts, dict) and isinstance(rates, dict)
        for name in CLASS_NAMES:
            total[name] += int(counts[name])
            per_component_rates[name].append(float(rates[name]))
    return {
        "component_count": len(items),
        "query_count": query_count,
        "class_counts": total,
        "pooled_query_rates": {name: total[name] / query_count for name in CLASS_NAMES},
        "equal_component_mean_rates": {
            name: sum(per_component_rates[name]) / len(items) for name in CLASS_NAMES
        },
        "per_component_rate_min": {name: min(per_component_rates[name]) for name in CLASS_NAMES},
        "per_component_rate_max": {name: max(per_component_rates[name]) for name in CLASS_NAMES},
    }


def _require_runtime_environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        raise RuntimeError("R10 must run without visible CUDA devices")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"R10 requires {name}=1")


def _check_physical_output(output: Path, scratch: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    if output.parent.resolve() != scratch.resolve() or not scratch.is_dir():
        raise ValueError("R10 output must be a new direct child of existing research_tmp")
    mounts = []
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 3 and scratch.resolve().is_relative_to(Path(fields[1])):
            mounts.append((len(fields[1]), fields[2]))
    if not mounts or max(mounts)[1] in {"tmpfs", "ramfs"}:
        raise ValueError("R10 output storage is RAM-backed or cannot be verified")


def _guard_memory(stage: str) -> dict[str, int | str | None]:
    current = _own_cgroup_current_bytes()
    if current + MEMORY_EXTRA_BUDGET_BYTES > MEMORY_HIGH_BYTES:
        raise MemoryError(f"R10 memory margin missing at {stage}: current={current}")
    return {
        "stage": stage,
        "own_cgroup_current_bytes": current,
        "process_rss_bytes": _process_rss_bytes(),
        "conservative_extra_budget_bytes": MEMORY_EXTRA_BUDGET_BYTES,
        "memory_high_bytes": MEMORY_HIGH_BYTES,
    }


def _own_cgroup_current_bytes() -> int:
    cgroup_text = Path("/proc/self/cgroup").read_text()
    path = _scoped_cgroup_memory_path(cgroup_text)
    if not path.is_file():
        raise RuntimeError(f"scoped R10 cgroup memory.current is unavailable: {path}")
    return int(path.read_text().strip())


def _scoped_cgroup_memory_path(cgroup_text: str) -> Path:
    """Resolve only the named R10 systemd cgroup; never fall back to host root."""
    entries = []
    for line in cgroup_text.splitlines():
        hierarchy, separator, rest = line.partition(":")
        if not separator:
            continue
        controllers, separator, relative = rest.partition(":")
        if hierarchy == "0" and separator and not controllers:
            entries.append(relative)
    if len(entries) != 1:
        raise RuntimeError("expected exactly one cgroup-v2 membership for this process")
    relative = entries[0]
    if not relative.startswith("/") or relative == "/" or ".." in relative.split("/"):
        raise RuntimeError("R10 requires a non-root absolute cgroup-v2 membership")
    normalized = posixpath.normpath(relative)
    parts = normalized.split("/")
    required = {
        "wnn-campaign.slice",
        "computer-vision-homework-eda19-r10-v1.service",
    }
    if not required.issubset(parts):
        raise RuntimeError("process is outside the required scoped R10 cgroup")
    return Path("/sys/fs/cgroup") / normalized.lstrip("/") / "memory.current"


def _process_rss_bytes() -> int | None:
    status = Path("/proc/self/status")
    if not status.is_file():
        return None
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def _write_json(path: Path, value: object) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def _append_json_line(path: Path, value: object) -> None:
    payload = json.dumps(value, sort_keys=True, allow_nan=False).encode() + b"\n"
    with path.open("ab") as stream:
        stream.write(payload)


def main() -> None:
    """Run the one fixed target-free R10 diagnostic."""
    _run(parse_args())


if __name__ == "__main__":
    main()
