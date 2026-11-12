"""Render immutable training diagnostics from verified native rank-zero JSONL only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

L4_HOSTS = frozenset(("b300", "l4", "l4-followups", "l4-ema"))
RUN_SCHEMA = "wnn.rank0-training-diagnostics.r3"


def file_sha256(path: Path) -> str:
    """Hash a regular, non-link file in bounded memory."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object without following a symlink."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def iter_leaves(
    value: Any, path: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Yield nested JSON leaf values and their key paths."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from iter_leaves(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_leaves(child, (*path, str(index)))
    else:
        yield path, value


def finite_number(value: Any) -> float | None:
    """Return a finite numeric scalar; booleans are not telemetry values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def validate_native_run(
    artifacts: Path, run_name: str
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    """Bind one native L4 run to its host owner and collection receipt."""
    mirror_root = artifacts / "live-dashboard" / "runs"
    native_root = artifacts / "live-dashboard" / "native"
    mirror, native = mirror_root / run_name, native_root / run_name
    for directory in (mirror, native):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"run directory is absent or a link: {directory}")
    owner_path = mirror / ".sequence_mirror_owner.json"
    native_owner_path = native / ".sequence_mirror_owner.json"
    collection_path = native / "collection_receipt.json"
    owner, native_owner = read_json(owner_path), read_json(native_owner_path)
    receipt = read_json(collection_path)
    host, source_directory = owner.get("host"), owner.get("source_directory")
    if (
        owner != native_owner
        or host not in L4_HOSTS
        or not isinstance(source_directory, str)
        or not run_name.startswith(f"{host}-")
        or receipt.get("host") != host
        or receipt.get("source_directory") != source_directory
    ):
        raise ValueError(
            "native/mirror ownership or source receipt does not match the exact L4 host"
        )
    return mirror, native, owner, receipt


def _numeric_objectives(record: dict[str, Any]) -> dict[str, float]:
    diagnostics = record.get("optimizer_update_diagnostics")
    rates = record.get("learning_rates")
    if not isinstance(rates, dict) and isinstance(diagnostics, dict):
        rates = diagnostics.get("applied_learning_rates")
    values: dict[str, float] = {}
    if isinstance(rates, dict):
        for group, value in rates.items():
            numeric = finite_number(value)
            if numeric is not None:
                values[str(group)] = numeric
    if not values:
        numeric = finite_number(record.get("learning_rate"))
        if numeric is not None:
            values["default"] = numeric
    return values


def parse_rank0(path: Path, collection: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Stream exact rank0 bytes, checking the collection offset/tail and update order."""
    before = path.stat()
    if path.is_symlink() or not path.is_file():
        raise ValueError("native rank0 journal is absent or a link")
    entry_key = f"{path.parent.name}/rank0.jsonl"
    files = collection.get("files")
    entry = files.get(entry_key) if isinstance(files, dict) else None
    if not isinstance(entry, dict) or entry.get("offset") != before.st_size:
        raise ValueError(
            "rank0 byte length does not match the completed native collection receipt"
        )
    marker_hex = entry.get("marker")
    if not isinstance(marker_hex, str):
        raise TypeError("collection receipt does not carry the rank0 tail marker")
    marker = bytes.fromhex(marker_hex)

    digest = hashlib.sha256()
    last_line = b""
    steps: list[int] = []
    epochs: list[int] = []
    rate_series: dict[str, list[float]] = {}
    grad_norm: list[float] = []
    clipped: list[float] = []
    objectives = {"wmse": [], "kd": [], "correlation": []}
    serialized_nonfinite: list[float] = []
    explicit_counter: list[float] = []
    explicit_counter_keys: set[str] = set()
    total_bytes = 0

    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.endswith(b"\n"):
                raise ValueError(f"rank0 line {line_number} is not newline-complete")
            digest.update(line)
            total_bytes += len(line)
            last_line = line
            try:
                record = json.loads(
                    line.decode("utf-8"),
                    parse_constant=lambda token: {
                        "NaN": math.nan,
                        "Infinity": math.inf,
                        "-Infinity": -math.inf,
                    }[token],
                )
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"rank0 line {line_number} is malformed") from error
            if not isinstance(record, dict) or record.get("rank") != 0:
                raise ValueError(f"rank0 line {line_number} is not a rank-zero object")
            step, epoch = record.get("step"), record.get("epoch")
            if type(step) is not int or type(epoch) is not int:
                raise ValueError(f"rank0 line {line_number} lacks integer step/epoch")
            if steps and step <= steps[-1]:
                raise ValueError("rank0 steps are not strictly increasing")
            steps.append(step)
            epochs.append(epoch)

            rates = _numeric_objectives(record)
            for group in rates.keys() - rate_series.keys():
                rate_series[group] = [math.nan] * (len(steps) - 1)
            for group, values in rate_series.items():
                values.append(rates.get(group, math.nan))

            norm = finite_number(record.get("gradient_norm_preclip"))
            grad_norm.append(norm if norm is not None else math.nan)
            raw_clip_flag = record.get("gradient_clipped")
            clip_flag = (
                float(raw_clip_flag)
                if isinstance(raw_clip_flag, bool)
                else finite_number(raw_clip_flag)
            )
            clipped.append(
                1.0
                if clip_flag is not None and clip_flag > 0
                else math.nan
                if clip_flag is None
                else 0.0
            )
            for name, key in (
                ("wmse", "objective_wmse"),
                ("kd", "kd_loss"),
                ("correlation", "objective_correlation_loss"),
            ):
                value = finite_number(record.get(key))
                objectives[name].append(value if value is not None else math.nan)

            nonfinite_count = 0
            explicit_value = 0.0
            for keys, value in iter_leaves(record):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                numeric = float(value)
                if not math.isfinite(numeric):
                    nonfinite_count += 1
                if any("nonfinite" in key.lower().replace("_", "") for key in keys):
                    explicit_counter_keys.add(".".join(keys))
                    if math.isfinite(numeric):
                        explicit_value += numeric
            serialized_nonfinite.append(float(nonfinite_count))
            explicit_counter.append(
                explicit_value if explicit_counter_keys else math.nan
            )

    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or total_bytes != before.st_size
    ):
        raise ValueError("native rank0 changed while diagnostics were being read")
    if not last_line.endswith(marker):
        raise ValueError(
            "native rank0 tail does not match the collection receipt marker"
        )
    if not steps:
        raise ValueError("native rank0 journal has no update records")

    arrays: dict[str, Any] = {
        "step": np.asarray(steps, dtype=np.int64),
        "epoch": np.asarray(epochs, dtype=np.int64),
        "learning_rates": {
            key: np.asarray(value, dtype=np.float64)
            for key, value in rate_series.items()
        },
        "gradient_norm_preclip": np.asarray(grad_norm, dtype=np.float64),
        "gradient_clipped": np.asarray(clipped, dtype=np.float64),
        "objective_wmse": np.asarray(objectives["wmse"], dtype=np.float64),
        "kd_loss": np.asarray(objectives["kd"], dtype=np.float64),
        "objective_correlation_loss": np.asarray(
            objectives["correlation"], dtype=np.float64
        ),
        "serialized_nonfinite_values": np.asarray(
            serialized_nonfinite, dtype=np.float64
        ),
        "explicit_nonfinite_counter": np.asarray(explicit_counter, dtype=np.float64),
    }
    arrays["explicit_nonfinite_counter_keys"] = sorted(explicit_counter_keys)
    return arrays, digest.hexdigest()


def downsample_indices(values: np.ndarray, maximum: int) -> np.ndarray:
    """Return per-series min/max envelope indices, retaining endpoints."""
    count = int(values.size)
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    # Reserve room for endpoints so the result never exceeds ``maximum``.
    bins = max(1, (maximum - 2) // 2)
    edges = np.linspace(0, count, bins + 1, dtype=np.int64)
    selected = {0, count - 1}
    for start, stop in pairwise(edges):
        if stop <= start:
            continue
        block = values[start:stop]
        finite = np.flatnonzero(np.isfinite(block))
        if finite.size == 0:
            selected.add(int(start))
        else:
            selected.add(int(start + finite[np.argmin(block[finite])]))
            selected.add(int(start + finite[np.argmax(block[finite])]))
    return np.asarray(sorted(selected), dtype=np.int64)


def rolling_fraction(flags: np.ndarray, window: int = 200) -> np.ndarray:
    """Compute a trailing clipping fraction over observed update flags."""
    observed = np.isfinite(flags).astype(np.float64)
    values = np.nan_to_num(flags, nan=0.0)
    cumulative_values = np.concatenate(([0.0], np.cumsum(values)))
    cumulative_observed = np.concatenate(([0.0], np.cumsum(observed)))
    indices = np.arange(values.size, dtype=np.int64)
    starts = np.maximum(0, indices - window + 1)
    sums = cumulative_values[indices + 1] - cumulative_values[starts]
    counts = cumulative_observed[indices + 1] - cumulative_observed[starts]
    return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)


def _plot_line(
    ax: Any,
    steps: np.ndarray,
    values: np.ndarray,
    label: str,
    point_limit: int,
    **kwargs: Any,
) -> int:
    indices = downsample_indices(values, point_limit)
    ax.plot(steps[indices], values[indices], label=label, linewidth=1.0, **kwargs)
    return int(indices.size)


def render_diagnostic(
    run_name: str,
    arrays: dict[str, Any],
    metadata: dict[str, Any],
    rank0_sha256: str,
    source_sha256: str,
    collection_sha256: str,
    owner: dict[str, Any],
    output_directory: Path,
    maximum_points: int,
) -> dict[str, Any]:
    """Write one PNG/SVG diagnostics figure and a byte-hash receipt."""
    steps, epochs = arrays["step"], arrays["epoch"]
    point_counts: dict[str, int] = {}
    config = metadata.get("config", {})
    clip_limit = (
        finite_number(config.get("grad_clip")) if isinstance(config, dict) else None
    )
    explicit_keys = arrays["explicit_nonfinite_counter_keys"]

    figure, axes = plt.subplots(
        4, 1, figsize=(13, 14), sharex=True, constrained_layout=True
    )
    figure.suptitle(f"{run_name}: rank-zero training diagnostics", fontsize=14)

    ax = axes[0]
    rates = arrays["learning_rates"]
    for group, series in sorted(rates.items()):
        point_counts[f"learning_rate:{group}"] = _plot_line(
            ax, steps, series, group, maximum_points
        )
    ax.set_yscale("log")
    ax.set_ylabel("Learning rate")
    ax.grid(True, alpha=0.25)
    if rates:
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.01, 0.85, "No learning-rate fields in rank0", transform=ax.transAxes)

    ax = axes[1]
    gradient = arrays["gradient_norm_preclip"]
    point_counts["gradient_norm_preclip"] = _plot_line(
        ax, steps, gradient, "preclip gradient norm", maximum_points
    )
    if clip_limit is not None:
        ax.axhline(
            clip_limit,
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            label=f"clip threshold {clip_limit:g}",
        )
    ax.set_ylabel("Preclip norm")
    ax.grid(True, alpha=0.25)
    clip_ax = ax.twinx()
    clip_rate = rolling_fraction(arrays["gradient_clipped"])
    point_counts["clip_rate_200_updates"] = _plot_line(
        clip_ax,
        steps,
        clip_rate * 100.0,
        "clipped updates / trailing 200 (%)",
        maximum_points,
        color="tab:orange",
        alpha=0.8,
    )
    clip_ax.set_ylabel("Clip rate (%)")
    handles_a, labels_a = ax.get_legend_handles_labels()
    handles_b, labels_b = clip_ax.get_legend_handles_labels()
    if handles_a or handles_b:
        ax.legend(handles_a + handles_b, labels_a + labels_b, loc="best", fontsize=8)

    ax = axes[2]
    point_counts["objective_wmse"] = _plot_line(
        ax,
        steps,
        arrays["objective_wmse"],
        "objective WMSE",
        maximum_points,
        color="tab:blue",
    )
    ax.set_ylabel("Objective WMSE")
    ax.grid(True, alpha=0.25)
    terms_ax = ax.twinx()
    point_counts["kd_loss"] = _plot_line(
        terms_ax,
        steps,
        arrays["kd_loss"],
        f"raw kd_loss (configured KD weight={config.get('kd_weight', 'unset')})",
        maximum_points,
        color="tab:green",
        alpha=0.8,
    )
    point_counts["objective_correlation_loss"] = _plot_line(
        terms_ax,
        steps,
        arrays["objective_correlation_loss"],
        "raw correlation diagnostic",
        maximum_points,
        color="tab:purple",
        alpha=0.8,
    )
    terms_ax.set_ylabel("Raw logged diagnostic")
    handles_a, labels_a = ax.get_legend_handles_labels()
    handles_b, labels_b = terms_ax.get_legend_handles_labels()
    ax.legend(handles_a + handles_b, labels_a + labels_b, loc="best", fontsize=8)

    ax = axes[3]
    if explicit_keys:
        nonfinite_series = arrays["explicit_nonfinite_counter"]
        label = "logged nonfinite counter(s)"
        ylabel = "Logged count"
    else:
        nonfinite_series = arrays["serialized_nonfinite_values"]
        label = "nonfinite numeric values in serialized rank0 record"
        ylabel = "Observed fields / update"
        ax.text(
            0.01,
            0.88,
            "No explicit trainer nonfinite counter is emitted in rank0; this panel checks serialized numeric fields only.",
            transform=ax.transAxes,
            fontsize=8,
        )
    point_counts["nonfinite_diagnostic"] = _plot_line(
        ax, steps, nonfinite_series, label, maximum_points, color="tab:red"
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Optimizer update (native rank0 step)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    transitions = np.flatnonzero(epochs[1:] != epochs[:-1]) + 1
    for boundary in transitions:
        for ax in axes:
            ax.axvline(steps[boundary], color="gray", linewidth=0.5, alpha=0.25)
    axes[-1].set_xlim(int(steps[0]), int(steps[-1]))

    metadata_source = metadata.get("source", {})
    source_sha = (
        metadata_source.get("sha256") if isinstance(metadata_source, dict) else None
    )
    fig_meta = {
        "Title": f"{run_name} rank-zero training diagnostics",
        "Creator": RUN_SCHEMA,
    }
    png_path, svg_path = (
        output_directory / "training_diagnostics.png",
        output_directory / "training_diagnostics.svg",
    )
    figure.savefig(png_path, dpi=150, metadata=fig_meta)
    figure.savefig(svg_path, metadata=fig_meta)
    plt.close(figure)

    nonfinite_values = int(np.nansum(arrays["serialized_nonfinite_values"]))
    receipt = {
        "schema": RUN_SCHEMA,
        "run": run_name,
        "host": owner.get("host"),
        "source_directory": owner.get("source_directory"),
        "source_sha256": source_sha or source_sha256,
        "collection_receipt_sha256": collection_sha256,
        "rank0_sha256": rank0_sha256,
        "records": int(steps.size),
        "step_start": int(steps[0]),
        "step_end": int(steps[-1]),
        "epoch_ids": sorted({int(epoch) for epoch in epochs}),
        "gradient_clip_threshold": clip_limit,
        "explicit_nonfinite_counter_keys": explicit_keys,
        "serialized_nonfinite_numeric_values": nonfinite_values,
        "downsampling": {
            "method": "independent per-series min/max envelope over equal-index bins; first and last points retained",
            "max_points_per_series": maximum_points,
            "observed_points_per_series": point_counts,
        },
        "rank0_source": "verified native collection under live-dashboard/native; its captured byte offset and tail marker match collection_receipt.json",
        "mirror_owner_verified": True,
        "model_loads": 0,
        "gpu_use": 0,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_script_sha256": file_sha256(Path(__file__).resolve()),
        "output_files_sha256": {
            png_path.name: file_sha256(png_path),
            svg_path.name: file_sha256(svg_path),
        },
    }
    return receipt


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Create JSON exactly once within a fresh staging tree."""
    payload = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    with path.open("xb") as stream:
        stream.write(payload)


def process_run(
    artifacts: Path, run_name: str, stage_root: Path, maximum_points: int
) -> dict[str, Any]:
    """Validate, stream and render one completed native run."""
    mirror, native, owner, collection = validate_native_run(artifacts, run_name)
    metadata_path = native / "run_metadata.json"
    metadata = read_json(metadata_path)
    rank0_path = native / "rank0.jsonl"
    arrays, rank0_sha = parse_rank0(rank0_path, collection)
    source_sha = str(metadata.get("source", {}).get("sha256", ""))
    run_output = stage_root / run_name
    run_output.mkdir()
    receipt = render_diagnostic(
        run_name=run_name,
        arrays=arrays,
        metadata=metadata,
        rank0_sha256=rank0_sha,
        source_sha256=source_sha,
        collection_sha256=file_sha256(native / "collection_receipt.json"),
        owner=owner,
        output_directory=run_output,
        maximum_points=maximum_points,
    )
    receipt["run_metadata_sha256"] = file_sha256(metadata_path)
    receipt["native_rank0_bytes"] = rank0_path.stat().st_size
    receipt["mirror_rank0_sha256"] = file_sha256(mirror / "rank0.jsonl")
    if receipt["mirror_rank0_sha256"] != rank0_sha:
        raise ValueError("native and bridge rank0 bytes differ")
    atomic_json(run_output / "diagnostics_receipt.json", receipt)
    return {
        "run": run_name,
        "status": "rendered",
        "rank0_sha256": rank0_sha,
        "receipt_sha256": file_sha256(run_output / "diagnostics_receipt.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output-name", default="rank0-diagnostics-r1")
    parser.add_argument("--max-points", type=int, default=2400)
    args = parser.parse_args()
    if args.max_points < 20 or args.max_points > 20000:
        parser.error("max-points must be in 20..20000")
    artifacts = args.artifacts.resolve()
    live = artifacts / "live-dashboard"
    if not live.is_dir():
        parser.error("artifacts root has no live-dashboard directory")
    final_root = artifacts / args.output_name
    if final_root.exists() or final_root.is_symlink():
        raise FileExistsError(f"immutable output already exists: {final_root}")
    staging = artifacts / f".{args.output_name}.pending-{uuid.uuid4().hex}"
    staging.mkdir()
    runs = sorted(
        path.name
        for path in (live / "native").iterdir()
        if path.is_dir()
        and path.name.endswith("-fit")
        and path.name.startswith(("b300-", "l4-", "l4-followups-", "l4-ema-"))
    )
    if not runs:
        raise RuntimeError("no L4 native fit run directories found")
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for run_name in runs:
        try:
            results.append(process_run(artifacts, run_name, staging, args.max_points))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            errors.append(
                {"run": run_name, "error": f"{type(error).__name__}: {error}"}
            )
    manifest = {
        "schema": RUN_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "artifacts_root": str(artifacts),
        "output_root": str(final_root),
        "source_script_sha256": file_sha256(Path(__file__).resolve()),
        "candidate_runs": len(runs),
        "rendered_runs": len(results),
        "skipped_or_failed_runs": errors,
        "results": results,
        "downsampling": {
            "method": "independent per-series min/max envelope over equal-index bins; first and last points retained",
            "max_points_per_series": args.max_points,
        },
        "cpu_threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "model_loads": 0,
        "gpu_use": 0,
    }
    atomic_json(staging / "manifest.json", manifest)
    if final_root.exists():
        raise FileExistsError(f"immutable output appeared during render: {final_root}")
    os.rename(staging, final_root)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
