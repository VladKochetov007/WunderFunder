"""Create-only CLI; verify source and panel pins before importing model helpers."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

PACKET = Path(__file__).resolve().parent
BUNDLE = PACKET.parent


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _source_path(name: str) -> Path:
    if name.startswith("canonical:"):
        relative = name.removeprefix("canonical:")
        for entry in sys.path:
            candidate = Path(entry) / relative
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"canonical source not found on sys.path: {relative}")
    return (BUNDLE / name).resolve()


def _verify_preparation(expected_sha256: str) -> tuple[dict, dict[str, str]]:
    manifest_path = PACKET / "PREPARATION_MANIFEST.json"
    manifest_sha = _digest(manifest_path)
    if manifest_sha != expected_sha256:
        raise ValueError("R3 preparation manifest SHA256 differs from the launch pin")
    manifest = json.loads(manifest_path.read_bytes())
    observed = {}
    for name, expected in manifest["files"].items():
        path = _source_path(name)
        actual = _digest(path)
        if actual != expected:
            raise ValueError(f"pinned source mismatch: {name}: {actual}")
        observed[name] = actual
    return manifest, observed


def _sample_records(manifest: dict) -> list[dict]:
    selected = []
    for item in manifest["records"]:
        fields = ("role", "source", "group", "component", "seq_ix", "scored_rows", "files")
        selected.append({key: item[key] for key in fields if key in item})
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sample-sha256", required=True)
    parser.add_argument("--expected-preparation-manifest-sha256", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sample_manifest_path = args.sample / "SAMPLE.json"
    sample_sha256 = _digest(sample_manifest_path)
    if sample_sha256 != args.expected_sample_sha256:
        raise ValueError("R3 sample manifest differs from the launch pin")
    preparation, observed_sources = _verify_preparation(args.expected_preparation_manifest_sha256)
    sample_manifest = json.loads(sample_manifest_path.read_bytes())
    if (
        sample_manifest.get("split_sha256")
        != "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
    ):
        raise ValueError("R3 sample is not bound to canonical F0")
    selection = sample_manifest["selection"]
    fit_low32 = selection["fit_low32_components"]
    fit_order = selection["fit_full64_component_order"]
    if (
        len(fit_low32) != 32
        or len(fit_order) != 64
        or fit_order[:32] != fit_low32
        or len(set(fit_order)) != 64
        or len(sample_manifest["records"]) != 96
    ):
        raise ValueError("R3 sample membership is not the frozen 64/32 panel")
    fit_records = [item for item in sample_manifest["records"] if item["role"] == "fit"]
    eval_records = [item for item in sample_manifest["records"] if item["role"] == "eval"]
    fit_components = [int(item["component"]) for item in fit_records]
    eval_components = [int(item["component"]) for item in eval_records]
    if (
        len(fit_records) != 64
        or len(set(fit_components)) != 64
        or set(fit_components) != set(fit_order)
        or len(eval_records) != 32
        or len(set(eval_components)) != 32
        or set(fit_components) & set(eval_components)
    ):
        raise ValueError("R3 membership has duplicate, missing, or overlapping components")
    from idea_screens_r3.ridge_probe import run

    sample_started = time.process_time()
    sample_module = importlib.import_module("idea_screens_r1.sample_data")
    sequences, loaded_manifest = sample_module.load_sample(args.sample)
    sample_load_cpu = time.process_time() - sample_started
    if loaded_manifest != sample_manifest or _digest(sample_manifest_path) != sample_sha256:
        raise ValueError("R3 panel manifest changed while selected array hashes were checked")
    source_manifest = {
        "schema": "nf19.r3_source_manifest.v1",
        "sample_path": str(args.sample.resolve()),
        "sample_manifest_sha256": sample_sha256,
        "preparation_manifest_sha256": args.expected_preparation_manifest_sha256,
        "preparation_manifest_files": observed_sources,
        "source_snapshot": preparation.get("source_snapshot", "source-a266b036"),
        "canonical_split_sha256": sample_manifest["split_sha256"],
        "selection": sample_manifest["selection"],
        "sample_load_cpu_s": sample_load_cpu,
        "sample_environment": {
            "python": platform.python_version(),
            "numpy": __import__("numpy").__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "records": _sample_records(sample_manifest),
        "metric_id": "global_wp_clip2_v1",
    }
    sample_binding = {
        "sha256": sample_sha256,
        "fit_low32_components": [int(value) for value in fit_low32],
        "fit_full64_component_order": [int(value) for value in fit_order],
        "records": source_manifest["records"],
    }
    output_was_present = args.output.exists()
    try:
        run(sequences, args.output, sample_binding, source_manifest)
    except Exception as exc:
        if (
            not output_was_present
            and args.output.is_dir()
            and not (args.output / "FAILURE.json").exists()
        ):
            _write_json(
                args.output / "FAILURE.json",
                {
                    "utc": dt.datetime.now(dt.UTC).isoformat(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        raise


if __name__ == "__main__":
    main()
