"""Pinned create-only CLI for the fixed conditional-moment decoder probe."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

PACKET = Path(__file__).resolve().parent
BUNDLE = PACKET.parent
SAMPLE_SHA256 = "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
REFERENCE_SHA256 = "a7e19f6358fbd1f61496efd7c1712b80cb7d582f22445a1e3c12988851b323dc"
CONTRACT_SHA256 = "50194152914197c777ce7ace3df415096de4ad6cc4ecbcad96dded36b4332586"
WORKING_ARRAY_ESTIMATE_BYTES = 268_435_456
R5_SOURCE_FILES = (
    "idea_screens_r5/__init__.py",
    "idea_screens_r5/analysis_diagnostics.py",
    "idea_screens_r5/analysis_fit.py",
    "idea_screens_r5/analysis_score.py",
    "idea_screens_r5/analysis_report.py",
    "idea_screens_r5/run_probe.py",
    "idea_screens_r5/test_probe.py",
    "idea_screens_r5/CONTRACT.md",
)


def _digest(path: Path) -> str:
    """Hash a file incrementally without loading a sample array into memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


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
    """Verify the exact preparation manifest and every registered source byte."""
    manifest_path = PACKET / "PREPARATION_MANIFEST.json"
    if _digest(manifest_path) != expected_sha256:
        raise ValueError("R5 preparation manifest SHA256 differs from launch pin")
    manifest = json.loads(manifest_path.read_bytes())
    files = manifest.get("files")
    if not isinstance(files, dict) or not set(R5_SOURCE_FILES) <= files.keys():
        raise ValueError("R5 source manifest lacks the frozen module/contract closure")
    observed = {}
    for name, expected in files.items():
        path = _source_path(name)
        actual = _digest(path)
        if actual != expected:
            raise ValueError(f"pinned R5 source mismatch: {name}: {actual}")
        observed[name] = actual
    return manifest, observed


def _validate_sample_manifest(manifest: dict) -> tuple[list[int], list[int]]:
    """Check the fixed R3 64/32 metadata panel without opening array files."""
    if (
        manifest.get("schema") != "idea_screens_r3.sample.v2"
        or manifest.get("split_sha256")
        != "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
    ):
        raise ValueError("R5 requires the accepted source-bound R3 F0 v2 sample")
    selection = manifest["selection"]
    fit_order = [int(value) for value in selection["fit_full64_component_order"]]
    low32 = [int(value) for value in selection["fit_low32_components"]]
    eval_order = [int(value) for value in selection["eval32_component_order"]]
    records = manifest["records"]
    fit = [item for item in records if item["role"] == "fit"]
    evaluation = [item for item in records if item["role"] == "eval"]
    fit_components = [int(item["component"]) for item in fit]
    eval_components = [int(item["component"]) for item in evaluation]
    if (
        len(records) != 96
        or len(fit) != 64
        or len(set(fit_components)) != 64
        or set(fit_components) != set(fit_order)
        or len(fit_order) != 64
        or len(set(fit_order)) != 64
        or low32 != fit_order[:32]
        or len(low32) != 32
        or len(evaluation) != 32
        or len(set(eval_components)) != 32
        or set(eval_components) != set(eval_order)
        or fit_components
        and set(fit_components) & set(eval_components)
    ):
        raise ValueError("R5 sample membership differs from its fixed disjoint 64/32 panel")
    if any(item["source"] != "train" for item in fit) or any(
        item["source"] != "valid" for item in evaluation
    ):
        raise ValueError("R5 sample role/source binding differs")
    if len({(item["source"], item["group"]) for item in records}) != 96:
        raise ValueError("R5 sample contains duplicate source/group records")
    for item in records:
        files = item.get("files", {})
        if not {"x", "y", "scored"} <= files.keys():
            raise ValueError("R5 sample record lacks bound x/y/scored files")
        if any(not {"name", "sha256", "bytes"} <= value.keys() for value in files.values()):
            raise ValueError("R5 sample file binding is incomplete")
    return fit_order, eval_order


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--expected-sample-sha256", required=True)
    parser.add_argument("--expected-preparation-manifest-sha256", required=True)
    parser.add_argument("--expected-reference-result-sha256", required=True)
    return parser.parse_args()


def _maxrss_kib() -> int:
    """Report process peak resident memory in KiB on Linux or macOS."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value // 1024 if sys.platform == "darwin" else value)


def main() -> None:
    """Validate all stdlib-readable pins, then fit and score the two decoders."""
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.expected_sample_sha256 != SAMPLE_SHA256:
        raise ValueError("R5 launch sample pin is not the accepted R3 sample")
    if args.expected_reference_result_sha256 != REFERENCE_SHA256:
        raise ValueError("R5 launch reference pin differs from frozen R3 RESULT")
    if _digest(PACKET / "CONTRACT.md") != CONTRACT_SHA256:
        raise ValueError("R5 contract bytes differ from the accepted fixed design")
    sample_path = args.sample / "SAMPLE.json"
    sample_sha = _digest(sample_path)
    if sample_sha != SAMPLE_SHA256:
        raise ValueError("R3 sample manifest differs from the fixed R5 sample pin")
    sample_manifest = json.loads(sample_path.read_bytes())
    fit_order, eval_order = _validate_sample_manifest(sample_manifest)
    reference_sha = _digest(args.reference_result)
    if reference_sha != REFERENCE_SHA256:
        raise ValueError("frozen R3 reference RESULT differs from the fixed R5 pin")
    reference = json.loads(args.reference_result.read_bytes())
    if reference.get("sample_manifest_sha256") != sample_sha:
        raise ValueError("R3 reference result does not bind the fixed R5 panel")
    preparation, source_files = _verify_preparation(args.expected_preparation_manifest_sha256)
    if str(BUNDLE) not in sys.path:
        sys.path.insert(0, str(BUNDLE))
    sample_module = importlib.import_module("idea_screens_r1.sample_data")
    from idea_screens_r5.analysis_fit import collect_fit_rows, fit_moment_model
    from idea_screens_r5.analysis_report import build_result, digest, save_model, write_json
    from idea_screens_r5.analysis_score import score_panel

    args.output.mkdir(parents=False, exist_ok=False)
    source_manifest = {
        "schema": "bounded_r5_source_manifest_v1",
        "contract_sha256": CONTRACT_SHA256,
        "sample_manifest_sha256": sample_sha,
        "preparation_manifest_sha256": args.expected_preparation_manifest_sha256,
        "preparation_schema": preparation.get("schema"),
        "preparation_files": source_files,
        "reference_result_sha256": reference_sha,
        "canonical_split_sha256": sample_manifest["split_sha256"],
        "fit_full64_component_order": fit_order,
        "eval32_component_order": eval_order,
        "records": [
            {
                key: item[key]
                for key in (
                    "role",
                    "source",
                    "group",
                    "component",
                    "seq_ix",
                    "scored_rows",
                    "files",
                )
                if key in item
            }
            for item in sample_manifest["records"]
        ],
        "metric_id": "global_wp_clip2_v1",
        "sample_path": str(args.sample.resolve()),
        "reference_result_path": str(args.reference_result.resolve()),
        "working_array_bytes_estimate": WORKING_ARRAY_ESTIMATE_BYTES,
    }
    source_path = args.output / "SOURCE_MANIFEST.json"
    write_json(source_path, source_manifest)
    source_sha = digest(source_path)
    started_wall, started_cpu = time.monotonic(), time.process_time()
    output_created = True
    try:
        write_json(
            args.output / "START.json",
            {
                "utc": dt.datetime.now(dt.UTC).isoformat(),
                "sample_manifest_sha256": sample_sha,
                "preparation_manifest_sha256": args.expected_preparation_manifest_sha256,
                "reference_result_sha256": reference_sha,
                "source_manifest_sha256": source_sha,
                "python": platform.python_version(),
                "numpy": importlib.import_module("numpy").__version__,
                "sequence_count": len(sample_manifest["records"]),
                "working_array_bytes_estimate": WORKING_ARRAY_ESTIMATE_BYTES,
            },
        )
        sequences, loaded_manifest = sample_module.load_sample(args.sample)
        if loaded_manifest != sample_manifest or _digest(sample_path) != sample_sha:
            raise ValueError("R3 sample manifest changed during verified sample loading")
        progress_path = args.output / "incremental_progress.jsonl"
        with progress_path.open("x", encoding="utf-8") as progress_stream:

            def progress(event: dict) -> None:
                progress_stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
                progress_stream.flush()
                os.fsync(progress_stream.fileno())

            fit_features, fit_targets, fit_counts = collect_fit_rows(sequences, fit_order, progress)
            progress({"stage": "fit_rows_collected", **fit_counts})
            model = fit_moment_model(fit_features, fit_targets)
            model_archive = save_model(args.output / "moment_model.npz", model)
            progress({"stage": "fixed_moment_heads_fit", "alpha": 100, "archive": model_archive})
            del fit_features, fit_targets
            scoring, diagnostics = score_panel(sequences, model, progress)
        cost = {
            "total_wall_s": time.monotonic() - started_wall,
            "total_process_cpu_s": time.process_time() - started_cpu,
            "max_rss_kib": _maxrss_kib(),
            "working_array_bytes_estimate": WORKING_ARRAY_ESTIMATE_BYTES,
            "working_array_estimate_scope": "numpy analysis arrays; excludes interpreter and native-library overhead",
        }
        build_result(
            args.output,
            model,
            fit_counts,
            scoring,
            diagnostics,
            reference,
            reference_sha,
            sample_sha,
            source_sha,
            model_archive,
            cost,
        )
    except Exception as exc:
        if output_created and not (args.output / "FAILURE.json").exists():
            with (args.output / "FAILURE.json").open("x", encoding="utf-8") as stream:
                json.dump(
                    {
                        "utc": dt.datetime.now(dt.UTC).isoformat(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "sample_manifest_sha256": sample_sha,
                        "preparation_manifest_sha256": args.expected_preparation_manifest_sha256,
                        "reference_result_sha256": reference_sha,
                    },
                    stream,
                    indent=2,
                    sort_keys=True,
                )
                stream.write("\n")
        raise


if __name__ == "__main__":
    main()
