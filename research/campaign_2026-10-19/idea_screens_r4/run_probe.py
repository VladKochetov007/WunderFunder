"""Pinned create-only CLI for the four fixed R4 random-feature cases."""

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
CONTRACT_SHA256 = "8af9ae4223d236987bd873fe041fc8e15014126d68852f2d335e24fff4abd38f"


def _maxrss_kib() -> int:
    """Return process peak resident memory in KiB on macOS and Linux."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value // 1024 if sys.platform == "darwin" else value)


def _digest(path: Path) -> str:
    """Hash a source file or bounded manifest without reading it all at once."""
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
    manifest_path = PACKET / "PREPARATION_MANIFEST.json"
    if _digest(manifest_path) != expected_sha256:
        raise ValueError("R4 preparation manifest SHA256 differs from launch pin")
    manifest = json.loads(manifest_path.read_bytes())
    observed = {}
    for name, expected in manifest["files"].items():
        path = _source_path(name)
        actual = _digest(path)
        if actual != expected:
            raise ValueError(f"pinned R4 source mismatch: {name}: {actual}")
        observed[name] = actual
    return manifest, observed


def _validate_sample_manifest(manifest: dict) -> list[int]:
    if manifest.get("schema") != "idea_screens_r3.sample.v2":
        raise ValueError("R4 requires the accepted R3 v2 sample schema")
    selection = manifest["selection"]
    low32 = [int(value) for value in selection["fit_low32_components"]]
    fit_order = [int(value) for value in selection["fit_full64_component_order"]]
    records = manifest["records"]
    fit = [item for item in records if item["role"] == "fit"]
    evaluation = [item for item in records if item["role"] == "eval"]
    fit_components = [int(item["component"]) for item in fit]
    eval_components = [int(item["component"]) for item in evaluation]
    if (
        len(fit) != 64
        or len(set(fit_components)) != 64
        or set(fit_components) != set(fit_order)
        or len(fit_order) != 64
        or len(set(fit_order)) != 64
        or low32 != fit_order[:32]
        or len(low32) != 32
        or len(evaluation) != 32
        or len(set(eval_components)) != 32
        or set(fit_components) & set(eval_components)
    ):
        raise ValueError("R4 sample membership is not the fixed disjoint 64/32 component panel")
    if any(item["source"] != "train" for item in fit) or any(
        item["source"] != "valid" for item in evaluation
    ):
        raise ValueError("R4 sample role/source binding differs")
    if any(not {"x", "y", "scored"} <= item["files"].keys() for item in records):
        raise ValueError("R4 sample record lacks a bound x/y/scored file")
    return fit_order


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--expected-sample-sha256", required=True)
    parser.add_argument("--expected-preparation-manifest-sha256", required=True)
    parser.add_argument("--expected-reference-result-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    """Verify all fixed pins before model/sample imports, then run bounded analysis."""
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.expected_sample_sha256 != SAMPLE_SHA256:
        raise ValueError("R4 launch sample pin is not the accepted R3 sample")
    if args.expected_reference_result_sha256 != REFERENCE_SHA256:
        raise ValueError("R4 launch reference pin differs from frozen R3 result")
    if _digest(PACKET / "CONTRACT.md") != CONTRACT_SHA256:
        raise ValueError("R4 contract bytes differ from the accepted fixed design")
    sample_path = args.sample / "SAMPLE.json"
    sample_sha = _digest(sample_path)
    if sample_sha != SAMPLE_SHA256:
        raise ValueError("R3 sample manifest differs from fixed R4 sample pin")
    sample_manifest = json.loads(sample_path.read_bytes())
    fit_order = _validate_sample_manifest(sample_manifest)
    if (
        sample_manifest.get("split_sha256")
        != "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
    ):
        raise ValueError("R4 sample does not bind canonical F0")
    reference_sha = _digest(args.reference_result)
    if reference_sha != REFERENCE_SHA256:
        raise ValueError("R3 reference RESULT differs from fixed R4 pin")
    reference = json.loads(args.reference_result.read_bytes())
    if reference.get("sample_manifest_sha256") != sample_sha:
        raise ValueError("R3 reference result does not bind the pinned R4 sample")
    preparation, source_files = _verify_preparation(args.expected_preparation_manifest_sha256)

    if str(BUNDLE) not in sys.path:
        sys.path.insert(0, str(BUNDLE))
    sample_module = importlib.import_module("idea_screens_r1.sample_data")
    output_created = False
    try:
        args.output.mkdir(parents=False, exist_ok=False)
        output_created = True
        source_manifest = {
            "schema": "bounded_r4_source_manifest_v1",
            "contract_sha256": CONTRACT_SHA256,
            "sample_manifest_sha256": sample_sha,
            "preparation_manifest_sha256": args.expected_preparation_manifest_sha256,
            "preparation_files": source_files,
            "preparation_schema": preparation.get("schema"),
            "reference_result_path": str(args.reference_result.resolve()),
            "reference_result_sha256": reference_sha,
            "canonical_split_sha256": sample_manifest["split_sha256"],
            "fit_full64_component_order": fit_order,
            "selection": sample_manifest["selection"],
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
        }
        from idea_screens_r4.analysis_fit import collect_fit_rows, cross_validate, refit_final
        from idea_screens_r4.analysis_report import build_result, digest, save_models, write_json
        from idea_screens_r4.analysis_score import score_outer

        source_path = args.output / "SOURCE_MANIFEST.json"
        write_json(source_path, source_manifest)
        source_sha = digest(source_path)
        started_wall, started_cpu = time.monotonic(), time.process_time()
        sequences, loaded_manifest = sample_module.load_sample(args.sample)
        if loaded_manifest != sample_manifest or _digest(sample_path) != sample_sha:
            raise ValueError("R3 sample manifest changed while selected array hashes were checked")
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
                "sequences": len(sequences),
            },
        )
        progress_path = args.output / "incremental_progress.jsonl"
        with progress_path.open("x", encoding="utf-8") as progress_stream:

            def progress(event: dict) -> None:
                progress_stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
                progress_stream.flush()
                os.fsync(progress_stream.fileno())

            base_rows, targets, row_components, fit_counts = collect_fit_rows(sequences, fit_order)
            progress({"stage": "fit_rows_collected", **fit_counts})
            cv_tables, selections, draws, cv_counts = cross_validate(
                sequences, base_rows, targets, row_components, fit_order, progress
            )
            final_models, normalizers, projection_stats, refit_seconds = refit_final(
                base_rows, targets, selections, draws, progress
            )
            archives = save_models(args.output, final_models, draws, normalizers)
            progress({"stage": "final_models_archived", "archives": archives})
            del base_rows, targets, row_components
            outer, outer_cost = score_outer(sequences, final_models, draws, normalizers, progress)
        build_result(
            args.output,
            sample_sha,
            args.expected_preparation_manifest_sha256,
            reference_sha,
            source_sha,
            source_manifest,
            fit_counts,
            cv_tables,
            selections,
            cv_counts,
            final_models,
            refit_seconds,
            archives,
            draws,
            normalizers,
            projection_stats,
            outer,
            outer_cost,
            time.monotonic() - started_wall,
            time.process_time() - started_cpu,
            _maxrss_kib(),
            reference,
        )
    except Exception as exc:
        if output_created and not (args.output / "FAILURE.json").exists():
            failure = {
                "utc": dt.datetime.now(dt.UTC).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "sample_manifest_sha256": sample_sha,
                "preparation_manifest_sha256": args.expected_preparation_manifest_sha256,
                "reference_result_sha256": reference_sha,
            }
            with (args.output / "FAILURE.json").open("x", encoding="utf-8") as stream:
                json.dump(failure, stream, indent=2, sort_keys=True)
                stream.write("\n")
        raise


if __name__ == "__main__":
    main()
