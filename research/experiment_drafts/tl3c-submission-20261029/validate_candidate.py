"""Receiver-only score, callback equivalence and capacity evidence for one exact ZIP.

Run each candidate and ISA level in a fresh offline Python 3.11 Linux container.
This file is intentionally not run during local submission preparation.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from checkpoint_export import TIED_FEATURE_SOURCES  # noqa: E402
from validation_io import (  # noqa: E402
    SPLIT_SHA,
    constrain_receiver,
    extract_candidate,
    read_json,
    selected_groups,
    sha256,
)


def arguments() -> argparse.Namespace:
    """Keep a small engineering profile separate from a full scientific score."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("zip", "receipt", "data", "split", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--level", choices=("v2", "avx2", "v3"), required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--max-sequences", type=int)
    scope.add_argument("--full-valid", action="store_true")
    parser.add_argument("--test-sequences", type=int)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-metadata", type=Path)
    parser.add_argument(
        "--comparison-mode",
        choices=("equivalent", "callback-equivalent", "quality"),
        default="equivalent",
    )
    parser.add_argument("--equivalence-atol", type=float, default=1e-5)
    parser.add_argument("--max-wp-drop", type=float, default=0.0002)
    args = parser.parse_args()
    if args.test_sequences is not None and args.test_sequences < 1:
        parser.error("--test-sequences must be positive")
    if not (0 <= args.equivalence_atol < 1 and 0 <= args.max_wp_drop < 1):
        parser.error("comparison thresholds must be finite and in [0,1)")
    if args.comparison_mode == "quality" and not args.full_valid:
        parser.error("quality comparison requires --full-valid")
    if args.comparison_mode == "quality" and args.reference is None:
        parser.error("quality comparison requires a reference prediction bundle")
    return args


def member_signature(receipt: dict) -> list[dict]:
    """Identify the same scientific predictor across f32 and dynamic-int8 packaging."""
    return [
        {
            "run": item["run"],
            "coef": item["coef"],
            "family": item["family"],
            "checkpoint_sha256": item["checkpoint"]["sha256"],
        }
        for item in receipt["members"]
    ]


def evaluate(args: argparse.Namespace, report: dict, package: Path) -> bool:
    """Stream one complete group at a time without running training or importing Torch."""
    import numpy as np
    import pyarrow.parquet as pq
    from validation_runtime import (
        differences,
        load_model,
        load_sequence,
        open_reference,
        replay,
        state_checks,
    )

    from utils import SEQUENCE_LENGTH, GlobalAccumulator

    receipt = read_json(args.receipt)
    report["libraries"] = {"numpy": np.__version__, "pyarrow": sys.modules["pyarrow"].__version__}
    report["artifact"] = extract_candidate(args.zip, receipt, package)
    report["build_receipt_sha256"] = sha256(args.receipt)
    report["member_signature"] = member_signature(receipt)
    report["reference_kind"] = f"callback_{receipt['mode']}"
    groups = selected_groups(args.split, args.max_sequences, args.full_valid)
    data_sha = sha256(args.data)
    if data_sha != args.data_sha256:
        raise ValueError("validation parquet SHA256 mismatch")
    report.update(data_sha256=data_sha, split_sha256=SPLIT_SHA, row_groups=groups)
    reference, metadata = open_reference(
        args.reference, args.reference_metadata, data_sha, groups, SPLIT_SHA
    )
    if metadata is not None and metadata.get("member_signature") != report["member_signature"]:
        raise ValueError("reference does not bind the same checkpoint/coefficients/families")
    if metadata is not None and metadata["reference_kind"] == "torch_chunk4000":
        source = metadata.get("source", {})
        if (
            metadata.get("chunk_rows") != 4000
            or source.get("commit") != receipt.get("training_reference_commit")
            or any(
                source.get("verified_files", {}).get(name) != digest
                for name, digest in TIED_FEATURE_SOURCES.items()
            )
        ):
            raise ValueError(
                "Torch reference is not bound to the admitted chunk4000 feature sources"
            )
    if (
        metadata is not None
        and args.comparison_mode == "equivalent"
        and metadata["reference_kind"] != "torch_chunk4000"
    ):
        raise ValueError(
            "equivalent mode requires the admitted Torch chunk4000 reference; "
            "use callback-equivalent for native ISA equivalence only"
        )
    if args.comparison_mode == "quality" and (
        receipt["mode"] != "dynamic_int8"
        or metadata is None
        or metadata["reference_kind"] != "callback_f32"
    ):
        raise ValueError(
            "quality mode requires a dynamic-int8 candidate and its f32 callback reference"
        )
    if (
        metadata is not None
        and args.comparison_mode == "callback-equivalent"
        and metadata["reference_kind"] != f"callback_{receipt['mode']}"
    ):
        raise ValueError("callback-equivalent requires the same arithmetic mode for both ISA paths")
    parquet = pq.ParquetFile(args.data)
    if max(groups) >= parquet.num_row_groups:
        raise ValueError("split references missing validation groups")
    output = args.output_dir / "predictions.npy"
    predictions = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32, shape=(len(groups), SEQUENCE_LENGTH, 2)
    )
    actual_score, reference_score = GlobalAccumulator(), GlobalAccumulator()
    started = time.perf_counter()
    model, native = load_model(package, args.level)
    startup_seconds = time.perf_counter() - started
    seconds, callback_seconds, ids, comparison = [], [], [], []
    first = second = first_prediction = None
    for index, group in enumerate(groups):
        start = time.perf_counter()
        stream = load_sequence(parquet, group)
        if stream["seq_ix"] in ids:
            raise ValueError("duplicate validation sequence ID")
        ids.append(stream["seq_ix"])
        actual, elapsed = replay(model, stream)
        actual_score.add(stream["targets"], actual, stream["score"])
        predictions[index] = actual
        if reference is not None:
            if metadata.get("sequence_ids", [])[index : index + 1] != [stream["seq_ix"]]:
                raise ValueError("reference sequence IDs differ from parquet")
            comparison.append(differences(actual, reference[index], stream["need"]))
            reference_score.add(stream["targets"], reference[index], stream["score"])
        if index == 0:
            first, first_prediction = stream, actual.copy()
        elif index == 1:
            second = stream
        callback_seconds.append(elapsed)
        seconds.append(time.perf_counter() - start)
    predictions.flush()
    del predictions
    loop_wall_seconds = time.perf_counter() - started
    report["state_checks"] = state_checks(model, first, first_prediction, second)
    score = actual_score.result()
    if args.full_valid and score["selected_rows"] != 3_528_273:
        raise ValueError("full official valid must contain exactly 3528273 scored rows")
    report.update(
        native=native,
        sequence_ids=ids,
        metrics=score,
        score_scope="full_official_valid_exposed"
        if args.full_valid
        else "profile_subset_not_primary",
        predictions={"path": output.name, "sha256": sha256(output)},
    )
    report["timing"] = {
        "startup_seconds": startup_seconds,
        "validation_loop_wall_seconds": loop_wall_seconds,
        "callback_seconds": sum(callback_seconds),
        "callback_rows": len(groups) * SEQUENCE_LENGTH,
        "callback_us_per_row": sum(callback_seconds) * 1e6 / (len(groups) * SEQUENCE_LENGTH),
        "per_sequence_end_to_end_seconds": seconds,
        "includes": "loop includes decode, callbacks, scoring, reference comparison and prediction writes",
        "excludes": "input hashes, ZIP extraction, post-loop A-B-A checks and final output hash",
    }
    report["runtime_projection"] = {"status": "unknown_test_cardinality"}
    if args.test_sequences:
        estimate = startup_seconds + max(seconds) * args.test_sequences
        report["runtime_projection"] = {
            "status": "projection_only_not_measured_test_runtime",
            "test_sequences": args.test_sequences,
            "seconds_using_slowest_observed_sequence": estimate,
            "under_60_minutes_projected": estimate < 3600,
            "basis": "same host and CPU level; slowest observed decode+callback+validation sequence",
        }
    if reference is None:
        report["comparison"] = {"status": "pending_reference", "chunk4000_parity": "unverified"}
        return True
    maximum = max(row["max_abs"] for row in comparison)
    boundary_max = max(row["boundary4000_max_abs"] for row in comparison)
    other = reference_score.result()
    delta = score["wp"] - other["wp"]
    equivalent = maximum <= args.equivalence_atol
    quality = args.full_valid and delta >= -args.max_wp_drop
    report["comparison"] = {
        "status": "compared",
        "mode": args.comparison_mode,
        "reference_kind": metadata["reference_kind"],
        "reference_metrics": other,
        "reference_prediction_sha256": metadata["predictions"]["sha256"],
        "max_abs": maximum,
        "boundary4000_max_abs": boundary_max,
        "boundary4000_rows": comparison[0]["boundary4000_rows"],
        "equivalence_atol": args.equivalence_atol,
        "numerically_equivalent": equivalent,
        "wp_delta": delta,
        "max_wp_drop": args.max_wp_drop,
        "full_valid_quality_passed": quality,
        "chunk4000_parity": ("passed" if equivalent else "FAILED")
        if metadata["reference_kind"] == "torch_chunk4000"
        else "unverified_no_torch_reference",
        "interpretation": "a non-equivalent native stream is a separately scored candidate, not the Torch predictor",
    }
    return bool(quality if args.comparison_mode == "quality" else equivalent)


def main() -> None:
    """Write an immutable receiver receipt; never declare submission approval."""
    args = arguments()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "tl3c-callback-validation-v1",
        "submission_approved": False,
        "scope": "receiver execution only; no training or portal submission",
    }
    passed = False
    try:
        report["receiver"] = constrain_receiver()
        report["scorer_source_sha256"] = sha256(ROOT / "utils.py")
        with tempfile.TemporaryDirectory(prefix="tl3c-receiver-") as temporary:
            passed = evaluate(args, report, Path(temporary).resolve())
        report["validation_completed"] = True
        report["requested_comparison_passed"] = (
            None if report["comparison"]["status"] == "pending_reference" else passed
        )
    except Exception as error:
        report.update(validation_completed=False, error=f"{type(error).__name__}: {error}")
    finally:
        report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
