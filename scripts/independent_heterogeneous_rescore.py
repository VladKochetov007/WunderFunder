"""Independently rescore all final models and twelve frozen ensembles on CPU."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

from scripts.independent_heterogeneous_inputs import load_sources, sha256
from scripts.independent_heterogeneous_numeric import MIXTURES, TwoPassPearson, compose


def peak_rss_bytes() -> int | None:
    """Return Linux peak resident memory without importing GPU/model libraries."""
    if sys.platform != "linux":
        return None
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def write_json(path: Path, value: object) -> None:
    """Preserve one fresh evidence file without overwriting an earlier result."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def compare_scores(
    scores: dict[str, dict[str, float | int]], report: dict, tolerance: float
) -> tuple[list[dict], float]:
    """Compare every independently computed target/pooled score to the actual report."""
    expected = dict(report["member_scores"])
    expected.update({row["name"]: row["score"] for row in report["rows"]})
    if set(scores) != set(expected) or len(scores) != 25:
        raise ValueError("all thirteen singletons and twelve ensembles must be compared")
    records, maximum = [], 0.0
    for name, actual in scores.items():
        row = expected[name]
        if row.get("selected_rows") != 3528273 or row.get("metric_id") != "global_wp_clip2_v1":
            raise ValueError(f"reported metric/coverage differs: {name}")
        differences = {}
        for key in ("t0", "t1", "wp"):
            if type(row.get(key)) not in (int, float) or not math.isfinite(row[key]):
                raise ValueError(f"reported score is nonfinite or absent: {name}/{key}")
            differences[key] = abs(actual[key] - row[key])
            maximum = max(maximum, differences[key])
        records.append(
            {
                "name": name,
                "independent": actual,
                "production": row,
                "absolute_differences": differences,
                "passed": max(differences.values()) <= tolerance,
            }
        )
    return records, maximum


def run(args: argparse.Namespace) -> bool:
    """Stream actual frozen arrays twice and retain an independent agreement receipt."""
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "request.json",
        {
            "manifest": str(args.manifest.resolve()),
            "report": str(args.report.resolve()),
            "max_wall_seconds": args.max_wall_seconds,
            "tolerance": args.tolerance,
            "cpu_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "gpu_usage": "none; NumPy/PyArrow only; no Torch or production scorer imports",
        },
    )

    def check_limits() -> None:
        """Stop only this read-only verifier if its declared work bounds are exceeded."""
        if time.monotonic() - started > args.max_wall_seconds:
            raise TimeoutError("independent rescore wall-clock limit exceeded")
        peak = peak_rss_bytes()
        if peak is not None and peak > 2 * 1024**3:
            raise RuntimeError("independent verifier exceeded its 2 GiB resident-memory budget")

    try:
        sources, production = load_sources(args.manifest, args.report)
        names = [*sources.banks, *MIXTURES]
        accumulator = TwoPassPearson(names)
        for pass_index in (0, 1):
            for index, group in enumerate(sources.groups):
                check_limits()
                targets, raw = sources.selected(group)
                accumulator.add(targets, compose(raw), pass_index)
                sources.release_mapped_pages()
                if (index + 1) % 100 == 0 or index + 1 == len(sources.groups):
                    print(
                        json.dumps(
                            {
                                "pass": pass_index + 1,
                                "completed_sequences": index + 1,
                                "total_sequences": len(sources.groups),
                                "selected_rows": accumulator.rows[pass_index],
                                "elapsed_seconds": time.monotonic() - started,
                                "peak_rss_bytes": peak_rss_bytes(),
                            }
                        ),
                        flush=True,
                    )
        if accumulator.rows != [3528273, 3528273]:
            raise ValueError("independent passes did not cover the complete official scoring mask")
        scores = accumulator.result()
        comparisons, maximum = compare_scores(scores, production, args.tolerance)
        for path, expected in sources.evidence.items():
            check_limits()
            if sha256(Path(path)) != expected:
                raise ValueError(f"input evidence changed during independent rescoring: {path}")
        code = {
            name: sha256(Path(__file__).with_name(name))
            for name in (
                "independent_heterogeneous_rescore.py",
                "independent_heterogeneous_inputs.py",
                "independent_heterogeneous_numeric.py",
            )
        }
        passed = maximum <= args.tolerance
        result = {
            "schema": "wnn.independent-heterogeneous-rescore.v1",
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "status": "passed" if passed else "score_disagreement",
            "method": "Independent float64 weighted means, then a second centered variance/covariance pass; target and prediction clipped after raw FP32 hierarchical mixture construction.",
            "production_scorer_imported": False,
            "model_inference": False,
            "new_fits": 0,
            "valid_sequences": len(sources.groups),
            "selected_rows_per_pass": accumulator.rows,
            "mixtures": {name: dict(terms) for name, terms in MIXTURES.items()},
            "comparisons": comparisons,
            "max_absolute_difference": maximum,
            "tolerance": args.tolerance,
            "input_evidence_sha256": sources.evidence,
            "verifier_source_sha256": code,
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": peak_rss_bytes(),
            "exposure": "The same exposed official-valid data and fixed fitted models; numerical verification is not fresh confirmation, training replication, serving compliance or public/private transfer.",
            "ancestry_scope": "Actual artifact pins and direct source memberships are checked. The scientific producer/teacher-cache ancestry audit remains a separate evidence layer.",
        }
        write_json(args.output / "RESULT.json", result)
        lines = [
            "| Output | Independent WP | Production WP | Max target/WP difference | Status |",
            "|---|---:|---:|---:|---|",
        ]
        for row in comparisons:
            lines.append(
                f"| {row['name']} | {row['independent']['wp']:.12f} | {row['production']['wp']:.12f} | {max(row['absolute_differences'].values()):.3g} | {'pass' if row['passed'] else 'FAIL'} |"
            )
        with (args.output / "COMPARISON.md").open("x", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "outputs": len(comparisons),
                    "max_absolute_difference": maximum,
                }
            ),
            flush=True,
        )
        return passed
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        write_json(
            args.output / "FAILURE.json",
            {
                "kind": type(error).__name__,
                "message": str(error),
                "elapsed_seconds": time.monotonic() - started,
                "new_fits": 0,
                "production_artifacts_changed": False,
            },
        )
        raise


def main() -> None:
    """Run a bounded CPU-only check after every final source and table row exists."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=int, default=1800)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    if not 0 < args.max_wall_seconds <= 1800 or args.tolerance != 1e-10:
        parser.error("require at most 1800 seconds and the frozen tolerance 1e-10")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "ARROW_NUM_THREADS"):
        os.environ[key] = "1"
    raise SystemExit(0 if run(args) else 1)


if __name__ == "__main__":
    main()
