"""Score frozen heterogeneous mixtures against canonical TL3 on full official valid."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.heterogeneous_ensemble_contract import (
    DEFAULT_ENSEMBLES,
    TL3_WEIGHTS,
    JsonObject,
    PredictionSource,
    load_bank,
    load_contract,
    load_shards,
    validate_role,
)  # noqa: E402
from scripts.heterogeneous_galam_probe import load_probe  # noqa: E402
from wnn.gmetric import GlobalAccumulator  # noqa: E402
from wnn.splits import file_sha256  # noqa: E402


def ensemble_requirements(
    ensembles: dict[str, dict[str, float]], members: set[str]
) -> dict[str, set[str]]:
    """Validate positive convex mixtures and resolve their acyclic member dependencies."""
    if ensembles.get("TL3") != TL3_WEIGHTS or members & ensembles.keys():
        raise ValueError("canonical TL3 weights or unique member/ensemble names differ")
    requirements: dict[str, set[str]] = {}

    def visit(name: str, active: set[str]) -> set[str]:
        if name in members:
            return {name}
        if name in active or name not in ensembles:
            raise ValueError(f"cyclic or unknown mixture member: {name}")
        if name in requirements:
            return requirements[name]
        weights = ensembles[name]
        if (
            not isinstance(weights, dict)
            or not weights
            or any(
                type(weight) not in (float, int) or not math.isfinite(weight) or weight <= 0
                for weight in weights.values()
            )
            or not math.isclose(sum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError(f"ensemble requires positive finite weights summing to one: {name}")
        result = set().union(*(visit(child, active | {name}) for child in weights))
        requirements[name] = result
        return result

    for name in ensembles:
        visit(name, set())
    return requirements


def blend_predictions(
    name: str, values: dict[str, np.ndarray], ensembles: dict[str, dict[str, float]]
) -> np.ndarray:
    """Compose raw FP32 outputs in declared order; official clipping happens afterwards."""
    if name in values:
        return values[name]
    result = None
    for member, weight in ensembles[name].items():
        prediction = blend_predictions(member, values, ensembles)
        term = np.float32(weight) * np.asarray(prediction, dtype=np.float32)
        if result is None:
            result = term.copy()
        else:
            if result.shape != term.shape:
                raise ValueError("ensemble members have different prediction shapes")
            result += term
    if result is None:
        raise ValueError("empty mixture")
    values[name] = result
    return result


def score_groups(
    parquet: pq.ParquetFile,
    groups: list[int],
    sources: dict[str, PredictionSource],
    ensembles: dict[str, dict[str, float]],
    available: list[str],
) -> tuple[dict[str, JsonObject], dict[str, JsonObject], int]:
    """Pool complete source sequences and apply the original mask separately per target."""
    ensemble_scores = {name: GlobalAccumulator() for name in available}
    member_scores = {name: GlobalAccumulator() for name in sources}
    selected_rows = 0
    for group in groups:
        table = parquet.read_row_group(group, columns=["t0", "t1", "is_scored"], use_threads=False)
        if table.num_rows != 20_000:
            raise ValueError("evaluation row group is not one complete20000-row sequence")
        targets = np.column_stack([table[name].to_numpy() for name in ("t0", "t1")]).astype(
            np.float32
        )
        mask = table["is_scored"].to_numpy().astype(bool, copy=True)
        mask[:99] = False
        selected_rows += int(mask.sum())
        values = {name: source.sequence(group) for name, source in sources.items()}
        for name, accumulator in member_scores.items():
            accumulator.add(targets, values[name], mask)
        for name, accumulator in ensemble_scores.items():
            accumulator.add(targets, blend_predictions(name, values, ensembles), mask)
    return (
        {name: score.result() for name, score in ensemble_scores.items()},
        {name: score.result() for name, score in member_scores.items()},
        selected_rows,
    )


def markdown_table(rows: list[JsonObject]) -> str:
    """Render unavailable comparisons explicitly without inventing zero scores or deltas."""
    lines = ["| Ensemble | t0 | t1 | WP | Delta vs TL3 | Status |", "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        score = row["score"]
        numbers = (
            ["—"] * 4
            if score is None
            else [f"{score[key]:.9f}" for key in ("t0", "t1", "wp")]
            + ["—" if row["delta_to_tl3"] is None else f"{row['delta_to_tl3']:+.9f}"]
        )
        status = row["status"] + (
            ": " + ", ".join(row["missing_members"]) if row["missing_members"] else ""
        )
        lines.append("| " + " | ".join([row["name"], *numbers, status]) + " |")
    return "\n".join(lines) + "\n"


def grade(manifest_path: Path, output: Path) -> JsonObject:
    """Validate immutable producer assets, score available endpoints and write a fresh report."""
    if output.exists():
        raise FileExistsError("preserve prior results; choose a fresh output directory")
    manifest_hash = file_sha256(manifest_path)
    manifest, split, validation, order = load_contract(manifest_path)
    ensembles = manifest.get("ensembles", DEFAULT_ENSEMBLES)
    if not isinstance(ensembles, dict):
        raise ValueError("ensembles must be an object")
    requirements = ensemble_requirements(ensembles, set(manifest["members"]))
    sources, pending = {}, {}
    for name, record in manifest["members"].items():
        if not isinstance(record, dict):
            raise ValueError(f"member record is not an object: {name}")
        validate_role(name, record)
        if record.get("state") == "pending":
            if not isinstance(record.get("reason"), str) or not record["reason"].strip():
                raise ValueError("a pending member requires an explicit reason")
            pending[name] = record["reason"]
            continue
        if (
            type(record.get("epoch")) is not int
            or record["epoch"] < 0
            or not isinstance(record.get("run"), str)
        ):
            raise ValueError("available member requires run and zero-based epoch")
        if record.get("kind") == "legacy_bank":
            sources[name] = load_bank(record, manifest_path.resolve().parent, split, order)
        elif record.get("kind") == "sequence_shards":
            sources[name] = load_shards(record, manifest_path.resolve().parent, split)
        elif record.get("kind") == "galam_scored_probe":
            sources[name] = load_probe(record, manifest_path.resolve().parent, split, validation)
        else:
            raise ValueError(f"unsupported explicit producer kind: {record.get('kind')}")
    available = [name for name in ensembles if requirements[name] <= sources.keys()]
    if not sources:
        raise ValueError("no predictions exist; this command grades actual artifacts")
    parquet = pq.ParquetFile(validation)
    if parquet.num_row_groups != 1873:
        raise ValueError("official valid must contain1873 whole sequences")
    scores, members, selected_rows = score_groups(
        parquet, list(split.evaluation_groups), sources, ensembles, available
    )
    if selected_rows != 3_528_273:
        raise ValueError("full official-valid scored-row count differs from the admitted mask")
    baseline = scores.get("TL3")
    rows = [
        {
            "name": name,
            "weights": weights,
            "score": scores.get(name),
            "delta_to_tl3": None
            if baseline is None or name not in scores
            else scores[name]["wp"] - baseline["wp"],
            "status": "complete" if name in scores else "unavailable",
            "missing_members": sorted(requirements[name] - sources.keys()),
        }
        for name, weights in ensembles.items()
    ]
    if file_sha256(manifest_path) != manifest_hash:
        raise ValueError("input manifest changed during evaluation")
    report = {
        "schema": "wnn.heterogeneous-ensemble-score.v1",
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "status": "complete" if len(available) == len(ensembles) else "partial",
        "baseline": "TL3",
        "baseline_available": baseline is not None,
        "input_manifest_sha256": manifest_hash,
        "input_manifest": manifest,
        "split_sha256": split.sha256,
        "valid_sequences": len(split.evaluation_groups),
        "selected_rows": selected_rows,
        "rows": rows,
        "member_scores": members,
        "pending_members": pending,
        "paired_student_deltas": {
            str(seed): None
            if not {f"control_s{seed}", f"hetero_s{seed}"} <= members.keys()
            else members[f"hetero_s{seed}"]["wp"] - members[f"control_s{seed}"]["wp"]
            for seed in (1301, 1302)
        },
        "exposure": "full official valid is development-exposed; fixed weights, no public-score forecast",
        "uncertainty": "fixed-model scores; no training-variance, significance or new holdout claim",
        "ancestry_limit": "producer membership is checked; legacy consumed teacher prediction bytes require separate ancestry receipts",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "COMPARISON.md").write_text(markdown_table(rows), encoding="utf-8")
    return report


def main() -> None:
    """Grade pinned existing predictions; never fit models or tune mixture coefficients."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = grade(args.manifest, args.output)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "baseline_available", "valid_sequences", "selected_rows")
            }
        )
    )


if __name__ == "__main__":
    main()
