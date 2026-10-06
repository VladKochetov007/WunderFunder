from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
COLUMNS = [
    "seq_ix",
    "step_in_seq",
    "need_prediction",
    "is_scored",
    "i0_p0",
    "i0_p11",
    "a6",
    "t0",
    "t1",
]
CLOCKS = ("bid", "ask", "or", "and")


def changed(values: np.ndarray) -> np.ndarray:
    """Mark raw-code changes without interpreting the anonymous coordinate."""
    return np.concatenate((np.zeros(1, dtype=bool), values[1:] != values[:-1]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Preserve the numerical data behind each descriptive figure."""
    with path.open("x", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_group(
    table: dict[str, np.ndarray], node: dict[str, object]
) -> list[dict[str, object]]:
    """Count each independently named event law on one fitting component."""
    if len(table["seq_ix"]) != 20_000 or not np.all(table["seq_ix"] == node["seq_ix"]):
        raise ValueError("cohort/group identity mismatch")
    if not np.array_equal(table["step_in_seq"], np.arange(20_000)):
        raise ValueError("expected ordered, consecutive sequence steps")
    for name in ("need_prediction", "is_scored"):
        if not np.isin(table[name], (0, 1)).all():
            raise ValueError(f"expected a binary row mask: {name}")
    for name in ("i0_p0", "i0_p11", "a6", "t0", "t1"):
        if not np.isfinite(table[name]).all():
            raise ValueError(f"nonfinite raw diagnostic input: {name}")
    bid, ask, auxiliary = changed(table["i0_p0"]), changed(table["i0_p11"]), changed(table["a6"])
    clocks = {"bid": bid, "ask": ask, "or": bid | ask, "and": bid & ask}
    eligible = table["need_prediction"].astype(bool) & (table["step_in_seq"] >= 99)
    selections = {"eligible": eligible, "scored": eligible & table["is_scored"].astype(bool)}
    rows: list[dict[str, object]] = []
    for population, selected in selections.items():
        for clock_name, event in clocks.items():
            for target_name in ("t0", "t1"):
                target = table[target_name]
                target_change = changed(target)
                rows.append(
                    {
                        "row_group": node["row_group"],
                        "seq_ix": node["seq_ix"],
                        "component": node["component"],
                        "role": node["role"],
                        "class_diagnostic": node["label_class_diagnostic_only"],
                        "population": population,
                        "clock": clock_name,
                        "target": target_name,
                        "rows": int(selected.sum()),
                        "events": int((selected & event).sum()),
                        "target_changes": int((selected & target_change).sum()),
                        "changes_without_event": int((selected & target_change & ~event).sum()),
                        "events_without_change": int((selected & event & ~target_change).sum()),
                        "a6_clock_disagreements": int((selected & (auxiliary != event)).sum()),
                        "absolute_target_mass": float(
                            np.abs(np.clip(target[selected], -2, 2)).sum(dtype=np.float64)
                        ),
                    }
                )
    return rows


def main() -> None:
    """Stream fit-only raw data and publish immutable descriptive artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a fresh EDA artifact directory")
    cohort_path = ROOT / "research/campaign_2026-10-05/target_mechanics/cohort.json"
    cohort = json.loads(cohort_path.read_text())
    nodes = [node for node in cohort["cohort"] if node["role"] == "fit"]
    if len(nodes) != 24 or len({node["component"] for node in nodes}) != 24:
        raise ValueError("expected the frozen 24 distinct fitting components")
    data = ROOT / "datasets/valid.parquet"
    parquet = pq.ParquetFile(data)
    rows: list[dict[str, object]] = []
    histogram = np.zeros((80, 80), dtype=np.int64)
    edges = np.linspace(-2, 2, 81)
    for node in nodes:
        arrow = parquet.read_row_group(int(node["row_group"]), columns=COLUMNS, use_threads=False)
        table = {name: arrow[name].to_numpy() for name in COLUMNS}
        rows.extend(summarize_group(table, node))
        eligible = table["need_prediction"].astype(bool) & (table["step_in_seq"] >= 99)
        counts, _, _ = np.histogram2d(
            np.clip(table["t0"][eligible], -2, 2),
            np.clip(table["t1"][eligible], -2, 2),
            bins=(edges, edges),
        )
        histogram += counts.astype(np.int64)
        del table, arrow
    args.output.mkdir(parents=True)
    write_csv(args.output / "per_component.csv", rows)
    sums: dict[tuple[object, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    measurements = [
        name
        for name in rows[0]
        if name
        not in {
            "row_group",
            "seq_ix",
            "component",
            "role",
            "class_diagnostic",
            "population",
            "clock",
            "target",
        }
    ]
    for row in rows:
        key = (row["class_diagnostic"], row["population"], row["clock"], row["target"])
        for name in measurements:
            sums[key][name] += float(row[name])
    aggregate = [
        {
            "class_diagnostic": key[0],
            "population": key[1],
            "clock": key[2],
            "target": key[3],
            **values,
        }
        for key, values in sorted(sums.items())
    ]
    write_csv(args.output / "aggregate.csv", aggregate)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, target in zip(axes, ("t0", "t1"), strict=True):
        missed = np.zeros((4, 4))
        for row in aggregate:
            if row["population"] == "eligible" and row["target"] == target:
                missed[int(row["class_diagnostic"]), CLOCKS.index(str(row["clock"]))] = row[
                    "changes_without_event"
                ] / max(row["target_changes"], 1)
        view = axis.imshow(missed, vmin=0, vmax=1, cmap="magma")
        axis.set(
            xticks=range(4),
            xticklabels=CLOCKS,
            yticks=range(4),
            ylabel="Diagnostic class",
            title=f"{target}: change without clock event",
        )
        for class_id in range(4):
            for clock_id in range(4):
                axis.text(
                    clock_id,
                    class_id,
                    f"{missed[class_id, clock_id]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if missed[class_id, clock_id] < 0.5 else "black",
                )
        figure.colorbar(view, ax=axis, label="Fraction of raw target changes")
    figure.savefig(args.output / "event_alignment.png", dpi=170)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    view = axis.imshow(np.log1p(histogram.T), origin="lower", extent=(-2, 2, -2, 2), cmap="viridis")
    axis.set(
        xlabel="Clipped t0", ylabel="Clipped t1", title="Fit-only joint support: log(1 + row count)"
    )
    figure.colorbar(view, ax=axis)
    figure.savefig(args.output / "joint_support.png", dpi=170)
    plt.close(figure)
    np.savez_compressed(args.output / "joint_histogram.npz", counts=histogram, edges=edges)
    report = {
        "study": "E492",
        "hypothesis": "H120",
        "fit_components": len(nodes),
        "evaluation_components_loaded": 0,
        "scientific_fits_completed": 0,
        "cohort_sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset_size_bytes": data.stat().st_size,
        "row_definition": "need_prediction and step>=99",
        "target_change_definition": "exact raw target inequality to preceding row",
        "class_use": "label-derived diagnostic grouping only",
        "aggregate": aggregate,
        "limitations": [
            "exposed fitting cohort",
            "association is not predictability",
            "no model or public-score claim",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"fit_components": len(nodes), "figures": 2, "scientific_fits_completed": 0}))


if __name__ == "__main__":
    main()
