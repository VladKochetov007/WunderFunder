#!/usr/bin/env python3
"""Plot the measured subset replay wall times with mmap setup amortized."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

SCRIPT = Path(__file__).resolve()
LINEAGE = SCRIPT.parent
INPUT = LINEAGE / "E528_RESULTS_V2.json"
PLOTS = LINEAGE / "plots"


def load_series(document: dict[str, Any]) -> tuple[list[int], list[float], list[float], float, int | None]:
    design = document["timing_design"]
    epochs = [int(value) for value in design["epoch_counts_K"]]
    setup_seconds = float(document["break_even"]["cache_setup_seconds_including_required_verification"])
    parquet = [
        float(design["method_by_K"][str(k)]["parquet"]["median_total_K_replay_wall_seconds"])
        for k in epochs
    ]
    mapped = [
        setup_seconds + float(design["method_by_K"][str(k)]["mmap"]["median_total_K_replay_wall_seconds"])
        for k in epochs
    ]
    first_break_even = document["break_even"]["strict_first_K_or_null"]
    return epochs, parquet, mapped, setup_seconds, first_break_even


def plot() -> tuple[Path, Path]:
    document = json.loads(INPUT.read_text(encoding="utf-8"))
    epochs, parquet, mapped, setup_seconds, first_break_even = load_series(document)
    PLOTS.mkdir(exist_ok=True)
    positions = np.arange(len(epochs), dtype=np.float64)
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    figure.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.29)
    axis.bar(positions - width / 2, parquet, width, label="Parquet decode each pass", color="#386cb0")
    axis.bar(positions + width / 2, mapped, width, label="mmap replay + one-time setup", color="#fdb462")
    axis.set_xticks(positions, [str(k) for k in epochs])
    axis.set_xlabel("Replays of the same 20-sequence subset (K)")
    axis.set_ylabel("Median measured replay wall (seconds)")
    axis.set_title("Decoded-feature reuse amortizes on the 20-sequence subset")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(frameon=False)
    for x, y in zip(positions - width / 2, parquet, strict=True):
        axis.annotate(f"{y:.2f}", (x, y), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    for x, y in zip(positions + width / 2, mapped, strict=True):
        axis.annotate(f"{y:.2f}", (x, y), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    break_even_text = "null" if first_break_even is None else str(first_break_even)
    figure.text(
        0.12,
        0.025,
        f"Cache build + SHA + exact parity + mmap open = {setup_seconds:.3f}s; strict formula break-even K={break_even_text}.\n"
        "2 repeats/method/K; page-cache state unknown; no cold-cache claim. Subset only.",
        fontsize=8,
        va="bottom",
    )
    png_path = PLOTS / "E528_cache_break_even_A2.png"
    svg_path = PLOTS / "E528_cache_break_even_A2.svg"
    figure.savefig(png_path, dpi=160)
    figure.savefig(svg_path)
    plt.close(figure)
    return png_path, svg_path


if __name__ == "__main__":
    outputs = plot()
    for path in outputs:
        print(f"{path.relative_to(LINEAGE)}\t{path.stat().st_size}")
