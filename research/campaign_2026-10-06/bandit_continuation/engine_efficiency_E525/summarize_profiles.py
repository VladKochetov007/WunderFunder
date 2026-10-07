"""Summarize measured CPU samples and matched engine timings; no score forecast."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    folder = Path(__file__).resolve().parent
    plot_dir = folder / "plots"
    profiles = []
    for path in sorted(plot_dir.glob("*_native_profile.json")):
        source = json.loads(path.read_text())
        frames = source["shared"]["frames"]
        leaf: Counter[str] = Counter()
        for profile in source["profiles"]:
            for sample, weight in zip(profile["samples"], profile["weights"], strict=True):
                if sample:
                    leaf[frames[sample[-1]]["name"]] += weight
        total = leaf.total()
        profiles.append(
            {
                "case": path.stem.removesuffix("_native_profile"),
                "total_sample_weight_seconds": total,
                "top_leaf_frames": [
                    {"name": name, "fraction": weight / total}
                    for name, weight in leaf.most_common(12)
                ],
            }
        )
    (folder / "NATIVE_PROFILE_SUMMARY_V1.json").write_text(
        json.dumps(
            {
                "scope": "Sampled native leaf frames including startup; inlining limits attribution",
                "profiles": profiles,
            },
            indent=2,
        )
        + "\n"
    )
    first = json.loads((folder / "matched_engines_v1.json").read_text())
    second = json.loads((folder / "matched_quant_engines_v2.json").read_text())
    records = first["results"] + second["results"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True)
    for axis, members in zip(axes, (1, 2), strict=True):
        subset = [row for row in records if row["members"] == members and row["case"] != "cpp_int8"]
        subset.sort(key=lambda row: row["median_us_per_row"])
        medians = np.array([row["median_us_per_row"] for row in subset])
        lower = medians - np.array([min(row["trials_us_per_row"]) for row in subset])
        upper = np.array([max(row["trials_us_per_row"]) for row in subset]) - medians
        colors = ["#3178a8" if row["case"].startswith("rust") else "#cf8235" for row in subset]
        axis.barh(
            [row["case"] for row in subset],
            medians,
            color=colors,
            xerr=np.stack([lower, upper]),
            capsize=3,
        )
        axis.invert_yaxis()
        axis.set_title(f"{members} member(s), same own weights")
        axis.set_xlabel("Median µs/raw row (lower is faster)")
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "Matched CPU inference: three trials; bars show observed range\nTwo complete streams, one CPU; cpp_auto uses AVX-512 VNNI"
    )
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(plot_dir / f"matched_engine_latency.{extension}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
