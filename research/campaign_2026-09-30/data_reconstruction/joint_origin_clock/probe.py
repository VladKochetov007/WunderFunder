"""Compare age-like input coordinates after joint lower-endpoint observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PREVIOUS = HERE.parent / "event_rate_features/selection.json"
MANIFEST = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
LOWER = np.float32(-5.199337482452393)
UPPER = np.float32(5.199337482452393)
PAIRS = [(5, 6)] + [(index, clock) for index in (0, 1, 4) for clock in (5, 6)]
COLUMNS = [f"i0_p{index}" for index in range(22)]
COLUMNS += [f"i0_{group}{index}" for group in ("dp", "dv") for index in range(4)]
COLUMNS += [f"a{index}" for index in range(8)]


def sha(path: Path) -> str:
    """Hash frozen sources and membership metadata."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path: Path, value: object) -> None:
    """Write once to retain the original experiment evidence."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def freeze() -> None:
    """Choose sixteen fresh components without reading any data values."""
    previous = json.loads(PREVIOUS.read_text())
    excluded = set(previous["excluded_components"])
    excluded.update(row["component"] for key in ("fit", "evaluation") for row in previous[key])
    rows = json.loads(MANIFEST.read_text())["sequences"]["train"]
    rows.sort(
        key=lambda row: hashlib.sha256(
            f"joint-origin-clock-v1:{row['row_group']}".encode()
        ).digest()
    )
    selected = []
    used = set(excluded)
    for row in rows:
        if row["component"] not in used:
            selected.append({key: row[key] for key in ("row_group", "component")})
            used.add(row["component"])
        if len(selected) == 16:
            break
    if len(selected) != 16:
        raise ValueError("Insufficient fresh components")
    publish(
        HERE / "selection.json",
        {
            "selected": selected,
            "excluded_components": sorted(excluded),
            "previous_selection_sha256": sha(PREVIOUS),
            "manifest_sha256": sha(MANIFEST),
            "source_sha256": sha(Path(__file__)),
            "preregistration_sha256": sha(HERE / "preregistration.json"),
            "features_targets_masks_read": False,
        },
    )


def rho(values: np.ndarray, pair: tuple[int, int]) -> float:
    """Calculate average-tie Spearman; constants and <2 rows return zero."""
    if len(values) < 2:
        return 0.0
    ranked = rankdata(values[:, pair], axis=0, method="average")
    ranked -= ranked.mean(0)
    norms = np.sqrt(np.square(ranked).sum(0))
    if np.any(norms == 0):
        return 0.0
    return float(np.dot(ranked[:, 0] / norms[0], ranked[:, 1] / norms[1]))


def describe(a: np.ndarray, eligible: np.ndarray) -> dict[str, object]:
    """Record all fixed correlations and distinct-value support."""
    values = a[eligible].astype(np.float64)
    return {
        "rows": len(values),
        "distinct_values": [int(np.unique(values[:, index]).size) for index in range(8)],
        "pair_spearman": [rho(values, pair) for pair in PAIRS],
    }


def probe() -> None:
    """Read input-only TRAIN blocks and evaluate the frozen conditions."""
    selection = json.loads((HERE / "selection.json").read_text())
    if selection["source_sha256"] != sha(Path(__file__)):
        raise ValueError("Source changed after freezing")
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    rows = []
    for chosen in selection["selected"]:
        table = source.read_row_group(chosen["row_group"], columns=COLUMNS, use_threads=False)
        x = np.column_stack([table.column(name).to_numpy() for name in COLUMNS])
        if x.shape != (20000, 38) or not np.isfinite(x).all():
            raise ValueError("Invalid feature block")
        a = x[:, -8:]
        scored_steps = np.arange(len(x)) >= 99
        both_lower = (a[:, 5] == LOWER) & (a[:, 6] == LOWER)
        previous_lower = np.r_[False, both_lower[:-1]]
        current_interior = (
            (a[:, 5] > LOWER) & (a[:, 6] > LOWER) & (a[:, 5] < UPPER) & (a[:, 6] < UPPER)
        )
        primary = scored_steps & previous_lower & current_interior
        price_change = np.r_[False, np.any(x[1:, :22] != x[:-1, :22], axis=1)]
        trade_change = np.r_[False, np.any(x[1:, 22:30] != x[:-1, 22:30], axis=1)]
        rows.append(
            {
                **chosen,
                "all_evaluation_steps": describe(a, scored_steps),
                "primary_joint_origin": describe(a, primary),
                "endpoint_support": {
                    "previous_both_lower": int((scored_steps & previous_lower).sum()),
                    "previous_lower_current_lower_or_saturated": int(
                        (scored_steps & previous_lower & ~current_interior).sum()
                    ),
                    "both_lower_current": int((scored_steps & both_lower).sum()),
                    "both_lower_with_price_and_trade_refresh": int(
                        (scored_steps & both_lower & price_change & trade_change).sum()
                    ),
                    "both_lower_without_price_or_trade_refresh": int(
                        (scored_steps & both_lower & ~price_change & ~trade_change).sum()
                    ),
                },
            }
        )
    summaries = []
    for j, pair in enumerate(PAIRS):
        supported = [row for row in rows if row["primary_joint_origin"]["rows"] >= 100]
        positive = sum(row["primary_joint_origin"]["pair_spearman"][j] >= 0.99 for row in supported)
        negative = sum(
            row["primary_joint_origin"]["pair_spearman"][j] <= -0.99 for row in supported
        )
        summaries.append(
            {
                "pair": [f"a{index}" for index in pair],
                "components_with_at_least100_primary_rows": len(supported),
                "positive_rho_ge0.99_components": positive,
                "negative_rho_le_minus0.99_components": negative,
                "strong_common_order_support": max(positive, negative) >= 8,
            }
        )
    publish(
        HERE / "results.json",
        {
            "selection_sha256": sha(HERE / "selection.json"),
            "source_sha256": sha(Path(__file__)),
            "fixed_pairs": PAIRS,
            "per_component": rows,
            "summaries": summaries,
            "targets_masks_VALID_external_data_read": False,
            "claim_limit": "Endpoint-conditioned monotonic association only. CDF saturation may merge different small ages. Physical zero resets, common time units, source identity, target law and public gain remain unproven.",
        },
    )


def main() -> None:
    """Execute one bounded phase and persist measured process resources."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "probe"])
    args = parser.parse_args()
    start = time.monotonic()
    {"freeze": freeze, "probe": probe}[args.phase]()
    publish(
        HERE / f"{args.phase}_resources.json",
        {
            "wall_seconds": time.monotonic() - start,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "address_space_limit_bytes": resource.getrlimit(resource.RLIMIT_AS)[0],
            "cpu_time_limit_seconds": resource.getrlimit(resource.RLIMIT_CPU)[0],
            "thread_pool_workers": 1,
            "gpu_seconds": 0,
            "cgroup_or_durable_execution_claim": False,
        },
    )


if __name__ == "__main__":
    main()
