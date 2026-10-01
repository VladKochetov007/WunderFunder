"""Compare book copulas with external physical books, without fitting a model."""

from __future__ import annotations

import csv
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
MD = Path("/home/vlad/development/ultimate/data/MD/lob.csv")
SELECTION = HERE.parent / "additional_feature_relations/selection.json"


def sha(path: Path) -> str:
    """Hash a small source or analysis commitment."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path: Path, value: object) -> None:
    """Preserve each result as an exclusive artifact."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def describe(x: np.ndarray) -> dict[str, object]:
    """Summarize rank dependence invariant to individual monotone transforms."""
    finite = np.isfinite(x).all(1)
    x = x[finite]
    if len(x) < 100:
        return {"rows": len(x), "status": "insufficient finite rows"}
    ranks = rankdata(x, method="average", axis=0)
    ranks -= ranks.mean(0)
    norm = np.sqrt(np.square(ranks).sum(0))
    ranks /= np.maximum(norm, 1e-30)
    ranks[:, norm == 0] = 0
    correlation = ranks.T @ ranks
    eigenvalues = np.linalg.eigvalsh(correlation)[::-1]
    trace = float(eigenvalues.sum())
    within = np.r_[
        correlation[:11, :11][np.triu_indices(11, 1)],
        correlation[11:, 11:][np.triu_indices(11, 1)],
    ]
    cross = correlation[:11, 11:].ravel()
    return {
        "status": "complete",
        "rows": len(x),
        "nonfinite_rows": int((~finite).sum()),
        "active_columns": int((norm > 0).sum()),
        "price_or_quantity_spectrum": (eigenvalues / max(trace, 1e-30)).tolist(),
        "top_eigenvalue_share": float(eigenvalues[0] / max(trace, 1e-30)),
        "top_two_share": float(eigenvalues[:2].sum() / max(trace, 1e-30)),
        "effective_rank": float(trace * trace / max(np.square(eigenvalues).sum(), 1e-30)),
        "within_side_rho_quantiles": np.quantile(within, [0.1, 0.5, 0.9]).tolist(),
        "cross_side_rho_quantiles": np.quantile(cross, [0.1, 0.5, 0.9]).tolist(),
        "distinct_values_per_column": [len(np.unique(column)) for column in x.T],
    }


def external() -> dict[str, object]:
    """Read the fixed complete L11 snapshot prefix, without reconstructing depth."""
    digest = hashlib.sha256()
    prices, volumes = [], []
    names = [
        f"{side}s[{index}].{kind}"
        for side in ("bid", "ask")
        for index in range(11)
        for kind in ("price", "amount")
    ]
    with MD.open() as stream:
        header = stream.readline()
        digest.update(header.encode())
        columns = next(csv.reader([header]))
        indices = [columns.index(name) for name in names]
        for _ in range(20000):
            line = stream.readline()
            if not line:
                break
            digest.update(line.encode())
            fields = next(csv.reader([line]))
            values = [float(fields[index]) if fields[index] else np.nan for index in indices]
            prices.append(values[::2])
            volumes.append(values[1::2])
    p, v = np.asarray(prices), np.asarray(volumes)
    valid = np.isfinite(p).all(1) & np.isfinite(v).all(1) & (p > 0).all(1) & (v >= 0).all(1)
    p, v = p[valid], v[valid]
    midpoint = 0.5 * (p[:, 0] + p[:, 11])
    variants = {
        "absolute_prices": p,
        "midpoint_centered_prices": p - midpoint[:, None],
        "midpoint_fractional_prices": p / midpoint[:, None] - 1,
        "side_top_centered_prices": p - np.repeat(p[:, [0, 11]], 11, axis=1),
    }
    return {
        "source": {
            "path": str(MD),
            "size_bytes": MD.stat().st_size,
            "mtime_ns": MD.stat().st_mtime_ns,
            "consumed_prefix_utf8_sha256": digest.hexdigest(),
        },
        "snapshots_consumed": len(prices),
        "eligible_snapshots": len(p),
        "ineligible_snapshots": int((~valid).sum()),
        "price_variants": {name: describe(value) for name, value in variants.items()},
        "volume": describe(v),
        "limits": "One external source/prefix; fixed physical MD depth ordering applies only to MD. No external fitted state or rows enter competition training/inference.",
    }


def main() -> None:
    """Compute only declared descriptive statistics under bounded CPU resources."""
    started = time.monotonic()
    contract = json.loads((HERE / "preregistration.json").read_text())
    if (
        sha(Path(__file__)) != contract["source_sha256"]
        or sha(SELECTION) != contract["selection_sha256"]
    ):
        raise ValueError("Source or membership changed after preregistration")
    selected = json.loads(SELECTION.read_text())
    groups = selected["evaluation"]
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    columns = [
        f"i{instrument}_{group}{index}"
        for instrument in (0, 1)
        for group in ("p", "v")
        for index in range(22)
    ]
    summaries = []
    for item in groups:
        table = source.read_row_group(item["row_group"], columns=columns, use_threads=False)
        for instrument in (0, 1):
            for group in ("p", "v"):
                x = np.column_stack(
                    [
                        table.column(f"i{instrument}_{group}{index}").to_numpy()
                        for index in range(22)
                    ]
                )
                summaries.append(
                    {
                        "row_group": item["row_group"],
                        "component": item["component"],
                        "instrument": instrument,
                        "feature_group": group,
                        **describe(x),
                    }
                )
    outside = external()
    result = {
        "preregistration_sha256": sha(HERE / "preregistration.json"),
        "competition": summaries,
        "external": outside,
        "wall_seconds": time.monotonic() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "address_space_limit": resource.getrlimit(resource.RLIMIT_AS),
        "target_mask_VALID_columns_read": False,
        "no_model_or_external_transform_fitted": True,
        "claim_scope": "Copula shape and dimensionality; similarity does not identify physical coordinates/source. Overall spectrum is permutation invariant; side summaries use published side membership, never assumed competition depth order.",
    }
    publish(HERE / "results.json", result)
    for group in ("p", "v"):
        selected_stats = [row for row in summaries if row["feature_group"] == group]
        print(
            json.dumps(
                {
                    "provided_group": group,
                    "median_top_share": float(
                        np.median([row["top_eigenvalue_share"] for row in selected_stats])
                    ),
                    "median_cross_side_rho": float(
                        np.median([row["cross_side_rho_quantiles"][1] for row in selected_stats])
                    ),
                }
            )
        )
    print(
        json.dumps(
            {
                "external_price_top_share": {
                    name: value["top_eigenvalue_share"]
                    for name, value in outside["price_variants"].items()
                },
                "wall_seconds": result["wall_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
