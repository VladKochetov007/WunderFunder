"""Test parsimonious input-only explanations of the first five extra features."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.special import ndtr
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MANIFEST = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
FEATURES = [
    f"i{instrument}_{group}{index}"
    for instrument in (0, 1)
    for group, width in (("p", 22), ("v", 22), ("dp", 4), ("dv", 4))
    for index in range(width)
] + [f"a{index}" for index in range(8)]
EXCLUDE = [
    "research/campaign_2026-09-30/data_reconstruction/selection.json",
    "research/campaign_2026-09-29/provided_decode/sample_manifest.json",
    "research/campaign_2026-09-29/clock_reproduction/preregistration.json",
    "research/campaign_2026-09-30/event_content/exposure/selection.json",
    "research/campaign_2026-09-30/spread_aux/selection.json",
]


def sha(path: Path) -> str:
    """Identify small source, membership and result files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path: Path, value: object) -> None:
    """Write once so a second run cannot silently replace evidence."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def freeze() -> None:
    """Choose disjoint components without reading feature blocks or targets."""
    manifest = json.loads(MANIFEST.read_text())
    rows = manifest["sequences"]["train"]
    components = {row["row_group"]: row["component"] for row in rows}
    valid_components = {
        row["row_group"]: row["component"] for row in manifest["sequences"]["valid"]
    }
    excluded: set[int] = set()
    for name in EXCLUDE:
        selection = json.loads((ROOT / name).read_text())
        for key in ("fit_train_groups", "test_train_groups", "map_fit_train_groups"):
            # The spread cache's full fitting membership is not a small study.
            if "spread_aux" in name and key == "fit_train_groups":
                continue
            excluded.update(components[group] for group in selection.get(key, []))
        excluded.update(row["component"] for row in selection.get("selected", []))
        excluded.update(
            valid_components[group] for group in selection.get("evaluation_valid_groups", [])
        )
    chosen = []
    used = set(excluded)
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"additional-feature-relations-v1:{row['row_group']}".encode()
        ).digest(),
    )
    for row in ordered:
        if row["component"] not in used:
            chosen.append({key: row[key] for key in ("row_group", "component")})
            used.add(row["component"])
        if len(chosen) == 32:
            break
    if len(chosen) != 32:
        raise ValueError("Insufficient distinct fresh components")
    publish(
        HERE / "selection.json",
        {
            "fit": chosen[:16],
            "evaluation": chosen[16:],
            "excluded_components": sorted(excluded),
            "canonical_manifest_sha256": sha(MANIFEST),
            "source_sha256": sha(Path(__file__)),
            "prior_selection_sha256": {name: sha(ROOT / name) for name in EXCLUDE},
            "no_features_targets_masks_read": True,
        },
    )


def age(event: np.ndarray) -> np.ndarray:
    """Measure causal row age from the latest observed event or sequence origin."""
    step = np.arange(event.size)
    return step - np.maximum.accumulate(np.where(event, step, 0))


def recent(event: np.ndarray) -> np.ndarray:
    """Count current and past input refreshes in the fixed 100-row window."""
    prefix = np.r_[0, np.cumsum(event)]
    step = np.arange(event.size)
    return prefix[step + 1] - prefix[np.maximum(0, step + 1 - 100)]


def candidates(x: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Form encoded-value and CDF-coordinate proxies without physical units."""
    result: dict[str, np.ndarray] = {}
    for instrument in (0, 1):
        block = x[:, instrument * 52 : (instrument + 1) * 52].astype(np.float64)
        p, v, dp, dv = block[:, :22], block[:, 22:44], block[:, 44:48], block[:, 48:52]
        u = ndtr(v)
        prefix = f"i{instrument}_"
        result[prefix + "volume_z_sum"] = v.sum(1)
        result[prefix + "volume_z_side_contrast"] = v[:, :11].sum(1) - v[:, 11:].sum(1)
        result[prefix + "volume_z_abs_sum"] = np.abs(v).sum(1)
        result[prefix + "volume_cdf_sum"] = u.sum(1)
        result[prefix + "volume_cdf_bid_fraction"] = u[:, :11].sum(1) / np.maximum(u.sum(1), 1e-12)
        result[prefix + "price_z_range"] = np.ptp(p, axis=1)
        result[prefix + "price_z_side_contrast"] = p[:, :11].mean(1) - p[:, 11:].mean(1)
        result[prefix + "trade_price_z_sum"] = dp.sum(1)
        result[prefix + "trade_volume_z_sum"] = dv.sum(1)
        result[prefix + "trade_volume_cdf_sum"] = ndtr(dv).sum(1)
        result[prefix + "trade_encoded_cross_product"] = (dp * dv).sum(1)
        for name, values in (("price", p), ("volume", v), ("trade", block[:, 44:52])):
            event = np.r_[False, np.any(values[1:] != values[:-1], axis=1)]
            result[prefix + name + "_row_age"] = age(event)
            result[prefix + name + "_refresh_count100"] = recent(event)
    result["quote_endpoint_row_age"] = age(x[:, 110] == np.float32(-5.199337482452393))
    result["step_in_sequence"] = np.arange(x.shape[0])
    return list(result), np.column_stack(list(result.values()))


def correlations(c: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Compute tied-rank Spearman correlations without fitted transforms."""
    if not np.isfinite(c).all() or not np.isfinite(a).all():
        raise ValueError("Nonfinite raw inputs or candidate values")
    ranked = rankdata(np.column_stack([c, a]), axis=0, method="average")
    ranked -= ranked.mean(0)
    norms = np.sqrt(np.square(ranked).sum(0))
    normalized = ranked / np.maximum(norms, 1e-30)
    normalized[:, norms == 0] = 0
    return normalized[:, : c.shape[1]].T @ normalized[:, c.shape[1] :]


def read_groups(groups: list[dict[str, int]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Stream complete TRAIN groups; no target or mask columns are requested."""
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    all_c, all_a = [], []
    names: list[str] = []
    for row in groups:
        table = source.read_row_group(row["row_group"], columns=FEATURES, use_threads=False)
        x = np.column_stack([table.column(name).to_numpy() for name in FEATURES])
        names, c = candidates(x)
        all_c.append(c)
        all_a.append(x[:, 104:109].astype(np.float64))
    return names, np.concatenate(all_c), np.concatenate(all_a)


def fit() -> None:
    """Choose exactly one explanatory proxy per a field using fitting inputs."""
    selection = json.loads((HERE / "selection.json").read_text())
    if selection["source_sha256"] != sha(Path(__file__)):
        raise ValueError("Source changed after selection freeze")
    names, c, a = read_groups(selection["fit"])
    rho = correlations(c, a)
    indices = np.abs(rho).argmax(0)
    publish(
        HERE / "fit.json",
        {
            "selection_sha256": sha(HERE / "selection.json"),
            "candidate_names": names,
            "fit_spearman": rho.tolist(),
            "selected_indices": indices.tolist(),
            "selected_fit_correlations": [float(rho[index, j]) for j, index in enumerate(indices)],
            "fit_rows": len(a),
            "heldout_features_or_targets_read": False,
        },
    )


def evaluate() -> None:
    """Grade the fixed five proxies on component-held-out input-only rows."""
    selection = json.loads((HERE / "selection.json").read_text())
    if selection["source_sha256"] != sha(Path(__file__)):
        raise ValueError("Source changed after selection freeze")
    fitted = json.loads((HERE / "fit.json").read_text())
    names, c, a = read_groups(selection["evaluation"])
    if names != fitted["candidate_names"]:
        raise ValueError("Candidate schema changed")
    indices = fitted["selected_indices"]
    rho = correlations(c[:, indices], a)
    per_component = [
        np.diag(correlations(c[start : start + 20000, indices], a[start : start + 20000])).tolist()
        for start in range(0, len(a), 20000)
    ]
    summaries = []
    for j, index in enumerate(indices):
        within = np.array(per_component)[:, j]
        sign = np.sign(fitted["selected_fit_correlations"][j])
        summaries.append(
            {
                "feature": f"a{j}",
                "fixed_proxy": names[index],
                "fit_spearman": fitted["selected_fit_correlations"][j],
                "evaluation_spearman": float(rho[j, j]),
                "within_component_median_spearman": float(np.median(within)),
                "within_component_sign_adjusted_min": float((sign * within).min()),
                "components_with_sign_adjusted_rho_ge_0.95": int((sign * within >= 0.95).sum()),
            }
        )
    publish(
        HERE / "results.json",
        {
            "selection_sha256": sha(HERE / "selection.json"),
            "fit_sha256": sha(HERE / "fit.json"),
            "source_sha256": sha(Path(__file__)),
            "evaluation_rows": len(a),
            "summaries": summaries,
            "per_component_spearman": per_component,
            "targets_masks_valid_external_data_read": False,
            "claim_limit": "Encoded/CDF coordinate correlations only; no physical volume units, exact identity, target law, trained submission state or score gain follows.",
        },
    )
    print(json.dumps(summaries))


def main() -> None:
    """Run one frozen short phase and preserve measured process resources."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "fit", "evaluate"])
    args = parser.parse_args()
    start = time.monotonic()
    {"freeze": freeze, "fit": fit, "evaluate": evaluate}[args.phase]()
    publish(
        HERE / f"{args.phase}_resources.json",
        {
            "wall_seconds": time.monotonic() - start,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "address_space_limits": resource.getrlimit(resource.RLIMIT_AS),
            "cpu_limits": resource.getrlimit(resource.RLIMIT_CPU),
            "no_durable_or_cgroup_claim": True,
        },
    )


if __name__ == "__main__":
    main()
