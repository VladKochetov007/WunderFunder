"""Probe causal event-rate explanations without reading targets or scoring masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.signal import lfilter
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PREVIOUS = HERE.parent / "additional_feature_relations/selection.json"
MANIFEST = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
FEATURES = [
    f"i{instrument}_{group}{index}"
    for instrument in (0, 1)
    for group, width in (("p", 22), ("v", 22), ("dp", 4), ("dv", 4))
    for index in range(width)
] + [f"a{index}" for index in range(8)]
OUTPUT_INDICES = [0, 1, 2, 3, 4, 7]
HALF_LIVES = [8, 64, 512]


def sha(path: Path) -> str:
    """Identify immutable input manifests and source files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path: Path, value: object) -> None:
    """Preserve original evidence by refusing to replace a file."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def freeze() -> None:
    """Choose fresh canonical components using metadata alone."""
    prior = json.loads(PREVIOUS.read_text())
    excluded = set(prior["excluded_components"])
    excluded.update(row["component"] for key in ("fit", "evaluation") for row in prior[key])
    manifest = json.loads(MANIFEST.read_text())
    rows = manifest["sequences"]["train"]
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"event-rate-features-v1:{row['row_group']}".encode()
        ).digest(),
    )
    selected = []
    used = set(excluded)
    for row in ordered:
        if row["component"] not in used:
            selected.append({key: row[key] for key in ("row_group", "component")})
            used.add(row["component"])
        if len(selected) == 32:
            break
    if len(selected) != 32:
        raise ValueError("Insufficient fresh components")
    publish(
        HERE / "selection.json",
        {
            "fit": selected[:16],
            "evaluation": selected[16:],
            "excluded_components": sorted(excluded),
            "canonical_manifest_sha256": sha(MANIFEST),
            "previous_selection_sha256": sha(PREVIOUS),
            "source_sha256": sha(Path(__file__)),
            "preregistration_sha256": sha(HERE / "preregistration.json"),
            "features_targets_masks_read": False,
        },
    )


def events_and_rates(x: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Construct fixed exponentially decayed refresh rates in row units."""
    events = {}
    for instrument in (0, 1):
        offset = instrument * 52
        for name, start, stop in (("price", 0, 22), ("volume", 22, 44), ("trade", 44, 52)):
            block = x[:, offset + start : offset + stop]
            events[f"i{instrument}_{name}"] = np.r_[False, np.any(block[1:] != block[:-1], axis=1)]
    market_events = np.column_stack(list(events.values()))
    events["quote_endpoint"] = x[:, 110] == np.float32(-5.199337482452393)
    events["quote_endpoint"][0] = False
    names, rates = [], []
    for name, event in events.items():
        for half_life in HALF_LIVES:
            decay = float(np.exp2(-1.0 / half_life))
            names.append(f"{name}_rate_hl{half_life}")
            rates.append(lfilter([1.0 - decay], [1.0, -decay], event.astype(np.float64)))
    # These eight controls are never eligible for fitting proxy selection.
    step = np.arange(len(x))
    for name, event in events.items():
        names.append(f"{name}_row_age_control")
        rates.append(step - np.maximum.accumulate(np.where(event, step, 0)))
    names.append("step_index_control")
    rates.append(step)
    return names, np.column_stack(rates), market_events


def correlations(c: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Return tied Spearman correlations, with constant inputs defined as zero."""
    ranked = rankdata(np.column_stack([c, a]), axis=0, method="average")
    ranked -= ranked.mean(0)
    norms = np.sqrt(np.square(ranked).sum(0))
    ranked /= np.maximum(norms, 1e-30)
    ranked[:, norms == 0] = 0
    return ranked[:, : c.shape[1]].T @ ranked[:, c.shape[1] :]


def profile(a: np.ndarray, events: np.ndarray) -> dict[str, object]:
    """Describe movements on rows with no published book or trade refresh."""
    # Delta ending at row99 is included; the preceding row is available causally.
    delta = a[99:].astype(np.float64) - a[98:-1].astype(np.float64)
    quiet = ~events[99:].any(1)
    counts = []
    for index in range(8):
        change = delta[:, index]
        counts.append(
            {
                "feature": f"a{index}",
                "quiet_negative": int((change[quiet] < 0).sum()),
                "quiet_zero": int((change[quiet] == 0).sum()),
                "quiet_positive": int((change[quiet] > 0).sum()),
                "refresh_negative": int((change[~quiet] < 0).sum()),
                "refresh_zero": int((change[~quiet] == 0).sum()),
                "refresh_positive": int((change[~quiet] > 0).sum()),
                "distinct_values": int(np.unique(a[99:, index]).size),
                "lower_endpoint_rows": int((a[99:, index] == np.float32(-5.199337482452393)).sum()),
                "upper_endpoint_rows": int((a[99:, index] == np.float32(5.199337482452393)).sum()),
            }
        )
    return {"rows": len(delta), "quiet_rows": int(quiet.sum()), "features": counts}


def read_groups(
    groups: list[dict[str, int]],
) -> tuple[list[str], np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Stream TRAIN feature columns; labels and masks are never requested."""
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    all_c, all_a, profiles = [], [], []
    names: list[str] = []
    for row in groups:
        table = source.read_row_group(row["row_group"], columns=FEATURES, use_threads=False)
        x = np.column_stack([table.column(name).to_numpy() for name in FEATURES])
        if x.shape != (20000, 112) or not np.isfinite(x).all():
            raise ValueError("Unexpected feature shape or nonfinite inputs")
        names, c, events = events_and_rates(x)
        a = x[:, 104:112]
        all_c.append(c[99:])
        all_a.append(a[99:, OUTPUT_INDICES].astype(np.float64))
        profiles.append({**row, **profile(a, events)})
    return names, np.concatenate(all_c), np.concatenate(all_a), profiles


def run_phase(phase: str) -> None:
    """Fit a fixed proxy choice or evaluate choices on distinct components."""
    selection = json.loads((HERE / "selection.json").read_text())
    if selection["source_sha256"] != sha(Path(__file__)):
        raise ValueError("Source changed after preregistration")
    names, c, a, profiles = read_groups(selection["fit" if phase == "fit" else "evaluation"])
    if phase == "fit":
        rho = correlations(c, a)
        indices = np.abs(rho[:21]).argmax(0)
        publish(
            HERE / "fit.json",
            {
                "selection_sha256": sha(HERE / "selection.json"),
                "candidate_names": names,
                "output_features": [f"a{index}" for index in OUTPUT_INDICES],
                "fit_correlations": rho.tolist(),
                "selected_indices": indices.tolist(),
                "selected_fit_correlations": [
                    float(rho[index, j]) for j, index in enumerate(indices)
                ],
                "fit_profiles": profiles,
                "evaluation_data_read": False,
            },
        )
        return
    fitted = json.loads((HERE / "fit.json").read_text())
    if names != fitted["candidate_names"]:
        raise ValueError("Candidate ordering changed")
    indices = fitted["selected_indices"]
    rho = correlations(c[:, indices], a)
    per_component = np.array(
        [
            np.diag(correlations(c[start : start + 19901, indices], a[start : start + 19901]))
            for start in range(0, len(a), 19901)
        ]
    )
    age_indices = [21 + index // 3 for index in indices]
    age_rho = np.diag(correlations(c[:, age_indices], a))
    step_rho = correlations(c[:, [-1]], a)[0]
    summaries = []
    for j, index in enumerate(indices):
        within = per_component[:, j]
        sign = np.sign(fitted["selected_fit_correlations"][j])
        support = int((sign * within >= 0.95).sum())
        summaries.append(
            {
                "feature": fitted["output_features"][j],
                "fixed_rate": names[index],
                "fit_spearman": fitted["selected_fit_correlations"][j],
                "evaluation_spearman": float(rho[j, j]),
                "within_component_median": float(np.median(within)),
                "support_components": support,
                "strong_rank_association_gate": support >= 12,
                "matched_event_row_age_control_spearman": float(age_rho[j]),
                "step_index_control_spearman": float(step_rho[j]),
            }
        )
    nonsaturated = []
    for component in range(16):
        start = component * 19901
        stop = start + 19901
        diagnostics = []
        for j, index in enumerate(indices):
            values = a[start:stop, j]
            supported = (values != np.float32(-5.199337482452393)) & (
                values != np.float32(5.199337482452393)
            )
            value = None
            if supported.sum() >= 2:
                value = float(
                    correlations(c[start:stop][supported][:, [index]], values[supported, None])[
                        0, 0
                    ]
                )
            diagnostics.append({"rows": int(supported.sum()), "spearman": value})
        nonsaturated.append(diagnostics)
    publish(
        HERE / "results.json",
        {
            "selection_sha256": sha(HERE / "selection.json"),
            "fit_sha256": sha(HERE / "fit.json"),
            "source_sha256": sha(Path(__file__)),
            "evaluation_rows": len(a),
            "summaries": summaries,
            "per_component_correlations": per_component.tolist(),
            "nonsaturated_per_component": nonsaturated,
            "evaluation_profiles": profiles,
            "targets_masks_VALID_external_data_read": False,
            "claim_limit": "Strong monotonic association only; rank does not identify numeric rates. No physical time unit, exact feature identity, target law or public score improvement established.",
        },
    )


def main() -> None:
    """Run one bounded phase and save measured resource evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "fit", "evaluate"])
    args = parser.parse_args()
    start = time.monotonic()
    if args.phase == "freeze":
        freeze()
    else:
        run_phase(args.phase)
    publish(
        HERE / f"{args.phase}_resources.json",
        {
            "wall_seconds": time.monotonic() - start,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "address_space_limit_bytes": resource.getrlimit(resource.RLIMIT_AS)[0],
            "cpu_time_limit_seconds": resource.getrlimit(resource.RLIMIT_CPU)[0],
            "thread_pools": 1,
            "gpu_seconds": 0,
            "cgroup_or_durable_execution_claim": False,
        },
    )


if __name__ == "__main__":
    main()
