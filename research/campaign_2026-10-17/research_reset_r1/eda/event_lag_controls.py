#!/usr/bin/env python3
"""Label-free, fixed-lag event-dependence audit on a seeded sequence sample.

Reads only selected price/volume feature columns. It does not load targets,
fit a model, choose lags from labels, or pool rows as independent observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PRICE_COLUMNS = {
    "i0": [f"i0_p{i}" for i in range(22)],
    "i1": [f"i1_p{i}" for i in range(22)],
}
VOLUME_COLUMNS = {
    "i0": [f"i0_v{i}" for i in range(22)],
    "i1": [f"i1_v{i}" for i in range(22)],
}
LAGS = (1, 4, 16, 64)
CONTROL_SHIFT = 1000
HALVES = 2


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_manifest(summary_path: Path) -> dict:
    """Choose first 8 group-sorted cached entries per seeded class/split."""
    source = json.loads(summary_path.read_text())
    if source.get("seed") != 20261017 or source.get("per_class") != 32:
        raise ValueError("Unexpected seeded summary; refusing to change membership")
    entries = []
    for split in ("train", "valid"):
        strata = source["splits"][split]["strata"]
        for cls in ("0", "1", "2", "3"):
            seqs = strata[cls]["sequences"]
            ordered = sorted(seqs, key=lambda r: r["row_group"])
            if len(ordered) != 32:
                raise ValueError(f"Expected 32 seeded entries in {split}/{cls}")
            for rank, row in enumerate(ordered[:8]):
                entries.append({
                    "split": split,
                    "retrospective_class_stratum": int(cls),
                    "rank_in_seeded_32_by_row_group": rank,
                    "row_group": int(row["row_group"]),
                    "seq_ix": int(row["seq_ix"]),
                    "expected_rows": 20_000,
                })
    return {
        "seed": 20261017,
        "seeded_summary_sha256": sha256_file(summary_path),
        "selection": "first 8 row-group-sorted entries from each existing seeded 32-sequence split/class stratum; no reselection",
        "sequence_count": len(entries),
        "expected_total_rows": len(entries) * 20_000,
        "entries": entries,
    }


def change_any(values: np.ndarray) -> np.ndarray:
    """Event at t means at least one component differs from t-1; first row false."""
    out = np.zeros(values.shape[0], dtype=bool)
    if len(out) > 1:
        out[1:] = np.any(values[1:] != values[:-1], axis=1)
    return out


def event_flags(columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    p0_changes = np.zeros(len(columns["i0_p0"]), dtype=bool)
    p11_changes = np.zeros_like(p0_changes)
    if len(p0_changes) > 1:
        p0_changes[1:] = columns["i0_p0"][1:] != columns["i0_p0"][:-1]
        p11_changes[1:] = columns["i0_p11"][1:] != columns["i0_p11"][:-1]
    return {
        "i0_price_p0_and_p11": p0_changes & p11_changes,
        "i0_price_any22": change_any(np.column_stack(
            [columns[c] for c in PRICE_COLUMNS["i0"]])),
        "i1_price_any22": change_any(np.column_stack(
            [columns[c] for c in PRICE_COLUMNS["i1"]])),
        "i0_volume_any22": change_any(np.column_stack(
            [columns[c] for c in VOLUME_COLUMNS["i0"]])),
        "i1_volume_any22": change_any(np.column_stack(
            [columns[c] for c in VOLUME_COLUMNS["i1"]])),
    }


def phi(a: np.ndarray, b: np.ndarray) -> float | None:
    """Binary covariance normalized by the Bernoulli marginal standard deviations."""
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("Invalid aligned binary vectors")
    pa_, pb_ = float(a.mean()), float(b.mean())
    denom = np.sqrt(pa_ * (1.0 - pa_) * pb_ * (1.0 - pb_))
    if denom == 0:
        return None
    return float((np.mean(a & b) - pa_ * pb_) / denom)


def cross_phi(i1: np.ndarray, i0: np.ndarray, lag: int) -> float | None:
    """Positive lag pairs i1[t] with later i0[t+lag]; negative reverses direction."""
    if lag > 0:
        return phi(i1[:-lag], i0[lag:])
    k = -lag
    return phi(i1[k:], i0[:-k])


def summarize_values(values: list[float | None]) -> dict:
    vals = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(vals):
        return {"n_sequences": 0, "median_sequence_value": None,
                "mean_sequence_value": None, "fraction_positive": None}
    return {
        "n_sequences": int(len(vals)),
        "median_sequence_value": float(np.median(vals)),
        "mean_sequence_value": float(np.mean(vals)),
        "fraction_positive": float(np.mean(vals > 0)),
    }


def audit_sequence(entry: dict, columns: dict[str, np.ndarray]) -> dict:
    n = len(columns["i0_p0"])
    if n != entry["expected_rows"]:
        raise ValueError(f"Expected 20,000 rows, got {n}: {entry}")
    flags = event_flags(columns)
    half_results = []
    half_size = n // HALVES
    for half in range(HALVES):
        lo, hi = half * half_size, (half + 1) * half_size
        f = {k: v[lo:hi] for k, v in flags.items()}
        i1, i0 = f["i1_price_any22"], f["i0_price_p0_and_p11"]
        shifted_i1 = np.roll(i1, CONTROL_SHIFT)
        cross = {}
        for lag in LAGS:
            forward = cross_phi(i1, i0, lag)
            reverse = cross_phi(i1, i0, -lag)
            control_forward = cross_phi(shifted_i1, i0, lag)
            control_reverse = cross_phi(shifted_i1, i0, -lag)
            cross[str(lag)] = {
                "i1_leads_i0_phi": forward,
                "reverse_direction_phi": reverse,
                "lead_minus_reverse": (forward - reverse)
                    if forward is not None and reverse is not None else None,
                "shift1000_i1_control_lead_phi": control_forward,
                "shift1000_i1_control_reverse_phi": control_reverse,
                "shift1000_control_lead_minus_reverse": (
                    control_forward - control_reverse
                    if control_forward is not None and control_reverse is not None else None),
                "observed_lead_phi_minus_shift_control": (
                    forward - control_forward
                    if forward is not None and control_forward is not None else None),
            }
        autocorr = {}
        for event_name in ("i0_price_p0_and_p11", "i0_price_any22",
                           "i1_price_any22", "i0_volume_any22", "i1_volume_any22"):
            event = f[event_name]
            autocorr[event_name] = {}
            for lag in LAGS:
                autocorr[event_name][str(lag)] = phi(event[:-lag], event[lag:])
        half_results.append({
            "half": half,
            "n_rows": hi - lo,
            "event_counts": {k: int(v.sum()) for k, v in f.items()},
            "cross_instrument": cross,
            "within_instrument_event_autocorrelation": autocorr,
        })
    return {
        "split": entry["split"],
        "retrospective_class_stratum": entry["retrospective_class_stratum"],
        "row_group": entry["row_group"],
        "seq_ix": entry["seq_ix"],
        "rows": n,
        "halves": half_results,
    }


def aggregate_sequences(rows: list[dict]) -> dict:
    """Summarize half-averaged sequence values; halves/rows are not iid units."""
    out: dict = {"n_sequences": len(rows), "lags": {}}
    for lag in LAGS:
        key = str(lag)
        fields = (
            "i1_leads_i0_phi", "reverse_direction_phi", "lead_minus_reverse",
            "shift1000_i1_control_lead_phi", "shift1000_i1_control_reverse_phi",
            "shift1000_control_lead_minus_reverse", "observed_lead_phi_minus_shift_control",
        )
        lag_out = {}
        per_seq: dict[str, list[float | None]] = {f: [] for f in fields}
        per_half_delta, per_half_control_delta = [], []
        for row in rows:
            vals = [h["cross_instrument"][key] for h in row["halves"]]
            for field in fields:
                xs = [v[field] for v in vals if v[field] is not None]
                per_seq[field].append(float(np.mean(xs)) if xs else None)
            per_half_delta.extend(v["lead_minus_reverse"] for v in vals
                                  if v["lead_minus_reverse"] is not None)
            per_half_control_delta.extend(v["shift1000_control_lead_minus_reverse"] for v in vals
                                          if v["shift1000_control_lead_minus_reverse"] is not None)
        for field in fields:
            lag_out[field] = summarize_values(per_seq[field])
        lag_out["half_sign_consistency"] = {
            "lead_minus_reverse_fraction_positive": (
                float(np.mean(np.asarray(per_half_delta) > 0)) if per_half_delta else None),
            "shift_control_delta_fraction_positive": (
                float(np.mean(np.asarray(per_half_control_delta) > 0))
                if per_half_control_delta else None),
            "n_halves": len(per_half_delta),
        }
        out["lags"][key] = lag_out
    auto = {}
    for event_name in ("i0_price_p0_and_p11", "i0_price_any22", "i1_price_any22",
                       "i0_volume_any22", "i1_volume_any22"):
        auto[event_name] = {}
        for lag in LAGS:
            per_seq_values = []
            for row in rows:
                vals = [h["within_instrument_event_autocorrelation"][event_name][str(lag)]
                        for h in row["halves"]]
                vals = [v for v in vals if v is not None]
                per_seq_values.append(float(np.mean(vals)) if vals else None)
            auto[event_name][str(lag)] = summarize_values(per_seq_values)
    out["within_instrument_autocorrelation"] = auto
    return out


def run(data_dir: Path, manifest: dict) -> dict:
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    start_cpu = resource.getrusage(resource.RUSAGE_SELF)
    start_wall = time.time()
    results = []
    files = {}
    columns = [*PRICE_COLUMNS["i0"], *PRICE_COLUMNS["i1"],
               *VOLUME_COLUMNS["i0"], *VOLUME_COLUMNS["i1"]]
    for split in ("train", "valid"):
        data_path = data_dir / f"{split}.parquet"
        parquet = pq.ParquetFile(data_path)
        files[split] = {
            "basename": data_path.name,
            "bytes": data_path.stat().st_size,
            "mtime_ns": data_path.stat().st_mtime_ns,
        }
        entries = [e for e in manifest["entries"] if e["split"] == split]
        for entry in entries:
            table = parquet.read_row_group(entry["row_group"], columns=columns, use_threads=False)
            arrs = {c: table[c].to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
                    for c in columns}
            if len(arrs["i0_p0"]) != entry["expected_rows"]:
                raise ValueError(f"Row group length mismatch: {entry}")
            results.append(audit_sequence(entry, arrs))
            del table, arrs
    by_stratum = {}
    for split in ("train", "valid"):
        for cls in range(4):
            key = f"{split}_class{cls}"
            group = [r for r in results if r["split"] == split
                     and r["retrospective_class_stratum"] == cls]
            by_stratum[key] = aggregate_sequences(group)
    end_cpu = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "status": "completed",
        "analysis": "fixed-lag binary event dependence only; no labels, fitting, or prediction",
        "manifest": manifest,
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "data_files": files,
        "feature_groups": {
            "i0_price": PRICE_COLUMNS["i0"], "i1_price": PRICE_COLUMNS["i1"],
            "i0_volume": VOLUME_COLUMNS["i0"], "i1_volume": VOLUME_COLUMNS["i1"],
            "excluded": ["i0_dp*", "i0_dv*", "i1_dp*", "i1_dv*", "a0..a7",
                         "targets", "need_prediction", "is_scored", "seq_ix as predictor"],
            "i0_price_p0_and_p11": ["i0_p0", "i0_p11"],
        },
        "method": {
            "events": "component changed from preceding row; each 10,000-row half starts with false events",
            "binary_dependence": "phi coefficient / binary Pearson correlation",
            "positive_lag": "i1 event at t paired with later i0 event at t+lag",
            "lags_rows": list(LAGS),
            "negative_control": f"circularly roll i1 price event by {CONTROL_SHIFT} rows within each half; preserves event count and cyclic i1 autocorrelation, randomizes phase relative to i0",
            "aggregation": "per-sequence average of its two halves; medians/means/sign fractions over sequences, no row-level standard errors or iid claims",
        },
        "sequence_count": len(results),
        "observed_rows": sum(r["rows"] for r in results),
        "per_sequence": results,
        "per_stratum": by_stratum,
        "runtime": {
            "wall_seconds": time.time() - start_wall,
            "user_cpu_seconds": end_cpu.ru_utime - start_cpu.ru_utime,
            "system_cpu_seconds": end_cpu.ru_stime - start_cpu.ru_stime,
            "maxrss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.manifest_out:
        if not args.summary:
            parser.error("--manifest-out requires --summary")
        args.manifest_out.write_text(json.dumps(make_manifest(args.summary), indent=2) + "\n")
        return
    if not args.data_dir or not args.manifest or not args.output:
        parser.error("analysis requires --data-dir, --manifest, and --output")
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("sequence_count") != 64 or manifest.get("expected_total_rows") != 1_280_000:
        raise ValueError("Unexpected manifest; refusing to run")
    result = run(args.data_dir, manifest)
    result["script_sha256"] = sha256_file(Path(__file__))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["runtime"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
