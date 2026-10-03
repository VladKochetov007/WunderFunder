from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from decoder import ENDPOINT, HERE, QUANTA, ROOT, decode, digest, load_tables, read_group
from path_laws import FAMILIES, functional


def sample_indices(event_rows: np.ndarray) -> np.ndarray:
    """Choose input-only origin indices without filtering decoded paths or labels."""
    selected = np.flatnonzero(event_rows >= 99)[::5]
    if len(selected) > 128:
        selected = selected[np.linspace(0, len(selected) - 1, 128, dtype=int)]
    return selected


def window(
    decoded: dict, events: np.ndarray, index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a bounded event path and keep broken-prefix mass explicit."""
    rows = events[index : index + 33]
    origin = events[index]
    segment = decoded["segment"][origin]
    valid = (decoded["segment"][rows] == segment) & (segment >= 0)
    return decoded["ask"][rows], decoded["bid"][rows], valid


def origin_observables(
    ask: np.ndarray, bid: np.ndarray, valid: np.ndarray, targets: np.ndarray
) -> dict:
    """Check implied exit touch/extremum geometry; labels remain diagnostic only."""
    answer = []
    levels = [ask[0] + targets[0], bid[0] - targets[1]]
    for side in range(2):
        first = -1
        extreme = None
        for j in range(1, len(ask)):
            if not valid[j]:
                break
            quote = bid[j] if side == 0 else ask[j]
            if abs(quote - levels[side]) < 0.02:
                first = j
                extreme = (
                    bool(quote >= bid[1 : j + 1].max())
                    if side == 0
                    else bool(quote <= ask[1 : j + 1].min())
                )
                break
        answer.append({"first_exit_side_touch_lag": first, "first_touch_running_extreme": extreme})
    return {"target_tick_pair": targets.tolist(), "exit_side_touch": answer}


def process(
    source: pq.ParquetFile, group: int, component: int, spreads: dict, moves: dict, controls: bool
) -> tuple[list[dict], dict]:
    """Stream one fixed group, evaluating every preselected origin including unknowns."""
    x, a6, y, seq_id = read_group(source, group)
    decoded = decode(x, spreads, moves)
    ev = np.flatnonzero(a6.astype(np.float32) == ENDPOINT)
    selected = sample_indices(ev)
    rng = np.random.default_rng(20261003 + group)
    permutation = rng.permutation(len(ev))
    records = []
    prefix_ok = True
    for index in selected:
        row = int(ev[index])
        cls = int(decoded["class"][row])
        q = float(QUANTA[cls]) if cls >= 0 else None
        ask, bid, valid = window(decoded, ev, int(index))
        if cls < 0:
            valid = np.zeros_like(valid)
        next_row = int(ev[index + 1]) if index + 1 < len(ev) else len(y)
        weight = float(np.abs(np.clip(y[row], -2, 2)).sum()) * (next_row - row)
        entry = {
            "group": group,
            "component": component,
            "seq_ix": seq_id,
            "step": row,
            "event_index": int(index),
            "fit_code_class": cls,
            "quantum_from_fit_code": q,
            "target_raw": y[row].tolist(),
            "represented_plateau_rows": next_row - row,
            "diagnostic_weight": weight,
            "laws": {},
        }
        if q is not None:
            entry["label_lattice_residual"] = float(
                np.max(np.abs(y[row] - np.rint(y[row] / q) * q))
            )
            entry["target_geometry"] = origin_observables(ask, bid, valid, y[row] / q)
        conditions = {"aligned": (ask, bid, valid)}
        if controls:
            conditions["identity"] = (ask.copy(), bid.copy(), valid.copy())
            for name, donor in [
                ("shift997", (int(index) + 997) % len(ev)),
                ("shuffle", int(permutation[index])),
            ]:
                da, db, dv = window(decoded, ev, donor)
                da, db = da - db[0] + bid[0], db - db[0] + bid[0]
                da[0], db[0] = ask[0], bid[0]
                dv[0] &= valid[0]
                if not valid[0]:
                    dv[:] = False
                conditions[name] = (da, db, dv)
        for family in FAMILIES:
            entry["laws"][family] = {}
            for name, (a, b, v) in conditions.items():
                output = functional(a, b, v, family)
                reasons = output["reasons"]
                if cls < 0:
                    reasons = ["missing_origin_class"] * 2
                pair = output["pair"] * q if q is not None else np.full(2, np.nan)
                entry["laws"][family][name] = {
                    "prediction_raw": pair.tolist(),
                    "lags": output["lags"].tolist(),
                    "reasons": reasons,
                }
            if controls:
                a, b = entry["laws"][family]["aligned"], entry["laws"][family]["identity"]
                if a["reasons"] != b["reasons"] or not np.array_equal(
                    a["prediction_raw"], b["prediction_raw"], equal_nan=True
                ):
                    raise AssertionError("Identity path control changed output")
        records.append(entry)
    # Prefix replay checks sample input-only boundaries, not label-selected rows.
    for end in [256, 1024]:
        prefix = decode(x[:end], spreads, moves)
        for field in ["ask", "bid", "segment", "class", "spread"]:
            prefix_ok &= bool(np.array_equal(decoded[field][:end], prefix[field]))
    events = np.r_[0, np.flatnonzero(np.any(x[1:] != x[:-1], axis=1)) + 1]
    keys = [row.tobytes() for row in np.round(x[events], 4).astype(np.float32)]
    transport = Counter()
    for i, row in enumerate(events[1:], 1):
        cls = int(decoded["class"][row])
        move = moves.get((keys[i - 1], keys[i]))
        if cls >= 0 and move is not None:
            q = QUANTA[cls]
            ticks = np.rint(y[[row - 1, row]] / q).astype(int)
            if (ticks <= 0).all():
                transport["known_interior_transition_checks"] += 1
                dy = ticks[1] - ticks[0]
                transport["exact_interior_transition_checks"] += move == (-int(dy[0]), int(dy[1]))
    summary = {
        "group": group,
        "component": component,
        "seq_ix": seq_id,
        "origins": len(records),
        "sampled_steps": [r["step"] for r in records],
        "all_endpoint_events": len(ev),
        "prefix_checks_pass": prefix_ok,
        "decoder_counts": decoded["decoder_counts"],
        "known_transition_transport": dict(transport),
        "array_sha256": hashlib.sha256(x.tobytes() + a6.tobytes() + y.tobytes()).hexdigest(),
    }
    return records, summary


def summary_stats(
    records: list[dict], family: str, condition: str, calibration: np.ndarray | None = None
) -> dict:
    """Report complete and unresolved origin counts and weight on the same denominator."""
    total_weight = sum(r["diagnostic_weight"] for r in records)
    reasons = Counter()
    reason_weights = Counter()
    errors, weights, predictions, lags = [], [], [], []
    exact, exact_weight = 0, 0.0
    for r in records:
        item = r["laws"][family][condition]
        p = np.array(item["prediction_raw"])
        if np.isfinite(p).all():
            if calibration is not None:
                p = np.r_[1, p] @ calibration
            error = p - np.array(r["target_raw"])
            errors.append(error)
            weights.append(r["diagnostic_weight"])
            predictions.append(p)
            lags.extend(item["lags"])
            is_exact = bool(np.all(np.abs(error) <= 0.002))
            exact += is_exact
            exact_weight += is_exact * r["diagnostic_weight"]
            reason = "resolved_pair"
        else:
            reason = next(
                (
                    v
                    for v in [
                        "missing_origin_class",
                        "unknown_origin",
                        "unknown_transition",
                        "sequence_end",
                        "horizon_censored",
                    ]
                    if v in item["reasons"]
                ),
                "unresolved",
            )
        reasons[reason] += 1
        reason_weights[reason] += r["diagnostic_weight"]
    result = {
        "all_origins": len(records),
        "all_weight": total_weight,
        "reason_counts": dict(reasons),
        "reason_weight_shares": {k: v / total_weight for k, v in reason_weights.items()},
        "complete_pairs": len(errors),
        "exact_pairs": exact,
        "exact_fraction_of_all_origins": exact / len(records),
        "exact_weight_fraction_all": exact_weight / total_weight,
        "exact_fraction_of_complete": exact / len(errors) if errors else None,
    }
    if errors:
        e, w, p = np.array(errors), np.array(weights), np.array(predictions)
        result.update(
            {
                "weighted_raw_mae_pair": np.average(np.abs(e), weights=w, axis=0).tolist(),
                "weighted_sum_mae": float(np.average(np.abs(e.sum(axis=1)), weights=w)),
                "weighted_difference_mae": float(np.average(np.abs(e[:, 0] - e[:, 1]), weights=w)),
                "predicted_both_positive_fraction": float(np.mean((p > 0).all(axis=1))),
                "stop_lag_quantiles": np.quantile(lags, [0.5, 0.9, 0.99]).tolist(),
            }
        )
    return result


def main() -> None:
    """Fit only a frozen diagnostic affine map, then grade fixed TRAIN test components."""
    started = time.monotonic()
    state = json.loads((HERE / "STATE.json").read_text())
    state["phase"] = "calibration_on_fit_before_test"
    (HERE / "STATE.json").write_text(json.dumps(state, indent=2))
    panel = json.loads((HERE / "selection.json").read_text())
    spreads, moves = load_tables()
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    if "--resume" in sys.argv:
        calibration = json.loads((HERE / "affine_calibration.json").read_text())
        assert digest(HERE / "affine_calibration.json") == digest(
            HERE / "affine_calibration_initial.json"
        )
        fit_sample_count = calibration[FAMILIES[0]]["fit_all_origins"]
    else:
        fit_records = []
        for g, c in zip(panel["fit_groups"], panel["fit_components"], strict=True):
            rows, _ = process(source, g, c, spreads, moves, False)
            fit_records.extend(rows)
        calibration = {}
        for family in FAMILIES:
            rows = [
                r
                for r in fit_records
                if np.isfinite(r["laws"][family]["aligned"]["prediction_raw"]).all()
            ]
            design = np.array([[1] + r["laws"][family]["aligned"]["prediction_raw"] for r in rows])
            targets = np.array([r["target_raw"] for r in rows])
            coef = (
                np.linalg.lstsq(design, targets, rcond=None)[0]
                if len(rows) >= 3
                else np.full((3, 2), np.nan)
            )
            calibration[family] = {
                "coefficients": coef.tolist(),
                "fit_complete_origins": len(rows),
                "fit_all_origins": len(fit_records),
            }
        (HERE / "affine_calibration.json").write_text(json.dumps(calibration, indent=2))
        fit_sample_count = len(fit_records)
    state["phase"] = "test_grade_frozen_calibration"
    state["affine_sha256"] = digest(HERE / "affine_calibration.json")
    (HERE / "STATE.json").write_text(json.dumps(state, indent=2))
    test_records, sequences = [], []
    for g, c in zip(panel["test_groups"], panel["test_components"], strict=True):
        rows, sequence = process(source, g, c, spreads, moves, True)
        test_records.extend(rows)
        sequences.append(sequence)
        print(
            f"test group={g} component={c} id={sequence['seq_ix']} origins={len(rows)}", flush=True
        )
    scores = {}
    for family in FAMILIES:
        scores[family] = {
            condition: summary_stats(test_records, family, condition)
            for condition in ["aligned", "identity", "shift997", "shuffle"]
        }
        scores[family]["affine_aligned_secondary"] = summary_stats(
            test_records, family, "aligned", np.array(calibration[family]["coefficients"])
        )
    result = {
        "scope": "Previously exposed TRAIN component-held-out diagnostics only",
        "families": scores,
        "fit_samples_origins": fit_sample_count,
        "test_samples": sequences,
        "observed_both_positive_origins": int(
            sum(np.all(np.array(r["target_raw"]) > 0) for r in test_records)
        ),
        "identity_pass": True,
        "all_prefix_checks_pass": all(r["prefix_checks_pass"] for r in sequences),
        "hashes": {
            f: digest(HERE / f)
            for f in [
                "PREREG.json",
                "selection.json",
                "decoder_tables.json",
                "affine_calibration.json",
                "evaluate.py",
                "path_laws.py",
            ]
        },
        "cost": {
            "wall_seconds": time.monotonic() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime
            + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        },
    }
    (HERE / "sampled_origins.json").write_text(json.dumps(test_records))
    (HERE / "results.json").write_text(json.dumps(result, indent=2))
    state["phase"] = "fixed_test_complete"
    state["test_ids"] = [
        {"group": r["group"], "component": r["component"], "seq_ix": r["seq_ix"]} for r in sequences
    ]
    state["pending"] = ["inspect results", "verify scripts", "final report and hashes"]
    (HERE / "STATE.json").write_text(json.dumps(state, indent=2))
    print(json.dumps({"family_scores": scores, "cost": result["cost"]}))


if __name__ == "__main__":
    main()
