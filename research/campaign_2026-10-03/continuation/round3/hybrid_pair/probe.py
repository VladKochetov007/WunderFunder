from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from laws import LAWS, run_law
from synthetic import json_safe

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PARENT = ROOT / "research/campaign_2026-10-03/continuation/round2/first_passage"
FROZEN_HASHES = {
    "decoder.py": "4b1105e6a1ed6d30e8c573a5b272a99418d2d9fdbc8daf312803f0493af27ec8",
    "decoder_tables.json": "43d9d1e256e9ede1a06ddfd58429ff3c3cdf50ccf815ec07a677c977d70467bb",
    "selection.json": "d63d62a7d0ad63aaa3b958d07390b0efccdea77f263d94e44bf8734654354e57",
    "sampled_origins.json": "b87b9a6f68bf2958a98713e568813c88828197b3ba6429970eb40287208ba7a0",
}


def digest(path: Path) -> str:
    """Hash small source and selected-data artifacts, not entire parquet files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_parent() -> None:
    """Refuse any mutable parent input, keeping the round2 decoder and panel frozen."""
    for name, expected in FROZEN_HASHES.items():
        if digest(PARENT / name) != expected:
            raise ValueError(f"Frozen parent changed: {name}")


def load_decoder() -> object:
    """Import inspected frozen pure replay helpers without running its fitting entrypoint."""
    spec = importlib.util.spec_from_file_location("round2_frozen_decoder", PARENT / "decoder.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load frozen decoder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def group_summary(records: list[dict], law: str, condition: str, regime: str = "all") -> dict:
    """Count complete and unresolved pairs against the full preselected denominator."""
    rows = [r for r in records if regime == "all" or r["regime"] == regime]
    reasons, reason_weight = Counter(), Counter()
    errors, predictions, weights = [], [], []
    exact_pair, exact_sides = 0, np.zeros(2, dtype=int)
    exact_weight = 0.0
    outside_sides = Counter()
    for r in rows:
        out = r["outputs"][law][condition]
        p = np.asarray(out["prediction_raw"], dtype=float)
        w = r["weight"]
        if np.isfinite(p).all():
            error = p - np.array(r["target_raw"])
            exact = np.abs(error) <= 0.002
            is_exact = bool(exact.all())
            exact_pair += int(is_exact)
            exact_weight += w * is_exact
            exact_sides += exact
            errors.append(error)
            predictions.append(p)
            weights.append(w)
            if r["regime"] == "outside":
                for side, y in enumerate(r["target_raw"]):
                    outside_sides["winner_exact" if y > 0 else "loser_exact"] += int(exact[side])
            reason = "complete_pair"
        else:
            reason = next(
                (
                    s
                    for s in [
                        "missing_origin_class",
                        "unknown_origin",
                        "unknown_transition",
                        "sequence_end",
                        "horizon_censored",
                    ]
                    if s in out["reasons"]
                ),
                "unresolved",
            )
        reasons[reason] += 1
        reason_weight[reason] += w
    total_weight = sum(r["weight"] for r in rows)
    result = {
        "origins": len(rows),
        "all_weight": total_weight,
        "reason_counts": dict(reasons),
        "reason_weight_shares": {
            k: v / total_weight if total_weight else None for k, v in reason_weight.items()
        },
        "complete_pairs": len(errors),
        "exact_pairs": exact_pair,
        "exact_sides": exact_sides.tolist(),
        "outside_side_exact": dict(outside_sides),
        "exact_fraction_all_origins": exact_pair / len(rows) if rows else None,
        "exact_fraction_complete": exact_pair / len(errors) if errors else None,
        "exact_weight_share_all": exact_weight / total_weight if total_weight else None,
    }
    if errors:
        e, p, w = np.array(errors), np.array(predictions), np.array(weights)
        result.update(
            {
                "raw_pair_mae": np.mean(np.abs(e), axis=0).tolist(),
                "raw_pair_rmse": np.sqrt(np.mean(e * e, axis=0)).tolist(),
                "weighted_raw_pair_mae": np.average(np.abs(e), weights=w, axis=0).tolist(),
                "sum_mae": float(np.abs(e.sum(axis=1)).mean()),
                "difference_mae": float(np.abs(e[:, 0] - e[:, 1]).mean()),
                "predicted_both_positive": int((p > 0).all(axis=1).sum()),
            }
        )
    return result


def common_summary(records: list[dict], law: str, control: str, regime: str) -> dict:
    """Compare shifted paths and old extrema on common complete cases, separately from coverage."""
    pairs = []
    for r in records:
        if regime != "all" and r["regime"] != regime:
            continue
        aligned = r["outputs"][law]["aligned"]
        other = (
            r["outputs"]["extrema_only"]["aligned"]
            if control == "extrema_only"
            else r["outputs"][law][control]
        )
        p, q = (
            np.asarray(aligned["prediction_raw"], float),
            np.asarray(other["prediction_raw"], float),
        )
        if np.isfinite(p).all() and np.isfinite(q).all():
            y = np.array(r["target_raw"])
            pairs.append(
                (bool(np.all(np.abs(p - y) <= 0.002)), bool(np.all(np.abs(q - y) <= 0.002)))
            )
    return {
        "common_complete_origins": len(pairs),
        "aligned_exact": sum(a for a, _ in pairs),
        "control_exact": sum(b for _, b in pairs),
    }


def main() -> None:
    """Score two frozen hybrids on raw replay of the unchanged exposed TRAIN panel."""
    started = time.monotonic()
    state = json.loads((HERE / "STATE.json").read_text())
    state["phase"] = "scoring_frozen_laws_after_synthetic"
    state["commands"].append(
        "run_cpu.sh round3-hybrid-pair env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research/campaign_2026-10-03/continuation/round3/hybrid_pair/probe.py"
    )
    (HERE / "STATE.json").write_text(json.dumps(state, indent=2))
    assert (
        json.loads((HERE / "synthetic_results.json").read_text())["stage"]
        == "SYNTHETIC PASSED BEFORE RAW TRAIN SCORING"
    )
    verify_parent()
    decoder = load_decoder()
    spreads, moves = decoder.load_tables()
    parent_records = json.loads((PARENT / "sampled_origins.json").read_text())
    old_sequences = {
        r["group"]: r for r in json.loads((PARENT / "results.json").read_text())["test_samples"]
    }
    panel = json.loads((PARENT / "selection.json").read_text())
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    records, sequences = [], []
    control_checked, no_unknown_skips = 0, 0
    for group, component in zip(panel["test_groups"], panel["test_components"], strict=True):
        x, a6, y, seq_id = decoder.read_group(source, group)
        data_hash = hashlib.sha256(x.tobytes() + a6.tobytes() + y.tobytes()).hexdigest()
        assert data_hash == old_sequences[group]["array_sha256"]
        decoded = decoder.decode(x, spreads, moves)
        events = np.flatnonzero(a6.astype(np.float32) == decoder.ENDPOINT)
        permutation = np.random.default_rng(20261003 + group).permutation(len(events))
        origins = [r for r in parent_records if r["group"] == group]
        prefix_ok = True
        for end in [256, 1024]:
            prefix = decoder.decode(x[:end], spreads, moves)
            prefix_ok &= all(
                np.array_equal(prefix[f], decoded[f][:end])
                for f in ["ask", "bid", "segment", "class", "spread"]
            )
        assert prefix_ok
        for old in origins:
            row, index = old["step"], old["event_index"]
            assert int(events[index]) == row and seq_id == old["seq_ix"]
            assert np.array_equal(y[row], old["target_raw"])
            cls = int(decoded["class"][row])
            q = float(decoder.QUANTA[cls]) if cls >= 0 else None
            assert cls == old["fit_code_class"] and q == old["quantum_from_fit_code"]
            base_rows = events[index : index + 33]
            origin_segment = decoded["segment"][row]
            ask, bid = decoded["ask"][base_rows], decoded["bid"][base_rows]
            valid = (decoded["segment"][base_rows] == origin_segment) & (origin_segment >= 0)
            if cls < 0:
                valid[:] = False
            paths = {
                "aligned": (ask, bid, valid),
                "identity": (ask.copy(), bid.copy(), valid.copy()),
            }
            for name, donor in [
                ("shift997", (index + 997) % len(events)),
                ("shuffle", int(permutation[index])),
            ]:
                donor_rows = events[donor : donor + 33]
                donor_segment = decoded["segment"][events[donor]]
                da = decoded["ask"][donor_rows] - decoded["bid"][events[donor]] + bid[0]
                db = decoded["bid"][donor_rows] - decoded["bid"][events[donor]] + bid[0]
                dv = (decoded["segment"][donor_rows] == donor_segment) & (donor_segment >= 0)
                da[0], db[0] = ask[0], bid[0]
                dv[0] &= valid[0]
                if not valid[0]:
                    dv[:] = False
                paths[name] = (da, db, dv)
            record = {
                "group": group,
                "component": component,
                "seq_ix": seq_id,
                "step": row,
                "event_index": index,
                "regime": "outside" if (y[row] > 0).any() else "interior",
                "fit_code_class": cls,
                "quantum": q,
                "target_raw": y[row].tolist(),
                "entry_spread_ticks": int(decoded["spread"][row]),
                "diagnostic_path": {
                    "ask": ask.tolist(),
                    "bid": bid.tolist(),
                    "known_contiguous": valid.tolist(),
                    "event_steps": base_rows.tolist(),
                },
                "weight": old["diagnostic_weight"],
                "outputs": {},
            }
            for law in LAWS:
                record["outputs"][law] = {}
                for condition, (a, b, v) in paths.items():
                    out = run_law(a, b, v, law)
                    if cls < 0:
                        out["reasons"] = ["missing_origin_class"] * 2
                    raw = out["pair"] * q if q is not None else np.full(2, np.nan)
                    if np.isfinite(raw).all():
                        assert out["required_prefix_lag"] >= 1
                        assert v[: out["required_prefix_lag"] + 1].all()
                        no_unknown_skips += 1
                    item = {
                        "prediction_raw": raw.tolist(),
                        "reasons": out["reasons"],
                        "lags": out["lags"],
                        "required_prefix_lag": out["required_prefix_lag"],
                        "selected_side": out["selected_side"],
                        "selected_lag": out["selected_lag"],
                        "raw_residual": (raw - y[row]).tolist(),
                    }
                    record["outputs"][law][condition] = item
                    if law == "extrema_only":
                        parent = old["laws"]["trailing_extrema"][condition]
                        assert parent["reasons"] == item["reasons"]
                        assert parent["lags"] == item["lags"]
                        assert np.array_equal(parent["prediction_raw"], raw, equal_nan=True)
                        control_checked += 1
                a, b = record["outputs"][law]["aligned"], record["outputs"][law]["identity"]
                assert a["reasons"] == b["reasons"] and a["lags"] == b["lags"]
                assert np.array_equal(a["prediction_raw"], b["prediction_raw"], equal_nan=True)
            records.append(record)
        sequences.append(
            {
                "group": group,
                "component": component,
                "seq_ix": seq_id,
                "origins": len(origins),
                "steps": [r["step"] for r in origins],
                "selected_array_sha256": data_hash,
                "prefix_checks_pass": prefix_ok,
                "decoder_counts": decoded["decoder_counts"],
            }
        )
        print(
            f"group={group} component={component} seq={seq_id} origins={len(origins)}", flush=True
        )
    assert len(records) == 1393
    assert sum(r["regime"] == "outside" for r in records) == 445
    scores = {
        law: {
            condition: {
                regime: group_summary(records, law, condition, regime)
                for regime in ["all", "outside", "interior"]
            }
            for condition in ["aligned", "identity", "shift997", "shuffle"]
        }
        for law in LAWS
    }
    paired = {
        law: {
            c: {
                regime: common_summary(records, law, c, regime)
                for regime in ["all", "outside", "interior"]
            }
            for c in ["extrema_only", "shift997", "shuffle"]
        }
        for law in LAWS[:2]
    }
    verify_parent()
    cost = {
        "wall_seconds": time.monotonic() - started,
        "cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime
        + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_seconds": 0,
    }
    result = {
        "status": "two frozen conditional coordinate laws, no generator identity",
        "scores": scores,
        "paired_controls": paired,
        "test_sequences": sequences,
        "all_origins": len(records),
        "outside_origins": 445,
        "interior_origins": 948,
        "extrema_control_exact_parent_checks": control_checked,
        "complete_pairs_verified_no_unknown_skip": no_unknown_skips,
        "identity_pass": True,
        "parent_hashes_before_after": FROZEN_HASHES,
        "prereg_sha256": digest(HERE / "PREREG.json"),
        "source_hashes": {p: digest(HERE / p) for p in ["laws.py", "synthetic.py", "probe.py"]},
        "cost": cost,
    }
    (HERE / "sampled_outputs.json").write_text(json.dumps(json_safe(records), allow_nan=False))
    (HERE / "results.json").write_text(json.dumps(json_safe(result), indent=2, allow_nan=False))
    state["phase"] = "fixed_scoring_complete"
    state["test_IDs"] = [{k: s[k] for k in ["group", "component", "seq_ix"]} for s in sequences]
    state["pending"] = [
        "summarize conditional matches/mismatches",
        "lint and final report/hash/cost",
    ]
    (HERE / "STATE.json").write_text(json.dumps(state, indent=2))
    print(json.dumps({"aligned": {law: scores[law]["aligned"] for law in LAWS}, "cost": cost}))


if __name__ == "__main__":
    main()
