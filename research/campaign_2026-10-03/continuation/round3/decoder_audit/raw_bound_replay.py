from __future__ import annotations

import hashlib
import json
import logging
import resource
import time
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from audit_common import (
    ENDPOINT,
    FROZEN,
    OWN,
    PRICE_COLUMNS,
    QUANTA,
    ROOT,
    TOLERANCE,
    digest,
    phase,
    price_code,
    save_json,
)

LOGGER = logging.getLogger(__name__)


def audit_inputs() -> dict[str, object]:
    """Check only authorized frozen files against the supplied provenance."""
    provenance = json.loads((FROZEN / "PROVENANCE.json").read_text())
    names = [
        "REPORT.md",
        "PREREG.json",
        "BOUND_PREREG.json",
        "selection.json",
        "decoder_tables.json",
        "sampled_origins.json",
        "sampled_origins_strict.json",
        "results.json",
        "fit_results.json",
        "loser_bound_results.json",
        "decoder.py",
        "evaluate.py",
        "loser_bound.py",
    ]
    verification = {
        name: {
            "sha256": digest(FROZEN / name),
            "recorded_sha256": provenance[name]["sha256"],
            "match": digest(FROZEN / name) == provenance[name]["sha256"],
        }
        for name in names
    }
    tables = json.loads((FROZEN / "decoder_tables.json").read_text())
    return {
        "frozen_artifact_checks": verification,
        "all_declared_artifact_hashes_match": all(
            row["match"] for row in verification.values()
        ),
        "fit_time_decoder_source_hash": tables["source_hashes"]["decoder.py"],
        "current_decoder_source_hash": digest(FROZEN / "decoder.py"),
        "fit_time_source_hash_matches_current": tables["source_hashes"]["decoder.py"]
        == digest(FROZEN / "decoder.py"),
        "own_prereg_sha256": digest(OWN / "PREREG.json"),
        "source_implementation_read_after_commitment": True,
        "implementation_reused": False,
    }


def origin_record(
    supplied: dict[str, object],
    prices: np.ndarray,
    targets: np.ndarray,
    events: np.ndarray,
    lookup: dict[str, tuple[int, int]],
    event_index: dict[int, int],
) -> dict[str, object]:
    """Extract raw targets and apply the registered inequality to each leg."""
    step = int(supplied["step"])
    raw = targets[step]
    code = price_code(prices[step])
    mapping = lookup.get(code)
    cls, spread = mapping if mapping is not None else (-1, -1)
    q = QUANTA[cls] if cls >= 0 else None
    bound = -spread * q if q is not None else None
    exterior = bool(np.any(raw > 0))
    index = event_index[step]
    next_event = int(events[index + 1]) if index + 1 < len(events) else len(targets)
    represented = next_event - step
    weight = float(np.abs(np.clip(raw, -2, 2)).sum() * represented)
    violations = raw < bound - TOLERANCE if bound is not None else None
    clipped = np.clip(raw, -2, 2)
    clipped_violations = clipped < bound - TOLERANCE if bound is not None else None
    stored_q = supplied["quantum_from_fit_code"]
    record = {
        "group": int(supplied["group"]),
        "component": int(supplied["component"]),
        "seq_ix": int(supplied["seq_ix"]),
        "step": step,
        "event_index": index,
        "code_hex": code,
        "class": cls,
        "quantum": q,
        "spread_ticks": spread if mapping else None,
        "bound_raw": bound,
        "t0_raw": float(raw[0]),
        "t1_raw": float(raw[1]),
        "outside": exterior,
        "both_positive": bool(np.all(raw > 0)),
        "mapped": mapping is not None,
        "raw_violation": bool(violations.any()) if violations is not None else None,
        "t0_violates": bool(violations[0]) if violations is not None else None,
        "t1_violates": bool(violations[1]) if violations is not None else None,
        "clipped_violation": bool(clipped_violations.any())
        if clipped_violations is not None
        else None,
        "bound_margin_raw": float(raw.min() - bound) if bound is not None else None,
        "lattice_residual_raw": float(np.max(np.abs(raw - np.rint(raw / q) * q)))
        if q
        else None,
        "interior_sum_matches": bool(abs(raw.sum() + spread * q) <= TOLERANCE)
        if not exterior and q is not None
        else None,
        "recorded_target_matches": bool(np.array_equal(raw, supplied["target_raw"])),
        "recorded_class_matches": cls == supplied["fit_code_class"],
        "recorded_quantum_matches": q == stored_q,
        "recorded_event_index_matches": index == supplied["event_index"],
        "recorded_plateau_matches": represented == supplied["represented_plateau_rows"],
        "recorded_weight_matches": weight == supplied["diagnostic_weight"],
        "represented_plateau_rows": represented,
        "diagnostic_weight": weight,
        "labels_constant_until_next_endpoint": bool(
            np.all(targets[step:next_event] == raw)
        ),
    }
    if q:
        record["recorded_lattice_residual_matches"] = (
            abs(record["lattice_residual_raw"] - supplied["label_lattice_residual"])
            < 1e-12
        )
    for law, outputs in supplied["laws"].items():
        reasons = outputs["aligned"]["reasons"]
        record[f"{law}_reasons"] = "|".join(reasons)
        record[f"{law}_complete"] = all(reason == "resolved" for reason in reasons)
    return record


def counts(frame: pd.DataFrame) -> dict[str, int | float]:
    """Keep all origin denominators and missing maps visible in every stratum."""
    outside = frame.outside.astype(bool)
    mapped = frame.mapped.astype(bool)
    violated = frame.raw_violation.fillna(False).astype(bool)
    clipped = frame.clipped_violation.fillna(False).astype(bool)
    return {
        "all": len(frame),
        "interior": int((~outside).sum()),
        "outside": int(outside.sum()),
        "both_positive": int(frame.both_positive.sum()),
        "mapped": int(mapped.sum()),
        "missing_map": int((~mapped).sum()),
        "mapped_interior": int((mapped & ~outside).sum()),
        "mapped_outside": int((mapped & outside).sum()),
        "missing_outside": int((~mapped & outside).sum()),
        "raw_violations_all": int(violated.sum()),
        "raw_violations_outside": int((violated & outside).sum()),
        "clipped_violations_outside": int((clipped & outside).sum()),
        "t0_violations_all": int(frame.t0_violates.fillna(False).sum()),
        "t1_violations_all": int(frame.t1_violates.fillna(False).sum()),
        "interior_sum_exact": int(frame.interior_sum_matches.fillna(False).sum()),
        "all_weight": float(frame.diagnostic_weight.sum()),
        "outside_weight": float(frame.loc[outside, "diagnostic_weight"].sum()),
        "violation_outside_weight": float(
            frame.loc[violated & outside, "diagnostic_weight"].sum()
        ),
    }


def main() -> None:
    """Independently replay all frozen origins from raw TRAIN without fitting."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start = time.monotonic()
    phase("raw_replay_running")
    inputs = audit_inputs()
    tables = json.loads((FROZEN / "decoder_tables.json").read_text())
    lookup = {key: (int(cls), int(spread)) for key, cls, spread in tables["spreads"]}
    if len(lookup) != len(tables["spreads"]):
        raise ValueError("Duplicate frozen spread keys")
    samples = json.loads((FROZEN / "sampled_origins_strict.json").read_text())
    original = json.loads((FROZEN / "sampled_origins.json").read_text())
    if len(samples) != 1393 or len(original) != 1393:
        raise ValueError("Frozen origin denominator changed")
    inputs["strict_and_original_origin_targets_identical"] = all(
        a["target_raw"] == b["target_raw"]
        and a["step"] == b["step"]
        and a["group"] == b["group"]
        for a, b in zip(samples, original, strict=True)
    )
    selection = json.loads((FROZEN / "selection.json").read_text())
    results = json.loads((FROZEN / "results.json").read_text())
    recorded = {row["group"]: row for row in results["test_samples"]}
    groups = defaultdict(list)
    for row in samples:
        groups[row["group"]].append(row)
    if set(groups) != set(selection["test_groups"]):
        raise ValueError("Origin groups disagree with the frozen selection")
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    records, group_checks = [], []
    for group, component in zip(
        selection["test_groups"], selection["test_components"], strict=True
    ):
        table = source.read_row_group(
            group,
            columns=PRICE_COLUMNS + ["a6", "t0", "t1", "seq_ix", "step_in_seq"],
            use_threads=False,
        )
        prices = np.column_stack(
            [table[name].to_numpy() for name in PRICE_COLUMNS]
        ).astype("<f4")
        targets = np.column_stack(
            [table[name].to_numpy() for name in ["t0", "t1"]]
        ).astype("<f8")
        a6 = table["a6"].to_numpy().astype("<f4")
        ids = table["seq_ix"].to_numpy()
        steps = table["step_in_seq"].to_numpy()
        events = np.flatnonzero(a6 == ENDPOINT)
        selected = [index for index, step in enumerate(events) if step >= 99][::5]
        if len(selected) > 128:
            selected = [
                selected[index]
                for index in np.linspace(0, len(selected) - 1, 128).astype(int)
            ]
        selected_steps = [int(events[index]) for index in selected]
        array_hash = hashlib.sha256(
            prices.tobytes() + a6.tobytes() + targets.tobytes()
        ).hexdigest()
        saved = recorded[group]
        checks = {
            "group": group,
            "component": component,
            "seq_ix": int(ids[0]),
            "all_seq_ids_equal": bool(np.all(ids == saved["seq_ix"])),
            "raw_step_order_ok": bool(np.array_equal(steps, np.arange(len(steps)))),
            "metadata_component_ok": all(
                row["component"] == component for row in groups[group]
            ),
            "metadata_origin_seq_ids_ok": all(
                row["seq_ix"] == int(ids[0]) for row in groups[group]
            ),
            "array_sha256": array_hash,
            "recorded_array_hash_matches": array_hash == saved["array_sha256"],
            "sample_selection_replays": selected_steps
            == [row["step"] for row in groups[group]]
            == saved["sampled_steps"],
            "endpoint_count_matches": len(events) == saved["all_endpoint_events"],
        }
        event_index = {int(row): index for index, row in enumerate(events)}
        for supplied in groups[group]:
            records.append(
                origin_record(supplied, prices, targets, events, lookup, event_index)
            )
        group_checks.append(checks)
        LOGGER.info(
            "group=%s component=%s origins=%s raw_hash_ok=%s",
            group,
            component,
            len(groups[group]),
            checks["recorded_array_hash_matches"],
        )
    frame = pd.DataFrame(records)
    frame.to_csv(OWN / "all_origins_raw_replay.csv", index=False)
    save_json("all_origins_raw_replay.json", records)
    summary = {
        "counts": counts(frame),
        "group_checks": group_checks,
        "by_component": {
            str(key): counts(part) for key, part in frame.groupby("component")
        },
        "by_class": {str(key): counts(part) for key, part in frame.groupby("class")},
        "component_class": {
            f"{comp}/{cls}": counts(part)
            for (comp, cls), part in frame.groupby(["component", "class"])
        },
        "recorded_field_mismatches": {
            name: int((~frame[name].dropna().astype(bool)).sum())
            for name in frame
            if name.startswith("recorded_")
        },
        "censoring_from_frozen_metadata": {
            law: {
                str(reasons): counts(part)
                for reasons, part in frame.groupby(f"{law}_reasons")
            }
            for law in ["fixed_extrema", "competing_barriers", "trailing_extrema"]
        },
        "maximum_mapped_lattice_residual_raw": float(frame.lattice_residual_raw.max()),
        "nonconstant_represented_label_spans": int(
            (~frame.labels_constant_until_next_endpoint).sum()
        ),
        "raw_minimum_outside_bound_margin": float(
            frame.loc[frame.outside & frame.mapped, "bound_margin_raw"].min()
        ),
        "raw_maximum_outside_bound_margin": float(
            frame.loc[frame.outside & frame.mapped, "bound_margin_raw"].max()
        ),
        "cost": {
            "wall_seconds": time.monotonic() - start,
            "cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime
            + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    save_json("raw_replay_results.json", summary)
    save_json("input_provenance_checks.json", inputs)
    phase(
        "raw_replay_complete",
        inspected_group_ids=selection["test_groups"],
        inspected_seq_ids=[row["seq_ix"] for row in group_checks],
        cost_raw=summary["cost"],
        last_failure=None,
    )
    LOGGER.info("COUNTS %s", json.dumps(summary["counts"]))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        failure = traceback.format_exc()
        (OWN / "raw_replay_failure.log").write_text(failure)
        phase("raw_replay_failed", last_failure=failure)
        raise
