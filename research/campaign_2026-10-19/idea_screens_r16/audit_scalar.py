"""One-shot metadata/scalar audit for a captured R16 run; never opens arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SAMPLE = ROOT / "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_SAMPLE.json"
MEMBERSHIP = ROOT / "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_MEMBERSHIP.json"
MANIFEST = HERE / "PREPARATION_MANIFEST.json"
EXPECTED_SAMPLE = "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
EXPECTED_MEMBERSHIP = "e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217"
EXPECTED_SPLIT = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
WIDTH = {"i0": 52, "i1": 52, "tail": 8}
FLOOR = 1e-12


def raw(path: Path) -> bytes:
    return path.read_bytes()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def obj(path: Path) -> tuple[dict[str, Any], bytes]:
    data = raw(path)
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value, data


def close(a: float, b: float) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-10 * max(1.0, abs(a), abs(b))


def same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return close(float(a), float(b))


def verify_sample_order(sample: dict[str, Any]) -> tuple[list[dict[str, Any]], list[int]]:
    records = sample["records"]
    fit = records[:64]
    evaluation = records[64:]
    assert len(records) == 96 and len(fit) == 64 and len(evaluation) == 32
    assert all(r["role"] == "fit" and r["source"] == "train" for r in fit)
    assert all(r["role"] == "eval" and r["source"] == "valid" for r in evaluation)
    fit_order = [int(r["component"]) for r in fit]
    eval_order = [int(r["component"]) for r in evaluation]
    assert len(set(fit_order)) == 64 and len(set(eval_order)) == 32
    assert set(fit_order).isdisjoint(eval_order)
    assert sample["selection"]["fit_full64_component_order"] == fit_order
    return fit, fit_order


def verify_membership(sample: dict[str, Any], membership: dict[str, Any]) -> None:
    assert sample["schema"] == "idea_screens_r3.sample.v2"
    assert membership["schema"] == "idea_screens_r3.membership.v2"
    assert sample["split_sha256"] == membership["split_sha256"] == EXPECTED_SPLIT
    assert sample["selection"] == membership["selection"]
    assert sample["metadata_counts"] == membership["metadata_counts"]
    identity_fields = (
        "role", "source", "group", "row_group", "seq_ix", "component",
        "included_in_low32", "fit_low32_order", "eligible_groups_in_component",
        "selection_probabilities",
    )
    assert len(sample["records"]) == len(membership["records"]) == 96
    for left, right in zip(sample["records"], membership["records"], strict=True):
        assert all(left.get(key) == right.get(key) for key in identity_fields)


def check_input_receipts(source: dict[str, Any], fit: list[dict[str, Any]]) -> int:
    expected: list[dict[str, Any]] = []
    for record in fit:
        common = {key: record[key] for key in ("role", "source", "group", "component", "seq_ix")}
        for key in ("x", "scored"):
            binding = record["files"][key]
            expected.append({**common, "key": key, "name": binding["name"],
                             "sha256": binding["sha256"], "bytes": binding["bytes"]})
    actual = source["input_file_receipts"]
    assert actual == expected and len(actual) == 128
    assert source["fit_input_files_hashed_and_mapped"] == 128
    assert source["target_values_accessed"] is False
    assert source["y_payloads_opened"] == 0
    assert source["evaluation_array_payloads_opened"] == 0
    assert source["evaluation_components_metadata_only"] == 32
    return len(actual)


def audit_edge(report: dict[str, Any], held: list[int]) -> dict[str, Any]:
    components = report["components"]
    assert report["component_count"] == 32 and len(components) == 32
    assert [int(c["component"]) for c in components] == held
    assert report["held_rows"] == 15_904
    width = WIDTH[report["destination"]]
    assert report["destination_channels"] == width
    assert all(c["rows"] == 497 and c["destination_channels"] == width for c in components)
    assert sum(int(c["rows"]) for c in components) == report["held_rows"]

    base = math.fsum(float(c["baseline_sse"]) for c in components)
    corrected = math.fsum(float(c["corrected_sse"]) for c in components)
    energy = math.fsum(float(c["standardized_delta_energy"]) for c in components)
    assert close(base, float(report["baseline_sse"]))
    assert close(corrected, float(report["corrected_sse"]))
    assert close(energy, float(report["standardized_delta_energy"]))
    gain = (base - corrected) / base if base > FLOOR else None
    assert same(gain, report["relative_sse_reduction"])

    gains: list[float] = []
    undefined = False
    for component in components:
        b, c = float(component["baseline_sse"]), float(component["corrected_sse"])
        g = (b - c) / b if b > FLOOR else None
        assert same(g, component["relative_sse_reduction"])
        if g is None:
            undefined = True
        else:
            gains.append(g)
    med = None if undefined else median(gains)
    assert same(med, report["component_median_relative_sse_reduction"])
    assert report["component_median_defined"] is (med is not None)
    return {"source": report["source"], "destination": report["destination"],
            "direction": report["direction"], "rows": report["held_rows"],
            "baseline_sse": base, "corrected_sse": corrected,
            "standardized_delta_energy": energy, "relative_sse_reduction": gain,
            "component_median_relative_sse_reduction": med,
            "undefined_component_denominators": 32 - len(gains)}


def audit(capture: Path, validation_path: Path | None, stop_path: Path | None) -> dict[str, Any]:
    start, start_bytes = obj(capture / "REMOTE_START.json")
    source, source_bytes = obj(capture / "REMOTE_SOURCE.json")
    result, result_bytes = obj(capture / "REMOTE_RESULT.json")
    done, done_bytes = obj(capture / "REMOTE_DONE.json")
    manifest, manifest_bytes = obj(MANIFEST)
    sample, sample_bytes = obj(SAMPLE)
    membership, membership_bytes = obj(MEMBERSHIP)
    assert start["schema"] == "nf19.r16.start.v1"
    assert source["schema"] == "nf19.r16.source.v1"
    assert result["schema"] == "nf19.r16.result.v1"
    assert done["schema"] == "nf19.r16.done.v1" and done["status"] == "complete"
    assert sha(sample_bytes) == EXPECTED_SAMPLE and sha(membership_bytes) == EXPECTED_MEMBERSHIP
    assert sample["split_sha256"] == membership["split_sha256"] == EXPECTED_SPLIT
    verify_membership(sample, membership)
    fit, fit_order = verify_sample_order(sample)

    manifest_hash = sha(manifest_bytes)
    source_hash = sha(source_bytes)
    result_hash = sha(result_bytes)
    assert start["sample_sha256_expected"] == EXPECTED_SAMPLE
    assert start["preparation_manifest_sha256"] == manifest_hash
    assert source["sample_sha256"] == EXPECTED_SAMPLE
    assert source["membership_sha256"] == EXPECTED_MEMBERSHIP
    assert source["split_sha256"] == EXPECTED_SPLIT
    assert result["source_sha256"] == source_hash
    assert done["result_sha256"] == result_hash
    assert done["result_bytes"] == len(result_bytes)
    assert done["input_file_count_hashed"] == 128
    source_pins = source["source_pins"]
    assert source_pins == manifest["files"]
    for relative, expected_hash in source_pins.items():
        assert sha(raw(ROOT / relative)) == expected_hash, f"local source pin differs: {relative}"
    assert set(source["runtime_imports"]) == set(manifest["runtime_py"])
    assert source["runtime"]["torch_imported_observed"] is False
    assert source["runtime"]["pyarrow_imported_observed"] is False
    assert check_input_receipts(source, fit) == 128

    analysis = result["analysis"]
    assert analysis["fit_components"] == 64
    assert analysis["fit_rows_per_direction"] == analysis["held_rows_per_direction"] == 15_904
    assert analysis["ordered_edges"] == 6 and analysis["directional_evaluations"] == 12
    assert analysis["baseline_ridge_fits"] == 6
    assert analysis["source_projection_fits"] == analysis["correction_ridge_fits"] == 12
    dirs = analysis["direction_diagnostics"]
    left, right = fit_order[:32], fit_order[32:]
    assert dirs["L_to_R"]["fit_components"] == left and dirs["L_to_R"]["held_components"] == right
    assert dirs["R_to_L"]["fit_components"] == right and dirs["R_to_L"]["held_components"] == left
    assert dirs["L_to_R"]["fit_rows"] == dirs["L_to_R"]["held_rows"] == 15_904
    assert dirs["R_to_L"]["fit_rows"] == dirs["R_to_L"]["held_rows"] == 15_904

    expected_dirs = {(s, d, direction) for s in WIDTH for d in WIDTH if s != d
                     for direction in ("L_to_R", "R_to_L")}
    reports = analysis["edge_scores"]
    assert len(reports) == 12
    assert {(r["source"], r["destination"], r["direction"]) for r in reports} == expected_dirs
    checked = [audit_edge(r, right if r["direction"] == "L_to_R" else left) for r in reports]
    by_key = {(r["source"], r["destination"], r["direction"]): r for r in reports}
    expected_pairs = {(s, d) for s in WIDTH for d in WIDTH if s != d}
    pairs = analysis["paired_edges"]
    assert len(pairs) == 6 and {(p["source"], p["destination"]) for p in pairs} == expected_pairs
    stable_count = 0
    for pair in pairs:
        a = by_key[(pair["source"], pair["destination"], "L_to_R")]
        b = by_key[(pair["source"], pair["destination"], "R_to_L")]
        assert same(pair["L_to_R_relative_sse_reduction"], a["relative_sse_reduction"])
        assert same(pair["R_to_L_relative_sse_reduction"], b["relative_sse_reduction"])
        assert same(
            pair["L_to_R_component_median_relative_sse_reduction"],
            a["component_median_relative_sse_reduction"],
        )
        assert same(
            pair["R_to_L_component_median_relative_sse_reduction"],
            b["component_median_relative_sse_reduction"],
        )
        stable = all(
            isinstance(r.get("relative_sse_reduction"), (int, float))
            and isinstance(r.get("component_median_relative_sse_reduction"), (int, float))
            and r["relative_sse_reduction"] >= 0.01
            and r["component_median_relative_sse_reduction"] > 0.0
            for r in (a, b)
        )
        assert pair["stable"] is stable
        stable_count += int(stable)
    assert analysis["stable_edge_count"] == stable_count
    assert analysis["spending_gate_passed"] is (stable_count > 0)

    archive = result["parameter_archive"]
    assert archive == done["parameter_archive"]
    assert archive["contains_predictions"] is False
    # Metadata-only: the archive path/hash/size are recorded but archive bytes are never opened.
    root_evidence: dict[str, Any] = {}
    for label, path in (("capture_validation", validation_path), ("stop_receipt", stop_path)):
        if path is not None:
            value, payload = obj(path)
            root_evidence[label] = {"path": str(path), "sha256": sha(payload), "record": value}
    return {
        "schema": "nf19.r16.scalar_audit.v1",
        "status": "PASS_SCALAR_METADATA_AUDIT",
        "source_manifest_sha256": manifest_hash,
        "sample_sha256": EXPECTED_SAMPLE,
        "membership_sha256": EXPECTED_MEMBERSHIP,
        "start_sha256": sha(start_bytes),
        "source_sha256": source_hash,
        "result_sha256": result_hash,
        "done_sha256": sha(done_bytes),
        "result_bytes": len(result_bytes),
        "directional_edge_audits": checked,
        "stable_edge_count": stable_count,
        "spending_gate_passed": stable_count > 0,
        "fit_component_order": fit_order,
        "fit_halves": {"L": left, "R": right},
        "input_receipts_checked": 128,
        "fit_rows_per_half": 15_904,
        "held_rows_per_half": 15_904,
        "held_component_count_per_endpoint": 32,
        "directional_endpoints": 12,
        "edge_procedure_counts": {
            "baseline_ridge_fits": analysis["baseline_ridge_fits"],
            "source_projection_fits": analysis["source_projection_fits"],
            "correction_ridge_fits": analysis["correction_ridge_fits"],
        },
        "source_pins_match_manifest": len(source_pins),
        "runtime_imports_match_manifest": len(manifest["runtime_py"]),
        "archive_metadata_only": archive,
        "root_evidence": root_evidence,
        "limits": [
            "Scalar/component sufficient statistics and JSON receipts only; no NPY, NPZ, model, or target payload was opened.",
            "The 64 components form two fixed 32-component halves; six edges and two directions are not independent replications.",
            "This is an input-innovation screen on the exposed R3 fit panel, not a Y-quality result or causal graph.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--stop", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "RESULT_AUDIT.json")
    args = parser.parse_args()
    value = audit(args.capture, args.validation, args.stop)
    payload = json.dumps(value, indent=2, allow_nan=False).encode() + b"\n"
    with args.output.open("xb") as stream:
        stream.write(payload)
        stream.flush()


if __name__ == "__main__":
    main()
