"""Independently audit R9's compact JSON receipts without reading arrays or weights."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CAPTURE = Path(__file__).resolve().parent / "terminal_capture"
BRANCHES = ("recent256", "zero")
KINDS = ("raw", "clipped")
TARGETS = ("target_0", "target_1", "pooled")
ADDITIVE_FIELDS = (
    "count",
    "delta_square_sum",
    "reference_square_sum",
    "absolute_difference_sum",
)
METRIC_FIELDS = (
    "reference_rms",
    "rms_difference",
    "relative_rms_difference",
    "mean_absolute_difference",
    "max_absolute_difference",
)


def _bytes_sha(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _read(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = CAPTURE / name
    size, digest = _bytes_sha(path)
    return json.loads(path.read_bytes()), {"bytes": size, "sha256": digest}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-12)


def _parse_events(value: str) -> dict[str, int]:
    return {parts[0]: int(parts[1]) for line in value.splitlines() if (parts := line.split())}


def _summary_from_sums(values: dict[str, float | int]) -> dict[str, float | int]:
    count = int(values["count"])
    delta_square_sum = float(values["delta_square_sum"])
    reference_square_sum = float(values["reference_square_sum"])
    absolute_difference_sum = float(values["absolute_difference_sum"])
    return {
        **values,
        "reference_rms": math.sqrt(reference_square_sum / count),
        "rms_difference": math.sqrt(delta_square_sum / count),
        "relative_rms_difference": math.sqrt(
            delta_square_sum / max(reference_square_sum, 1e-12)
        ),
        "mean_absolute_difference": absolute_difference_sum / count,
    }


def _audit() -> dict[str, Any]:
    result, result_file = _read("REMOTE_RESULT.json")
    source, source_file = _read("REMOTE_SOURCE.json")
    done, done_file = _read("REMOTE_DONE.json")
    start, start_file = _read("REMOTE_START.json")
    root_admission, admission_file = _read("REMOTE_ROOT_ADMISSION.json")
    root_launch, launch_file = _read("REMOTE_ROOT_LAUNCH.json")
    root_smoke, smoke_file = _read("REMOTE_ROOT_IMPORT_SMOKE.json")
    capture_summary, capture_file = _read("CAPTURE_SUMMARY.json")
    validation, validation_file = _read("VALIDATION.json")
    stop, stop_file = _read("STOP_RECEIPT.json")

    manifest_path = ROOT / "PREPARATION_MANIFEST.json"
    manifest_bytes, manifest_sha = _bytes_sha(manifest_path)
    manifest = json.loads(manifest_path.read_bytes())
    pins = manifest["files"]
    _require(manifest_sha == start["preparation_manifest_sha256"], "manifest SHA differs")
    _require(source["preparation_manifest_sha256"] == manifest_sha, "source manifest SHA differs")
    _require(result["schema"] == "nf19.r9_history_probe.result.v1", "unexpected result schema")
    _require(done["status"] == result["status"] == "completed_no_update_carried_state_diagnostic", "result incomplete")
    _require(done["result_sha256"] == result_file["sha256"], "DONE result hash mismatch")
    _require(done["result_bytes"] == result_file["bytes"], "DONE result byte count mismatch")
    _require(source["source_file_pins"] == pins, "runtime source pins differ from preparation manifest")
    _require(len(pins) == 43, "unexpected source pin count")

    source_sample = source["sample"]
    sample_path = ROOT.parent / "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_SAMPLE.json"
    sample_size, sample_sha = _bytes_sha(sample_path)
    sample = json.loads(sample_path.read_bytes())
    expected_components = sample["selection"]["fit_full64_component_order"][:8]
    expected_sequences = [
        {
            key: record[key]
            for key in ("source", "group", "row_group", "seq_ix", "component")
        }
        for record in sample["records"][:8]
    ]
    _require(sample_sha == source_sample["sample_sha256"] == start["sample_sha256"], "R3 sample identity mismatch")
    _require(source_sample["membership_sha256"] == start["membership_sha256"], "R3 membership identity mismatch")
    _require(source_sample["selected_components"] == expected_components, "first-eight component selection differs")
    _require(source_sample["selected_sequences"] == expected_sequences, "selected sequence metadata differs from producer fixture")
    _require(source_sample["selected_npy_files_hashed_and_mapped"] == 24, "expected selected NPY integrity count differs")
    _require(source["target_file_count_hashed_and_mapped"] == 8, "target-file integrity count differs")
    _require(source["target_files_hashed_and_mapped_for_integrity"] is True, "target integrity statement missing")
    _require(source["target_values_accessed"] is False and result["target_values_accessed"] is False, "target values were reported accessed")
    _require(source["evaluation_arrays_read"] == 0 and source_sample["eval_npy_files_read"] == 0, "evaluation arrays were reported read")

    expected_source_sha = {
        key: value for key, value in pins.items() if not key.startswith("canonical:")
    }
    expected_canonical_sha = {
        key.removeprefix("canonical:wnn/"): value
        for key, value in pins.items()
        if key.startswith("canonical:wnn/")
    }
    _require(len(expected_canonical_sha) == 25, "unexpected canonical pin count")
    _require(root_launch["source_sha256"] == root_admission["source_sha256"], "admission/launch payload hashes differ")
    _require(root_launch["canonical_source_sha256"] == root_admission["canonical_source_sha256"], "admission/launch canonical hashes differ")
    for key, value in expected_source_sha.items():
        _require(root_launch["source_sha256"].get(key) == value, f"staged payload pin differs: {key}")
    for key, value in expected_canonical_sha.items():
        _require(root_launch["canonical_source_sha256"].get(key) == value, f"canonical pin differs: {key}")
    _require(source["source_file_pins"] == pins, "source receipt pins mismatch")
    _require(source["runtime"]["source_pin_count"] == len(pins), "runtime pin count differs")
    _require(len(source["runtime"]["canonical_inventory_after_r9_imports"]) == 25, "post-import canonical inventory incomplete")
    _require(len(source["runtime"]["runtime_import_paths"]) == 12, "runtime import inventory incomplete")

    smoke = root_smoke["runtime_import_inventory"]
    _require(root_smoke["status"] == "PASS" and root_smoke["data_reads"] == 0, "root read-sentinel smoke failed")
    _require(root_smoke["cuda_initialized"] is False, "CUDA was initialized during smoke")
    _require(smoke["runtime"]["reader_calls"] == 0, "runtime smoke called a reader")
    _require(len(smoke["runtime_import_paths"]) == 12, "root smoke runtime import set incomplete")
    _require(len(smoke["runtime"]["wnn_source_inventory"]) == 25, "root smoke canonical inventory incomplete")
    _require(root_smoke["torch"] == "2.14.0+cu130" and root_smoke["numpy"] == "2.2.6", "B smoke runtime version mismatch")

    invocation = capture_summary["expected_identity"]["invocation_id"]
    unit = "computer-vision-homework-eda19-r9-v1.service"
    _require(root_launch["unit"] == root_admission["unit"] == stop["unit"] == unit, "unit identity mismatch")
    _require(invocation == stop["invocation"] == capture_summary["actual_unit_properties"]["InvocationID"], "unit invocation mismatch")
    _require(root_launch["returncode"] == 0 and stop["ssh_returncode"] == 0 and stop["stop_returncode"] == 0, "launch or stop command failed")
    _require(invocation in root_launch["stderr"], "launch invocation is not bound in stderr")
    _require(capture_summary["caps_match"] is True and capture_summary["terminal_exit_success"] is True, "capture did not validate configured caps or terminal status")
    _require(validation["status"] == "PASS_CAPTURE_VALIDATION_ONLY", "capture validation status differs")
    _require(validation["invocation_id"] == invocation and validation["failure_json_present"] is False, "capture validation identity/failure mismatch")
    props = capture_summary["actual_unit_properties"]
    _require(props["CPUQuotaPerSecUSec"] == "500ms", "actual CPU quota differs")
    _require(props["MemoryHigh"] == "1400000000" and props["MemoryMax"] == "1500000000", "actual memory caps differ")
    _require(props["MemorySwapMax"] == "0" and props["RuntimeMaxUSec"] == "10min", "actual swap/runtime caps differ")
    _require(props["ExecMainStatus"] == "0" and props["SubState"] == "exited", "terminal unit did not exit successfully")
    _require(capture_summary["own_cgroup"]["exists"] is False, "unexpected own-cgroup status")
    _require(capture_summary["own_cgroup"]["control_group_property"] == "", "own cgroup was unexpectedly available")
    _require(props["MemoryPeak"] == "[not set]" and props["MemoryCurrent"] == "[not set]", "unexpected per-unit memory sample")
    _require(stop["status"] == "STOPPED_EXACT_OWNED_UNIT" and stop["stop_called"] is True, "exact unit stop was not validated")
    _require(stop["post_state"]["LoadState"] == "not-found" and stop["post_state"]["SubState"] == "dead", "unit remains present after stop")
    _require(stop["post_state"]["ExecMainStatus"] == "0" and stop["post_state"]["Result"] == "success", "post-stop unit status differs")
    _require(all(item["pass"] for item in stop["precheck_validation"].values()), "stop precheck did not bind exact unit properties")

    cap = validation["cgroups"]
    _require(cap["populated_campaign_descendants"] == 0 and cap["terminal_control_group_empty_no_fallback"] is True, "unexpected populated descendants or cgroup fallback")
    _require(all(value == 0 for value in cap["parent_memory_events_delta"].values()), "parent memory event counters changed")
    _require(all(value == 0 for value in cap["parent_memory_events_local_delta"].values()), "parent local memory event counters changed")
    admission_events = _parse_events(root_admission["admission"]["memory_events"])
    initial_events = _parse_events(root_admission["admission"]["memory_events_initial"])
    for events in (admission_events, initial_events):
        _require(events.get("oom") == 0 and events.get("oom_kill") == 0, "OOM event recorded at launch gate")

    counts = result["traversal_counts"]
    expected_counts = {
        "full_prefix_rows": 130048,
        "recent_reset_rows": 4096,
        "rollout_rows": 6144,
        "model_forward_rows_total": 140288,
        "expected_model_forward_rows_total": 140288,
        "prepared_input_rows": 132096,
        "model_forward_calls": 464,
    }
    for key, expected in expected_counts.items():
        _require(counts.get(key) == expected, f"unexpected traversal count for {key}")
    _require(counts["model_forward_rows_match_design"] is True, "traversal total/design flag differs")
    _require(done["traversal_counts"] == counts, "DONE traversal counts differ")
    _require(result["window_count"] == 16 and result["anchors"] == [8192, 16384], "window count or anchor set differs")
    _require(result["snapshot_k"] == [0, 1, 16, 64, 128], "snapshot positions differ")
    _require(result["model_state_unchanged"] is True and done["model_state_unchanged"] is True, "model state changed")
    _require(result["model_state_fingerprint_before"] == result["model_state_fingerprint_after"], "model fingerprints differ")
    _require(result["parameter_grad_fields_empty"] is True, "parameter gradients were populated")
    _require(result["backward_or_optimizer_used"] is False and result["scorer_called"] is False, "an update or scorer was used")
    _require(result["input_width"] == 211 and result["raw_input_width"] == 112 and result["causal_event_width"] == 99, "input dimensions differ")
    _require(result["normalization"] == "saved model mean/std only; unchanged from checkpoint", "normalization description differs")

    expected_windows = {
        (sequence_index, anchor)
        for sequence_index in range(8)
        for anchor in (8192, 16384)
    }
    seen_windows: set[tuple[int, int]] = set()
    pooled: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {
        branch: {
            kind: {
                target: {
                    "count": 0,
                    "delta_square_sum": 0.0,
                    "reference_square_sum": 0.0,
                    "absolute_difference_sum": 0.0,
                    "max_absolute_difference": 0.0,
                }
                for target in TARGETS
            }
            for kind in KINDS
        }
        for branch in BRANCHES
    }
    per_window_relative: dict[str, list[float]] = {branch: [] for branch in BRANCHES}
    state_k128: dict[str, list[float]] = {branch: [] for branch in BRANCHES}
    for window in result["windows"]:
        sequence_index = int(window["sequence_index"])
        anchor = int(window["anchor"])
        key = (sequence_index, anchor)
        _require(key not in seen_windows, "duplicate window"); seen_windows.add(key)
        _require(window["component"] == expected_components[sequence_index], "window component differs from fixed sequence")
        selected = expected_sequences[sequence_index]
        for field in ("source", "group", "seq_ix"):
            _require(window[field] == selected[field], f"window {field} differs from sample metadata")
        _require(window["source"] == "train" and window["window_rows"] == 128, "window role/length differs")
        _require(set(window["state_snapshots"]) == {"0", "1", "16", "64", "128"}, "window state snapshots missing")
        endpoints = window["endpoint_predictions"]
        _require([item["k"] for item in endpoints] == [1, 16, 64, 128], "window endpoint order differs")
        for snapshot_k, snapshot in window["state_snapshots"].items():
            _require(set(snapshot) == {"full", "recent256", "zero"}, "snapshot branch set differs")
            for branch, state in snapshot.items():
                for field in ("reference_l2", "alternative_l2", "reference_rms", "difference_l2", "difference_rms", "relative_difference_l2", "max_absolute_difference"):
                    _require(math.isfinite(float(state[field])), "nonfinite state statistic")
                if branch == "full":
                    _require(state["difference_l2"] == 0.0 and state["difference_rms"] == 0.0, "full-state self distance is nonzero")
                    _require(_close(state["reference_l2"], state["alternative_l2"]), "full-state norm self check differs")
                if snapshot_k == "0" and branch == "zero":
                    _require(state["alternative_l2"] == 0.0, "zero branch did not start at zero state")
        _require(
            window["state_snapshots"]["1"]["zero"] != window["state_snapshots"]["128"]["zero"],
            "k=1 and k=128 zero-state snapshots are identical",
        )
        for branch in BRANCHES:
            per_window_relative[branch].append(
                float(window["prediction_differences"][branch]["raw"]["pooled"]["relative_rms_difference"])
            )
            state_k128[branch].append(
                float(window["state_snapshots"]["128"][branch]["relative_difference_l2"])
            )
            for kind in KINDS:
                for target in TARGETS:
                    values = window["prediction_differences"][branch][kind][target]
                    expected_count = 256 if target == "pooled" else 128
                    _require(values["count"] == expected_count, "per-window statistic count differs")
                    for field in (*ADDITIVE_FIELDS[1:], "max_absolute_difference", *METRIC_FIELDS):
                        _require(math.isfinite(float(values[field])), "nonfinite prediction statistic")
                    dest = pooled[branch][kind][target]
                    for field in ADDITIVE_FIELDS:
                        dest[field] += values[field]
                    dest["max_absolute_difference"] = max(
                        float(dest["max_absolute_difference"]), float(values["max_absolute_difference"])
                    )
        for endpoint in endpoints:
            _require(math.isfinite(float(endpoint["full_prediction_l2"])), "nonfinite endpoint prediction norm")
            for branch in BRANCHES:
                for target_data in endpoint[branch]["target_differences"].values():
                    _require(math.isfinite(float(target_data["absolute_difference"])), "nonfinite endpoint delta")
                    _require(math.isfinite(float(target_data["relative_absolute_difference"])), "nonfinite endpoint relative delta")
    _require(seen_windows == expected_windows, "fixed 16-window panel is incomplete")

    reconstructed: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    for branch in BRANCHES:
        reconstructed[branch] = {}
        for kind in KINDS:
            reconstructed[branch][kind] = {}
            for target in TARGETS:
                values = pooled[branch][kind][target]
                values["count"] = int(values["count"])
                summary = _summary_from_sums(values)
                stored = result["pooled_prediction_differences"][branch][kind][target]
                expected_count = 4096 if target == "pooled" else 2048
                _require(summary["count"] == expected_count, "pooled prediction count differs")
                for field in (*ADDITIVE_FIELDS, "max_absolute_difference", *METRIC_FIELDS):
                    _require(_close(float(summary[field]), float(stored[field])), f"pooled {branch}/{kind}/{target} {field} differs")
                reconstructed[branch][kind][target] = summary
    _require(
        reconstructed["recent256"]["raw"] == reconstructed["recent256"]["clipped"]
        and reconstructed["zero"]["raw"] == reconstructed["zero"]["clipped"],
        "raw/clipped reconstructed sufficient statistics differ",
    )

    progress_path = CAPTURE / "REMOTE_PROGRESS.jsonl"
    progress = [json.loads(line) for line in progress_path.read_text().splitlines() if line]
    _require(sum(item.get("event") == "window_completed" for item in progress) == 16, "progress receipt does not show 16 completed windows")
    _require(any(item.get("event") == "initialized" for item in progress), "progress initialization event missing")
    raw_capture_size, raw_capture_sha = _bytes_sha(CAPTURE / "REMOTE_CAPTURE_RAW.json")
    _require(raw_capture_size == validation["raw_envelope"]["bytes"], "raw capture byte count differs")
    _require(raw_capture_sha == validation["raw_envelope"]["sha256"], "raw capture hash differs")
    stop_stdout_size, stop_stdout_sha = _bytes_sha(CAPTURE / "REMOTE_STOP_RAW.json")
    _require(stop_stdout_size == stop["ssh_stdout_raw"]["bytes"], "stop raw stdout byte count differs")
    _require(stop_stdout_sha == stop["ssh_stdout_raw"]["sha256"], "stop raw stdout hash differs")

    ranges = {}
    for branch in BRANCHES:
        values = per_window_relative[branch]
        state_values = state_k128[branch]
        ranges[branch] = {
            "per_window_raw_relative_rms": {
                "min": min(values),
                "median": statistics.median(values),
                "max": max(values),
                "count": len(values),
            },
            "k128_state_relative_l2": {
                "min": min(state_values),
                "median": statistics.median(state_values),
                "max": max(state_values),
                "count": len(state_values),
            },
        }

    audit_script_size, audit_script_sha = _bytes_sha(Path(__file__))
    audit = {
        "schema": "nf19.r9_history_probe.result_audit.v1",
        "status": "PASS",
        "audit_script": {"path": str(Path(__file__).resolve()), "bytes": audit_script_size, "sha256": audit_script_sha},
        "identities": {
            "result": result_file,
            "source": source_file,
            "done": done_file,
            "start": start_file,
            "root_admission": admission_file,
            "root_launch": launch_file,
            "root_import_smoke": smoke_file,
            "capture_summary": capture_file,
            "capture_validation": validation_file,
            "stop_receipt": stop_file,
            "raw_capture_envelope": {"bytes": raw_capture_size, "sha256": raw_capture_sha},
            "stop_stdout": {"bytes": stop_stdout_size, "sha256": stop_stdout_sha},
            "preparation_manifest": {"bytes": manifest_bytes, "sha256": manifest_sha},
            "sample_fixture": {"bytes": sample_size, "sha256": sample_sha},
        },
        "source_and_panel": {
            "source_pin_count": len(pins),
            "canonical_pin_count": len(expected_canonical_sha),
            "runtime_import_count": len(source["runtime"]["runtime_import_paths"]),
            "canonical_runtime_inventory_count": len(source["runtime"]["canonical_inventory_after_r9_imports"]),
            "sample_sha256": sample_sha,
            "membership_sha256": source_sample["membership_sha256"],
            "selected_components": expected_components,
            "selected_sequences": expected_sequences,
            "selected_fit_npy_hashes_and_maps": source_sample["selected_npy_files_hashed_and_mapped"],
            "y_files_hashed_and_mapped_for_integrity": source["target_file_count_hashed_and_mapped"],
            "target_values_accessed": False,
            "evaluation_arrays_read": 0,
            "checkpoint_sha256": source["checkpoint_sha256"],
            "checkpoint_bytes": source["checkpoint_bytes"],
            "model_input_width": result["input_width"],
            "raw_width": result["raw_input_width"],
            "causal_event_width": result["causal_event_width"],
            "saved_normalization_unchanged": True,
        },
        "execution_and_state": {
            "window_count": len(result["windows"]),
            "anchors": result["anchors"],
            "snapshot_k": result["snapshot_k"],
            "traversal_counts": counts,
            "model_fingerprint_unchanged": result["model_state_unchanged"],
            "parameter_grad_fields_empty": result["parameter_grad_fields_empty"],
            "backward_or_optimizer_used": result["backward_or_optimizer_used"],
            "scorer_called": result["scorer_called"],
            "target_values_accessed": result["target_values_accessed"],
            "per_window_state_and_endpoint_checks": "PASS",
        },
        "reconstructed_pooled_prediction_statistics": reconstructed,
        "per_window_ranges": ranges,
        "runtime_and_stop": {
            "unit": stop["unit"],
            "invocation": stop["invocation"],
            "unit_exit_status": props["ExecMainStatus"],
            "unit_caps_match": capture_summary["caps_match"],
            "cpu_quota": props["CPUQuotaPerSecUSec"],
            "memory_high_bytes": int(props["MemoryHigh"]),
            "memory_max_bytes": int(props["MemoryMax"]),
            "swap_max_bytes": int(props["MemorySwapMax"]),
            "runtime_limit": props["RuntimeMaxUSec"],
            "parent_memory_event_deltas": cap["parent_memory_events_delta"],
            "parent_local_memory_event_deltas": cap["parent_memory_events_local_delta"],
            "own_cgroup_memory_sample_available": False,
            "unit_memory_peak_bytes": None,
            "stop_status": stop["status"],
            "stop_returncode": stop["stop_returncode"],
            "post_stop_state": stop["post_state"],
        },
        "interpretation_limits": [
            "This is a fixed-checkpoint output-sensitivity diagnostic on eight exposed fit sequences and sixteen dependent windows, not a target-quality or population-validation result.",
            "RECENT256 versus FULL changes recurrent carry while preserving the identical canonical causal feature history; the observed difference is conditional on this checkpoint and panel.",
            "Output dependence on erased carry does not establish label utility, prove that 256 rows are sufficient/insufficient for quality, or isolate the source of carried information.",
            "The inherited reader hashes and maps selected y files for integrity; no y values were accessed. The terminal unit ControlGroup was empty, so no per-unit cgroup memory peak/current is claimed."
        ],
    }
    output_path = ROOT / "RESULT_AUDIT.json"
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


if __name__ == "__main__":
    report = _audit()
    print(json.dumps({"status": report["status"], "result_sha256": report["identities"]["result"]["sha256"], "audit_path": str(ROOT / "RESULT_AUDIT.json")}, sort_keys=True))
