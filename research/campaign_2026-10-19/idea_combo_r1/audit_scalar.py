"""Compact JSON/sufficient-statistic audit for the single combo endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
REPO_ROOT = CAMPAIGN.parents[1]
R15_RESULT = HERE / "REFERENCE_R15_RESULT.json"
MANIFEST = HERE / "PREPARATION_MANIFEST.json"
R15_RESULT_SHA = "ab149a3fcf9d58c1221300ddca3eba356a16181fe5239ea1b3d7646bfe9099c3"
R15_ARCHIVE_SHA = "9f3404f921bb27ffa92912a88b51142a5d1fa76ab29df417c12c808f30d4269f"
R15_ARCHIVE_BYTES = 178_584
R15_SPARSE_DICTIONARY_SHA = "7a62eac9aa42658c38b92ebe90b84489f7dcb9c8750e9b33e872db8a7b124b06"
R15_SPARSE_WP = 0.4824412038543008
R15_DENSE_WP = 0.4798730901246624
SAMPLE_SHA = "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
MEMBERSHIP_SHA = "e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217"
SPLIT_SHA = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
SUMS_KEY = "official_clipped_sufficient_statistics_w_wy_wp_wyy_wpp_wyp"


def read_object(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value, payload


def read_capture_object(capture: Path, stem: str) -> tuple[dict[str, Any], bytes]:
    """Read the captured REMOTE_ filename, with plain-name support for fixtures."""
    candidates = (capture / f"REMOTE_{stem}.json", capture / f"{stem}.json")
    for path in candidates:
        if path.is_file():
            return read_object(path)
    raise FileNotFoundError(f"missing captured {stem}.json in {capture}")


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def near(a: float, b: float, tol: float = 1e-10) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def wp_from_six(six: list[float]) -> float:
    """Recompute raw-sum weighted Pearson; canonical score remains authority."""
    w, wy, wp, wyy, wpp, wyp = map(float, six)
    if w < 0.0 or not all(math.isfinite(x) for x in (w, wy, wp, wyy, wpp, wyp)):
        raise ValueError("invalid six-sum record")
    if w < 1e-8:
        return 0.0
    my, mp = wy / w, wp / w
    second_y, second_p = wyy / w, wpp / w
    vy = second_y - my * my
    vp = second_p - mp * mp
    eps = 8.0 * 2.220446049250313e-16
    if vy <= eps * max(abs(second_y), my * my):
        vy = 0.0
    if vp <= eps * max(abs(second_p), mp * mp):
        vp = 0.0
    if math.sqrt(max(vy, 0.0)) <= 1e-8 or math.sqrt(max(vp, 0.0)) <= 1e-8:
        return 0.0
    return max(-1.0, min(1.0, (wyp / w - my * mp) / math.sqrt(vy * vp)))


def summarize(deltas: list[float]) -> dict[str, Any]:
    return {
        "count": len(deltas),
        "positive": sum(value > 0.0 for value in deltas),
        "zero": sum(value == 0.0 for value in deltas),
        "negative": sum(value < 0.0 for value in deltas),
        "min": min(deltas),
        "median": median(deltas),
        "max": max(deltas),
    }


def audit(capture: Path, extras: dict[str, Path]) -> dict[str, Any]:
    start, start_bytes = read_capture_object(capture, "START")
    source, source_bytes = read_capture_object(capture, "SOURCE")
    result, result_bytes = read_capture_object(capture, "RESULT")
    done, done_bytes = read_capture_object(capture, "DONE")
    reference, reference_bytes = read_object(R15_RESULT)
    manifest, manifest_bytes = read_object(MANIFEST)
    assert result["schema"] == "nf19.combo.result.v1"
    assert source["schema"] == "nf19.combo.source.v1"
    assert done["schema"] == "nf19.combo.done.v1"
    assert sha(reference_bytes) == R15_RESULT_SHA
    assert sha(manifest_bytes) == result["preparation_manifest_sha256"]
    assert start["preparation_manifest_sha256"] == sha(manifest_bytes)
    assert source["preparation_manifest_sha256"] == sha(manifest_bytes)
    assert (
        source["sample_sha256"] == result["sample_sha256"] == start["sample_sha256"] == SAMPLE_SHA
    )
    assert (
        source["membership_sha256"]
        == result["membership_sha256"]
        == start["membership_sha256"]
        == MEMBERSHIP_SHA
    )
    assert source["split_sha256"] == result["split_sha256"] == start["split_sha256"] == SPLIT_SHA
    source_hash, result_hash = sha(source_bytes), sha(result_bytes)
    assert result["source_sha256"] == done["source_sha256"] == source_hash
    assert done["result_sha256"] == result_hash
    assert done["result_bytes"] == len(result_bytes)
    assert (
        result["reference_r15_result_sha256"]
        == source["r15_reference_result_sha256"]
        == R15_RESULT_SHA
    )
    assert (
        result["reference_r15_archive"]["sha256"]
        == source["r15_reference_archive"]["sha256"]
        == R15_ARCHIVE_SHA
    )
    assert (
        result["reference_r15_archive"]["bytes"]
        == source["r15_reference_archive"]["bytes"]
        == R15_ARCHIVE_BYTES
    )
    assert (
        result["reference_r15_archive"]["sparse_dictionary_fingerprint"]
        == source["r15_reference_archive"]["sparse_dictionary_fingerprint"]
        == R15_SPARSE_DICTIONARY_SHA
    )
    assert source["r15_old_predictors_reinferred_or_regraded"] is False

    pins = manifest["files"]
    assert len(pins) == 38 and len(manifest["runtime_py"]) == 31
    assert source["source_pins"] == pins
    assert set(source["runtime_imports"]) == set(manifest["runtime_py"])
    local_pin_errors: list[str] = []
    for key, expected in pins.items():
        if key.startswith("canonical:"):
            local_path = REPO_ROOT / "src" / key.partition(":")[2]
        else:
            local_path = CAMPAIGN / key
        if not local_path.is_file() or sha(local_path.read_bytes()) != expected:
            local_pin_errors.append(key)
    assert not local_pin_errors, local_pin_errors
    assert result["fit_normalizers_exactly_match_stored_r15"] is True
    assert source["mapped_npy_file_count"] == 288
    assert source["target_files_hashed_and_mapped_for_integrity"] is True
    assert source["fit_target_values_used_for_supervised_training"] is True
    assert source["eval_target_values_used_only_for_the_new_final_endpoint_score"] is True
    assert source["fit_components"] == 64 and source["fit_stride_rows"] == 31_808
    assert source["eval_components"] == 32 and source["eval_scored_rows"] == 59_436
    assert source["all_model_parameters_cpu_float32"] is True
    assert source["torch_runtime"] == {
        "torch_threads": 1,
        "torch_interop_threads": 1,
        "cuda_initialized": False,
        "autocast_cuda_enabled": False,
        "autocast_cpu_enabled": False,
        "tf32_matmul_enabled": False,
        "tf32_cudnn_enabled": False,
    }

    training = result["training"]
    assert (
        result["expected_updates"]
        == training["attempted_updates"]
        == training["successful_updates"]
        == 756
    )
    assert training["nonfinite_count"] == result["nonfinite_count"] == 0
    assert result["epochs"] == training["epochs_completed"] == 12
    assert result["row_exposures"] == training["row_exposures"] == 381_696
    assert (
        training["initial_sha256"]
        == "bce617ce88cb42d0c2cce82f3ec479c90909adc2b9e8adcd4cb6dc1d50963405"
    )
    assert training["variance_head_zero_initialized"] is True
    assert result["all_model_parameters_cpu_float32"] is True

    arms = reference["evaluation"]["arms"]
    sparse, dense = arms["sparse"], arms["dense"]
    assert sparse["official_metric"]["wp"] == R15_SPARSE_WP
    assert dense["official_metric"]["wp"] == R15_DENSE_WP
    candidate = result["evaluation"]
    assert candidate["metric_id"] == "global_wp_clip2_v1"
    assert candidate["selected_rows"] == 59_436 and candidate["evaluation_components"] == 32
    assert candidate["label_mask_parity_against_stored_sparse_and_dense"] is True
    sums = candidate[SUMS_KEY]
    components = sorted(sums, key=int)
    assert len(components) == 32
    assert candidate["component_rows"] == sparse["component_rows"] == dense["component_rows"]
    assert (
        set(candidate["component_scores"])
        == set(candidate["component_drop_one_scores"])
        == set(components)
    )

    # The captured result must preserve the exact target/mask contribution sums;
    # prediction columns may differ, so WPP/WYP are not compared with the old arms.
    max_label_sum_error = 0.0
    for arm in (sparse, dense):
        old_sums = arm[SUMS_KEY]
        assert set(old_sums) == set(components)
        for component in components:
            old = old_sums[component]
            new = sums[component]
            assert len(old) == len(new) == 2
            for target in range(2):
                for column in (0, 1, 3):  # W, WY, WYY: labels and masks only
                    error = abs(float(old[target][column]) - float(new[target][column]))
                    max_label_sum_error = max(max_label_sum_error, error)
                    assert error <= 1e-10
        assert candidate["component_rows"] == arm["component_rows"]

    pooled_six = [
        [math.fsum(float(sums[c][t][j]) for c in components) for j in range(6)] for t in range(2)
    ]
    recomputed_targets = [wp_from_six(side) for side in pooled_six]
    official = candidate["official_metric"]
    raw_metric_error = max(abs(recomputed_targets[i] - float(official[f"t{i}"])) for i in range(2))
    assert raw_metric_error <= 5e-6  # raw six-sums vs canonical centered streaming accumulator
    assert near(official["wp"], 0.5 * math.fsum(recomputed_targets), 5e-6)

    component_errors: list[float] = []
    primary_component: list[float] = []
    secondary_component: list[float] = []
    primary_drop: list[float] = []
    secondary_drop: list[float] = []
    drop_errors: list[float] = []
    for component in components:
        new_metric = candidate["component_scores"][component]
        new_t = [wp_from_six(sums[component][t]) for t in range(2)]
        component_errors.append(max(abs(new_t[i] - float(new_metric[f"t{i}"])) for i in range(2)))
        old_s = sparse["component_scores"][component]
        old_d = dense["component_scores"][component]
        primary_component.append(float(new_metric["wp"]) - float(old_s["wp"]))
        secondary_component.append(float(new_metric["wp"]) - float(old_d["wp"]))
        new_drop = candidate["component_drop_one_scores"][component]
        old_s_drop = sparse["component_drop_one_scores"][component]
        old_d_drop = dense["component_drop_one_scores"][component]
        primary_drop.append(float(new_drop["wp"]) - float(old_s_drop["wp"]))
        secondary_drop.append(float(new_drop["wp"]) - float(old_d_drop["wp"]))
        remain = [
            [
                math.fsum(float(sums[c][t][j]) for c in components if c != component)
                for j in range(6)
            ]
            for t in range(2)
        ]
        drop_t = [wp_from_six(side) for side in remain]
        drop_errors.append(max(abs(drop_t[i] - float(new_drop[f"t{i}"])) for i in range(2)))
    assert max(component_errors) <= 5e-6
    assert max(drop_errors) <= 5e-6

    primary = float(official["wp"]) - R15_SPARSE_WP
    secondary = float(official["wp"]) - R15_DENSE_WP
    assert near(primary, float(candidate["primary_delta_wp_vs_stored_sparse_wmse"]))
    assert near(secondary, float(candidate["secondary_delta_wp_vs_stored_dense_wmse"]))
    assert (
        near(primary, float(result["primary_delta_wp_vs_stored_sparse_wmse"]))
        if "primary_delta_wp_vs_stored_sparse_wmse" in result
        else True
    )

    extra_records: dict[str, dict[str, Any]] = {}
    extra_evidence: dict[str, Any] = {}
    for label, path in extras.items():
        value, payload = read_object(path)
        extra_records[label] = value
        extra_evidence[label] = {"path": str(path), "sha256": sha(payload)}

    admission = extra_records["admission"]
    launch = extra_records["launch"]
    smoke = extra_records["import_smoke"]
    validation = extra_records["capture_validation"]
    stop_envelope = extra_records["stop_receipt"]
    launch_transport = extra_records["launch_transport"]
    unit = "computer-vision-homework-idea-combo-r1-v1.service"
    assert admission["unit"] == launch["unit"] == unit
    assert admission["preparation_manifest_sha256"] == sha(manifest_bytes)
    assert admission["sample"]["sample_sha256"] == SAMPLE_SHA
    assert admission["sample"]["membership_sha256"] == MEMBERSHIP_SHA
    caps = admission["caps"]
    assert caps["blas_threads"] == 1 and caps["cpu_cores"] == 0.5
    assert caps["cuda_visible_devices"] == ""
    assert caps["memory_high_bytes"] == 1_400_000_000
    assert caps["memory_max_bytes"] == 1_500_000_000
    assert caps["memory_swap_max_bytes"] == 0 and caps["runtime_max_s"] == 600
    assert caps["tmpdir"].endswith("/research_tmp")
    invocation = launch["invocation_id"]
    assert invocation == "66d47211356441deba6a6ef9b712e844"
    assert launch["systemd_run_returncode"] == 0
    assert launch_transport["returncode"] == 0 and launch_transport["unknown_outcome"] is False
    launch_props = launch["systemd_properties"]["properties"]
    assert launch_props["InvocationID"] == invocation
    assert launch_props["CPUQuotaPerSecUSec"] == "500ms"
    assert launch_props["MemoryHigh"] == "1400000000"
    assert launch_props["MemoryMax"] == "1500000000"
    assert launch_props["MemorySwapMax"] == "0"
    assert launch_props["RuntimeMaxUSec"] == "10min"
    assert launch_transport["stdout_bytes"] == len(
        (extras["launch_transport"].parent / "launch.stdout").read_bytes()
    )
    assert launch_transport["stdout_sha256"] == sha(
        (extras["launch_transport"].parent / "launch.stdout").read_bytes()
    )
    assert launch_transport["stderr_sha256"] == sha(
        (extras["launch_transport"].parent / "launch.stderr").read_bytes()
    )
    assert smoke["status"] == "PASS"
    assert smoke["preparation_manifest_sha256"] == sha(manifest_bytes)
    assert smoke["runtime_source_sha256"] == {
        path: pins[path] for path in manifest["runtime_py"].values()
    }
    assert smoke["runtime_import_paths"] == source["runtime_imports"]
    assert smoke["reader_sentinel_calls"] == []
    assert smoke["array_reads"] == smoke["checkpoint_reads"] == smoke["parquet_reads"] == 0
    assert smoke["cuda_initialized"] is False

    assert validation["status"] == "PASS_BEFORE_EXACT_STOP"
    assert validation["unit"] == unit and validation["invocation_id"] == invocation
    assert all(validation["checks"].values())
    raw_names = {
        "ROOT_ADMISSION.json": "REMOTE_ROOT_ADMISSION.json",
        "ROOT_LAUNCH.json": "REMOTE_ROOT_LAUNCH.json",
        "ROOT_IMPORT_SMOKE.json": "REMOTE_ROOT_IMPORT_SMOKE.json",
        "PROBE_LOG.txt": "REMOTE_PROBE_LOG.txt",
        "START.json": "REMOTE_START.json",
        "SOURCE.json": "REMOTE_SOURCE.json",
        "RESULT.json": "REMOTE_RESULT.json",
        "DONE.json": "REMOTE_DONE.json",
        "PROGRESS.jsonl": "REMOTE_PROGRESS.jsonl",
    }
    for validated_name, remote_name in raw_names.items():
        raw = (capture / remote_name).read_bytes()
        validated = validation["files"][validated_name]
        assert len(raw) == validated["bytes"]
        assert sha(raw) == validated["sha256"]

    stop = stop_envelope["stop"]
    assert stop_envelope["returncode"] == stop["returncode"] == 0
    assert stop["unit"] == unit
    assert stop["before"]["InvocationID"] == invocation
    assert stop["before"]["MemoryHigh"] == "1400000000"
    assert stop["before"]["MemoryMax"] == "1500000000"
    assert stop["before"]["MemorySwapMax"] == "0"
    assert stop["before"]["CPUQuotaPerSecUSec"] == "500ms"
    assert stop["before"]["RuntimeMaxUSec"] == "10min"
    assert stop["after"]["ActiveState"] == "inactive"
    assert stop["after"]["SubState"] == "dead"
    assert stop["after"]["LoadState"] == "not-found"
    stop_stdout = extras["stop_receipt"].parent / "stop.stdout"
    assert sha(stop_stdout.read_bytes()) == stop_envelope["stdout_sha256"]
    assert len(stop_stdout.read_bytes()) == stop_envelope["stdout_bytes"]
    assert stop_envelope["capture_validation_sha256"] == sha(
        extras["capture_validation"].read_bytes()
    )

    new_metric = candidate["official_metric"]
    sparse_metric = sparse["official_metric"]
    dense_metric = dense["official_metric"]
    primary_target_deltas = [
        float(new_metric[f"t{i}"]) - float(sparse_metric[f"t{i}"]) for i in range(2)
    ]
    secondary_target_deltas = [
        float(new_metric[f"t{i}"]) - float(dense_metric[f"t{i}"]) for i in range(2)
    ]

    return {
        "schema": "nf19.combo.scalar_audit.v1",
        "status": "PASS_SCALAR_AUDIT",
        "result_sha256": result_hash,
        "source_sha256": source_hash,
        "done_sha256": sha(done_bytes),
        "result_bytes": len(result_bytes),
        "reference_r15_result_sha256": R15_RESULT_SHA,
        "preparation_manifest_sha256": sha(manifest_bytes),
        "source_pins": len(pins),
        "runtime_modules": len(manifest["runtime_py"]),
        "evaluation": {
            "rows": candidate["selected_rows"],
            "components": len(components),
            "max_label_mask_six_sum_error": max_label_sum_error,
            "pooled_raw_six_sum_targets": recomputed_targets,
            "official_targets": [official["t0"], official["t1"]],
            "raw_vs_centered_score_max_error": raw_metric_error,
            "component_raw_vs_official_max_error": max(component_errors),
            "drop_one_raw_vs_official_max_error": max(drop_errors),
            "candidate_pooled_wp": official["wp"],
            "r15_sparse_wmse_wp": R15_SPARSE_WP,
            "r15_dense_wmse_wp": R15_DENSE_WP,
            "primary_sparse_nll_minus_stored_sparse_wmse": primary,
            "secondary_sparse_nll_minus_stored_dense_wmse": secondary,
            "primary_target_deltas_vs_stored_sparse_wmse": primary_target_deltas,
            "secondary_target_deltas_vs_stored_dense_wmse": secondary_target_deltas,
            "primary_component_deltas": primary_component,
            "secondary_component_deltas": secondary_component,
            "primary_component_delta_summary": summarize(primary_component),
            "secondary_component_delta_summary": summarize(secondary_component),
            "primary_drop_one_deltas": primary_drop,
            "secondary_drop_one_deltas": secondary_drop,
            "primary_drop_one_delta_summary": summarize(primary_drop),
            "secondary_drop_one_delta_summary": summarize(secondary_drop),
            "component_sensitivities_are_not_independent_replications": True,
        },
        "fit": {
            "updates_attempted_successful": training["successful_updates"],
            "nonfinite_count": training["nonfinite_count"],
            "epochs": result["epochs"],
            "row_exposures": result["row_exposures"],
            "initialization_sha256": training["initial_sha256"],
            "variance_head_zero_initialized": training["variance_head_zero_initialized"],
            "variance_head_changed_under_nll": training["variance_head_changed_under_nll"],
            "normalizers_exact_match_r15": result["fit_normalizers_exactly_match_stored_r15"],
        },
        "archive_metadata_only": {
            "r15_reference": source["r15_reference_archive"],
            "new_candidate": result["model_archive"],
            "archive_payloads_opened_by_auditor": False,
        },
        "root_evidence": {
            **extra_evidence,
            "caps": admission["caps"],
            "invocation_id": invocation,
            "launch_script_sha256": launch_transport["script_sha256"],
            "root_startup_review_sha256": launch_transport["root_startup_review_sha256"],
            "systemd_exit_code": launch_props["ExecMainStatus"],
            "systemd_memory_peak_bytes": stop["before"].get("MemoryPeak"),
            "capture_validation_status": validation["status"],
            "exact_stop_returncode": stop_envelope["returncode"],
            "post_stop": {
                "load_state": stop["after"].get("LoadState"),
                "active_state": stop["after"].get("ActiveState"),
                "sub_state": stop["after"].get("SubState"),
            },
        },
        "limits": [
            "Audit reads compact JSON and source hashes only; it does not open R15 NPZ, candidate NPZ, NPY arrays, or model payloads and does not rerun prediction/scoring.",
            "The paired reference is stored R15 sparse-WMSE; there is no dense-NLL arm, so the result is not a sparse-by-NLL synergy estimate.",
            "The panel is exposed and adaptively reused. Components and leave-one summaries are sensitivity descriptions, not independent replications or fresh-fold evidence.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--launch", type=Path)
    parser.add_argument("--import-smoke", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--stop", type=Path)
    parser.add_argument("--launch-transport", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "RESULT_AUDIT.json")
    args = parser.parse_args()
    extras = {
        key: path
        for key, path in (
            ("admission", args.admission),
            ("launch", args.launch),
            ("import_smoke", args.import_smoke),
            ("capture_validation", args.validation),
            ("stop_receipt", args.stop),
            ("launch_transport", args.launch_transport),
        )
        if path is not None
    }
    audit_record = audit(args.capture, extras)
    payload = json.dumps(audit_record, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with args.output.open("xb") as stream:
        stream.write(payload)
        stream.flush()


if __name__ == "__main__":
    main()
