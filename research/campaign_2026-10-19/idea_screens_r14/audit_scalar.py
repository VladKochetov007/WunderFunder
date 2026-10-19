"""Independent scalar audit of the compact R14 terminal receipts.

Reads JSON receipts only. It never opens the producer parameter archive or any
sample, model, or prediction payload.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
RECORDS = HERE / "records"
CAPTURE = RECORDS / "terminal_capture"
EXPECTED_MANIFEST = "3ed272bc6d1efd6ec016c68f73a5efcfe3211c011cb1f60cbb2b28e80b1f7257"
EXPECTED_RECEIPT = "0315ae81ebdb4e92754135a12dd64052cf78ddd07330f653b60dba22ad08da2e"


def read_json(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    return json.loads(raw), raw


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(left: float, right: float, *, atol: float = 1e-12, rtol: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol)


def score_from_sums(values: list[float]) -> float:
    """Reconstruct one clipped weighted Pearson from W,WY,WP,WYY,WPP,WYP."""
    weight, wy, wp, wyy, wpp, wyp = map(float, values)
    if weight < 1e-8:
        return 0.0
    mean_y, mean_p = wy / weight, wp / weight
    second_y, second_p = wyy / weight, wpp / weight
    var_y, var_p = second_y - mean_y * mean_y, second_p - mean_p * mean_p
    tolerance = 8.0 * sys.float_info.epsilon
    if var_y <= tolerance * max(abs(second_y), mean_y * mean_y):
        var_y = 0.0
    if var_p <= tolerance * max(abs(second_p), mean_p * mean_p):
        var_p = 0.0
    sd_y, sd_p = math.sqrt(max(var_y, 0.0)), math.sqrt(max(var_p, 0.0))
    if sd_y <= 1e-8 or sd_p <= 1e-8:
        return 0.0
    covariance = wyp / weight - mean_y * mean_p
    return max(-1.0, min(1.0, covariance / (sd_y * sd_p)))


def pool_sums(parts: dict, omitted: str | None = None) -> list[list[float]]:
    keys = sorted(key for key in parts if key != omitted)
    return [
        [math.fsum(float(parts[key][target][field]) for key in keys) for field in range(6)]
        for target in range(2)
    ]


def local_pin_path(repo: Path, campaign: Path, key: str) -> Path:
    if key.startswith("canonical:"):
        return repo / "src" / key.split(":", 1)[1]
    return campaign / key


def main() -> None:
    repo = HERE.parents[2]
    campaign = HERE.parent
    manifest, manifest_bytes = read_json(HERE / "PREPARATION_MANIFEST.json")
    receipt, receipt_bytes = read_json(HERE / "TEST_RECEIPT.json")
    require(sha(manifest_bytes) == EXPECTED_MANIFEST, "preparation manifest SHA differs")
    require(sha(receipt_bytes) == EXPECTED_RECEIPT, "test receipt SHA differs")
    pins = manifest["files"]
    pin_errors = []
    for key, expected in pins.items():
        path = local_pin_path(repo, campaign, key)
        if not path.is_file() or sha(path.read_bytes()) != expected:
            pin_errors.append(key)
    require(not pin_errors and len(pins) == 27, f"local source pins differ: {pin_errors}")

    result, result_bytes = read_json(CAPTURE / "REMOTE_RESULT.json")
    source, source_bytes = read_json(CAPTURE / "REMOTE_SOURCE.json")
    done, done_bytes = read_json(CAPTURE / "REMOTE_DONE.json")
    start, _ = read_json(CAPTURE / "REMOTE_START.json")
    validation, _ = read_json(RECORDS / "ROOT_CAPTURE_VALIDATION.json")
    admission, _ = read_json(CAPTURE / "REMOTE_ROOT_ADMISSION.json")
    smoke, _ = read_json(CAPTURE / "REMOTE_ROOT_IMPORT_SMOKE.json")
    launch, _ = read_json(CAPTURE / "REMOTE_ROOT_LAUNCH.json")
    transport, _ = read_json(CAPTURE / "TRANSPORT.json")
    stop_envelope, _ = read_json(RECORDS / "STOP_RECEIPT.json")
    stop_raw, stop_raw_bytes = read_json(RECORDS / "stop.stdout")
    require(result["schema"] == "nf19.r14.result.v1", "unexpected result schema")
    require(source["schema"] == "nf19.r14.source.v1", "unexpected source schema")
    require(done["schema"] == "nf19.r14.done.v1", "unexpected done schema")
    result_sha, source_sha, done_sha = sha(result_bytes), sha(source_bytes), sha(done_bytes)
    require(done["result_sha256"] == result_sha, "DONE/result hash mismatch")
    require(done["result_bytes"] == len(result_bytes), "DONE/result byte count mismatch")
    require(result["source_sha256"] == source_sha, "RESULT/source hash mismatch")
    require(validation["status"] == "PASS_BEFORE_STOP", "capture validation did not pass")
    require(validation["result_sha256"] == result_sha, "capture validation/result hash mismatch")
    require(validation["source_sha256"] == source_sha, "capture validation/source hash mismatch")
    require(validation["done_sha256"] == done_sha, "capture validation/DONE hash mismatch")
    require(validation["files_validated"] == 9, "capture validation file count differs")
    require(validation["source_pins_verified"] == 27, "capture source-pin count differs")
    require(validation["runtime_imports_verified"] == 20, "capture runtime count differs")
    require(source["source_pins"] == pins, "remote and frozen source pin maps differ")
    require(len(source["runtime_imports"]) == 20, "remote runtime import count differs")
    require(set(source["runtime_imports"]) == set(manifest["runtime_py"]), "runtime names differ")
    require(source["preparation_manifest_sha256"] == EXPECTED_MANIFEST, "source manifest differs")
    require(source["sample_sha256"] == start["sample_sha256"] == admission["sample"]["sample_sha256"], "sample hash differs")
    require(source["membership_sha256"] == start["membership_sha256"] == admission["sample"]["membership_sha256"], "membership hash differs")
    require(source["split_sha256"] == start["split_sha256"] == admission["sample"]["split_sha256"], "split hash differs")
    require(source["source_revision"] == "a266b036", "canonical source revision differs")
    require(source["input_npy_files_hashed_and_mapped"] == 288, "mapped NPY count differs")
    require(source["complete_sequences"] == 96 and source["fit_components"] == 64, "fit panel counts differ")
    require(source["fit_stride_rows"] == 31_808, "fit stride-row count differs")
    require(source["eval_components"] == 32 and source["eval_scored_rows"] == 59_436, "eval panel counts differ")
    require(source["fit_target_values_used_for_objective"] is True, "fit target-use flag differs")
    require(source["eval_target_values_used_for_fit_or_normalization"] is False, "eval fit-use flag differs")
    require(source["eval_target_values_used_for_secondary_scores_after_fit"] is True, "eval scoring flag differs")
    require(source["arrays_or_model_payloads_transferred_from_B"] is False, "unexpected payload transfer")
    require(start["output_path"] == admission["output"], "START/admission output path differs")

    require(smoke["status"] == "PASS" and smoke["actual_owned_prefix_module_count"] == 20, "import smoke failed")
    require(smoke["preparation_manifest_sha256"] == EXPECTED_MANIFEST, "smoke manifest differs")
    require(smoke["array_reads"] == smoke["parquet_reads"] == smoke["checkpoint_reads"] == 0, "smoke read payload")
    require(smoke["reader_sentinel_calls"] == [] and not smoke["torch_imported"], "smoke sentinels differ")
    require(smoke["cuda_visible_devices"] == "", "CUDA was visible during smoke")
    require(smoke["actual_owned_modules"] == smoke["expected_owned_modules"], "smoke module inventory differs")
    require(smoke["runtime_import_paths"] == source["runtime_imports"], "smoke and runtime import paths differ")
    require(validation["invocation_id"] == launch["invocation_id"], "validation/launch invocation differs")
    require(validation["unit"] == launch["unit"] == admission["unit"], "unit identity differs")
    require(launch["systemd_run_returncode"] == 0, "systemd launch failed")
    require(transport["returncode"] == 0 and transport["stdout_sha256"] == validation["capture_sha256"], "terminal capture transport differs")

    cap = admission["caps"]
    require(cap == {
        "blas_threads": 1,
        "cpu_cores": 0.5,
        "cuda_visible_devices": "",
        "memory_high_bytes": 1_400_000_000,
        "memory_max_bytes": 1_500_000_000,
        "memory_swap_max_bytes": 0,
        "runtime_max_s": 600,
        "tmpdir": "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp",
    }, "admitted resource contract differs")
    launch_properties = {}
    for line in launch["systemd_properties"]["stdout"].splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            launch_properties[key] = value
    require(launch_properties.get("CPUQuotaPerSecUSec") == "500ms", "CPU quota differs")
    require(launch_properties.get("MemoryHigh") == "1400000000", "memory high differs")
    require(launch_properties.get("MemoryMax") == "1500000000", "memory max differs")
    require(launch_properties.get("MemorySwapMax") == "0", "swap cap differs")
    require(launch_properties.get("RuntimeMaxUSec") == "10min", "runtime cap differs")

    stop = stop_envelope["receipt"]
    require(stop_envelope["returncode"] == 0 and sha(stop_raw_bytes) == stop_envelope["stdout_sha256"], "stop receipt binding differs")
    require(stop["returncode"] == stop_raw["returncode"] == 0, "stop command failed")
    require(stop["unit"] == stop_raw["unit"] == launch["unit"], "stop unit differs")
    require(stop["before"] == stop_raw["before"] and stop["after"] == stop_raw["after"], "stop envelope differs from raw stop record")
    before, after = stop_raw["before"], stop_raw["after"]
    require(before["InvocationID"] == launch["invocation_id"], "stop invocation differs")
    require(before["Result"] == "success" and before["ExecMainStatus"] == "0", "unit did not exit successfully")
    require(before["CPUQuotaPerSecUSec"] == "500ms", "terminal CPU quota differs")
    require(before["MemoryHigh"] == "1400000000" and before["MemoryMax"] == "1500000000", "terminal memory cap differs")
    require(before["MemorySwapMax"] == "0" and before["RuntimeMaxUSec"] == "10min", "terminal swap/runtime cap differs")
    require(0 <= int(before["MemoryPeak"]) < 1_400_000_000, "terminal memory peak exceeds memory high")
    require(after["LoadState"] == "not-found" and after["ActiveState"] == "inactive" and after["SubState"] == "dead", "unit was not fully stopped")

    counts = result["counts"]
    require(counts["fit_components"] == 64 and counts["fit_stride_rows"] == 31_808, "result fit counts differ")
    require(counts["eval_components"] == 32 and counts["eval_scored_rows"] == 59_436, "result eval counts differ")
    require(counts["coefficients_per_target"] == 449 and counts["attempted_iterations_per_arm_target"] == 128, "model dimensions differ")
    require(counts["target_iterations_attempted"] == done["target_iterations_attempted"] == 512, "iteration total differs")
    require(counts["target_nonzero_steps"] == 512 and counts["target_noop_steps"] == 0, "step counts differ")
    objective = result["objective"]
    require(objective["ridge_lambda"] == 100.0 and objective["intercept_penalty"] == 0.0, "ridge objective differs")
    require(objective["target_clipping"] == [-2.0, 2.0] and objective["target_weights"] == "absolute clipped fit target", "target objective differs")
    require(result["coordinate_map"]["empirical_feature_covariance_whitening"] is False, "map scope flag differs")

    optimization = result["optimization"]
    gap_summary: dict[str, object] = {}
    iteration_count = 0
    eta_error = gap_identity_error = 0.0
    for arm in ("arm_a", "arm_b"):
        target_paths = optimization[arm]
        require(len(target_paths) == 2, f"{arm} target path count differs")
        terminal = []
        for target, path in enumerate(target_paths):
            iterations = path["iterations"]
            checkpoints = path["checkpoints"]
            require(len(iterations) == 128 and path["attempted_target_iterations"] == 128, f"{arm}/{target} iteration count differs")
            require(path["nonzero_target_steps"] == 128 and path["no_op_target_steps"] == 0, f"{arm}/{target} no-op count differs")
            require(path["initial_gap_energy"] > 0 and close(path["initial_gap_energy"], path["star_beta_h_beta"]), f"{arm}/{target} initial energy differs")
            require(close(path["initial_relative_gap"], 1.0), f"{arm}/{target} initial relative gap differs")
            require([item["iteration"] for item in checkpoints] == [0, 1, 2, 4, 8, 16, 32, 64, 128], f"{arm}/{target} checkpoints differ")
            weight_sum = float(checkpoints[0]["weight_sum"])
            require(weight_sum > 0, f"{arm}/{target} objective weight is empty")
            previous_risk = float(iterations[0]["risk"])
            previous_gap = float(iterations[0]["relative_gap"])
            for index, item in enumerate(iterations, 1):
                require(item["iteration"] == index and item["finite"] is True, f"{arm}/{target} iteration record invalid")
                require(item["objective_scale"] == "unnormalized_weighted_ridge_quadratic", f"{arm}/{target} objective scale differs")
                require(item["step_status"] == "nonzero_step", f"{arm}/{target} expected step is absent")
                expected_eta = float(item["gradient_squared_norm"]) / float(item["line_curvature"])
                eta_error = max(eta_error, abs(float(item["eta"]) - expected_eta))
                require(close(item["gradient_norm"] ** 2, item["gradient_squared_norm"], atol=1e-6), f"{arm}/{target} gradient norm mismatch")
                require(float(item["line_curvature"]) > 0 and close(item["eta"], expected_eta), f"{arm}/{target} line step mismatch")
                require(float(item["risk"]) <= previous_risk + 1e-11 * max(1.0, abs(previous_risk)), f"{arm}/{target} risk increased")
                require(float(item["relative_gap"]) <= previous_gap + 1e-11, f"{arm}/{target} gap increased")
                reconstructed_gap = (float(item["risk"]) - float(checkpoints[0]["star_risk"])) * weight_sum
                stated_gap = float(item["relative_gap"]) * float(path["initial_gap_energy"])
                gap_identity_error = max(gap_identity_error, abs(reconstructed_gap - stated_gap))
                previous_risk, previous_gap = float(item["risk"]), float(item["relative_gap"])
            for checkpoint in checkpoints:
                require(close(checkpoint["delta_h_delta"] / path["initial_gap_energy"], checkpoint["relative_gap"]), f"{arm}/{target} checkpoint gap mismatch")
            terminal.append(float(path["checkpoints"][-1]["relative_gap"]))
            iteration_count += len(iterations)
        stated_terminal = [float(item["checkpoints"][-1]["relative_gap"]) for item in target_paths]
        require(all(close(a, b) for a, b in zip(stated_terminal, optimization["terminal_relative_gap_by_target"][arm], strict=True)), f"{arm} terminal gap vector differs")
        require(close(sum(stated_terminal) / 2.0, optimization["mean_terminal_relative_gap"][arm]), f"{arm} mean terminal gap differs")
        gap_summary[arm] = {"target_gaps": stated_terminal, "mean_gap": sum(stated_terminal) / 2.0}
    require(iteration_count == 512, "audited iteration records do not total 512")
    energies_a = [item["initial_gap_energy"] for item in optimization["arm_a"]]
    energies_b = [item["initial_gap_energy"] for item in optimization["arm_b"]]
    require(all(close(a, b) for a, b in zip(energies_a, energies_b, strict=True)), "mapped arms have different initial objective gaps")
    expected_gates = [
        float(b["checkpoints"][-1]["relative_gap"]) <= 0.8 * float(a["checkpoints"][-1]["relative_gap"])
        and float(a["checkpoints"][-1]["relative_gap"]) > 0
        for a, b in zip(optimization["arm_a"], optimization["arm_b"], strict=True)
    ]
    require(optimization["predeclared_b_over_a_gap_le_0_8_by_target"] == expected_gates, "gap gate vector differs")
    require(optimization["spending_gate"] is all(expected_gates), "aggregate gap gate differs")
    spectra = result["spectra"]
    condition_ratios = [spectra["arm_b"][i]["condition_number"] / spectra["arm_a"][i]["condition_number"] for i in range(2)]
    require(all(ratio > 1 for ratio in condition_ratios), "reported B condition number is not worse for both targets")

    scores = result["secondary_eval_scores"]
    require(scores["selected_rows"] == 59_436, "secondary score row count differs")
    expected_endpoints = {"arm_a_128", "arm_b_128", "common_optimum"}
    require(set(scores["endpoints"]) == expected_endpoints, "secondary endpoint set differs")
    score_audit = {}
    pooled_errors = component_errors = drop_errors = 0.0
    component_keys: set[str] | None = None
    for endpoint, data in scores["endpoints"].items():
        parts = data["component_official_clipped_six_sums_w_wy_wp_wyy_wpp_wyp"]
        components = set(parts)
        require(len(components) == 32 and len(data["component_scores"]) == 32 and len(data["drop_one_component_scores"]) == 32, f"{endpoint} component/drop-one count differs")
        if component_keys is None:
            component_keys = components
        require(components == component_keys == set(data["component_scores"]) == set(data["drop_one_component_scores"]), f"{endpoint} component sets differ")
        row_counts = {key: int(data["component_scores"][key]["selected_rows"]) for key in components}
        total_rows = sum(row_counts.values())
        require(total_rows == data["pooled"]["selected_rows"] == scores["selected_rows"], f"{endpoint} component row counts do not pool")
        reconstructed = pool_sums(parts)
        pooled_values = [score_from_sums(reconstructed[target]) for target in range(2)]
        pooled = data["pooled"]
        for target in range(2):
            pooled_errors = max(pooled_errors, abs(pooled_values[target] - float(pooled[f"t{target}"])))
        pooled_errors = max(pooled_errors, abs(sum(pooled_values) / 2.0 - float(pooled["wp"])))
        for component in sorted(components):
            component_scores = [score_from_sums(parts[component][target]) for target in range(2)]
            stored = data["component_scores"][component]
            component_errors = max(component_errors, *(abs(component_scores[t] - float(stored[f"t{t}"])) for t in range(2)))
            component_errors = max(component_errors, abs(sum(component_scores) / 2.0 - float(stored["wp"])))
            require(int(data["drop_one_component_scores"][component]["selected_rows"]) == total_rows - row_counts[component], f"{endpoint}/{component} drop-one row count differs")
            without = pool_sums(parts, omitted=component)
            drop_values = [score_from_sums(without[target]) for target in range(2)]
            drop = data["drop_one_component_scores"][component]
            for target in range(2):
                drop_errors = max(drop_errors, abs(drop_values[target] - float(drop[f"t{target}"])))
            drop_errors = max(drop_errors, abs(sum(drop_values) / 2.0 - float(drop["wp"])))
        require(pooled["metric_id"] == "global_wp_clip2_v1", f"{endpoint} scorer identity differs")
        score_audit[endpoint] = {
            "t0": float(pooled["t0"]),
            "t1": float(pooled["t1"]),
            "wp": float(pooled["wp"]),
            "selected_rows": total_rows,
            "components": 32,
            "drop_one_scores_reconstructed": 32,
        }
    require(pooled_errors < 1e-10 and component_errors < 1e-10 and drop_errors < 1e-10, "six-sum reconstruction exceeds tolerance")

    caps = result["parameter_archive"]
    require(caps["path_name"] == "PARAMETERS.npz" and caps["bytes"] == done["parameter_archive"]["bytes"], "archive metadata differs")
    require(caps["sha256"] == done["parameter_archive"]["sha256"] and caps["contains_predictions"] is False, "archive producer metadata differs")
    require(done["elapsed_wall_seconds"] <= 600.0, "reported wall duration exceeds runtime cap")
    require(done["elapsed_cpu_seconds"] <= 0.51 * done["elapsed_wall_seconds"], "reported CPU time conflicts with 0.5-core cap")
    stop_status = {
        "returncode": stop_raw["returncode"],
        "before_state": {key: before.get(key) for key in ("InvocationID", "Result", "ActiveState", "SubState", "ExecMainStatus", "CPUQuotaPerSecUSec", "MemoryHigh", "MemoryMax", "MemorySwapMax", "MemoryPeak")},
        "after_state": {key: after.get(key) for key in ("LoadState", "ActiveState", "SubState")},
        "memory_peak_bytes": int(before["MemoryPeak"]),
        "stdout_sha256": stop_envelope["stdout_sha256"],
    }
    audit = {
        "schema": "nf19.r14.result_audit.v1",
        "status": "PASS_SCALAR_AUDIT",
        "bindings": {
            "result_sha256": result_sha,
            "source_sha256": source_sha,
            "done_sha256": done_sha,
            "capture_validation_sha256": sha((RECORDS / "ROOT_CAPTURE_VALIDATION.json").read_bytes()),
            "result_audit_script_sha256": sha(Path(__file__).read_bytes()),
            "preparation_manifest_sha256": sha(manifest_bytes),
            "test_receipt_sha256": sha(receipt_bytes),
            "invocation_id": validation["invocation_id"],
            "unit": validation["unit"],
        },
        "provenance_and_resources": {
            "local_manifest_pins_verified": len(pins),
            "remote_source_pin_map_matches_manifest": True,
            "runtime_modules_and_import_smoke": 20,
            "sample_sha256": source["sample_sha256"],
            "membership_sha256": source["membership_sha256"],
            "split_sha256": source["split_sha256"],
            "mapped_npy_files": source["input_npy_files_hashed_and_mapped"],
            "source_role_counts": {"fit_components": source["fit_components"], "eval_components": source["eval_components"], "fit_rows": source["fit_stride_rows"], "eval_scored_rows": source["eval_scored_rows"]},
            "launch_caps": cap,
            "elapsed_wall_seconds": done["elapsed_wall_seconds"],
            "elapsed_cpu_seconds": done["elapsed_cpu_seconds"],
            "systemd_memory_peak_bytes": stop_status["memory_peak_bytes"],
            "exact_stop": stop_status,
            "capture_validation": validation["status"],
            "producer_archive_sha256": caps["sha256"],
            "producer_archive_bytes": caps["bytes"],
            "archive_opened_by_auditor": False,
        },
        "optimization_audit": {
            "attempted_iterations": iteration_count,
            "all_iterations_finite": True,
            "all_steps_nonzero": True,
            "max_eta_formula_abs_error": eta_error,
            "max_risk_gap_identity_abs_error": gap_identity_error,
            "arm_a": gap_summary["arm_a"],
            "arm_b": gap_summary["arm_b"],
            "arm_b_over_arm_a_terminal_gap": [
                float(optimization["arm_b"][i]["checkpoints"][-1]["relative_gap"])
                / float(optimization["arm_a"][i]["checkpoints"][-1]["relative_gap"])
                for i in range(2)
            ],
            "condition_number_b_over_a": condition_ratios,
            "per_target_spending_gates": expected_gates,
            "aggregate_spending_gate": optimization["spending_gate"],
        },
        "secondary_scores": {
            "scope": "One fixed exposed eval panel, scored after fitting; secondary only.",
            "endpoints": score_audit,
            "pooled_six_sum_max_abs_error": pooled_errors,
            "component_six_sum_max_abs_error": component_errors,
            "drop_one_six_sum_max_abs_error": drop_errors,
            "arm_b_minus_arm_a_wp": score_audit["arm_b_128"]["wp"] - score_audit["arm_a_128"]["wp"],
            "common_optimum_minus_arm_a_wp": score_audit["common_optimum"]["wp"] - score_audit["arm_a_128"]["wp"],
        },
        "interpretation_limit": "The transformed arm fails the predeclared optimization-gap spending gate and has worse reported Hessian condition numbers on both targets. The secondary panel WP is descriptive; this does not establish generalization, an unbiased full-fold effect, or a production/GRU result.",
        "audit_method": "Standard-library scalar reconstruction from compact JSON and six stored additive sums only. No scorer/data/array/model/NPZ payload was reopened, no fitting or prediction was replayed.",
    }
    encoded = json.dumps(audit, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    output_path = HERE / "RESULT_AUDIT.json"
    with output_path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()


if __name__ == "__main__":
    main()
