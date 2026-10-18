"""Root-authority, measured-density, projection, and resource receipt gates."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from control_manifest import verify_control_manifest
from frozen_panel import (
    CAMPAIGN_CAP_SECONDS,
    FIT_CAP_SECONDS,
    OLD_UNIT,
    R01_EXECUTE_SHA256,
    R01_MANIFEST_SHA256,
    R01_PANEL_SHA256,
    R01_POLICY_SHA256,
    R01_ROLLING_SHA256,
    sha256_file,
)

EXPECTED_ROOT = Path("/data/epanov_s/tabdata")
EXPECTED_PACKET = EXPECTED_ROOT / "runtime/b300-r01-v1"
EXPECTED_STATE = EXPECTED_ROOT / "state/r01-execution-v4"
EXPECTED_MAIN_PID = 520360
EXPECTED_MAIN_START_TICKS = 228617705
EXPECTED_MAIN_CGROUP = (
    "/user.slice/user-969845684.slice/user@969845684.service/app.slice/"
    "tabdata-r01-execution-v4.service"
)
EXPECTED_STARTED_UNIX = 1792345872.0381842
EXPECTED_DEADLINE_UNIX = EXPECTED_STARTED_UNIX + CAMPAIGN_CAP_SECONDS
EXPECTED_SOURCE_TREE_SHA256 = "f4ef9ca8450a3e76d6b6f8295259e70aec5024d3a4f3296558bb0323add74130"
EXPECTED_ACTOR_SHA256 = "e6a155ee13a7e907bb7fca2c1ebed07879e8466422015546e20360af9b404220"
DENSITY_SHA256 = "5e029fd3cb023bafe8c45156d604bb12d0e3caf437ab8794f4c4d39ca2635e9a"
DENSITY_RESOURCE_SHA256 = "14f33367b83e15803e426b788f3dcfdaffb1ca188b04aa709d95d05b76b37e94"
PROJECTION_SHA256 = "05c2044cfe638e55be3cca1f8094d18c36f460001e04baf88f4cb2434a5eeefb"
RESOURCE_PREFLIGHT_SHA256 = "714c4e20ab6ee998fa4548343db9a7307137312212599f7600be78cf5b2ef3fc"
EXPECTED_NEW_CPUS = tuple(range(16)) + tuple(range(48, 64))


def _require_sha(path: Path, digest: str, label: str) -> None:
    """Require one receipt path to have the root-pinned bytes."""
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{label} SHA256 must be 64 hexadecimal characters")
    if any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} SHA256 must be lowercase hexadecimal")
    if sha256_file(path) != digest:
        raise ValueError(f"{label} bytes differ from root authority")


def load_authority(path: Path, *, require_execute: bool) -> dict[str, Any]:
    """Validate root's literal takeover authority against frozen host/source pins."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "tabdata-r01-density4-takeover.v1":
        raise ValueError("unknown R01 density4 authority schema")
    if type(value.get("execute_enabled")) is not bool:
        raise ValueError("execute_enabled must be an explicit boolean")
    if require_execute and value["execute_enabled"] is not True:
        raise ValueError("root authority has not enabled successor execution")
    expected = {
        "root": str(EXPECTED_ROOT),
        "packet_root": str(EXPECTED_PACKET),
        "old_state": str(EXPECTED_STATE),
        "old_unit": OLD_UNIT,
        "old_main_pid": EXPECTED_MAIN_PID,
        "old_main_start_ticks": EXPECTED_MAIN_START_TICKS,
        "old_main_cgroup_path": EXPECTED_MAIN_CGROUP,
        "old_unit_started_unix": EXPECTED_STARTED_UNIX,
        "campaign_deadline_unix": EXPECTED_DEADLINE_UNIX,
        "fit_cap_seconds": FIT_CAP_SECONDS,
        "panel_sha256": R01_PANEL_SHA256,
        "execution_manifest_sha256": R01_MANIFEST_SHA256,
        "execute_sha256": R01_EXECUTE_SHA256,
        "rolling_sha256": R01_ROLLING_SHA256,
        "rolling_policy_sha256": R01_POLICY_SHA256,
        "source_tree_identity_sha256": EXPECTED_SOURCE_TREE_SHA256,
        "actor_bundle_sha256": EXPECTED_ACTOR_SHA256,
        "new_unit": "tabdata-r01-density4-v1",
        "new_cpu_ids": list(EXPECTED_NEW_CPUS),
        "new_memory_max_bytes": 192 * 1024**3,
        "new_memory_swap_max_bytes": 0,
        "new_pids_max": 512,
        "max_fit_seconds": FIT_CAP_SECONDS,
        "density_receipt_sha256": DENSITY_SHA256,
        "density_resource_receipt_schema": "tabdata-r01-profile-resource-reconciliation.v1",
        "projection_receipt_sha256": PROJECTION_SHA256,
        "resource_preflight_sha256": RESOURCE_PREFLIGHT_SHA256,
    }
    for key, expected_value in expected.items():
        actual = value.get(key)
        if type(expected_value) is int:
            valid = type(actual) is int and actual == expected_value
        elif type(expected_value) is float:
            valid = type(actual) in {int, float} and math.isclose(
                float(actual), expected_value, rel_tol=0.0, abs_tol=0.0001
            )
        else:
            valid = actual == expected_value
        if not valid:
            raise ValueError(f"root authority field {key} differs from frozen R01 state")
    for key in (
        "main_cmdline_sha256",
        "density_receipt_path",
        "projection_receipt_path",
        "resource_preflight_path",
        "density_resource_receipt_path",
        "live_resource_receipt_path",
        "control_manifest_path",
        "control_bundle_sha256",
        "successor_state_path",
    ):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"root authority must provide {key}")
    reserve = value.get("postprocess_reserve_seconds")
    if (
        isinstance(reserve, bool)
        or not isinstance(reserve, (int, float))
        or not math.isfinite(reserve)
        or reserve <= 0
    ):
        raise ValueError("root authority must provide a positive finite postprocess reserve")
    baseline = value.get("baseline_oom_events")
    required_events = ("high", "max", "oom", "oom_group_kill", "oom_kill")
    if not isinstance(baseline, dict) or any(
        type(baseline.get(key)) is not int for key in required_events
    ):
        raise ValueError("root authority must pin all aggregate OOM event baselines")
    _require_sha(Path(value["density_receipt_path"]), DENSITY_SHA256, "density feasibility receipt")
    _require_sha(Path(value["projection_receipt_path"]), PROJECTION_SHA256, "fit queue projection")
    _require_sha(
        Path(value["resource_preflight_path"]), RESOURCE_PREFLIGHT_SHA256, "aggregate preflight"
    )
    resource_sha = value.get("density_resource_receipt_sha256")
    if resource_sha != DENSITY_RESOURCE_SHA256:
        raise ValueError("root authority profile resource SHA differs from the reconciled evidence")
    validate_density_resource_reconciliation(
        Path(value["density_resource_receipt_path"]), resource_sha
    )
    control_sha = value["control_bundle_sha256"]
    if not isinstance(control_sha, str) or len(control_sha) != 64:
        raise ValueError("root authority must pin the complete successor control bundle")
    verify_control_manifest(
        Path(__file__).resolve().parent,
        Path(value["control_manifest_path"]),
        control_sha,
    )
    return value


def validate_density_receipt(path: Path) -> dict[str, Any]:
    """Verify the two finite mixed R01 profiles against the exact source envelope."""
    _require_sha(path, DENSITY_SHA256, "density feasibility receipt")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema") != "tabdata-density4-r01-feasibility.v1"
        or value.get("scientific_recipe_unchanged") is not True
        or value.get("original_fits_and_controllers_untouched") is not True
        or "frozen a266 FP32 recipe" not in str(value.get("precision_basis", ""))
    ):
        raise ValueError("density receipt does not bind the frozen a266 engineering profiles")
    results = value.get("results")
    if not isinstance(results, dict) or set(results) != {"2", "7"}:
        raise ValueError("density receipt must contain the exact GPU2/GPU7 mixed profile pair")
    expected = {
        "2": {
            "arm": "wide",
            "uuid": "GPU-df6fe7d0-64e5-7af2-467d-c1699cb6fc67",
            "cpus": [56, 58],
            "after_min": 4.6176,
            "before_min": 2.6492,
        },
        "7": {
            "arm": "deep",
            "uuid": "GPU-be85f55c-71a1-f389-eeed-55c63a0e3ce7",
            "cpus": [61, 63],
            "after_min": 4.6540,
            "before_min": 3.3552,
        },
    }
    for gpu, pin in expected.items():
        record = results[gpu]
        if not isinstance(record, dict) or record.get("added_r01_arm") != pin["arm"]:
            raise ValueError(f"density GPU{gpu} arm differs from the measured mixed pair")
        updates = record.get("updates")
        envelope = record.get("envelope")
        if not isinstance(updates, dict) or not isinstance(envelope, dict):
            raise ValueError(f"density GPU{gpu} profile omits update or envelope evidence")
        required_updates = {
            "optimizer_attempts": 340,
            "optimizer_completed_updates": 340,
            "amp_skipped_updates": 0,
            "grad_scaler_enabled": False,
            "loss_checks": 340,
            "loss_finite": True,
            "gradient_finite": True,
            "gradient_stats_steps": 340,
            "gradient_stats_nonfinite_steps": 0,
            "counts_consistent": True,
            "valid_for_throughput": True,
        }
        if any(
            type(updates.get(key)) is not type(expected) or updates.get(key) != expected
            for key, expected in required_updates.items()
        ):
            raise ValueError(f"density GPU{gpu} does not pass the 340-update finite-FP32 gate")
        if (
            envelope.get("schema") != "tabdata-b300-process-envelope.v1"
            or envelope.get("torch") != "2.14.0+cu130"
            or envelope.get("cuda") != "13.0"
            or "B300" not in str(envelope.get("gpu"))
            or envelope.get("visible_devices") != pin["uuid"]
            or envelope.get("cpu_affinity") != pin["cpus"]
            or envelope.get("torch_threads") != 2
            or envelope.get("allocator_budget_bytes") != 6 * 1024**3
        ):
            raise ValueError(f"density GPU{gpu} process envelope differs from the frozen profile")
        rate = record.get("added_sequences_per_second_lower_estimate")
        before = record.get("before_r01_sequences_per_second")
        after = record.get("after_r01_sequences_per_second")
        if any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
            for item in (rate, before, after)
        ):
            raise ValueError(f"density GPU{gpu} throughput measurements are malformed")
        if rate <= 0 or after < pin["after_min"] or before < pin["before_min"] or after <= before:
            raise ValueError(f"density GPU{gpu} aggregate R01 rate does not support scaleout")
    return value


def validate_density_resource_reconciliation(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Require separate root evidence because the original profile unit missed its resource receipt."""
    _require_sha(path, expected_sha256, "density profile resource reconciliation")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or (
        value.get("schema") != "tabdata-r01-profile-resource-reconciliation.v1"
        or value.get("verified") is not True
        or value.get("feasibility_sha256") != DENSITY_SHA256
        or value.get("source_manifest_sha256") != R01_MANIFEST_SHA256
        or value.get("live_monitor_required") is not True
    ):
        raise ValueError("resource reconciliation is not bound to the frozen profile inputs")
    profiles = value.get("profiles")
    expected = {
        2: ("wide", "GPU-df6fe7d0-64e5-7af2-467d-c1699cb6fc67", [56, 58]),
        7: ("deep", "GPU-be85f55c-71a1-f389-eeed-55c63a0e3ce7", [61, 63]),
    }
    if not isinstance(profiles, list) or len(profiles) != 2:
        raise ValueError("resource reconciliation must cover exactly GPU2 and GPU7")
    by_gpu = {
        profile.get("gpu_index"): profile for profile in profiles if isinstance(profile, dict)
    }
    if set(by_gpu) != set(expected):
        raise ValueError("resource reconciliation GPU identities are duplicated or changed")
    for gpu, (arm, uuid, cpus) in expected.items():
        profile = by_gpu[gpu]
        if not isinstance(profile, dict) or (
            profile.get("arm") != arm
            or profile.get("gpu_uuid") != uuid
            or profile.get("cpu_affinity") != cpus
            or profile.get("pid") != profile.get("pgid")
            or profile.get("exited") is not True
            or profile.get("finite_completed_updates") != 340
            or profile.get("precision_recipe") != "fp32"
            or profile.get("allocator_budget_bytes") != 6 * 1024**3
            or profile.get("normal_trainer_return_observed") is not True
        ):
            raise ValueError(
                f"GPU{gpu} resource proof does not bind the finite normal-exit profile"
            )
        for key in ("log_sha256", "request_sha256"):
            digest = profile.get(key)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError(f"GPU{gpu} {key} is missing or malformed")
    limits = value.get("unit_limits")
    if not isinstance(limits, dict) or (
        limits.get("RuntimeMaxUSec") != "10min"
        or limits.get("MemoryMax") != str(32 * 1024**3)
        or limits.get("MemorySwapMax") != "0"
        or limits.get("TasksMax") != "128"
        or limits.get("ActiveState") != "failed"
        or limits.get("ExecMainCode") != "1"
        or limits.get("ExecMainStatus") != "1"
    ):
        raise ValueError("profile resource receipt must retain the actual failed bounded unit")
    reason = str(value.get("failure_reason", "")).lower()
    if "postprocessing" not in reason or "returned normally" not in reason:
        raise ValueError("profile unit failure is not limited to its postprocessing receipt gate")
    if (
        value.get("profile_memory_peak_observed_bytes") is not None
        or value.get("profile_oom_counters_observed") is not None
    ):
        raise ValueError("unknown profile peak and OOM counters must remain explicitly unknown")
    r01_events = value.get("original_r01_memory_events")
    if not isinstance(r01_events, dict) or any(
        r01_events.get(key) != "0" for key in ("high", "max", "oom", "oom_kill", "oom_group_kill")
    ):
        raise ValueError("original R01 unit memory-event evidence is incomplete or nonzero")
    if value.get("original_pr15_completed_fits") != 16:
        raise ValueError("resource reconciliation omits the completed original PR15 cohort")
    if (
        not isinstance(value.get("foreign_clients_still_alive"), list)
        or len(value["foreign_clients_still_alive"]) < 3
    ):
        raise ValueError("resource reconciliation does not show the preserved foreign GPU clients")
    return value


def validate_projection_receipt(
    path: Path,
    deadline_unix: float,
    postprocess_reserve: float,
    density: dict[str, Any],
) -> dict[str, Any]:
    """Verify queue workload, per-fit caps, and available delivery time."""
    _require_sha(path, PROJECTION_SHA256, "density fit queue projection")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or (
        value.get("schema") != "tabdata-density4-fit-queue-projection.v1"
        or value.get("density_receipt_sha256") != DENSITY_SHA256
    ):
        raise ValueError("fit projection does not derive from the admitted density receipt")
    families = value.get("families")
    r01 = families.get("r01") if isinstance(families, dict) else None
    if not isinstance(r01, dict):
        raise ValueError("fit projection lacks the R01 queue")
    before, after = r01.get("before_hours"), r01.get("after_hours")
    if (
        isinstance(before, bool)
        or isinstance(after, bool)
        or not isinstance(before, (int, float))
        or not isinstance(after, (int, float))
        or not math.isfinite(before)
        or not math.isfinite(after)
        or not math.isclose(before, 50.14696378223102, rel_tol=0, abs_tol=0.001)
        or not math.isclose(after, 38.45541027770323, rel_tol=0, abs_tol=0.001)
        or after >= before
    ):
        raise ValueError("fit projection differs from the measured R01 50.15-to-38.46h tail")
    details = value.get("details")
    rows = (
        [row for row in details if isinstance(row, dict) and row.get("family") == "r01"]
        if isinstance(details, list)
        else []
    )
    if len(rows) != 60 or len({row.get("id") for row in rows}) != 60:
        raise ValueError("fit projection must bind all 60 unique R01 cells")
    profiles = density["results"]
    seconds_per_batch = {
        "wide": float(profiles["2"]["profile"]["step_wall"]) / 1000,
        "deep": float(profiles["7"]["profile"]["step_wall"]) / 1000,
    }
    fit_estimates: dict[str, float] = {}
    for row in rows:
        job_id = row.get("id")
        total_batches = row.get("total_fit_batches")
        completed_batches = row.get("completed_fit_batches")
        if (
            not isinstance(job_id, str)
            or type(total_batches) is not int
            or type(completed_batches) is not int
            or not 0 <= completed_batches <= total_batches
        ):
            raise ValueError("per-fit projection rows contain invalid optimizer-batch counts")
        if "_wide_" in job_id:
            arm = "wide"
        elif "_deep_" in job_id:
            arm = "deep"
        else:
            raise ValueError(f"projection row has unknown arm identity: {job_id}")
        remaining = total_batches - completed_batches
        fit_estimates[job_id] = remaining * seconds_per_batch[arm] * 1.25 + 300
    max_estimate = max(fit_estimates.values())
    if max_estimate >= FIT_CAP_SECONDS:
        raise ValueError("measured worst-shape fit estimate exceeds the frozen 12-hour cap")
    remaining_seconds = deadline_unix - time.time()
    required_seconds = float(after) * 3600 + postprocess_reserve
    if remaining_seconds < required_seconds:
        raise TimeoutError(
            f"fit-only tail plus delivery reserve needs {required_seconds / 3600:.2f}h, "
            f"but only {remaining_seconds / 3600:.2f}h remain"
        )
    return {
        "source_receipt": value,
        "fit_only_tail_hours": float(after),
        "baseline_fit_only_tail_hours": float(before),
        "fit_estimate_seconds_by_cell": fit_estimates,
        "max_single_fit_estimate_seconds": max_estimate,
        "postprocess_reserve_seconds": postprocess_reserve,
        "projected_finish_unix": time.time() + required_seconds,
    }


def validate_live_resources(
    path: Path,
    baseline_oom: dict[str, int],
    *,
    max_age_seconds: float = 20.0,
    now_unix: float | None = None,
) -> dict[str, Any]:
    """Require fresh host-watcher evidence for both units and aggregate resources."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("aggregate resource receipt must be a JSON object")
    now = time.time() if now_unix is None else now_unix
    observed = value.get("observed_unix")
    if (
        isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or not math.isfinite(observed)
    ):
        raise ValueError("aggregate resource receipt lacks a finite observation timestamp")
    if observed > now + 2 or now - observed > max_age_seconds:
        raise TimeoutError("aggregate resource receipt is stale")
    validate_campaign_resource_sample(value, baseline_oom)
    return value


def validate_campaign_resource_sample(
    receipt: dict[str, Any], baseline_oom: dict[str, int]
) -> None:
    """Check each unit's hard cap and whole-host headroom without a fake aggregate cap."""
    if receipt.get("scope") != "b300_campaign_aggregate" or receipt.get("verified") is not True:
        raise ValueError("live resource sample must cover the B300 campaign and be verified")
    units = receipt.get("units")
    required_units = {
        OLD_UNIT,
        "tabdata-r01-density4-v1",
        "tabdata-pr15-execution-v2",
    }
    if not isinstance(units, dict) or not required_units <= set(units):
        raise ValueError(
            "live sample must include original R01, successor, and current PR15 v2 units"
        )
    campaign_units = {
        name: record
        for name, record in units.items()
        if isinstance(name, str) and name.startswith("tabdata-")
    }
    if any(not isinstance(name, str) for name in units):
        raise ValueError("campaign resource unit names must be strings")
    if any(not isinstance(record, dict) for record in campaign_units.values()):
        raise ValueError("every active tabdata campaign unit must have a resource record")
    event_totals = {key: 0 for key in baseline_oom}
    current_total = 0
    for name, unit in campaign_units.items():
        if not isinstance(unit, dict):
            raise ValueError(f"unit resource record is malformed: {name}")
        for key, expected in (
            ("memory_max_bytes", 192 * 1024**3),
            ("memory_swap_max_bytes", 0),
            ("pids_max", 512),
        ):
            if type(unit.get(key)) is not int or unit[key] != expected:
                raise ValueError(f"{name} {key} differs from its unit-level cap")
        lifecycle = unit.get("lifecycle")
        if name == "tabdata-pr15-execution-v2" and lifecycle == "terminal_success":
            if (
                unit.get("memory_current_bytes") != 0
                or unit.get("memory_current_basis")
                != "cgroup_absent_and_no_process_members"
                or unit.get("memory_swap_current_bytes") != 0
                or unit.get("pids_current") != 0
            ):
                raise ValueError("terminal PR15 zero memory/swap/pids must reflect absent cgroup and actors")
            terminal = unit.get("terminal_receipt")
            if not isinstance(terminal, dict) or (
                terminal.get("schema") != "tabdata-pr15-terminal-success.v1"
                or terminal.get("completed_successor_cells") != 104
                or terminal.get("preserved_original_fits") != 16
                or terminal.get("results_count") != 104
                or terminal.get("systemd_active_state") != "inactive"
                or terminal.get("systemd_main_pid") != 0
                or terminal.get("cgroup_absent") is not True
                or terminal.get("process_count") != 0
            ):
                raise ValueError("terminal PR15 receipt does not prove successful actor-free completion")
            load_state = terminal.get("systemd_load_state")
            if load_state == "loaded":
                if (
                    terminal.get("systemd_result_observed") != "success"
                    or terminal.get("hard_caps_basis")
                    != "systemd_properties_and_last_observed_cgroup"
                ):
                    raise ValueError("loaded terminal PR15 lacks successful systemd and cap evidence")
            elif load_state == "not-found":
                if (
                    terminal.get("systemd_result_observed") is not None
                    or terminal.get("hard_caps_basis")
                    != "last_observed_live_cgroup_and_systemd_properties"
                ):
                    raise ValueError("garbage-collected PR15 must use its saved live cap evidence")
            else:
                raise ValueError("terminal PR15 systemd load state is unknown")
            for digest_key in ("complete_sha256", "results_sha256"):
                digest = terminal.get(digest_key)
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise ValueError(f"terminal PR15 {digest_key} is missing or malformed")
            proof = terminal.get("terminal_proof")
            proof_observed = proof.get("first_terminal_observed_unix") if isinstance(proof, dict) else None
            last_observed = unit.get("memory_events_last_observed_unix")
            if (
                not isinstance(proof, dict)
                or proof.get("schema") != "tabdata-pr15-terminal-proof.v1"
                or proof.get("complete_sha256") != terminal.get("complete_sha256")
                or proof.get("results_sha256") != terminal.get("results_sha256")
                or proof.get("completed_successor_cells") != 104
                or proof.get("preserved_original_fits") != 16
                or proof.get("results_count") != 104
                or proof.get("last_live_observed_unix") != last_observed
                or proof.get("last_live_control_group")
                != unit.get("last_observed_control_group")
                or proof.get("last_live_memory_events")
                != unit.get("memory_events_last_observed")
                or proof.get("last_live_memory_max_bytes") != unit.get("memory_max_bytes")
                or proof.get("last_live_memory_swap_max_bytes")
                != unit.get("memory_swap_max_bytes")
                or proof.get("last_live_pids_max") != unit.get("pids_max")
                or proof.get("systemd_load_state") != load_state
                or proof.get("systemd_result_observed")
                != terminal.get("systemd_result_observed")
                or proof.get("hard_caps_basis") != terminal.get("hard_caps_basis")
                or isinstance(last_observed, bool)
                or not isinstance(last_observed, (int, float))
                or not math.isfinite(last_observed)
                or isinstance(proof_observed, bool)
                or not isinstance(proof_observed, (int, float))
                or not math.isfinite(proof_observed)
                or not 0 <= proof_observed - float(last_observed) <= 20
            ):
                raise ValueError("terminal PR15 proof is stale or differs from last observed cgroup history")
            if (
                not isinstance(unit.get("last_observed_control_group"), str)
                or not isinstance(unit.get("memory_events_last_observed_unix"), (int, float))
                or isinstance(unit.get("memory_events_last_observed_unix"), bool)
                or not math.isfinite(unit["memory_events_last_observed_unix"])
                or unit["memory_events_last_observed_unix"] <= 0
                or "memory_events" in unit
            ):
                raise ValueError("terminal PR15 must carry only timestamped last-observed event counters")
            events = unit.get("memory_events_last_observed")
            current = 0
        else:
            if lifecycle != "live":
                raise ValueError(f"{name} must have an actual live cgroup observation")
            current = unit.get("memory_current_bytes")
            if type(current) is not int or current < 0 or current >= 180 * 1024**3:
                raise ValueError(f"{name} current memory is missing or at the admission ceiling")
            events = unit.get("memory_events")
            if not isinstance(events, dict):
                raise ValueError(f"{name} lacks cgroup memory event counts")
        if not isinstance(events, dict):
            raise ValueError(f"{name} lacks observed cgroup memory event counts")
        current_total += current
        for key in event_totals:
            count = events.get(key)
            if type(count) is not int or count < 0:
                raise ValueError(f"{name} has invalid memory event counter {key}")
            event_totals[key] += count
    for key, expected in baseline_oom.items():
        if event_totals.get(key) != expected:
            raise ValueError(f"campaign OOM counter changed since the frozen baseline: {key}")
    host = receipt.get("host_memory")
    if not isinstance(host, dict):
        raise ValueError("live sample omits whole-host memory evidence")
    available, total = host.get("MemAvailable_bytes"), host.get("MemTotal_bytes")
    if type(available) is not int or type(total) is not int or available < 64 * 1024**3:
        raise ValueError("whole-host available memory is below the 64-GiB reserve")
    if current_total > total or receipt.get("memory_swap_current_bytes") != 0:
        raise ValueError("whole-host current memory or swap evidence is inconsistent")
    rss = receipt.get("worker_group_rss_bytes")
    if not isinstance(rss, dict) or not rss:
        raise ValueError("live sample omits process-group RSS measurements")
    for group_id, amount in rss.items():
        if (
            not isinstance(group_id, str)
            or type(amount) is not int
            or not 0 <= amount <= 24 * 1024**3
        ):
            raise ValueError(f"worker group RSS violates its 24-GiB process cap: {group_id}")


def resource_baseline(path: Path) -> dict[str, int]:
    """Verify the immutable root preflight and return summed unit event baselines."""
    _require_sha(path, RESOURCE_PREFLIGHT_SHA256, "aggregate resource preflight")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("units"), dict):
        raise ValueError("resource preflight has no unit records")
    totals = {key: 0 for key in ("high", "max", "oom", "oom_group_kill", "oom_kill")}
    for unit in value["units"].values():
        events = unit.get("memory_events") if isinstance(unit, dict) else None
        if not isinstance(events, dict):
            raise ValueError("resource preflight unit lacks memory event counters")
        for key in totals:
            try:
                count = int(events.get(key))
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid preflight OOM counter {key}") from error
            if count < 0:
                raise ValueError(f"negative preflight OOM counter {key}")
            totals[key] += count
    if (
        value.get("proposed_successor_memory_max_bytes") != 192 * 1024**3
        or value.get("proposed_successor_tasks_max") != 512
        or value.get("proposed_successor_cpu_affinity") != list(EXPECTED_NEW_CPUS)
    ):
        raise ValueError("resource preflight does not bind the successor hard envelope")
    return totals
