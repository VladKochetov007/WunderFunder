"""Reclaim only charged file cache after a pinned student-frontier phase."""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import math
import os
import subprocess
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAX_JSON_BYTES = 1_000_000
MAX_PLAN_BYTES = 512_000
MAX_SOURCE_BYTES = 20_000_000
MAX_CGROUP_TEXT_BYTES = 64_000
ACTIVE_STATUSES = {"reserved", "waiting", "running"}
UNRESOLVED_STATUSES = {"preflight", "launch_uncertain"}
TERMINAL_STATUSES = {"failed", "finished", "not_started"}
PLAN_PHASES = (
    "full-fit",
    "post-fit-assertions",
    "frozen-grader-input-check",
    "shape-matched-inference-resource-profile",
    "validate-shape-matched-inference-resource-profile",
    "full-valid-prediction",
    "official-fold0-grade",
    "postgrade-assertions",
)
SHAPE_ALLOWLIST = {(256, 2), (256, 1), (192, 2), (512, 1), (384, 2), (640, 1), (512, 2), (448, 3)}
REQUIRED_PROFILE_OBSERVERS = {
    "attempt_memory_events_final",
    "attempt_memory_peak_final",
    "attempt_systemd_runtime",
    "nvidia_smi_runtime",
    "nvidia_smi_final",
    "parent_hierarchical_memory_events_final",
}
OOM_FIELDS = ("oom", "oom_kill", "oom_group_kill")
MAX_RECLAIM_BYTES = 6_000_000_000
MIN_RETAINED_FILE_BYTES = 6_000_000_000
TARGET_PARENT_BYTES = 8_500_000_000
WORKING_LIMIT_BYTES = 17_000_000_000
PROFILE_NEXT_RESERVE_BYTES = 6_000_000_000
FIT_NEXT_RESERVE_BYTES = 5_000_000_000


def sha256_file(path: Path, maximum_bytes: int = MAX_SOURCE_BYTES) -> str:
    """Hash one immutable file only after enforcing a finite size bound."""
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"{path} is empty or exceeds its {maximum_bytes}-byte hash bound")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    if path.stat().st_size != size:
        raise ValueError(f"{path} changed while it was being hashed")
    return digest.hexdigest()


def read_json(path: Path, maximum_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    """Read a small JSON object only after bounding its on-disk length."""
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"{path} exceeds its {maximum_bytes}-byte JSON read bound")
    with path.open("rb") as source:
        raw = source.read(maximum_bytes + 1)
    if len(raw) != size:
        raise ValueError(f"{path} changed while it was being read")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def argv_value(command: list[str], name: str) -> str:
    """Read one exact value from a frozen argument vector."""
    indexes = [index for index, token in enumerate(command) if token == name]
    if len(indexes) != 1 or indexes[0] + 1 >= len(command):
        raise ValueError(f"expected one value for {name}")
    return command[indexes[0] + 1]


def step(plan: dict[str, Any], index: int, phase: str) -> dict[str, Any]:
    """Return a required sequence step at its frozen index."""
    steps = plan.get("steps")
    if not isinstance(steps, list) or len(steps) != len(PLAN_PHASES):
        raise ValueError("base plan does not contain the exact eight-step R4 sequence")
    item = steps[index]
    if not isinstance(item, dict) or item.get("phase") != phase:
        raise ValueError(f"base plan step {index} is not {phase}")
    return item


def load_bound_plan(
    plan_path: Path, expected_sha256: str, run: str, hidden: int, layers: int
) -> tuple[dict[str, Any], str]:
    """Require the immutable R4 per-shape plan as the handoff evidence contract."""
    actual_sha256 = sha256_file(plan_path.resolve(strict=True), MAX_PLAN_BYTES)
    if actual_sha256 != expected_sha256:
        raise ValueError("base plan hash differs from the admitted handoff contract")
    plan = read_json(plan_path, MAX_PLAN_BYTES)
    shape = [hidden, layers]
    if tuple(shape) not in SHAPE_ALLOWLIST:
        raise ValueError("shape is outside the admitted student-frontier ladder")
    if (
        plan.get("format") != "wnn.bandit-sequence-plan"
        or plan.get("shape") != shape
        or plan.get("run") != run
        or plan.get("step_count") != len(PLAN_PHASES)
    ):
        raise ValueError("base plan campaign, shape, run or step count differs")
    for index, phase in enumerate(PLAN_PHASES):
        item = step(plan, index, phase)
        command = item.get("expected_command")
        if not isinstance(command, list) or any(not isinstance(token, str) for token in command):
            raise ValueError(f"base plan {phase} command is malformed")
    return plan, actual_sha256


def source_value(record: dict[str, Any], source_path: Path) -> str:
    """Find a recorded source hash using its absolute or root-relative key."""
    hashes = record.get("source_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("controller attempt lacks its frozen source hashes")
    candidates = [str(source_path.resolve())]
    root = Path(str(record.get("working_directory", "")))
    if root.is_dir():
        with suppress(ValueError):
            candidates.append(str(source_path.resolve().relative_to(root.resolve())))
    matches = [hashes[key] for key in candidates if key in hashes]
    if len(matches) != 1 or not isinstance(matches[0], str):
        raise ValueError(f"controller source manifest does not pin {source_path}")
    return matches[0]


def validate_pinned_predictor(
    profile_contract: dict[str, Any],
    profile_command: list[str],
    profile_attempt: dict[str, Any],
) -> tuple[Path, str]:
    """Require the profile's copied student predictor and exact recorded source hash."""
    source = profile_contract.get("source")
    if not isinstance(source, dict):
        raise ValueError("profile contract lacks source pins")
    predictor_path = Path(str(source.get("profiled_predictor_copy", ""))).resolve(strict=True)
    predictor_sha256 = str(source.get("profiled_predictor_copy_sha256", ""))
    upstream_predictor = Path(str(source.get("upstream_predictor", ""))).resolve(strict=True)
    if (
        predictor_path == upstream_predictor
        or predictor_path.name == "predict_run.py"
        or predictor_path.name != "predict_student_frontier_f0_frozen.py"
        or predictor_path.as_posix() not in profile_command
        or sha256_file(predictor_path, MAX_SOURCE_BYTES) != predictor_sha256
        or source_value(profile_attempt, predictor_path) != predictor_sha256
    ):
        raise ValueError("profile did not use the exact pinned student predictor source")
    return predictor_path, predictor_sha256


def checked_attempt(
    runtime: Path,
    item: dict[str, Any],
    *,
    expected_kind: str,
    contract_path: Path,
    contract_sha256: str,
) -> dict[str, Any]:
    """Read a terminal controller record and compare its exact frozen command."""
    attempt_id = str(item.get("attempt", ""))
    record_path = runtime / "attempts" / f"{attempt_id}.json"
    record = read_json(record_path)
    expected_command = item.get("expected_command")
    if not isinstance(expected_command, list):
        raise ValueError(f"frozen command is missing for {attempt_id}")
    expected = {
        "id": attempt_id,
        "kind": expected_kind,
        "status": "finished",
        "phase": "terminal",
        "exit_status": 0,
        "failure": None,
        "contract": str(contract_path.resolve()),
        "contract_sha256": contract_sha256,
        "command": expected_command,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"attempt {attempt_id} {key} differs from the frozen plan")
    return record


def require_script_source(record: dict[str, Any], command: list[str]) -> Path:
    """Require the one command script to remain source-hash bound."""
    candidates = [Path(token) for token in command if token.endswith(".py")]
    candidates = [
        candidate for candidate in candidates if candidate.is_absolute() and candidate.is_file()
    ]
    if len(candidates) != 1:
        raise ValueError("controller command does not identify exactly one Python source")
    script_path = candidates[0].resolve(strict=True)
    if source_value(record, script_path) != sha256_file(script_path, MAX_SOURCE_BYTES):
        raise ValueError(
            f"controller attempt was not run from the current pinned source: {script_path}"
        )
    return script_path


def load_phase_evidence(
    plan_path: Path,
    plan_sha256: str,
    runtime: Path,
    run: str,
    hidden: int,
    layers: int,
) -> dict[str, Any]:
    """Bind terminal profile, export and grade receipts to the frozen R4 plan."""
    plan, actual_plan_sha256 = load_bound_plan(plan_path, plan_sha256, run, hidden, layers)
    profile_step = step(plan, 3, PLAN_PHASES[3])
    profile_gate_step = step(plan, 4, PLAN_PHASES[4])
    export_step = step(plan, 5, PLAN_PHASES[5])
    grade_step = step(plan, 6, PLAN_PHASES[6])
    grade_gate_step = step(plan, 7, PLAN_PHASES[7])

    profile_command = profile_step["expected_command"]
    profile_contract_path = Path(argv_value(profile_step["launch_command"], "--contract"))
    profile_contract_sha256 = str(plan["profile_contract_sha256"])
    if str(profile_contract_path.resolve()) != str(Path(plan["profile_contract"]).resolve()):
        raise ValueError("profile worker contract path differs from the plan-level pin")
    if sha256_file(profile_contract_path, MAX_PLAN_BYTES) != profile_contract_sha256:
        raise ValueError("profile contract differs from the frozen plan SHA256")
    profile_contract = read_json(profile_contract_path, MAX_PLAN_BYTES)
    model = next(
        (
            item
            for item in profile_contract.get("models", [])
            if item.get("hidden") == hidden and item.get("layers") == layers
        ),
        None,
    )
    if not isinstance(model, dict) or model.get("run") != run:
        raise ValueError("profile contract does not contain this exact run and shape")
    if profile_step.get("attempt") != model.get("profile_attempt_id"):
        raise ValueError("profile attempt ID differs from the profile contract")
    if profile_command != model.get("worker_command"):
        raise ValueError("profile command differs from its exact contract command")
    profile_attempt_record = checked_attempt(
        runtime,
        profile_step,
        expected_kind="gpu",
        contract_path=profile_contract_path,
        contract_sha256=profile_contract_sha256,
    )
    predictor_path, predictor_sha256 = validate_pinned_predictor(
        profile_contract, profile_command, profile_attempt_record
    )
    if not terminal_profile_resources(profile_attempt_record):
        raise ValueError("profile attempt lacks bounded resource observers or has OOM evidence")

    profile_receipt_path = Path(str(model["profile_validation_receipt_path"]))
    profile_gate_command = profile_gate_step["expected_command"]
    profile_gate_contract_path = Path(argv_value(profile_gate_step["launch_command"], "--contract"))
    if profile_gate_contract_path.resolve() != profile_contract_path.resolve():
        raise ValueError("profile validator contract differs from the profile contract")
    profile_gate_attempt = checked_attempt(
        runtime,
        profile_gate_step,
        expected_kind="cpu",
        contract_path=profile_gate_contract_path,
        contract_sha256=profile_contract_sha256,
    )
    require_script_source(profile_gate_attempt, profile_gate_command)
    if Path(argv_value(profile_gate_command, "--out")).resolve() != profile_receipt_path.resolve():
        raise ValueError("profile validator output path differs from the profile contract")
    profile_receipt = read_json(profile_receipt_path)
    fit_receipt_path = Path(str(model["fit_receipt_path"]))
    fit_receipt = read_json(fit_receipt_path)
    for key, expected in {
        "format": "wnn.student-frontier-prediction-profile-validation",
        "profile_contract_sha256": profile_contract_sha256,
        "attempt_id": profile_step["attempt"],
        "run": run,
        "shape": [hidden, layers],
        "epoch": 0,
        "checkpoint_sha256": fit_receipt.get("checkpoint_sha256"),
        "batch_size": 16,
        "sequence_rows": 20_000,
        "precision": "fp32",
        "prediction_path": model["output_path"],
        "controller_status": "finished",
        "controller_exit_status": 0,
        "oom_counters_zero": True,
        "all_outputs_finite": True,
    }.items():
        if profile_receipt.get(key) != expected:
            raise ValueError(f"profile receipt {key} differs from the exact terminal profile")
    if not str(profile_receipt.get("validation", "")).startswith("passed;"):
        raise ValueError("profile validation receipt does not record a passed terminal gate")
    profile_prediction = Path(str(model["output_path"]))
    if profile_prediction.stat().st_size != 16 * 20_000 * 2 * 4 + 128:
        raise ValueError("profile prediction bytes differ from the exact B16 output geometry")
    if sha256_file(profile_prediction, 16 * 20_000 * 2 * 4 + 1024) != profile_receipt.get(
        "prediction_sha256"
    ):
        raise ValueError("profile prediction hash differs from its validated receipt")
    if (
        profile_receipt.get("peak_host_bytes") != profile_attempt_record.get("peak_ram_bytes")
        or profile_receipt.get("peak_observed_vram_bytes")
        != profile_attempt_record.get("peak_observed_vram_bytes")
        or not 0 < int(profile_receipt.get("peak_cuda_reserved_bytes", 0)) <= 3_500_000_000
    ):
        raise ValueError(
            "profile receipt resource peaks differ from the terminal controller record"
        )

    export_contract_path = Path(argv_value(export_step["launch_command"], "--contract"))
    if export_contract_path.resolve() != profile_receipt_path.resolve():
        raise ValueError("full export is not gated by this shape's profile receipt")
    dependency = export_step.get("profile_gate_dependency")
    if not isinstance(dependency, dict) or dependency != {
        "contract_path": str(profile_receipt_path),
        "contract_must_exist_before_launch": True,
        "source_manifest_hashes_receipt": True,
    }:
        raise ValueError("full export dependency does not bind the matching profile receipt")
    export_command = export_step["expected_command"]
    if (
        predictor_path.as_posix() not in export_command
        or any(Path(token).name == "predict_run.py" for token in export_command)
        or "--campaign-full-valid-feature-only" not in export_command
        or "--campaign-profile-first-batch" in export_command
    ):
        raise ValueError("planned full export does not use the exact student predictor")

    grade_manifest_path = Path(str(plan["grader_inputs_manifest"]))
    grade_manifest_sha256 = str(plan["grader_inputs_manifest_sha256"])
    if sha256_file(grade_manifest_path, MAX_PLAN_BYTES) != grade_manifest_sha256:
        raise ValueError("grader-input manifest differs from the sequence pin")
    grade_manifest = read_json(grade_manifest_path, MAX_PLAN_BYTES)
    if Path(argv_value(grade_step["launch_command"], "--contract")).resolve() != (
        grade_manifest_path.resolve()
    ):
        raise ValueError("grade worker contract differs from the grader-input manifest")
    if Path(argv_value(grade_gate_step["launch_command"], "--contract")).resolve() != (
        grade_manifest_path.resolve()
    ):
        raise ValueError("grade validator contract differs from the grader-input manifest")

    return {
        "plan": plan,
        "plan_sha256": actual_plan_sha256,
        "profile_contract_path": profile_contract_path.resolve(),
        "profile_contract_sha256": profile_contract_sha256,
        "profile_contract": profile_contract,
        "model": model,
        "profile_receipt_path": profile_receipt_path.resolve(),
        "profile_receipt": profile_receipt,
        "profile_attempt_record": profile_attempt_record,
        "profile_gate_attempt_record": profile_gate_attempt,
        "profile_gate_command": profile_gate_command,
        "export_step": export_step,
        "expected_export_command_sha256": hashlib.sha256(
            json.dumps(export_command, separators=(",", ":")).encode()
        ).hexdigest(),
        "grade_step": grade_step,
        "grade_gate_step": grade_gate_step,
        "grade_manifest_path": grade_manifest_path.resolve(),
        "grade_manifest_sha256": grade_manifest_sha256,
        "grade_manifest": grade_manifest,
        "fit_receipt": fit_receipt,
        "predictor_path": predictor_path,
        "predictor_sha256": predictor_sha256,
    }


def terminal_gpu_resources(record: dict[str, Any], reservation_bytes: int) -> bool:
    """Require measured GPU-worker bounds, observers and zero OOM evidence."""
    reservation = record.get("reservation")
    if (
        not isinstance(reservation, dict)
        or reservation.get("ram_bytes") != reservation_bytes
        or reservation.get("cpus") != 1
    ):
        return False
    peak_host = record.get("peak_ram_bytes")
    peak_vram = record.get("peak_observed_vram_bytes")
    if type(peak_host) is not int or not 0 < peak_host <= reservation_bytes:
        return False
    if type(peak_vram) is not int or not 0 < peak_vram < 5_000_000_000:
        return False
    statuses = record.get("observer_status")
    if not isinstance(statuses, dict) or any(
        statuses.get(name) != "ok" for name in REQUIRED_PROFILE_OBSERVERS
    ):
        return False
    if record.get("observer_events") != []:
        return False
    for key in (
        "memory_event_deltas",
        "final_memory_event_deltas",
        "parent_memory_event_deltas",
        "final_parent_memory_event_deltas",
        "parent_hierarchical_memory_event_deltas",
        "final_parent_hierarchical_memory_event_deltas",
    ):
        events = record.get(key)
        if not isinstance(events, dict) or any(events.get(name) != 0 for name in OOM_FIELDS):
            return False
    return True


def terminal_profile_resources(record: dict[str, Any]) -> bool:
    """Require the profile controller's measured resource and OOM evidence."""
    return terminal_gpu_resources(record, 5_000_000_000)


def validate_export_and_grade(evidence: dict[str, Any]) -> dict[str, Any]:
    """For postgrade handoff, require exact export and validated official grade."""
    export_step = evidence["export_step"]
    grade_step = evidence["grade_step"]
    grade_gate_step = evidence["grade_gate_step"]
    run = evidence["plan"]["run"]
    predictor_path = evidence["predictor_path"]
    predictor_sha256 = evidence["predictor_sha256"]
    profile_receipt_path = evidence["profile_receipt_path"]
    profile_receipt_sha256 = sha256_file(profile_receipt_path)

    export_record = checked_attempt(
        Path(evidence["plan"]["runtime"]),
        export_step,
        expected_kind="gpu",
        contract_path=profile_receipt_path,
        contract_sha256=profile_receipt_sha256,
    )
    export_command = export_step["expected_command"]
    if (
        predictor_path.as_posix() not in export_command
        or any(Path(token).name == "predict_run.py" for token in export_command)
        or source_value(export_record, predictor_path) != predictor_sha256
        or "--campaign-full-valid-feature-only" not in export_command
        or "--campaign-profile-first-batch" in export_command
    ):
        raise ValueError("full export is not the exact pinned causal student predictor")
    if not terminal_gpu_resources(export_record, 6_000_000_000):
        raise ValueError("full export lacks bounded resource observers or has OOM evidence")
    output_path = Path(evidence["plan"]["disk_path"]) / "cache" / f"pred_{run}_valid.npy"
    expected_bytes = int(evidence["plan"]["prediction_outputs_bytes_each"])
    if output_path.stat().st_size != expected_bytes:
        raise ValueError("full prediction size differs from the frozen sequence output geometry")
    sidecar_path = output_path.with_suffix(".json")
    sidecar = read_json(sidecar_path)
    for key, expected in {
        "run": run,
        "split": "valid",
        "epoch": 0,
        "checkpoint_sha256": evidence["profile_receipt"]["checkpoint_sha256"],
        "split_manifest_sha256": evidence["grade_manifest"]["split"]["sha256"],
        "campaign_profile_only": False,
        "campaign_export_scope": "full-valid-feature-only",
        "feature_only": True,
        "inference_precision": "fp32",
    }.items():
        if sidecar.get(key) != expected:
            raise ValueError(f"full prediction sidecar {key} differs from the terminal export")
    prediction_sha256 = sidecar.get("prediction_sha256")
    if (
        not isinstance(prediction_sha256, str)
        or len(prediction_sha256) != 64
        or any(character not in "0123456789abcdef" for character in prediction_sha256)
    ):
        raise ValueError("full prediction sidecar lacks a SHA256 identity")

    manifest_path = evidence["grade_manifest_path"]
    manifest_sha256 = evidence["grade_manifest_sha256"]
    grade_record = checked_attempt(
        Path(evidence["plan"]["runtime"]),
        grade_step,
        expected_kind="cpu",
        contract_path=manifest_path,
        contract_sha256=manifest_sha256,
    )
    grade_source_path = require_script_source(grade_record, grade_step["expected_command"])
    expected_grade_source = evidence["grade_manifest"]["grader"]["score_fold0_panel"]["path"]
    expected_grade_source_sha256 = evidence["grade_manifest"]["grader"]["score_fold0_panel"][
        "sha256"
    ]
    if (
        str(grade_source_path.resolve()) != str(Path(expected_grade_source).resolve())
        or sha256_file(grade_source_path, MAX_SOURCE_BYTES) != expected_grade_source_sha256
    ):
        raise ValueError("grade attempt does not use the manifest-pinned scorer")

    grade_gate_command = grade_gate_step["expected_command"]
    grade_gate_record = checked_attempt(
        Path(evidence["plan"]["runtime"]),
        grade_gate_step,
        expected_kind="cpu",
        contract_path=manifest_path,
        contract_sha256=manifest_sha256,
    )
    grade_gate_source_path = require_script_source(grade_gate_record, grade_gate_command)
    if grade_gate_source_path.name != "validate_student_frontier_grade.py":
        raise ValueError("grade-validation attempt uses an unexpected validator source")

    report_path = Path(argv_value(grade_step["expected_command"], "--out"))
    report = read_json(report_path, 2_000_000)
    grade_receipt_path = Path(argv_value(grade_gate_command, "--out"))
    grade_receipt = read_json(grade_receipt_path)
    if not str(grade_receipt.get("validation", "")).startswith("passed;"):
        raise ValueError("grade validation receipt does not record a passed terminal gate")
    reference = None
    if "--reference" in grade_step["expected_command"]:
        reference = argv_value(grade_step["expected_command"], "--reference")
    official = report.get("official")
    if not isinstance(official, dict):
        raise ValueError("grade report lacks its official metric object")
    score = official.get("wp")
    for key, expected in {
        "format": "wnn.student-frontier-fold0-grade-validation",
        "run": run,
        "candidate": f"{run}@ep0",
        "official_metric": "global_wp_clip2_v1",
        "official_wp": float(score) if isinstance(score, (int, float)) else None,
        "split_manifest_sha256": evidence["grade_manifest"]["split"]["sha256"],
        "valid_targets_sha256": evidence["grade_manifest"]["grader"]["targets"]["sha256"],
        "evaluation_groups": 468,
        "reference": reference,
    }.items():
        if grade_receipt.get(key) != expected:
            raise ValueError(f"grade receipt {key} differs from the frozen fold-0 comparison")
    if sha256_file(report_path, 2_000_000) != grade_receipt.get("grade_report_sha256"):
        raise ValueError("grade report hash differs from its terminal grade validation receipt")
    if report.get("candidate") != f"{run}@ep0" or report.get("evaluation_groups") != 468:
        raise ValueError("grade report candidate or fold coverage differs")
    if not isinstance(official, dict) or official.get("metric_id") != "global_wp_clip2_v1":
        raise ValueError("grade report does not contain the exact official metric")
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise ValueError("grade score is missing or nonfinite")
    if report.get("split_manifest_sha256") != evidence["grade_manifest"]["split"]["sha256"]:
        raise ValueError("grade report split hash differs from the frozen manifest")
    if (
        report.get("valid_targets_sha256")
        != evidence["grade_manifest"]["grader"]["targets"]["sha256"]
    ):
        raise ValueError("grade report target hash differs from the frozen manifest")
    if (
        not isinstance(report.get("exposure"), str)
        or "not a public forecast" not in report["exposure"]
    ):
        raise ValueError("grade report lacks its exposed-development label")
    return {
        "export_attempt_id": export_step["attempt"],
        "export_attempt_sha256": sha256_file(
            Path(evidence["plan"]["runtime"]) / "attempts" / f"{export_step['attempt']}.json"
        ),
        "full_prediction_path": str(output_path),
        "full_prediction_bytes": expected_bytes,
        "full_prediction_sha256": str(sidecar["prediction_sha256"]),
        "grade_attempt_id": grade_step["attempt"],
        "grade_attempt_sha256": sha256_file(
            Path(evidence["plan"]["runtime"]) / "attempts" / f"{grade_step['attempt']}.json"
        ),
        "grade_validation_attempt_id": grade_gate_step["attempt"],
        "grade_validation_attempt_sha256": sha256_file(
            Path(evidence["plan"]["runtime"]) / "attempts" / f"{grade_gate_step['attempt']}.json"
        ),
        "grade_report_path": str(report_path),
        "grade_report_sha256": grade_receipt["grade_report_sha256"],
        "grade_validation_receipt_path": str(grade_receipt_path),
        "grade_validation_receipt_sha256": sha256_file(grade_receipt_path),
        "official_wp": float(score),
    }


def read_memory_snapshot(cgroup: Path) -> dict[str, int]:
    """Read current, file and anonymous charges from the parent cgroup."""
    rows = dict(line.split() for line in read_cgroup_text(cgroup / "memory.stat").splitlines())
    return {
        "current": int(read_cgroup_text(cgroup / "memory.current")),
        "file": int(rows["file"]),
        "anon": int(rows["anon"]),
    }


def read_cgroup_text(path: Path) -> str:
    """Read bounded cgroup pseudo-file contents; these files report stat size zero."""
    with path.open("rb") as source:
        raw = source.read(MAX_CGROUP_TEXT_BYTES + 1)
    if not raw or len(raw) > MAX_CGROUP_TEXT_BYTES:
        raise ValueError(f"{path} is empty or exceeds its bounded cgroup read")
    return raw.decode("ascii").strip()


def systemd_parent_properties() -> dict[str, str]:
    """Read current campaign parent properties from user systemd."""
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            "wnn-campaign.slice",
            "--property=CPUQuotaPerSecUSec,MemoryHigh,MemoryMax,MemorySwapMax,ControlGroup",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def current_cpu_quota(cgroup: Path) -> tuple[int, int]:
    """Read the cgroup v2 CPU quota without interpreting a missing field."""
    fields = read_cgroup_text(cgroup / "cpu.max").split()
    if len(fields) != 2 or fields[0] == "max" or not all(field.isdigit() for field in fields):
        raise ValueError("parent cpu.max does not expose a finite quota")
    quota, period = (int(field) for field in fields)
    if quota <= 0 or period <= 0 or quota > 5 * period:
        raise ValueError("parent CPU quota exceeds the current five-core ceiling")
    return quota, period


def assert_no_active_gpu(runtime: Path, self_attempt: str) -> dict[str, Any]:
    """Reject active or unresolved GPU records while both controller locks are held."""
    attempts_dir = runtime / "attempts"
    if not attempts_dir.is_dir():
        raise ValueError("runtime attempt ledger is missing")
    own_record: dict[str, Any] | None = None
    examined = 0
    active_gpu: list[str] = []
    unresolved: list[str] = []
    for path in sorted(attempts_dir.glob("*.json")):
        item = read_json(path, MAX_JSON_BYTES)
        examined += 1
        attempt_id = str(item.get("id", path.stem))
        status = item.get("status")
        kind = item.get("kind")
        if attempt_id == self_attempt:
            own_record = item
            continue
        if status in UNRESOLVED_STATUSES:
            unresolved.append(attempt_id)
        elif status in ACTIVE_STATUSES and (kind == "gpu" or kind not in {"cpu", "gpu"}):
            active_gpu.append(attempt_id)
        elif status == "lost" and kind == "gpu":
            unit = item.get("unit")
            if not isinstance(unit, str) or not unit:
                unresolved.append(attempt_id)
            else:
                result = subprocess.run(
                    ["systemctl", "--user", "show", unit, "--property=ActiveState"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                active_state = next(
                    (
                        line.split("=", 1)[1]
                        for line in result.stdout.splitlines()
                        if line.startswith("ActiveState=")
                    ),
                    "unknown",
                )
                if active_state != "inactive":
                    unresolved.append(attempt_id)
        elif status not in TERMINAL_STATUSES and status != "lost":
            unresolved.append(attempt_id)
    if own_record is None:
        raise ValueError("handoff worker has no durable controller attempt record")
    own_command = own_record.get("command")
    if (
        own_record.get("kind") != "cpu"
        or own_record.get("status") != "running"
        or not isinstance(own_command, list)
        or not any(
            isinstance(token, str) and "reclaim_student_frontier_cache.py" in token
            for token in own_command
        )
    ):
        raise ValueError("handoff must run as its own active bounded CPU controller attempt")
    if active_gpu or unresolved:
        raise RuntimeError(
            f"GPU or unresolved attempts prevent file-cache handoff: active_gpu={active_gpu}, unresolved={unresolved}"
        )
    return {
        "attempt_records_examined": examined,
        "active_gpu_attempts": [],
        "unresolved_attempts": [],
    }


def validate_self_attempt(
    runtime: Path,
    self_attempt: str,
    plan_path: Path,
    plan_sha256: str,
    scope: str,
    out_path: Path,
) -> dict[str, Any]:
    """Bind this helper invocation to the CPU guard and exact base-plan contract."""
    record_path = runtime / "attempts" / f"{self_attempt}.json"
    record = read_json(record_path)
    if (
        record.get("id") != self_attempt
        or record.get("kind") != "cpu"
        or record.get("status") not in {"running", "waiting"}
        or Path(str(record.get("contract", ""))).resolve() != plan_path.resolve()
        or record.get("contract_sha256") != plan_sha256
    ):
        raise ValueError("handoff worker is not bound to its exact CPU controller contract")
    reservation = record.get("reservation")
    if (
        not isinstance(reservation, dict)
        or reservation.get("cpus") != 1
        or reservation.get("ram_bytes") != 400_000_000
    ):
        raise ValueError("handoff worker does not have the frozen one-CPU/400 MB bound")
    command = record.get("command")
    if not isinstance(command, list) or any(not isinstance(token, str) for token in command):
        raise ValueError("handoff controller command is malformed")
    script_path = Path(__file__).resolve(strict=True)
    if source_value(record, script_path) != sha256_file(script_path):
        raise ValueError("handoff worker source differs from its admitted controller source")
    if (
        "--scope" not in command
        or argv_value(command, "--scope") != scope
        or Path(argv_value(command, "--base-plan")).resolve() != plan_path.resolve()
        or argv_value(command, "--base-plan-sha256") != plan_sha256
        or argv_value(command, "--self-attempt") != self_attempt
        or Path(argv_value(command, "--out")).resolve() != out_path.resolve()
    ):
        raise ValueError("handoff worker command differs from its scope, plan or output receipt")
    return record


def reclaim_cache(
    *,
    runtime: Path,
    self_attempt: str,
    parent_group: str,
    cgroup_root: Path,
    next_reserve_bytes: int,
) -> dict[str, Any]:
    """Reclaim file cache under controller/GPU locks without changing limits."""
    if next_reserve_bytes not in {PROFILE_NEXT_RESERVE_BYTES, FIT_NEXT_RESERVE_BYTES}:
        raise ValueError("unexpected next worker reservation")
    expected_current = TARGET_PARENT_BYTES
    parent = systemd_parent_properties()
    if (
        parent.get("CPUQuotaPerSecUSec") != "5s"
        or parent.get("MemoryHigh") != "18000000000"
        or parent.get("MemoryMax") != "20000000000"
        or parent.get("MemorySwapMax") != "0"
        or parent.get("ControlGroup") != parent_group
        or not parent_group.endswith("/wnn-campaign.slice")
    ):
        raise RuntimeError(
            "campaign parent identity or resource limits differ from the frozen policy"
        )
    cgroup = cgroup_root.joinpath(*parent_group.strip("/").split("/"))
    if not cgroup.is_dir() or cgroup.resolve().is_relative_to(cgroup_root.resolve()) is False:
        raise ValueError("campaign parent cgroup path is invalid")
    actual_quota, period = current_cpu_quota(cgroup)
    before_limits = {
        "cpu_max": read_cgroup_text(cgroup / "cpu.max"),
        "memory_high": read_cgroup_text(cgroup / "memory.high"),
        "memory_max": read_cgroup_text(cgroup / "memory.max"),
        "memory_swap_max": read_cgroup_text(cgroup / "memory.swap.max"),
    }
    if not 20_000_000_000 - 4096 <= int(before_limits["memory_max"]) <= 20_000_000_000:
        raise RuntimeError("cgroup memory.max differs from the frozen 20 GB limit")
    if not 18_000_000_000 - 4096 <= int(before_limits["memory_high"]) <= 18_000_000_000:
        raise RuntimeError("cgroup memory.high differs from the frozen 18 GB limit")
    if int(before_limits["memory_swap_max"]) != 0:
        raise RuntimeError("cgroup swap must remain disabled")

    with (runtime / "controller.lock").open("a+") as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (runtime / "gpu.lock").open("a+") as gpu_lock:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ledger = assert_no_active_gpu(runtime, self_attempt)
            locked_parent = systemd_parent_properties()
            if locked_parent != parent:
                raise RuntimeError("campaign parent properties changed before file-cache reclaim")
            before = read_memory_snapshot(cgroup)
            request = min(
                MAX_RECLAIM_BYTES,
                max(0, before["current"] - expected_current),
                max(0, before["file"] - MIN_RETAINED_FILE_BYTES),
            )
            warning = None
            if request:
                try:
                    (cgroup / "memory.reclaim").write_text(
                        f"{request} swappiness=0\n", encoding="ascii"
                    )
                except OSError as error:
                    if error.errno != errno.EAGAIN:
                        raise
                    warning = str(error)
            after = read_memory_snapshot(cgroup)
            after_limits = {
                "cpu_max": read_cgroup_text(cgroup / "cpu.max"),
                "memory_high": read_cgroup_text(cgroup / "memory.high"),
                "memory_max": read_cgroup_text(cgroup / "memory.max"),
                "memory_swap_max": read_cgroup_text(cgroup / "memory.swap.max"),
            }
            if after_limits != before_limits:
                raise RuntimeError("resource limits changed during file-cache reclaim")
            projected = after["current"] + next_reserve_bytes
            return {
                "status": "ready"
                if after["current"] <= expected_current and projected <= WORKING_LIMIT_BYTES
                else "needs_review",
                "cgroup": str(cgroup),
                "before": before,
                "after": after,
                "requested_file_reclaim_bytes": request,
                "swappiness": 0,
                "kernel_partial_reclaim_warning": warning,
                "limits_before": before_limits,
                "limits_after": after_limits,
                "limits_changed": False,
                "parent_cpu_quota_microseconds": actual_quota,
                "parent_cpu_period_microseconds": period,
                "next_reserve_bytes": next_reserve_bytes,
                "projected_current_plus_next_reserve_bytes": projected,
                "committed_working_limit_bytes": WORKING_LIMIT_BYTES,
                "ledger_lock_check": ledger,
            }


def run_handoff(
    *,
    root: Path,
    runtime: Path,
    plan_path: Path,
    plan_sha256: str,
    scope: str,
    run: str,
    hidden: int,
    layers: int,
    self_attempt: str,
    out_path: Path,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> dict[str, Any]:
    """Validate exact terminal evidence, then perform the scoped handoff."""
    if scope not in {"after-profile", "after-grade"}:
        raise ValueError("scope must be after-profile or after-grade")
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite handoff receipt: {out_path}")
    root = root.resolve(strict=True)
    runtime = runtime.resolve(strict=True)
    plan_path = plan_path.resolve(strict=True)
    out_path = out_path.resolve()
    evidence = load_phase_evidence(plan_path, plan_sha256, runtime, run, hidden, layers)
    base_plan = evidence["plan"]
    if Path(base_plan["runtime"]).resolve() != runtime:
        raise ValueError("base plan runtime differs from this handoff")
    if Path(base_plan["disk_path"]).resolve() != root:
        raise ValueError("base plan project root differs from this handoff")
    own_record = validate_self_attempt(
        runtime, self_attempt, plan_path, plan_sha256, scope, out_path
    )
    next_reserve = (
        PROFILE_NEXT_RESERVE_BYTES if scope == "after-profile" else FIT_NEXT_RESERVE_BYTES
    )
    phase_evidence: dict[str, Any] = {}
    if scope == "after-grade":
        phase_evidence = validate_export_and_grade(evidence)
    parent = own_record.get("parent_properties_at_start")
    if not isinstance(parent, dict) or not isinstance(parent.get("ControlGroup"), str):
        raise ValueError("handoff controller record lacks its verified parent cgroup")
    parent_group = str(parent["ControlGroup"])
    resources = reclaim_cache(
        runtime=runtime,
        self_attempt=self_attempt,
        parent_group=parent_group,
        cgroup_root=cgroup_root,
        next_reserve_bytes=next_reserve,
    )
    receipt = {
        "format": "wnn.student-frontier-scoped-file-cache-handoff",
        "version": 1,
        "status": resources["status"],
        "observed_utc": datetime.now(UTC).isoformat(),
        "scope": scope,
        "run": run,
        "shape": [hidden, layers],
        "handoff_attempt_id": self_attempt,
        "handoff_attempt_sha256": sha256_file(
            runtime / "attempts" / f"{self_attempt}.json", MAX_JSON_BYTES
        ),
        "base_plan_path": str(plan_path),
        "base_plan_sha256": plan_sha256,
        "profile_contract_path": str(evidence["profile_contract_path"]),
        "profile_contract_sha256": evidence["profile_contract_sha256"],
        "profile_attempt_id": evidence["profile_attempt_record"]["id"],
        "profile_attempt_sha256": sha256_file(
            runtime / "attempts" / f"{evidence['profile_attempt_record']['id']}.json",
            MAX_JSON_BYTES,
        ),
        "profile_validation_receipt_path": str(evidence["profile_receipt_path"]),
        "profile_validation_receipt_sha256": sha256_file(evidence["profile_receipt_path"]),
        "profile_gate_attempt_id": evidence["profile_gate_attempt_record"]["id"],
        "profile_gate_attempt_sha256": sha256_file(
            runtime / "attempts" / f"{evidence['profile_gate_attempt_record']['id']}.json",
            MAX_JSON_BYTES,
        ),
        "predictor_source_path": str(evidence["predictor_path"]),
        "predictor_source_sha256": evidence["predictor_sha256"],
        "phase_evidence": phase_evidence,
        "resource_handoff": resources,
        "validation": "passed; matching source-bound terminal evidence and idle-GPU cache handoff; no quality selection",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out_path, receipt)
    if receipt["status"] != "ready":
        raise RuntimeError(
            "file-cache handoff did not restore the next phase's working-memory headroom"
        )
    return receipt


def atomic_json(path: Path, value: object) -> None:
    """Write one immutable receipt without replacing prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    if path.exists():
        temporary.unlink()
        raise FileExistsError(f"refusing to overwrite handoff receipt: {path}")
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main(argv: list[str] | None = None) -> int:
    """Run one bounded, source- and terminal-evidence-bound cache handoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--base-plan-sha256", required=True)
    parser.add_argument("--scope", choices=("after-profile", "after-grade"), required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--self-attempt", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cgroup-root", type=Path, default=Path("/sys/fs/cgroup"))
    args = parser.parse_args(argv)
    try:
        receipt = run_handoff(
            root=args.root,
            runtime=args.runtime,
            plan_path=args.base_plan,
            plan_sha256=args.base_plan_sha256,
            scope=args.scope,
            run=args.run,
            hidden=args.hidden,
            layers=args.layers,
            self_attempt=args.self_attempt,
            out_path=args.out,
            cgroup_root=args.cgroup_root,
        )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        if not args.out.exists():
            atomic_json(
                args.out,
                {
                    "format": "wnn.student-frontier-scoped-file-cache-handoff",
                    "version": 1,
                    "status": "needs_review",
                    "observed_utc": datetime.now(UTC).isoformat(),
                    "scope": args.scope,
                    "run": args.run,
                    "shape": [args.hidden, args.layers],
                    "handoff_attempt_id": args.self_attempt,
                    "base_plan_path": str(args.base_plan),
                    "base_plan_sha256": args.base_plan_sha256,
                    "error": f"{type(error).__name__}: {error}",
                    "resource_limits_changed": False,
                    "validation": "blocked; no subsequent phase may be admitted from this receipt",
                },
            )
        raise
    print(json.dumps({"status": receipt["status"], "scope": args.scope, "run": args.run}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
