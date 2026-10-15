#!/usr/bin/env python3
"""Materialize byte-pinned input contracts for the alpha-0 continuation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
CAMPAIGN = Path(__file__).resolve().parent
R2_VALIDATION = CAMPAIGN / "ALPHA0_SOURCEBOUND_VALIDATION_R2.json"
FULL_HASH_RECEIPT = CAMPAIGN / "ALPHA0_DATASET_BYTE_BINDING_R3.json"
INITIAL_HASH_RECEIPT = CAMPAIGN / "ALPHA0_DATASET_BYTE_BINDING_R3_INITIAL.json"
CHECKER = CAMPAIGN / "verify_alpha0_input_binding_r3.py"
CHECKER_IN_MAIN = (
    ROOT
    / "research/campaign_2026-10-15/student_quality_tradeoff_r1/verify_alpha0_input_binding_r3.py"
)
STAGES = ("p1", "p2", "p3")
DATA_PIN_SOURCE_PATHS = {
    "registry": "research/campaign_2026-10-04/portable_inputs.json",
    "e520_spec": "research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/fit_v4_observer_fallback_candidate_v1/E520_GRADER_SPEC.json",
    "h390_start": "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/H390_R3_EXECUTION_START_R1.json",
}
R2_CONTRACTS = {
    stage: CAMPAIGN / f"ALPHA0_{stage.upper()}_FIT_CONTRACT_R2.json" for stage in STAGES
}
R3_CONTRACTS = {
    stage: CAMPAIGN / f"ALPHA0_{stage.upper()}_INPUT_BINDING_R3.json" for stage in STAGES
}
GATES = CAMPAIGN / "ALPHA0_STAGE_INPUT_GATES_R3.json"
VALIDATION = CAMPAIGN / "ALPHA0_INPUT_BINDING_VALIDATION_R3.json"


def sha256_file(path: Path) -> str:
    """Hash a small metadata or contract file without loading dataset payloads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    """Read a bounded project JSON receipt; Any represents decoded JSON values."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: dict[str, Any]) -> None:
    """Create one immutable-by-convention packet item without replacing evidence."""
    if path.exists():
        raise FileExistsError(f"refusing to replace R3 packet item: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def relative_root_path(path: str) -> str:
    """Convert an absolute project path into a portable repo-relative path."""
    value = Path(path)
    return value.relative_to(ROOT).as_posix() if value.is_absolute() else value.as_posix()


def command_for(stage: str) -> list[str]:
    """Return the bounded input-gate command for one stage, with no fit side effect."""
    common = [
        "env",
        "WNN_SLICE=wnn-campaign.slice",
        "WNN_CPU_QUOTA=100%",
        "WNN_THREADS=1",
        "WNN_MEM_MAX=300000000",
        "WNN_RUNTIME_MAX=600s" if stage == "p1" else "WNN_RUNTIME_MAX=120s",
        str(ROOT / "scripts/limited"),
        "/usr/bin/python3",
        str(CHECKER_IN_MAIN),
    ]
    output = ROOT / "research/campaign_2026-10-15/student_quality_tradeoff_r1"
    if stage == "p1":
        return [
            *common,
            "--mode",
            "establish",
            "--stage",
            "p1",
            "--receipt-out",
            str(output / "ALPHA0_RUNTIME_INPUT_BINDING_P1.json"),
        ]
    previous = "p1" if stage == "p2" else "p2"
    return [
        *common,
        "--mode",
        "verify-stage",
        "--stage",
        stage,
        "--receipt",
        str(output / f"ALPHA0_RUNTIME_INPUT_BINDING_{previous.upper()}.json"),
        "--receipt-out",
        str(output / f"ALPHA0_RUNTIME_INPUT_BINDING_{stage.upper()}.json"),
    ]


def materialize() -> None:
    """Bind R2 worker vectors to whole-file data hashes and A9 endpoint bytes."""
    for path in (*R3_CONTRACTS.values(), GATES, VALIDATION):
        if path.exists():
            raise FileExistsError(f"R3 packet output already exists: {path}")

    full = read_json(FULL_HASH_RECEIPT)
    initial = read_json(INITIAL_HASH_RECEIPT)
    r2 = read_json(R2_VALIDATION)
    full_sha = sha256_file(FULL_HASH_RECEIPT)
    r2_sha = sha256_file(R2_VALIDATION)
    checker_sha = sha256_file(CHECKER)
    materializer_sha = sha256_file(Path(__file__))
    if full["status"] != "PASS_FULL_DATA_AND_A9_CHECKPOINT_SHA; NO_FIT_NO_LAUNCH":
        raise ValueError("full byte receipt did not pass")
    if full.get("fit_authorized") or full.get("launch_authorized"):
        raise ValueError("full byte receipt unexpectedly authorizes work")
    if "receipt_revision" in initial:
        raise ValueError("initial receipt should retain the original ambiguous field label")
    if full["receipt_revision"]["supersedes_initial_receipt_sha256"] != sha256_file(
        INITIAL_HASH_RECEIPT
    ):
        raise ValueError("corrected receipt does not bind the preserved initial receipt")
    if r2_sha != "4d5ae05c6a496446a84d91746a31a25bc20d042f18f681db50f7b6d28aaacb99":
        raise ValueError("R2 source validation receipt changed")

    data_pins = {
        name: {
            "path": item["path"],
            "sha256": item["sha256"],
            "bytes": item["bytes"],
            "footer_bytes": item["footer_bytes"],
            "footer_sha256": item["footer_sha256"],
            "original_mtime_ns": item["original_mtime_ns"],
            "pin_source_receipts": {
                key: {
                    "path": path,
                    "sha256": full["source_receipts_sha256"][key],
                }
                for key, path in DATA_PIN_SOURCE_PATHS.items()
            },
        }
        for name, item in full["datasets"].items()
    }
    stage_entries = {}
    contract_hashes = {}
    for stage in STAGES:
        path = R2_CONTRACTS[stage]
        base = read_json(path)
        stage_pin = r2["stages"][stage]
        contract_hash = sha256_file(path)
        if contract_hash != r2["materialized_files_sha256"][path.name]:
            raise ValueError(f"R2 stage contract hash differs: {stage}")
        if (
            base["source_bindings"]["alpha0_command_sha256"]
            != stage_pin["alpha0_worker_argv_sha256"]
        ):
            raise ValueError(f"R2 worker vector and validation disagree: {stage}")
        attempt_path = Path(base["stage"]["actual_a9_attempt_receipt"])
        attempt_sha = sha256_file(attempt_path)
        expected_attempt_sha = base["source_bindings"]["frozen_a9_attempt_receipt_sha256"]
        if attempt_sha != expected_attempt_sha:
            raise ValueError(f"A9 attempt receipt changed: {stage}")
        endpoint = full["a9_reusable_checkpoints"][stage]
        if sha256_file(Path(endpoint["provenance_path"])) != endpoint["provenance_sha256"]:
            raise ValueError(f"A9 endpoint provenance changed: {stage}")
        for output in [base["stage"]["output_directory"], *base["stage"]["expected_outputs"]]:
            if (ROOT / output).exists():
                raise FileExistsError(f"alpha-0 output appeared after preflight: {output}")

        binding = {
            "schema": "wnn.student-warmparent-alpha0-stage-input-binding.r3",
            "status": "PASS_PINNED_INPUT_BINDING; PREFLIGHT_REQUIRED; NO_FIT_NO_LAUNCH",
            "created_utc": datetime.now(UTC).isoformat(),
            "stage": stage,
            "fit_authorized": False,
            "launch_authorized": False,
            "base_r2_contract_path": path.name,
            "base_r2_contract_sha256": contract_hash,
            "r2_source_validation_path": R2_VALIDATION.name,
            "r2_source_validation_sha256": r2_sha,
            "alpha0_worker_argv_sha256": stage_pin["alpha0_worker_argv_sha256"],
            "datasets": data_pins,
            "source_receipts_sha256": full["source_receipts_sha256"],
            "full_hash_receipt": {
                "path": FULL_HASH_RECEIPT.name,
                "sha256": full_sha,
                "status": full["status"],
            },
            "a9_attempt_receipt_path": relative_root_path(str(attempt_path)),
            "a9_attempt_receipt_sha256": expected_attempt_sha,
            "a9_reusable_checkpoint": {
                "path": endpoint["relative_path"],
                "sha256": endpoint["sha256"],
                "bytes": endpoint["bytes"],
                "epoch": endpoint["epoch"],
                "provenance_path": relative_root_path(endpoint["provenance_path"]),
                "provenance_sha256": endpoint["provenance_sha256"],
            },
            "a9_p1_data_pin_limit": full["historical_binding_limit"],
            "alpha0_run_name": base["run_name"],
            "alpha0_output_directory": base["stage"]["output_directory"],
            "expected_outputs_absent_at_preflight": base["stage"]["expected_outputs"],
            "input_gate": "Run the matching command in ALPHA0_STAGE_INPUT_GATES_R3.json; a pass is not fit admission.",
        }
        write_new(R3_CONTRACTS[stage], binding)
        stage_entries[stage] = {
            "contract_path": R3_CONTRACTS[stage].name,
            "contract_sha256": sha256_file(R3_CONTRACTS[stage]),
            "command_argv": command_for(stage),
            "stage_requires_fresh_root_admission": True,
            "fit_or_launch_authorized_by_gate": False,
        }
        contract_hashes[path.name] = contract_hash

    gates = {
        "schema": "wnn.student-warmparent-alpha0-stage-input-gates.r3",
        "status": "COMMANDS_PINNED; NO_FIT_NO_LAUNCH",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_materializer_path": Path(__file__).name,
        "source_materializer_sha256": materializer_sha,
        "working_directory": str(ROOT),
        "input_checker_path": CHECKER_IN_MAIN.relative_to(ROOT).as_posix(),
        "input_checker_source_sha256": checker_sha,
        "purpose": "Bounded source/data/checkpoint identity checks only; no trainer invocation.",
        "external_predicates_before_any_fit": [
            "Independent critic issues GO on this exact successor commit.",
            "The current H384 complete sequence, export and grade chain succeeds.",
            "Root records a fresh per-stage resource admission after validating the stage boundary.",
        ],
        "stage_commands": stage_entries,
        "p1_hashing_policy": "One streaming train+valid SHA pass and A9 endpoint rehash before P1; do not repeat whole-file hashing per later stage.",
        "p2_p3_identity_policy": "Verify device/inode/size/mtime/ctime and pinned checkpoint provenance; any identity change requires fresh full hashing.",
        "all_gates_no_fit_no_launch": True,
    }
    write_new(GATES, gates)

    files = (*R3_CONTRACTS.values(), GATES)
    validation = {
        "schema": "wnn.student-warmparent-alpha0-input-binding-validation.r3",
        "status": "PASS_PINNED_SOURCE_CONTRACTS_AND_BOUNDED_GATE_COMMANDS; NO_FIT_NO_LAUNCH",
        "created_utc": datetime.now(UTC).isoformat(),
        "fit_authorized": False,
        "launch_authorized": False,
        "full_hash_receipt_path": FULL_HASH_RECEIPT.name,
        "full_hash_receipt_sha256": full_sha,
        "initial_receipt_path": INITIAL_HASH_RECEIPT.name,
        "initial_receipt_sha256": sha256_file(INITIAL_HASH_RECEIPT),
        "source_validation_r2_sha256": r2_sha,
        "r2_contract_sha256": contract_hashes,
        "data_pins": {
            name: {"path": item["path"], "sha256": item["sha256"], "bytes": item["bytes"]}
            for name, item in data_pins.items()
        },
        "a9_checkpoint_sha256": {
            stage: full["a9_reusable_checkpoints"][stage]["sha256"] for stage in STAGES
        },
        "alpha0_output_paths_absent_when_materialized": True,
        "stage_binding_sha256": {path.name: sha256_file(path) for path in R3_CONTRACTS.values()},
        "input_gate_path": GATES.name,
        "input_gate_sha256": sha256_file(GATES),
        "input_checker_sha256": checker_sha,
        "materializer_sha256": materializer_sha,
        "historical_binding_limit": full["historical_binding_limit"],
        "preflight_measurement_note": "The original full-stream receipt did not pin the checker source hash at execution. Its exact dataset/checkpoint digests and cgroup cap are preserved; the corrected current checker is SHA-pinned here and its stage-consumer path is smoke-tested separately.",
        "resource_scope_for_full_stream": full["runtime"],
        "r3_packet_files_sha256": {path.name: sha256_file(path) for path in files},
        "no_fit_no_launch": True,
    }
    write_new(VALIDATION, validation)
    print(
        json.dumps(
            {
                "status": validation["status"],
                "contracts": validation["stage_binding_sha256"],
                "gates_sha256": validation["input_gate_sha256"],
                "validation_sha256": sha256_file(VALIDATION),
                "fit_authorized": False,
                "launch_authorized": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    materialize()
