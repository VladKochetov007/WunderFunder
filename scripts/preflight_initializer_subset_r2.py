"""Check the R3-bound initializer subset packet without loading Torch or payload data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MAX_JSON_BYTES = 2_000_000


def sha256_file(path: Path) -> str:
    """Hash a file in fixed-size reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_json(path: Path) -> tuple[dict[str, Any], bytes]:
    """Read a small JSON artifact only after checking its on-disk length."""
    size = path.stat().st_size
    if size <= 0 or size > MAX_JSON_BYTES:
        raise ValueError(f"JSON exceeds the {MAX_JSON_BYTES}-byte preflight bound: {path}")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError(f"JSON changed while being read: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value, raw


def check_hash(path: Path, expected: str, label: str) -> None:
    """Fail closed when one declared source or artifact hash differs."""
    if sha256_file(path) != expected:
        raise ValueError(f"{label} hash differs from the frozen contract: {path}")


def validate(contract_path: Path) -> dict[str, Any]:
    """Verify source, profile and subset identities without Torch or parquet reads."""
    contract, contract_raw = bounded_json(contract_path)
    if contract.get("format") != "wnn.initializer-only-complete-component-screen":
        raise ValueError("initializer subset contract format is unsupported")
    if contract.get("version") != 2 or contract.get("quality_claim") is not None:
        raise ValueError("expected the non-quality R2 initializer recipe contract")
    if (
        contract.get("status")
        != "FROZEN_R3_PROFILE_BOUND_RECIPE_PENDING_NATIVE_GATE_AND_ROOT_FIT_ADMISSION"
    ):
        raise ValueError("initializer recipe is not in its reviewed pending-gate status")

    recipe_root = Path(contract["source_roots"]["recipe"]).resolve(strict=True)
    for relative, expected in contract["source_sha256"].items():
        check_hash(recipe_root / relative, expected, f"recipe source {relative}")
    for source_name in ("official_grade", "position_buckets"):
        source = contract["scoring_sources"][source_name]
        root = Path(source["root"]).resolve(strict=True)
        for relative, expected in source["source_sha256"].items():
            check_hash(root / relative, expected, f"{source_name} source {relative}")
    support = contract["support_sources"]["cpu_limited_wrapper"]
    support_root = Path(support["root"]).resolve(strict=True)
    for relative, expected in support["source_sha256"].items():
        check_hash(support_root / relative, expected, f"support source {relative}")

    profile = contract["profile_binding"]
    profile_contract_path = Path(profile["contract_path"]).resolve(strict=True)
    profile_contract_copy = Path(profile["contract_copy"]).resolve(strict=True)
    check_hash(profile_contract_path, profile["contract_sha256"], "R3 profile contract")
    check_hash(profile_contract_copy, profile["contract_sha256"], "archived R3 contract copy")
    profile_contract, _ = bounded_json(profile_contract_path)
    if (
        profile_contract.get("format") != "wnn.initializer-complete-sequence-profile-contract"
        or profile_contract.get("version") != 2
        or profile_contract.get("source_sha256", {}).get(
            "scripts/profile_initializer_frontier_r3.py"
        )
        != profile["source_sha256"]
    ):
        raise ValueError("profile contract is not the frozen R3 device-resolution contract")

    profile_result_path = Path(profile["result_path"]).resolve(strict=True)
    profile_result_copy = Path(profile["result_copy"]).resolve(strict=True)
    check_hash(profile_result_path, profile["result_sha256"], "R3 profile result")
    check_hash(profile_result_copy, profile["result_sha256"], "archived R3 result copy")
    result, _ = bounded_json(profile_result_path)
    if (
        result.get("contract_sha256") != profile["contract_sha256"]
        or result.get("quality_claim") is not None
        or result.get("execution_device", {}).get("resolved") != "cuda:0"
        or result.get("profile_scope", {}).get("kind") != "resource_only"
    ):
        raise ValueError("R3 profile result is stale, misbound or not resource-only")

    attempt_path = Path(profile["attempt_receipt_copy"]).resolve(strict=True)
    admission_path = Path(profile["admission_receipt_copy"]).resolve(strict=True)
    check_hash(
        Path(profile["attempt_receipt_original"]).resolve(strict=True),
        profile["attempt_receipt_sha256"],
        "canonical R3 controller attempt",
    )
    check_hash(
        Path(profile["admission_receipt_original"]).resolve(strict=True),
        profile["admission_receipt_sha256"],
        "canonical R3 root admission",
    )
    check_hash(attempt_path, profile["attempt_receipt_sha256"], "R3 controller attempt")
    check_hash(admission_path, profile["admission_receipt_sha256"], "R3 root admission")
    attempt, _ = bounded_json(attempt_path)
    admission, _ = bounded_json(admission_path)
    if (
        attempt.get("id") != profile["attempt_id"]
        or attempt.get("status") != "finished"
        or attempt.get("phase") != "terminal"
        or attempt.get("exit_status") != 0
        or attempt.get("contract_sha256") != profile["contract_sha256"]
        or attempt.get("peak_ram_bytes") != profile["host_peak_bytes"]
        or attempt.get("peak_observed_vram_bytes") != profile["observed_vram_peak_bytes"]
        or attempt.get("memory_event_deltas", {}).get("oom") != 0
        or attempt.get("memory_event_deltas", {}).get("oom_kill") != 0
        or attempt.get("parent_hierarchical_memory_event_deltas", {}).get("oom") != 0
        or admission.get("returncode") != 0
    ):
        raise ValueError("R3 controller records do not prove the frozen successful profile")

    artifact_root = Path(profile["payload_root"]).resolve(strict=True)
    observed_artifacts: dict[str, dict[str, str]] = {}
    for arm in result["arms"]:
        mode = arm["mode"]
        checkpoint = artifact_root / arm["checkpoint_reload"]["path"]
        native_archive = artifact_root / arm["native_export"]["archive"]
        check_hash(checkpoint, arm["checkpoint_reload"]["sha256"], f"{mode} checkpoint")
        check_hash(native_archive, arm["native_export"]["archive_sha256"], f"{mode} export")
        observed_artifacts[mode] = {
            "checkpoint_sha256": sha256_file(checkpoint),
            "native_export_sha256": sha256_file(native_archive),
        }

    subset_path = recipe_root / contract["basis"]["subset_manifest"]
    check_hash(subset_path, contract["basis"]["subset_manifest_sha256"], "subset manifest")
    subset, _ = bounded_json(subset_path)
    if (
        subset.get("selection", {}).get("selected_sequence_count") != 258
        or len(subset.get("fit_order", [])) != 258
        or len(subset.get("selected_source_group_seq_ix_component", [])) != 258
        or subset.get("fit_order_sha256") != contract["cohort"]["fit_order_sha256"]
        or subset.get("selected_source_group_seq_ix_component_sha256")
        != contract["cohort"]["source_group_seq_ix_component_sha256"]
    ):
        raise ValueError("complete-component fit subset differs from the pinned 258-sequence order")

    gate_path = recipe_root / contract["ancestry"]["native_profile_gate_output_path"]
    if gate_path.exists():
        raise FileExistsError(f"preserving any prior native gate output: {gate_path}")
    if contract["ancestry"].get("native_profile_gate_result_sha256") is not None:
        raise ValueError("recipe contract unexpectedly claims an existing native gate result")

    return {
        "format": "wnn.initializer-subset-r3-profile-preflight",
        "version": 1,
        "contract_sha256": hashlib.sha256(contract_raw).hexdigest(),
        "recipe_source_sha256": contract["source_sha256"],
        "profile_contract_sha256": profile["contract_sha256"],
        "profile_result_sha256": profile["result_sha256"],
        "profile_attempt_id": profile["attempt_id"],
        "profile_attempt_receipt_sha256": profile["attempt_receipt_sha256"],
        "profile_admission_receipt_sha256": profile["admission_receipt_sha256"],
        "subset_manifest_sha256": contract["basis"]["subset_manifest_sha256"],
        "selected_sequences": 258,
        "fit_order_sha256": subset["fit_order_sha256"],
        "profile_payload_sha256": observed_artifacts,
        "native_gate_output": str(gate_path),
        "native_gate_result": "PENDING",
        "quality_claim": None,
        "torch_imported": False,
        "parquet_read": False,
        "cuda_allocated": False,
        "fit_or_prediction_performed": False,
    }


def main() -> None:
    """Write one immutable standard-library preflight receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"preserving prior preflight receipt: {args.output}")
    result = validate(args.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as output:
        output.write(raw)
    print(json.dumps({"result": "PASS", "output": str(args.output)}))


if __name__ == "__main__":
    main()
