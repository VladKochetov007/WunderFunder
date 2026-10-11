from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
OPS = (
    MAIN / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011"
)
R1_DIR = (
    Path("/home/vlad/development/wunder/wnn_r4_terminal_collector_20261012")
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/clean_t3_f0_r4_terminal_collection_r1"
)
R1_COLLECTOR = R1_DIR / "collect_r4_terminal.py"
R1_COLLECTOR_SHA256 = "2de1b1d2bf5b843887220b720839248c868f97cf5d598ef552d6ba1f5983e19e"
R1_INVENTORY_RECEIPT = (
    R1_DIR / "read_only_inventory_attempt_001/R4_READONLY_INVENTORY_RECEIPT_R1.json"
)
R4_ROOT_RELEASE = OPS / "clean_t3_fold0_h390_r4_root_release"
R4_COLLECTION = R4_ROOT_RELEASE / "terminal_collection_r1"
R4_ATTEMPTS = R4_COLLECTION / "attempts"
R4_ATTEMPT = R4_ATTEMPTS / "attempt_001"
R4_INVENTORY = R4_COLLECTION / "PROVIDER_INVENTORY_R1.json"
R4_ADMISSION = R4_ROOT_RELEASE / "ROOT_TERMINAL_DOWNLOAD_ADMISSION_R1.json"
R4_STATUS = R4_ROOT_RELEASE / "ROOT_PROVIDER_POLL_R2.json"
R4_PUSH = R4_ROOT_RELEASE / "ROOT_KERNEL_PUSH_R1.json"
R4_GENERIC_RECEIPT = R4_COLLECTION / "TERMINAL_COLLECTION_RECEIPT_R2.json"
R4_DIAGNOSTICS = R4_COLLECTION / "R4_TERMINAL_DIAGNOSTICS_R2.json"
DOWNLOAD_ADMISSION_SHA256 = "3eee6881ec10d8d4b18b5769733982aa9baf47fa28e5a10cbed8698a310e546e"
DOWNLOAD_INVENTORY_SHA256 = "1b834f5e77d6ebd493a93b8e9ae0e2791827078d5b9f3fb05a2f12b21114627b"
KERNEL = "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r4-runner"
PUSH_RECEIPT_NAME = "ROOT_KERNEL_PUSH_R1.json"
MAX_DOWNLOAD_BYTES = 1024**3
MIN_FREE_BYTES = 4 * 1024**3
INNER_RUNTIME_SECONDS = 420


def sha256_file(path: Path) -> str:
    """Hash one local file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path, maximum_bytes: int = 32 * 1024**2) -> dict[str, Any]:
    """Read a bounded JSON object receipt without following symlinks."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum_bytes:
        raise ValueError(f"required receipt is missing, symlinked, or oversized: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"receipt must contain one JSON object: {path}")
    return value


def load_r1_collector() -> Any:
    """Load the frozen R1 R4 validator and bind its exact source hash."""
    if sha256_file(R1_COLLECTOR) != R1_COLLECTOR_SHA256:
        raise ValueError("frozen R1 collector source changed")
    spec = importlib.util.spec_from_file_location("r4_terminal_collector_r1", R1_COLLECTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the frozen R1 collector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_attempt() -> tuple[Any, Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the existing admitted ERROR download without provider access."""
    collector = load_r1_collector()
    static = collector.validate_frozen_r4_bindings()
    if sha256_file(R4_ADMISSION) != DOWNLOAD_ADMISSION_SHA256:
        raise ValueError("exact root terminal-download admission hash changed")
    if sha256_file(R1_INVENTORY_RECEIPT) != DOWNLOAD_INVENTORY_SHA256:
        raise ValueError("exact predownload inventory receipt hash changed")
    admission = collector.validate_download_admission(
        R4_ADMISSION, R4_STATUS, R1_INVENTORY_RECEIPT, R4_COLLECTION
    )
    if not R4_ATTEMPT.is_dir() or R4_ATTEMPT.is_symlink():
        raise ValueError("the already-downloaded attempt_001 is missing or unsafe")
    attempts = sorted(path.name for path in R4_ATTEMPTS.iterdir() if path.is_dir())
    if attempts != ["attempt_001"]:
        raise ValueError("R2 may close only the single existing attempt_001")
    if R4_GENERIC_RECEIPT.exists() or R4_GENERIC_RECEIPT.is_symlink():
        raise FileExistsError("R2 generic receipt already exists; refusing replacement")
    if R4_DIAGNOSTICS.exists() or R4_DIAGNOSTICS.is_symlink():
        raise FileExistsError("R2 diagnostics receipt already exists; refusing replacement")

    inventory_receipt = read_object(R1_INVENTORY_RECEIPT)
    provider_inventory = read_object(R4_INVENTORY)
    if provider_inventory.get("complete") is not True:
        raise ValueError("downloaded provider inventory is incomplete")
    expected_names = admission["expected_output_names"]
    collector.require_exact_inventory_names(expected_names, provider_inventory.get("files"))
    if provider_inventory.get("response_log") != inventory_receipt["endpoint_inventory"].get(
        "response_log"
    ):
        raise ValueError("downloaded inventory response-log binding differs from preinventory")

    status_file = read_object(R4_ATTEMPT / "provider_status.json")
    _, admitted_status = collector.parse_status_snapshot(
        R4_STATUS, "2026-10-11T21:30:16.866625+00:00"
    )
    if (
        status_file.get("kernel") != KERNEL
        or status_file.get("status") != "KernelWorkerStatus.ERROR"
        or admitted_status.get("kernel") != KERNEL
        or admitted_status.get("status") != status_file.get("status")
        or admitted_status.get("failure_message", "") != status_file.get("failure_message", "")
    ):
        raise ValueError("downloaded attempt status differs from the admitted terminal ERROR")

    result = read_object(R4_ATTEMPT / "attempt_result.json")
    if (
        result.get("cli_version_match") is not True
        or result.get("logs", {}).get("exit_code") != 0
        or result.get("output", {}).get("exit_code") != 0
    ):
        raise ValueError("the recorded logs/output download is not successful and reusable")
    adapter = collector.load_adapter()
    source_bindings = adapter.verify_collector_sources()
    cli_info = adapter.terminal.kaggle_cli_version(str(MAIN / ".venv/bin/kaggle"))
    if cli_info != {"exit_code": 0, "version": "Kaggle CLI 2.2.4"}:
        raise ValueError("pinned Kaggle CLI is no longer available for receipt closure")
    return (
        collector,
        adapter,
        static,
        admission,
        {
            "provider_inventory": provider_inventory,
            "status_file": status_file,
            "result": result,
            "cli_info": cli_info,
            "source_bindings": source_bindings,
        },
    )


def build_generic_receipt(
    adapter: Any,
    status_file: dict[str, Any],
    result: dict[str, Any],
    cli_info: dict[str, Any],
) -> dict[str, Any]:
    """Create the reviewed generic receipt using the exact R4 admission and push."""
    target = {
        "kernel": KERNEL,
        "resource": "gpu",
        "root_admission_sha256": DOWNLOAD_ADMISSION_SHA256,
        "push_receipt": PUSH_RECEIPT_NAME,
    }
    config = {
        "repo_root": str(MAIN),
        "config_dir": str(R4_ROOT_RELEASE),
        "minimum_free_bytes": MIN_FREE_BYTES,
        "maximum_download_bytes": MAX_DOWNLOAD_BYTES,
        "kaggle_cli": str(MAIN / ".venv/bin/kaggle"),
    }
    return adapter.terminal._terminal_receipt(
        target=target,
        status_row=status_file,
        attempt_dir=R4_ATTEMPT,
        artifact_dir=R4_ATTEMPTS,
        result=result,
        cli_info=cli_info,
        config=config,
    )


def build_diagnostics(
    collector: Any,
    adapter: Any,
    admission: dict[str, Any],
    evidence: dict[str, Any],
    generic_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the actual pipeline failure and generic consumer checks to R2."""
    inventory = evidence["provider_inventory"]
    comparison = adapter.compare_provider_inventory(
        inventory["files"],
        R4_ATTEMPT,
        provider_response_log=inventory.get("response_log"),
    )
    low_level_complete = adapter.terminal.collection_is_complete(R4_GENERIC_RECEIPT, R4_ATTEMPTS)
    complete = adapter.terminal_evidence_complete(low_level_complete, inventory, comparison)
    if not complete:
        raise ValueError("actual downloaded attempt fails the generic terminal consumer")

    diagnostics = collector.partial_terminal_diagnostics(
        R4_ATTEMPT, comparison, evidence["status_file"]
    )
    output_root = R4_ATTEMPT / "outputs"
    pipeline_path = collector.unique_suffix_file(
        output_root, "results/teacher_cohort_pipeline.json"
    )
    pipeline = collector.read_small_json(pipeline_path)
    guard_path = collector.unique_suffix_file(
        output_root, "results/paired_profile_guard_receipt.json"
    )
    guard = collector.read_small_json(guard_path)
    preflight_path = collector.unique_suffix_file(
        output_root, "results/profile_preflight_inventory.json"
    )
    preflight = collector.read_small_json(preflight_path)

    wrapper = {
        "schema": "wnn.kaggle.r4_terminal_receipt_repair.r2",
        "status": "terminal_download_complete_science_not_validated",
        "repair_scope": "receipt_creation_only_same_attempt_no_redownload",
        "repair_reason": "R1 omitted root_admission_sha256 from the generic target after downloads completed",
        "kernel": KERNEL,
        "root_download_admission_sha256": DOWNLOAD_ADMISSION_SHA256,
        "root_fit_dispatch_sha256": collector.R4_FINAL_DISPATCH_SHA256,
        "root_kernel_push_sha256": sha256_file(R4_PUSH),
        "r1_collector_sha256": R1_COLLECTOR_SHA256,
        "r2_collector_source_bindings": evidence["source_bindings"],
        "generic_terminal_receipt": {
            "path": str(R4_GENERIC_RECEIPT),
            "sha256": generic_sha256,
            "root_admission_sha256": DOWNLOAD_ADMISSION_SHA256,
            "complete": low_level_complete,
        },
        "consumer_verification": {
            "provider_inventory_matches_download": comparison.get("matches") is True,
            "terminal_evidence_complete": complete,
            "provider_file_count": comparison.get("provider_file_count"),
            "downloaded_file_count": comparison.get("downloaded_file_count"),
            "response_log_match": comparison.get("response_log_match"),
            "inventory_sha256": sha256_file(R4_INVENTORY),
        },
        "pipeline": {
            "status": pipeline.get("status"),
            "error_type": pipeline.get("error_type"),
            "error": pipeline.get("error"),
            "completed_stages": [
                {
                    "name": row.get("name"),
                    "status": row.get("status"),
                    "returncode": row.get("returncode"),
                    "elapsed_seconds": row.get("elapsed_seconds"),
                    "workers": row.get("workers"),
                }
                for row in pipeline.get("stages", [])
                if isinstance(row, dict)
            ],
            "paired_profile_guard": guard,
            "host_preflight_inventory": preflight,
            "pipeline_sha256": sha256_file(pipeline_path),
            "profile_guard_sha256": sha256_file(guard_path),
            "profile_preflight_sha256": sha256_file(preflight_path),
            "fit_stage_started": any(
                row.get("name") == "teacher_fits" for row in pipeline.get("stages", [])
            ),
            "export_stage_started": any(
                row.get("name") == "teacher_exports" for row in pipeline.get("stages", [])
            ),
            "runner_stage_summaries": diagnostics.get("stage_summaries"),
            "expected_fit_artifacts_present": diagnostics.get("expected_fit_artifacts_present"),
            "expected_prediction_arrays_present": diagnostics.get(
                "expected_prediction_arrays_present"
            ),
            "worker_logs_present": diagnostics.get("worker_logs_present"),
        },
        "scientific_interpretation": (
            "Infrastructure execution failure after paired profiling; no teacher fit or export "
            "was started, and the paired profile guard passed. This is not a negative model result."
        ),
        "root_download_inventory_receipt_sha256": DOWNLOAD_INVENTORY_SHA256,
        "root_admission_collection_dir": admission.get("collection_dir"),
        "downloaded_attempt": str(R4_ATTEMPT),
        "strict_science_validation": "not_run_terminal_error",
    }
    return wrapper, comparison


def close_existing_attempt(*, write_receipts: bool) -> dict[str, Any]:
    """Validate and optionally seal receipt closure for the existing attempt only."""
    collector, adapter, _static, admission, evidence = validate_attempt()
    generic = build_generic_receipt(
        adapter,
        evidence["status_file"],
        evidence["result"],
        evidence["cli_info"],
    )
    if not write_receipts:
        comparison = adapter.compare_provider_inventory(
            evidence["provider_inventory"]["files"],
            R4_ATTEMPT,
            provider_response_log=evidence["provider_inventory"].get("response_log"),
        )
        return {
            "status": "validated_existing_attempt_no_writes",
            "root_download_admission_sha256": DOWNLOAD_ADMISSION_SHA256,
            "inventory_matches": comparison.get("matches") is True,
            "attempt_result": evidence["result"],
        }
    if sha256_file(R4_ADMISSION) != DOWNLOAD_ADMISSION_SHA256:
        raise ValueError("root terminal-download admission changed before receipt write")
    generic_sha256 = adapter.write_immutable_json(R4_GENERIC_RECEIPT, generic)
    wrapper, comparison = build_diagnostics(collector, adapter, admission, evidence, generic_sha256)
    wrapper_sha256 = adapter.write_immutable_json(R4_DIAGNOSTICS, wrapper)
    return {
        "status": wrapper["status"],
        "root_download_admission_sha256": DOWNLOAD_ADMISSION_SHA256,
        "generic_terminal_receipt": str(R4_GENERIC_RECEIPT),
        "generic_terminal_receipt_sha256": generic_sha256,
        "diagnostics_receipt": str(R4_DIAGNOSTICS),
        "diagnostics_receipt_sha256": wrapper_sha256,
        "provider_inventory_matches": comparison.get("matches") is True,
        "pipeline_error_type": wrapper["pipeline"]["error_type"],
        "pipeline_error": wrapper["pipeline"]["error"],
        "fit_stage_started": wrapper["pipeline"]["fit_stage_started"],
        "export_stage_started": wrapper["pipeline"]["export_stage_started"],
    }


def main() -> int:
    """Close the R4 receipt only after explicit --write on the pinned local attempt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write new R2 receipts exclusively")
    args = parser.parse_args()
    try:
        result = close_existing_attempt(write_receipts=args.write)
    except (OSError, ValueError, RuntimeError, FileExistsError, KeyError) as error:
        print(
            json.dumps(
                {"status": "blocked", "error_type": type(error).__name__, "error": str(error)}
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
