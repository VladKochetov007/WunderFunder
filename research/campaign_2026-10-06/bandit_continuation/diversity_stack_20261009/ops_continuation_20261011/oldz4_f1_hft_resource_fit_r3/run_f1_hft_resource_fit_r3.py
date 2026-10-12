from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

# JSON receipts have heterogeneous nested values, so `Any` is unavoidable here.
from typing import Any

PROJECT_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
OPS_ROOT = PROJECT_ROOT / (
    "research/campaign_2026-10-06/bandit_continuation/"
    "diversity_stack_20261009/ops_continuation_20261011"
)
R2_PACKET = OPS_ROOT / "oldz4_f1_core_hft_r2"
PACKET_ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = PACKET_ROOT / "HFT_RESOURCE_ONLY_FIT_CONTRACT_R3.json"
PROFILE_EVIDENCE_PATH = PACKET_ROOT / "F1_HFT_PROFILE_RESOURCE_EVIDENCE_R3.json"
ROOT_RELEASE_PATH = PACKET_ROOT / "ROOT_F1_HFT_RESOURCE_ONLY_FIT_ADMISSION_R3.json"
PROFILE_ATTEMPT_PATH = PROJECT_ROOT / (
    "research/campaign_2026-10-06/bandit_continuation/runtime/attempts/"
    "oldz4-r2-f1-s3-hft-profile.json"
)
PROFILE_LOG_PATH = PROJECT_ROOT / (
    "research/campaign_2026-10-06/bandit_continuation/runtime/logs/oldz4-r2-f1-s3-hft-profile.log"
)


def sha256_file(path: Path) -> str:
    """Hash a file incrementally before accepting a root resource-only release."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject non-object evidence."""

    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"non-finite JSON token is forbidden: {token}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    """Fail closed on any missing authorization, changed input, or profile fact."""
    if not condition:
        raise ValueError(message)


def canonical_equal(left: Any, right: Any) -> bool:
    """Compare JSON evidence without Python's bool/int equality ambiguity."""
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def project_path(path: str | Path) -> Path:
    """Resolve a bound R2 path from the project root, independent of cwd."""
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def import_runner(path: Path) -> ModuleType:
    """Import the exact frozen R2 stage consumer without writing bytecode."""
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("PYTHONDONTWRITEBYTECODE=1 is required")
    spec = importlib.util.spec_from_file_location("oldz4_f1_hft_r2_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import frozen R2 runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_profile_evidence(contract: dict[str, Any], evidence: dict[str, Any]) -> None:
    """Bind terminal attempt, raw log, strict failure, and measured resources."""
    pins = contract["hashes"]
    evidence_hashes = evidence.get("hashes")
    require(
        isinstance(evidence_hashes, dict)
        and evidence_hashes
        == {key: value for key, value in pins.items() if key != "profile_resource_evidence_sha256"},
        "profile evidence hash map differs from the resource-only contract",
    )
    require(
        contract.get("profile_evidence_sha256") == pins["profile_resource_evidence_sha256"],
        "contract's profile evidence digest fields disagree",
    )
    require(
        sha256_file(PROFILE_ATTEMPT_PATH) == pins["profile_attempt_receipt_sha256"],
        "profile attempt receipt hash changed",
    )
    require(
        sha256_file(PROFILE_LOG_PATH) == pins["profile_log_sha256"], "raw profile log hash changed"
    )
    require(
        sha256_file(PROFILE_EVIDENCE_PATH) == pins["profile_resource_evidence_sha256"],
        "profile evidence receipt hash changed",
    )
    launch_path = R2_PACKET / "ROOT_F1_HFT_PROFILE_LAUNCH_R2.json"
    require(
        sha256_file(launch_path) == pins["profile_launch_receipt_sha256"],
        "root profile launch receipt hash changed",
    )
    launch = read_object(launch_path)
    require(
        launch.get("schema") == "wnn.oldz4-hft-profile-launch.v1"
        and launch.get("decision") == "PASS",
        "actual profile launch receipt is not valid",
    )
    require(
        launch.get("profile_run") == evidence["profile_run"],
        "profile launch receipt names a different run",
    )
    require(
        launch.get("profile_contract_sha256") == pins["profile_contract_sha256"],
        "profile launch receipt binds a different profile contract",
    )
    require(
        launch.get("core_crop_receipt_sha256") == pins["core_crop_receipt_sha256"],
        "profile launch receipt binds a different crop receipt",
    )
    require(
        launch.get("profile_success_or_fit_admission_claim") is False,
        "profile launch receipt must not misstate strict profile success",
    )

    attempt = read_object(PROFILE_ATTEMPT_PATH)
    require(attempt.get("id") == "oldz4-r2-f1-s3-hft-profile", "unexpected profile attempt")
    require(
        attempt.get("status") == "failed" and attempt.get("phase") == "terminal",
        "R2 profile is not the recorded failed terminal attempt",
    )
    require(attempt.get("exit_status") == 1, "R2 profile exit status changed")
    require(
        attempt.get("finished_utc") == evidence.get("profile_finished_utc")
        and attempt.get("started_utc") == evidence.get("profile_started_utc"),
        "profile terminal timestamps differ from the frozen resource evidence",
    )
    require(
        attempt.get("contract_sha256") == pins["profile_contract_sha256"],
        "profile attempt used another profile contract",
    )
    source_hashes = attempt.get("source_sha256", {})
    require(
        source_hashes.get(str(R2_PACKET.relative_to(PROJECT_ROOT) / "HFT_PROFILE_CONTRACT_R2.json"))
        == pins["profile_contract_sha256"],
        "profile attempt source map differs at profile contract",
    )
    require(
        source_hashes.get(str(R2_PACKET.relative_to(PROJECT_ROOT) / "run_f1_core_hft_stage_r2.py"))
        == pins["r2_runner_sha256"],
        "profile attempt source map differs at stage runner",
    )
    require(
        source_hashes.get("scripts/train_rnn.py") == pins["trainer_sha256"],
        "profile attempt source map differs at the C337 trainer",
    )

    log = PROFILE_LOG_PATH.read_text(encoding="utf-8")
    line = next(
        (entry for entry in log.splitlines() if entry.startswith("PROFILE_UPDATE_EVIDENCE ")), None
    )
    require(line is not None, "raw profile log lacks PROFILE_UPDATE_EVIDENCE")
    observed = json.loads(line[len("PROFILE_UPDATE_EVIDENCE ") :])
    require(
        canonical_equal(observed, evidence["profile_update_evidence"]),
        "parsed profile evidence differs from frozen evidence receipt",
    )
    require(
        "RuntimeError: profile update evidence is incomplete or nonfinite" in log,
        "raw log no longer records the strict profile guard failure",
    )
    expected = contract["profile_evidence_expected"]
    require(
        canonical_equal({key: observed.get(key) for key in expected}, expected),
        "profile update counters differ from the recorded 268/270 resource-only case",
    )
    require(
        observed.get("gradient_finite") is False
        and observed.get("gradient_stats_nonfinite_steps") == 2
        and observed.get("valid_for_throughput") is False,
        "resource-only evidence must preserve the failed strict numerical gate",
    )
    observed_resources = {
        "peak_worker_ram_bytes": attempt.get("peak_ram_bytes"),
        "peak_observed_vram_bytes": attempt.get("peak_observed_vram_bytes"),
        "worker_reservation": attempt.get("reservation"),
        "memory_event_deltas": attempt.get("memory_event_deltas"),
        "final_memory_event_deltas": attempt.get("final_memory_event_deltas"),
        "parent_memory_event_deltas": attempt.get("final_parent_memory_event_deltas"),
        "parent_hierarchical_memory_event_deltas": attempt.get(
            "final_parent_hierarchical_memory_event_deltas"
        ),
        "resource_exit_status": attempt.get("exit_status"),
    }
    require(
        canonical_equal(evidence.get("profile_resource_observations"), observed_resources),
        "raw profile resource observations differ from the frozen evidence receipt",
    )
    observations = evidence["profile_resource_observations"]
    limits = contract["resource_limits"]
    require(
        canonical_equal(observations["worker_reservation"], {"cpus": 1, "ram_bytes": 6800000000}),
        "actual profile worker reservation differs from the reviewed fit envelope",
    )
    require(
        observations["peak_worker_ram_bytes"] <= limits["worker_memory_max_bytes"],
        "profile worker RAM exceeded the fit reservation",
    )
    require(
        observations["peak_observed_vram_bytes"] <= limits["allocator_vram_bytes"],
        "profile VRAM exceeded the fit allocator budget",
    )
    for source in (
        observations["memory_event_deltas"],
        observations["final_memory_event_deltas"],
        observations["parent_memory_event_deltas"],
        observations["parent_hierarchical_memory_event_deltas"],
    ):
        expected_event_keys = {
            "high",
            "low",
            "max",
            "oom",
            "oom_group_kill",
            "oom_kill",
            "sock_throttled",
        }
        require(
            isinstance(source, dict)
            and set(source) == expected_event_keys
            and all(type(value) is int and value == 0 for value in source.values()),
            "profile memory-event delta is nonzero or malformed",
        )


def verify_bound_files(contract: dict[str, Any]) -> None:
    """Rehash every frozen R2 input and source immediately before dispatch."""
    bindings = read_object(R2_PACKET / "SOURCE_INPUT_BINDINGS_R2.json")
    r4 = bindings["r4"]
    c337 = bindings["c337_source"]
    source_manifest_path = Path(c337["snapshot_bindings_path"])
    if not source_manifest_path.is_absolute():
        source_manifest_path = PROJECT_ROOT / source_manifest_path
    source_manifest = read_object(source_manifest_path)
    hft_root = Path(source_manifest["hft"]["snapshot_root"])
    if not hft_root.is_absolute():
        hft_root = PROJECT_ROOT / hft_root
    paths = {
        "r2_packet_manifest_sha256": R2_PACKET / "PACKET_FILE_MANIFEST_R2B.json",
        "r2_runner_sha256": R2_PACKET / "run_f1_core_hft_stage_r2.py",
        "source_input_bindings_sha256": R2_PACKET / "SOURCE_INPUT_BINDINGS_R2.json",
        "r4_input_bindings_sha256": project_path(r4["input_bindings_path"]),
        "r4_source_bindings_sha256": project_path(r4["source_bindings_path"]),
        "source_snapshot_bindings_sha256": project_path(c337["snapshot_bindings_path"]),
        "trainer_sha256": hft_root / "scripts/train_rnn.py",
        "profile_contract_sha256": R2_PACKET / "HFT_PROFILE_CONTRACT_R2.json",
        "fit_contract_sha256": R2_PACKET / "HFT_FIT_CONTRACT_R2.json",
        "export_contract_sha256": R2_PACKET / "HFT_EXPORT_CONTRACT_R2.json",
        "core_crop_contract_sha256": R2_PACKET / "CORE_CROP_CONTRACT_R2.json",
        "core_crop_receipt_sha256": R2_PACKET / "CORE_CROP_PRODUCER_RECEIPT_R2.json",
        "profile_launch_receipt_sha256": R2_PACKET / "ROOT_F1_HFT_PROFILE_LAUNCH_R2.json",
        "profile_attempt_receipt_sha256": PROFILE_ATTEMPT_PATH,
        "profile_log_sha256": PROFILE_LOG_PATH,
        "profile_resource_evidence_sha256": PROFILE_EVIDENCE_PATH,
        "student_cp1_terminal_gate_sha256": R2_PACKET / "ROOT_F1_STUDENT_CP1_TERMINAL_GATE_R2.json",
    }
    pins = contract["hashes"]
    for key, path in paths.items():
        require(
            path.is_file() and sha256_file(path) == pins[key], f"frozen R2 input changed: {key}"
        )


def check_root_release(
    contract: dict[str, Any], evidence: dict[str, Any], release: dict[str, Any], release_sha: str
) -> None:
    """Require the separate root authorization and every frozen source binding."""
    hashes = contract["hashes"]
    expected_release_hashes = {
        **hashes,
        "resource_contract_sha256": sha256_file(CONTRACT_PATH),
        "resource_adapter_sha256": sha256_file(Path(__file__).resolve()),
    }
    required_fields = set(contract["required_root_release_fields"])
    require(
        required_fields <= set(release),
        f"root release is missing required fields: {sorted(required_fields - set(release))}",
    )
    require(
        release_sha == sha256_file(ROOT_RELEASE_PATH),
        "root release SHA does not match the supplied binding",
    )
    require(
        release.get("schema") == contract["root_release"]["schema"],
        "unexpected root resource-release schema",
    )
    require(
        release.get("decision") == "PASS_FOR_ONE_BOUNDED_ORIGINAL_FIT",
        "root did not explicitly admit one original fit",
    )
    require(release.get("attempt") == contract["attempt"], "root release names another fit attempt")
    require(release.get("fit_run") == contract["fit_run"], "root release names another fit run")
    require(
        release.get("profile_attempt_id") == evidence["profile_attempt_id"],
        "root release names another terminal resource profile",
    )
    require(
        release.get("fit_argv_unchanged") is True,
        "root release does not affirm unchanged original fit argv",
    )
    require(
        release.get("profile_strict_gate_passed") is False,
        "root release must preserve strict profile failure",
    )
    require(
        release.get("profile_valid_for_throughput") is False,
        "root release must not claim throughput validity",
    )
    for key, value in expected_release_hashes.items():
        require(release.get(key) == value, f"root release hash binding mismatch: {key}")
    require(
        canonical_equal(
            release.get("profile_update_evidence"), contract["profile_evidence_expected"]
        ),
        "root release profile evidence differs from the failed actual profile",
    )
    require(
        canonical_equal(
            release.get("profile_resource_observations"), evidence["profile_resource_observations"]
        ),
        "root release resource evidence differs from the actual profile",
    )
    require(
        canonical_equal(release.get("resource_limits"), contract["resource_limits"]),
        "root release resource limits differ from the prepared fit contract",
    )
    require(
        all(
            type(release["resource_limits"][key]) is int
            for key in (
                "cpu_cores",
                "worker_memory_max_bytes",
                "allocator_vram_bytes",
                "runtime_max_seconds",
                "loader_workers",
                "threads",
                "swap_bytes",
            )
        ),
        "root release resource limits must use exact integers",
    )


def check_worker_environment(contract: dict[str, Any]) -> None:
    """Ensure the admitted worker retains the recipe's bounded loader/runtime setup."""
    expected = {
        "WNN_SLICE": "wnn-campaign.slice",
        "WNN_CPU_QUOTA": "100%",
        "WNN_MEM_MAX": "6800000000",
        "WNN_RUNTIME_MAX": "1800s",
        "WNN_LOADER_WORKERS": "0",
        "WNN_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "WNN_GPU_MEMORY_BYTES": str(contract["resource_limits"]["allocator_vram_bytes"]),
        "WNN_LOADER_RAM_GB": "6",
    }
    missing = {
        key: os.environ.get(key) for key, value in expected.items() if os.environ.get(key) != value
    }
    require(not missing, f"resource-only worker environment differs: {missing}")


def main() -> None:
    """Dispatch one unchanged R2 fit only under a new resource-only root release."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-release-sha256", required=True)
    args = parser.parse_args()
    require(len(args.root_release_sha256) == 64, "root release SHA must be 64 hex characters")
    require(
        all(char in "0123456789abcdef" for char in args.root_release_sha256),
        "root release SHA must be lowercase hex",
    )

    contract = read_object(CONTRACT_PATH)
    evidence = read_object(PROFILE_EVIDENCE_PATH)
    release = read_object(ROOT_RELEASE_PATH)
    verify_bound_files(contract)
    check_root_release(contract, evidence, release, args.root_release_sha256)
    check_profile_evidence(contract, evidence)
    check_worker_environment(contract)

    output_dir = PROJECT_ROOT / "artifacts/runs" / contract["fit_run"]
    require(not output_dir.exists(), f"refuse to overwrite an existing fit output: {output_dir}")

    runner_path = R2_PACKET / "run_f1_core_hft_stage_r2.py"
    require(
        sha256_file(runner_path) == contract["hashes"]["r2_runner_sha256"],
        "frozen R2 stage consumer changed",
    )
    require(
        sha256_file(R2_PACKET / "PACKET_FILE_MANIFEST_R2B.json")
        == contract["hashes"]["r2_packet_manifest_sha256"],
        "R2B packet manifest changed",
    )
    require(
        sha256_file(R2_PACKET / "HFT_FIT_CONTRACT_R2.json")
        == contract["hashes"]["fit_contract_sha256"],
        "original R2 fit contract changed",
    )

    runner = import_runner(runner_path)
    runner._require_environment()
    bindings = runner.verify_packet_bindings()
    fit_contract = runner.read_object(R2_PACKET / "HFT_FIT_CONTRACT_R2.json")
    require(fit_contract["run"] == contract["fit_run"], "original fit run differs from R3 contract")
    require(fit_contract.get("fit_authorized") is False, "preserve the original R2 strict fit gate")
    argv_hash = hashlib.sha256(
        json.dumps(fit_contract["trainer_argv"], separators=(",", ":")).encode()
    ).hexdigest()
    require(
        argv_hash == contract["hashes"]["fit_trainer_argv_sha256"], "original R2 fit argv changed"
    )
    require(
        bindings["trainer_sha256"] == contract["hashes"]["trainer_sha256"],
        "actual trainer source differs from resource-only contract",
    )
    crop_receipt, crop_receipt_sha = runner.validate_core_receipt()
    require(
        crop_receipt_sha == contract["hashes"]["core_crop_receipt_sha256"],
        "actual R2 crop receipt changed",
    )
    require(
        crop_receipt.get("status") == "PASS_ACTUAL_CORE_CROP", "actual crop receipt is not valid"
    )
    trainer, trainer_path = runner.import_trainer()
    require(
        sha256_file(trainer_path) == contract["hashes"]["trainer_sha256"],
        "imported C337 trainer source changed",
    )
    membership = runner.validate_inputs(trainer)
    require(
        membership.get("source_qualified") is True
        and membership.get("outer_evaluation_fit_overlap") == 0,
        "F1 source-qualified membership gate failed",
    )

    require(fit_contract["run"] == contract["fit_run"], "validated fit output run changed")
    previous_argv = sys.argv[:]
    try:
        sys.argv = [str(trainer_path), *fit_contract["trainer_argv"]]
        trainer.main()
    finally:
        sys.argv = previous_argv
    print(
        json.dumps(
            {
                "status": "DISPATCHED_ORIGINAL_FIT_RESOURCE_ONLY",
                "attempt": contract["attempt"],
                "fit_run": contract["fit_run"],
                "strict_profile_gate_passed": False,
                "profile_valid_for_throughput": False,
                "resource_only_root_release_sha256": args.root_release_sha256,
                "crop_receipt_sha256": crop_receipt_sha,
                "membership": membership,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
