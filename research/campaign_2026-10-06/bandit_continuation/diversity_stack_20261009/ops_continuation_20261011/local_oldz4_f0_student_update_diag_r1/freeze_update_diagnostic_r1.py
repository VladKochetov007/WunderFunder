from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAIN_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
PROFILE_PACKET = (
    MAIN_ROOT
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009"
    / "ops_continuation_20261011/local_oldz4_f0_student_profile_r2"
)
BASE_CONTRACT = PROFILE_PACKET / "ROOT_ADMITTED_PROFILE_CONTRACT_R2.json"
PREFIX_RUN = "oldz4_h230_student_update_diag_f0_s3_r1"
RELOAD_RUN = "oldz4_h230_student_update_reload_f0_s3_r1"
UPDATE_COUNTER_SHA256 = "7033ea007b9c7dcf498d3ab1e31aaeacb5725b7009fe719e9071c05b99272a76"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def find_flag(command: list[str], flag: str) -> int:
    indexes = [index for index, value in enumerate(command) if value == flag]
    if len(indexes) != 1:
        raise ValueError(f"expected exactly one {flag}, found {len(indexes)}")
    return indexes[0]


def replace_value(command: list[str], flag: str, value: str) -> None:
    index = find_flag(command, flag)
    if index + 1 >= len(command):
        raise ValueError(f"{flag} has no value")
    command[index + 1] = value


def remove_flag_pair(command: list[str], flag: str) -> None:
    index = find_flag(command, flag)
    if index + 1 >= len(command):
        raise ValueError(f"{flag} has no value")
    del command[index : index + 2]


def build_stage_command(base_command: list[str], run_name: str) -> list[str]:
    command = list(base_command)
    replace_value(command, "--name", run_name)
    replace_value(command, "--epochs", "1")
    replace_value(command, "--checkpoint-epoch", "0")
    remove_flag_pair(command, "--profile-warmup")
    remove_flag_pair(command, "--profile-steps")
    command.extend(["--max-batches", "2", "--save-training-state"])
    return command


def stage_output(run_name: str) -> Path:
    return MAIN_ROOT / "artifacts/runs" / run_name


def build_contract(base: dict[str, Any]) -> dict[str, Any]:
    expected_command = base["profile"]["expected_command"]
    if not isinstance(expected_command, list):
        raise ValueError("base profile command is not a list")
    command = [str(part) for part in expected_command]
    required_environment = (
        "WNN_LOADER_WORKERS=0",
        "WNN_THREADS=1",
        "WNN_GPU_MEMORY_BYTES=4300000000",
    )
    if (
        "600s" not in command
        or "--profile-steps" not in command
        or any(value not in command for value in required_environment)
    ):
        raise ValueError("base profile command differs from the reviewed R2 resource/argv shape")

    prefix_command = build_stage_command(command, PREFIX_RUN)
    reload_command = build_stage_command(command, RELOAD_RUN)
    reload_command.extend(["--resume-state", str(stage_output(PREFIX_RUN) / "training_state.pt")])

    input_paths = {
        "executed_trainer": base["inputs"]["executed_trainer"],
        "optimizer_module": str(
            Path(base["inputs"]["executed_trainer"]).parents[1] / "src/wnn/optim.py"
        ),
        "update_counter_module": str(
            Path(base["inputs"]["executed_trainer"]).parents[1] / "src/wnn/run_identity.py"
        ),
        "source_manifest": base["inputs"]["source_manifest"],
        "source_snapshot_bindings": base["inputs"]["source_bindings"],
        "input_bindings": base["inputs"]["input_bindings"],
        "split_manifest": base["inputs"]["split_manifest"],
        "parent_checkpoint": base["inputs"]["initializer"]["checkpoint"],
        "propensity_checkpoint": base["inputs"]["propensity_model"],
        "cache_metadata": str(Path(base["inputs"]["cache_root"]) / "metadata.json"),
    }
    input_hashes = {name: sha256_file(Path(path)) for name, path in input_paths.items()}
    declared_hashes = {
        "executed_trainer": base["inputs"]["executed_trainer_sha256"],
        "optimizer_module": base["inputs"]["optim_sha256"],
        "update_counter_module": UPDATE_COUNTER_SHA256,
        "source_manifest": base["inputs"]["source_manifest_sha256"],
        "source_snapshot_bindings": base["inputs"]["source_bindings_sha256"],
        "input_bindings": base["inputs"]["input_bindings_sha256"],
        "split_manifest": base["inputs"]["split_manifest_sha256"],
        "parent_checkpoint": base["inputs"]["initializer"]["sha256"],
        "propensity_checkpoint": base["inputs"]["propensity_sha256"],
        "cache_metadata": base["inputs"]["cache_metadata_sha256"],
    }
    if input_hashes != declared_hashes:
        raise ValueError(f"R2 source/input digest mismatch: {input_hashes} != {declared_hashes}")

    return {
        "schema": "wnn.oldz4_student_update_diagnostic_contract.r1",
        "status": "prepared_not_admitted",
        "launch_authorized": False,
        "fit_authorized": False,
        "base_profile_contract": {
            "path": str(BASE_CONTRACT),
            "sha256": sha256_file(BASE_CONTRACT),
            "attempt_id": base["attempt_id"],
            "expected_command_sha256": canonical_sha256(command),
        },
        "source_snapshot": {
            "worktree": base["inputs"]["source_worktree"],
            "commit": base["inputs"]["source_worktree_commit"],
            "manifest_sha256": base["inputs"]["source_manifest_sha256"],
            "snapshot_bindings_sha256": base["inputs"]["source_bindings_sha256"],
            "trainer_sha256": base["inputs"]["executed_trainer_sha256"],
            "optimizer_sha256": base["inputs"]["optim_sha256"],
            "update_counter_sha256": UPDATE_COUNTER_SHA256,
            "input_hashes": input_hashes,
        },
        "recipe_and_data": {
            "interpretation": "Two-minibatch finite-update/checkpoint/reload diagnostic only; changed horizon and one-cycle schedule, not a scientific recipe score or production student fit.",
            "parent_run": base["inputs"]["initializer"]["run"],
            "parent_checkpoint": base["inputs"]["initializer"]["checkpoint"],
            "parent_checkpoint_sha256": base["inputs"]["initializer"]["sha256"],
            "parent_epoch": base["inputs"]["initializer"]["epoch"],
            "split_manifest": base["inputs"]["split_manifest"],
            "split_manifest_sha256": base["inputs"]["split_manifest_sha256"],
            "membership": base["inputs"]["membership"],
            "cache_metadata": base["inputs"]["cache_metadata"],
            "cache_metadata_sha256": base["inputs"]["cache_metadata_sha256"],
            "cache_ancestry": base["inputs"]["cache_ancestry"],
            "propensity_model": base["inputs"]["propensity_model"],
            "propensity_sha256": base["inputs"]["propensity_sha256"],
            "seed": 3,
            "batch_size": 16,
            "chunk_rows": 2000,
            "batches": 2,
            "optimizer_attempts_expected": 20,
            "fit_evaluation": "disabled",
        },
        "resources": {
            "cpu_reservation_cores": 2,
            "memory_reservation_bytes": 6_500_000_000,
            "memory_high_bytes": 6_100_000_000,
            "memory_max_bytes": 6_500_000_000,
            "memory_swap_max_bytes": 0,
            "parent_memory_high_bytes": 18_000_000_000,
            "parent_memory_max_bytes": 20_000_000_000,
            "timeout_seconds": 600,
            "timeout_kill_grace_seconds": 20,
            "gpu_allocator_bytes": 4_300_000_000,
            "observed_vram_ceiling_bytes": 5_000_000_000,
            "loader_workers": 0,
            "threads": 1,
        },
        "allowed_profile_to_diagnostic_changes": [
            "rename to a fresh diagnostic run",
            "set epochs=1 and fixed checkpoint epoch=0",
            "remove profile warmup/steps and use max-batches=2",
            "enable full training-state save",
        ],
        "stages": {
            "update_prefix": {
                "attempt_id": "oldz4-r3-f0-s3-student-update-diag-r1",
                "run_name": PREFIX_RUN,
                "output_dir": str(stage_output(PREFIX_RUN)),
                "command": prefix_command,
                "must_not_exist_before_admission": str(stage_output(PREFIX_RUN)),
            },
            "no_op_reload": {
                "attempt_id": "oldz4-r3-f0-s3-student-update-reload-r1",
                "run_name": RELOAD_RUN,
                "output_dir": str(stage_output(RELOAD_RUN)),
                "command": reload_command,
                "resume_input": str(stage_output(PREFIX_RUN) / "training_state.pt"),
                "must_not_exist_before_admission": str(stage_output(RELOAD_RUN)),
            },
        },
        "producer_checks_required_before_fit_admission": [
            "exact source/data/parent/cache/propensity hashes match this contract",
            "fit membership excludes all 468 held-fold groups",
            "profile command resolves the frozen research/.../scripts/train_rnn.py entrypoint and its pinned PYTHONPATH",
            "both commands keep all scientific/data arguments identical to R2 except the four declared diagnostic changes",
        ],
        "terminal_acceptance": {
            "prefix": [
                "optimizer_attempts == 20 and completed_batches == 2",
                "amp_skipped_updates and optimizer_completed_updates are explicit integer counters whose sum is 20",
                "all saved model, optimizer, scheduler, scaler and RNG tensor state is finite",
                "fixed model_ep0.pt and training_state.pt exist with hashes and source/provenance bindings",
                "no new MemoryMax/OOM/OOM-kill event; observed VRAM is within the stated ceiling",
            ],
            "reload": [
                "resume-contract exact-match succeeds with the same one-epoch/two-batch arguments and source/input bindings",
                "no new optimizer updates occur because saved next_epoch == epochs == 1",
                "reloaded fixed model_ep0.pt tensors equal the prefix model_ep0.pt tensors",
                "reloaded state and checkpoint are finite and have complete hashes/provenance",
            ],
            "nonfinite_gradient_norms": "Preserve the count and compare with explicit AMP skip counters; do not infer a skip from GradStats alone. Root reviews any mismatch.",
            "not_a_pass": "Process exit 0 alone, a profile row, or a checkpoint without explicit update counts and exact reload proof.",
        },
        "prepared_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def validate(contract_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    base = json.loads(BASE_CONTRACT.read_text(encoding="utf-8"))
    expected = build_contract(base)
    expected.pop("prepared_utc")
    observed = dict(contract)
    observed.pop("prepared_utc", None)
    if observed != expected:
        raise ValueError("frozen contract differs from source-bound derivation")
    if contract["launch_authorized"] or contract["fit_authorized"]:
        raise ValueError("preparation packet must remain unadmitted")
    runtime_attempts = (
        MAIN_ROOT / "research/campaign_2026-10-06/bandit_continuation/runtime/attempts"
    )
    for stage in contract["stages"].values():
        output = Path(stage["output_dir"])
        if output.exists():
            raise ValueError(f"diagnostic output already exists: {output}")
        attempt_receipt = runtime_attempts / f"{stage['attempt_id']}.json"
        if attempt_receipt.exists():
            raise ValueError(f"diagnostic attempt ID is already recorded: {attempt_receipt}")
    return {
        "schema": "wnn.oldz4_student_update_diagnostic_static_validation.r1",
        "contract_sha256": sha256_file(contract_path),
        "base_contract_sha256": sha256_file(BASE_CONTRACT),
        "source_hashes_verified": len(contract["source_snapshot"]["input_hashes"]),
        "prefix_argv_sha256": canonical_sha256(contract["stages"]["update_prefix"]["command"]),
        "reload_argv_sha256": canonical_sha256(contract["stages"]["no_op_reload"]["command"]),
        "fit_authorized": False,
        "launch_authorized": False,
        "static_result": "PASS_NOT_ADMITTED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--freeze", action="store_true", help="write a fresh unadmitted R1 command contract"
    )
    mode.add_argument(
        "--check", action="store_true", help="rederive and validate the frozen R1 contract"
    )
    args = parser.parse_args()
    contract_path = Path(__file__).with_name("ROOT_STUDENT_UPDATE_DIAGNOSTIC_CONTRACT_R1.json")
    if args.freeze:
        if contract_path.exists():
            raise SystemExit(f"refusing to overwrite frozen contract: {contract_path}")
        base = json.loads(BASE_CONTRACT.read_text(encoding="utf-8"))
        contract = build_contract(base)
        contract_path.write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    receipt = validate(contract_path)
    receipt_path = Path(__file__).with_name("STATIC_VALIDATION_R1.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
