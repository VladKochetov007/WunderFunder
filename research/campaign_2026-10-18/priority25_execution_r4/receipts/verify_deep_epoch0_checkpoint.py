#!/usr/bin/env python3
"""Bounded structural verification for the deep F0 epoch-0 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch


SOURCE_COMMIT = "a266b036b02e8e012c6eb31072291dad618a169d"
SOURCE_ARCHIVE_SHA256 = "288b4c9dd67b48114240bbf593901fc539de15aa709a1aa63388dccf64a4478a"
EXPECTED_SOURCE_FILES = 138
EXPECTED_PARAMETERS = 2_151_554
EXPECTED_STEPS_PER_EPOCH = 14_920
EXPECTED_EPOCHS = 6
EXPECTED_TOTAL_STEPS = EXPECTED_STEPS_PER_EPOCH * EXPECTED_EPOCHS
RUN_NAME = "rr18_r01_deep_f0_s1001"


def sha256_file(path: Path) -> str:
    """Hash one artifact with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    """Fail closed when a checkpoint invariant does not hold."""
    if not condition:
        raise ValueError(message)


def _tensor_values(value: object, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
    """Visit optimizer-state tensors without coercing its nested schema."""
    if isinstance(value, torch.Tensor):
        return [(prefix, value)]
    if isinstance(value, Mapping):
        return [
            item
            for key, child in value.items()
            for item in _tensor_values(child, f"{prefix}.{key}" if prefix else str(key))
        ]
    if isinstance(value, (list, tuple)):
        return [
            item
            for index, child in enumerate(value)
            for item in _tensor_values(child, f"{prefix}[{index}]")
        ]
    return []


def _assert_equal_contracts(
    wide_contract: Mapping[str, object], deep_contract: Mapping[str, object]
) -> None:
    """Require the pair to differ in architecture only."""
    wide_options = wide_contract["options"]
    deep_options = deep_contract["options"]
    require(isinstance(wide_options, Mapping), "wide contract has no options")
    require(isinstance(deep_options, Mapping), "deep contract has no options")
    changed = {
        key
        for key in set(wide_options) | set(deep_options)
        if wide_options.get(key) != deep_options.get(key)
    }
    require(changed == {"hidden", "layers"}, f"paired option drift: {sorted(changed)}")
    require(
        wide_options["hidden"] == 512 and wide_options["layers"] == 1,
        "wide architecture differs",
    )
    require(
        deep_options["hidden"] == 384 and deep_options["layers"] == 2,
        "deep architecture differs",
    )
    non_options = set(wide_contract) | set(deep_contract)
    non_options.discard("options")
    changed_contract = sorted(
        key for key in non_options if wide_contract.get(key) != deep_contract.get(key)
    )
    require(
        not changed_contract,
        f"paired non-architecture contract fields differ: {changed_contract}",
    )


def _verify_sources(source_root: Path, contract: Mapping[str, object]) -> int:
    """Hash every source file bound into the canonical continuation contract."""
    sources = contract.get("sources")
    require(isinstance(sources, Mapping), "contract has no source map")
    require(len(sources) == EXPECTED_SOURCE_FILES, "contract source count is not 138")
    for relative, expected in sources.items():
        path = (
            source_root / "scripts/train_rnn.py"
            if relative == "trainer"
            else source_root / "src/wnn" / str(relative)
        )
        require(path.is_file(), f"source file is missing: {relative}")
        require(sha256_file(path) == expected, f"source hash differs: {relative}")
    return len(sources)


def _verify_registry(contract: Mapping[str, object], registry_path: Path) -> str:
    """Match contract data identities to the previously sealed portability registry."""
    registry = json.loads(registry_path.read_text())
    entries = registry["sources"]
    datasets = contract["datasets"]
    require(isinstance(datasets, Mapping), "contract has no dataset identities")
    for split in ("train", "valid"):
        entry = entries[f"datasets/{split}.parquet"]
        require(
            datasets[split]["sha256"] == entry["sha256"]
            and datasets[split]["bytes"] == entry["original_identity"]["bytes"],
            f"{split} dataset identity differs from the sealed registry",
        )
    return sha256_file(registry_path)


def _verify_source_archive(manifest_path: Path) -> None:
    """Bind the extracted per-file checks to the frozen portable archive record."""
    manifest = json.loads(manifest_path.read_text())
    require(manifest["source_revision"] == "a266b036", "portable source revision differs")
    require(
        any(item["sha256"] == SOURCE_ARCHIVE_SHA256 for item in manifest["archives"]),
        "portable source archive SHA256 differs",
    )


def verify(args: argparse.Namespace) -> dict[str, object]:
    """Validate one collected epoch boundary without initializing CUDA."""
    source_root = args.source_root.resolve()
    snapshot = args.snapshot_dir.resolve()
    wide_snapshot = args.wide_snapshot_dir.resolve()
    sys.path.insert(0, str(source_root / "src"))
    from wnn.training_state import load_training_state

    collection = json.loads(args.collection_receipt.read_text())
    wide = json.loads(args.wide_verification.read_text())
    contract = load_training_state(snapshot / "training_state.pt")
    wide_state = load_training_state(wide_snapshot / "training_state.pt")
    state_contract = contract["contract"]
    require(contract["next_epoch"] == 1, "checkpoint is not epoch-0 boundary next_epoch=1")
    require(state_contract["epochs"] == EXPECTED_EPOCHS, "fixed epoch horizon differs")
    require(state_contract["total_steps"] == EXPECTED_TOTAL_STEPS, "fixed schedule steps differ")
    options = state_contract["options"]
    require(options["precision"] == "fp32", "fit is not declared FP32")
    require(options["seed"] == 1001 and options["valid_fold"] == 0, "fit cell identity differs")
    require(
        options["distill"] == "" and options["propensity"] == "",
        "fit consumed a teacher or propensity",
    )
    require(
        options["init"] == ""
        and not options["final_distill_initializer"]
        and not options["inherit_init_preprocessing"],
        "fit was not cold initialized",
    )
    require(state_contract["teacher_predictions"] == {}, "teacher predictions are present")
    require(state_contract["consumed_caches"] == {}, "auxiliary target caches are present")
    require(
        state_contract["ancestors"]["initial_checkpoint_sha256"] is None,
        "initial checkpoint ancestry is present",
    )
    require(
        state_contract["ancestors"]["propensity_checkpoint_sha256"] is None,
        "propensity ancestry is present",
    )
    require(state_contract["runtime"]["matmul_tf32"] is False, "CUDA matmul TF32 was enabled")
    require(state_contract["runtime"]["cudnn_tf32"] is False, "cuDNN TF32 was enabled")
    require(contract["scaler"] == {}, "FP32 scaler state is unexpectedly nonempty")
    require(contract["ema"] is None and contract["swa"] is None, "unexpected averaging state")
    require(
        wide["status"] == "passed" and wide["next_epoch"] == 1,
        "wide baseline receipt is not valid",
    )
    require(wide_state["next_epoch"] == 1, "wide checkpoint is not the epoch-0 boundary")

    source_count = _verify_sources(source_root, state_contract)
    _verify_source_archive(args.archive_manifest)
    require(
        state_contract["sources"]["trainer"] == sha256_file(source_root / "scripts/train_rnn.py"),
        "trainer entrypoint source binding differs",
    )
    registry_sha256 = _verify_registry(state_contract, args.data_registry)
    wide_collection = json.loads(args.wide_collection_receipt.read_text())
    require(
        wide_collection["run"] == "rr18_r01_wide_f0_s1001"
        and wide_collection["epoch_index"] == 0
        and wide_collection["source_snapshot_hash_match"] is True,
        "wide baseline collection is not a stable epoch-0 snapshot",
    )
    wide_checkpoint_sha = sha256_file(wide_snapshot / "training_state.pt")
    require(
        wide_checkpoint_sha == wide_collection["artifacts"]["training_state.pt"]["sha256"]
        == wide["artifacts"]["training_state.pt"]["sha256"],
        "wide training-state bytes do not match both baseline receipts",
    )
    _assert_equal_contracts(wide_state["contract"], state_contract)
    for identity_name in ("fit_membership", "normalization"):
        require(
            state_contract[identity_name] == wide_state["contract"][identity_name],
            f"paired {identity_name} identity differs",
        )
    artifacts = collection["artifacts"]
    require(
        collection["run"] == RUN_NAME and collection["epoch_index"] == 0,
        "collection cell differs",
    )
    require(collection["source_snapshot_hash_match"] is True, "collection was not stable")
    require(
        collection["source_stats_before"] == collection["source_stats_after"],
        "source changed during collection",
    )
    artifact_results: dict[str, dict[str, object]] = {}
    for name in ("model_ep0.pt", "provenance.json", "training_state.pt"):
        path = snapshot / name
        entry = artifacts[name]
        actual_sha = sha256_file(path)
        actual_size = path.stat().st_size
        require(actual_sha == entry["sha256"], f"collected {name} hash differs")
        require(actual_size == entry["bytes"], f"collected {name} size differs")
        artifact_results[name] = {"bytes": actual_size, "sha256": actual_sha}

    raw = torch.load(snapshot / "model_ep0.pt", map_location="cpu", weights_only=False)
    provenance = json.loads((snapshot / "provenance.json").read_text())
    require(
        provenance["fit_membership_identity"] == state_contract["fit_membership"],
        "external provenance fitting membership differs from contract",
    )
    require(
        provenance["normalization_identity"] == state_contract["normalization"],
        "external provenance normalization differs from contract",
    )
    for name in ("hidden", "layers", "seed", "valid_fold", "precision", "epochs"):
        require(provenance["args"][name] == options[name], f"provenance argument differs: {name}")
    raw_model = raw["model"]
    state_model = contract["model"]
    require(set(raw_model) == set(state_model), "raw and resume model tensor names differ")
    for name in raw_model:
        require(
            torch.equal(raw_model[name], state_model[name]),
            f"raw and resume tensor differs: {name}",
        )
    require(
        provenance["checkpoint_sha256"]["model_ep0.pt"]
        == artifact_results["model_ep0.pt"]["sha256"],
        "provenance does not bind model_ep0.pt",
    )
    for field in (
        "args",
        "trainer_implementation",
        "fit_membership_identity",
        "normalization_identity",
    ):
        if field in provenance:
            require(
                raw["provenance"].get(field) == provenance[field],
                f"embedded provenance differs: {field}",
            )

    model_tensors = _tensor_values(state_model)
    require(model_tensors, "checkpoint model has no tensors")
    model_finite_fp32 = all(
        tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all())
        for _, tensor in model_tensors
    )
    require(model_finite_fp32, "model tensors are not all finite FP32")
    parameter_count = sum(
        tensor.numel() for name, tensor in state_model.items() if name not in {"mean", "std"}
    )
    require(parameter_count == EXPECTED_PARAMETERS, f"parameter count differs: {parameter_count}")

    optimizer = contract["optimizer"]
    optimizer_states = optimizer["state"]
    optimizer_parameter_ids = [
        parameter_id for group in optimizer["param_groups"] for parameter_id in group["params"]
    ]
    require(
        len(optimizer_parameter_ids) == len(set(optimizer_parameter_ids)),
        "optimizer parameter IDs are not unique",
    )
    require(
        set(optimizer_states) == set(optimizer_parameter_ids),
        "optimizer state does not cover every trained parameter",
    )
    wide_optimizer_states = wide_state["optimizer"]["state"]
    require(
        len(wide_optimizer_states) == wide["optimizer_states_verified"],
        "wide baseline optimizer-state count differs from its verification receipt",
    )
    optimizer_tensors = [
        item
        for parameter_state in optimizer_states.values()
        for item in _tensor_values(parameter_state)
    ]
    floating_optimizer = [
        (name, tensor) for name, tensor in optimizer_tensors if tensor.is_floating_point()
    ]
    require(
        all(
            tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all())
            for _, tensor in floating_optimizer
        ),
        "floating optimizer state is not all finite FP32",
    )
    require(
        all(
            "exp_avg" in path or "exp_avg_sq" in path or path == "step"
            for path, _ in floating_optimizer
        ),
        "unexpected floating optimizer state field",
    )

    progress = contract["progress"]
    history = progress["history"]
    require(
        len(history) == 1 and history[0]["epoch"] == 0,
        "checkpoint history is not one completed epoch",
    )
    epoch = history[0]
    fit_summary = epoch["fit_summary"]
    update = epoch["update_diagnostics"]
    expected_counts = {
        "optimizer_attempts": EXPECTED_STEPS_PER_EPOCH,
        "amp_skipped_updates": 0,
        "optimizer_completed_updates": EXPECTED_STEPS_PER_EPOCH,
        "ema_updates": 0,
        "completed_batches": 1492,
    }
    require(
        update["epoch"] == expected_counts,
        "epoch update counts do not match the first full epoch",
    )
    require(progress["cumulative_updates"] == expected_counts, "cumulative update counts differ")
    grad_stats = update["grad_stats"]
    require(grad_stats["steps"] == EXPECTED_STEPS_PER_EPOCH, "gradient-stat steps differ")
    require(grad_stats["nonfinite_steps"] == 0, "nonfinite gradient steps were recorded")
    require(
        math.isfinite(float(fit_summary["objective_mean"])),
        "epoch mean fit objective is nonfinite",
    )
    require(
        fit_summary["optimizer_windows"] == EXPECTED_STEPS_PER_EPOCH,
        "fit summary window count differs",
    )
    scheduler = contract["scheduler"]
    require(scheduler["last_epoch"] == EXPECTED_STEPS_PER_EPOCH, "scheduler epoch differs")
    require(
        scheduler["_step_count"] == EXPECTED_STEPS_PER_EPOCH + 1,
        "scheduler step count differs",
    )
    require(not torch.cuda.is_initialized(), "CUDA initialized during CPU verification")
    require(
        all(tensor.device.type == "cpu" for _, tensor in model_tensors + optimizer_tensors),
        "loaded tensors are not on CPU",
    )

    return {
        "kind": "actual_epoch_checkpoint_structural_verification",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "passed",
        "run": RUN_NAME,
        "epoch_index": 0,
        "next_epoch": contract["next_epoch"],
        "snapshot_directory": str(snapshot),
        "collection_receipt_sha256": sha256_file(args.collection_receipt),
        "artifacts": artifact_results,
        "canonical_loader": "source-a266b036/src/wnn/training_state.py:load_training_state",
        "fitting_source_revision": SOURCE_COMMIT,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "wide_baseline_training_state_sha256": wide_checkpoint_sha,
        "wide_baseline_collection_receipt_sha256": sha256_file(args.wide_collection_receipt),
        "prior_verification_attempts": [
            {
                "attempt": 1,
                "unit": "computer-vision-homework-r01-deep-ep0-verify-20261018.service",
                "status": "failed_at_pair_contract_comparison",
                "reason": (
                    "The first attempt compared the deep full contract to the wide summary receipt, "
                    "which omits the source map. The comparison target was corrected to the "
                    "hash-bound wide training_state.pt."
                ),
            },
            {
                "attempt": 2,
                "unit": "computer-vision-homework-r01-deep-ep0-verify-r2-20261018.service",
                "status": "failed_at_optimizer_state_cardinality_check",
                "reason": (
                    "The first implementation reused the wide arm's ten optimizer-state count. "
                    "The deep arm adds a recurrent layer, so validation now checks complete "
                    "optimizer parameter-ID coverage instead of a wide-only count."
                ),
            }
        ],
        "source_files_verified": source_count,
        "split_manifest_sha256": options["split_manifest_sha256"],
        "dataset_identities": state_contract["datasets"],
        "fit_membership_identity": state_contract["fit_membership"],
        "normalization_identity": state_contract["normalization"],
        "model_tensors_verified": len(model_tensors),
        "trainable_parameter_count": parameter_count,
        "wide_trainable_parameter_count": wide["trainable_parameter_count"],
        "deep_minus_wide_parameters": parameter_count - wide["trainable_parameter_count"],
        "optimizer_states_verified": len(optimizer_states),
        "optimizer_floating_tensors_verified": len(floating_optimizer),
        "completed_update_counts": update["epoch"],
        "fixed_schedule_total_steps": state_contract["total_steps"],
        "scheduler_last_epoch": scheduler["last_epoch"],
        "scheduler_step_count": scheduler["_step_count"],
        "raw_checkpoint_resume_model_equal": True,
        "all_floating_model_and_adam_moments_finite_fp32": True,
        "normalization_and_membership_bindings_match": True,
        "cold_initializer_and_no_teacher": True,
        "dataset_registry_sha256": registry_sha256,
        "dataset_hashes_match_sealed_registry": True,
        "paired_contract_architecture_only": {
            "wide": {"hidden": 512, "layers": 1},
            "deep": {"hidden": 384, "layers": 2},
        },
        "epoch_summary": {
            "epoch": 0,
            "train_seconds": epoch["train_seconds"],
            "evaluation": "withheld",
            "fit_summary": fit_summary,
            "nonfinite_gradient_steps": grad_stats["nonfinite_steps"],
        },
        "cuda_initialized_during_verification": torch.cuda.is_initialized(),
        "support_limits": {
            "cpu_quota": "50%",
            "memory_max_bytes": 1_500_000_000,
            "memory_swap_max_bytes": 0,
            "runtime_max_seconds": 120,
            "gpu": False,
        },
        "limitations": [
            "Structural/hash/finite-state checks only; no GPU continuation was executed, so "
            "resume parity is unmeasured.",
            "Raw parquet files were not rehashed; dataset bytes and SHA256 were matched to the "
            "previously sealed registry.",
            "This is one intermediate epoch of one arm, not a fixed endpoint, independent grade, "
            "CP30 comparison or convergence result.",
        ],
    }


def main() -> None:
    """Run the isolated checkpoint verification and emit one JSON receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--wide-snapshot-dir", type=Path, required=True)
    parser.add_argument("--collection-receipt", type=Path, required=True)
    parser.add_argument("--wide-verification", type=Path, required=True)
    parser.add_argument("--wide-collection-receipt", type=Path, required=True)
    parser.add_argument("--data-registry", type=Path, required=True)
    parser.add_argument("--archive-manifest", type=Path, required=True)
    args = parser.parse_args()
    result = json.dumps(verify(args), indent=2, allow_nan=False) + "\n"
    output = Path(__file__).with_name("RANK01_DEEP_F0_S1001_EP0_VERIFICATION.json")
    output.write_text(result)


if __name__ == "__main__":
    main()
