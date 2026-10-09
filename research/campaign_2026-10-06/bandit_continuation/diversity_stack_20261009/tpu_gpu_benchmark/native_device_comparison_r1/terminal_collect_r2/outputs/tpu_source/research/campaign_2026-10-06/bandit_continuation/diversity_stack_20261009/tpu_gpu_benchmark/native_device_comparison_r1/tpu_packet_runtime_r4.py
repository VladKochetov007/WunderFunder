"""R4 TPU consumer with fail-closed full-profile fit headroom admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fit_headroom_gate_r4 import (
    EXPECTED_TPU_DEVICE_COUNT,
    evaluate_actual_device_inventory,
    evaluate_profile_headroom,
    sample_live_resources,
    validate_policy,
)

CACHE_MANIFEST_SHA = "7400b75504f592011577dd3b126e9813749c21ce9126e5c4fa04c8e6b61e8f3b"
CACHE_KEY = "949587f77794935c08589bd5d5e81641062de547526b92fc1b01bc4109cd82a6"
MEMBERSHIP_SHA = "dbd9ea6d7b0b752495745ea1ebd4bd858af9305c40f0ada7de7386c5359f3acd"
SPLIT_SHA = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
REGISTRY_SHA = "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c"
CPU_KERNEL = "vladkochetov07/wnn-parsed-cache-source-stage-20261009-v1"
R4_FIT_KERNEL = "vladkochetov07/wnn-loss-pair-gpu-20261009-v4"
GPU_KERNEL = "vladkochetov07/wnn-loss-pair-export-recovery-20261009-r1"
R4_TERMINAL_RECEIPT_SHA = "1bed2d7befbdd25b487f15f0b8788b8e514783058a2c7d5f683b31ff70a7500c"
R4_FIT_REVIEW_SHA = "701987320ad045a4688e633cef19d59fb8b940cb1a75f53b089deaeb0a070e47"
R5_RECOVERY_ASSET_SHA = "0d9d5b6d53017b8ef837c369d7aaa47ce2a676c37a1165cda71d6735b446bcfd"
R5_ROOT_REVIEW_SHA = "60e9770de432a6b39d3ae41c818d60b2d47b5298f630b33ee3483a0e7fecc7f9"
R4_ORIGINAL_CACHE_RECEIPT_SHA = "f6452f4b03abf7c66a219008458e431a8c1b6a2ffba961f63e5f57d1e936076f"
R5_PREDICTOR_ORIGINAL_SHA = "74d1eac17add3f9461e1205bac0eb5b6a78facd567898d679b15bdfa5bed4f5c"
R4_CODE_DATASET = "vladkochetov07/wnn-loss-pair-code-20261009-r4"
REQUIRED_SHARED_ROLES = {
    "cache_manifest",
    "cache_validation_receipt",
    "cache_reader_profile",
    "source_stage_receipt",
    "source_verification",
    "initial_state",
    "normalization",
    "normalization_metadata",
    "sample_orders",
    "onecycle_trace",
    "profile_inputs",
    "profile_inputs_receipt",
    "device_profile_gate",
    "fold0_asset_validation_receipt",
}
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
# Heterogeneous historical producer records and JSON artifact contracts.
Record = dict[str, Any]


def digest(path: Path) -> str:
    """Hash a bounded stream without loading mounted arrays into RAM."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> Record:
    """Reject non-object contract artifacts at the boundary."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, record: Record) -> None:
    """Preserve each phase/admission/output identity as strict JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def discover_pinned(input_root: Path, filename: str, expected: str) -> Path:
    """Discover actual nested mount paths by bytes, never by assumed slug folder."""
    candidates = list(input_root.rglob(filename))
    matched = [path for path in candidates if digest(path) == expected]
    if len(matched) != 1:
        raise ValueError(
            f"{filename}: expected exactly one pinned nested artifact, found {len(matched)}"
        )
    return matched[0]


def verify_shared_contract(path: Path, admission: Record) -> tuple[Record, dict[str, Path]]:
    """Validate completed R5 export recovery and every actual shared role."""
    if digest(path) != admission.get("gpu_training_contract_sha256"):
        raise ValueError("GPU producer training contract differs from root pin")
    contract = read_json(path)
    if (
        contract.get("schema") != "wnn.loss-pair.training-contract.v1"
        or contract.get("format_version") != 1
        or contract.get("state") != "complete"
    ):
        raise ValueError("unsupported actual GPU producer contract")
    recovery = contract.get("recovery")
    if not isinstance(recovery, dict) or any(
        recovery.get(key) != expected
        for key, expected in {
            "mode": "export_only",
            "predecessor_kernel": R4_FIT_KERNEL,
            "predecessor_terminal_receipt_sha256": R4_TERMINAL_RECEIPT_SHA,
            "predecessor_root_fit_review_sha256": R4_FIT_REVIEW_SHA,
            "recovery_asset_manifest_sha256": R5_RECOVERY_ASSET_SHA,
            "checkpoint_cache_receipt_sha256": R4_ORIGINAL_CACHE_RECEIPT_SHA,
            "predictor_original_sha256": R5_PREDICTOR_ORIGINAL_SHA,
            "fits_retrained": False,
            "checkpoint_files_rewritten": False,
        }.items()
    ):
        raise ValueError("GPU contract is not the reviewed R5 export-only recovery")
    evaluation = contract.get("evaluation")
    if (
        not isinstance(evaluation, dict)
        or evaluation.get("state") != "pending_independent_grade"
        or evaluation.get("metric_id") != "global_wp_clip2_v1"
        or evaluation.get("fold") != 0
        or evaluation.get("fixed_evaluation_count") != 468
    ):
        raise ValueError("R5 export contract evaluation identity differs")
    expected_cache = {
        "manifest_sha256": CACHE_MANIFEST_SHA,
        "cache_key": CACHE_KEY,
        "membership_sha256": MEMBERSHIP_SHA,
        "split_manifest_sha256": SPLIT_SHA,
        "portable_registry_sha256": REGISTRY_SHA,
        "fold": 0,
    }
    if any(contract.get("cache", {}).get(key) != value for key, value in expected_cache.items()):
        raise ValueError("GPU producer contract binds different cache/membership")
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("GPU producer contract lacks artifact role mapping")
    if not set(artifacts) >= REQUIRED_SHARED_ROLES:
        raise ValueError(
            f"R5 producer omits shared roles: {sorted(REQUIRED_SHARED_ROLES - set(artifacts))}"
        )
    resolved = {}
    for name in REQUIRED_SHARED_ROLES:
        item = artifacts[name]
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("GPU shared paths must be relative to out/training_contract.json")
        artifact = path.parent / relative
        artifact.resolve(strict=True).relative_to(path.parent.resolve(strict=True))
        if (
            artifact.is_symlink()
            or not artifact.is_file()
            or artifact.stat().st_size != item["size_bytes"]
            or digest(artifact) != item["sha256"]
        ):
            raise ValueError(f"GPU shared artifact changed: {name}")
        resolved[name] = artifact
    _verify_r5_recipe(contract, artifacts, resolved, recovery)
    return contract, resolved


def _verify_r5_recipe(
    contract: Record, artifacts: Record, resolved: dict[str, Path], recovery: Record
) -> None:
    """Confirm R5 retained the actual completed FP32 control recipe."""
    source = contract.get("source")
    cache = contract.get("cache")
    model = contract.get("model")
    fit = contract.get("fit")
    if (
        not isinstance(source, dict)
        or source.get("dataset") != R4_CODE_DATASET
        or source.get("source_stage_kernel") != CPU_KERNEL
        or not isinstance(cache, dict)
        or cache.get("validation_receipt_sha256") != artifacts["cache_validation_receipt"]["sha256"]
        or cache.get("reader_profile_sha256") != artifacts["cache_reader_profile"]["sha256"]
        or cache.get("source_verification_sha256") != artifacts["source_verification"]["sha256"]
        or cache.get("fold") != 0
        or cache.get("fit_train_count") != 350
        or cache.get("fit_valid_count_not_used_for_fit") != 28
        or cache.get("evaluation_valid_count") != 468
        or not isinstance(model, dict)
        or model.get("family") != "two-layer GRU"
        or model.get("input_width") != 211
        or model.get("hidden_size") != 256
        or model.get("layers") != 2
        or model.get("activation") != "GELU"
        or model.get("input_layout") != {"raw_features": 112, "causal_event_features": 99}
        or model.get("outputs") != 2
        or model.get("precision") != "float32"
        or model.get("tf32") is not False
        or not isinstance(fit, dict)
        or fit.get("batch_size") != 32
        or fit.get("chunk_rows") != 2000
        or fit.get("epochs") != 2
        or fit.get("updates_per_arm") != 220
        or fit.get("optimizer") != "AdamW"
        or fit.get("scheduler") != "OneCycleLR"
        or fit.get("global_gradient_clip") != 1.0
        or fit.get("learning_rate_max") != 0.001
        or fit.get("checkpoint_epoch_zero_indexed") != 1
    ):
        raise ValueError("R5 shared contract differs from the completed matched FP32 control")
    arms = contract.get("arms")
    if (
        not isinstance(arms, dict)
        or not isinstance(arms.get("wmse"), dict)
        or not isinstance(arms.get("huber_d1"), dict)
        or arms["wmse"].get("loss") != "wmse"
        or arms["huber_d1"].get("loss") != "huber"
        or arms["huber_d1"].get("huber_delta") != 1.0
        or any(arms[name].get("updates_attempted") != 220 for name in ("wmse", "huber_d1"))
    ):
        raise ValueError("R5 completed fit arms do not match the WMSE/Huber control pair")
    profile = read_json(resolved["device_profile_gate"])
    profile_recipe = profile.get("profile")
    if (
        profile.get("state") != "passed"
        or not isinstance(profile_recipe, dict)
        or profile_recipe.get("batch_size") != 32
        or profile_recipe.get("chunk_rows") != 2000
        or profile_recipe.get("input_width") != 211
        or profile_recipe.get("hidden_size") != 256
        or profile_recipe.get("layers") != 2
        or profile_recipe.get("precision") != "float32"
        or profile_recipe.get("attempted_updates") != 20
        or profile_recipe.get("tf32", False) is not False
    ):
        raise ValueError("R5 does not carry the actual-shape FP32 device gate")
    profile_inputs_receipt = read_json(resolved["profile_inputs_receipt"])
    if (
        profile_inputs_receipt.get("profile_inputs_sha256") != artifacts["profile_inputs"]["sha256"]
        or profile_inputs_receipt.get("initial_state_sha256")
        != artifacts["initial_state"]["sha256"]
        or profile_inputs_receipt.get("cache_validation_receipt_sha256")
        != artifacts["cache_validation_receipt"]["sha256"]
    ):
        raise ValueError("R5 actual first-batch fixture receipt does not bind its shared files")
    if (
        recovery.get("fresh_runtime_cache_receipt_sha256")
        != artifacts["cache_validation_receipt"]["sha256"]
    ):
        raise ValueError("R5 recovery and shared cache receipt identities differ")


def check_admission(
    admission: Record,
    source_manifest_sha: str,
    source_manifest: Record,
    *,
    require_push: bool = True,
) -> None:
    """Never turn a prepared packet or passing parity into root spending authority."""
    gpu_producer = source_manifest.get("gpu_export_recovery_producer_r5")
    if not isinstance(gpu_producer, dict):
        raise ValueError("source manifest has no R5 producer binding")
    pinned_contract_sha = gpu_producer.get("training_contract_sha256")
    pinned_review_sha = gpu_producer.get("root_terminal_review_sha256")
    if pinned_review_sha != R5_ROOT_REVIEW_SHA:
        raise ValueError("source manifest does not bind the reviewed R5 terminal receipt")
    expected = {
        "schema": "wnn.loss-pair-tpu.root-admission.v1",
        "source_manifest_sha256": source_manifest_sha,
        "cache_manifest_sha256": CACHE_MANIFEST_SHA,
        "cache_key": CACHE_KEY,
        "gpu_kernel": GPU_KERNEL,
        "cpu_kernel": CPU_KERNEL,
        "gpu_training_contract_sha256": pinned_contract_sha,
        "gpu_root_terminal_review_sha256": pinned_review_sha,
    }
    if any(admission.get(k) != value for k, value in expected.items()):
        raise ValueError("root admission identity differs from packet source/data contract")
    if require_push and admission.get("push_allowed") is not True:
        raise ValueError("packet has pending root/provider/producer admission")
    if admission.get("runtime_seconds", 0) > 3300 or admission.get("runtime_seconds", 0) <= 0:
        raise ValueError("worker runtime exceeds finite packet envelope")
    precision = admission.get("tpu_compute_precision", "fp32")
    if precision not in {"fp32", "bf16"}:
        raise ValueError("unsupported native TPU precision")
    if precision == "bf16" and admission.get("native_bf16_fit_allowed") is not True:
        raise ValueError("native BF16 profile/fit requires separate root admission")
    if admission.get("conditional_fit_allowed") is True:
        _policy, policy_errors = validate_policy(admission.get("fit_headroom_policy"))
        if policy_errors:
            raise ValueError("conditional fit admission omits approved headroom reserves")


def check_profile_binding(receipt: Record, first_ids: list[str], normalization_sha: str) -> None:
    """Require the pre-update fixture to belong to this exact ordered fit batch."""
    expected = {
        "cache_manifest_sha256": CACHE_MANIFEST_SHA,
        "cache_key": CACHE_KEY,
        "normalization_sha256": normalization_sha,
        "sequence_ids": first_ids,
    }
    if len(first_ids) != 32 or any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("GPU profile fixture cache/normalization/first-batch order differs")


def main() -> None:
    """Validate mounted inputs, profile actual TPU arithmetic, then conditional fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/out"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    admission = read_json(args.admission)
    manifest = read_json(args.source_manifest)
    check_admission(
        admission,
        digest(args.source_manifest),
        manifest,
        require_push=not args.offline,
    )
    for row in manifest["files"]:
        path = ROOT / row["path"]
        if path.stat().st_size != row["size_bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"sealed source changed: {row['path']}")
    if args.offline:
        # Source validation can run without provider admission or imported JAX.
        write_json(
            args.output_root / "offline_source_receipt.json",
            {
                "status": "pass",
                "source_manifest_sha256": digest(args.source_manifest),
                "fit_admitted": False,
            },
        )
        return
    cache_manifest = discover_pinned(args.input_root, "manifest.json", CACHE_MANIFEST_SHA)
    gpu_contract = discover_pinned(
        args.input_root, "training_contract.json", admission["gpu_training_contract_sha256"]
    )
    _, shared = verify_shared_contract(gpu_contract, admission)
    write_json(
        args.output_root / "phase.json",
        {
            "phase": "cache_mount_validation",
            "cache_manifest": str(cache_manifest),
            "gpu_contract": str(gpu_contract),
        },
    )
    sys.path.insert(0, str(ROOT / "src"))
    from wnn.parsed_sequence_cache_reader import ParsedSequenceCache, verify_parsed_cache

    validation_path = args.output_root / "tpu_mount_reader_validation.json"
    _, validation_sha = verify_parsed_cache(cache_manifest, CACHE_MANIFEST_SHA, validation_path)
    # Metadata and NPZ layout checks precede expensive actual-device compilation.
    # The supervising process never creates an accelerator array or model fit.
    os.environ["JAX_PLATFORMS"] = "cpu"
    import numpy as np
    from tpu_worker import EXPECTED_RECIPE, _cache_batch, _read_onecycle_trace, _read_orders
    from wnn_jax.rnn import validate_state_dict

    cache_metadata = read_json(cache_manifest)
    membership = read_json(cache_manifest.parent / cache_metadata["membership_artifact"]["path"])
    orders = _read_orders(shared["sample_orders"], membership)
    _read_onecycle_trace(shared["onecycle_trace"])
    check_profile_binding(
        read_json(shared["profile_inputs_receipt"]),
        read_json(shared["sample_orders"])["epochs"]["0"][:32],
        digest(shared["normalization"]),
    )
    with np.load(shared["initial_state"], allow_pickle=False) as initial:
        validate_state_dict(initial, 211, 256, 2, 2)
        if any(initial[name].dtype != np.dtype("<f4") for name in initial.files):
            raise ValueError("shared matched-FP32 initial state contains a different dtype")
        with np.load(shared["normalization"], allow_pickle=False) as normalization:
            if set(normalization.files) != {"mean", "std"} or any(
                not np.array_equal(initial[name], normalization[name])
                for name in normalization.files
            ):
                raise ValueError("shared normalization differs from initialized model buffers")
    cache = ParsedSequenceCache.open(
        cache_manifest,
        expected_manifest_sha256=CACHE_MANIFEST_SHA,
        validation_receipt_path=validation_path,
        expected_validation_receipt_sha256=validation_sha,
        expected_cache_key=CACHE_KEY,
        expected_split_manifest_sha256=SPLIT_SHA,
        expected_fold=0,
    )
    first_batch_ids = orders["epochs"]["0"][:32]
    cache_features, cache_targets, _cache_scored, observed_ids = _cache_batch(
        cache, orders["by_id"], first_batch_ids, "fit_train"
    )
    with np.load(shared["profile_inputs"], allow_pickle=False) as fixture:
        fixture_inputs = fixture["inputs"]
        fixture_targets = fixture["targets"]
        fixture_mask = fixture["row_mask"].astype(bool)
    expected_mask = np.broadcast_to(np.arange(2000, dtype=np.int32)[None, :] >= 99, (32, 2000))
    if observed_ids != first_batch_ids:
        raise ValueError("native cache loader returned a different first-batch order")
    if (
        not np.array_equal(cache_features[:, :2000], fixture_inputs)
        or not np.array_equal(cache_targets[:, :2000], fixture_targets)
        or not np.array_equal(expected_mask, fixture_mask)
    ):
        raise ValueError("native cache first batch differs from the GPU producer fixture")
    fixture_path_receipt = {
        "status": "pass",
        "cache_manifest_sha256": CACHE_MANIFEST_SHA,
        "cache_validation_receipt_sha256": validation_sha,
        "profile_inputs_sha256": digest(shared["profile_inputs"]),
        "first_batch_sequence_ids": observed_ids,
        "shape": [32, 2000, 211],
        "features_equal": True,
        "targets_equal": True,
        "training_mask_equal": True,
    }
    write_json(args.output_root / "producer_consumer_fixture_receipt.json", fixture_path_receipt)
    cache.close()
    del cache_features, cache_targets, _cache_scored, cache
    parity_root = args.output_root / "device_parity"
    write_json(
        args.output_root / "phase.json",
        {"phase": "actual_device_parity", "elapsed_seconds": time.monotonic() - start},
    )
    parity_command = [
        sys.executable,
        str(HERE / "run_device_parity.py"),
        "--initial-state",
        str(shared["initial_state"]),
        "--profile-inputs",
        str(shared["profile_inputs"]),
        "--profile-inputs-receipt",
        str(shared["profile_inputs_receipt"]),
        "--onecycle-trace",
        str(shared["onecycle_trace"]),
        "--output-root",
        str(parity_root),
    ]
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "tpu"
    env["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
    env["OMP_NUM_THREADS"] = "1"
    with (args.output_root / "device_parity.log").open("w") as log:
        subprocess.run(
            parity_command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=min(1200, admission["runtime_seconds"]),
            check=True,
        )
    parity = read_json(parity_root / "device_parity_receipt.json")
    if parity.get("status") != "pass":
        raise ValueError("actual-device gate did not pass")
    precision = admission.get("tpu_compute_precision", "fp32")
    measured_profile = parity["profile"]
    expected_device_count = EXPECTED_TPU_DEVICE_COUNT
    if precision == "bf16":
        native_root = args.output_root / "native_bf16_profile"
        with (args.output_root / "native_bf16_profile.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "run_native_profile.py"),
                    "--initial-state",
                    str(shared["initial_state"]),
                    "--profile-inputs",
                    str(shared["profile_inputs"]),
                    "--profile-inputs-receipt",
                    str(shared["profile_inputs_receipt"]),
                    "--fp32-receipt",
                    str(parity_root / "device_parity_receipt.json"),
                    "--onecycle-trace",
                    str(shared["onecycle_trace"]),
                    "--output-root",
                    str(native_root),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=max(1, int(admission["runtime_seconds"] - (time.monotonic() - start))),
                check=True,
            )
        native = read_json(native_root / "native_profile_receipt.json")
        if native.get("status") != "pass":
            raise ValueError("native BF16 gate failed")
        measured_profile = native["profile"]
        expected_device_count = native.get("device_count")
    headroom_gate = evaluate_profile_headroom(
        admission.get("fit_headroom_policy"),
        measured_profile,
        sample_live_resources(args.output_root),
        expected_device_count,
    )
    if parity.get("actual_data_profile") is not True:
        headroom_gate["passed"] = False
        headroom_gate["errors"].append("profile_is_not_actual_data")
    inventory_gate = evaluate_actual_device_inventory(
        parity.get("platform"), parity.get("accelerator_count"), parity.get("device_inventory")
    )
    headroom_gate["device_inventory_gate"] = inventory_gate
    if not inventory_gate["passed"]:
        headroom_gate["passed"] = False
        headroom_gate["errors"].extend(inventory_gate["errors"])
    headroom_path = args.output_root / "fit_headroom_gate.json"
    write_json(headroom_path, headroom_gate)
    if admission.get("conditional_fit_allowed") is not True:
        write_json(
            args.output_root / "terminal_receipt.json",
            {
                "status": "parity_complete_fit_not_admitted",
                "parity": parity,
                "compute_precision_profiled": precision,
                "native_profile": native if precision == "bf16" else None,
                "fit_headroom_gate": headroom_gate,
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        return
    if not headroom_gate["passed"]:
        write_json(
            args.output_root / "terminal_receipt.json",
            {
                "status": "profile_complete_fit_deferred_headroom",
                "defer_reason": "full-shape profile or live host/cgroup/disk reserves are missing or below root policy",
                "fit_headroom_gate": headroom_gate,
                "parity": parity,
                "compute_precision_profiled": precision,
                "native_profile": native if precision == "bf16" else None,
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        return
    remaining = admission["runtime_seconds"] - (time.monotonic() - start)
    estimated_warm_fit_seconds = sum(
        row["warm_update_seconds_max"] * 220 for row in measured_profile.values()
    )
    compile_reserve_seconds = sum(
        max(0, row["compile_plus_first_update_seconds"] - row["warm_update_seconds"])
        for row in measured_profile.values()
    )
    estimated_fit_seconds = estimated_warm_fit_seconds + compile_reserve_seconds
    estimate_basis = {
        "warm_fit_seconds": estimated_warm_fit_seconds,
        "repeat_compile_reserve_seconds": compile_reserve_seconds,
        "host_io_export_reserve_seconds": admission.get("evaluation_export_reserve_seconds", 600),
        "warm_measurement_basis": "maximum of five synchronized discarded full batch32 updates per loss",
        "compute_precision": precision,
        "limits": "Repeated warm full-shape steps exclude host preparation/exports. Full pair remains deadline-bounded; defer is execution evidence, not a scientific failure.",
    }
    export_reserve = admission.get("evaluation_export_reserve_seconds", 600)
    if estimated_fit_seconds + export_reserve > remaining:
        write_json(
            args.output_root / "terminal_receipt.json",
            {
                "status": "parity_complete_pair_deferred_runtime",
                "estimated_fit_seconds": estimated_fit_seconds,
                "remaining_seconds": remaining,
                "estimate_basis": estimate_basis,
                "parity": parity,
                "fit_headroom_gate": headroom_gate,
            },
        )
        return
    recipe = {**EXPECTED_RECIPE, "precision": precision, "losses": ["wmse", "huber"]}
    artifacts = {}
    bindings = {
        **{name: shared[name] for name in ("initial_state", "sample_orders", "onecycle_trace")},
        "cache_manifest": cache_manifest,
        "cache_validation_receipt": validation_path,
        "device_parity_receipt": parity_root / "device_parity_receipt.json",
        "fit_headroom_gate": headroom_path,
    }
    if precision == "bf16":
        bindings["native_profile_receipt"] = native_root / "native_profile_receipt.json"
    for name, path in bindings.items():
        artifacts[name] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": digest(path),
        }
    contract = {
        "schema": "wnn.loss-pair-tpu-run-contract.v1",
        "recipe": recipe,
        "identity": {
            "fold": 0,
            "cache_key": CACHE_KEY,
            "split_manifest_sha256": SPLIT_SHA,
            "membership_sha256": MEMBERSHIP_SHA,
            "portable_registry_sha256": REGISTRY_SHA,
        },
        "artifacts": artifacts,
        "ancestry": {
            "gpu_training_contract_sha256": digest(gpu_contract),
            "root_admission_sha256": digest(args.admission),
            "cache_mount_validation_sha256": validation_sha,
        },
        "fit_headroom_policy": admission["fit_headroom_policy"],
        "fit_headroom_gate_sha256": digest(headroom_path),
    }
    contract_path = args.output_root / "tpu_training_contract.json"
    write_json(contract_path, contract)
    write_json(
        args.output_root / "phase.json",
        {
            "phase": "matched_fit_pair",
            "elapsed_seconds": time.monotonic() - start,
            "estimated_fit_seconds": estimated_fit_seconds,
            "remaining_seconds": remaining,
            "estimate_basis": estimate_basis,
            "fit_headroom_gate_sha256": digest(headroom_path),
        },
    )
    with (args.output_root / "fit_pair.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                str(HERE / "tpu_worker.py"),
                "--contract",
                str(contract_path),
                "--output-root",
                str(args.output_root / "fit_pair"),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=max(1, int(remaining)),
            check=True,
        )
    write_json(
        args.output_root / "terminal_receipt.json",
        {
            "status": "complete",
            "fit_terminal": read_json(args.output_root / "fit_pair/terminal_receipt.json"),
            "parity": parity,
            "elapsed_seconds": time.monotonic() - start,
        },
    )


if __name__ == "__main__":
    main()
