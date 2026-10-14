#!/usr/bin/env python3
"""Profile the complete-sequence initializer-only path on one clean F0 sequence.

The preflight subcommand uses only the standard library and binds the frozen
control, clean-cache profile, source-qualified sequence and runtime sources.
The execute subcommand is a root-admitted CUDA resource smoke, not a fit or
quality evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

FORMAT = "wnn.initializer-complete-sequence-profile-contract"
RESULT_FORMAT = "wnn.initializer-complete-sequence-profile-result"
PROFILE_MODES = ("zero", "constant", "first_row", "prefix")
HOST_RESERVATION_BYTES = 5_000_000_000
CUDA_ALLOCATOR_CAP_BYTES = 3_500_000_000
OBSERVED_VRAM_LIMIT_BYTES = 5_000_000_000
SEQUENCE_ROWS = 20_000
CHUNK_ROWS = 2_000
N_FEATURES = 211
MAX_CONTRACT_BYTES = 64_000
MAX_SMALL_JSON_BYTES = 8_000_000
MAX_CACHE_METADATA_BYTES = 512_000
MAX_SPLIT_BYTES = 2_000_000


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Hash an in-memory byte string."""
    return hashlib.sha256(value).hexdigest()


def bounded_json(path: Path, limit: int, label: str) -> tuple[dict[str, Any], bytes]:
    """Parse a JSON object only after checking its on-disk size."""
    size = path.stat().st_size
    if size <= 0 or size > limit:
        raise ValueError(f"{label} is outside its {limit}-byte read bound")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError(f"{label} changed while being read")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, raw


def source_manifest(script: Path, repo: Path) -> dict[str, str]:
    """Bind the initializer harness and implementation to exact source files."""
    paths = (
        script,
        repo / "src/wnn/initializer_training.py",
        repo / "src/wnn/state_initialization.py",
    )
    return {str(path.relative_to(repo)): sha256_file(path) for path in paths}


def check_control(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Verify the immutable fitted H256×2 control and its actual producer receipt."""
    pins = contract["control"]
    checkpoint_path = root / pins["checkpoint"]
    receipt_path = root / pins["fit_validation"]
    provenance_path = root / pins["provenance"]
    for path, key in (
        (checkpoint_path, "checkpoint_sha256"),
        (receipt_path, "fit_validation_sha256"),
        (provenance_path, "provenance_sha256"),
    ):
        if sha256_file(path) != pins[key]:
            raise ValueError(f"frozen initializer backbone input changed: {path}")

    receipt, _ = bounded_json(receipt_path, MAX_SMALL_JSON_BYTES, "fit validation receipt")
    provenance, _ = bounded_json(provenance_path, MAX_SMALL_JSON_BYTES, "fit provenance")
    fit_source_root = Path(contract["training_source"]["source_root"]).resolve(strict=True)
    actual_training_sources = {
        relative: sha256_file(fit_source_root / relative)
        for relative in contract["training_source"]["source_sha256"]
    }
    if actual_training_sources != contract["training_source"]["source_sha256"]:
        raise ValueError("frozen fit producer source files changed")
    if provenance.get("student_frontier", {}).get("source_sha256") != actual_training_sources:
        raise ValueError("fitted H256 control provenance was produced by another trainer source")
    args = provenance.get("args", {})
    expected_args = {
        "name": pins["run"],
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "hidden": 256,
        "layers": 2,
        "dropout": 0.0,
        "readout_activation": "gelu",
        "seed": 3,
        "epochs": 1,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 0,
        "precision": "fp32",
        "distill_cache_variant": "homogeneous",
        "distill_alpha": 0.9,
        "propensity": "c20260929_prop_f0",
    }
    if any(args.get(name) != value for name, value in expected_args.items()):
        raise ValueError("frozen control provenance no longer matches the contracted H256×2 fit")
    if (
        provenance.get("split_manifest_sha256") != contract["split_sha256"]
        or provenance.get("distillation_cache", {}).get("metadata_sha256")
        != contract["cache"]["metadata_sha256"]
        or provenance.get("student_frontier", {})
        .get("cache_binding", {})
        .get("verified_target_files")
        != contract["cache"]["target_files"]
    ):
        raise ValueError("frozen control provenance is not bound to the pinned clean F0 targets")
    if (
        receipt.get("run") != pins["run"]
        or receipt.get("shape") != [256, 2]
        or receipt.get("checkpoint_sha256") != pins["checkpoint_sha256"]
        or receipt.get("optimizer_attempts") != 14_920
        or receipt.get("optimizer_completed_updates") != 14_920
        or receipt.get("amp_skipped_updates") != 0
        or receipt.get("nonfinite_gradient_steps") != 0
        or receipt.get("validation", "").split(";")[0] != "passed"
    ):
        raise ValueError(
            "control fit validation receipt does not prove the contracted terminal checkpoint"
        )
    if receipt.get("normalization_identity") != contract["normalization"]["identity"]:
        raise ValueError(
            "control checkpoint normalization differs from the common 211-feature buffers"
        )
    return {
        "run": pins["run"],
        "checkpoint": pins["checkpoint"],
        "checkpoint_sha256": pins["checkpoint_sha256"],
        "fit_validation": pins["fit_validation"],
        "fit_validation_sha256": pins["fit_validation_sha256"],
        "provenance": pins["provenance"],
        "provenance_sha256": pins["provenance_sha256"],
        "completed_updates": receipt["optimizer_completed_updates"],
        "normalization_identity": receipt["normalization_identity"],
        "split_manifest_sha256": provenance["split_manifest_sha256"],
    }


def check_cache_packet(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Verify the pinned B8 strict-consumer result and source-qualified member."""
    pins = contract["cache_packet"]
    b8_root = Path(pins["source_root"]).resolve(strict=True)
    b8_contract_path = root / pins["contract"]
    b8_result_path = root / pins["result"]
    b8_helper_path = b8_root / pins["helper"]
    for path, key in (
        (b8_contract_path, "contract_sha256"),
        (b8_result_path, "result_sha256"),
        (root / pins["attempt_receipt"], "attempt_receipt_sha256"),
        (b8_helper_path, "helper_sha256"),
    ):
        if sha256_file(path) != pins[key]:
            raise ValueError(f"reviewed source-qualified cache packet changed: {path}")

    b8_contract, raw_contract = bounded_json(
        b8_contract_path, MAX_CONTRACT_BYTES, "B8 profile contract"
    )
    b8_result, _ = bounded_json(b8_result_path, MAX_SMALL_JSON_BYTES, "B8 profile result")
    b8_attempt, _ = bounded_json(
        root / pins["attempt_receipt"], MAX_SMALL_JSON_BYTES, "B8 terminal attempt receipt"
    )
    if (
        sha256_bytes(raw_contract) != pins["contract_sha256"]
        or b8_result.get("profile_contract_sha256") != pins["contract_sha256"]
        or b8_result.get("profile_source_sha256") != pins["source_manifest"]
        or b8_contract.get("source_sha256") != pins["source_manifest"]
        or pins["source_manifest"].get(pins["helper"]) != pins["helper_sha256"]
    ):
        raise ValueError("B8 profile result is detached from its frozen contract/source")
    terminal = pins["terminal_resource_measurement"]
    if (
        b8_attempt.get("contract_sha256") != pins["contract_sha256"]
        or b8_attempt.get("status") != "finished"
        or b8_attempt.get("exit_status") != 0
        or b8_attempt.get("peak_ram_bytes") != terminal["host_peak_bytes"]
        or b8_attempt.get("peak_observed_vram_bytes") != terminal["observed_vram_peak_bytes"]
        or b8_attempt.get("final_memory_event_deltas", {}).get("oom") != 0
        or b8_attempt.get("final_memory_event_deltas", {}).get("oom_kill") != 0
    ):
        raise ValueError("pinned B8 resource result lacks a successful bounded controller receipt")
    sampler = b8_result.get("source_qualified_sampler", {})
    first_member = sampler.get("first_batch_members", [None])[0]
    first_seq_ix = sampler.get("first_batch_seq_ix", [None])[0]
    if (
        first_member != contract["sample"]["source_member"]
        or first_seq_ix != contract["sample"]["seq_ix"]
        or sampler.get("overlapping_row_group_probe", {}).get("source_qualification_preserved")
        is not True
    ):
        raise ValueError(
            "profile sequence is not the actual source-qualified seed-3 epoch-0 member"
        )
    overlap = sampler.get("overlapping_row_group_probe", {})
    if (
        sampler.get("all_cache_rows_source_qualified") != 11_940
        or overlap.get("row_group") != 37
        or overlap.get("train_cache_row") != 36
        or overlap.get("valid_cache_row") != 27
    ):
        raise ValueError("overlapping train/valid row-group IDs lost source-local cache mapping")
    if (
        b8_result.get("cache_metadata_sha256") != contract["cache"]["metadata_sha256"]
        or b8_result.get("cache_verified_files") != contract["cache"]["target_files"]
        or b8_result.get("normalization_sha256") != contract["normalization"]["hashes"]
        or b8_result.get("split_manifest_sha256") != contract["split_sha256"]
        or b8_result.get("cache_ancestry") != contract["cache_ancestry"]
    ):
        raise ValueError(
            "reviewed B8 result does not match the initializer arm's cache/norm/split pins"
        )

    metadata_path = root / contract["cache"]["root"] / "metadata.json"
    metadata, metadata_raw = bounded_json(
        metadata_path, MAX_CACHE_METADATA_BYTES, "clean target metadata"
    )
    if sha256_bytes(metadata_raw) != contract["cache"]["metadata_sha256"]:
        raise ValueError("clean homogeneous target metadata changed")
    if metadata.get("student_contract") != b8_contract["cache"]["historical_student_contract"]:
        raise ValueError("immutable producer's historical consumer profile metadata changed")
    if metadata.get("target_contract") != b8_contract["cache"]["target_contract"]:
        raise ValueError("immutable clean target formula changed")
    for name, expected_sha in contract["cache"]["target_files"].items():
        metadata_file = metadata.get("files", {}).get(name)
        if not isinstance(metadata_file, dict):
            raise ValueError(f"strict producer metadata lacks target {name}")
        path = root / contract["cache"]["root"] / metadata_file["path"]
        if (
            path.stat().st_size != metadata_file["bytes"]
            or metadata_file["bytes"] != contract["cache"]["target_file_bytes"][name]
            or metadata_file.get("sha256") != expected_sha
        ):
            raise ValueError(f"clean target payload identity/size changed for {name}")

    split_path = root / contract["split_path"]
    if sha256_file(split_path) != contract["split_sha256"]:
        raise ValueError("clean F0 source-qualified split manifest changed")
    split, _ = bounded_json(split_path, MAX_SPLIT_BYTES, "clean F0 split manifest")
    matched = [
        sequence
        for sequence in split.get("sequences", {}).get(contract["sample"]["source"], [])
        if sequence.get("row_group") == contract["sample"]["row_group"]
    ]
    if len(matched) != 1 or matched[0].get("seq_ix") != contract["sample"]["seq_ix"]:
        raise ValueError("source-qualified row_group/seq_ix does not match the frozen split")
    return {
        "b8_contract_sha256": pins["contract_sha256"],
        "b8_result_sha256": pins["result_sha256"],
        "b8_helper_sha256": pins["helper_sha256"],
        "split_manifest_sha256": contract["split_sha256"],
        "cache_metadata_sha256": contract["cache"]["metadata_sha256"],
        "target_files": contract["cache"]["target_files"],
        "cache_ancestry": contract["cache_ancestry"],
        "sample": {
            "source": contract["sample"]["source"],
            "row_group": contract["sample"]["row_group"],
            "seq_ix": contract["sample"]["seq_ix"],
            "epoch0_dataset_position": sampler["first_batch_positions"][0],
        },
        "historical_producer_student_contract_unchanged": True,
        "full_target_payload_hashes_reverified_by": "pinned B8 strict consumer at GPU profile execution",
    }


def static_preflight(script: Path, contract_path: Path, output: Path | None) -> dict[str, Any]:
    """Run source/metadata/checkpoint checks without importing NumPy or PyTorch."""
    contract, raw = bounded_json(contract_path, MAX_CONTRACT_BYTES, "initializer profile contract")
    contract_sha = sha256_bytes(raw)
    if contract.get("format") != FORMAT or contract.get("version") != 1:
        raise ValueError("unsupported initializer profile contract")
    root = Path(contract["data_root"]).resolve(strict=True)
    repo = script.resolve().parents[1]
    if repo != Path(contract["source_root"]).resolve():
        raise ValueError("initializer profile script is not running from its frozen source root")
    actual_sources = source_manifest(script.resolve(), repo)
    if actual_sources != contract.get("source_sha256"):
        raise ValueError("initializer profile source hashes differ from the frozen contract")
    if contract.get("precision") != "float32" or contract.get("chunk_rows") != CHUNK_ROWS:
        raise ValueError("initializer profile precision or chunk size changed")
    if contract.get("modes") != list(PROFILE_MODES):
        raise ValueError("initializer profile mode allowlist changed")
    if (
        contract.get("resource_envelope", {}).get("host_reservation_bytes")
        != HOST_RESERVATION_BYTES
        or contract.get("resource_envelope", {}).get("cuda_allocator_cap_bytes")
        != CUDA_ALLOCATOR_CAP_BYTES
        or contract.get("resource_envelope", {}).get("observed_vram_ceiling_bytes")
        != OBSERVED_VRAM_LIMIT_BYTES
    ):
        raise ValueError("initializer profile resource ceiling changed")
    cache_receipt = check_cache_packet(root, contract)
    fit_receipt = check_control(root, contract)
    result = {
        "format": "wnn.initializer-complete-sequence-profile-preflight",
        "version": 1,
        "contract_sha256": contract_sha,
        "source_root": str(repo),
        "source_sha256": actual_sources,
        "control": fit_receipt,
        "clean_cache_binding": cache_receipt,
        "cache_ancestry": contract["cache_ancestry"],
        "normalization": contract["normalization"],
        "propensity": contract["propensity"],
        "profile_scope": contract["profile_scope"],
        "resource_envelope": contract["resource_envelope"],
        "gpu_launch_performed": False,
        "quality_claim": None,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"preflight receipt already exists; preserve attempts: {output}")
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
    return result


def current_cgroup_bytes(filename: str) -> int | None:
    """Read one cgroup-v2 memory counter for the current process."""
    try:
        entries = [
            line.split(":", 2)[2]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        ]
        if len(entries) != 1:
            return None
        path = (Path("/sys/fs/cgroup") / entries[0].lstrip("/")).resolve()
        root = Path("/sys/fs/cgroup").resolve()
        if not path.is_relative_to(root):
            return None
        value = (path / filename).read_text().strip()
        return int(value) if value.isascii() and value.isdigit() else None
    except (OSError, ValueError):
        return None


def nvidia_smi_used_bytes() -> int:
    """Return the sum of observed GPU memory usage, rounded up to MiB."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not values or any(not value.isascii() or not value.isdigit() for value in values):
        raise RuntimeError(f"unparseable nvidia-smi memory output: {result.stdout!r}")
    return sum((int(value) + 1) * 1024**2 for value in values)


def finite_tensors(state: dict[str, Any], torch: Any) -> bool:
    """Check every tensor in a module state dictionary for finite values."""
    return all(
        torch.isfinite(value).all().item() for value in state.values() if torch.is_tensor(value)
    )


def tensor_identity(value: Any, torch: Any) -> str:
    """Hash canonical little-endian FP32 tensor bytes."""
    array = (
        value.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
    )
    return sha256_bytes(array.tobytes())


def make_base_model(
    trainer: Any, contract: dict[str, Any], state: dict[str, Any], device: Any
) -> Any:
    """Construct the exact plain H256×2 GRU and install the frozen CP0 state."""
    shape = contract["control"]["shape"]
    args = {
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "hidden": shape[0],
        "layers": shape[1],
        "dropout": 0.0,
        "readout_activation": "gelu",
        "numeric_map": "none",
        "ple_intervals": 8,
    }
    backbone = trainer.make_model(args, N_FEATURES, 2)
    incompatible = backbone.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("frozen H256×2 checkpoint does not match the current student model")
    return backbone.to(device=device)


def initialize_inputs(
    trainer: Any, batch: Any, propensity_model: Any, torch: Any
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Build the exact 211 inputs, blended targets and main F0 label/propensity weights."""
    x_cpu, target_cpu, scored_cpu, prop_cpu = (value[:1] for value in batch)
    device = torch.device("cuda")
    x = x_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
    target_pack = target_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
    scored = scored_cpu.to(device=device, non_blocking=True)
    prop_cached = prop_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
    trainer.HEURISTICS = False
    trainer.RAW_ONLY = False
    trainer.ONLINE_NORM = False
    trainer.RESEARCH_FEATURES = "none"
    trainer.PRIVILEGED_MODE = "none"
    trainer.INPUT_PROPENSITY = None
    inputs = trainer.build_inputs(x)
    if tuple(inputs.shape) != (1, SEQUENCE_ROWS, N_FEATURES):
        raise ValueError("production feature builder did not return the declared 211 causal inputs")
    labels = target_pack[..., :2]
    teacher = target_pack[..., 2:4]
    target = trainer.blended_target(labels, teacher, None, slice(0, SEQUENCE_ROWS), 0.9, 0, 0.0)
    valid = torch.arange(SEQUENCE_ROWS, device=device)[None] >= trainer.WARMUP
    row_weight = trainer.main_row_weights(
        inputs,
        labels,
        valid,
        scored,
        propensity_model,
        prop_cached,
        argparse.Namespace(
            mask_weight=False,
            prop_power=2.0,
            prop_eps=0.03,
            class_weight="",
            valid_true_mask=0.03,
            sync_free=True,
            since_weight=0.0,
            episode_weight=0.0,
            episode_floor=0.5,
            since_floor=0.25,
            prop_cache=True,
        ),
        x,
    )
    weights = labels.float().clamp(-2.0, 2.0).abs() * row_weight[..., None].float()
    weights[:, :99] = 0
    if (
        not torch.isfinite(inputs).all()
        or not torch.isfinite(target).all()
        or not torch.isfinite(weights).all()
        or torch.any(weights < 0)
    ):
        raise FloatingPointError(
            "profile inputs, targets or nonnegative loss weights are not finite"
        )
    total_mass = float(weights.sum().detach().cpu())
    if not total_mass > 0:
        raise ValueError("source-qualified sequence has no positive main loss weight")
    return (
        x,
        inputs,
        target,
        {
            "weights": weights,
            "total_weight_mass": total_mass,
            "target_formula": "0.1*clamp(label,-2,2)+0.9*clamp(homogeneous_teacher,-2,2)",
            "row_weight": "abs(clamp(label,-2,2))*main_row_weights; p^2+0.03 and valid_true_mask epsilon 0.03",
            "target_sha256": tensor_identity(target, torch),
            "weight_sha256": tensor_identity(weights, torch),
            "input_sha256": tensor_identity(inputs, torch),
            "finite_nonnegative_weights": True,
        },
    )


def initializer_state_dict(module: Any) -> dict[str, Any]:
    """Return only learned initializer tensors, excluding the frozen backbone."""
    return {
        name: value.detach().cpu().contiguous()
        for name, value in module.state_dict().items()
        if not name.startswith("backbone.")
    }


def export_native(
    module: Any,
    mode: str,
    output_dir: Path,
    np: Any,
    torch: Any,
    *,
    write_archive: bool = True,
) -> dict[str, Any]:
    """Validate and write the native initializer payload for the profiled arm."""
    from wnn.state_initialization import export_native_initializer

    config = export_native_initializer(module)
    result: dict[str, Any] = {"mode": mode}
    arrays: dict[str, Any] = {}
    for name, value in config.items():
        if isinstance(value, np.ndarray):
            if (
                value.dtype != np.float32
                or not value.flags.c_contiguous
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"native initializer field {name} is not finite contiguous FP32")
            arrays[name] = value
            result[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": sha256_bytes(value.astype("<f4", copy=False).tobytes(order="C")),
            }
        elif isinstance(value, float):
            if not np.isfinite(value):
                raise ValueError("native residual bound is not finite")
            result[name] = value
        else:
            result[name] = value
    if arrays and write_archive:
        archive_path = output_dir / f"INITIALIZER_{mode.upper()}_NATIVE_R1.npz"
        np.savez(archive_path, **arrays)
        result["archive"] = archive_path.name
        result["archive_sha256"] = sha256_file(archive_path)
        result["archive_bytes"] = archive_path.stat().st_size
    return result


def profile_mode(
    mode: str,
    trainer: Any,
    base_state: dict[str, Any],
    x: Any,
    inputs: Any,
    targets: Any,
    weights: Any,
    contract: dict[str, Any],
    output_dir: Path,
    torch: Any,
    np: Any,
) -> dict[str, Any]:
    """Run one complete-sequence FP32 update and verify state/export invariants."""
    arm_started = time.perf_counter()
    from wnn.initializer_training import initializer_sequence_loss
    from wnn.state_initialization import InitializedGRU

    torch.manual_seed(int(contract["seed"]))
    torch.cuda.manual_seed_all(int(contract["seed"]))
    device = torch.device("cuda")
    backbone = make_base_model(trainer, contract, base_state, device)
    backbone.eval()
    backbone._inference_fp32 = True
    backbone.requires_grad_(False)
    model = InitializedGRU(backbone, mode, residual_bound=float(contract["residual_bound"]))
    model.train()
    backbone_before = {
        name: value.detach().to(device="cpu").clone()
        for name, value in backbone.state_dict().items()
    }
    with torch.no_grad():
        zero_start, zero_end = model(inputs)
        zero_prediction = zero_start[..., :2].float()
    if mode == "zero":
        expected = {
            "mode": mode,
            "initial_prediction_sha256": tensor_identity(zero_prediction, torch),
            "zero_reference_prediction_sha256": tensor_identity(zero_prediction, torch),
            "initialization_matches_zero_reference": True,
            "updates_attempted": 0,
            "updates_completed": 0,
            "loss": None,
            "gradient_norm_before_clip": None,
            "parameter_update_observed": False,
            "backbone_unchanged": True,
            "backbone_sha256": contract["control"]["checkpoint_sha256"],
            "native_export": {"mode": "zero"},
            "checkpoint_reload": "not applicable; zero mode has no learned initializer parameters",
        }
        del model, backbone, zero_start, zero_end, zero_prediction
        return expected

    initial_sha = tensor_identity(zero_prediction, torch)
    reference_sha = contract["zero_reference_prediction_sha256"]
    if initial_sha != reference_sha:
        raise ValueError(
            f"{mode} zero-initialized predictions differ from the zero-state reference"
        )
    del zero_start, zero_end, zero_prediction

    initializer_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not initializer_parameters or any(
        parameter.requires_grad for parameter in backbone.parameters()
    ):
        raise ValueError("profile optimizer must see initializer parameters only")
    initializer_before = [
        parameter.detach().to(device="cpu").clone() for parameter in initializer_parameters
    ]
    optimizer = torch.optim.AdamW(
        initializer_parameters,
        lr=float(contract["optimizer"]["lr"]),
        betas=tuple(contract["optimizer"]["betas"]),
        eps=float(contract["optimizer"]["eps"]),
        weight_decay=float(contract["optimizer"]["weight_decay"]),
        fused=False,
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss = initializer_sequence_loss(
        model,
        inputs,
        targets,
        weights,
        chunk_rows=CHUNK_ROWS,
        checkpoint_chunks=True,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError(f"{mode} full-sequence loss is nonfinite")
    loss.backward()
    if any(parameter.grad is None for parameter in initializer_parameters):
        raise FloatingPointError(f"{mode} initializer has a missing gradient")
    if any(not torch.isfinite(parameter.grad).all().item() for parameter in initializer_parameters):
        raise FloatingPointError(f"{mode} initializer has a nonfinite gradient")
    gradient_norm = torch.nn.utils.clip_grad_norm_(initializer_parameters, max_norm=1.0)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError(f"{mode} initializer gradient norm is nonfinite")
    has_nonzero_gradient = any(
        torch.count_nonzero(parameter.grad).item() for parameter in initializer_parameters
    )
    if not has_nonzero_gradient:
        raise FloatingPointError(f"{mode} initializer received no nonzero gradient")
    optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if not finite_tensors(model.state_dict(), torch):
        raise FloatingPointError(f"{mode} update produced a nonfinite model tensor")
    backbone_unchanged = all(
        torch.equal(value, backbone.state_dict()[name].detach().to(device="cpu"))
        for name, value in backbone_before.items()
    )
    if not backbone_unchanged:
        raise RuntimeError(f"{mode} optimizer modified the frozen H256×2 backbone")
    if any(parameter.grad is not None for parameter in backbone.parameters()):
        raise RuntimeError(f"{mode} produced a backbone gradient despite the freeze")

    model.eval()
    with torch.no_grad():
        full_prediction, full_state = model(inputs, None)
        chunk_predictions = []
        chunk_state = None
        for start in range(0, SEQUENCE_ROWS, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, SEQUENCE_ROWS)
            prediction, chunk_state = model(inputs[:, start:stop], chunk_state)
            chunk_predictions.append(prediction)
        chunk_prediction = torch.cat(chunk_predictions, dim=1)
        max_chunk_difference = float((full_prediction - chunk_prediction).abs().max().cpu())
        torch.testing.assert_close(full_prediction, chunk_prediction, rtol=2e-5, atol=2e-6)
        reset_prediction, reset_state = model(inputs, None)
        max_reset_difference = float((full_prediction - reset_prediction).abs().max().cpu())
        torch.testing.assert_close(full_prediction, reset_prediction, rtol=0, atol=0)
        # Process a nonterminal prefix, then prove that an explicit None starts a fresh sequence.
        stale_prefix, stale_state = model(inputs[:, :1_000], None)
        carried_prediction, _ = model(inputs[:, 1_000:2_000], stale_state)
        torch.testing.assert_close(
            carried_prediction, full_prediction[:, 1_000:2_000], rtol=2e-5, atol=2e-6
        )
        del stale_prefix, carried_prediction
        fresh_chunk, fresh_state = model(inputs[:, :2_000], None)
        max_fresh_reset_difference = float(
            (fresh_chunk - full_prediction[:, :2_000]).abs().max().cpu()
        )
        torch.testing.assert_close(fresh_chunk, full_prediction[:, :2_000], rtol=2e-5, atol=2e-6)
    export = export_native(model, mode, output_dir, np, torch)

    init_checkpoint_path = output_dir / f"INITIALIZER_{mode.upper()}_R1.pt"
    init_state = initializer_state_dict(model)
    if not init_state:
        raise ValueError(f"{mode} initializer checkpoint has no learned parameter tensors")
    torch.save(
        {
            "format": "wnn.initializer-only-profile-checkpoint",
            "version": 1,
            "mode": mode,
            "residual_bound": float(contract["residual_bound"]),
            "backbone_checkpoint_sha256": contract["control"]["checkpoint_sha256"],
            "normalization_sha256": contract["normalization"]["hashes"],
            "initializer_state_dict": init_state,
        },
        init_checkpoint_path,
    )
    checkpoint_sha = sha256_file(init_checkpoint_path)
    saved = torch.load(init_checkpoint_path, map_location="cpu", weights_only=False)
    if (
        saved.get("format") != "wnn.initializer-only-profile-checkpoint"
        or saved.get("backbone_checkpoint_sha256") != contract["control"]["checkpoint_sha256"]
        or saved.get("mode") != mode
    ):
        raise ValueError(f"{mode} profile checkpoint metadata failed reload")
    reloaded_backbone = make_base_model(trainer, contract, base_state, device)
    reloaded_backbone.requires_grad_(False)
    reloaded = InitializedGRU(
        reloaded_backbone, mode, residual_bound=float(contract["residual_bound"])
    )
    merged = reloaded.state_dict()
    for name, value in saved["initializer_state_dict"].items():
        if name not in merged or name.startswith("backbone."):
            raise ValueError(f"{mode} profile checkpoint contains an unsupported initializer key")
        merged[name] = value
    reloaded.load_state_dict(merged, strict=True)
    reloaded.eval()
    with torch.no_grad():
        reloaded_prediction, _ = reloaded(inputs, None)
    reload_difference = float((full_prediction - reloaded_prediction).abs().max().cpu())
    torch.testing.assert_close(full_prediction, reloaded_prediction, rtol=0, atol=0)
    reloaded_export = export_native(reloaded, mode, output_dir, np, torch, write_archive=False)
    export_fields = {
        name: value
        for name, value in export.items()
        if name
        not in {
            "archive",
            "archive_sha256",
            "archive_bytes",
        }
    }
    if reloaded_export != export_fields:
        raise RuntimeError(f"{mode} native export changed after checkpoint reload")

    parameter_update_observed = any(
        not torch.equal(before, parameter.detach().to(device="cpu"))
        for before, parameter in zip(initializer_before, initializer_parameters, strict=True)
    )
    if not parameter_update_observed:
        raise RuntimeError(f"{mode} optimizer step changed no initializer parameter")

    torch.cuda.synchronize(device)
    cuda_allocated = int(torch.cuda.max_memory_allocated(device))
    cuda_reserved = int(torch.cuda.max_memory_reserved(device))
    observed_vram = nvidia_smi_used_bytes()
    if cuda_reserved > CUDA_ALLOCATOR_CAP_BYTES:
        raise RuntimeError("PyTorch CUDA reservation exceeded the explicit 3.5 GB allocator cap")
    if observed_vram >= OBSERVED_VRAM_LIMIT_BYTES:
        raise RuntimeError("observed GPU memory reached or exceeded the fixed 5 GB ceiling")
    result = {
        "mode": mode,
        "batch_size": 1,
        "sequence_rows": SEQUENCE_ROWS,
        "chunk_rows": CHUNK_ROWS,
        "chunks": SEQUENCE_ROWS // CHUNK_ROWS,
        "precision": "float32; autocast disabled; TF32 disabled",
        "checkpointed_recurrent_graph_across_all_chunks": True,
        "updates_attempted": 1,
        "updates_completed": 1,
        "loss": float(loss.detach().cpu()),
        "loss_finite": True,
        "global_sequence_weight_denominator": float(weights.sum().detach().cpu()),
        "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
        "nonzero_initializer_gradient": True,
        "parameter_update_observed": parameter_update_observed,
        "backbone_unchanged": backbone_unchanged,
        "checkpoint_reload": {
            "verified": True,
            "path": init_checkpoint_path.name,
            "sha256": checkpoint_sha,
            "bytes": init_checkpoint_path.stat().st_size,
            "max_prediction_abs_difference": reload_difference,
        },
        "native_export": export,
        "full_vs_10_chunk_max_abs_difference": max_chunk_difference,
        "reset_full_prediction_max_abs_difference": max_reset_difference,
        "reset_after_prior_prefix_max_abs_difference": max_fresh_reset_difference,
        "reset_after_prior_prefix_verified": True,
        "profile_elapsed_seconds": elapsed,
        "arm_total_elapsed_seconds": time.perf_counter() - arm_started,
        "cuda_peak_allocated_bytes": cuda_allocated,
        "cuda_peak_reserved_bytes": cuda_reserved,
        "observed_vram_bytes_after_arm": observed_vram,
        "host_memory_current_bytes_after_arm": current_cgroup_bytes("memory.current"),
        "host_memory_peak_bytes_after_arm": current_cgroup_bytes("memory.peak"),
        "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "quality_claim": None,
    }
    del optimizer, model, reloaded, reloaded_backbone, saved, merged, init_state
    del full_prediction, full_state, chunk_predictions, chunk_state, chunk_prediction
    del reset_prediction, reset_state, stale_state, fresh_state, fresh_chunk
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return result


def execute_profile(
    script: Path, contract: dict[str, Any], contract_sha256: str, output: Path
) -> dict[str, Any]:
    """Run strict cache validation and the one-sequence CUDA resource profile."""
    import numpy as np
    import torch

    root = Path(contract["data_root"]).resolve(strict=True)
    repo = Path(contract["source_root"]).resolve(strict=True)
    os.environ["WNN_PROJECT_ROOT"] = str(root)
    if os.environ.get("WNN_GPU_MEMORY_BYTES") != str(CUDA_ALLOCATOR_CAP_BYTES):
        raise RuntimeError("profile requires WNN_GPU_MEMORY_BYTES=3500000000")
    if not torch.cuda.is_available():
        raise RuntimeError("initializer profile was admitted without a usable CUDA device")
    if torch.__version__ != contract["torch_version"]:
        raise RuntimeError(
            f"Torch build changed: expected {contract['torch_version']}, got {torch.__version__}"
        )
    if current_cgroup_bytes("memory.max") not in (None, HOST_RESERVATION_BYTES):
        raise RuntimeError("initializer worker is outside the declared 5 GB host memory envelope")

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    b8_root = Path(contract["cache_packet"]["source_root"]).resolve(strict=True)
    fit_root = Path(contract["training_source"]["source_root"]).resolve(strict=True)
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(b8_root / "src"))
    sys.path.insert(0, str(fit_root / "src"))
    sys.path.insert(0, str(b8_root / "scripts"))
    b8_helper = b8_root / contract["cache_packet"]["helper"]
    helper_source = importlib.util.spec_from_file_location(
        "pinned_student_frontier_batch", b8_helper
    )
    if helper_source is None or helper_source.loader is None:
        raise RuntimeError("could not import the reviewed B8 cache/source-order helper")
    batch_helper = importlib.util.module_from_spec(helper_source)
    helper_source.loader.exec_module(batch_helper)
    import wnn

    current_initializer_package = str(repo / "src/wnn")
    if current_initializer_package not in wnn.__path__:
        wnn.__path__.append(current_initializer_package)
    if batch_helper.sha256_file(b8_helper) != contract["cache_packet"]["helper_sha256"]:
        raise ValueError("reviewed B8 batch helper changed before profile execution")
    expected_sources = batch_helper.source_manifest(b8_helper, b8_root)
    b8_contract_path = root / contract["cache_packet"]["contract"]
    b8_contract, _ = bounded_json(b8_contract_path, MAX_CONTRACT_BYTES, "B8 profile contract")
    if expected_sources != b8_contract.get("source_sha256"):
        raise ValueError("reviewed strict cache/order helper's source manifest changed")

    trainer_spec = importlib.util.spec_from_file_location(
        "train_rnn", fit_root / "scripts/train_rnn.py"
    )
    if trainer_spec is None or trainer_spec.loader is None:
        raise RuntimeError("could not import the pinned producer-compatible trainer")
    trainer = importlib.util.module_from_spec(trainer_spec)
    sys.modules["train_rnn"] = trainer
    trainer_spec.loader.exec_module(trainer)
    split, cache_validation, cache_rows, order = batch_helper.validate_order_and_sources(
        root, b8_contract, trainer
    )
    ancestry = batch_helper.verify_ancestry(root, b8_contract, cache_validation["metadata"])
    if (
        order["first_batch_members"][0] != contract["sample"]["source_member"]
        or order["first_batch_seq_ix"][0] != contract["sample"]["seq_ix"]
    ):
        raise ValueError(
            "strict cache validation selected a different source-qualified profile sequence"
        )
    mean, std = batch_helper.load_bound_norm(root, b8_contract, split)
    batch, _, loader_seconds, batch_bytes = batch_helper.build_first_batch(
        root, b8_contract, trainer, split, cache_validation, cache_rows, order
    )
    verified_target_files = cache_validation["verified_files"]
    del split, cache_validation, cache_rows

    checkpoint_path = root / contract["control"]["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != 0 or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("the frozen control checkpoint is not its fixed terminal epoch 0")
    base_state = {
        name: value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        for name, value in checkpoint["model"].items()
    }
    if not finite_tensors(base_state, torch):
        raise FloatingPointError("frozen H256×2 CP0 contains nonfinite tensor values")
    if (
        "mean" not in base_state
        or "std" not in base_state
        or tensor_identity(base_state["mean"], torch) != contract["normalization"]["hashes"]["mean"]
        or tensor_identity(base_state["std"], torch) != contract["normalization"]["hashes"]["std"]
    ):
        raise ValueError(
            "fitted control checkpoint does not carry the pinned common mean/std buffers"
        )
    del checkpoint

    from wnn.resources import configure_torch

    configure_torch()
    if not torch.cuda.is_available() or nvidia_smi_used_bytes() >= OBSERVED_VRAM_LIMIT_BYTES:
        raise RuntimeError("observed VRAM is already at or above the declared 5 GB ceiling")
    trainer.HEURISTICS = False
    trainer.RAW_ONLY = False
    trainer.ONLINE_NORM = False
    trainer.RESEARCH_FEATURES = "none"
    trainer.PRIVILEGED_MODE = "none"
    trainer.INPUT_PROPENSITY = None
    from wnn.propensity import load_propensity

    propensity_model = load_propensity(
        root / contract["propensity"]["checkpoint"], torch.device("cuda")
    )
    x, inputs, targets, supervision = initialize_inputs(trainer, batch, propensity_model, torch)
    weights = supervision.pop("weights")
    # Establish the zero-state reference using the exact frozen CP0 backbone.
    zero_backbone = make_base_model(trainer, contract, base_state, torch.device("cuda"))
    zero_backbone.requires_grad_(False)
    from wnn.state_initialization import InitializedGRU

    zero_model = InitializedGRU(
        zero_backbone, "zero", residual_bound=float(contract["residual_bound"])
    )
    with torch.no_grad():
        zero_prediction, _ = zero_model(inputs, None)
    zero_reference_sha = tensor_identity(zero_prediction[..., :2], torch)
    contract["zero_reference_prediction_sha256"] = zero_reference_sha
    zero_export = {"mode": "zero", "prediction_sha256": zero_reference_sha}
    del zero_model, zero_backbone, zero_prediction
    torch.cuda.empty_cache()

    output_dir = output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_artifacts = [
        output_dir / f"INITIALIZER_{mode.upper()}_NATIVE_R1.npz" for mode in PROFILE_MODES[1:]
    ] + [output_dir / f"INITIALIZER_{mode.upper()}_R1.pt" for mode in PROFILE_MODES[1:]]
    if any(path.exists() for path in expected_artifacts):
        raise FileExistsError(
            "initializer profile artifacts already exist; preserve failed attempts"
        )
    profile_results = []
    for mode in PROFILE_MODES[1:]:
        current = nvidia_smi_used_bytes()
        if current >= OBSERVED_VRAM_LIMIT_BYTES:
            raise RuntimeError("observed VRAM reached the ceiling before the next initializer arm")
        result = profile_mode(
            mode,
            trainer,
            base_state,
            x,
            inputs,
            targets,
            weights,
            contract,
            output_dir,
            torch,
            np,
        )
        result["observed_vram_bytes_before_arm"] = current
        result["supervision"] = {
            "total_weight_mass": supervision["total_weight_mass"],
            "target_formula": supervision["target_formula"],
            "row_weight": supervision["row_weight"],
            "input_sha256": supervision["input_sha256"],
            "target_sha256": supervision["target_sha256"],
            "weight_sha256": supervision["weight_sha256"],
            "finite_nonnegative_weights": supervision["finite_nonnegative_weights"],
        }
        profile_results.append(result)
        torch.cuda.empty_cache()
    output_result = {
        "format": RESULT_FORMAT,
        "version": 1,
        "contract_sha256": contract_sha256,
        "source_sha256": source_manifest(script, repo),
        "control": contract["control"],
        "cache_ancestry": ancestry,
        "normalization": contract["normalization"],
        "sample": contract["sample"],
        "supervision": {
            "total_weight_mass": supervision["total_weight_mass"],
            "target_formula": supervision["target_formula"],
            "row_weight": supervision["row_weight"],
            "input_sha256": supervision["input_sha256"],
            "target_sha256": supervision["target_sha256"],
            "weight_sha256": supervision["weight_sha256"],
            "finite_nonnegative_weights": supervision["finite_nonnegative_weights"],
        },
        "profile_scope": contract["profile_scope"],
        "zero_reference": zero_export,
        "arms": profile_results,
        "batch_loader_seconds": loader_seconds,
        "B8_returned_batch_tensor_bytes": batch_bytes,
        "strict_cache_verified_files": verified_target_files,
        "host_memory_current_bytes": current_cgroup_bytes("memory.current"),
        "host_memory_peak_bytes": current_cgroup_bytes("memory.peak"),
        "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "observed_vram_bytes_at_completion": nvidia_smi_used_bytes(),
        "quality_claim": None,
        "interpretation": "resource-only one-update complete-sequence initializer smoke; not a fit, score, or architecture selection",
    }
    if output_result["observed_vram_bytes_at_completion"] >= OBSERVED_VRAM_LIMIT_BYTES:
        raise RuntimeError("observed VRAM reached the fixed 5 GB ceiling at profile completion")
    output.write_text(
        json.dumps(output_result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_result


def parse_args() -> argparse.Namespace:
    """Parse one of the two explicit profile modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run static preflight or the separately admitted actual-data profile."""
    args = parse_args()
    root = args.data_root.resolve(strict=True)
    if os.environ.get("WNN_PROJECT_ROOT") != str(root):
        raise ValueError("WNN_PROJECT_ROOT must be exactly the declared immutable data/source root")
    contract_path = args.contract.resolve(strict=True)
    contract, raw = bounded_json(contract_path, MAX_CONTRACT_BYTES, "initializer profile contract")
    if root != Path(contract["data_root"]).resolve(strict=True):
        raise ValueError("--data-root differs from the frozen initializer profile contract")
    if args.preflight_only:
        static_preflight(Path(__file__).resolve(), contract_path, args.output.resolve())
        return 0
    contract_sha = sha256_bytes(raw)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"profile result path already exists; preserve attempts: {output}")
    static_preflight(Path(__file__).resolve(), contract_path, None)
    execute_profile(Path(__file__).resolve(), contract, contract_sha, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
