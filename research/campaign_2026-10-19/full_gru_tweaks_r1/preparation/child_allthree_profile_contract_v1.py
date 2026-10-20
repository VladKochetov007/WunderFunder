"""Pinned identities and warm-start invariants for the all-three child profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PARENT_RUN = "fg19_parent_h256x2_f0_s1001_fit_r2_v1"
PARENT_UNIT = "computer-vision-homework-fg19-parent-fit-r2-v1.service"
PARENT_INVOCATION = "5186b3016ff54b1fa754c10a1f87b3e4"
PARENT_EP3_SHA256 = "6027fc1db38fbc82d7cf1bb8c48dcb89f75bcc57ff91e9a7127f803cea11ef45"
PARENT_SPLIT_SHA256 = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
DICTIONARY_ACCEPTANCE_SHA256 = "4802815134d09bbbfeb697689a17ac82bcbbd6541e7969d5fec56867ecd3066f"
DICTIONARY_SHA256 = "9cf76503b579d2a88e7342b5f727e1ad5b3945c0938f1ae62d9738442a68b659"
DICTIONARY_BYTES = 22557
PROFILE_RUN = "fg19_child_allthree_f0_s1001_profile_v1"
SOURCE_COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"

# `Any` is limited to trusted tensor-state mappings returned by PyTorch serialization.


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: str, label: str) -> str:
    """Reject missing, malformed, or placeholder SHA-256 identities."""
    if (
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or value == "0" * 64
    ):
        raise ValueError(f"{label} must be an explicit lowercase SHA-256 digest")
    return value


def tensor_fingerprint(state: Mapping[str, Any]) -> str:
    """Hash names, tensor metadata, and bytes in stable key order."""
    import torch

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"model state entry {name!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        descriptor = json.dumps(
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_parent_extension(
    parent_state: Mapping[str, Any], child_state: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove the 211→467 inputs and 23→25 outputs preserve the parent function."""
    import torch

    if set(parent_state) != set(child_state):
        raise ValueError("child model state keys differ from the parent checkpoint")
    parent_input, child_input = parent_state.get("inp.weight"), child_state.get("inp.weight")
    parent_head, child_head = parent_state.get("head.2.weight"), child_state.get("head.2.weight")
    parent_bias, child_bias = parent_state.get("head.2.bias"), child_state.get("head.2.bias")
    values = (parent_input, child_input, parent_head, child_head, parent_bias, child_bias)
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise ValueError("parent/child projection or output head tensors are missing")
    if tuple(parent_input.shape) != (256, 211) or tuple(child_input.shape) != (256, 467):
        raise ValueError("profile model must preserve the 211→467 input projection layout")
    if tuple(parent_head.shape) != (23, 256) or tuple(child_head.shape) != (25, 256):
        raise ValueError("profile model must preserve the 23→25 output-head layout")
    if tuple(parent_bias.shape) != (23,) or tuple(child_bias.shape) != (25,):
        raise ValueError("parent/child output biases do not match 23→25 extension")
    for name, width, suffix in (("mean", 211, 0.0), ("std", 211, 1.0)):
        before, after = parent_state.get(name), child_state.get(name)
        if not isinstance(before, torch.Tensor) or not isinstance(after, torch.Tensor):
            raise ValueError(f"parent/child {name} normalization buffer is missing")
        if tuple(before.shape) != (width,) or tuple(after.shape) != (467,):
            raise ValueError(f"{name} normalization buffers do not match 211→467 layout")
        if not torch.equal(before.detach().cpu(), after[:width].detach().cpu()):
            raise ValueError(f"parent {name} normalization prefix changed")
        expected = torch.full_like(after[width:], suffix)
        if not torch.equal(after[width:].detach().cpu(), expected.detach().cpu()):
            raise ValueError(f"child {name} normalization suffix is not {suffix:g}")
    for name, before, after, width in (
        ("inp.weight", parent_input, child_input, 211),
        ("head.2.weight", parent_head, child_head, 23),
        ("head.2.bias", parent_bias, child_bias, 23),
    ):
        prefix = after[:, :width] if name == "inp.weight" else after[:width]
        suffix = after[:, width:] if name == "inp.weight" else after[width:]
        if not torch.equal(before.detach().cpu(), prefix.detach().cpu()):
            raise ValueError(f"parent prefix changed in {name}")
        if not torch.equal(suffix.detach().cpu(), torch.zeros_like(suffix).cpu()):
            raise ValueError(f"new child values in {name} are not zero initialized")
    special = {"inp.weight", "mean", "std", "head.2.weight", "head.2.bias"}
    for name in set(parent_state) - special:
        before, after = parent_state[name], child_state[name]
        if not isinstance(before, torch.Tensor) or not isinstance(after, torch.Tensor):
            raise TypeError(f"state entry {name!r} is not a tensor")
        if tuple(before.shape) != tuple(after.shape) or not torch.equal(
            before.detach().cpu(), after.detach().cpu()
        ):
            raise ValueError(f"parent trunk or auxiliary tensor changed in {name}")
    if not all(torch.isfinite(value).all().item() for value in child_state.values()):
        raise FloatingPointError("warm-loaded child contains nonfinite state")
    return {
        "parent_state_fingerprint": tensor_fingerprint(parent_state),
        "initial_child_state_fingerprint": tensor_fingerprint(child_state),
        "parent_input_width": 211,
        "child_input_width": 467,
        "parent_output_width": 23,
        "child_output_width": 25,
        "parent_trunk_and_prefix_equal": True,
        "new_input_columns_zero": True,
        "new_logvariance_rows_zero": True,
        "normalization_prefix_equal": True,
        "normalization_suffix_mean_zero_std_one": True,
        "finite_initial_state": True,
    }


def validate_parent_acceptance(
    path: Path, expected_sha256: str, checkpoint_sha256: str
) -> dict[str, Any]:
    """Require the exact accepted native parent fit and fixed epoch-3 checkpoint."""
    expected_sha256 = require_sha256(expected_sha256, "parent acceptance SHA-256")
    if sha256_file(path) != expected_sha256:
        raise ValueError("parent terminal acceptance bytes differ from the expected digest")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    native = receipt.get("native_unit")
    checkpoint = receipt.get("files", {}).get("model_ep3.pt", {})
    typed_exit = (("main_pid", 0), ("exec_main_code", 1), ("exec_main_status", 0))
    if any(type(receipt.get(key)) is not int or receipt[key] != value for key, value in typed_exit):
        raise ValueError("parent acceptance lacks typed successful native terminal fields")
    if (
        receipt.get("status") != "ACCEPTED"
        or receipt.get("selected_epoch") != 3
        or receipt.get("successful_updates") != 29840
        or receipt.get("nonfinite_updates") != 0
        or not receipt.get("strict_fp32_finite_model_optimizer_verified")
        or not isinstance(native, dict)
        or native.get("unit") != PARENT_UNIT
        or native.get("InvocationID") != PARENT_INVOCATION
        or native.get("ActiveState") != "active"
        or native.get("SubState") != "exited"
        or native.get("MainPID") != "0"
        or native.get("ExecMainCode") != "1"
        or native.get("ExecMainStatus") != "0"
        or checkpoint.get("sha256") != checkpoint_sha256
    ):
        raise ValueError("parent acceptance is not the accepted fixed-epoch-3 fit identity")
    return receipt


def validate_exact_stop(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Require exact stop of the accepted parent unit and no remaining processes."""
    expected_sha256 = require_sha256(expected_sha256, "parent exact-stop SHA-256")
    if sha256_file(path) != expected_sha256:
        raise ValueError("parent exact-stop bytes differ from the expected digest")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    before, after, stopped = (
        receipt.get("before", {}),
        receipt.get("after", {}),
        receipt.get("stop", {}),
    )
    if (
        receipt.get("unit") != PARENT_UNIT
        or receipt.get("invocation_id") != PARENT_INVOCATION
        or before.get("InvocationID") != PARENT_INVOCATION
        or before.get("ActiveState") != "active"
        or before.get("SubState") != "exited"
        or before.get("MainPID") != "0"
        or stopped.get("returncode") != 0
        or after.get("LoadState") != "not-found"
        or after.get("ActiveState") != "inactive"
        or after.get("SubState") != "dead"
        or after.get("MainPID") != "0"
        or receipt.get("populated_after") != {}
    ):
        raise ValueError("exact-stop receipt does not prove the parent fit is stopped")
    return receipt


def validate_profile_no_short_circuit(args: argparse.Namespace) -> None:
    """Reject canonical trainer modes that return before the profile can update."""
    if args.student_frontier_preflight_only or args.calibrate_distill_reference:
        raise ValueError("profile cannot enable trainer preflight or calibration short-circuits")


def validate_inherited_preprocessing(args: argparse.Namespace) -> None:
    """Require the parent checkpoint's saved input normalization to be inherited."""
    if args.inherit_init_preprocessing is not True:
        raise ValueError("profile requires inherited parent preprocessing")


def validate_profile_args(
    args: argparse.Namespace, expected_cache: Path, expected_distill_cache_root: Path
) -> None:
    """Reject drift from the one bounded source-bound all-three profile."""
    expected = {
        "name": PROFILE_RUN,
        "device": "cuda",
        "seed": 1001,
        "train_seqs": 10535,
        "train_source": "train+valid",
        "valid_fold": 0,
        "valid_repeat": 1,
        "valid_train_fraction": 1.0,
        "batch_order": "seeded",
        "batch": 16,
        "chunk": 2000,
        "update_rows": 0,
        "profile_warmup": 2,
        "profile_steps": 4,
        "max_batches": 6,
        "epochs": 6,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 5,
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "hidden": 256,
        "layers": 2,
        "head_layers": 1,
        "dropout": 0.0,
        "init": f"{PARENT_RUN}@ep3",
        "inherit_init_preprocessing": True,
        "representation_mode": "ema_sparse",
        "mean_variance_head": 1,
        "mean_variance_objective": "nll",
        "loss": "wmse",
        "loss_norm": "chunk",
        "distill_cache_variant": "homogeneous",
        "distill_cache_scale": 1.0,
        "distill_alpha": 0.9,
        "distill_alpha_far": None,
        "distill_boot_h": 0,
        "distill_reanchor": 0.0,
        "distill_family_weight": 0.0,
        "aux_struct": 0.3,
        "aux_next": 0.3,
        "aux": "",
        "aux_next_k": "",
        "lr": 0.0001,
        "wd": 0.01,
        "beta2": 0.999,
        "sched": "onecycle",
        "warmup_frac": 0.05,
        "clip": 1.0,
        "precision": "fp32",
        "deterministic": False,
        "optim": "adamw",
        "numeric_map": "none",
        "research_features": "none",
        "privileged_mode": "none",
        "readout_activation": "gelu",
        "rnn_init": "default",
        "qat": "none",
        "recurrent_rank": 0,
        "final_model_adaptation": False,
        "final_distill_initializer": False,
        "init_qmc": False,
        "prop_cache": False,
        "mask_weight": False,
        "class_weight": "",
        "class_balance": False,
        "train_classes": "",
        "valid_true_mask": -1.0,
        "since_weight": 0.0,
        "since_floor": 0.25,
        "episode_weight": 0.0,
        "episode_floor": 0.5,
        "distill_denominator": "chunk",
        "distill_reference_mass": 0.0,
        "antithetic": False,
    }
    validate_inherited_preprocessing(args)
    for key, value in expected.items():
        if getattr(args, key) != value:
            raise ValueError(f"all-three profile requires {key}={value!r}")
    validate_profile_no_short_circuit(args)
    if not args.no_eval_during_fit or not args.grad_stats:
        raise ValueError("profile requires no evaluation and canonical FP32 gradient evidence")
    forbidden = (
        "save_training_state",
        "resume_state",
        "parity_dump",
        "fast",
        "sync_free",
        "fused_adam",
        "pin_memory",
        "ema_teacher",
        "eval_ema",
        "swa_frac",
        "td_alpha",
        "aux_quote",
        "aux_tier",
        "aux_mask",
        "aux_sign",
        "include_heldout_valid",
        "epoch_samples",
        "propensity",
        "prop_input",
        "distill",
        "heuristics",
        "raw_only",
        "online_norm",
        "init_qmc",
        "final_model_adaptation",
        "final_distill_initializer",
        "prop_cache",
    )
    if any(getattr(args, name) for name in forbidden):
        raise ValueError("profile trainer flags enable an unbound mode or persistent output")
    if args.representation_cache is None or Path(args.representation_cache).resolve(
        strict=True
    ) != expected_cache.resolve(strict=True):
        raise ValueError("training argv representation cache differs from the bound artifact")
    if Path(args.distill_cache_root).resolve(strict=True) != expected_distill_cache_root.resolve(
        strict=True
    ):
        raise ValueError("Main T3 cache path differs from the fresh admission pin")
