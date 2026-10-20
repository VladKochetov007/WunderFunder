"""V1 bounded integrity check for the completed seed-4 strict-FP32 teacher fit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from propensity_terminal_support import (
    _group_identity,
    _hash_file,
    _json_file,
    _load_torch_file,
    _numpy_safe_globals,
    _safe_relative,
    _stable_file,
    _tensor_summary,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

RUN = "fg19_t3_h384x2_e3_s4_v1"
UNIT = "computer-vision-homework-fg19-teacher-s4-v1.service"
INVOCATION = "b6d71599e0044970b8641e3e241c40a4"
COMMIT = "af21d6d8409c46b36830f0bdddd50e67ee17025f"
RECIPE_SHA = "c460b098d09773786552e5b2c98596fab3319a05108d49bd5693da43ee8a7d88"
MANIFEST_SHA = "0c83367d4ee3afd7e9f382111060d8c125b52420ce27cd5cc309d338287af7fd"
ADMISSION_SHA = "0f68060beeb73a0fc92892dfa9621fa06a28a6a6f9d175f5adb87ae1fc2852c2"
LAUNCH_SHA = "f303534d17c770fd993d979ca407b050a8e8a1207bbe96a32224a392448b16f9"
LAUNCH_ACCEPTANCE_SHA = "532917357d95b15be5dfdbcd4707643eac731c6b40f8d11f3ce42ea9a2bfbfca"
SPLIT_SHA = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
PROPENSITY_SHA = "6facf4ca9045118b8e18ca590053c97e2f5f5122c92f228f75002f0c76a2800e"
PROFILE_SHA = "d19eed47c0e447810d35e1661b91546004c8aabc42e5aca7f1928fb2eb53ab37"
UPDATES = 22380
PER_EPOCH = 7460
BATCHES = 373
MODEL_SHAPES = {
    "mean": (211,),
    "std": (211,),
    "inp.weight": (384, 211),
    "inp.bias": (384,),
    "gru.weight_ih_l0": (1152, 384),
    "gru.weight_hh_l0": (1152, 384),
    "gru.bias_ih_l0": (1152,),
    "gru.bias_hh_l0": (1152,),
    "gru.weight_ih_l1": (1152, 384),
    "gru.weight_hh_l1": (1152, 384),
    "gru.bias_ih_l1": (1152,),
    "gru.bias_hh_l1": (1152,),
    "head.0.weight": (384, 768),
    "head.0.bias": (384,),
    "head.2.weight": (23, 384),
    "head.2.bias": (23,),
}
PARAMETER_COUNT = 2_159_639
OPTIMIZER_GROUP_PARAMETER_SHAPES = (
    ((384, 211), (384,), (1152,), (1152,), (1152,), (1152,), (384,), (23, 384), (23,)),
    ((1152, 384), (1152, 384), (1152, 384), (1152, 384)),
    ((384, 768),),
)
CRITICAL_ARGS = (
    ("name", RUN),
    ("seed", 4),
    ("device", "cuda"),
    ("arch", "gru"),
    ("cell", "gru"),
    ("enc", "linear"),
    ("hidden", 384),
    ("encoder_width", None),
    ("recurrent_width", None),
    ("readout_width", None),
    ("layers", 2),
    ("head_layers", 1),
    ("readout_activation", "gelu"),
    ("numeric_map", "none"),
    ("epochs", 3),
    ("checkpoint_policy", "fixed"),
    ("checkpoint_epoch", 2),
    ("batch", 32),
    ("chunk", 1000),
    ("optim", "muon"),
    ("muon_scope", "all"),
    ("muon_ortho_dtype", "fp32"),
    ("strict_muon_fp32", True),
    ("lr", 0.001),
    ("wd", 0.01),
    ("beta2", 0.999),
    ("sched", "onecycle"),
    ("warmup_frac", 0.05),
    ("clip", 1.0),
    ("loss", "wmse"),
    ("loss_norm", "chunk"),
    ("precision", "fp32"),
    ("propensity", "fg19_propensity_clean_f0_seed0_e4"),
    ("prop_power", 2.0),
    ("prop_eps", 0.03),
    ("valid_true_mask", 0.03),
    ("aux_struct", 0.3),
    ("aux_next", 0.3),
    ("since_weight", 32.0),
    ("since_floor", 0.25),
    ("train_source", "train+valid"),
    ("eval_source", "vfold"),
    ("n_valid_folds", 4),
    ("split_manifest_sha256", SPLIT_SHA),
    ("include_heldout_valid", False),
    ("eval_ema", False),
    ("no_eval_during_fit", True),
    ("save_training_state", True),
    ("grad_stats", True),
    ("sync_free", True),
    ("pin_memory", True),
    ("prop_cache", False),
    ("fast", False),
    ("fused_adam", False),
    ("max_batches", 0),
    ("profile_steps", 0),
    ("deterministic", False),
    ("activation_backend", "none"),
    ("init", ""),
    ("resume_state", None),
    ("init_qmc", False),
    ("rnn_init", "default"),
    ("dropout", 0.0),
    ("head_lr_mult", 1.0),
    ("optim_switch", ""),
    ("momentum", 0.0),
    ("no_gradient_clip", False),
    ("batch_order", "random"),
    ("antithetic", False),
    ("thin_prob", 0.0),
    ("mask_weight", False),
    ("valid_repeat", 1),
    ("valid_train_fraction", 1.0),
    ("class_balance", False),
    ("train_classes", ""),
    ("aux", ""),
    ("aux_weight", 0.5),
    ("aux_tier", 0.0),
    ("aux_mask", 0.0),
    ("aux_sign", 0.0),
    ("aux_quote", 0.0),
    ("aux_next_k", ""),
    ("quote_cache_root", ""),
    ("spread_cache_root", ""),
    ("spread_aux_mode", "original"),
    ("mean_variance_head", 0),
    ("mean_variance_objective", "wmse"),
    ("representation_mode", None),
    ("swa_frac", 0.0),
    ("ema_teacher", 0.0),
    ("td_alpha", 0.0),
    ("valid_fold", 0),
    ("research_features", "none"),
    ("prop_input", ""),
    ("heuristics", False),
    ("raw_only", False),
    ("online_norm", False),
)


def _eq(actual: object, expected: object, label: str) -> None:
    """Raise a focused error when a terminal field differs from its binding."""
    if actual != expected:
        raise ValueError(f"{label} differs")


def _teacher_safe_globals(np: object) -> list[object]:
    """Allow only the observed NumPy RNG and PyTorch version metadata classes."""
    return [*_numpy_safe_globals(np), torch.torch_version.TorchVersion]


def _check_exit(evidence: Mapping[str, object], launch: Mapping[str, object]) -> None:
    """Bind successful native exit evidence to the admitted invocation."""
    fields = (("unit", UNIT), ("invocation_id", INVOCATION))
    for item in (launch, evidence):
        for key, expected in fields:
            _eq(item.get(key), expected, f"{key} identity")
    _eq(evidence.get("run_name"), RUN, "run name")
    _eq(
        (evidence.get("active_state"), evidence.get("sub_state")),
        ("active", "exited"),
        "native unit state",
    )
    for key, expected in (("main_pid", 0), ("exec_main_code", 1), ("exec_main_status", 0)):
        if type(evidence.get(key)) is not int:
            raise ValueError(f"{key} must be an integer from native exit evidence")
        _eq(evidence[key], expected, key)


def _check_source(
    root: Path, manifest_path: Path, admission: Mapping[str, object]
) -> dict[str, object]:
    """Verify every source manifest byte against the admitted digest map."""
    manifest, digest = _json_file(manifest_path)
    _eq(digest, MANIFEST_SHA, "source manifest SHA")
    _eq(manifest.get("source_commit"), COMMIT, "source commit")
    entries, admitted = manifest.get("files"), admission.get("source_files")
    if not isinstance(entries, list) or not isinstance(admitted, dict):
        raise ValueError("source manifest schema is malformed")
    if any(not isinstance(item, dict) or not isinstance(item.get("path"), str) for item in entries):
        raise ValueError("source manifest contains a malformed entry")
    mapped = {item["path"]: item["sha256"] for item in entries if isinstance(item, dict)}
    _eq(mapped, admitted, "source/admission map")
    _eq((len(entries), len(mapped)), (164, 164), "source file count/uniqueness")
    source_bytes = 0
    for item in entries:
        size = item.get("bytes")
        if type(size) is not int or size < 0:
            raise ValueError("source manifest byte counts must be nonnegative integers")
        source_bytes += size
        if source_bytes > 128_000_000:
            raise ValueError("pinned source tree exceeds the bounded hashing budget")
        path = _safe_relative(root, str(item["path"]))
        actual, info = _hash_file(path, size)
        _eq((actual, info["bytes"]), (item["sha256"], size), str(item["path"]))
    return {"commit": COMMIT, "files": len(mapped), "manifest_sha256": digest}


def _check_components(
    components: dict[str, dict[int, int]], train: list[int], valid: list[int], evaluation: list[int]
) -> None:
    """Reject overlap by globally assigned component IDs across source splits."""
    fit = {components["train"][int(group)] for group in train}
    fit.update(components["valid"][int(group)] for group in valid)
    held = {components["valid"][int(group)] for group in evaluation}
    if not fit.isdisjoint(held):
        raise ValueError("fit/evaluation components overlap")


def _expected_membership(
    args: Mapping[str, object], split: Mapping[str, object]
) -> tuple[list[int], list[int], list[int]]:
    """Reproduce the pinned loader's fixed train, fit-valid, and eval ordering."""
    _eq(
        (args.get("train_source"), args.get("valid_fold"), args.get("valid_train_fraction")),
        ("train+valid", 0, 1.0),
        "membership selection options",
    )
    fit, evaluation = split.get("fit"), split.get("evaluation")
    if not isinstance(fit, dict) or not isinstance(evaluation, dict):
        raise ValueError("campaign split membership is malformed")
    train_groups, valid_groups, eval_groups = (
        fit.get("train"),
        fit.get("valid"),
        evaluation.get("valid"),
    )
    if not all(isinstance(groups, list) for groups in (train_groups, valid_groups, eval_groups)):
        raise ValueError("campaign split group lists are missing")
    for groups in (train_groups, valid_groups, eval_groups):
        if any(type(group) is not int for group in groups) or len(set(groups)) != len(groups):
            raise ValueError("campaign split groups must be unique integer IDs")
    train_count, valid_count = args.get("train_seqs"), args.get("valid_seqs")
    full_valid = args.get("full_valid")
    if type(train_count) is not int or train_count < 0:
        raise ValueError("train sequence cap is invalid")
    if type(valid_count) is not int or valid_count < 0 or type(full_valid) is not bool:
        raise ValueError("evaluation sequence cap is invalid")
    if args.get("eval_source") != "vfold":
        raise ValueError("evaluation source differs from fixed campaign fold")
    train_order = (
        np.random.default_rng(int(args["seed"])).permutation(train_groups)[:train_count].tolist()
    )
    eval_count = len(eval_groups) if full_valid else min(valid_count, len(eval_groups))
    eval_order = np.random.default_rng(12345).permutation(eval_groups)[:eval_count].tolist()
    # In the pinned recipe, source uses all campaign valid groups and fraction=1,
    # so canonical select_membership keeps this list in split-manifest order.
    return train_order, list(valid_groups), eval_order


def _check_membership_order(
    actual: tuple[object, object, object], expected: tuple[list[int], list[int], list[int]]
) -> None:
    """Require exact ordered membership, even if a forged digest matches a permutation."""
    labels = ("train", "fit-valid", "evaluation")
    for label, observed, canonical in zip(labels, actual, expected, strict=True):
        if not isinstance(observed, list) or any(type(group) is not int for group in observed):
            raise ValueError(f"{label} membership must be an ordered integer list")
        _eq(observed, canonical, f"{label} ordered membership")


def _check_membership_identities(identity: object, train: list[int], valid: list[int]) -> None:
    """Verify self-reported membership hashes and their combined row count."""
    if not isinstance(identity, dict):
        raise ValueError("fit membership identity is malformed")
    _eq(identity.get("train"), _group_identity(train, np), "train membership hash")
    _eq(identity.get("valid"), _group_identity(valid, np), "valid membership hash")
    _eq(identity.get("dataset_length"), len(train) + len(valid), "fit dataset length")


def _check_model_layout(model: object) -> tuple[int, int]:
    """Require the exact 211-input legacy GRU tensor inventory and output width."""
    if not isinstance(model, dict) or set(model) != set(MODEL_SHAPES):
        raise ValueError("checkpoint tensor inventory differs from the bound GRU")
    parameters = 0
    for name, expected_shape in MODEL_SHAPES.items():
        summary = _tensor_summary(model[name], torch)
        _eq(tuple(summary["shape"]), expected_shape, f"tensor shape: {name}")
        if name not in {"mean", "std"}:
            parameters += int(np.prod(expected_shape))
    _eq(parameters, PARAMETER_COUNT, "parameter count")
    return len(model), parameters


def _check_membership(
    provenance: Mapping[str, object],
    split: Mapping[str, object],
    args: Mapping[str, object],
) -> dict[str, int]:
    """Compare loader-ordered row groups with the fixed component-disjoint split."""
    fit, seqs = provenance.get("fit_row_groups"), split.get("sequences")
    if not isinstance(fit, dict) or not isinstance(seqs, dict):
        raise ValueError("membership metadata is malformed")
    train, valid, evaluation = (
        fit.get("train"),
        fit.get("valid"),
        provenance.get("evaluation_row_groups"),
    )
    if not all(isinstance(rows, list) for rows in (train, valid, evaluation)):
        raise ValueError("membership vectors are missing")
    expected = _expected_membership(args, split)
    _check_membership_order((train, valid, evaluation), expected)
    _check_membership_identities(provenance.get("fit_membership_identity"), train, valid)
    components = {
        source: {int(row["row_group"]): int(row["component"]) for row in rows}
        for source, rows in seqs.items()
    }
    _check_components(components, train, valid, evaluation)
    counts = {"fit_train": len(train), "fit_valid": len(valid), "evaluation": len(evaluation)}
    _eq(counts, {"fit_train": 10535, "fit_valid": 1405, "evaluation": 468}, "split counts")
    return counts


def _check_normalization_identity(
    model: object,
    declared: object,
    tensor_identity: object,
    label: str,
) -> None:
    """Bind the two stored normalization buffers to their canonical provenance hashes."""
    if not isinstance(model, dict) or not callable(tensor_identity):
        raise ValueError(f"{label} normalization tensors are missing")
    if not isinstance(declared, dict):
        raise ValueError(f"{label} normalization identity is missing")
    actual = {}
    for name in ("mean", "std"):
        if name not in model:
            raise ValueError(f"{label} normalization tensor {name} is missing")
        actual[name] = tensor_identity(model[name])
    _eq(actual, declared, f"{label} checkpoint normalization identity")


def _check_counters(
    history: object, progress: Mapping[str, object], provenance: Mapping[str, object]
) -> None:
    """Reconcile the three fit-only epochs and total update counters."""
    diagnostics = provenance.get("update_diagnostics")
    if (
        not isinstance(history, list)
        or len(history) != 3
        or not isinstance(diagnostics, list)
        or len(diagnostics) != 3
    ):
        raise ValueError("three-epoch history or diagnostics are missing")
    _eq(progress.get("history"), history, "training history")
    _eq(progress.get("update_diagnostics"), diagnostics, "training diagnostics")
    for epoch, (record, item) in enumerate(zip(history, diagnostics, strict=True)):
        _eq((record.get("epoch"), record.get("evaluation")), (epoch, "withheld"), "history scope")
        if "weighted_pearson" in record:
            raise ValueError("fit history unexpectedly contains an evaluation score")
        _eq(
            record.get("update_diagnostics"),
            {k: v for k, v in item.items() if k != "epoch_index"},
            "history counters",
        )
        _eq(item.get("epoch_index"), epoch, "diagnostic order")
        counts, grads = item.get("epoch"), item.get("grad_stats")
        _eq(
            tuple(
                counts.get(k)
                for k in (
                    "optimizer_attempts",
                    "optimizer_completed_updates",
                    "completed_batches",
                    "amp_skipped_updates",
                )
            ),
            (PER_EPOCH, PER_EPOCH, BATCHES, 0),
            "epoch counters",
        )
        _eq(
            (grads.get("steps"), grads.get("nonfinite_steps")),
            (PER_EPOCH, 0),
            "gradient finiteness",
        )
    total = provenance.get("cumulative_updates")
    _eq(
        tuple(
            total.get(k)
            for k in (
                "optimizer_attempts",
                "optimizer_completed_updates",
                "completed_batches",
                "amp_skipped_updates",
            )
        ),
        (UPDATES, UPDATES, 3 * BATCHES, 0),
        "total counters",
    )
    _eq(
        (progress.get("step_count"), progress.get("cumulative_updates")),
        (UPDATES, total),
        "training-state counters",
    )


def _model_fingerprint(model: object) -> tuple[str, int]:
    """Hash tensor metadata/bytes without returning model values."""
    if not isinstance(model, dict) or not model:
        raise ValueError("checkpoint model state is missing")
    records = {}
    for name, tensor in model.items():
        item = _tensor_summary(tensor, torch)
        _eq(item["dtype"], "torch.float32", f"model dtype: {name}")
        records[name] = item
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), len(records)


def _check_optimizer(optimizer: object) -> dict[str, int]:
    """Require source-derived Muon groups, parameter shapes, steps, and finite state."""
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("param_groups"), list):
        raise ValueError("optimizer state is malformed")
    groups, states = optimizer["param_groups"], optimizer.get("state")
    if len(groups) != 3 or not isinstance(states, dict):
        raise ValueError("strict Muon groups differ")
    expected_group_flags = (
        {"muon": False, "blocks": 1, "muon_fp32_ortho": False},
        {"muon": True, "blocks": 3, "muon_fp32_ortho": True},
        {"muon": True, "blocks": 1, "muon_fp32_ortho": True},
    )
    parameter_ids: list[list[int]] = []
    for index, (group, flags, shapes) in enumerate(
        zip(groups, expected_group_flags, OPTIMIZER_GROUP_PARAMETER_SHAPES, strict=True)
    ):
        if not isinstance(group, dict):
            raise ValueError("optimizer parameter group is malformed")
        for name, expected in flags.items():
            _eq(group.get(name), expected, f"optimizer group {index} {name}")
        ids = group.get("params")
        if not isinstance(ids, list) or len(ids) != len(shapes):
            raise ValueError(f"optimizer group {index} parameter membership differs")
        if any(type(parameter_id) is not int for parameter_id in ids):
            raise ValueError("optimizer parameter identifiers must be integers")
        parameter_ids.append(ids)
        _eq(group.get("weight_decay"), 0.01, f"optimizer group {index} weight decay")
        _eq(group.get("eps"), 1e-8, f"optimizer group {index} epsilon")
        _eq(group.get("ns_steps"), 5, f"optimizer group {index} orthogonalization steps")
        _eq(group.get("rms"), 0.2, f"optimizer group {index} update scale")
        beta_pair = group.get("betas")
        if (
            not isinstance(beta_pair, (list, tuple))
            or len(beta_pair) != 2
            or beta_pair[1] != 0.999
            or not 0.0 < float(beta_pair[0]) < 1.0
        ):
            raise ValueError(f"optimizer group {index} beta settings differ")
        lr = group.get("lr")
        if not isinstance(lr, (float, int)) or not np.isfinite(lr) or lr <= 0:
            raise ValueError(f"optimizer group {index} learning rate is invalid")
    flat_ids = [parameter_id for ids in parameter_ids for parameter_id in ids]
    if len(set(flat_ids)) != 14 or set(states) != set(flat_ids):
        raise ValueError("optimizer state does not cover exactly 14 model parameters")

    floating_state_count = 0
    for group_index, (ids, shapes) in enumerate(
        zip(parameter_ids, OPTIMIZER_GROUP_PARAMETER_SHAPES, strict=True)
    ):
        for parameter_id, shape in zip(ids, shapes, strict=True):
            state = states[parameter_id]
            if not isinstance(state, dict):
                raise ValueError("optimizer parameter state is malformed")
            if group_index == 0:
                if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise ValueError("AdamW parameter state keys differ")
                if type(state["step"]) is not int or state["step"] != UPDATES:
                    raise ValueError("AdamW parameter step differs from completed updates")
                tensors = (state["exp_avg"], state["exp_avg_sq"])
            else:
                if set(state) != {"momentum_buffer"}:
                    raise ValueError("Muon parameter state keys differ")
                tensors = (state["momentum_buffer"],)
            for tensor in tensors:
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.dtype != torch.float32
                    or tuple(tensor.shape) != shape
                    or not bool(torch.isfinite(tensor).all().item())
                ):
                    raise ValueError(
                        "optimizer state tensor is not finite FP32 with expected shape"
                    )
                floating_state_count += 1
    _eq(floating_state_count, 23, "optimizer floating-state tensor count")
    return {
        "groups": len(groups),
        "adamw_parameters": len(parameter_ids[0]),
        "recurrent_muon_parameters": len(parameter_ids[1]),
        "interior_muon_parameters": len(parameter_ids[2]),
        "floating_tensors": floating_state_count,
    }


def _check_contract(
    state: dict[str, object],
    provenance: dict[str, object],
    admission: dict[str, object],
    split_sha: str,
    launch_args: Mapping[str, object],
) -> None:
    """Bind resume ancestry, data identities, full source map and trainer subset."""
    contract, pinned = state.get("contract"), admission["source_files"]
    if not isinstance(contract, dict):
        raise ValueError("resume contract is missing")
    _check_runtime(contract.get("runtime"))
    _eq(
        (contract.get("fit_membership"), contract.get("normalization")),
        (provenance.get("fit_membership_identity"), provenance.get("normalization_identity")),
        "resume identities",
    )
    _eq(
        contract.get("ancestors", {}).get("propensity_checkpoint_sha256"),
        PROPENSITY_SHA,
        "resume propensity",
    )
    _eq(
        (
            contract["total_steps"],
            contract["options"]["seed"],
            contract["options"]["epochs"],
            provenance["split_manifest_sha256"],
        ),
        (UPDATES, 4, 3, split_sha),
        "resume recipe/split",
    )
    excluded = {
        "name",
        "resume_state",
        "save_training_state",
        "stop_after_epoch",
        "split_manifest",
        "fit_groups_manifest",
    }
    expected_options = {key: value for key, value in launch_args.items() if key not in excluded}
    _eq(contract.get("options"), expected_options, "resume options vs launched CLI")
    for key in ("train", "valid"):
        expected = admission["data"][f"datasets/{key}.parquet"]
        _eq(
            contract["datasets"][key],
            {"bytes": expected["bytes"], "sha256": expected["registered_sha256"]},
            f"{key} dataset",
        )
    expected = {
        name.removeprefix("src/wnn/"): value
        for name, value in pinned.items()
        if name.startswith("src/wnn/") and name.endswith(".py")
    }
    expected["trainer"] = pinned["scripts/train_rnn.py"]
    _eq(contract.get("sources"), expected, "resume sources")
    impl = provenance.get("trainer_implementation")
    trainer = {
        name: value
        for name, value in pinned.items()
        if name == "scripts/train_rnn.py"
        or name in {"pyproject.toml", "uv.lock"}
        or name.startswith("src/wnn/training/")
        and name.endswith(".py")
    }
    _eq(impl.get("files"), trainer, "trainer source files")
    _eq(
        impl.get("sha256"),
        hashlib.sha256(
            json.dumps(trainer, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "trainer source digest",
    )


def _check_runtime(runtime: object) -> None:
    """Require the admitted single-thread strict-FP32 backend settings."""
    if not isinstance(runtime, dict):
        raise ValueError("resume runtime metadata is missing")
    if type(runtime.get("threads")) is not int:
        raise ValueError("runtime thread count must be an integer")
    _eq(runtime.get("threads"), 1, "runtime thread count")
    for name in ("matmul_tf32", "cudnn_tf32"):
        if type(runtime.get(name)) is not bool:
            raise ValueError(f"runtime {name} must be a boolean")
        _eq(runtime[name], False, f"runtime {name}")


def _check_checkpoint_hashes(
    provenance: Mapping[str, object], files: Mapping[str, tuple[str, Mapping[str, int]]]
) -> None:
    """Bind every published inference checkpoint to its initial bounded prehash."""
    recorded = provenance.get("checkpoint_sha256")
    if not isinstance(recorded, dict):
        raise ValueError("checkpoint digest map is missing")
    for name in ("model.pt", "model_ep0.pt", "model_ep1.pt", "model_ep2.pt"):
        _eq(recorded.get(name), files[name][0], f"checkpoint hash {name}")


def _launch_cli_args(launch: Mapping[str, object], source_root: Path) -> dict[str, object]:
    """Parse the actual launch command with the fully pinned pure-argparse CLI."""
    command = launch.get("command")
    if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
        raise ValueError("raw launch command is malformed")
    positions = [
        index for index, item in enumerate(command) if item.endswith("/scripts/train_rnn.py")
    ]
    if len(positions) != 1:
        raise ValueError("raw launch must name exactly one pinned trainer entrypoint")
    index = positions[0]
    script = _safe_relative(source_root, "scripts/train_rnn.py")
    _eq(Path(command[index]).resolve(strict=True), script, "launched trainer path")
    src = (source_root / "src").resolve(strict=True)
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if "wnn" in sys.modules:
        package_path = Path(sys.modules["wnn"].__file__).resolve(strict=True)
        if not package_path.is_relative_to(src):
            raise ValueError("wnn was imported from outside the pinned source tree")
    from wnn.training.cli import build_parser

    cli_path = Path(sys.modules["wnn.training.cli"].__file__).resolve(strict=True)
    _eq(cli_path, (src / "wnn/training/cli.py").resolve(strict=True), "training CLI source path")
    namespace = build_parser().parse_args(command[index + 1 :])
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(namespace).items()
    }


def _canonical_tensor_identity(source_root: Path):
    """Import the exact identity helper only after the complete source tree is verified."""
    source = (source_root / "src").resolve(strict=True)
    if "wnn" in sys.modules:
        package_path = Path(sys.modules["wnn"].__file__).resolve(strict=True)
        _eq(package_path, source / "wnn/__init__.py", "normalization package source path")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from wnn.run_identity import tensor_identity

    module_path = Path(sys.modules["wnn.run_identity"].__file__).resolve(strict=True)
    _eq(module_path, source / "wnn/run_identity.py", "normalization identity source path")
    return tensor_identity


def _effective_runtime_args(
    launch_args: Mapping[str, object], split: Mapping[str, object], split_sha: str
) -> dict[str, object]:
    """Apply only the deterministic campaign-split fields added after argparse."""
    _eq(split_sha, SPLIT_SHA, "campaign split digest")
    _eq((split.get("fold"), split.get("n_folds")), (0, 4), "campaign split fold")
    if not isinstance(launch_args.get("split_manifest"), str):
        raise ValueError("the admitted command must select the campaign split manifest")
    expected = dict(launch_args)
    expected.update(
        {
            "valid_fold": 0,
            "n_valid_folds": 4,
            "eval_source": "vfold",
            "split_manifest_sha256": split_sha,
        }
    )
    return expected


def _check_args(args: object, launch_args: Mapping[str, object]) -> None:
    """Validate fixed model, objective, optimizer, fit scope, and runtime flags."""
    if not isinstance(args, dict):
        raise ValueError("checkpoint arguments are missing")
    _eq(args, dict(launch_args), "checkpoint arguments vs launched CLI")
    for key, value in CRITICAL_ARGS:
        _eq(args.get(key), value, f"argument {key}")


def verify_terminal(
    run_dir: Path,
    source_root: Path,
    split_path: Path,
    manifest_path: Path,
    admission_path: Path,
    launch_path: Path,
    launch_acceptance_path: Path,
    terminal_path: Path,
) -> dict[str, object]:
    """Verify terminal evidence without inference, dataset reads, or optimizer execution."""
    admission, admission_sha = _json_file(admission_path)
    launch, launch_sha = _json_file(launch_path)
    launch_acceptance, launch_acceptance_sha = _json_file(launch_acceptance_path)
    terminal, _ = _json_file(terminal_path)
    _eq(
        (admission_sha, launch_sha, launch_acceptance_sha),
        (ADMISSION_SHA, LAUNCH_SHA, LAUNCH_ACCEPTANCE_SHA),
        "admission/launch bytes",
    )
    _eq(launch.get("unit"), UNIT, "raw launch unit")
    if type(launch.get("returncode")) is not int:
        raise ValueError("systemd-run launch return code must be an integer")
    _eq(launch.get("returncode"), 0, "systemd-run launch return code")
    _eq(
        (
            launch_acceptance.get("schema"),
            launch_acceptance.get("status"),
            launch_acceptance.get("unit"),
            launch_acceptance.get("invocation_id"),
            launch_acceptance.get("admission_sha256"),
            launch_acceptance.get("launch_sha256"),
            launch_acceptance.get("recipe_sha256"),
            launch_acceptance.get("profile_acceptance_sha256"),
            launch_acceptance.get("source_commit"),
            launch_acceptance.get("source_file_count"),
            launch_acceptance.get("expected_updates"),
        ),
        (
            "fg19.teacher-s4-launch-acceptance.v1",
            "ADMITTED_RUNNING_NOT_TERMINAL",
            UNIT,
            INVOCATION,
            admission_sha,
            launch_sha,
            RECIPE_SHA,
            PROFILE_SHA,
            COMMIT,
            164,
            UPDATES,
        ),
        "normalized launch acceptance binding",
    )
    _eq(
        (
            admission.get("recipe_sha256"),
            admission.get("source_commit"),
            admission.get("profile_acceptance_sha256"),
            admission.get("propensity", {}).get("model_sha256"),
        ),
        (RECIPE_SHA, COMMIT, PROFILE_SHA, PROPENSITY_SHA),
        "admitted ancestry",
    )
    _eq(launch.get("post", {}).get("InvocationID"), INVOCATION, "raw launch invocation")
    _check_exit(terminal, launch_acceptance)
    split, split_sha = _json_file(split_path)
    _eq(
        (split_sha, admission.get("split_identity", {}).get("sha256")),
        (SPLIT_SHA, SPLIT_SHA),
        "split bytes",
    )
    source = _check_source(source_root, manifest_path, admission)
    launch_args = _launch_cli_args(launch, source_root)
    tensor_identity = _canonical_tensor_identity(source_root)
    runtime_args = _effective_runtime_args(launch_args, split, split_sha)
    names = (
        "model.pt",
        "model_ep0.pt",
        "model_ep1.pt",
        "model_ep2.pt",
        "training_state.pt",
        "provenance.json",
        "history.json",
    )
    paths = {name: run_dir / name for name in names}
    limits = {
        "model.pt": 32_000_000,
        "model_ep0.pt": 32_000_000,
        "model_ep1.pt": 32_000_000,
        "model_ep2.pt": 32_000_000,
        "training_state.pt": 64_000_000,
        "provenance.json": 16_000_000,
        "history.json": 16_000_000,
    }
    files = {name: _hash_file(path, limits[name]) for name, path in paths.items()}
    if sum(info["bytes"] for _, info in files.values()) > 128_000_000:
        raise ValueError("aggregate terminal artifact size exceeds the bounded verifier budget")
    provenance, provenance_sha = _json_file(paths["provenance.json"])
    _eq(provenance_sha, files["provenance.json"][0], "provenance changed")
    raw, history_sha, _ = _stable_file(paths["history.json"], 16_000_000)
    _eq(history_sha, files["history.json"][0], "history changed")
    history = json.loads(raw)
    _eq(
        (provenance.get("selected_epoch"), provenance.get("propensity_checkpoint_sha256")),
        (2, PROPENSITY_SHA),
        "selected checkpoint/parent",
    )
    _check_args(provenance.get("args"), runtime_args)
    counts = _check_membership(provenance, split, runtime_args)
    checkpoints = []
    model_layout = None
    for name in ("model.pt", "model_ep2.pt"):
        payload, payload_sha, payload_info = _load_torch_file(
            paths[name], torch, allowlist=_teacher_safe_globals(np)
        )
        _eq((payload_sha, payload_info), files[name], f"stable {name} payload")
        _eq(payload.get("epoch"), 2, f"{name} epoch")
        _check_args(payload.get("args"), runtime_args)
        _eq(payload.get("n_in"), 211, f"{name} legacy input width")
        _eq(payload["model"]["head.2.weight"].shape[0], 23, f"{name} legacy output width")
        embedded = payload.get("provenance")
        for key in (
            "args",
            "split_manifest_sha256",
            "fit_row_groups",
            "evaluation_row_groups",
            "fit_membership_identity",
            "normalization_identity",
            "trainer_implementation",
            "propensity_checkpoint_sha256",
            "update_diagnostics",
            "cumulative_updates",
        ):
            _eq(embedded.get(key), provenance.get(key), f"embedded provenance {key}")
        _check_normalization_identity(
            payload.get("model"),
            provenance.get("normalization_identity"),
            tensor_identity,
            name,
        )
        layout = _check_model_layout(payload.get("model"))
        if model_layout is None:
            model_layout = layout
        else:
            _eq(layout, model_layout, "selected/epoch-2 model layout")
        checkpoints.append(_model_fingerprint(payload.get("model")))
    _eq(checkpoints[0], checkpoints[1], "selected/epoch-2 model")
    state, state_sha, state_info = _load_torch_file(
        paths["training_state.pt"], torch, allowlist=_teacher_safe_globals(np)
    )
    _eq((state_sha, state_info), files["training_state.pt"], "stable training-state payload")
    _eq(
        (state.get("format"), state.get("version"), state.get("boundary"), state.get("next_epoch")),
        ("wnn.epoch-training-state", 1, "epoch", 3),
        "resume boundary",
    )
    _eq((state.get("ema"), state.get("swa")), (None, None), "unexpected averaging state")
    _eq(_check_model_layout(state.get("model")), model_layout, "resume model layout")
    _eq(_model_fingerprint(state.get("model")), checkpoints[0], "resume/inference model")
    _check_normalization_identity(
        state.get("model"),
        provenance.get("normalization_identity"),
        tensor_identity,
        "resume",
    )
    optimizer = _check_optimizer(state.get("optimizer"))
    _check_counters(history, state.get("progress", {}), provenance)
    _check_contract(state, provenance, admission, split_sha, runtime_args)
    _check_checkpoint_hashes(provenance, files)
    return {
        "schema": "fg19.teacher-s4-terminal-check.v1",
        "status": "PASS",
        "run_name": RUN,
        "unit": UNIT,
        "invocation_id": INVOCATION,
        "launch_sha256": launch_sha,
        "launch_acceptance_sha256": launch_acceptance_sha,
        "source": source,
        "split_sha256": split_sha,
        "membership_counts": counts,
        "updates": UPDATES,
        "epochs": 3,
        "model_tensor_count": model_layout[0],
        "parameter_count": model_layout[1],
        "model_tensor_fingerprint": checkpoints[0][0],
        "optimizer": optimizer,
        "files": {
            name: {"sha256": pair[0], "bytes": pair[1]["bytes"]} for name, pair in files.items()
        },
    }


def main() -> int:
    """Run the checker and create a compact JSON receipt without overwriting."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "run-dir",
        "source-root",
        "split",
        "source-manifest",
        "admission",
        "launch",
        "launch-acceptance",
        "terminal-evidence",
        "output",
    ):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    result = verify_terminal(
        args.run_dir,
        args.source_root,
        args.split,
        args.source_manifest,
        args.admission,
        args.launch,
        args.launch_acceptance,
        args.terminal_evidence,
    )
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
