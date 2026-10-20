"""Bounded terminal integrity checker for the pinned F0 seed-1001 CONTROL fit."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import parent_fit_terminal_check_v3 as parent_v3
import torch
from propensity_terminal_support import _hash_file, _json_file, _stable_file, _tensor_summary

if TYPE_CHECKING:
    from collections.abc import Mapping

RUN = "fg19_child_control_f0_s1001_fit_v1"
UNIT = "computer-vision-homework-fg19-child-control-fit-v1.service"
INVOCATION = "e66e0d0703314e4fb5d47dcdbfcaf202"
COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
SOURCE_MANIFEST_SHA = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
ADMISSION_SHA = "f749d0c4ba4925506320f1a480d7e76164a2b0f84b7efac5b432cb18522c5973"
LAUNCH_SHA = "afd19428c8b6034a9dd40fac78d6adb78b70f3d95f1f964917cbadc828ed27e9"
LAUNCH_ACCEPTANCE_SHA = "caa7e1560171ee6813a6766649b82dc512432c240b400a286a11cea39ad29a15"
LAUNCH_BODY_SHA = "124b90f9cc76c17c9a62e22b934dc15e579417f0b769eb86f44334ba55f6b954"
SPLIT_SHA = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
TRAINER_ARGV_SHA = "69e735294550802364f29a94a3dbc68873aa2e4500fe543e4ffb1b90f594b9a2"
PARENT_RUN = "fg19_parent_h256x2_f0_s1001_fit_r2_v1"
PARENT_UNIT = "computer-vision-homework-fg19-parent-fit-r2-v1.service"
PARENT_INVOCATION = "5186b3016ff54b1fa754c10a1f87b3e4"
PARENT_ACCEPTANCE_SHA = "5b9ddc32f871f4dae29334b4f573d1e38efd9f8435cf2d7fde47e47f97e658b8"
PARENT_STOP_SHA = "15a1e42a90a0962c9b45b6043a7e1e152057848725391fffcd611862e5c6ffa8"
PARENT_CHECKPOINT_SHA = "6027fc1db38fbc82d7cf1bb8c48dcb89f75bcc57ff91e9a7127f803cea11ef45"
PARENT_CHECKPOINT_BYTES = 3_988_711
DICTIONARY_ACCEPTANCE_SHA = "4802815134d09bbbfeb697689a17ac82bcbbd6541e7969d5fec56867ecd3066f"
DICTIONARY_STOP_SHA = "2eb44a4381de06f28ed3d7438ace320ef3274eae37c89074fc481f58f5d88158"
DICTIONARY_SHA = "9cf76503b579d2a88e7342b5f727e1ad5b3945c0938f1ae62d9738442a68b659"
DICTIONARY_BYTES = 22_557
PROFILE_ACCEPTANCE_SHA = "9235b1670e80de8e18d322ab6e40dac102a7a8d8c028250f7b08262d8692c045"
PROFILE_STOP_SHA = "a0437f4da0b2534721909087bab1ddf40e068c4ba6911870472b7374e4bb7dce"
# Canonical digest of initial_state in the exact PROFILE_ACCEPTANCE_SHA receipt.
PROFILE_INITIAL_STATE_SHA = "254b29510a93a689ef3f259e65890fb6be76cfe112b483b15fd91763ea74c2cd"
PROFILE_UNIT = "computer-vision-homework-fg19-child-allthree-profile-v1.service"
PROFILE_INVOCATION = "1d62c3fb47ac4bd5be8e8ec8fe6759cd"
PROFILE_RUN = "fg19_child_allthree_f0_s1001_profile_v1"
UPDATES_PER_EPOCH = 7_460
BATCHES_PER_EPOCH = 746
EPOCHS = 6
UPDATES = 44_760
BATCHES = 4_476
FIT_TRAIN = 10_535
FIT_VALID = 1_405
EVALUATION = 468
MAX_JSON_BYTES = 16_000_000
MAX_MODEL_BYTES = 32_000_000
MAX_STATE_BYTES = 64_000_000
MAX_DICTIONARY_BYTES = 1_000_000
MAX_LOG_BYTES = 16_000_000
MAX_AGGREGATE_BYTES = 128_000_000
EXPECTED_TORCH = "2.14.0+cu130"

MODEL_SHAPES = {
    "mean": (467,),
    "std": (467,),
    "inp.weight": (256, 467),
    "inp.bias": (256,),
    "gru.weight_ih_l0": (768, 256),
    "gru.weight_hh_l0": (768, 256),
    "gru.bias_ih_l0": (768,),
    "gru.bias_hh_l0": (768,),
    "gru.weight_ih_l1": (768, 256),
    "gru.weight_hh_l1": (768, 256),
    "gru.bias_ih_l1": (768,),
    "gru.bias_hh_l1": (768,),
    "head.0.weight": (256, 512),
    "head.0.bias": (256,),
    "head.2.weight": (25, 256),
    "head.2.bias": (25,),
}
PARAMETER_SHAPES = tuple(
    shape for name, shape in MODEL_SHAPES.items() if name not in {"mean", "std"}
)
CONTRACT_KEYS = {
    "epochs",
    "total_steps",
    "options",
    "sources",
    "datasets",
    "teacher_predictions",
    "consumed_caches",
    "fit_membership",
    "normalization",
    "ancestors",
    "loader",
    "runtime",
}
TRAINING_STATE_KEYS = {
    "format",
    "version",
    "boundary",
    "contract",
    "next_epoch",
    "model",
    "optimizer",
    "scheduler",
    "scaler",
    "ema",
    "swa",
    "loader",
    "auxiliary",
    "progress",
    "rng",
}
CHECKPOINT_PROVENANCE_KEYS = (
    "args",
    "split_manifest_sha256",
    "fit_row_groups",
    "evaluation_row_groups",
    "fit_membership_identity",
    "normalization_identity",
    "trainer_implementation",
    "distillation_cache",
    "update_diagnostics",
    "cumulative_updates",
    "initial_checkpoint_sha256",
    "propensity_checkpoint_sha256",
)


def _eq(actual: object, expected: object, label: str) -> None:
    """Require an exact value match without JSON-type coercion."""
    if actual != expected:
        raise ValueError(f"{label} differs")


def _int(value: object, expected: int, label: str) -> None:
    """Require an exact integer and reject bools or numeric strings."""
    if type(value) is not int or value != expected:
        raise ValueError(f"{label} must be integer {expected}")


def _is_sha256(value: object) -> bool:
    """Return whether value is a lowercase SHA-256 digest."""
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _check_control_launch_records(
    launch: Mapping[str, object],
    launch_sha: str,
    launch_acceptance: Mapping[str, object],
    launch_acceptance_sha: str,
    admission_sha: str,
) -> None:
    """Bind original CONTROL launch bytes and the separate accepted body digest."""
    _eq(
        (admission_sha, launch_sha, launch_acceptance_sha),
        (ADMISSION_SHA, LAUNCH_SHA, LAUNCH_ACCEPTANCE_SHA),
        "CONTROL launch receipt pins",
    )
    submission = launch.get("submission")
    if not isinstance(submission, dict):
        raise ValueError("CONTROL launch submission is missing")
    _eq(
        (
            launch.get("schema"),
            launch.get("unit"),
            submission.get("outcome"),
            submission.get("returncode"),
        ),
        ("fg19-child-control-fit-launch.v1", UNIT, "submitted", 0),
        "CONTROL launch record",
    )
    _eq(
        (
            launch_acceptance.get("schema"),
            launch_acceptance.get("status"),
            launch_acceptance.get("unit"),
            launch_acceptance.get("invocation_id"),
            launch_acceptance.get("admission_sha256"),
            launch_acceptance.get("launch_sha256"),
            launch_acceptance.get("body_sha256"),
        ),
        (
            "fg19-child-control-fit-launch-acceptance.v1",
            "ACTUAL_LAUNCH_CAPTURED_NOT_COMPLETION",
            UNIT,
            INVOCATION,
            admission_sha,
            launch_sha,
            LAUNCH_BODY_SHA,
        ),
        "CONTROL launch acceptance",
    )


def _check_terminal(terminal: Mapping[str, object]) -> None:
    """Require the root-captured typed exit for this exact running invocation."""
    _eq(
        (
            terminal.get("schema"),
            terminal.get("unit"),
            terminal.get("run_name"),
            terminal.get("invocation_id"),
        ),
        ("fg19.child-control-fit-native-terminal-evidence.v1", UNIT, RUN, INVOCATION),
        "native terminal identity",
    )
    _eq(
        (terminal.get("active_state"), terminal.get("sub_state")),
        ("active", "exited"),
        "native terminal state",
    )
    for key, value in (("main_pid", 0), ("exec_main_code", 1), ("exec_main_status", 0)):
        _int(terminal.get(key), value, key)
    if type(terminal.get("raw_bytes")) is not int or terminal["raw_bytes"] <= 0:
        raise ValueError("native terminal raw byte count is invalid")
    if not _is_sha256(terminal.get("raw_sha256")):
        raise ValueError("native terminal raw digest is malformed")
    for key in ("captured_utc", "native_exit_utc"):
        value = terminal.get(key)
        if not isinstance(value, str):
            raise ValueError(f"native terminal {key} is missing")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"native terminal {key} is not ISO-8601") from error
        if parsed.tzinfo is None:
            raise ValueError(f"native terminal {key} lacks an offset")
    if not isinstance(terminal.get("raw_path"), str) or not terminal["raw_path"]:
        raise ValueError("native terminal raw path is missing")


def _model_fingerprint(model: object) -> tuple[str, int]:
    """Validate the finite FP32 467→25 child state and fingerprint its tensors."""
    if not isinstance(model, dict) or set(model) != set(MODEL_SHAPES):
        raise ValueError("child checkpoint tensor inventory differs from H256x2 467→25 GRU")
    records: dict[str, object] = {}
    for name, shape in MODEL_SHAPES.items():
        record = _tensor_summary(model[name], torch)
        _eq((record["dtype"], tuple(record["shape"])), ("torch.float32", shape), name)
        records[name] = record
    if bool((model["std"] < 0.001).any().item()):
        raise ValueError("child standard deviation violates the source clamp")
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), len(records)


def _tensor_rng_signature(tensor: object, label: str) -> dict[str, object]:
    """Validate one producer-shaped RNG byte tensor and return its observed digest."""
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8 or tensor.ndim != 1:
        raise ValueError(f"{label} RNG tensor must be one-dimensional uint8")
    expected_size = 5056 if label == "CPU Torch" else 16
    _int(tensor.numel(), expected_size, f"{label} RNG cardinality")
    raw = tensor.detach().cpu().contiguous().numpy().tobytes()
    return {
        "type": "torch.Tensor",
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _check_loader_and_rng(state: Mapping[str, object]) -> dict[str, object]:
    """Validate the producer's epoch-5 EpochOrderSampler and source-shaped RNGs."""
    loader = state.get("loader")
    expected_loader = {
        "sampler_type": "wnn.training_sampling.EpochOrderSampler",
        "batch_size": 16,
        "drop_last": True,
        "num_workers": 0,
        "dataset_length": 11_940,
        "loader_generator": None,
        "sampler_generator": None,
        "sampler_state": None,
        "sampler_epoch": 5,
    }
    _eq(loader, expected_loader, "saved epoch-boundary loader")
    rng = state.get("rng")
    if not isinstance(rng, dict) or set(rng) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("saved RNG state schema differs")
    python_rng = rng["python"]
    if (
        not isinstance(python_rng, tuple)
        or len(python_rng) != 3
        or type(python_rng[0]) is not int
        or not isinstance(python_rng[1], tuple)
        or len(python_rng[1]) != 625
        or any(type(value) is not int for value in python_rng[1])
        or (python_rng[2] is not None and type(python_rng[2]) is not float)
    ):
        raise ValueError("Python RNG state has an unsupported producer format")
    numpy_rng = rng["numpy"]
    if (
        not isinstance(numpy_rng, tuple)
        or len(numpy_rng) != 5
        or numpy_rng[0] != "MT19937"
        or not isinstance(numpy_rng[1], np.ndarray)
        or numpy_rng[1].dtype != np.uint32
        or numpy_rng[1].shape != (624,)
        or type(numpy_rng[2]) is not int
        or type(numpy_rng[3]) is not int
        or type(numpy_rng[4]) not in {int, float}
    ):
        raise ValueError("NumPy RNG state has an unsupported producer format")
    cpu = _tensor_rng_signature(rng["torch"], "CPU Torch")
    cuda = rng["cuda"]
    if not isinstance(cuda, list) or len(cuda) != 1:
        raise ValueError("single visible CUDA RNG state is malformed")
    cuda_signature = _tensor_rng_signature(cuda[0], "CUDA")
    return {
        "torch": cpu,
        "cuda": {"type": "builtins.list", "cardinality": 1, "components": [cuda_signature]},
    }


def _check_sample_order(item: object, epoch: int) -> dict[str, object]:
    """Check logged sampler identity while treating its child RNG hashes as observed data."""
    if not isinstance(item, dict):
        raise ValueError("sample-order identity is missing")
    expected = {
        "sampler_class": "EpochOrderSampler",
        "batch_size": 16,
        "drop_last": True,
        "loader_generator": "global CPU RNG",
        "sampler_generator": "global CPU RNG",
        "num_workers": 0,
        "persistent_workers": False,
        "sampler_replacement": None,
        "sampler_num_samples": None,
        "seeded_order": {
            "seed": 1001,
            "epoch": epoch,
            "antithetic": False,
            "shared_dropped_tail": True,
        },
    }
    _eq({key: item.get(key) for key in expected}, expected, f"epoch {epoch} sample-order contract")
    loader_rng, sampler_rng = item.get("loader_rng"), item.get("sampler_rng")
    for label, entry in (("loader", loader_rng), ("sampler", sampler_rng)):
        if not isinstance(entry, dict):
            raise ValueError(f"epoch {epoch} {label} RNG identity is missing")
        _eq(
            (entry.get("shape"), entry.get("dtype")),
            ([5056], "torch.uint8"),
            f"epoch {epoch} {label} RNG format",
        )
        if not _is_sha256(entry.get("bytes_sha256")):
            raise ValueError(f"epoch {epoch} {label} RNG digest is malformed")
    _eq(loader_rng, sampler_rng, f"epoch {epoch} shared global-CPU RNG identity")
    return {
        "epoch": epoch,
        "rng_sha256": loader_rng["bytes_sha256"],
        "rng_shape": [5056],
        "rng_dtype": "torch.uint8",
    }


def _check_scheduler(scheduler: object) -> tuple[float, float]:
    """Validate the child OneCycleLR horizon, phase split, and cosine endpoint."""
    if not isinstance(scheduler, dict):
        raise ValueError("saved OneCycle scheduler is missing")
    _eq(
        set(scheduler),
        {
            "total_steps",
            "_schedule_phases",
            "_anneal_func_type",
            "cycle_momentum",
            "use_beta1",
            "base_lrs",
            "last_epoch",
            "_step_count",
            "_is_initial",
            "_get_lr_called_within_step",
            "_last_lr",
        },
        "OneCycle scheduler schema",
    )
    _int(scheduler.get("total_steps"), UPDATES, "scheduler total steps")
    _int(scheduler.get("last_epoch"), UPDATES, "scheduler cursor")
    _int(scheduler.get("_step_count"), UPDATES + 1, "scheduler step count")
    if scheduler.get("cycle_momentum") is not True:
        raise ValueError("AdamW OneCycle beta schedule was not retained")
    _eq(
        (
            scheduler.get("_anneal_func_type"),
            scheduler.get("use_beta1"),
            scheduler.get("_is_initial"),
            scheduler.get("_get_lr_called_within_step"),
        ),
        ("cos", True, False, False),
        "OneCycle policy",
    )
    phases = [
        {
            "end_step": 2237.0,
            "start_lr": "initial_lr",
            "end_lr": "max_lr",
            "start_momentum": "max_momentum",
            "end_momentum": "base_momentum",
        },
        {
            "end_step": UPDATES - 1,
            "start_lr": "max_lr",
            "end_lr": "min_lr",
            "start_momentum": "base_momentum",
            "end_momentum": "max_momentum",
        },
    ]
    _eq(scheduler.get("_schedule_phases"), phases, "OneCycle phase geometry")
    _eq(scheduler.get("base_lrs"), [4e-6], "OneCycle base learning rate")
    phase_start = phases[0]["end_step"]
    phase_end = phases[1]["end_step"]
    terminal_fraction = (UPDATES - phase_start) / (phase_end - phase_start)
    cosine_out = math.cos(math.pi * terminal_fraction) + 1
    expected_lr = 4e-8 + (1e-4 - 4e-8) / 2.0 * cosine_out
    expected_beta1 = 0.95 + (0.85 - 0.95) / 2.0 * cosine_out
    rates = scheduler.get("_last_lr")
    if (
        not isinstance(rates, list)
        or len(rates) != 1
        or type(rates[0]) not in {int, float}
        or not math.isfinite(rates[0])
        or rates[0] < 0
    ):
        raise ValueError("OneCycle final learning rate is malformed")
    _eq(rates[0], expected_lr, "OneCycle terminal learning rate")
    return expected_lr, expected_beta1


def _check_optimizer(
    optimizer: object, scheduler_lr: float, scheduler_beta1: float
) -> dict[str, object]:
    """Validate terminal OneCycle group values and AdamW moments."""
    if (
        not isinstance(optimizer, dict)
        or not isinstance(optimizer.get("param_groups"), list)
        or not isinstance(optimizer.get("state"), dict)
    ):
        raise ValueError("AdamW state is malformed")
    groups, states = optimizer["param_groups"], optimizer["state"]
    if len(groups) != 1:
        raise ValueError("CONTROL requires one AdamW parameter group")
    group = groups[0]
    _eq(
        {
            key: group.get(key)
            for key in ("initial_lr", "max_lr", "min_lr", "base_momentum", "max_momentum")
        },
        {
            "initial_lr": 4e-6,
            "max_lr": 1e-4,
            "min_lr": 4e-8,
            "base_momentum": 0.85,
            "max_momentum": 0.95,
        },
        "AdamW OneCycle group endpoints",
    )
    betas = group.get("betas")
    if (
        not isinstance(betas, (tuple, list))
        or len(betas) != 2
        or type(betas[0]) not in {int, float}
        or type(betas[1]) not in {int, float}
    ):
        raise ValueError("AdamW beta values are malformed")
    if not (0 < betas[0] < 1 and betas[1] == 0.999) or not all(
        math.isfinite(value) for value in betas
    ):
        raise ValueError("AdamW terminal beta schedule differs")
    _eq(betas[0], scheduler_beta1, "AdamW/OneCycle terminal beta1")
    _eq((group.get("weight_decay"), group.get("eps")), (0.01, 1e-8), "AdamW hyperparameters")
    learning_rate = group.get("lr")
    if (
        type(learning_rate) not in {int, float}
        or not math.isfinite(learning_rate)
        or learning_rate < 0
    ):
        raise ValueError("AdamW terminal learning rate is invalid")
    _eq(learning_rate, scheduler_lr, "AdamW/scheduler learning rate")
    for key in ("amsgrad", "maximize", "capturable", "differentiable", "fused"):
        if key in group and group[key] is not False:
            raise ValueError(f"AdamW policy differs: {key}")
    ids = group.get("params")
    if (
        not isinstance(ids, list)
        or len(ids) != len(PARAMETER_SHAPES)
        or any(type(value) is not int for value in ids)
    ):
        raise ValueError("AdamW parameter identity count differs")
    if len(set(ids)) != len(ids) or set(states) != set(ids):
        raise ValueError("AdamW states do not cover the parameter group exactly")
    for parameter_id, shape in zip(ids, PARAMETER_SHAPES, strict=True):
        state = states[parameter_id]
        if not isinstance(state, dict) or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("AdamW parameter state schema differs")
        step = state["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.ndim != 0
            or not torch.isfinite(step).item()
            or float(step.item()) != UPDATES
        ):
            raise ValueError("AdamW parameter step differs from completed optimizer updates")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = state[name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float32
                or tuple(tensor.shape) != shape
                or not bool(torch.isfinite(tensor).all().item())
            ):
                raise ValueError(f"AdamW {name} is not finite FP32 with its model parameter shape")
            if name == "exp_avg_sq" and bool((tensor < 0).any().item()):
                raise ValueError("AdamW second moment contains a negative value")
    return {
        "groups": 1,
        "parameters": len(ids),
        "moment_tensors": 2 * len(ids),
        "step": UPDATES,
        "learning_rate": learning_rate,
        "beta1": betas[0],
    }


def _check_counters(
    history: object, state: Mapping[str, object], provenance: Mapping[str, object]
) -> list[dict[str, object]]:
    """Reconcile six no-grade epochs across history, diagnostics and saved state."""
    diagnostics = provenance.get("update_diagnostics")
    if (
        not isinstance(history, list)
        or len(history) != EPOCHS
        or not isinstance(diagnostics, list)
        or len(diagnostics) != EPOCHS
    ):
        raise ValueError("six-epoch fit history and diagnostics are required")
    parent_v3._finite_tree(history)
    progress = state.get("progress")
    if not isinstance(progress, dict) or set(progress) != {
        "step_count",
        "best",
        "history",
        "cumulative_updates",
        "update_diagnostics",
        "fitting_seconds",
    }:
        raise ValueError("saved training progress state is malformed")
    if type(progress.get("best")) is not float or progress["best"] != -1.0:
        raise ValueError("fit-only CONTROL must retain the unscored best sentinel")
    seconds = progress.get("fitting_seconds")
    if type(seconds) not in {int, float} or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("saved fit duration is malformed")
    _eq(
        (progress.get("history"), progress.get("update_diagnostics")),
        (history, diagnostics),
        "saved state/history",
    )
    observed_order: list[dict[str, object]] = []
    for epoch, (record, item) in enumerate(zip(history, diagnostics, strict=True)):
        if not isinstance(record, dict) or not isinstance(item, dict):
            raise ValueError("fit history row is malformed")
        _eq(
            (record.get("epoch"), record.get("evaluation"), item.get("epoch_index")),
            (epoch, "withheld", epoch),
            "fit-only history order",
        )
        if "weighted_pearson" in record:
            raise ValueError("CONTROL history contains evaluation output")
        _eq(
            record.get("update_diagnostics"),
            {key: value for key, value in item.items() if key != "epoch_index"},
            "history update counters",
        )
        nested = item.get("epoch")
        cumulative = item.get("cumulative")
        if not isinstance(nested, dict) or not isinstance(cumulative, dict):
            raise ValueError("per-epoch update counters are missing")
        for key, expected in (
            ("optimizer_attempts", UPDATES_PER_EPOCH),
            ("optimizer_completed_updates", UPDATES_PER_EPOCH),
            ("completed_batches", BATCHES_PER_EPOCH),
            ("amp_skipped_updates", 0),
            ("ema_updates", 0),
        ):
            _int(nested.get(key), expected, f"epoch {epoch} {key}")
        for key, expected in (
            ("optimizer_attempts", (epoch + 1) * UPDATES_PER_EPOCH),
            ("optimizer_completed_updates", (epoch + 1) * UPDATES_PER_EPOCH),
            ("completed_batches", (epoch + 1) * BATCHES_PER_EPOCH),
            ("amp_skipped_updates", 0),
            ("ema_updates", 0),
        ):
            _int(cumulative.get(key), expected, f"epoch {epoch} cumulative {key}")
        gradients = item.get("grad_stats")
        if not isinstance(gradients, dict):
            raise ValueError("pre-clip gradient statistics are missing")
        _int(gradients.get("epoch"), epoch, f"epoch {epoch} gradient-stat index")
        _int(gradients.get("steps"), UPDATES_PER_EPOCH, f"epoch {epoch} gradient steps")
        _int(gradients.get("nonfinite_steps"), 0, f"epoch {epoch} nonfinite gradients")
        sample = _check_sample_order(item.get("sample_order_identity"), epoch)
        observed_order.append(sample)
        normalization = item.get("normalization_identity")
        if not isinstance(normalization, dict):
            raise ValueError("epoch normalization identity is missing")
        _eq(set(normalization), {"mean", "std"}, "normalization identity schema")
    totals = provenance.get("cumulative_updates")
    if not isinstance(totals, dict):
        raise ValueError("final cumulative update counters are missing")
    for key, expected in (
        ("optimizer_attempts", UPDATES),
        ("optimizer_completed_updates", UPDATES),
        ("completed_batches", BATCHES),
        ("amp_skipped_updates", 0),
        ("ema_updates", 0),
    ):
        _int(totals.get(key), expected, f"final cumulative {key}")
    _int(progress.get("step_count"), UPDATES, "saved optimizer step count")
    _eq(progress.get("cumulative_updates"), totals, "state/provenance cumulative updates")
    return observed_order


def _check_profile_initial_state(record: Mapping[str, object]) -> None:
    """Require the already accepted source-qualified 211→467/23→25 extension proof."""
    expected = {
        "schema": "fg19-child-allthree-profile-initial-state.v1",
        "parent_run": PARENT_RUN,
        "parent_invocation_id": PARENT_INVOCATION,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA,
        "parent_acceptance_sha256": PARENT_ACCEPTANCE_SHA,
        "parent_exact_stop_sha256": PARENT_STOP_SHA,
        "dictionary_acceptance_sha256": DICTIONARY_ACCEPTANCE_SHA,
        "dictionary_cache_sha256": DICTIONARY_SHA,
        "split_manifest_sha256": SPLIT_SHA,
        "source_commit": COMMIT,
        "parent_acceptance_status": "ACCEPTED",
    }
    _eq({key: record.get(key) for key in expected}, expected, "accepted profile ancestry")
    profile = record.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("accepted profile geometry is missing")
    _eq(
        {
            key: profile.get(key)
            for key in ("sequence_length", "batch", "chunk", "input_representation", "objective")
        },
        {
            "sequence_length": 20_000,
            "batch": 16,
            "chunk": 2_000,
            "input_representation": "ema_sparse",
            "objective": "nll",
        },
        "profile worst-shape recipe",
    )
    initialization = record.get("initialization")
    if not isinstance(initialization, dict):
        raise ValueError("accepted profile initialization proof is missing")
    _eq(
        {
            key: initialization.get(key)
            for key in (
                "parent_input_width",
                "child_input_width",
                "parent_output_width",
                "child_output_width",
                "parent_trunk_and_prefix_equal",
                "new_input_columns_zero",
                "new_logvariance_rows_zero",
                "normalization_prefix_equal",
                "normalization_suffix_mean_zero_std_one",
                "finite_initial_state",
            )
        },
        {
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
        },
        "profile extension proof",
    )
    for key in ("parent_state_fingerprint", "initial_child_state_fingerprint"):
        if not _is_sha256(initialization.get(key)):
            raise ValueError(f"profile {key} is missing")


def _read_pinned_json(
    path: Path, expected_sha: str, limit: int = 1_000_000
) -> tuple[dict[str, object], str]:
    """Read one bounded no-follow JSON receipt and bind its exact bytes."""
    raw, digest, _ = _stable_file(path, limit)
    _eq(digest, expected_sha, f"receipt digest {path.name}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"receipt is not a JSON object: {path.name}")
    return value, digest


def _check_stop_receipt(
    stop: Mapping[str, object],
    unit: str,
    invocation: str,
    acceptance_sha: str,
    *,
    strict_schema: str | None = None,
) -> None:
    """Require a native-success stop receipt for one already completed predecessor."""
    before, after, stopped = stop.get("before"), stop.get("after"), stop.get("stop")
    if not all(isinstance(item, dict) for item in (before, after)):
        raise ValueError("predecessor stop receipt is incomplete")
    if strict_schema is not None:
        _eq(stop.get("schema"), strict_schema, "predecessor stop schema")
    _eq(
        (stop.get("unit"), stop.get("invocation_id")),
        (unit, invocation),
        "predecessor stop identity",
    )
    _eq(
        (
            before.get("InvocationID"),
            before.get("ActiveState"),
            before.get("SubState"),
            before.get("MainPID"),
            before.get("ExecMainCode"),
            before.get("ExecMainStatus"),
        ),
        (invocation, "active", "exited", "0", "1", "0"),
        "predecessor exit",
    )
    _eq(
        (
            after.get("LoadState"),
            after.get("ActiveState"),
            after.get("SubState"),
            after.get("MainPID"),
            after.get("InvocationID"),
        ),
        ("not-found", "inactive", "dead", "0", ""),
        "predecessor stopped state",
    )
    if "acceptance_sha256" in stop:
        _eq(stop.get("acceptance_sha256"), acceptance_sha, "stop acceptance binding")
    returncode = (
        stopped.get("returncode") if isinstance(stopped, dict) else stop.get("stop_returncode")
    )
    _eq(returncode, 0, "predecessor stop command")
    populated = stop.get("populated_after", stop.get("populated", {}))
    _eq(populated, {}, "predecessor populated descendants")


def _check_predecessor_receipts(
    evidence_root: Path, profile_root: Path, admission: Mapping[str, object]
) -> dict[str, object]:
    """Validate the exact accepted parent, dictionary and profile ancestry receipts."""
    staged = admission.get("staged_receipt_sha256")
    if not isinstance(staged, dict):
        raise ValueError("admission lacks staged predecessor receipt digests")
    expected_staged = {
        "PARENT_TERMINAL_ACCEPTANCE.json": PARENT_ACCEPTANCE_SHA,
        "PARENT_EXACT_STOP.json": PARENT_STOP_SHA,
        "DICTIONARY_ACCEPTANCE.json": DICTIONARY_ACCEPTANCE_SHA,
        "DICTIONARY_EXACT_STOP.json": DICTIONARY_STOP_SHA,
        "PROFILE_TERMINAL_ACCEPTANCE.json": PROFILE_ACCEPTANCE_SHA,
        "PROFILE_EXACT_STOP.json": PROFILE_STOP_SHA,
    }
    _eq(
        {name: staged.get(name) for name in expected_staged},
        expected_staged,
        "staged predecessor digest map",
    )
    parent, _ = _read_pinned_json(
        evidence_root / "PARENT_TERMINAL_ACCEPTANCE.json", PARENT_ACCEPTANCE_SHA
    )
    parent_stop, _ = _read_pinned_json(evidence_root / "PARENT_EXACT_STOP.json", PARENT_STOP_SHA)
    dictionary, _ = _read_pinned_json(
        evidence_root / "DICTIONARY_ACCEPTANCE.json", DICTIONARY_ACCEPTANCE_SHA
    )
    dictionary_stop, _ = _read_pinned_json(
        evidence_root / "DICTIONARY_EXACT_STOP.json", DICTIONARY_STOP_SHA
    )
    profile, _ = _read_pinned_json(
        evidence_root / "PROFILE_TERMINAL_ACCEPTANCE.json", PROFILE_ACCEPTANCE_SHA
    )
    profile_stop, _ = _read_pinned_json(evidence_root / "PROFILE_EXACT_STOP.json", PROFILE_STOP_SHA)
    _eq(
        (
            parent.get("schema"),
            parent.get("status"),
            parent.get("run_name"),
            parent.get("unit"),
            parent.get("invocation_id"),
            parent.get("selected_epoch"),
            parent.get("successful_updates"),
            parent.get("next_epoch"),
            parent.get("quality_score"),
        ),
        (
            "fg19.parent-fit-r2-terminal-acceptance.v1",
            "ACCEPTED",
            PARENT_RUN,
            PARENT_UNIT,
            PARENT_INVOCATION,
            3,
            29_840,
            4,
            None,
        ),
        "parent acceptance",
    )
    _eq(
        parent.get("files", {}).get("model_ep3.pt"),
        {"bytes": PARENT_CHECKPOINT_BYTES, "sha256": PARENT_CHECKPOINT_SHA},
        "accepted parent checkpoint",
    )
    _check_stop_receipt(parent_stop, PARENT_UNIT, PARENT_INVOCATION, PARENT_ACCEPTANCE_SHA)
    _eq(
        (
            dictionary.get("schema"),
            dictionary.get("status"),
            dictionary.get("invocation"),
            dictionary.get("cache", {}).get("bytes"),
            dictionary.get("cache", {}).get("sha256"),
        ),
        (
            "fg19.dictionary-root-acceptance.v1",
            "ARTIFACT_AND_PROVENANCE_ACCEPTED",
            "4e6d90ff7d4344df92d61d0d1c3f01fb",
            DICTIONARY_BYTES,
            DICTIONARY_SHA,
        ),
        "shared dictionary acceptance",
    )
    _eq(
        dictionary_stop.get("before", {}).get("InvocationID"),
        "4e6d90ff7d4344df92d61d0d1c3f01fb",
        "dictionary stop invocation",
    )
    _eq(
        dictionary_stop.get("before", {}).get("ActiveState"),
        "active",
        "dictionary stop before state",
    )
    _eq(
        dictionary_stop.get("before", {}).get("SubState"),
        "exited",
        "dictionary stop before substate",
    )
    _eq(dictionary_stop.get("stop_returncode"), 0, "dictionary stop command")
    _eq(
        (
            dictionary_stop.get("after", {}).get("LoadState"),
            dictionary_stop.get("after", {}).get("ActiveState"),
            dictionary_stop.get("after", {}).get("SubState"),
            dictionary_stop.get("after", {}).get("InvocationID"),
            dictionary_stop.get("populated"),
        ),
        ("not-found", "inactive", "dead", "", {}),
        "dictionary exact stop",
    )
    _eq(
        (
            profile.get("schema"),
            profile.get("status"),
            profile.get("unit"),
            profile.get("invocation_id"),
            profile.get("source_commit"),
            profile.get("source_file_count"),
            profile.get("optimizer_attempts"),
            profile.get("optimizer_completed_updates"),
            profile.get("amp_skipped_updates"),
            profile.get("gradient_stats_nonfinite_steps"),
            profile.get("scientific_fits"),
            profile.get("quality_score"),
            profile.get("no_checkpoint"),
        ),
        (
            "fg19-child-allthree-profile-terminal-acceptance.v1",
            "ACCEPTED",
            PROFILE_UNIT,
            PROFILE_INVOCATION,
            COMMIT,
            167,
            60,
            60,
            0,
            0,
            0,
            None,
            True,
        ),
        "accepted profile",
    )
    initial_state = profile.get("initial_state")
    if not isinstance(initial_state, dict):
        raise ValueError("accepted profile lacks its initial-state proof")
    _check_profile_initial_state(initial_state)
    profile_files = profile.get("files")
    if not isinstance(profile_files, dict) or set(profile_files) != {
        "ADMISSION.json",
        "LAUNCH.json",
        "unit.log",
    }:
        raise ValueError("profile acceptance does not bind its exact admission/launch/log map")
    captured: dict[str, bytes] = {}
    for name, pin in profile_files.items():
        if (
            not isinstance(pin, dict)
            or type(pin.get("bytes")) is not int
            or not _is_sha256(pin.get("sha256"))
        ):
            raise ValueError(f"profile file pin is malformed: {name}")
        raw, digest, info = _stable_file(profile_root / name, 1_000_000)
        _eq(
            (len(raw), info["bytes"], digest),
            (pin["bytes"], pin["bytes"], pin["sha256"]),
            f"profile file {name}",
        )
        captured[name] = raw
    profile_admission = json.loads(captured["ADMISSION.json"])
    profile_launch = json.loads(captured["LAUNCH.json"])
    _eq(
        (
            profile_admission.get("schema"),
            profile_admission.get("unit"),
            profile_admission.get("run_name"),
            profile_admission.get("source_commit"),
            profile_admission.get("source_manifest_sha256"),
            profile_admission.get("source_files_verified"),
            profile_admission.get("parent_acceptance_sha256"),
            profile_admission.get("parent_exact_stop_sha256"),
            profile_admission.get("parent_checkpoint_sha256"),
            profile_admission.get("dictionary", {}).get("sha256"),
        ),
        (
            "fg19-child-allthree-profile-admission.v1",
            PROFILE_UNIT,
            PROFILE_RUN,
            COMMIT,
            SOURCE_MANIFEST_SHA,
            167,
            PARENT_ACCEPTANCE_SHA,
            PARENT_STOP_SHA,
            PARENT_CHECKPOINT_SHA,
            DICTIONARY_SHA,
        ),
        "profile admission ancestry",
    )
    _eq(
        (profile_launch.get("schema"), profile_launch.get("unit")),
        ("fg19-child-allthree-profile-launch.v1", PROFILE_UNIT),
        "profile launch identity",
    )
    lines = [
        line.removeprefix("CHILD_PROFILE_INITIAL_STATE ")
        for line in captured["unit.log"].decode("utf-8").splitlines()
        if line.startswith("CHILD_PROFILE_INITIAL_STATE ")
    ]
    if len(lines) != 1 or json.loads(lines[0]) != initial_state:
        raise ValueError("accepted profile initial-state proof differs from preserved native log")
    stop_hash = hashlib.sha256(
        json.dumps(initial_state, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _eq(stop_hash, PROFILE_INITIAL_STATE_SHA, "profile initial-state digest")
    _check_stop_receipt(profile_stop, PROFILE_UNIT, PROFILE_INVOCATION, PROFILE_ACCEPTANCE_SHA)
    return {
        "parent_acceptance_sha256": PARENT_ACCEPTANCE_SHA,
        "parent_exact_stop_sha256": PARENT_STOP_SHA,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA,
        "dictionary_acceptance_sha256": DICTIONARY_ACCEPTANCE_SHA,
        "dictionary_exact_stop_sha256": DICTIONARY_STOP_SHA,
        "dictionary_sha256": DICTIONARY_SHA,
        "profile_acceptance_sha256": PROFILE_ACCEPTANCE_SHA,
        "profile_exact_stop_sha256": PROFILE_STOP_SHA,
        "profile_initial_state_sha256": stop_hash,
    }


def _check_sample_log(
    log_path: Path, history: list[object], order_records: list[dict[str, object]]
) -> dict[str, object]:
    """Reconcile producer FIT/GRAD_STATS records with the final saved history."""
    raw, digest, info = _stable_file(log_path, MAX_LOG_BYTES)
    fit_rows: list[dict[str, object]] = []
    gradient_rows: list[dict[str, object]] = []
    selected: list[str] = []
    for line in raw.decode("utf-8").splitlines():
        if line.startswith("FIT "):
            value = json.loads(line[4:])
            if isinstance(value, dict):
                fit_rows.append(value)
        elif line.startswith("GRAD_STATS "):
            value = json.loads(line[len("GRAD_STATS ") :])
            if isinstance(value, dict):
                gradient_rows.append(value)
        elif line.startswith("SELECTED_FIXED "):
            selected.append(line.removeprefix("SELECTED_FIXED "))
    if len(fit_rows) != EPOCHS or len(gradient_rows) != EPOCHS:
        raise ValueError("native log does not contain exactly six FIT and GRAD_STATS records")
    for epoch, (fit, gradients, saved, order) in enumerate(
        zip(fit_rows, gradient_rows, history, order_records, strict=True)
    ):
        _eq(fit, saved, f"native FIT history epoch {epoch}")
        _eq(fit.get("epoch"), epoch, f"native FIT epoch {epoch}")
        _eq(
            gradients,
            saved.get("update_diagnostics", {}).get("grad_stats"),
            f"native GRAD_STATS epoch {epoch}",
        )
        _eq(
            saved.get("update_diagnostics", {})
            .get("sample_order_identity", {})
            .get("seeded_order", {})
            .get("epoch"),
            epoch,
            f"native sample-order epoch {epoch}",
        )
        _eq(order["epoch"], epoch, f"saved/log RNG identity epoch {epoch}")
    _eq(selected, ["5"], "fixed checkpoint selection log")
    return {
        "bytes": info["bytes"],
        "sha256": digest,
        "fit_records": len(fit_rows),
        "gradient_records": len(gradient_rows),
        "fixed_selection": selected[0],
    }


def _check_args_and_contract(
    state: Mapping[str, object],
    provenance: Mapping[str, object],
    admission: Mapping[str, object],
    launch_args: Mapping[str, object],
    manifest: Mapping[str, object],
    split_sha: str,
) -> None:
    """Bind the complete child CLI, ancestry, datasets, source map, cache and runtime."""
    contract = state.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("epoch continuation contract is missing")
    _eq(set(contract), CONTRACT_KEYS, "continuation contract schema")
    _eq(
        (contract.get("epochs"), contract.get("total_steps")),
        (EPOCHS, UPDATES),
        "continuation horizon",
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
    _eq(contract.get("options"), expected_options, "continuation CLI options")
    _eq(
        (contract.get("fit_membership"), contract.get("normalization")),
        (provenance.get("fit_membership_identity"), provenance.get("normalization_identity")),
        "continuation membership/normalization",
    )
    _eq(
        contract.get("ancestors"),
        {
            "initial_checkpoint_sha256": PARENT_CHECKPOINT_SHA,
            "propensity_checkpoint_sha256": None,
            "quote_cache": None,
            "spread_aux": None,
        },
        "child initializer ancestry",
    )
    data = admission.get("data_identity_stats_only")
    if not isinstance(data, dict):
        raise ValueError("admitted dataset identities are missing")
    expected_data = {
        name: {
            "bytes": data[f"datasets/{name}.parquet"]["bytes"],
            "sha256": data[f"datasets/{name}.parquet"]["registered_sha256"],
        }
        for name in ("train", "valid")
    }
    _eq(contract.get("datasets"), expected_data, "continuation dataset hashes")
    _eq(contract.get("teacher_predictions"), {}, "standalone teacher predictions")
    cache_meta = provenance.get("distillation_cache")
    if not isinstance(cache_meta, dict) or not isinstance(cache_meta.get("verified_files"), dict):
        raise ValueError("child cache provenance is missing")
    _eq(
        contract.get("consumed_caches"),
        {
            "distillation_cache_metadata": parent_v3.CACHE_METADATA_SHA,
            **{f"distillation_{key}": value for key, value in cache_meta["verified_files"].items()},
        },
        "consumed cache hashes",
    )
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("source manifest file list is missing")
    source_map = {item["path"]: item["sha256"] for item in files}
    expected_sources = {
        name.removeprefix("src/wnn/"): value
        for name, value in source_map.items()
        if name.startswith("src/wnn/") and name.endswith(".py")
    }
    expected_sources["trainer"] = source_map["scripts/train_rnn.py"]
    _eq(contract.get("sources"), expected_sources, "continuation source map")
    implementation_files = {
        name: value
        for name, value in source_map.items()
        if name == "scripts/train_rnn.py"
        or name in {"pyproject.toml", "uv.lock"}
        or (name.startswith("src/wnn/training/") and name.endswith(".py"))
    }
    implementation = provenance.get("trainer_implementation")
    if not isinstance(implementation, dict):
        raise ValueError("trainer implementation record is missing")
    implementation_sha = hashlib.sha256(
        json.dumps(implementation_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _eq(
        (implementation.get("files"), implementation.get("sha256")),
        (implementation_files, implementation_sha),
        "trainer implementation identity",
    )
    _eq(
        (
            provenance.get("initial_checkpoint_sha256"),
            provenance.get("propensity_checkpoint_sha256"),
        ),
        (PARENT_CHECKPOINT_SHA, None),
        "provenance initializer",
    )
    runtime = contract.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {
        "torch",
        "numpy",
        "python",
        "machine",
        "cublas_workspace_config",
        "devices",
        "cuda",
        "cudnn",
        "threads",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "matmul_tf32",
        "cudnn_tf32",
    }:
        raise ValueError("runtime identity schema differs")
    _eq(runtime.get("torch"), EXPECTED_TORCH, "source-qualified PyTorch version")
    _eq(
        (
            runtime.get("matmul_tf32"),
            runtime.get("cudnn_tf32"),
            runtime.get("cublas_workspace_config"),
            runtime.get("cudnn_benchmark"),
            runtime.get("cudnn_deterministic"),
        ),
        (False, False, ":4096:8", False, True),
        "strict source-pinned CUDA runtime",
    )
    _int(runtime.get("threads"), 1, "runtime thread count")
    if torch.__version__ != EXPECTED_TORCH:
        raise ValueError("checker PyTorch runtime differs from the fit runtime")
    devices = runtime.get("devices")
    if (
        not isinstance(devices, list)
        or len(devices) != 1
        or not isinstance(devices[0], dict)
        or set(devices[0]) != {"name", "capability"}
        or not isinstance(devices[0].get("name"), str)
        or not isinstance(devices[0].get("capability"), list)
        or any(type(value) is not int for value in devices[0]["capability"])
    ):
        raise ValueError("fit did not bind exactly one visible CUDA device")
    for key in ("numpy", "python", "machine", "cuda"):
        if not isinstance(runtime.get(key), str) or not runtime[key]:
            raise ValueError(f"runtime {key} identity is malformed")
    if type(runtime.get("cudnn")) is not int or runtime["cudnn"] <= 0:
        raise ValueError("runtime cuDNN version is malformed")
    _eq(
        contract.get("loader"),
        {
            "sampler_type": "wnn.training_sampling.EpochOrderSampler",
            "batch_size": 16,
            "drop_last": True,
            "num_workers": 0,
            "dataset_length": 11_940,
        },
        "continuation loader identity",
    )
    _eq(split_sha, SPLIT_SHA, "split manifest digest")


def _check_parent_checkpoint(
    source_root: Path, parent_acceptance: Mapping[str, object]
) -> tuple[object, str]:
    """Restricted-load the accepted small parent epoch-3 initializer."""
    path = source_root.parent / "data/artifacts/runs" / PARENT_RUN / "model_ep3.pt"
    payload, digest, stat_info = parent_v3._load_checkpoint(path, MAX_MODEL_BYTES)
    _eq(
        (digest, stat_info["bytes"]),
        (PARENT_CHECKPOINT_SHA, PARENT_CHECKPOINT_BYTES),
        "accepted parent checkpoint bytes",
    )
    if not isinstance(payload, dict):
        raise ValueError("parent initializer checkpoint is malformed")
    _eq((payload.get("epoch"), payload.get("n_in")), (3, 211), "parent initializer epoch/width")
    fingerprint = parent_v3._model_fingerprint(payload.get("model"))
    _eq(
        fingerprint,
        (parent_acceptance.get("model_tensor_fingerprint"), 16),
        "accepted parent model fingerprint",
    )
    return payload["model"], digest


def _check_child_normalizer(
    child_model: Mapping[str, object],
    parent_model: Mapping[str, object],
    provenance: Mapping[str, object],
    tensor_identity,
) -> dict[str, object]:
    """Confirm the saved child normalizer is the accepted parent prefix plus base suffix."""
    for key in ("mean", "std"):
        child = child_model[key]
        parent = parent_model[key]
        if not torch.equal(child[:211], parent):
            raise ValueError(f"child {key} does not preserve parent normalization prefix")
    if not torch.equal(child_model["mean"][211:], torch.zeros(256, dtype=torch.float32)):
        raise ValueError("base-mode child normalization suffix mean is not zero")
    if not torch.equal(child_model["std"][211:], torch.ones(256, dtype=torch.float32)):
        raise ValueError("base-mode child normalization suffix std is not one")
    identity = {key: tensor_identity(child_model[key]) for key in ("mean", "std")}
    _eq(identity, provenance.get("normalization_identity"), "child normalizer provenance")
    return identity


def _check_representation_payload(
    actual: object, expected: Mapping[str, object], label: str
) -> None:
    """Require an exact saved copy of the accepted dictionary transform."""
    if not isinstance(actual, dict):
        raise ValueError(f"{label} fitted input representation is missing")
    tensor_names = ("raw_mean", "raw_scale", "encoder_weight", "decoder_weight")
    _eq(
        set(actual),
        {"schema", *tensor_names, "fit_metadata", "state_sha256"},
        f"{label} representation schema",
    )
    _eq(
        (actual.get("schema"), actual.get("fit_metadata"), actual.get("state_sha256")),
        (expected.get("schema"), expected.get("fit_metadata"), expected.get("state_sha256")),
        f"{label} representation lineage",
    )
    for name in tensor_names:
        left, right = actual.get(name), expected.get(name)
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise ValueError(f"{label} representation tensor {name} is malformed")
        if (
            left.dtype != right.dtype
            or tuple(left.shape) != tuple(right.shape)
            or not torch.equal(left, right)
        ):
            raise ValueError(
                f"{label} representation tensor {name} differs from the accepted dictionary"
            )


def _check_dictionary_artifact(
    path: Path,
    split_sha: str,
    source_root: Path,
    representation_module: object,
    representation_fit_module: object,
    dictionary_module: object,
) -> tuple[dict[str, object], dict[str, object]]:
    """Restricted-load the accepted dictionary and validate its source-defined lineage."""
    for module, relative in (
        (representation_module, "src/wnn/training/input_representation.py"),
        (representation_fit_module, "src/wnn/training/input_representation_fit.py"),
        (dictionary_module, "src/wnn/training/input_representation_dictionary.py"),
    ):
        _eq(
            Path(module.__file__).resolve(strict=True),
            (source_root / relative).resolve(strict=True),
            f"dictionary helper source {relative}",
        )
    artifact, digest, stat_info = parent_v3._load_checkpoint(path, MAX_DICTIONARY_BYTES)
    _eq(
        (digest, stat_info["bytes"]),
        (DICTIONARY_SHA, DICTIONARY_BYTES),
        "accepted dictionary artifact bytes",
    )
    if not isinstance(artifact, dict) or set(artifact) != {
        "schema",
        "split_manifest_sha256",
        "fit_members",
        "fit_members_sha256",
        "recipe",
        "recipe_sha256",
        "representation",
    }:
        raise ValueError("accepted dictionary artifact schema differs")
    _eq(
        (artifact.get("schema"), artifact.get("split_manifest_sha256")),
        (dictionary_module.ARTIFACT_SCHEMA, split_sha),
        "dictionary artifact schema/fold",
    )
    members = artifact.get("fit_members")
    if not isinstance(members, list) or len(members) != dictionary_module.FIT_COMPONENTS:
        raise ValueError("accepted dictionary fit-member list is malformed")
    for member in members:
        if not isinstance(member, dict) or set(member) != {"source", "group", "component"}:
            raise ValueError("accepted dictionary fit member is malformed")
        _eq(member.get("source"), "train", "dictionary fit-member source")
        if any(
            type(member.get(key)) is not int or member[key] < 0 for key in ("group", "component")
        ):
            raise ValueError("accepted dictionary fit-member index is malformed")
    members_sha = hashlib.sha256(
        json.dumps(members, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "ascii"
        )
    ).hexdigest()
    _eq(artifact.get("fit_members_sha256"), members_sha, "dictionary fit-member digest")
    _eq(artifact.get("recipe"), dictionary_module.REPRESENTATION_RECIPE, "dictionary fixed recipe")
    _eq(
        artifact.get("recipe_sha256"),
        dictionary_module.representation_recipe_sha256(),
        "dictionary recipe digest",
    )
    serialized = artifact.get("representation")
    if not isinstance(serialized, dict):
        raise ValueError("accepted dictionary representation payload is missing")
    fitted = representation_module.deserialize_input_representation(serialized, "cpu")
    representation_fit_module.validate_fitted_representation(fitted, split_sha, members)
    if representation_module.input_representation_sha256(fitted) != serialized.get("state_sha256"):
        raise ValueError("accepted dictionary transform digest is invalid")
    _eq(
        (serialized.get("schema"), serialized.get("fit_metadata", {}).get("fit_members_sha256")),
        ("wnn.fitted_input_representation.v1", members_sha),
        "dictionary transform lineage",
    )
    return artifact, {
        "artifact_sha256": digest,
        "fit_members_sha256": members_sha,
        "fit_member_count": len(members),
        "recipe_sha256": artifact["recipe_sha256"],
        "state_sha256": serialized["state_sha256"],
    }


def _check_child_auxiliary(
    auxiliary: object,
    provenance: Mapping[str, object],
    dictionary_payload: Mapping[str, object],
    dictionary_path: Path,
) -> dict[str, object]:
    """Validate disabled auxiliary fields and the inherited frozen dictionary state."""
    if not isinstance(auxiliary, dict) or set(auxiliary) != {
        "norm_ema",
        "dynamics",
        "pool_memory",
        "readout_moments",
        "input_representation",
    }:
        raise ValueError("child auxiliary-state schema differs")
    expected_dynamics = {
        "switch": None,
        "caps": {},
        "eta": 0.0,
        "clip": 1.0,
        "no_clip": False,
        "seed": 1001 + 104729,
        "is_switched": False,
        "events": [],
        "generators": {},
    }
    _eq(auxiliary.get("dynamics"), expected_dynamics, "child fit-only optimizer dynamics")
    _eq(
        {key: auxiliary.get(key) for key in ("norm_ema", "pool_memory", "readout_moments")},
        {"norm_ema": None, "pool_memory": None, "readout_moments": None},
        "disabled child auxiliary state",
    )
    dictionary_representation = dictionary_payload.get("representation")
    if not isinstance(dictionary_representation, dict):
        raise ValueError("accepted dictionary transform is missing")
    _check_representation_payload(
        auxiliary.get("input_representation"), dictionary_representation, "training state"
    )
    identity = provenance.get("input_representation")
    if not isinstance(identity, dict) or set(identity) != {
        "mode",
        "state_sha256",
        "artifact_path",
        "artifact_file_sha256",
        "artifact_binding",
        "fit_metadata",
    }:
        raise ValueError("child fitted-representation provenance is malformed")
    _eq(
        identity,
        {
            "mode": "base",
            "state_sha256": dictionary_representation.get("state_sha256"),
            "artifact_path": str(dictionary_path),
            "artifact_file_sha256": DICTIONARY_SHA,
            "artifact_binding": "shared_fit_cache",
            "fit_metadata": dictionary_representation.get("fit_metadata"),
        },
        "child fitted-representation ancestry",
    )
    return {
        "state_sha256": identity["state_sha256"],
        "artifact_sha256": identity["artifact_file_sha256"],
        "artifact_binding": identity["artifact_binding"],
    }


def _check_paths(source_root: Path, run_dir: Path, split_path: Path, cache_root: Path) -> None:
    """Restrict the checker to the single immutable B source and CONTROL artifacts."""
    source = source_root.resolve(strict=True)
    base = source.parent
    _eq(source.name, "source-fg19-cache-68ff6829", "staged source directory")
    targets = {
        "source root": (source_root, base / "source-fg19-cache-68ff6829"),
        "run directory": (run_dir, base / "data/artifacts/runs" / RUN),
        "split manifest": (split_path, base / "splits/fold0.json"),
        "cache root": (cache_root, base / "research_tmp/fg19_t3_fit_homogeneous_cache_v1"),
    }
    for label, (path, expected) in targets.items():
        if path.is_symlink():
            raise ValueError(f"{label} must not be a symlink")
        _eq(path.resolve(strict=True), expected.resolve(strict=True), label)


def verify_terminal(
    run_dir: Path,
    source_root: Path,
    source_manifest: Path,
    split_path: Path,
    evidence_root: Path,
    profile_evidence_root: Path,
    admission_path: Path,
    launch_path: Path,
    launch_acceptance_path: Path,
    terminal_path: Path,
    cache_root: Path,
    cache_acceptance_path: Path,
    cache_stop_path: Path,
) -> dict[str, object]:
    """Verify captured exit, exact recipe and bounded completed artifacts.

    This function is intentionally post-exit: root captures native evidence first,
    then runs the checker, then performs acceptance and exact stop.
    """
    _check_paths(source_root, run_dir, split_path, cache_root)
    admission, admission_sha = _json_file(admission_path)
    launch, launch_sha = _json_file(launch_path)
    launch_acceptance, launch_acceptance_sha = _json_file(launch_acceptance_path)
    terminal, terminal_sha = _json_file(terminal_path)
    _check_control_launch_records(
        launch,
        launch_sha,
        launch_acceptance,
        launch_acceptance_sha,
        admission_sha,
    )
    _eq(
        (admission.get("schema"), admission.get("run_name"), admission.get("unit")),
        ("fg19-child-control-fit-admission.v1", RUN, UNIT),
        "CONTROL admission identity",
    )
    _eq(
        (
            admission.get("source_commit"),
            admission.get("source_manifest_sha256"),
            admission.get("source_files_verified"),
            admission.get("split_sha256"),
        ),
        (COMMIT, SOURCE_MANIFEST_SHA, 167, SPLIT_SHA),
        "CONTROL admission source/split",
    )
    _check_terminal(terminal)
    worker = admission.get("worker")
    if not isinstance(worker, dict):
        raise ValueError("admitted worker envelope is missing")
    parent_v3._check_worker(launch, admission, source_root)
    source = parent_v3._source_identity(source_root, source_manifest, admission)
    split, split_sha = _json_file(split_path)
    _eq(
        (split_sha, admission.get("split_sha256")),
        (SPLIT_SHA, SPLIT_SHA),
        "canonical split manifest",
    )
    cache = parent_v3._check_cache(
        admission, cache_root, cache_acceptance_path, cache_stop_path, split
    )
    lineage = _check_predecessor_receipts(evidence_root, profile_evidence_root, admission)
    _eq(
        admission.get("parent_acceptance_sha256"),
        PARENT_ACCEPTANCE_SHA,
        "admitted parent acceptance",
    )
    _eq(admission.get("parent_exact_stop_sha256"), PARENT_STOP_SHA, "admitted parent exact stop")
    _eq(
        admission.get("parent_checkpoint_sha256"),
        PARENT_CHECKPOINT_SHA,
        "admitted parent initializer",
    )
    _eq(
        admission.get("child_profile", {}).get("acceptance_sha256"),
        PROFILE_ACCEPTANCE_SHA,
        "admitted shape profile",
    )
    _eq(
        admission.get("child_profile", {}).get("exact_stop_sha256"),
        PROFILE_STOP_SHA,
        "admitted profile exact stop",
    )
    dictionary_identity = admission.get("dictionary")
    if not isinstance(dictionary_identity, dict):
        raise ValueError("shared dictionary identity is missing")
    _eq(
        (dictionary_identity.get("bytes"), dictionary_identity.get("sha256")),
        (DICTIONARY_BYTES, DICTIONARY_SHA),
        "admitted dictionary",
    )
    dictionary_path = (
        source_root.parent / "data/artifacts/research/fg19_dictionary_f0_v1/input_representation.pt"
    )
    parent_acceptance, _ = _read_pinned_json(
        evidence_root / "PARENT_TERMINAL_ACCEPTANCE.json", PARENT_ACCEPTANCE_SHA
    )
    parent_model, _ = _check_parent_checkpoint(source_root, parent_acceptance)
    from teacher_s5_terminal_check_v1 import _canonical_tensor_identity, _launch_cli_args

    launch_args = _launch_cli_args(launch, source_root)
    launch_args.update(
        {
            "valid_fold": 0,
            "n_valid_folds": 4,
            "eval_source": "vfold",
            "split_manifest_sha256": split_sha,
        }
    )
    command = launch.get("command")
    if not isinstance(command, list):
        raise ValueError("launch command is missing")
    positions = [
        index for index, value in enumerate(command) if value.endswith("/scripts/train_rnn.py")
    ]
    if len(positions) != 1:
        raise ValueError("launch must identify one canonical trainer entrypoint")
    argv_hash = hashlib.sha256(
        json.dumps(command[positions[0] + 1 :], separators=(",", ":")).encode()
    ).hexdigest()
    _eq(argv_hash, TRAINER_ARGV_SHA, "full CONTROL CLI argument digest")
    representation_module = importlib.import_module("wnn.training.input_representation")
    representation_fit_module = importlib.import_module("wnn.training.input_representation_fit")
    dictionary_module = importlib.import_module("wnn.training.input_representation_dictionary")
    dictionary_artifact, dictionary_identity = _check_dictionary_artifact(
        dictionary_path,
        split_sha,
        source_root,
        representation_module,
        representation_fit_module,
        dictionary_module,
    )
    _eq(launch_args.get("name"), RUN, "launched run name")
    _eq(launch_args.get("seed"), 1001, "launched seed")
    _eq(launch_args.get("batch_order"), "seeded", "launched sampler policy")
    _eq(launch_args.get("representation_mode"), "base", "launched representation")
    _eq(launch_args.get("mean_variance_head"), 1, "launched output head")
    _eq(launch_args.get("mean_variance_objective"), "wmse", "launched head objective")
    _eq(launch_args.get("loss"), "wmse", "launched fit objective")
    _eq(launch_args.get("checkpoint_epoch"), 5, "launched selected epoch")
    _eq(launch_args.get("epochs"), EPOCHS, "launched epoch horizon")
    _eq(launch_args.get("init"), f"{PARENT_RUN}@ep3", "launched parent initializer")
    _eq(launch_args.get("inherit_init_preprocessing"), True, "launched inherited preprocessing")
    _eq(launch_args.get("split_manifest"), str(split_path), "launched split path")
    _eq(launch_args.get("distill_cache_root"), str(cache_root), "launched cache path")
    _eq(launch_args.get("no_eval_during_fit"), True, "launched no-evaluation policy")
    expected_args = dict(launch_args)
    _eq(split_sha, SPLIT_SHA, "runtime split digest")
    run_files = {
        "model.pt": MAX_MODEL_BYTES,
        **{f"model_ep{epoch}.pt": MAX_MODEL_BYTES for epoch in range(EPOCHS)},
        "training_state.pt": MAX_STATE_BYTES,
        "provenance.json": MAX_JSON_BYTES,
        "history.json": MAX_JSON_BYTES,
    }
    paths = {name: run_dir / name for name in run_files}
    file_identities = {name: _hash_file(paths[name], limit) for name, limit in run_files.items()}
    if sum(entry[1]["bytes"] for entry in file_identities.values()) > MAX_AGGREGATE_BYTES:
        raise ValueError("child terminal artifacts exceed the aggregate byte cap")
    provenance, provenance_sha = _json_file(paths["provenance.json"])
    history_raw, history_sha, _ = _stable_file(paths["history.json"], MAX_JSON_BYTES)
    history = json.loads(history_raw)
    if not isinstance(history, list):
        raise ValueError("saved fit history is not a list")
    _eq(
        (provenance_sha, history_sha),
        (file_identities["provenance.json"][0], file_identities["history.json"][0]),
        "stable child metadata",
    )
    _eq(
        (provenance.get("selected_epoch"), provenance.get("split_manifest_sha256")),
        (5, SPLIT_SHA),
        "selected CONTROL endpoint",
    )
    _eq(provenance.get("args"), expected_args, "provenance arguments")
    _eq(provenance.get("seed"), 1001, "provenance seed")
    _eq(
        provenance.get("initial_checkpoint_sha256"), PARENT_CHECKPOINT_SHA, "provenance initializer"
    )
    _eq(provenance.get("propensity_checkpoint_sha256"), None, "unexpected propensity ancestry")
    membership = parent_v3._check_membership(provenance, split)
    parent_v3._check_cache_provenance(provenance, cache_root)
    tensor_identity = _canonical_tensor_identity(source_root)
    fingerprints: dict[str, tuple[str, int]] = {}
    models: dict[str, object] = {}
    for name in ("model.pt", "model_ep5.pt"):
        payload, digest, stat_info = parent_v3._load_checkpoint(paths[name], MAX_MODEL_BYTES)
        _eq((digest, stat_info), file_identities[name], f"stable {name}")
        _eq(
            (payload.get("epoch"), payload.get("n_in"), payload.get("args")),
            (5, 467, expected_args),
            f"{name} endpoint identity",
        )
        model = payload.get("model")
        fingerprints[name] = _model_fingerprint(model)
        models[name] = model
        child_provenance = payload.get("provenance")
        if not isinstance(child_provenance, dict):
            raise ValueError(f"{name} provenance is missing")
        _eq(
            {key: child_provenance.get(key) for key in CHECKPOINT_PROVENANCE_KEYS},
            {key: provenance.get(key) for key in CHECKPOINT_PROVENANCE_KEYS},
            f"{name} provenance identity",
        )
        _check_representation_payload(
            payload.get("input_representation"), dictionary_artifact["representation"], name
        )
    _eq(
        fingerprints["model.pt"],
        fingerprints["model_ep5.pt"],
        "selected child equals epoch-5 model",
    )
    normalizer = _check_child_normalizer(
        models["model.pt"], parent_model, provenance, tensor_identity
    )
    state, state_sha, state_stat = parent_v3._load_checkpoint(
        paths["training_state.pt"], MAX_STATE_BYTES
    )
    _eq((state_sha, state_stat), file_identities["training_state.pt"], "stable training state")
    _eq(
        (state.get("format"), state.get("version"), state.get("boundary"), state.get("next_epoch")),
        ("wnn.epoch-training-state", 1, "epoch", EPOCHS),
        "saved epoch boundary",
    )
    _eq(set(state), TRAINING_STATE_KEYS, "training-state schema")
    _eq(
        (state.get("ema"), state.get("swa"), state.get("scaler")),
        (None, None, {}),
        "disabled EMA/SWA/scaler state",
    )
    representation_identity = _check_child_auxiliary(
        state.get("auxiliary"), provenance, dictionary_artifact, dictionary_path
    )
    _eq(
        _model_fingerprint(state.get("model")),
        fingerprints["model.pt"],
        "training-state model fingerprint",
    )
    scheduler_lr, scheduler_beta1 = _check_scheduler(state.get("scheduler"))
    optimizer = _check_optimizer(state.get("optimizer"), scheduler_lr, scheduler_beta1)
    order_records = _check_counters(history, state, provenance)
    rng_signature = _check_loader_and_rng(state)
    manifest, _ = _json_file(source_manifest)
    _check_args_and_contract(state, provenance, admission, expected_args, manifest, split_sha)
    _eq(
        provenance.get("checkpoint_sha256"),
        {
            name: file_identities[name][0]
            for name in ("model.pt", *(f"model_ep{epoch}.pt" for epoch in range(EPOCHS)))
        },
        "checkpoint digest map",
    )
    log = _check_sample_log(evidence_root / "unit.log", history, order_records)
    return {
        "schema": "fg19.child-control-fit-terminal-check.v1",
        "status": "PASS",
        "run_name": RUN,
        "unit": UNIT,
        "invocation_id": INVOCATION,
        "terminal_evidence_sha256": terminal_sha,
        "launch_sha256": launch_sha,
        "launch_acceptance_sha256": launch_acceptance_sha,
        "source": source,
        "split_sha256": split_sha,
        "cache": cache,
        "membership_counts": membership,
        "observed_optimizer_updates": UPDATES,
        "next_epoch": EPOCHS,
        "selected_epoch": EPOCHS - 1,
        "model_tensor_count": fingerprints["model.pt"][1],
        "model_tensor_fingerprint": fingerprints["model.pt"][0],
        "normalization_identity": normalizer,
        "input_representation": representation_identity,
        "accepted_dictionary": dictionary_identity,
        "optimizer": optimizer,
        "rng_observed_after_restricted_load": rng_signature,
        "sample_order_observations": order_records,
        "predecessor_ancestry": lineage,
        "native_log": log,
        "cuda_resource_policy_source": {
            "path": "src/wnn/resources.py",
            "sha256": parent_v3.CUDA_POLICY_SOURCE_SHA,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        },
        "files": {
            name: {"sha256": value[0], "bytes": value[1]["bytes"]}
            for name, value in file_identities.items()
        },
    }


def main() -> int:
    """Verify the post-exit CONTROL record and create a private no-overwrite receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "run-dir",
        "source-root",
        "source-manifest",
        "split",
        "evidence-root",
        "profile-evidence-root",
        "admission",
        "launch",
        "launch-acceptance",
        "terminal-evidence",
        "cache-root",
        "cache-acceptance",
        "cache-exact-stop",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = verify_terminal(
        args.run_dir,
        args.source_root,
        args.source_manifest,
        args.split,
        args.evidence_root,
        args.profile_evidence_root,
        args.admission,
        args.launch,
        args.launch_acceptance,
        args.terminal_evidence,
        args.cache_root,
        args.cache_acceptance,
        args.cache_exact_stop,
    )
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
