from __future__ import annotations

import hashlib
from argparse import Namespace
from collections.abc import Mapping
from types import ModuleType
from typing import cast

from f3_checkpoint_common import (
    FOLD3_SHA256,
    expected_counts,
    is_resume_model_checkpoint,
    mapping,
    require,
)
from f3_checkpoint_semantics import validate_adam_state_descriptors


def _tensor_map_equal(
    left: Mapping[str, object], right: Mapping[str, object], torch: ModuleType
) -> bool:
    """Compare model tensor maps without converting device or dtype."""
    return set(left) == set(right) and all(torch.equal(left[key], right[key]) for key in left)  # type: ignore[attr-defined]


def _tensor_map_same_layout(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    """Compare tensor names and shapes while allowing checkpoint values to differ."""
    return set(left) == set(right) and all(
        tuple(left[key].shape) == tuple(right[key].shape) for key in left
    )


def _describe_adam_state(
    states: Mapping[str, object],
    parameter_ids: list[object],
    parameter_shapes: list[tuple[int, ...]],
    torch: ModuleType,
    expected_step_value: int,
) -> list[dict[str, object]]:
    """Validate real CPU Adam tensors and return descriptors for shape coverage."""
    descriptors: list[dict[str, object]] = []
    for parameter_id in parameter_ids:
        state = mapping(states[parameter_id], "Adam parameter state")
        require(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "Adam state fields")
        step, moment, square = (state[key] for key in ("step", "exp_avg", "exp_avg_sq"))
        require(
            all(torch.is_tensor(value) for value in (step, moment, square)),
            "Adam state tensor fields",
        )
        descriptors.append(
            {
                "state_fields": tuple(sorted(state)),
                "step_numel": step.numel(),
                "step_finite": bool(torch.isfinite(step).all()),
                "step_value": float(step.item()),
                "exp_avg_dtype": str(moment.dtype),
                "exp_avg_sq_dtype": str(square.dtype),
                "moments_finite": bool(
                    torch.isfinite(moment).all() and torch.isfinite(square).all()
                ),
                "exp_avg_shape": tuple(moment.shape),
                "exp_avg_sq_shape": tuple(square.shape),
            }
        )
    validate_adam_state_descriptors(parameter_shapes, descriptors, expected_step_value)
    return descriptors


def _tensor_values(value: object, torch: ModuleType) -> list[object]:
    """Flatten tensor objects from model and optimizer state without coercion."""
    if torch.is_tensor(value):  # type: ignore[attr-defined]
        return [value]
    if isinstance(value, Mapping):
        return [tensor for child in value.values() for tensor in _tensor_values(child, torch)]
    if isinstance(value, (list, tuple)):
        return [tensor for child in value for tensor in _tensor_values(child, torch)]
    return []


def verify_checkpoint_tensors(
    args: Namespace,
    state: Mapping[str, object],
    contract: Mapping[str, object],
    spec: Mapping[str, object],
    artifacts: Mapping[str, Mapping[str, object]],
    provenance: Mapping[str, object],
    torch: ModuleType,
) -> dict[str, int]:
    """Validate endpoint, raw checkpoint, normalization and Adam tensors on CPU."""
    model = mapping(state["model"], "resume model")
    torch_tensors = _tensor_values(model, torch)
    require(
        all(t.dtype == torch.float32 and bool(torch.isfinite(t).all()) for t in torch_tensors),
        "finite FP32 model tensors",
    )
    parameters = sum(t.numel() for key, t in model.items() if key not in ("mean", "std"))
    require(parameters == spec["parameters"], "trainable parameter count")
    require(len(torch_tensors) == (12 if args.arm == "wide" else 16), "model state tensor count")
    normalization = mapping(contract["normalization"], "contract normalization")
    for key in ("mean", "std"):
        tensor = model[key].detach().contiguous().cpu()
        actual = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "bytes_sha256": hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest(),
        }
        require(
            normalization[key] == actual and actual["shape"] == [211], f"normalization buffer {key}"
        )

    optimizer = mapping(state["optimizer"], "optimizer state")
    ids = [
        parameter
        for group in cast(list[dict[str, object]], optimizer["param_groups"])
        for parameter in cast(list[object], group["params"])
    ]
    states = mapping(optimizer["state"], "optimizer moments")
    require(len(ids) == len(set(ids)) and set(ids) == set(states), "Adam state parameter coverage")
    require(len(states) == (10 if args.arm == "wide" else 14), "Adam state count")
    optimizer_tensors = _tensor_values(states, torch)
    floating = [tensor for tensor in optimizer_tensors if tensor.is_floating_point()]
    require(
        all(
            tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all())
            for tensor in floating
        ),
        "finite FP32 Adam state",
    )
    model_parameter_shapes = [
        tuple(tensor.shape) for name, tensor in model.items() if name not in ("mean", "std")
    ]
    expected_step_value = expected_counts({"stage1": 2, "terminal": 6}[args.phase])[
        "optimizer_completed_updates"
    ]
    _describe_adam_state(states, ids, model_parameter_shapes, torch, expected_step_value)

    final_epoch = {"stage1": 1, "terminal": 5}[args.phase]
    checkpoint_names = (
        ["model_ep0.pt", "model_ep1.pt"] if args.phase == "stage1" else ["model_ep5.pt", "model.pt"]
    )
    for name in checkpoint_names:
        checkpoint = torch.load(args.snapshot_dir / name, map_location="cpu", weights_only=False)
        expected_epoch = (
            final_epoch
            if name == "model.pt" or name.endswith(f"{final_epoch}.pt")
            else int(name[8])
        )
        require(checkpoint.get("epoch") == expected_epoch, f"{name} epoch")
        raw_model = mapping(checkpoint["model"], f"{name} model")
        require(_tensor_map_same_layout(raw_model, model), f"{name} model layout")
        raw_args = mapping(checkpoint["args"], f"{name} args")
        expected_run = spec["run"]
        require(
            (
                raw_args.get("name"),
                raw_args.get("hidden"),
                raw_args.get("layers"),
                raw_args.get("seed"),
                raw_args.get("valid_fold"),
                raw_args.get("precision"),
            )
            == (expected_run, spec["hidden"], spec["layers"], 1001, 3, "fp32"),
            f"{name} checkpoint identity",
        )
        require(raw_args.get("split_manifest_sha256") == FOLD3_SHA256, f"{name} split manifest")
        embedded = mapping(checkpoint["provenance"], f"{name} embedded provenance")
        for field in (
            "args",
            "trainer_implementation",
            "fit_membership_identity",
            "normalization_identity",
            "resumed_from",
            "split_manifest_sha256",
        ):
            require(embedded.get(field) == provenance.get(field), f"{name} provenance {field}")
        require(
            provenance["checkpoint_sha256"][name] == artifacts[name]["sha256"],
            f"{name} producer hash",
        )
        raw_tensors = _tensor_values(raw_model, torch)
        require(
            all(t.dtype == torch.float32 and bool(torch.isfinite(t).all()) for t in raw_tensors),
            f"{name} tensors are not finite FP32",
        )
        if is_resume_model_checkpoint(name, args.phase):
            require(_tensor_map_equal(raw_model, model, torch), f"{name} differs from resume model")
    if args.phase == "terminal":
        raw = torch.load(args.snapshot_dir / "model_ep5.pt", map_location="cpu", weights_only=False)
        selected = torch.load(
            args.snapshot_dir / "model.pt", map_location="cpu", weights_only=False
        )
        require(
            _tensor_map_equal(
                mapping(raw["model"], "raw epoch5 model"),
                mapping(selected["model"], "selected model"),
                torch,
            ),
            "selected/raw equality",
        )
        require(provenance.get("selected_epoch") == 5, "fixed selected epoch")
    require(not torch.cuda.is_initialized(), "CUDA initialized")
    require(
        all(t.device.type == "cpu" for t in torch_tensors + optimizer_tensors),
        "checkpoint tensors not on CPU",
    )
    return {
        "model_tensors_verified": len(torch_tensors),
        "trainable_parameters": parameters,
        "adam_states_verified": len(states),
        "adam_floating_tensors_verified": len(floating),
    }
