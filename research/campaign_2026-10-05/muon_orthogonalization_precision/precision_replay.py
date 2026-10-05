"""Local E487 Muon precision replays; shared optimizer sources stay unchanged."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E485_DIR = ROOT / "research/campaign_2026-10-05/activation_duration"
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
E471_DIR = ROOT / "research/campaign_2026-10-05/recurrent_activation"
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
for directory in (ROOT / "src", E485_DIR, E470_DIR, E471_DIR, E474_DIR):
    sys.path.insert(0, str(directory))

import fit_activation_optimizer as e470  # noqa: E402
import fit_activation_placement as e474  # noqa: E402
import fit_duration as e485  # noqa: E402

from wnn.nn import RNNModel  # noqa: E402
from wnn.optim import NS_COEFFS, MuonAdamW, muon_param_groups  # noqa: E402

SEED = 271836
PRECISIONS = ("bf16_source", "fp32_diagnostic")
GRADIENT_SOURCES = ("native", "custom")
STRICT_RELATIVE_RMS = 0.02
STRICT_MAX_ABS = 2e-5
CUDA_FP32_CPU_ATOL = 1e-5
CUDA_FP32_CPU_RTOL = 1e-4
DEVICE_STOP_BYTES = 3_000_000_000
ALLOCATOR_MAX_BYTES = 2 * 1024**3
E485_FILES = {
    "fit_duration.py": "d9a1b9d243a9c9e50082fe0d9e7d98186d63cb0c566fcae0a20c00ccd42c4b48",
    "tests.py": "4f0579a73a7492ce960582d7f744f9b29152e7b5860219593c9fc63a89005534",
    "gpu_probe.py": "7f3b9a808eb8adbd0856b61fe8bd36ba53df2c1cd1455b697eb748b880d444fd",
    "GPU_PROBE_UPDATE_GATES.md": "52d547abd2da11cd712e5973b2ef1ac2eca70b449cbd143abf8141bea221d164",
}


class FP32MuonAdamW(MuonAdamW):
    """Use the ordinary MuonAdamW update except for FP32 NS intermediates."""

    @torch.no_grad()
    def step(self, closure=None):
        """Perform the registered update with a diagnostic FP32 orthogonalizer."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, weight_decay, eps = group["lr"], group["weight_decay"], group["eps"]
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                state = self.state[parameter]
                if group["muon"]:
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    buffer = state["momentum_buffer"]
                    buffer.lerp_(gradient, 1 - beta1)
                    blocks = group["blocks"]
                    rows, columns = parameter.shape[0] // blocks, parameter.shape[1]
                    momentum = gradient.lerp(buffer, beta1).reshape(blocks, rows, columns)
                    update = orthogonalize_fp32(momentum, group["ns_steps"])
                    parameter.mul_(1 - lr * weight_decay)
                    parameter.add_(
                        update.reshape(parameter.shape),
                        alpha=-lr * group["rms"] * math.sqrt(max(rows, columns)),
                    )
                    continue
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                parameter.mul_(1 - lr * weight_decay)
                exp_avg.lerp_(gradient, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                bias1 = 1 - beta1 ** state["step"]
                bias2 = 1 - beta2 ** state["step"]
                denominator = (exp_avg_sq.sqrt() / math.sqrt(bias2)).add_(eps)
                parameter.addcdiv_(exp_avg, denominator, value=-lr / bias1)
        return loss


def orthogonalize_fp32(gradient: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Apply the frozen quintic NS formula with FP32 intermediates only."""
    a, b, c = NS_COEFFS
    value = gradient.to(torch.float32)
    transposed = value.size(-2) > value.size(-1)
    if transposed:
        value = value.mT
    value = value / (value.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        gram = value @ value.mT
        value = a * value + (b * gram + c * gram @ gram) @ value
    if transposed:
        value = value.mT
    return value.to(gradient.dtype)


def require_finite(value: Tensor, label: str) -> None:
    """Reject nonfinite tensors before clipping, updates, or comparisons."""
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"nonfinite {label}")


def require_optimizer_state_finite(optimizer: torch.optim.Optimizer, label: str) -> None:
    """Check every populated tensor and numeric scalar in optimizer state."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                require_finite(value, f"{label} optimizer state {key}")
            elif isinstance(value, (float, int)) and not math.isfinite(float(value)):
                raise ValueError(f"nonfinite {label} optimizer state scalar {key}")


def validate_raw_feature_array(raw: np.ndarray, label: str) -> str:
    """Validate unsanitized features and return a digest for audit receipts."""
    values = np.asarray(raw)
    if values.ndim != 2 or values.shape[0] != 20_000 or values.shape[1] != 20:
        raise ValueError(f"unexpected raw feature shape for {label}: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"nonfinite raw feature before causal_features: {label}")
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def tensor_digest(value: Tensor) -> str:
    """Hash one tensor's contiguous CPU representation with dtype/shape."""
    cpu = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode())
    digest.update(np.asarray(cpu.shape, dtype=np.int64).tobytes())
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def mapping_digest(values: dict[str, Tensor]) -> str:
    """Hash a named tensor mapping in lexical key order."""
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode())
        digest.update(tensor_digest(values[name]).encode())
    return digest.hexdigest()


def state_digest(state: dict[str, Tensor]) -> str:
    """Hash a module state dictionary without storing device tensors."""
    return mapping_digest(state)


def optimizer_state_digest(
    optimizer: torch.optim.Optimizer, model: torch.nn.Module | None = None
) -> str:
    """Hash first-step optimizer state, including scalar counters."""
    digest = hashlib.sha256()
    named = (
        {id(parameter): name for name, parameter in model.named_parameters()}
        if model is not None
        else {id(parameter): name for name, parameter in optimizer_model_parameters(optimizer)}
    )
    for parameter, values in optimizer.state.items():
        name = named.get(id(parameter), f"unnamed-{len(named)}")
        digest.update(name.encode())
        for key in sorted(values):
            digest.update(key.encode())
            value = values[key]
            if isinstance(value, Tensor):
                digest.update(tensor_digest(value).encode())
            else:
                digest.update(repr(value).encode())
    return digest.hexdigest()


def optimizer_model_parameters(
    optimizer: torch.optim.Optimizer,
) -> list[tuple[str, torch.nn.Parameter]]:
    """Name optimizer parameters using attached diagnostic names."""
    result = []
    for group_index, group in enumerate(optimizer.param_groups):
        names = group.get("_e487_names", [])
        if len(names) != len(group["params"]):
            # Optimizers do not retain a model reference; names are installed by
            # make_optimizer, so missing metadata is a proof failure.
            raise ValueError(f"optimizer group {group_index} lacks parameter names")
        result.extend(zip(names, group["params"], strict=True))
    return result


def model_state(model: torch.nn.Module) -> dict[str, Tensor]:
    """Return a detached cloned state for the registered common initialization."""
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def check_common_initialization(model: torch.nn.Module, initial: dict[str, Tensor]) -> bool:
    """Require all state keys, dtypes, shapes and bytes to match exactly."""
    current = model.state_dict()
    if current.keys() != initial.keys():
        raise ValueError("model parameter/state keys differ from the common initialization")
    for name, value in current.items():
        first = value.detach().cpu().contiguous()
        second = initial[name].detach().cpu().contiguous()
        if first.dtype != second.dtype or first.shape != second.shape:
            raise ValueError(f"common initialization metadata differs for {name}")
        if first.numpy().tobytes() != second.numpy().tobytes():
            raise ValueError(f"common initialization bytes differ for {name}")
    return True


def _groups_with_names(model: torch.nn.Module) -> list[dict]:
    """Construct and annotate the unchanged E485 Muon/AdamW partition."""
    groups = muon_param_groups(model, scope="rnn")
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    assigned = [id(parameter) for group in groups for parameter in group["params"]]
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if len(assigned) != len(set(assigned)) or set(assigned) != trainable:
        raise ValueError("optimizer groups do not exactly partition trainable parameters")
    muon_names = {
        parameter_names[id(parameter)]
        for group in groups
        if group.get("muon")
        for parameter in group["params"]
    }
    if muon_names != e485.MUON_NAMES:
        raise ValueError(f"unexpected E485 Muon matrix group: {sorted(muon_names)}")
    for group in groups:
        group["_e487_names"] = [parameter_names[id(parameter)] for parameter in group["params"]]
    return groups


def make_optimizer(model: torch.nn.Module, precision: str) -> MuonAdamW:
    """Create either the untouched source optimizer or local FP32 diagnostic."""
    if precision not in PRECISIONS:
        raise ValueError(f"unregistered orthogonalizer precision: {precision}")
    groups = _groups_with_names(model)
    kwargs = {
        "lr": e485.LEARNING_RATE,
        "weight_decay": e485.WEIGHT_DECAY,
        "betas": e485.BETAS,
    }
    optimizer_class = MuonAdamW if precision == "bf16_source" else FP32MuonAdamW
    return optimizer_class(groups, **kwargs)


def named_parameter_state(model: torch.nn.Module) -> dict[str, Tensor]:
    """Clone named parameters to CPU for fixed-gradient replay."""
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}


def initialized_model(initial: dict[str, Tensor], model_kind: str) -> torch.nn.Module:
    """Construct native or custom wrapper over the identical parameter bytes."""
    if model_kind == "native":
        model = RNNModel(42, hidden=e485.HIDDEN, layers=2, dropout=0.0, n_out=2)
        model.load_state_dict(initial, strict=True)
    elif model_kind == "custom":
        model = e485.initialize_arm(0.0, initial)
    else:
        raise ValueError(f"unregistered replay model kind: {model_kind}")
    check_common_initialization(model, initial)
    return model


def capture_gradients(
    model: torch.nn.Module, x: Tensor, y: Tensor, mask: Tensor, source: str
) -> dict:
    """Capture one train-mode gradient snapshot and forward-parity receipt."""
    if source not in GRADIENT_SOURCES:
        raise ValueError(f"unregistered gradient source: {source}")
    model.train()
    x_live = x.detach().clone().requires_grad_(True)
    prediction, final_state = model(x_live)
    require_finite(prediction, f"{source} prediction")
    require_finite(final_state, f"{source} final state")
    loss = e470.loss_for_batch(prediction, y, mask)
    require_finite(loss, f"{source} loss")
    loss.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise ValueError(f"missing {source} gradient for {name}")
        require_finite(parameter.grad, f"{source} raw gradient {name}")
        gradients[name] = parameter.grad.detach().cpu().clone()
    if x_live.grad is None:
        raise ValueError(f"missing {source} input gradient")
    require_finite(x_live.grad, f"{source} input gradient")
    squared = sum(float(gradient.double().square().sum()) for gradient in gradients.values())
    raw_norm = math.sqrt(squared)
    if not math.isfinite(raw_norm) or raw_norm <= 0:
        raise ValueError(f"invalid {source} global raw gradient norm")
    return {
        "source": source,
        "loss": float(loss.detach().cpu()),
        "prediction": prediction.detach().cpu(),
        "final_state": final_state.detach().cpu(),
        "input_gradient": x_live.grad.detach().cpu().clone(),
        "raw_gradients": gradients,
        "raw_gradient_norm": raw_norm,
        "raw_gradient_sha256": mapping_digest(gradients),
        "input_gradient_sha256": tensor_digest(x_live.grad),
    }


def replay(
    initial: dict[str, Tensor],
    gradients: dict[str, Tensor],
    precision: str,
    device: torch.device,
    owner: str,
    model_kind: str = "custom",
) -> dict:
    """Apply one fixed gradient snapshot from identical bytes/fresh state."""
    model = initialized_model(initial, model_kind).to(device=device, dtype=torch.float32)
    check_common_initialization(model, initial)
    parameter_names = {name for name, _parameter in model.named_parameters()}
    if parameter_names != set(gradients):
        raise ValueError("replay gradient parameter names differ")
    parameters = dict(model.named_parameters())
    before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    optimizer = make_optimizer(model, precision)
    optimizer_before = optimizer_state_digest(optimizer, model)
    if optimizer.state:
        raise ValueError("replay optimizer state must be fresh/zero-buffer")
    raw_grads = {}
    for name, parameter in parameters.items():
        value = gradients[name].to(device=device, dtype=torch.float32).clone()
        require_finite(value, f"{owner} raw gradient {name}")
        parameter.grad = value
        raw_grads[name] = value.detach().clone()
    raw_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(value.double()) for value in raw_grads.values()]), 2
    )
    if not torch.isfinite(raw_norm):
        raise ValueError(f"nonfinite {owner} global gradient norm")
    raw_norm_value = float(raw_norm.cpu())
    clipped = bool(raw_norm_value > e485.CLIP_NORM)
    nn_grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), e485.CLIP_NORM, error_if_nonfinite=True
    )
    clipped_grads = {
        name: parameter.grad.detach().clone() for name, parameter in parameters.items()
    }
    clipped_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(value.double()) for value in clipped_grads.values()]),
        2,
    )
    require_finite(clipped_norm, f"{owner} clipped global gradient norm")
    optimizer.step()
    require_optimizer_state_finite(optimizer, owner)
    new_parameters = dict(model.named_parameters())
    deltas = {
        name: (new_parameters[name].detach() - before[name]).cpu().clone() for name in parameters
    }
    for name, parameter in new_parameters.items():
        require_finite(parameter, f"{owner} updated parameter {name}")
        require_finite(deltas[name], f"{owner} update delta {name}")
    return {
        "owner": owner,
        "model_kind": model_kind,
        "precision": precision,
        "state_keys": sorted(model.state_dict()),
        "parameter_sha256_before": state_digest(before),
        "raw_gradient_sha256": mapping_digest(raw_grads),
        "clipped_gradient_sha256": mapping_digest(clipped_grads),
        "optimizer_state_sha256_before": optimizer_before,
        "optimizer_state_sha256_after": optimizer_state_digest(optimizer, model),
        "optimizer_state_finite": True,
        "raw_gradient_norm": raw_norm_value,
        "clip_applied": clipped,
        "clip_return_norm": float(nn_grad_norm.detach().cpu()),
        "clipped_gradient_norm": float(clipped_norm.cpu()),
        "deltas": deltas,
        "updated_state": {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        },
    }


def rms_by_names(values: dict[str, Tensor], names: list[str]) -> float:
    """Return element-weighted RMS for a nonempty, finite tensor family."""
    if not names:
        raise ValueError("cannot summarize an empty tensor family")
    elements = 0
    squared = 0.0
    for name in names:
        value = values[name].detach().double()
        require_finite(value, f"RMS input {name}")
        elements += value.numel()
        squared += float(value.square().sum())
    return math.sqrt(squared / elements)


def compare_updates(reference: dict[str, Tensor], candidate: dict[str, Tensor]) -> dict:
    """Record all strict matrix/family gates without aborting on a finite failure."""
    if reference.keys() != candidate.keys():
        raise ValueError("update parameter names differ")
    all_names = sorted(reference)
    for name in all_names:
        require_finite(reference[name], f"reference update {name}")
        require_finite(candidate[name], f"candidate update {name}")
    groups: dict[str, list[str]] = {f"matrix:{name}": [name] for name in sorted(e485.MUON_NAMES)}
    groups.update(
        {
            "family:inp": [name for name in all_names if name.startswith("inp.")],
            "family:gru_bias": [name for name in all_names if name.startswith("gru.bias_")],
            "family:head": [name for name in all_names if name.startswith("head.")],
        }
    )
    covered = {name for group_names in groups.values() for name in group_names}
    if covered != set(all_names):
        raise ValueError(
            f"update gate does not cover parameters: {sorted(set(all_names) - covered)}"
        )
    metrics = {}
    for label, names in groups.items():
        reference_rms = rms_by_names(reference, names)
        candidate_rms = rms_by_names(candidate, names)
        error_values = {name: candidate[name] - reference[name] for name in names}
        error_rms = rms_by_names(error_values, names)
        max_abs = max(float(error_values[name].abs().max()) for name in names)
        if reference_rms == 0.0:
            relative = 0.0 if error_rms == 0.0 else None
            ratio_pass = error_rms == 0.0
        else:
            relative = error_rms / reference_rms
            ratio_pass = relative <= STRICT_RELATIVE_RMS
        metrics[label] = {
            "reference_update_rms": reference_rms,
            "candidate_update_rms": candidate_rms,
            "error_update_rms": error_rms,
            "relative_error_rms": relative,
            "max_abs_error": max_abs,
            "relative_rms_limit": STRICT_RELATIVE_RMS,
            "max_abs_limit": STRICT_MAX_ABS,
            "pass_relative_rms": ratio_pass,
            "pass_max_abs": max_abs <= STRICT_MAX_ABS,
            "passed": bool(ratio_pass and max_abs <= STRICT_MAX_ABS),
        }
    return {"metrics": metrics, "passed": all(item["passed"] for item in metrics.values())}


def precision_delta_sensitivity(
    bf16_replay: dict[str, Tensor], fp32_replay: dict[str, Tensor]
) -> dict[str, dict[str, float]]:
    """Report BF16-vs-FP32 delta change with no admission threshold."""
    if bf16_replay.keys() != fp32_replay.keys():
        raise ValueError("precision replay names differ")
    result = {}
    for name in sorted(bf16_replay):
        difference = fp32_replay[name] - bf16_replay[name]
        require_finite(difference, f"precision delta change {name}")
        result[name] = {
            "bf16_delta_rms": rms_by_names(bf16_replay, [name]),
            "fp32_delta_rms": rms_by_names(fp32_replay, [name]),
            "precision_change_rms": rms_by_names({name: difference}, [name]),
            "precision_change_max_abs": float(difference.abs().max()),
        }
    return result


def _tensor_maps_equal(first: dict[str, Tensor], second: dict[str, Tensor]) -> bool:
    """Return exact byte equality for same-device replay states."""
    return first.keys() == second.keys() and all(
        tensor_digest(first[name]) == tensor_digest(second[name]) for name in first
    )


def shared_gradient_control(
    initial: dict[str, Tensor],
    device: torch.device,
    before_replay: Callable[[], dict] | None = None,
    after_replay: Callable[[], dict] | None = None,
) -> dict:
    """Check deterministic same-gradient native/custom clones per precision."""
    controls = {}
    base_model = e485.initialize_arm(0.0, initial)
    synthetic = {}
    for index, (name, parameter) in enumerate(base_model.named_parameters()):
        sequence = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape_as(parameter)
        synthetic[name] = (sequence.remainder(31) - 15.0) * 0.01 + (index + 1) * 0.001
    for precision in PRECISIONS:
        native_before_memory = before_replay() if before_replay is not None else None
        native = replay(
            initial,
            synthetic,
            precision,
            device,
            f"shared-control/native/{precision}",
            model_kind="native",
        )
        native_memory = after_replay() if after_replay is not None else None
        custom_before_memory = before_replay() if before_replay is not None else None
        custom = replay(
            initial,
            synthetic,
            precision,
            device,
            f"shared-control/custom/{precision}",
            model_kind="custom",
        )
        custom_memory = after_replay() if after_replay is not None else None
        exact_deltas = _tensor_maps_equal(native["deltas"], custom["deltas"])
        exact_states = _tensor_maps_equal(native["updated_state"], custom["updated_state"])
        exact_opt_state = (
            native["optimizer_state_sha256_after"] == custom["optimizer_state_sha256_after"]
        )
        if not (exact_deltas and exact_states and exact_opt_state):
            raise ValueError(f"same-gradient duplicate control failed: {precision}")
        if precision == "bf16_source":
            frozen_model = e485.initialize_arm(0.0, initial).to(device=device, dtype=torch.float32)
            frozen_optimizer = e485.make_optimizer(frozen_model)
            frozen_before_memory = before_replay() if before_replay is not None else None
            frozen_before = {
                name: parameter.detach().clone()
                for name, parameter in frozen_model.named_parameters()
            }
            frozen_optimizer_before_hash = optimizer_state_digest(frozen_optimizer, frozen_model)
            parameter_map = dict(frozen_model.named_parameters())
            for name, parameter in parameter_map.items():
                parameter.grad = synthetic[name].to(device=device, dtype=torch.float32).clone()
            frozen_telemetry = e474.apply_optimizer_update(
                frozen_model,
                frozen_optimizer,
                e474.parameter_families(frozen_model),
                0,
                "E487/shared-gradient/frozen-E474",
            )
            require_optimizer_state_finite(frozen_optimizer, "shared-gradient frozen E474")
            frozen_state = {
                name: value.detach().cpu().clone()
                for name, value in frozen_model.state_dict().items()
            }
            frozen_memory = after_replay() if after_replay is not None else None
            frozen_deltas = {
                name: (parameter.detach() - frozen_before[name]).cpu().clone()
                for name, parameter in frozen_model.named_parameters()
            }
            frozen_clipped_gradients = {
                name: parameter.grad.detach().clone()
                for name, parameter in frozen_model.named_parameters()
            }
            frozen_clipped_hash = mapping_digest(frozen_clipped_gradients)
            frozen_optimizer_hash = optimizer_state_digest(frozen_optimizer, frozen_model)
            frozen_replay = {
                "owner": "shared-control/frozen-e474/bf16_source",
                "model_kind": "custom",
                "precision": "bf16_source",
                "parameter_sha256_before": state_digest(frozen_before),
                "raw_gradient_sha256": mapping_digest(synthetic),
                "clipped_gradient_sha256": frozen_clipped_hash,
                "optimizer_state_sha256_before": frozen_optimizer_before_hash,
                "optimizer_state_sha256_after": frozen_optimizer_hash,
                "deltas": frozen_deltas,
                "updated_state": frozen_state,
                "memory_after_replay": frozen_memory,
                "memory_before_replay": frozen_before_memory,
                "clip_applied": bool(frozen_telemetry is not None),
            }
            if not _tensor_maps_equal(custom["updated_state"], frozen_state):
                raise ValueError("BF16 shared-gradient control differs from frozen E474")
            if custom["optimizer_state_sha256_after"] != frozen_optimizer_hash:
                raise ValueError("BF16 shared-gradient optimizer state differs from frozen E474")
            if custom["clipped_gradient_sha256"] != frozen_clipped_hash:
                raise ValueError(
                    "BF16 shared-gradient clipped tensor bytes differ from frozen E474"
                )
        else:
            frozen_telemetry = None
            frozen_memory = None
            frozen_replay = None
            frozen_before_memory = None
        controls[precision] = {
            "same_gradient_delta_bytes_exact": exact_deltas,
            "same_gradient_model_state_bytes_exact": exact_states,
            "same_gradient_optimizer_state_hash_equal": exact_opt_state,
            "clip_activated": native["clip_applied"] and custom["clip_applied"],
            "raw_gradient_norm": native["raw_gradient_norm"],
            "frozen_e474_equal": True if precision == "bf16_source" else None,
            "frozen_e474_telemetry": frozen_telemetry,
            "memory_after_native": native_memory,
            "memory_after_custom": custom_memory,
            "memory_after_frozen_e474": frozen_memory,
            "memory_before_native": native_before_memory,
            "memory_before_custom": custom_before_memory,
            "memory_before_frozen_e474": frozen_before_memory,
            "shared_gradient_sha256": mapping_digest(synthetic),
            "replays": {
                "native": native,
                "custom": custom,
                "frozen_e474": frozen_replay,
            },
        }
        if not controls[precision]["clip_activated"]:
            raise ValueError(f"shared-gradient control did not clip: {precision}")
    return controls


def replay_crossed_gradients(
    initial: dict[str, Tensor],
    captured: dict[str, dict],
    device: torch.device,
    before_replay: Callable[[], dict] | None = None,
    after_replay: Callable[[], dict] | None = None,
) -> dict:
    """Run all four finite replay cells and record strict gate booleans."""
    if set(captured) != set(GRADIENT_SOURCES):
        raise ValueError("crossed replay requires both native and custom gradients")
    cells = {}
    clipped_hashes: dict[tuple[str, str], str] = {}
    for source in GRADIENT_SOURCES:
        raw = captured[source]["raw_gradients"]
        for precision in PRECISIONS:
            before_memory = before_replay() if before_replay is not None else None
            cell = replay(
                initial,
                raw,
                precision,
                device,
                f"crossed/{source}/{precision}",
                model_kind=source,
            )
            cell["memory_before_replay"] = before_memory
            cell["memory_after_replay"] = after_replay() if after_replay is not None else None
            if not cell["clip_applied"]:
                raise ValueError(f"registered raw gradient did not activate clipping: {source}")
            clipped_hashes[(source, precision)] = cell["clipped_gradient_sha256"]
            cells[f"{source}__{precision}"] = cell
    for source in GRADIENT_SOURCES:
        if clipped_hashes[(source, PRECISIONS[0])] != clipped_hashes[(source, PRECISIONS[1])]:
            raise ValueError(f"precision variants changed clipped gradient bytes: {source}")
    gates = {}
    sensitivities = {}
    for precision in PRECISIONS:
        native = cells[f"native__{precision}"]
        custom = cells[f"custom__{precision}"]
        gates[precision] = compare_updates(native["deltas"], custom["deltas"])
    for source in GRADIENT_SOURCES:
        sensitivities[source] = precision_delta_sensitivity(
            cells[f"{source}__bf16_source"]["deltas"],
            cells[f"{source}__fp32_diagnostic"]["deltas"],
        )
    return {
        "crossed_replays": cells,
        "native_custom_strict_gates": gates,
        "bf16_fp32_delta_sensitivity": sensitivities,
        "all_crossed_finite": all(
            cell["optimizer_state_finite"]
            and all(bool(torch.isfinite(value).all()) for value in cell["deltas"].values())
            and all(bool(torch.isfinite(value).all()) for value in cell["updated_state"].values())
            for cell in cells.values()
        ),
        "expected_gate_failures_are_recorded_not_raised": True,
    }


def capture_actual_fit_batch() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Load one canonical fit-only B4x256 batch and its common initialization."""
    e485.parent_source_hashes()
    manifest = json.loads(e470.COHORT_PATH.read_text())
    fit_nodes = [node for node in manifest["cohort"] if node.get("role") == "fit"]
    if len(fit_nodes) != 24 or len({node["row_group"] for node in fit_nodes}) != 24:
        raise ValueError("E487 raw-feature check requires 24 unique fit-role row groups")
    parquet = e470.pq.ParquetFile(e470.DATA_PATH)
    raw_feature_digests = []
    for node in fit_nodes:
        table = parquet.read_row_group(
            node["row_group"], columns=list(e470.FEATURES), use_threads=False
        )
        raw = np.column_stack([table[column].to_numpy() for column in e470.FEATURES])
        raw_feature_digests.append(validate_raw_feature_array(raw, str(node["row_group"])))
    result, x, y, need, _unused, nodes = e470.load_cohort(include_test=False)
    if len(nodes) != 24 or any(item.get("role") != "fit" for item in nodes):
        raise ValueError("E487 may load only the frozen 24 fit-role groups")
    if x.shape != (24, e485.ROWS_PER_GROUP, e485.FEATURES):
        raise ValueError(f"unexpected fit cohort feature shape: {x.shape}")
    if y.shape != (24, e485.ROWS_PER_GROUP, 2) or need.shape != (24, e485.ROWS_PER_GROUP):
        raise ValueError("fit cohort label or mask shape differs")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("nonfinite raw fit-only feature or label")
    order = e485.canonical_epoch_order(SEED, 0)
    row_indices = order[: e485.BATCH_SIZE].tolist()
    selected_nodes = [nodes[index] for index in row_indices]
    torch.manual_seed(SEED)
    native = RNNModel(42, hidden=e485.HIDDEN, layers=2, dropout=0.0, n_out=2)
    if sum(parameter.numel() for parameter in native.parameters()) != e485.EXPECTED_PARAMETERS:
        raise ValueError("native H128 model parameter count differs")
    initial = model_state(native)
    custom = e485.initialize_arm(0.0, initial)
    check_common_initialization(native, initial)
    check_common_initialization(custom, initial)
    batch_x = torch.from_numpy(x[row_indices, : e485.CHUNK_ROWS]).float()
    batch_y = torch.from_numpy(y[row_indices, : e485.CHUNK_ROWS]).float()
    batch_mask = torch.from_numpy(need[row_indices, : e485.CHUNK_ROWS])
    require_finite(batch_x, "raw fit-only features")
    require_finite(batch_y, "raw fit-only labels")
    return (
        {
            "cohort": result,
            "initial": initial,
            "native": native,
            "custom": custom,
            "row_indices": row_indices,
            "fit_row_groups": [item["row_group"] for item in selected_nodes],
            "raw_feature_validation": {
                "all_24_fit_groups_checked_before_causal_sanitization": True,
                "fit_row_groups_in_manifest_order": [node["row_group"] for node in fit_nodes],
                "raw_feature_sha256_in_manifest_order": raw_feature_digests,
            },
            "feature_batch": batch_x,
            "label_batch": batch_y,
            "need_mask": batch_mask,
        },
        x,
        y,
        need,
        nodes,
    )


def run_fit_only_cpu_preflight() -> dict:
    """Run source-bound one-batch captures/replays on fit roles only."""
    batch, _x, _y, _need, _nodes = capture_actual_fit_batch()
    initial = batch["initial"]
    native_receipt = capture_gradients(
        batch["native"], batch["feature_batch"], batch["label_batch"], batch["need_mask"], "native"
    )
    custom_receipt = capture_gradients(
        batch["custom"], batch["feature_batch"], batch["label_batch"], batch["need_mask"], "custom"
    )
    output_state_parity = {
        "max_abs_output": float(
            (native_receipt["prediction"] - custom_receipt["prediction"]).abs().max()
        ),
        "max_abs_state": float(
            (native_receipt["final_state"] - custom_receipt["final_state"]).abs().max()
        ),
        "max_abs_input_gradient": float(
            (native_receipt["input_gradient"] - custom_receipt["input_gradient"]).abs().max()
        ),
        "max_abs_parameter_gradient": max(
            float(
                (native_receipt["raw_gradients"][name] - custom_receipt["raw_gradients"][name])
                .abs()
                .max()
            )
            for name in native_receipt["raw_gradients"]
        ),
        "loss_native": native_receipt["loss"],
        "loss_custom": custom_receipt["loss"],
    }
    torch.testing.assert_close(
        custom_receipt["prediction"], native_receipt["prediction"], atol=2e-5, rtol=2e-4
    )
    torch.testing.assert_close(
        custom_receipt["final_state"], native_receipt["final_state"], atol=2e-5, rtol=2e-4
    )
    torch.testing.assert_close(
        custom_receipt["input_gradient"], native_receipt["input_gradient"], atol=5e-5, rtol=5e-4
    )
    for name in native_receipt["raw_gradients"]:
        torch.testing.assert_close(
            custom_receipt["raw_gradients"][name],
            native_receipt["raw_gradients"][name],
            atol=5e-5,
            rtol=5e-4,
            msg=name,
        )
    captured = {"native": native_receipt, "custom": custom_receipt}
    crossed = replay_crossed_gradients(initial, captured, torch.device("cpu"))
    shared = shared_gradient_control(initial, torch.device("cpu"))
    return {
        "kind": "E487 fit-only CPU actual B4x256 one-batch precision preflight",
        "status": "passed",
        "fit_only": True,
        "test_rows_loaded": False,
        "seed": SEED,
        "fit_row_indices": batch["row_indices"],
        "fit_row_groups": batch["fit_row_groups"],
        "raw_feature_validation": batch["raw_feature_validation"],
        "cohort_sha256": e474.file_hash(e470.COHORT_PATH),
        "dataset_sha256": e474.DATASET_SHA256,
        "initial_state_sha256": state_digest(initial),
        "initial_state_exact_native_custom": True,
        "native_custom_forward_gradient_parity": output_state_parity,
        "forward_gradient_tolerances": {
            "output_state_atol": 2e-5,
            "output_state_rtol": 2e-4,
            "input_parameter_gradient_atol": 5e-5,
            "input_parameter_gradient_rtol": 5e-4,
        },
        "gradient_capture": {
            source: {
                "loss": receipt["loss"],
                "raw_gradient_norm": receipt["raw_gradient_norm"],
                "raw_gradient_sha256": receipt["raw_gradient_sha256"],
                "input_gradient_sha256": receipt["input_gradient_sha256"],
            }
            for source, receipt in captured.items()
        },
        "shared_gradient_controls": _jsonable_shared_control(shared),
        "crossed_replay_summary": _jsonable_crossed(crossed),
        "cpu_precision_note": "bf16_source uses the frozen CPU FP32 orthogonalizer branch; only CUDA distinguishes source BF16 from diagnostic FP32",
        "parent_source_sha256": e485.parent_source_hashes(),
        "source_sha256": source_hashes(),
        "diagnostic_source_sha256": source_hashes()["precision_replay.py"],
    }


def _jsonable_crossed(crossed: dict) -> dict:
    """Drop tensor payloads but retain every finite arithmetic measurement."""
    cells = {}
    for key, cell in crossed["crossed_replays"].items():
        cells[key] = {
            name: value for name, value in cell.items() if name not in {"deltas", "updated_state"}
        }
        cells[key]["delta_rms_by_parameter"] = {
            name: rms_by_names(cell["deltas"], [name]) for name in sorted(cell["deltas"])
        }
    return {
        "crossed_replays": cells,
        "native_custom_strict_gates": crossed["native_custom_strict_gates"],
        "bf16_fp32_delta_sensitivity": crossed["bf16_fp32_delta_sensitivity"],
        "all_crossed_finite": crossed["all_crossed_finite"],
        "expected_gate_failures_are_recorded_not_raised": crossed[
            "expected_gate_failures_are_recorded_not_raised"
        ],
    }


def _jsonable_shared_control(control: dict) -> dict:
    """Retain hashes and per-parameter update evidence without tensor payloads."""
    result = {}
    for precision, item in control.items():
        payload = {key: value for key, value in item.items() if key != "replays"}
        replay_receipts = {}
        for replay_name, cell in item["replays"].items():
            if cell is None:
                replay_receipts[replay_name] = None
                continue
            replay_receipts[replay_name] = {
                key: value for key, value in cell.items() if key not in {"deltas", "updated_state"}
            }
            replay_receipts[replay_name]["delta_rms_by_parameter"] = {
                name: rms_by_names(cell["deltas"], [name]) for name in sorted(cell["deltas"])
            }
            replay_receipts[replay_name]["state_sha256_after"] = state_digest(cell["updated_state"])
        payload["replays"] = replay_receipts
        result[precision] = payload
    return result


def source_hashes() -> dict[str, str]:
    """Hash the versioned diagnostic packet and frozen E485 dependencies."""
    paths = {
        "precision_replay.py": HERE / "precision_replay.py",
        "tests.py": HERE / "tests.py",
        "cpu_preflight.py": HERE / "cpu_preflight.py",
        "gpu_probe.py": HERE / "gpu_probe.py",
        "E487_DESIGN.md": HERE / "E487_DESIGN.md",
        "CONTROLLER_CLARIFICATIONS.md": HERE / "CONTROLLER_CLARIFICATIONS.md",
        "SPEC_CRITIC.md": HERE / "SPEC_CRITIC.md",
    }
    return {name: e474.file_hash(path) for name, path in paths.items() if path.is_file()}
