"""Engineering-only eager/compiled diagnostic using synthetic rows and Muon."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from pathlib import Path

import torch
from torch import Tensor, nn

SOURCE_FILES = (
    "src/wnn/training/sequence_precision.py",
    "src/wnn/training/sequence_models.py",
    "src/wnn/training/sequence_compile.py",
    "src/wnn/training/sequence_attention.py",
    "src/wnn/training/sequence_attention_backends.py",
    "src/wnn/training/sequence_attention_capabilities.py",
    "src/wnn/training/sequence_attention_masks.py",
    "src/wnn/training/sequence_objectives.py",
    "src/wnn/training/sequence_loop.py",
    "src/wnn/training/sequence_optimizer.py",
    "src/wnn/training/sequence_fit_utils.py",
    "src/wnn/training/sequence_fit_artifacts.py",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _metrics(pairs: Iterable[tuple[Tensor, Tensor]]) -> dict[str, float | int]:
    """Compute max error, relative L2, and cosine without concatenating gradients."""
    device: torch.device | None = None
    difference_sq: Tensor | None = None
    reference_sq: Tensor | None = None
    candidate_sq: Tensor | None = None
    dot: Tensor | None = None
    max_absolute: Tensor | None = None
    count = 0
    for reference, candidate in pairs:
        if reference.shape != candidate.shape:
            raise ValueError("comparison tensors have different shapes")
        if reference.numel() == 0:
            continue
        left = reference.detach().float()
        right = candidate.detach().float()
        if device is None:
            device = left.device
            difference_sq = torch.zeros((), device=device)
            reference_sq = torch.zeros((), device=device)
            candidate_sq = torch.zeros((), device=device)
            dot = torch.zeros((), device=device)
            max_absolute = torch.zeros((), device=device)
        assert all(
            x is not None
            for x in (difference_sq, reference_sq, candidate_sq, dot, max_absolute)
        )
        difference = right - left
        difference_sq += difference.square().sum()
        reference_sq += left.square().sum()
        candidate_sq += right.square().sum()
        dot += (left * right).sum()
        max_absolute = torch.maximum(max_absolute, difference.abs().max())
        count += difference.numel()
    if not count or any(
        x is None
        for x in (difference_sq, reference_sq, candidate_sq, dot, max_absolute)
    ):
        raise ValueError("comparison requires at least one tensor element")
    return {
        "max_absolute_error": float(max_absolute.item()),
        "relative_l2_error": float(
            (difference_sq.sqrt() / reference_sq.sqrt().clamp_min(1e-12)).item()
        ),
        "cosine_similarity": float(
            (dot / (reference_sq.sqrt() * candidate_sq.sqrt()).clamp_min(1e-12)).item()
        ),
        "elements": count,
    }


def _finite_tensors(tensors: Iterable[Tensor], name: str) -> int:
    count = 0
    flags: list[Tensor] = []
    for tensor in tensors:
        flags.append(torch.isfinite(tensor).all())
        count += tensor.numel()
    if not count:
        raise ValueError(f"{name} is empty")
    if not bool(torch.stack(flags).all()):
        raise FloatingPointError(f"{name} contains non-finite values")
    return count


def _parameters_fp32(model: nn.Module) -> bool:
    return all(p.dtype == torch.float32 for p in model.parameters()) and all(
        not b.is_floating_point() or b.dtype == torch.float32 for b in model.buffers()
    )


def _cache_fp32_bounded(state: object, context_rows: int) -> bool:
    from wnn.training.sequence_attention import TransformerState

    return isinstance(state, TransformerState) and all(
        tensor.dtype == torch.float32 and tensor.shape[-2] <= context_rows - 1
        for tensor in (*state.keys, *state.values)
    )


def _model_outputs(
    model: nn.Module,
    values: Tensor,
    mode: str,
    precision: Callable[[str, str], AbstractContextManager[object]],
) -> tuple[Tensor, Tensor, object]:
    with precision(mode, values.device.type):
        prediction, hints, state = model(values)
    expected = torch.bfloat16 if mode in ("bf16", "bf16_bf16x9") else torch.float32
    if prediction.dtype != expected or (hints.numel() and hints.dtype != expected):
        raise TypeError(f"model outputs must be {expected} for {mode}")
    return prediction.float(), hints.float(), state


def _source_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in SOURCE_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"pinned source is missing {name}")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _run(args: argparse.Namespace) -> dict[str, object]:
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    root = args.source_root.resolve(strict=True)
    sys.path.insert(0, str(root / "src"))
    from wnn.training.sequence_config import load_config
    from wnn.training.sequence_fit_utils import make_optimizer
    from wnn.training.sequence_models import build_sequence_model
    from wnn.training.sequence_objectives import (
        objective_numerators,
        scaled_objective,
        window_denominators,
    )
    from wnn.training.sequence_precision import (
        backward_context,
        configure_precision_mode,
        forward_context,
        validate_process_environment,
    )

    config = load_config(args.config.resolve(strict=True))
    if config["optimizer"] != "muon":
        raise ValueError("R4 requires the production Muon optimizer")
    if config["model"]["architecture"] != "transformer":
        raise ValueError("R4 only supports the production Transformer path")
    if config["model"]["checkpoint_blocks"] is not False:
        raise ValueError("R4 requires checkpoint_blocks=false, the production setting")
    mode = str(config["precision_mode"])
    validate_process_environment(mode)
    configure_precision_mode(mode)
    if not torch.cuda.is_available():
        raise RuntimeError("R4 requires CUDA")
    torch.cuda.set_device(0)
    if mode in ("bf16", "bf16_bf16x9") and not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected CUDA device does not support BF16")
    device = torch.device("cuda", 0)
    if torch.get_default_dtype() != torch.float32:
        raise TypeError("R4 requires the canonical FP32 default dtype")

    batch_size = int(config["batch_size"])
    rows = int(config["chunk_rows"])
    if not 1 <= batch_size <= 8 or not 2 <= rows <= 2048:
        raise ValueError(
            "R4 bounds one-device synthetic work to batch<=8 and rows<=2048"
        )
    model_base = dict(config["model"])
    model_base["precision_mode"] = mode
    model_base["checkpoint_blocks"] = False
    model_base["compile_blocks"] = False
    backend = str(model_base["attention_backend"])
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    eager = build_sequence_model(dict(model_base)).to(device)
    compiled_model_config = dict(model_base)
    compiled_model_config["compile_blocks"] = True
    compiled = build_sequence_model(compiled_model_config).to(device)
    compiled.load_state_dict(eager.state_dict(), strict=True)
    if not _parameters_fp32(eager) or not _parameters_fp32(compiled):
        raise TypeError("all compiled and eager master parameters must remain FP32")

    environment_emulates_fp32 = os.environ.get("CUBLAS_EMULATE_SINGLE_PRECISION") == "1"
    reference_mode = "fp32_bf16x9" if environment_emulates_fp32 else "fp32"
    validate_process_environment(reference_mode)
    reference_config = dict(model_base)
    reference_config["precision_mode"] = reference_mode
    reference = build_sequence_model(reference_config).to(device)
    reference.load_state_dict(eager.state_dict(), strict=True)
    if not _parameters_fp32(reference):
        raise TypeError("FP32 reference must retain FP32 parameters and buffers")

    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    inputs = torch.randn((batch_size, rows, 112), generator=generator).to(device)
    target = (torch.randn((batch_size, rows, 2), generator=generator) * 0.3).to(device)
    teacher = target + torch.randn(target.shape, generator=generator).to(device) * 0.1
    hint_dim = int(model_base["hint_dim"])
    hint_target = torch.randn((batch_size, rows, hint_dim), generator=generator).to(
        device
    )
    mask = torch.ones((batch_size, rows), dtype=torch.bool, device=device)
    denominators = window_denominators(target, mask, hint_target)
    if (
        not bool(torch.isfinite(denominators).all())
        or denominators[0] <= 0
        or (float(config["hint_weight"]) > 0 and denominators[1] <= 0)
    ):
        raise ValueError("synthetic objective denominators must be positive and finite")

    models = {"eager": eager, "compiled": compiled, "fp32_reference": reference}
    outputs: dict[str, tuple[Tensor, Tensor]] = {}
    losses: dict[str, Tensor] = {}
    gradients: dict[str, list[Tensor]] = {}
    for name, model in models.items():
        selected_mode = reference_mode if name == "fp32_reference" else mode
        model.train()
        prediction, hints, state = _model_outputs(
            model, inputs, selected_mode, forward_context
        )
        if not _cache_fp32_bounded(state, int(model_base["context_rows"])):
            raise TypeError(f"{name} rolling cache must be FP32 and context-bounded")
        outputs[name] = (prediction.detach(), hints.detach())
        numerators = objective_numerators(
            prediction, hints, target, teacher, hint_target, mask
        )
        loss = scaled_objective(
            numerators,
            denominators,
            world_size=1,
            kd_weight=float(config["kd_weight"]),
            hint_weight=float(config["hint_weight"]),
        )
        if loss.dtype != torch.float32 or not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"{name} objective must be finite FP32")
        with backward_context(selected_mode, device.type):
            loss.backward()
        model_grads = [p.grad for p in model.parameters()]
        if any(grad is None or grad.dtype != torch.float32 for grad in model_grads):
            raise TypeError(f"{name} must produce FP32 gradients for every parameter")
        gradients[name] = [grad.detach() for grad in model_grads if grad is not None]
        _finite_tensors(gradients[name], f"{name} gradients")
        losses[name] = loss.detach()

    for name, model in models.items():
        altered = inputs.clone()
        cut = max(1, rows * 3 // 4)
        altered[:, cut:] += 100.0
        selected_mode = reference_mode if name == "fp32_reference" else mode
        with torch.no_grad():
            original_out, _, _ = _model_outputs(
                model, inputs, selected_mode, forward_context
            )
            altered_out, _, _ = _model_outputs(
                model, altered, selected_mode, forward_context
            )
        if not torch.equal(original_out[:, :cut], altered_out[:, :cut]):
            raise AssertionError(f"{name} future perturbation changed the exact prefix")

    comparisons: dict[str, dict[str, object]] = {
        name: {
            "outputs_vs_fp32_reference": _metrics(
                [
                    (outputs["fp32_reference"][0], outputs[name][0]),
                    (outputs["fp32_reference"][1], outputs[name][1]),
                ]
            ),
            "gradients_vs_fp32_reference": _metrics(
                zip(gradients["fp32_reference"], gradients[name], strict=True)
            ),
            "loss_fp32": float(losses[name].item()),
            "loss_abs_error_vs_reference": abs(
                float(losses[name].item() - losses["fp32_reference"].item())
            ),
        }
        for name in ("eager", "compiled")
    }
    comparisons["compiled"]["outputs_vs_eager"] = _metrics(
        [
            (outputs["eager"][0], outputs["compiled"][0]),
            (outputs["eager"][1], outputs["compiled"][1]),
        ]
    )
    comparisons["compiled"]["gradients_vs_eager"] = _metrics(
        zip(gradients["eager"], gradients["compiled"], strict=True)
    )
    comparisons["compiled"]["loss_abs_error_vs_eager"] = abs(
        float(losses["compiled"].item() - losses["eager"].item())
    )
    norms: dict[str, Tensor] = {}
    for name in ("fp32_reference", "eager", "compiled"):
        model = models[name]
        norms[name] = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["grad_clip"])
        )
        if not bool(torch.isfinite(norms[name])):
            raise FloatingPointError(f"{name} preclip gradient norm is non-finite")
        if name in comparisons:
            comparisons[name]["preclip_norm"] = float(norms[name].item())
            comparisons[name]["preclip_norm_relative_error_vs_reference"] = abs(
                float(norms[name].item() - norms["fp32_reference"].item())
            ) / max(float(norms["fp32_reference"].item()), 1e-12)
    comparisons["compiled"]["preclip_norm_relative_error_vs_eager"] = abs(
        float(norms["compiled"].item() - norms["eager"].item())
    ) / max(float(norms["eager"].item()), 1e-12)

    optimizers: dict[
        str, tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]
    ] = {}
    for name in ("eager", "compiled"):
        opt_config = dict(config)
        opt_config["model"] = dict(model_base)
        opt_config["model"]["compile_blocks"] = name == "compiled"
        schedule_updates = max(
            1000, int(2.0 / max(1e-6, 1.0 - float(config["warmup_fraction"]))) + 2
        )
        optimizers[name] = make_optimizer(
            models[name], opt_config, total_updates=schedule_updates
        )
        optimizers[name][0].step()
        optimizers[name][1].step()
        if not _finite_tensors(
            (p.detach() for p in models[name].parameters()),
            f"{name} updated parameters",
        ):
            raise FloatingPointError(
                f"{name} optimizer step produced non-finite parameters"
            )
        optimizer_state = [
            tensor
            for state in optimizers[name][0].state.values()
            for tensor in state.values()
            if isinstance(tensor, Tensor) and tensor.is_floating_point()
        ]
        if optimizer_state:
            if any(tensor.dtype != torch.float32 for tensor in optimizer_state):
                raise TypeError(f"{name} Muon floating state must remain FP32")
            _finite_tensors(optimizer_state, f"{name} optimizer state")
        reference_state = reference.state_dict()
        delta_sq = torch.zeros((), dtype=torch.float32, device=device)
        delta_max = torch.zeros((), dtype=torch.float32, device=device)
        delta_flags: list[Tensor] = []
        for key, parameter in models[name].state_dict().items():
            if not parameter.is_floating_point():
                continue
            delta = parameter.detach() - reference_state[key].detach()
            delta_float = delta.float()
            delta_flags.append(torch.isfinite(delta_float).all())
            delta_sq += delta_float.square().sum()
            delta_max = torch.maximum(delta_max, delta_float.abs().max())
        if not bool(torch.stack(delta_flags).all()):
            raise FloatingPointError(f"{name} Muon update deltas are non-finite")
        comparisons[name]["muon_update_delta_l2"] = float(delta_sq.sqrt().item())
        comparisons[name]["muon_update_delta_max_abs"] = float(delta_max.item())
        comparisons[name]["muon_optimizer_state_fp32_finite"] = True

    eager_state = eager.state_dict()
    compiled_state = compiled.state_dict()
    reference_state = reference.state_dict()
    update_pairs = (
        (
            eager_state[key].detach() - reference_state[key].detach(),
            compiled_state[key].detach() - reference_state[key].detach(),
        )
        for key in eager_state
        if eager_state[key].is_floating_point()
    )
    comparisons["compiled"]["muon_update_delta_vs_eager"] = _metrics(update_pairs)

    torch.cuda.synchronize(device)
    return {
        "schema": "pr32.compile-path-check.r4",
        "engineering_only": True,
        "quality_claim": False,
        "source_root": str(root),
        "source_sha256": _source_hashes(root),
        "config_sha256": hashlib.sha256(
            args.config.resolve(strict=True).read_bytes()
        ).hexdigest(),
        "precision_mode": mode,
        "raw_model_output_dtype": "bfloat16"
        if mode in ("bf16", "bf16_bf16x9")
        else "float32",
        "objective_dtype": "float32",
        "fp32_reference_mode": reference_mode,
        "fp32_reference_kind": "process-global cuBLAS-emulated FP32; not native IEEE FP32"
        if reference_mode == "fp32_bf16x9"
        else "native strict FP32",
        "attention_backend": backend,
        "checkpoint_blocks": False,
        "compile_blocks": {"eager": False, "compiled": True},
        "batch_size": batch_size,
        "rows": rows,
        "seed": args.seed,
        "scheduler_horizon_updates": schedule_updates,
        "synthetic_standard_normal_inputs": True,
        "exact_future_prefix": True,
        "comparisons": comparisons,
    }


def main() -> None:
    args = _arguments()
    receipt = _run(args)
    rendered = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)


if __name__ == "__main__":
    main()
