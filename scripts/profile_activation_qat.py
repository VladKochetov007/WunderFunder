"""Profile compiled activation-aware GRU forward/backward and save a receipt."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn

from wnn.activation_qat import ActivationQATModel
from wnn.nn import RNNModel
from wnn.training_qat import configure_qat, qat_weight_replacements


class _TrainingObjective(nn.Module):
    """Produce a scalar that exercises sequence outputs and final recurrent state."""

    def __init__(self, model: RNNModel) -> None:
        super().__init__()
        self.network = ActivationQATModel(model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        predictions, state = self.network(inputs)
        return predictions.square().mean() + state.square().mean() * 0.01


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=False, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_receipt(path: Path, receipt: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=1000)
    parser.add_argument("--features", type=int, default=112)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--measured-steps", type=int, default=3)
    parser.add_argument(
        "--weight-preset",
        choices=("recurrent", "recurrent_head", "recurrent_w4", "input_w4_recurrent_w8"),
        default="recurrent_w4",
    )
    return parser


def main() -> int:
    """Run one bounded compiled profile, preserving backend failures as receipts."""
    args = _parser().parse_args()
    receipt: dict[str, object] = {
        "schema": "activation_aware_qat_profile_v1",
        "started_utc": datetime.now(UTC).isoformat(),
        "source_revision": _git_revision(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "shape": {
            "batch": args.batch_size,
            "chunk": args.chunk,
            "features": args.features,
            "hidden": args.hidden,
            "layers": args.layers,
        },
        "weight_preset": args.weight_preset,
        "activation_scope": "recurrent_input_and_previous_hidden_only",
        "backend_status": "running",
        "quality_status": "not_evaluated",
    }
    _write_receipt(args.receipt, receipt)
    stage = "device_check"
    try:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device(args.device)
        torch.manual_seed(23117)
        model = RNNModel(args.features, hidden=args.hidden, layers=args.layers).to(device)
        configure_qat(model, args.weight_preset)
        # Populate detached represented-matrix caches before Dynamo traces the
        # forward path. Each traced call still creates fresh STE autograd nodes.
        qat_weight_replacements(model)
        objective = _TrainingObjective(model).to(device)
        compiled = torch.compile(objective, fullgraph=True)
        inputs = torch.randn(
            args.batch_size,
            args.chunk,
            args.features,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        def update() -> torch.Tensor:
            objective.zero_grad(set_to_none=True)
            inputs.grad = None
            loss = compiled(inputs)
            loss.backward()
            return loss

        stage = "compile_and_first_update"
        started = time.perf_counter()
        first_loss = update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compile_and_first_update_seconds = time.perf_counter() - started
        compile_peak_allocated_bytes = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        )
        compile_peak_reserved_bytes = (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        )

        for _ in range(2):
            update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        stage = "measured_forward_backward"
        durations = []
        latest_loss = first_loss
        for _ in range(args.measured_steps):
            started = time.perf_counter()
            latest_loss = update()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - started)

        receipt.update(
            {
                "backend_status": "profile_complete",
                "device": str(device),
                "device_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else platform.processor()
                ),
                "compile_and_first_update_seconds": compile_and_first_update_seconds,
                "compile_peak_allocated_bytes": compile_peak_allocated_bytes,
                "compile_peak_reserved_bytes": compile_peak_reserved_bytes,
                "measured_step_seconds": durations,
                "median_forward_backward_seconds": sorted(durations)[len(durations) // 2],
                "last_loss": float(latest_loss.detach().cpu()),
                "finite_gradients": all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                    for parameter in objective.parameters()
                ),
                "peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                ),
                "peak_reserved_bytes": (
                    torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
                ),
            }
        )
        return_code = 0
    except Exception as error:
        receipt.update(
            {
                "backend_status": "backend_failure",
                "failure_stage": stage,
                "failure_type": type(error).__name__,
                "failure_message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        return_code = 1
    receipt["finished_utc"] = datetime.now(UTC).isoformat()
    _write_receipt(args.receipt, receipt)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
