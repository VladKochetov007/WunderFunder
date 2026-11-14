"""Exercise the legacy optimizer factory on real CUDA updates with precision checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn


def main() -> None:
    """Reject the old factory and validate both fixed factory branches on CUDA."""
    source = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(source / "src"))
    from wnn import optim
    from wnn.training_dynamics import build_optimizer

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(314)
    original = optim.orthogonalize
    calls = []

    def checked(
        gradient: torch.Tensor,
        steps: int = 5,
        eps: float = 1e-7,
        *,
        strict_fp32: bool = False,
    ) -> torch.Tensor:
        """Assert actual factory-to-kernel precision propagation on every update."""
        assert strict_fp32, "factory dispatched legacy BF16 Newton-Schulz"
        assert gradient.is_cuda and gradient.dtype == torch.float32
        result = original(gradient, steps, eps, strict_fp32=strict_fp32)
        assert result.dtype == torch.float32 and torch.isfinite(result).all()
        calls.append({"shape": list(gradient.shape), "strict_fp32": strict_fp32})
        return result

    for head_scale in (1.0, 2.0):
        model = nn.Module()
        model.add_module("gru", nn.GRU(8, 16, 2, batch_first=True))
        model.add_module("head", nn.Linear(16, 2))
        model.cuda()
        args = argparse.Namespace(
            head_lr_mult=head_scale,
            zoneout=0,
            momentum=0,
            optim="muon",
            optim_switch="",
            fused_adam=False,
            muon_scope="rnn",
            precision="fp32",
            lr=0.001,
            beta2=0.999,
            wd=0.01,
        )
        optimizer = build_optimizer(model, args)
        assert all(group["strict_fp32"] for group in optimizer.param_groups), (
            "legacy factory left strict_fp32=False"
        )
        with patch.object(optim, "orthogonalize", checked):
            for _ in range(3):
                for parameter in model.parameters():
                    parameter.grad = torch.randn_like(parameter)
                optimizer.step()
        for state in optimizer.state.values():
            assert all(
                value.dtype == torch.float32
                for value in state.values()
                if isinstance(value, torch.Tensor) and value.is_floating_point()
            )
        # The adapter must not silently change explicitly reduced-precision legacy recipes.
        args.precision = "fp16"
        assert not any(group["strict_fp32"] for group in build_optimizer(model, args).param_groups)
    assert len(calls) == 24
    print(
        json.dumps(
            {
                "passed": True,
                "source": str(source),
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "factory_branches": 2,
                "cuda_optimizer_steps": 6,
                "strict_newton_schulz_calls": len(calls),
                "parameter_gradient_state_and_iterates": "FP32; internal kernel assertions active",
                "scientific_fits": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
