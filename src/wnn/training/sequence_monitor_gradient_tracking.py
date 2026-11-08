from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn


class OutputGradientTracker:
    """Track sampled output tensors that do or do not receive a backward gradient."""

    def __init__(self) -> None:
        self.expected: dict[str, int] = {}
        self.observed: dict[str, int] = {}
        self.explicit_missing: dict[str, int] = {}

    def observe(self, key: str, gradient: Tensor | None) -> None:
        """Record whether one registered output received a backward gradient."""
        self.observed[key] = self.observed.get(key, 0) + 1
        if gradient is None:
            self.explicit_missing[key] = self.explicit_missing.get(key, 0) + 1

    def register(self, key: str, tensor: Tensor, hook: Callable[[Tensor | None], None]) -> None:
        """Register one detached reporting hook on a differentiable output tensor."""
        self.expected[key] = self.expected.get(key, 0) + 1
        tensor.register_hook(hook)

    def metrics(self) -> dict[str, int]:
        """Return per-output missing counts and their aggregate without zero-filling."""
        origins = set(self.expected) | set(self.explicit_missing)
        result = {
            f"diagnostics/missing_output_gradient/{origin}_count": max(
                0, self.expected.get(origin, 0) - self.observed.get(origin, 0)
            )
            + self.explicit_missing.get(origin, 0)
            for origin in sorted(origins)
        }
        result["diagnostics/missing_output_gradient_count"] = sum(result.values())
        return result


def parameter_gradient_status(model: nn.Module) -> tuple[int, int]:
    """Count trainable parameters and missing grads before optimizer zeroing."""
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return len(trainable), sum(parameter.grad is None for parameter in trainable)
