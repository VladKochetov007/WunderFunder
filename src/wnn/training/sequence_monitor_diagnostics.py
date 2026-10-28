from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from wnn.training.sequence_monitor_gates import collect_recurrent_gates


class DiagnosticProbe:
    """Collect bounded sampled activation, gradient, gate, attention, and prediction stats."""

    def __init__(self, model: nn.Module) -> None:
        """Bind temporary hooks to the existing model without changing its forward math."""
        self.model = model.module if isinstance(model, DistributedDataParallel) else model
        self._sum: dict[str, Tensor] = {}
        self._square: dict[str, Tensor] = {}
        self._fraction: dict[str, Tensor] = {}
        self._maximum: dict[str, Tensor] = {}
        self._count: dict[str, int] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._observers: list[nn.Module] = []

    def __enter__(self) -> DiagnosticProbe:
        """Install ephemeral hooks for this optimizer window only."""
        for name, module in self.model.named_modules():
            path = _galam_path(name)
            if isinstance(module, (nn.LSTM, nn.GRU)) and path in ("lstm", "gru"):
                self._handles.append(module.register_forward_hook(self._recurrent_hook(path)))
                continue
            metric = _activation_metric(path)
            if metric is not None:
                self._handles.append(module.register_forward_hook(self._activation_hook(metric)))
            if path == "fusion.gates.2":
                self._handles.append(module.register_forward_hook(self._fusion_gate_hook()))
            if module.__class__.__name__ == "TemporalSelfAttention":
                previous = getattr(module, "_monitor_observer", None)
                if previous is not None:
                    continue
                module._monitor_observer = self._attention_observer(_attention_tag(path))
                self._observers.append(module)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Remove every hook and observer even when training raises."""
        for handle in self._handles:
            handle.remove()
        for module in self._observers:
            module._monitor_observer = None

    def observe_prediction(self, prediction: Tensor, mask: Tensor) -> None:
        """Accumulate prediction distribution over the active training objective mask."""
        selected = prediction.detach().float()[mask.detach().bool()]
        if selected.numel():
            self._add_distribution("train/prediction", selected, clipped_at=2.0)

    def observe_attention(
        self,
        tag: str,
        weights: Tensor,
        bias: Tensor,
        query_times: Tensor,
        key_times: Tensor,
    ) -> None:
        """Summarize actual softmax weights, timestamp lags, and applied bias values."""
        detached_weights = weights.detach().float()
        entropy = -(detached_weights * detached_weights.clamp_min(1e-12).log()).sum(dim=-1)
        lag = (query_times.detach().unsqueeze(-1) - key_times.detach().unsqueeze(-2)).abs()
        weighted_lag = (detached_weights * lag.unsqueeze(1)).sum(dim=-1)
        self._add_mean(f"attention/{tag}/entropy_mean", entropy)
        self._add_mean(f"attention/{tag}/lag_abs_mean", weighted_lag)
        self._add_distribution(f"relative_bias/{tag}", bias.detach().float())
        self._add_maximum(f"relative_bias/{tag}/abs_max", bias.detach().float().abs())

    def metrics(self) -> dict[str, int | float]:
        """Synchronize once and return finite sampled metrics, omitting absent branches."""
        keys = sorted(set(self._sum) | set(self._square) | set(self._fraction) | set(self._maximum))
        tensors = [
            value
            for key in keys
            for mapping in (self._sum, self._square, self._fraction, self._maximum)
            if (value := mapping.get(key)) is not None
        ]
        values = torch.stack(tensors).detach().cpu().tolist() if tensors else []
        decoded: dict[tuple[str, str], float] = {}
        cursor = 0
        for key in keys:
            for name, mapping in (
                ("sum", self._sum),
                ("square", self._square),
                ("fraction", self._fraction),
                ("maximum", self._maximum),
            ):
                if key in mapping:
                    decoded[(key, name)] = float(values[cursor])
                    cursor += 1
        result: dict[str, int | float] = {}
        for key in self._sum:
            result[key] = decoded[(key, "sum")] / max(1, self._count[key])
        for key in self._square:
            result[key] = (decoded[(key, "square")] / max(1, self._count[key])) ** 0.5
        for key in self._fraction:
            result[key] = decoded[(key, "fraction")] / max(1, self._count[key])
        result.update({key: value for (key, name), value in decoded.items() if name == "maximum"})
        self._add_distribution_metrics(result, "train/prediction", remove_rms=True)
        self._add_distribution_metrics(result, "relative_bias/local", remove_rms=False)
        self._add_distribution_metrics(result, "relative_bias/global", remove_rms=False)
        return result

    def _activation_hook(self, key: str) -> Callable[..., None]:
        def _observe(module: nn.Module, inputs: tuple[object, ...], output: object) -> None:
            tensor = _first_tensor(output)
            if tensor is None:
                return
            self._add_rms(key, tensor)
            if tensor.requires_grad:
                tensor.register_hook(lambda gradient: self._add_rms(_gradient_key(key), gradient))

        return _observe

    def _attention_observer(self, tag: str) -> Callable[[Tensor, Tensor, Tensor, Tensor], None]:
        def _observe(weights: Tensor, bias: Tensor, query_times: Tensor, key_times: Tensor) -> None:
            self.observe_attention(tag, weights, bias, query_times, key_times)

        return _observe

    def _recurrent_hook(self, name: str) -> Callable[..., None]:
        def _observe(module: nn.Module, inputs: tuple[object, ...], output: object) -> None:
            if not inputs or not isinstance(inputs[0], Tensor) or not isinstance(output, tuple):
                return
            sequence_output = output[0]
            state = output[1]
            if isinstance(sequence_output, Tensor):
                key = "local/recurrent_output_rms"
                self._add_rms(key, sequence_output)
                if sequence_output.requires_grad:
                    sequence_output.register_hook(
                        lambda gradient: self._add_rms(_gradient_key(key), gradient)
                    )
            if isinstance(state, tuple) and len(state) == 2:
                hidden, cell = state
                self._record_state("local/hidden_state_rms", hidden)
                self._record_state("local/cell_state_rms", cell)
            elif isinstance(state, Tensor):
                self._record_state("local/hidden_state_rms", state)
            if getattr(module, "dropout", 0.0) == 0.0:
                collect_recurrent_gates(
                    module,
                    inputs[0],
                    inputs[1] if len(inputs) > 1 else None,
                    self._record_gate,
                )

        return _observe

    def _fusion_gate_hook(self) -> Callable[..., None]:
        def _observe(module: nn.Module, inputs: tuple[object, ...], output: object) -> None:
            if not isinstance(output, Tensor):
                return
            gates = output.detach().float().sigmoid()
            self._add_mean("fusion/gate/mean", gates)
            saturated = (gates < 0.05) | (gates > 0.95)
            self._add_fraction("fusion/gate/saturated_fraction", saturated)

        return _observe

    def _record_state(self, key: str, value: object) -> None:
        if isinstance(value, Tensor):
            self._add_rms(key, value)
            if value.requires_grad:
                value.register_hook(lambda gradient: self._add_rms(_gradient_key(key), gradient))

    def _record_gate(self, name: str, value: Tensor, sigmoid: bool, layer: int) -> None:
        prefix = f"local/gate/{name}/layer_{layer}"
        self._add_mean(f"{prefix}/mean", value)
        saturated = (value < 0.05) | (value > 0.95) if sigmoid else value.abs() > 0.95
        self._add_fraction(f"{prefix}/saturated_fraction", saturated)

    def _add_mean(self, key: str, value: Tensor) -> None:
        flat = value.detach().float()
        self._sum[key] = self._sum.get(key, flat.new_zeros(())) + flat.sum()
        self._count[key] = self._count.get(key, 0) + flat.numel()

    def _add_rms(self, key: str, value: Tensor) -> None:
        flat = value.detach().float()
        self._square[key] = self._square.get(key, flat.new_zeros(())) + flat.square().sum()
        self._count[key] = self._count.get(key, 0) + flat.numel()

    def _add_fraction(self, key: str, value: Tensor) -> None:
        flat = value.detach().float()
        self._fraction[key] = self._fraction.get(key, flat.new_zeros(())) + flat.sum()
        self._count[key] = self._count.get(key, 0) + flat.numel()

    def _add_maximum(self, key: str, value: Tensor) -> None:
        maximum = value.detach().float().max()
        self._maximum[key] = (
            torch.maximum(self._maximum[key], maximum) if key in self._maximum else maximum
        )

    def _add_distribution(
        self, prefix: str, value: Tensor, clipped_at: float | None = None
    ) -> None:
        flat = value.detach().float()
        self._add_mean(f"{prefix}/mean", flat)
        self._add_rms(f"{prefix}/rms", flat)
        if clipped_at is not None:
            self._add_fraction(f"{prefix}/clipped_fraction", flat.abs() > clipped_at)

    def _add_distribution_metrics(
        self, result: dict[str, int | float], prefix: str, *, remove_rms: bool
    ) -> None:
        mean = result.get(f"{prefix}/mean")
        rms = result.get(f"{prefix}/rms")
        if mean is not None and rms is not None:
            result[f"{prefix}/std"] = max(0.0, rms * rms - mean * mean) ** 0.5
            if remove_rms:
                del result[f"{prefix}/rms"]


def _first_tensor(value: object) -> Tensor | None:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, tuple):
        return next((tensor for item in value if (tensor := _first_tensor(item)) is not None), None)
    return None


def _galam_path(name: str) -> str:
    if ".galam." in name:
        return name.split(".galam.", 1)[1]
    if name.startswith("galam."):
        return name.removeprefix("galam.")
    return name.removeprefix("module.")


def _attention_tag(path: str) -> str:
    return "local" if path.startswith("local_attention.") else "global"


def _activation_metric(path: str) -> str | None:
    if path.startswith("local_norms."):
        return f"local/activation_rms/{path.rsplit('.', 1)[-1]}"
    if path.startswith("global_attention_norms."):
        return f"global/attention_activation_rms/{path.rsplit('.', 1)[-1]}"
    if path.startswith("global_mixer_norms."):
        return f"global/mixer_activation_rms/{path.rsplit('.', 1)[-1]}"
    if path == "hidden_projection":
        return "local/hidden_projection_activation_rms"
    if path == "cell_projection":
        return "local/cell_projection_activation_rms"
    if path == "fusion":
        return "fusion/output_activation_rms"
    if path in ("fusion_norm", "feed_forward", "feed_forward_norm"):
        return f"fusion/{path}_activation_rms"
    if path.startswith(("local_attention.", "global_attention.")) and path.endswith(".output"):
        return f"attention/{_attention_tag(path)}/output_activation_rms"
    if ".relative_bias" in path and path.endswith("relative_bias"):
        return f"relative_bias/{_attention_tag(path)}/activation_rms"
    return None


def _gradient_key(key: str) -> str:
    return key.replace("activation_rms", "gradient_rms").replace("state_rms", "state_gradient_rms")
