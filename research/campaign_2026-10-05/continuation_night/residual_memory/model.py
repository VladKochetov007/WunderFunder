from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

UpdateMode = Literal["delta", "additive_fixed_decay"]


class ResidualMatrixMemory(nn.Module):
    """Causal, bounded input-memory residual over an existing model's predictions.

    The external state is a float32 ``[batch, key_dim, value_dim]`` matrix.
    Every row reads the matrix before writing features from that same input
    row. The zero-initialized output projection preserves base predictions
    exactly at construction in the tested FP32 and FP16 paths.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        key_dim: int = 16,
        value_dim: int = 16,
        update_mode: UpdateMode = "delta",
        fixed_decay: float = 0.999,
        learned_decay: bool = False,
        *,
        event_index: int,
        event_threshold: float = 0.5,
    ) -> None:
        """Create an input-only write, base-output query, residual memory.

        Args:
            input_dim: Width of current-row causal features written to memory.
            output_dim: Width of the base model prediction and correction.
            key_dim: Width of the normalized associative key.
            value_dim: Width of the bounded stored value.
            update_mode: Delta erase/write or additive fixed-decay control.
            fixed_decay: Per-row memory retention for the additive control.
            learned_decay: Learn additive-control retention instead of fixing it.
            event_index: Current-row event-feature column in ``write_inputs``.
            event_threshold: Strict threshold for selecting event versus quiet writes.

        Raises:
            ValueError: If dimensions, update mode, or decay are invalid.
        """
        super().__init__()
        if min(input_dim, output_dim, key_dim, value_dim) <= 0:
            raise ValueError("memory dimensions must be positive")
        if update_mode not in ("delta", "additive_fixed_decay"):
            raise ValueError(f"unsupported memory update mode: {update_mode}")
        if not 0.0 < fixed_decay <= 1.0:
            raise ValueError("fixed_decay must lie in (0, 1]")
        if learned_decay and update_mode != "additive_fixed_decay":
            raise ValueError("learned_decay applies only to the additive control")
        if learned_decay and fixed_decay >= 1.0:
            raise ValueError("learned fixed_decay must lie strictly between zero and one")
        if not 0 <= event_index < input_dim:
            raise ValueError("event_index must select a column in write_inputs")
        if not 0.0 <= event_threshold <= 1.0:
            raise ValueError("event_threshold must lie in [0, 1]")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.update_mode = update_mode
        self.event_index = event_index
        self.event_threshold = event_threshold
        self.query = nn.Linear(output_dim, key_dim, bias=False)
        self.write_key = nn.Linear(input_dim, key_dim, bias=False)
        self.write_value = nn.Linear(input_dim, value_dim)
        self.event_write_rate = nn.Linear(input_dim, 1)
        self.quiet_write_rate = nn.Linear(input_dim, 1)
        self.residual_out = nn.Linear(value_dim, output_dim)
        nn.init.zeros_(self.residual_out.weight)
        nn.init.zeros_(self.residual_out.bias)
        if learned_decay:
            initial_logit = torch.logit(torch.tensor(fixed_decay, dtype=torch.float32))
            self.decay_logit = nn.Parameter(initial_logit)
            self.register_buffer("fixed_decay", torch.empty(0), persistent=False)
        else:
            self.register_parameter("decay_logit", None)
            self.register_buffer("fixed_decay", torch.tensor(fixed_decay, dtype=torch.float32))

    def detach_state(self, state: Tensor | None) -> Tensor | None:
        """Detach carried memory at an explicit training credit boundary."""
        return None if state is None else state.detach()

    def forward(
        self,
        base_predictions: Tensor,
        write_inputs: Tensor,
        state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Read prior input memory, correct base outputs, then write current inputs.

        Args:
            base_predictions: Causal base-model predictions, shaped ``[B,T,O]``.
            write_inputs: Causal input features, shaped ``[B,T,I]``; never labels.
            state: Optional prior chunk memory, shaped ``[B,K,V]`` float32.

        Returns:
            A tuple containing corrected predictions ``[B,T,O]`` and the
            persistent matrix state ``[B,K,V]``. Call ``detach_state`` at the
            training plan's recurrent credit boundary and pass ``None`` when
            beginning an independent sequence.

        Raises:
            ValueError: If inputs or a carried state have incompatible shapes.
        """
        self._validate_inputs(base_predictions, write_inputs, state)
        batch_size, steps, _ = write_inputs.shape
        memory = (
            torch.zeros(
                batch_size,
                self.key_dim,
                self.value_dim,
                device=write_inputs.device,
                dtype=torch.float32,
            )
            if state is None
            else state
        )
        query = F.normalize(self.query(base_predictions.float()).float(), dim=-1, eps=1e-6)
        keys = F.normalize(self.write_key(write_inputs.float()).float(), dim=-1, eps=1e-6)
        values = torch.tanh(self.write_value(write_inputs.float()).float())
        event_rates = torch.sigmoid(self.event_write_rate(write_inputs.float()).float())
        quiet_rates = torch.sigmoid(self.quiet_write_rate(write_inputs.float()).float())
        event_mask = write_inputs[..., self.event_index] > self.event_threshold
        write_rates = torch.where(event_mask[..., None], event_rates, quiet_rates)
        outputs: list[Tensor] = []

        for row in range(steps):
            row_query = query[:, row]
            retrieved = torch.einsum("bk,bkv->bv", row_query, memory)
            correction = self.residual_out(retrieved.to(self.residual_out.weight.dtype))
            correction = correction.to(base_predictions.dtype)
            outputs.append(base_predictions[:, row] + correction)
            key = keys[:, row]
            value = values[:, row]
            rate = write_rates[:, row]
            if self.update_mode == "delta":
                prior_value = torch.einsum("bk,bkv->bv", key, memory)
                memory = memory + key[..., None] * (rate * (value - prior_value))[:, None, :]
            else:
                if self.decay_logit is None:
                    decay = self.fixed_decay
                else:
                    decay = torch.sigmoid(self.decay_logit)
                memory = decay * memory + key[..., None] * (rate * value)[:, None, :]

        return torch.stack(outputs, dim=1), memory

    def _validate_inputs(
        self,
        base_predictions: Tensor,
        write_inputs: Tensor,
        state: Tensor | None,
    ) -> None:
        if (
            base_predictions.ndim != 3
            or write_inputs.ndim != 3
            or base_predictions.shape[:2] != write_inputs.shape[:2]
            or base_predictions.shape[1] == 0
            or base_predictions.shape[-1] != self.output_dim
            or write_inputs.shape[-1] != self.input_dim
        ):
            raise ValueError("expected matching nonempty base/input sequences")
        if base_predictions.device != write_inputs.device:
            raise ValueError("base predictions and input features must share a device")
        if state is not None and (
            state.shape != (write_inputs.shape[0], self.key_dim, self.value_dim)
            or state.dtype != torch.float32
            or state.device != write_inputs.device
        ):
            raise ValueError("carried matrix has incompatible shape, dtype or device")
