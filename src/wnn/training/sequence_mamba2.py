"""Canonical sequence adapter for the source-pinned genuine CUDA Mamba-2 SSD."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from wnn.training.sequence_mamba2_backend import (
    Mamba2LayerState,
    Mamba2State,
    _load_mamba2_backend,
    _Mamba2Block,
)
from wnn.training.sequence_model_helpers import (
    HintProjector,
    validate_sequence_fp32_environment,
    validate_sequence_input_shape,
)


class Mamba2SequenceAdapter(nn.Module):
    """Pre-norm Mamba-2 residual stack with exact carried convolution/SSD state.

    The frozen backend wrapper is adapted from historical blob 7f60c071, using
    official mamba-ssm 2.2.4 commit 95d8aba8. It is CUDA-only. No CPU stand-in
    or Mamba-1 approximation is substituted when that backend is unavailable.
    """

    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        mixer_type, scan = _load_mamba2_backend()
        hidden = int(config["hidden"])
        self.register_buffer("mean", torch.zeros(112))
        self.register_buffer("std", torch.ones(112))
        self.project = nn.Linear(112, hidden)
        self.layers = nn.ModuleList(
            _Mamba2Block(
                hidden,
                int(config["ssm_state_size"]),
                int(config["ssm_expand"]),
                int(config["ssm_head_dim"]),
                int(config["ssm_conv_width"]),
                int(config["ssm_chunk_size"]),
                mixer_type,
                scan,
            )
            for _ in range(int(config["layers"]))
        )
        self.final_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 2)
        self.hint_projector = HintProjector(hidden, int(config["hint_dim"]), "learned")

    def forward(
        self, values: Tensor, state: Mamba2State | None = None
    ) -> tuple[Tensor, Tensor, Mamba2State]:
        """Predict the current CUDA chunk and return differentiable carried state."""
        validate_sequence_input_shape(values)
        validate_sequence_fp32_environment(self, values, "fp32")
        if not values.is_cuda or torch.version.cuda is None:
            raise RuntimeError("genuine Mamba-2 requires an NVIDIA CUDA tensor/backend")
        if state is not None and (
            not isinstance(state, Mamba2State) or len(state.layers) != len(self.layers)
        ):
            raise ValueError("Mamba-2 requires one compatible state per layer")
        hidden = self.project((values - self.mean) / self.std)
        next_states: list[Mamba2LayerState] = []
        for index, layer in enumerate(self.layers):
            hidden, layer_state = layer(hidden, None if state is None else state.layers[index])
            next_states.append(layer_state)
        latent = self.final_norm(hidden)
        return self.head(latent), self.hint_projector(latent), Mamba2State(tuple(next_states))
