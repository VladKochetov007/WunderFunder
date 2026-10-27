from __future__ import annotations

import torch
from torch import nn

from wnn.galam.attention import TemporalSelfAttention
from wnn.galam.config import GALAMConfig
from wnn.galam.fusion import TwoWayFusionGate
from wnn.galam.mixer import CausalConvMixer, PerTimeBatchNorm1d
from wnn.galam.state import AttentionCache, ConvMixerState, GALAMState


class GALAM(nn.Module):
    """Configurable causal reconstruction of the documented GALAM graph."""

    def __init__(self, config: GALAMConfig) -> None:
        """Build the parallel LSTM/attention and projected-input/attention memories."""
        super().__init__()
        self.config = config
        width = config.width
        self.local_input_norm = PerTimeBatchNorm1d(config.input_features)
        self.lstm = nn.LSTM(
            config.input_features, width, num_layers=config.lstm_layers, batch_first=True
        )
        self.local_attention = nn.ModuleList(
            self._attention(config) for _ in range(config.local_attention_blocks)
        )
        self.local_norms = nn.ModuleList(nn.RMSNorm(width) for _ in self.local_attention)
        self.hidden_projection = nn.Sequential(nn.Linear(width, width), nn.GELU())
        self.cell_projection = nn.Sequential(nn.Linear(width, width), nn.GELU())
        self.global_projection = nn.Linear(config.input_features, width)
        self.global_attention = nn.ModuleList(
            self._attention(config) for _ in range(config.global_blocks)
        )
        self.global_attention_norms = nn.ModuleList(
            nn.LayerNorm(width) for _ in self.global_attention
        )
        self.global_mixers = nn.ModuleList(
            CausalConvMixer(width, config.conv_hidden_width, config.conv_kernel_size)
            for _ in range(config.global_blocks)
        )
        self.global_mixer_norms = nn.ModuleList(nn.LayerNorm(width) for _ in self.global_mixers)
        self.fusion = TwoWayFusionGate(width, config.fusion_operand_order)
        self.fusion_norm = nn.RMSNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, config.feed_forward_width),
            nn.GELU(),
            nn.Linear(config.feed_forward_width, width),
        )
        self.feed_forward_norm = nn.RMSNorm(width)
        self.readout = nn.Linear(width, config.output_features)

    @staticmethod
    def _attention(config: GALAMConfig) -> TemporalSelfAttention:
        """Create one bounded temporal attention module from declared config fields."""
        return TemporalSelfAttention(
            config.width,
            config.attention_heads,
            config.attention_history,
            config.relative_bias_width,
            config.timestamp_scale,
        )

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> GALAMState:
        """Create zero recurrent state and empty bounded caches for a fresh sequence."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        shape = (self.config.lstm_layers, batch_size, self.config.width)
        zeros = torch.zeros(shape, device=device, dtype=dtype)
        return GALAMState(
            lstm_hidden=zeros.clone(),
            lstm_cell=zeros.clone(),
            local_attention=tuple(None for _ in self.local_attention),
            global_attention=tuple(None for _ in self.global_attention),
            global_mixers=tuple(None for _ in self.global_mixers),
            steps_seen=0,
        )

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor | None = None) -> torch.Tensor:
        """Predict every step of a fresh batch; inputs are batch, time, features."""
        if x.ndim != 3 or x.shape[-1] != self.config.input_features:
            raise ValueError("input must have shape [batch, time, configured input_features]")
        state = self.reset_state(x.shape[0], x.device, x.dtype)
        predictions, _ = self.forward_chunk(x, state, timestamps)
        return predictions

    def forward_chunk(
        self, x: torch.Tensor, state: GALAMState, timestamps: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, GALAMState]:
        """Process a chunk while carrying explicit state across chunk boundaries."""
        if x.ndim != 3 or x.shape[-1] != self.config.input_features:
            raise ValueError("chunk must have shape [batch, time, configured input_features]")
        self._validate_state(x, state)
        if timestamps is not None and timestamps.shape != x.shape[:2]:
            raise ValueError("timestamps must have shape [batch, time]")
        if x.shape[1] == 0:
            return x.new_empty((x.shape[0], 0, self.config.output_features)), state
        if self.config.timestamp_source == "provided" and timestamps is None:
            raise ValueError("this config requires absolute timestamps for every input step")
        if self.config.timestamp_source == "ordinal" and timestamps is not None:
            raise ValueError("ordinal timestamp mode does not accept caller-provided timestamps")
        outputs = []
        for index in range(x.shape[1]):
            time = None if timestamps is None else timestamps[:, index]
            output, state = self.stream_step(x[:, index], state, time)
            outputs.append(output)
        return torch.stack(outputs, dim=1), state

    def stream_step(
        self, x: torch.Tensor, state: GALAMState, timestamp: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, GALAMState]:
        """Process one timestamp using an explicit state and return its prediction."""
        if x.ndim != 2 or x.shape[-1] != self.config.input_features:
            raise ValueError("one step must have shape [batch, configured input_features]")
        self._validate_state(x, state)
        time = self._resolve_timestamp(x, state, timestamp)
        normalized = self.local_input_norm(x)
        _, (hidden, cell) = self.lstm(normalized.unsqueeze(1), (state.lstm_hidden, state.lstm_cell))
        hidden_step = hidden[-1]
        cell_step = cell[-1]
        local, local_cache = self._local_step(hidden_step, time, state.local_attention)
        global_memory, global_cache, mixer_state = self._global_step(
            x, time, state.global_attention, state.global_mixers
        )
        hidden_features = self.hidden_projection(hidden_step)
        cell_features = self.cell_projection(cell_step)
        fused = self.fusion(global_memory, local, hidden_features, cell_features)
        fused = self.fusion_norm(fused)
        decoded = self.feed_forward_norm(fused + self.feed_forward(fused))
        prediction = self.readout(decoded)
        next_state = GALAMState(
            lstm_hidden=hidden,
            lstm_cell=cell,
            local_attention=local_cache,
            global_attention=global_cache,
            global_mixers=mixer_state,
            steps_seen=state.steps_seen + 1,
        )
        return prediction, next_state

    def _resolve_timestamp(
        self, x: torch.Tensor, state: GALAMState, timestamp: torch.Tensor | None
    ) -> torch.Tensor:
        """Resolve either caller-provided absolute times or explicit ordinal positions."""
        if self.config.timestamp_source == "provided":
            if timestamp is None or timestamp.numel() != x.shape[0]:
                raise ValueError("provided timestamp must contain one absolute time per batch row")
            return timestamp.reshape(x.shape[0]).to(device=x.device, dtype=torch.float32)
        if timestamp is not None:
            raise ValueError("ordinal timestamp mode does not accept an explicit timestamp")
        return torch.full((x.shape[0],), state.steps_seen, device=x.device, dtype=torch.float32)

    def _validate_state(self, x: torch.Tensor, state: GALAMState) -> None:
        """Check recurrent state shape, placement, type, and cache depth before use."""
        expected = (self.config.lstm_layers, x.shape[0], self.config.width)
        if state.lstm_hidden.shape != expected or state.lstm_cell.shape != expected:
            raise ValueError("state LSTM tensors do not match model depth, batch, and width")
        if state.lstm_hidden.device != x.device or state.lstm_hidden.dtype != x.dtype:
            raise ValueError("state device and dtype must match the input tensor")
        if len(state.local_attention) != len(self.local_attention):
            raise ValueError("state local attention depth does not match the model")
        if len(state.global_attention) != len(self.global_attention):
            raise ValueError("state global attention depth does not match the model")
        if len(state.global_mixers) != len(self.global_mixers):
            raise ValueError("state mixer depth does not match the model")

    def _local_step(
        self,
        hidden: torch.Tensor,
        timestamp: torch.Tensor,
        caches: tuple[AttentionCache | None, ...],
    ) -> tuple[torch.Tensor, tuple[AttentionCache, ...]]:
        """Apply local temporal attention residuals and PostRMSNorm operations."""
        memory = hidden
        updated = []
        for module, norm, cache in zip(self.local_attention, self.local_norms, caches, strict=True):
            attended, next_cache = module.forward_step(memory, timestamp, cache)
            memory = norm(memory + attended)
            updated.append(next_cache)
        return memory, tuple(updated)

    def _global_step(
        self,
        x: torch.Tensor,
        timestamp: torch.Tensor,
        attention_caches: tuple[AttentionCache | None, ...],
        mixer_states: tuple[ConvMixerState | None, ...],
    ) -> tuple[torch.Tensor, tuple[AttentionCache, ...], tuple[ConvMixerState, ...]]:
        """Apply projected-input temporal attention and causal residual ConvMixers."""
        memory = self.global_projection(x)
        next_attention = []
        next_mixers = []
        for attention, attention_norm, mixer, mixer_norm, cache, mixer_state in zip(
            self.global_attention,
            self.global_attention_norms,
            self.global_mixers,
            self.global_mixer_norms,
            attention_caches,
            mixer_states,
            strict=True,
        ):
            attended, next_cache = attention.forward_step(memory, timestamp, cache)
            memory = attention_norm(memory + attended)
            mixed, next_mixer_state = mixer.forward_step(memory, mixer_state)
            memory = mixer_norm(memory + mixed)
            next_attention.append(next_cache)
            next_mixers.append(next_mixer_state)
        return memory, tuple(next_attention), tuple(next_mixers)


def build_galam(config: GALAMConfig) -> GALAM:
    """Construct a GALAM model from an explicit source or adaptation config."""
    return GALAM(config)
