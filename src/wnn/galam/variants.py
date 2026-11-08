from __future__ import annotations

from typing import cast

import torch
from torch import nn

from wnn.galam.attention import TemporalSelfAttention
from wnn.galam.config import GALAMConfig
from wnn.galam.mixer import CausalConvMixer, PerTimeBatchNorm1d
from wnn.galam.model import GALAM
from wnn.galam.state import AttentionCache, ConvMixerState, GALAMState
from wnn.galam.variant_config import (
    AdaptedGALAMVariant,
    GALAMVariant,
    validate_variant_config,
    variant_channel_order,
)
from wnn.galam.variant_fusion import BranchChannelProductGate


def build_galam_variant(config: GALAMConfig, variant: GALAMVariant) -> nn.Module:
    """Build the preserved LSTM template or one explicitly labelled adaptation."""
    if variant == "schematic_lstm_full":
        return GALAM(config)
    return GALAMAdaptation(config, cast(AdaptedGALAMVariant, variant))


class GALAMAdaptation(nn.Module):
    """Causal GRU-full and LSTM-local-only adaptations, not original GALAM variants.

    The full GRU uses global memory, local attention, and projected hidden state;
    the LSTM local-only ablation uses local attention plus projected hidden and
    cell state. Both deliberately use a three-width decoder. Their parameter
    counts and fusion graphs differ from the four-width LSTM schematic template.
    """

    def __init__(self, config: GALAMConfig, variant: AdaptedGALAMVariant) -> None:
        """Build only the recurrent and branch modules used by the selected adaptation."""
        super().__init__()
        if variant not in ("schematic_gru_full", "schematic_lstm_local_only"):
            raise ValueError("GALAMAdaptation only builds the explicit GRU and local-only variants")
        self.config = config
        self.variant = variant
        validate_variant_config(config, variant)
        self._is_gru = variant == "schematic_gru_full"
        self._has_global = self._is_gru
        width = config.width
        self.local_input_norm = PerTimeBatchNorm1d(config.input_features)
        if self._is_gru:
            self.gru = nn.GRU(
                config.input_features, width, num_layers=config.lstm_layers, batch_first=True
            )
            self.hidden_projection = nn.Sequential(nn.Linear(width, width), nn.GELU())
        else:
            self.lstm = nn.LSTM(
                config.input_features, width, num_layers=config.lstm_layers, batch_first=True
            )
            self.hidden_projection = nn.Sequential(nn.Linear(width, width), nn.GELU())
            self.cell_projection = nn.Sequential(nn.Linear(width, width), nn.GELU())
        channel_order = cast(tuple[str, str, str], variant_channel_order(config, variant))
        self.local_attention = nn.ModuleList(
            self._attention(config) for _ in range(config.local_attention_blocks)
        )
        self.local_norms = nn.ModuleList(nn.RMSNorm(width) for _ in self.local_attention)
        if self._has_global:
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
        decoder_width = 3 * width
        self.fusion = BranchChannelProductGate(width, channel_order)
        self.fusion_norm = nn.RMSNorm(decoder_width)
        self.feed_forward = nn.Sequential(
            nn.Linear(decoder_width, config.feed_forward_width),
            nn.GELU(),
            nn.Linear(config.feed_forward_width, decoder_width),
        )
        self.feed_forward_norm = nn.RMSNorm(decoder_width)
        self.readout = nn.Linear(decoder_width, config.output_features)

    @staticmethod
    def _attention(config: GALAMConfig) -> TemporalSelfAttention:
        """Create bounded causal attention with the source-traced relative-time path."""
        return TemporalSelfAttention(
            config.width,
            config.attention_heads,
            config.attention_history,
            config.relative_bias_width,
            config.timestamp_scale,
        )

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> GALAMState:
        """Create fresh recurrent state without inventing a GRU cell state."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        shape = (self.config.lstm_layers, batch_size, self.config.width)
        hidden = torch.zeros(shape, device=device, dtype=dtype)
        cell = None if self._is_gru else torch.zeros_like(hidden)
        return GALAMState(
            lstm_hidden=hidden,
            lstm_cell=cell,
            local_attention=tuple(None for _ in self.local_attention),
            global_attention=tuple(None for _ in self.global_attention) if self._has_global else (),
            global_mixers=tuple(None for _ in self.global_mixers) if self._has_global else (),
            steps_seen=0,
        )

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor | None = None) -> torch.Tensor:
        """Predict a fresh sequence shaped batch, time, features."""
        if x.ndim != 3 or x.shape[-1] != self.config.input_features:
            raise ValueError("input must have shape [batch, time, configured input_features]")
        predictions, _ = self.forward_chunk(
            x, self.reset_state(x.shape[0], x.device, x.dtype), timestamps
        )
        return predictions

    def forward_chunk(
        self, x: torch.Tensor, state: GALAMState, timestamps: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, GALAMState]:
        """Process a chunk causally while carrying caches and recurrent state."""
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
        """Process one causal input row and return its explicit next state."""
        if x.ndim != 2 or x.shape[-1] != self.config.input_features:
            raise ValueError("one step must have shape [batch, configured input_features]")
        self._validate_state(x, state)
        time = self._resolve_timestamp(x, state, timestamp)
        normalized = self.local_input_norm(x)
        if self._is_gru:
            recurrent = getattr(self, "gru", None)
            if not isinstance(recurrent, nn.GRU):
                raise RuntimeError("full GRU adaptation has no GRU module")
            _, hidden = recurrent(normalized.unsqueeze(1), state.lstm_hidden)
            cell = None
        else:
            recurrent = getattr(self, "lstm", None)
            if not isinstance(recurrent, nn.LSTM) or state.lstm_cell is None:
                raise RuntimeError("local-only LSTM adaptation requires an LSTM cell state")
            _, (hidden, cell) = recurrent(
                normalized.unsqueeze(1), (state.lstm_hidden, state.lstm_cell)
            )
        hidden_step = hidden[-1]
        local, local_cache = self._local_step(hidden_step, time, state.local_attention)
        values = {"local_memory": local, "hidden_projection": self.hidden_projection(hidden_step)}
        if self._is_gru:
            global_memory, global_cache, mixer_state = self._global_step(
                x, time, state.global_attention, state.global_mixers
            )
            values["global_memory"] = global_memory
        else:
            if cell is None:
                raise RuntimeError("local-only LSTM adaptation lost its cell state")
            values["cell_projection"] = self.cell_projection(cell[-1])
            global_cache, mixer_state = (), ()
        fused = self.fusion(values)
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
        """Resolve caller-provided absolute time or explicit ordinal step position."""
        if self.config.timestamp_source == "provided":
            if timestamp is None or timestamp.numel() != x.shape[0]:
                raise ValueError("provided timestamp must contain one absolute time per batch row")
            return timestamp.reshape(x.shape[0]).to(device=x.device, dtype=torch.float32)
        if timestamp is not None:
            raise ValueError("ordinal timestamp mode does not accept an explicit timestamp")
        return torch.full((x.shape[0],), state.steps_seen, device=x.device, dtype=torch.float32)

    def _validate_state(self, x: torch.Tensor, state: GALAMState) -> None:
        """Check state depth, device, dtype, and the selected branch topology."""
        expected = (self.config.lstm_layers, x.shape[0], self.config.width)
        if state.lstm_hidden.shape != expected:
            raise ValueError("state hidden tensor does not match recurrent depth, batch, and width")
        if self._is_gru != (state.lstm_cell is None):
            raise ValueError("state cell presence does not match the selected recurrent cell")
        if state.lstm_cell is not None and state.lstm_cell.shape != expected:
            raise ValueError("state cell tensor does not match recurrent depth, batch, and width")
        if state.lstm_hidden.device != x.device or state.lstm_hidden.dtype != x.dtype:
            raise ValueError("state device and dtype must match the input tensor")
        if state.lstm_cell is not None and (
            state.lstm_cell.device != x.device or state.lstm_cell.dtype != x.dtype
        ):
            raise ValueError("state cell device and dtype must match the input tensor")
        if len(state.local_attention) != len(self.local_attention):
            raise ValueError("state local attention depth does not match the model")
        expected_global = len(self.global_attention) if self._has_global else 0
        expected_mixers = len(self.global_mixers) if self._has_global else 0
        if len(state.global_attention) != expected_global:
            raise ValueError("state global attention depth does not match the model")
        if len(state.global_mixers) != expected_mixers:
            raise ValueError("state mixer depth does not match the model")

    def _local_step(
        self,
        hidden: torch.Tensor,
        timestamp: torch.Tensor,
        caches: tuple[AttentionCache | None, ...],
    ) -> tuple[torch.Tensor, tuple[AttentionCache, ...]]:
        """Apply local temporal attention with the preserved Post-RMSNorm residual."""
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
        """Apply the full GRU variant's causal attention and Post-LayerNorm ConvMixer path."""
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
