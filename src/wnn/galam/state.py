from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttentionCache:
    """Bounded projected key/value history for one causal attention module."""

    keys: torch.Tensor
    values: torch.Tensor
    timestamps: torch.Tensor


@dataclass(frozen=True)
class ConvMixerState:
    """Causal input tails for the two convolutions in one mixer."""

    first_inputs: torch.Tensor
    second_inputs: torch.Tensor


@dataclass(frozen=True)
class GALAMState:
    """Explicit recurrent, attention, convolution, and ordinal-position state."""

    lstm_hidden: torch.Tensor
    lstm_cell: torch.Tensor
    local_attention: tuple[AttentionCache | None, ...]
    global_attention: tuple[AttentionCache | None, ...]
    global_mixers: tuple[ConvMixerState | None, ...]
    steps_seen: int

    def detach(self) -> GALAMState:
        """Detach stored history at a caller-chosen truncated-BPTT boundary."""
        return GALAMState(
            lstm_hidden=self.lstm_hidden.detach(),
            lstm_cell=self.lstm_cell.detach(),
            local_attention=tuple(_detach_attention(cache) for cache in self.local_attention),
            global_attention=tuple(_detach_attention(cache) for cache in self.global_attention),
            global_mixers=tuple(_detach_mixer(state) for state in self.global_mixers),
            steps_seen=self.steps_seen,
        )


def _detach_attention(cache: AttentionCache | None) -> AttentionCache | None:
    """Detach one optional key/value cache without changing its bounded history."""
    if cache is None:
        return None
    return AttentionCache(cache.keys.detach(), cache.values.detach(), cache.timestamps.detach())


def _detach_mixer(state: ConvMixerState | None) -> ConvMixerState | None:
    """Detach one optional causal convolution tail."""
    if state is None:
        return None
    return ConvMixerState(state.first_inputs.detach(), state.second_inputs.detach())
