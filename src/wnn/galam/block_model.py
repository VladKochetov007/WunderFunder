from __future__ import annotations

import torch

from wnn.galam.block_attention import attention_chunk
from wnn.galam.block_mixer import mixer_chunk
from wnn.galam.config import GALAMConfig
from wnn.galam.state import AttentionCache, ConvMixerState, GALAMState
from wnn.galam.variants import GALAMAdaptation


class GALAMBlockGRU(GALAMAdaptation):
    """Use the original GRU-full parameters with chunk-batched execution.

    This changes floating-point operation order, not the graph or intended causal
    dependencies. Numerical, gradient and state parity must be measured on the
    receiver before a production fit. Streaming remains the inherited step path.
    """

    def __init__(self, config: GALAMConfig) -> None:
        """Reuse every original parameter name so immutable step weights can be loaded."""
        super().__init__(config, "schematic_gru_full")
        self.execution_backend = "block"

    def forward_chunk(
        self, x: torch.Tensor, state: GALAMState, timestamps: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, GALAMState]:
        """Batch ordinary chunks and preserve legacy semantics for diagnostic samples."""
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
        if self._has_step_observers():
            # Existing probes measure every recurrent hidden state and gate. They
            # would silently change meaning if attached only to the final state.
            return super().forward_chunk(x, state, timestamps)
        if timestamps is None:
            times = torch.arange(
                state.steps_seen,
                state.steps_seen + x.shape[1],
                device=x.device,
                dtype=torch.float32,
            ).expand(x.shape[0], -1)
        else:
            times = timestamps.to(device=x.device, dtype=torch.float32)
        normalized = self.local_input_norm(x)
        hidden_rows, hidden_state = self.gru(normalized, state.lstm_hidden)
        local, local_cache = self._local_chunk(hidden_rows, times, state.local_attention)
        global_memory, global_cache, mixer_states = self._global_chunk(
            x, times, state.global_attention, state.global_mixers
        )
        fused = self.fusion(
            {
                "global_memory": global_memory,
                "local_memory": local,
                "hidden_projection": self.hidden_projection(hidden_rows),
            }
        )
        fused = self.fusion_norm(fused)
        decoded = self.feed_forward_norm(fused + self.feed_forward(fused))
        return self.readout(decoded), GALAMState(
            lstm_hidden=hidden_state,
            lstm_cell=None,
            local_attention=local_cache,
            global_attention=global_cache,
            global_mixers=mixer_states,
            steps_seen=state.steps_seen + x.shape[1],
        )

    def _has_step_observers(self) -> bool:
        """Identify installed legacy diagnostics requiring the original per-step graph."""
        return any(
            module._monitor_observer is not None
            for module in (*self.local_attention, *self.global_attention)
        )

    def _local_chunk(
        self,
        hidden: torch.Tensor,
        timestamps: torch.Tensor,
        caches: tuple[AttentionCache | None, ...],
    ) -> tuple[torch.Tensor, tuple[AttentionCache, ...]]:
        """Apply the exact local residual attention stack to the recurrent trajectory."""
        memory = hidden
        updated = []
        for module, norm, cache in zip(self.local_attention, self.local_norms, caches, strict=True):
            attended, next_cache = attention_chunk(
                module,
                memory,
                timestamps,
                cache,
                shared_ordinal_times=self.config.timestamp_source == "ordinal",
            )
            memory = norm(memory + attended)
            updated.append(next_cache)
        return memory, tuple(updated)

    def _global_chunk(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        caches: tuple[AttentionCache | None, ...],
        mixer_states: tuple[ConvMixerState | None, ...],
    ) -> tuple[torch.Tensor, tuple[AttentionCache, ...], tuple[ConvMixerState, ...]]:
        """Batch the independent global attention and convolution branch."""
        memory = self.global_projection(x)
        next_attention = []
        next_mixers = []
        for attention, norm, mixer, mixer_norm, cache, mixer_state in zip(
            self.global_attention,
            self.global_attention_norms,
            self.global_mixers,
            self.global_mixer_norms,
            caches,
            mixer_states,
            strict=True,
        ):
            attended, next_cache = attention_chunk(
                attention,
                memory,
                timestamps,
                cache,
                shared_ordinal_times=self.config.timestamp_source == "ordinal",
            )
            memory = norm(memory + attended)
            mixed, next_mixer_state = mixer_chunk(mixer, memory, mixer_state)
            memory = mixer_norm(memory + mixed)
            next_attention.append(next_cache)
            next_mixers.append(next_mixer_state)
        return memory, tuple(next_attention), tuple(next_mixers)
