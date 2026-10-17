from __future__ import annotations

import attrs
import torch
import torch.nn.functional as F
from torch import Tensor, nn


@attrs.frozen
class CausalTransformerState:
    """Per-sequence attention caches at an explicit absolute-position offset."""

    keys: tuple[Tensor, ...]
    values: tuple[Tensor, ...]
    position_offset: int

    def detach(self) -> CausalTransformerState:
        """Detach cached history at a caller-owned TBPTT boundary."""
        return CausalTransformerState(
            tuple(key.detach() for key in self.keys),
            tuple(value.detach() for value in self.values),
            self.position_offset,
        )


class CausalTransformerLayer(nn.Module):
    """Pre-norm causal self-attention with a bounded, sliding KV cache."""

    def __init__(
        self, width: int, heads: int, window: int, ffn_mult: int, inv_freq: Tensor
    ) -> None:
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.window = window
        self.register_buffer("inv_freq", inv_freq.clone(), persistent=False)
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attn_out = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, ffn_mult * width),
            nn.GELU(),
            nn.Linear(ffn_mult * width, width),
        )

    def _rotate(self, tensor: Tensor, positions: Tensor) -> Tensor:
        batch, _, rows, _ = tensor.shape
        pairs = tensor.reshape(batch, self.heads, rows, self.head_width // 2, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        angles = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]
        cosine = angles.cos().to(tensor.dtype)[None, None, :, :]
        sine = angles.sin().to(tensor.dtype)[None, None, :, :]
        rotated = torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
        )
        return rotated.flatten(-2)

    def forward(
        self,
        inputs: Tensor,
        previous_key: Tensor | None,
        previous_value: Tensor | None,
        position_offset: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, rows, _ = inputs.shape
        q, key, value = self.qkv(self.norm1(inputs)).chunk(3, dim=-1)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.reshape(batch, rows, self.heads, self.head_width).transpose(1, 2)

        q, key, value = split_heads(q), split_heads(key), split_heads(value)
        query_positions = torch.arange(
            position_offset, position_offset + rows, device=inputs.device
        )
        key = self._rotate(key, query_positions)
        q = self._rotate(q, query_positions)

        cached_rows = 0 if previous_key is None else previous_key.shape[-2]
        if previous_key is not None:
            if previous_value is None or previous_value.shape != previous_key.shape:
                raise ValueError("key and value cache shapes must match")
            if previous_key.device != key.device or previous_key.dtype != key.dtype:
                raise ValueError("cached keys must match the current device and dtype")
            if previous_value.device != value.device or previous_value.dtype != value.dtype:
                raise ValueError("cached values must match the current device and dtype")
            all_keys = torch.cat((previous_key, key), dim=-2)
            all_values = torch.cat((previous_value, value), dim=-2)
        else:
            all_keys, all_values = key, value

        first_key_position = position_offset - cached_rows
        key_positions = torch.arange(
            first_key_position,
            position_offset + rows,
            device=inputs.device,
        )
        # Boolean SDPA masks use True for visible entries. Each query sees at
        # most the current row and the preceding window - 1 absolute positions.
        visible = (key_positions[None, :] <= query_positions[:, None]) & (
            key_positions[None, :] > query_positions[:, None] - self.window
        )
        attended = F.scaled_dot_product_attention(
            q, all_keys, all_values, attn_mask=visible[None, None, :, :]
        )
        attended = attended.transpose(1, 2).contiguous().reshape(batch, rows, self.width)
        output = inputs + self.attn_out(attended)
        output = output + self.feedforward(self.norm2(output))

        retained = self.window - 1
        if retained == 0:
            next_key = all_keys[:, :, :0, :]
            next_value = all_values[:, :, :0, :]
        else:
            next_key = all_keys[:, :, -retained:, :]
            next_value = all_values[:, :, -retained:, :]
        return output, next_key, next_value


class CausalTransformer(nn.Module):
    """Chunk-invariant causal Transformer for ordered sequence prediction.

    Inputs have shape ``[batch, time, n_in]``. ``None`` starts a fresh
    sequence; otherwise pass the returned state to continue it. Each attention
    layer retains only the previous ``window - 1`` keys and values. Long input
    calls are internally split into bounded query blocks, so attention does not
    form a full-sequence quadratic matrix. Current-call caches remain attached
    to autograd; callers detach the returned state between TBPTT updates.
    """

    _MAX_QUERY_ROWS = 128

    def __init__(
        self,
        n_in: int,
        hidden: int = 256,
        layers: int = 4,
        heads: int = 8,
        window: int = 256,
        ffn_mult: int = 4,
        n_out: int = 2,
    ) -> None:
        super().__init__()
        if min(n_in, hidden, layers, heads, window, ffn_mult, n_out) <= 0:
            raise ValueError("all model dimensions must be positive")
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        head_width = hidden // heads
        if head_width % 2:
            raise ValueError("RoPE requires an even per-head width")
        self.window = window
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.project = nn.Linear(n_in, hidden)
        inv_freq = 1.0 / (
            10000.0
            ** (torch.arange(0, head_width, 2, dtype=torch.float32) / head_width)
        )
        self.layers = nn.ModuleList(
            CausalTransformerLayer(hidden, heads, window, ffn_mult, inv_freq)
            for _ in range(layers)
        )
        # Keep head.2.weight compatible with existing output-width inspection.
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_out))

    def _validate_state(self, state: CausalTransformerState, batch: int) -> None:
        if state.position_offset < 0:
            raise ValueError("state position_offset must be nonnegative")
        if len(state.keys) != len(self.layers) or len(state.values) != len(self.layers):
            raise ValueError("state must contain one key and value cache per layer")
        max_history = self.window - 1
        expected_history = min(state.position_offset, max_history)
        for key, value in zip(state.keys, state.values, strict=True):
            if key.ndim != 4 or value.shape != key.shape:
                raise ValueError("each state cache must have matching [batch, heads, time, width] tensors")
            if (
                key.shape[0] != batch
                or key.shape[1] != self.layers[0].heads
                or key.shape[3] != self.layers[0].head_width
                or key.shape[2] != expected_history
            ):
                raise ValueError("state cache shape or history length is invalid")

    def forward(
        self, x: Tensor, state: CausalTransformerState | None = None
    ) -> tuple[Tensor, CausalTransformerState]:
        """Return predictions and the updated per-sequence sliding-window state."""
        if x.ndim != 3 or x.shape[0] <= 0 or x.shape[1] <= 0 or x.shape[-1] != self.mean.numel():
            raise ValueError("x must have shape [positive_batch, positive_time, n_in]")
        if state is not None:
            self._validate_state(state, x.shape[0])

        hidden = self.project((x - self.mean) / self.std)
        base_offset = 0 if state is None else state.position_offset
        keys: tuple[Tensor, ...] = () if state is None else state.keys
        values: tuple[Tensor, ...] = () if state is None else state.values
        hidden_blocks = []

        # Bound the query axis even when callers pass an entire 20k-row
        # sequence. This keeps FP32 SDPA score workspaces bounded for batch 32.
        query_rows = self._MAX_QUERY_ROWS
        for start in range(0, x.shape[1], query_rows):
            end = min(start + query_rows, x.shape[1])
            block = hidden[:, start:end]
            next_keys = []
            next_values = []
            for index, layer in enumerate(self.layers):
                previous_key = keys[index] if keys else None
                previous_value = values[index] if values else None
                block, next_key, next_value = layer(
                    block,
                    previous_key,
                    previous_value,
                    base_offset + start,
                )
                next_keys.append(next_key)
                next_values.append(next_value)
            keys = tuple(next_keys)
            values = tuple(next_values)
            hidden_blocks.append(block)

        encoded = torch.cat(hidden_blocks, dim=1)
        prediction = self.head(encoded)
        next_state = CausalTransformerState(keys, values, base_offset + x.shape[1])
        return prediction, next_state
