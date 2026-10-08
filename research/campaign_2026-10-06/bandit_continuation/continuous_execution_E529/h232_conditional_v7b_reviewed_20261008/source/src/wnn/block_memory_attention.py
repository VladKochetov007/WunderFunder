from __future__ import annotations

import attrs
import torch
import torch.nn.functional as F
from torch import nn


@attrs.frozen
class BlockState:
    """One layer's completed-block memory and unfinished causal block."""

    memory: torch.Tensor
    inputs: torch.Tensor
    output_sum: torch.Tensor

    def detach(self) -> BlockState:
        """Detach at an explicit training truncation boundary."""
        return BlockState(self.memory.detach(), self.inputs.detach(), self.output_sum.detach())


class BlockMemoryLayer(nn.Module):
    """Attend within a causal block and recurrently summarize completed blocks."""

    def __init__(self, width: int, heads: int, block_rows: int) -> None:
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads or block_rows <= 0:
            raise ValueError("positive width divisible by heads and positive block required")
        self.width, self.heads, self.block_rows = width, heads, block_rows
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.project = nn.Linear(width, width)
        self.feedforward = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )
        self.memory_update = nn.GRUCell(width, width)

    def forward(
        self, values: torch.Tensor, state: BlockState | None = None
    ) -> tuple[torch.Tensor, BlockState]:
        """Carry partial blocks so arbitrary chunk boundaries preserve predictions."""
        batch, rows, width = values.shape
        if rows == 0 or width != self.width:
            raise ValueError("nonempty sequence with configured width required")
        if state is None:
            state = BlockState(
                values.new_zeros(batch, width),
                values.new_empty(batch, 0, width),
                torch.zeros(batch, width, device=values.device, dtype=torch.float32),
            )
        if state.memory.shape != (batch, width) or state.inputs.shape[1] >= self.block_rows:
            raise ValueError("state batch/width or partial-block length is invalid")
        outputs = []
        offset = 0
        while offset < rows:
            previous = state.inputs.shape[1]
            count = min(self.block_rows - previous, rows - offset)
            current = values[:, offset : offset + count]
            context = torch.cat(
                [state.memory[:, None].to(values.dtype), state.inputs, current], dim=1
            )
            q, k, v = self.qkv(self.norm(context)).chunk(3, dim=-1)

            def split_heads(tensor: torch.Tensor) -> torch.Tensor:
                return tensor.reshape(batch, -1, self.heads, width // self.heads).transpose(1, 2)

            q, k, v = split_heads(q[:, -count:]), split_heads(k), split_heads(v)
            # True entries are visible. The memory token represents only earlier
            # completed blocks; unfinished-block tokens obey the row-wise mask.
            keys = torch.arange(context.shape[1], device=values.device)
            query_end = 1 + previous + torch.arange(count, device=values.device)
            visible = keys[None, :] <= query_end[:, None]
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=visible)
            attended = attended.transpose(1, 2).reshape(batch, count, width)
            output = current + self.project(attended)
            output = output + self.feedforward(output)
            outputs.append(output)
            output_sum = state.output_sum + output.float().sum(dim=1)
            if previous + count == self.block_rows:
                memory = self.memory_update(
                    (output_sum / self.block_rows).to(state.memory.dtype), state.memory
                )
                state = BlockState(
                    memory, values.new_empty(batch, 0, width), torch.zeros_like(output_sum)
                )
            else:
                state = BlockState(
                    state.memory, torch.cat([state.inputs, current], dim=1), output_sum
                )
            offset += count
        return torch.cat(outputs, dim=1), state


class BlockMemoryAttention(nn.Module):
    """Independent H218 teacher prototype; no production fit or serving claim."""

    def __init__(
        self,
        n_in: int = 211,
        width: int = 256,
        layers: int = 4,
        heads: int = 8,
        block_rows: int = 256,
        n_out: int = 2,
    ) -> None:
        super().__init__()
        if n_in <= 0 or layers <= 0 or n_out <= 0:
            raise ValueError("positive input, layer and output dimensions required")
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.project = nn.Linear(n_in, width)
        self.layers = nn.ModuleList(
            BlockMemoryLayer(width, heads, block_rows) for _ in range(layers)
        )
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, n_out))

    def forward(
        self, values: torch.Tensor, states: tuple[BlockState, ...] | None = None
    ) -> tuple[torch.Tensor, tuple[BlockState, ...]]:
        """Predict causally; pass None to reset every layer at a sequence boundary."""
        if states is not None and len(states) != len(self.layers):
            raise ValueError("one state per layer required")
        hidden = self.project((values - self.mean) / self.std)
        next_states = []
        for index, layer in enumerate(self.layers):
            hidden, state = layer(hidden, None if states is None else states[index])
            next_states.append(state)
        return self.head(hidden), tuple(next_states)
