"""Bounded normalized content memory; research implementation, unapplied."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

State = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
ENDPOINT = -5.199337482452393
ARMS = ("endpoint-content", "endpoint-position", "raw-code-content")


def walsh_table() -> Tensor:
    """Uncentered 128x96 orthogonal Walsh columns, including intercept."""
    return torch.tensor(
        [[1 - 2 * ((n & j).bit_count() % 2) for j in range(96)] for n in range(128)],
        dtype=torch.float32,
    )


class EventContentModel(nn.Module):
    """One GRU with row/event associative heads and a flat carried state."""

    def __init__(
        self,
        n_in: int,
        n_out: int = 2,
        arm: str = "endpoint-content",
        microblock: int = 64,
        superblock_tiles: int = 32,
    ) -> None:
        super().__init__()
        if (
            arm not in ARMS
            or not 1 <= microblock <= 64
            or not 1 <= superblock_tiles <= 32
            or n_in < 112
        ):
            raise ValueError("invalid arm, microblock, or missing raw features")
        self.arm, self.microblock = arm, microblock
        self.superblock_tiles = superblock_tiles
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.register_buffer("walsh", walsh_table())
        self.register_buffer("log_decay", -math.log(2) / torch.tensor([64.0, 512.0, 2.0, 16.0]))
        self.register_buffer("base_decay", self.log_decay.exp())
        self.register_buffer("norm_scale", 1 - self.base_decay)
        self.inp = nn.Linear(n_in, 96)
        self.gru = nn.GRU(96, 96, batch_first=True)
        self.query = nn.Linear(96, 32, bias=False)
        self.key = nn.Linear(96, 32, bias=False)
        self.value = nn.Linear(96, 64)
        self.read_gate = nn.Linear(96, 64)
        self.head = nn.Sequential(nn.Linear(256, 96), nn.GELU(), nn.Linear(96, n_out))

    def initial_state(self, x: Tensor) -> State:
        """External reset; only actual event gates write event heads."""
        b = x.shape[0]
        dtype = torch.float64 if self.mean.dtype == torch.float64 else torch.float32
        return (
            x.new_zeros(1, b, 96, dtype=self.mean.dtype),
            x.new_zeros(b, 4, 8, 16, dtype=dtype),
            x.new_zeros(b, 4, 8, dtype=dtype),
            torch.zeros(b, device=x.device, dtype=torch.int64),
            torch.zeros(b, device=x.device, dtype=torch.int64),
            x.new_zeros(b, 22, dtype=torch.float32),
        )

    def forward(self, x: Tensor, state: State | None = None) -> tuple[Tensor, State]:
        """Parallel exact bounded local scan; read after current insertion."""
        if x.ndim != 3 or x.shape[1] == 0:
            raise ValueError("expected nonempty B,T,D")
        state = self.initial_state(x) if state is None else state
        h, s, zmem, rowcount, writecount, prev = state
        # Source f32 predicate extracted before standardization or autocast maps.
        raw = x[..., :112].float()
        prices = raw[..., :22]
        previous = torch.cat((prev[:, None], prices[:, :-1]), dim=1)
        endpoint = raw[..., 110] == ENDPOINT
        code = (prices != previous).any(-1)
        ordinal = rowcount[:, None] + torch.arange(x.shape[1], device=x.device)
        writes = (code & (ordinal > 0)) if self.arm == "raw-code-content" else endpoint
        cumulative = writecount[:, None] + writes.to(torch.int64).cumsum(1)
        eventordinal = (cumulative - 1).clamp_min(0)
        enc = F.gelu(self.inp((x - self.mean) / self.std))
        hidden, h = self.gru(enc, h)
        # Both clocks' positions are generated in every arm.
        posrow = self.walsh[ordinal % 128]
        posevent = self.walsh[eventordinal % 128]
        content = hidden[:, :, None, :].expand(-1, -1, 4, -1)
        position = torch.stack((posrow, posrow, posevent, posevent), dim=2)
        source = position if self.arm == "endpoint-position" else content
        # Shared linear tensor sliced by head (same compute in all arms).
        dtype = torch.float64 if self.mean.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            source = source.to(dtype)
            q = (
                F.elu(
                    torch.einsum(
                        "bthd,hkd->bthk", source, self.query.weight.to(dtype).reshape(4, 8, 96)
                    ).clamp(-8, 8)
                )
                + 1
            )
            k = (
                F.elu(
                    torch.einsum(
                        "bthd,hkd->bthk", source, self.key.weight.to(dtype).reshape(4, 8, 96)
                    ).clamp(-8, 8)
                )
                + 1
            )
            v = torch.tanh(
                F.linear(hidden.to(dtype), self.value.weight.to(dtype), self.value.bias.to(dtype))
            ).reshape(*hidden.shape[:2], 4, 16)
            gate = torch.stack(
                (torch.ones_like(writes), torch.ones_like(writes), writes, writes), -1
            ).to(dtype)
            results = []
            stride = self.microblock * self.superblock_tiles
            for start in range(0, x.shape[1], stride):
                stop = min(start + stride, x.shape[1])
                retrieved, s, zmem = self._superblock(
                    q[:, start:stop],
                    k[:, start:stop],
                    v[:, start:stop],
                    gate[:, start:stop],
                    s,
                    zmem,
                )
                results.append(retrieved)
            retrieval = torch.cat(results, dim=1).flatten(2)
        read = torch.sigmoid(self.read_gate(hidden)) * retrieval.to(hidden.dtype)
        out = self.head(torch.cat((hidden, enc, read), dim=-1))
        return out, (h, s, zmem, rowcount + x.shape[1], cumulative[:, -1], prices[:, -1])

    def _superblock(
        self, q: Tensor, k: Tensor, v: Tensor, gate: Tensor, s: Tensor, zmem: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Batch local reads and strictly causal compressed-state tile carries."""
        b, t, _, _ = q.shape
        width = self.microblock
        tiles = (t + width - 1) // width
        padding = tiles * width - t

        def packed(field: Tensor) -> Tensor:
            if padding:
                field = torch.cat((field, field.new_zeros(b, padding, *field.shape[2:])), dim=1)
            return field.reshape(b, tiles, width, *field.shape[2:]).movedim(3, 1)

        qb, kb, vb, gb = (packed(field) for field in (q, k, v, gate))
        # Padded rows have zero gate in ALL heads; only real rows are returned.
        alpha = self.base_decay.to(q.dtype)[None, :, None, None]
        scale = self.norm_scale.to(q.dtype)[None, :, None, None]
        prefix = gb.cumsum(-1)
        tile_count = prefix[..., -1]
        end_count = tile_count.cumsum(-1)
        before_count = end_count - tile_count
        row_decay = alpha.pow(prefix)
        tail = alpha.pow(tile_count[..., None] - prefix) * gb * scale
        drive_s = torch.einsum("bhjt,bhjtk,bhjtv->bhjkv", tail, kb, vb)
        drive_z = torch.einsum("bhjt,bhjtk->bhjk", tail, kb)

        # Compress completed tiles before each query tile: at most32x32 states.
        carry_delta = (before_count[..., :, None] - end_count[..., None, :]).clamp_min(0)
        prior = torch.ones(tiles, tiles, device=q.device, dtype=torch.bool).tril(-1)
        carry_weight = alpha.pow(carry_delta).masked_fill(~prior, 0)
        before_decay = alpha.squeeze(-1).pow(before_count)
        s_before = before_decay[..., None, None] * s[:, :, None] + torch.einsum(
            "bhij,bhjkv->bhikv", carry_weight, drive_s
        )
        z_before = before_decay[..., None] * zmem[:, :, None] + torch.einsum(
            "bhij,bhjk->bhik", carry_weight, drive_z
        )

        local_delta = (prefix[..., :, None] - prefix[..., None, :]).clamp_min(0)
        causal = torch.ones(width, width, device=q.device, dtype=torch.bool).tril()
        local_decay = alpha[..., None].pow(local_delta).masked_fill(~causal, 0)
        weight = (
            torch.matmul(qb, kb.transpose(-1, -2))
            * local_decay
            * gb[..., None, :]
            * scale[..., None]
        )
        numerator = torch.matmul(qb, s_before) * row_decay[..., None] + torch.matmul(weight, vb)
        denominator = (qb * z_before[..., None, :]).sum(-1) * row_decay + weight.sum(-1)
        reads = numerator / torch.maximum(denominator, scale * 1e-6)[..., None]
        reads = reads.permute(0, 2, 3, 1, 4).reshape(b, tiles * width, 4, 16)[:, :t]

        final_weight = alpha.squeeze(-1).pow(end_count[..., -1, None] - end_count)
        incoming_decay = alpha.squeeze(-1).pow(end_count[..., -1, None]).squeeze(-1)
        s_out = incoming_decay[..., None, None] * s + torch.einsum(
            "bhj,bhjkv->bhkv", final_weight, drive_s
        )
        z_out = incoming_decay[..., None] * zmem + torch.einsum(
            "bhj,bhjk->bhk", final_weight, drive_z
        )
        return reads, s_out, z_out
