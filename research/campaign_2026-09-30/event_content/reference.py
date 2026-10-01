"""Independent literal recurrence: no parallel actor helpers called."""

from __future__ import annotations

import torch
from model import EventContentModel, State
from torch import Tensor
from torch.nn import functional as F


def sequential(
    model: EventContentModel, x: Tensor, state: State | None = None
) -> tuple[Tensor, State]:
    """Reference streaming uses individual GRU steps and direct S/Z updates."""
    h, s, zm, row, count, prev = model.initial_state(x) if state is None else state
    outputs = []
    dtype = torch.float64 if model.mean.dtype == torch.float64 else torch.float32
    for t in range(x.shape[1]):
        raw = x[:, t].float()
        code = (raw[:, :22] != prev).any(-1)
        mark = raw[:, 110] == -5.199337482452393
        write = (code & (row > 0)) if model.arm == "raw-code-content" else mark
        count = count + write.long()
        enc = F.gelu(model.inp((x[:, t] - model.mean) / model.std))
        hidden, h = model.gru(enc[:, None], h)
        hidden = hidden[:, 0]
        values = []
        next_s, next_z = [], []
        with torch.autocast(device_type=x.device.type, enabled=False):
            for j, _half in enumerate((64, 512, 2, 16)):
                ordinal = row if j < 2 else (count - 1).clamp_min(0)
                if model.arm == "endpoint-position":
                    # Construct the basis directly, independently of actor table.
                    ord_cpu = (ordinal % 128).tolist()
                    source = x.new_tensor(
                        [[1 - 2 * ((n & c).bit_count() % 2) for c in range(96)] for n in ord_cpu],
                        dtype=dtype,
                    )
                else:
                    source = hidden.to(dtype)
                q = (
                    F.elu(
                        F.linear(source, model.query.weight[j * 8 : (j + 1) * 8].to(dtype)).clamp(
                            -8, 8
                        )
                    )
                    + 1
                )
                k = (
                    F.elu(
                        F.linear(source, model.key.weight[j * 8 : (j + 1) * 8].to(dtype)).clamp(
                            -8, 8
                        )
                    )
                    + 1
                )
                v = torch.tanh(
                    F.linear(
                        hidden.to(dtype),
                        model.value.weight[j * 16 : (j + 1) * 16].to(dtype),
                        model.value.bias[j * 16 : (j + 1) * 16].to(dtype),
                    )
                )
                w = torch.ones_like(write, dtype=dtype) if j < 2 else write.to(dtype)
                # Explicit serialized base and scale define the recurrent state grammar.
                base = model.base_decay[j].to(dtype)
                scale = model.norm_scale[j].to(dtype)
                a = torch.where(w > 0, base, torch.ones_like(w))
                sj = (
                    a[:, None, None] * s[:, j]
                    + (w * scale)[:, None, None] * k[:, :, None] * v[:, None, :]
                )
                zj = a[:, None] * zm[:, j] + (w * scale)[:, None] * k
                r = (q[:, :, None] * sj).sum(1) / (
                    torch.maximum((q * zj).sum(1), scale * 1e-6)[:, None]
                )
                values.append(r)
                next_s.append(sj)
                next_z.append(zj)
            s, zm = torch.stack(next_s, 1), torch.stack(next_z, 1)
        read = torch.sigmoid(model.read_gate(hidden)) * torch.cat(values, -1).to(hidden.dtype)
        outputs.append(model.head(torch.cat((hidden, enc, read), -1)))
        row = row + 1
        prev = raw[:, :22]
    return torch.stack(outputs, 1), (h, s, zm, row, count, prev)
