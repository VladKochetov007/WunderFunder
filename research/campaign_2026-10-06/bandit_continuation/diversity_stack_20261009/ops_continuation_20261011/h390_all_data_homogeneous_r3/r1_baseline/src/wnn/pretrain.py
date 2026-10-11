"""Input-only next-row pretraining of the main encoder + GRU (research flag --pretrain-next-epochs)."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import nn

from wnn.extractor import N_NEXT


def pretrain_next_rows(model: nn.Module, batches: Callable[[], Iterable], build_inputs: Callable,
                       device: torch.device, epochs: int, chunk: int, lr: float = 1e-3, clip: float = 1.0,
                       max_batches: int = 0) -> list[float]:
    """Train encoder + GRU (and a throwaway decoder) to predict the next row's standardised raw channels.

    ``batches()`` yields (features, kept_length_or_None) of fitting sequences; labels are never read.
    The model's own mean/std standardise the inputs. The readout head is restored to its pre-pretraining
    values on return, so only encoder and GRU weights carry over. Truncated BPTT over ``chunk`` rows with the
    GRU state carried and detached between chunks; the state starts at zero for every sequence.
    Returns the mean loss per epoch.
    """
    head_state = {k: v.clone() for k, v in model.head.state_dict().items()}
    decoder = nn.Linear(model.gru.hidden_size, N_NEXT).to(device)
    params = [p for n, p in model.named_parameters() if not n.startswith("head.")] + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    model.train()
    history: list[float] = []
    for _ in range(epochs):
        total, count = 0.0, 0
        for b, (x, kept) in enumerate(batches()):
            if max_batches and b >= max_batches:
                break
            xn = (build_inputs(x.to(device)) - model.mean) / model.std
            kept = None if kept is None else kept.to(device)
            state = None
            for s in range(0, xn.shape[1] - 1, chunk):
                e = min(s + chunk, xn.shape[1] - 1)  # inputs rows s..e-1 predict rows s+1..e
                o, state = model.gru(model.encode(xn[:, s:e]), state)
                state = state.detach()
                err = ((decoder(o) - xn[:, s + 1:e + 1, :N_NEXT].clamp(-6, 6)) ** 2).mean(-1)
                if kept is not None:
                    rows = torch.arange(s, e, device=device)[None]
                    err = err * (rows < kept[:, None] - 1).float()
                loss = err.mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, clip)
                opt.step()
                total, count = total + float(loss.detach()), count + 1
        history.append(total / max(count, 1))
    model.head.load_state_dict(head_state)
    model.zero_grad(set_to_none=True)
    return history
