"""Small next-step GRU feature extractor, trained separately and frozen inside the main model's inputs."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from wnn.data import N_FEATURES
from wnn.features import N_EVENT_FEATURES
from wnn.splits import file_sha256

N_NEXT = 104  # raw price/volume channels predicted one row ahead
BASE_WIDTH = N_FEATURES + N_EVENT_FEATURES  # raw + event inputs the extractor reads
PROJECTED = 32


class NextStepExtractor(nn.Module):
    """Linear encoder -> GRU(128) -> 32-d projection (the exported feature) -> next-row decoder."""

    def __init__(self, n_in: int = BASE_WIDTH, hidden: int = 128, proj: int = PROJECTED, n_next: int = N_NEXT):
        super().__init__()
        self.dims = {"n_in": n_in, "hidden": hidden, "proj": proj, "n_next": n_next}
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.inp = nn.Linear(n_in, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, proj)
        self.out = nn.Linear(proj, n_next)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        """Return (features, next-row prediction, state); state None means a sequence start."""
        z = F.gelu(self.inp((x - self.mean) / self.std))
        o, h = self.gru(z, h)
        feat = self.proj(o)
        return feat, self.out(F.gelu(feat)), h


def next_row_loss(pred: torch.Tensor, standardized: torch.Tensor, kept: torch.Tensor | None = None) -> torch.Tensor:
    """MSE of row-t predictions against the standardised channels of row t+1 (last row has no target)."""
    target = standardized[:, 1:, :pred.shape[-1]].clamp(-6, 6)
    err = ((pred[:, :-1] - target) ** 2).mean(-1)
    if kept is None:
        return err.mean()
    ok = (torch.arange(err.shape[1], device=err.device)[None] < (kept[:, None] - 1)).float()
    return (err * ok).sum() / ok.sum().clamp_min(1.0)


def load_extractor(path: str | Path, device: torch.device) -> tuple[NextStepExtractor, str]:
    """Load, freeze and return the extractor with its file hash."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = NextStepExtractor(**checkpoint["dims"])
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    return model.to(device), file_sha256(Path(path))


def fit_extractor(model: NextStepExtractor, batches, device: torch.device, epochs: int, chunk: int = 2000,
                  lr: float = 1e-3, clip: float = 1.0) -> list[float]:
    """Next-row MSE fit on base inputs (B, T, n_in) of fitting sequences; labels are never read.

    ``batches`` is a callable returning a fresh iterable of tensors per epoch. Truncated BPTT with the state
    carried and detached between chunks and reset to None for every batch of whole sequences.
    """
    opt = torch.optim.AdamW(list(model.parameters()), lr=lr, weight_decay=0.0)
    history = []
    model.train()
    for _ in range(epochs):
        total, count = 0.0, 0
        for base in batches():
            base = base.to(device).float()
            state = None
            for s in range(0, base.shape[1] - 1, chunk):
                e = min(s + chunk, base.shape[1] - 1)  # rows s..e-1 predict rows s+1..e
                _, pred, state = model(base[:, s:e], state)
                state = state.detach()
                standardized = (base[:, s:e + 1] - model.mean) / model.std
                loss = next_row_loss(torch.cat([pred, pred[:, -1:]], dim=1), standardized)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                total, count = total + float(loss.detach()), count + 1
        history.append(total / max(count, 1))
    model.eval().requires_grad_(False)
    return history


@torch.no_grad()
def extractor_features(model: NextStepExtractor, base_inputs: torch.Tensor) -> torch.Tensor:
    """Projected hidden states for whole sequences; the state starts at zero for each sequence (reset)."""
    feat, _, _ = model(base_inputs.float(), None)
    return feat
