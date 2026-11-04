"""Single scheduler interface for TL3's independent Muon and AdamW OneCycles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from wnn.training.tl3_optimizer import TL3MuonAdamW

_FORMAT = "wnn.tl3-muon-onecycle.v1"


class TL3MuonOneCycle:
    """Advance the historical momentum and beta1 cycles through one scheduler."""

    def __init__(
        self,
        optimizer: TL3MuonAdamW,
        *,
        lr: float,
        muon_lr_mult: float,
        total_steps: int,
        warmup_frac: float,
    ) -> None:
        """Create one momentum-cycling schedule per underlying optimizer."""
        self.optimizer = optimizer
        common = {
            "total_steps": total_steps,
            "pct_start": warmup_frac,
            "final_div_factor": 100,
            "cycle_momentum": True,
        }
        self.muon_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer.muon_optimizer, max_lr=lr * muon_lr_mult, **common
        )
        self.adamw_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer.adamw_optimizer, max_lr=lr, **common
        )

    @property
    def last_epoch(self) -> int:
        """Return the shared completed-step index."""
        if self.muon_scheduler.last_epoch != self.adamw_scheduler.last_epoch:
            raise RuntimeError("TL3 Muon and AdamW schedule steps diverged")
        return self.muon_scheduler.last_epoch

    def step(self) -> None:
        """Advance Muon LR/momentum and AdamW LR/beta1 together."""
        self.muon_scheduler.step()
        self.adamw_scheduler.step()
        _ = self.last_epoch

    def get_last_lr(self) -> list[float]:
        """Return rates in the trainer-facing Muon-then-AdamW group order."""
        return self.muon_scheduler.get_last_lr() + self.adamw_scheduler.get_last_lr()

    def state_dict(self) -> dict[str, Any]:
        """Serialize both native schedule states."""
        return {
            "format": _FORMAT,
            "muon": self.muon_scheduler.state_dict(),
            "adamw": self.adamw_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore the two schedules and reject divergent continuation points."""
        if state_dict.get("format") != _FORMAT or not {"muon", "adamw"} <= state_dict.keys():
            raise ValueError("not a TL3 Muon/AdamW OneCycle scheduler state")
        self.muon_scheduler.load_state_dict(state_dict["muon"])
        self.adamw_scheduler.load_state_dict(state_dict["adamw"])
        _ = self.last_epoch
