from __future__ import annotations

import argparse
from pathlib import Path

# Historical checkpoint records and experimental backend/batch states have
# heterogeneous layouts; Any is retained only at those compatibility boundaries.
from typing import Any

import numpy as np
import torch
from attrs import define
from torch.utils.data import DataLoader

from wnn.final_submit_manifest import FinalFitMembership
from wnn.outcomes import ReadoutMoments
from wnn.resources import DutyCycle
from wnn.splits import CampaignSplit


@define
class OptionsState:
    """Options state owned by one training run; tensors are never copied here."""

    ap: argparse.ArgumentParser | None = None
    args: argparse.Namespace | None = None
    controlled_loss: bool = False
    continuation: bool | None = None
    detach_rows: int | None = None
    device: torch.device | None = None
    distillation_requested: bool | None = None
    duty: DutyCycle | None = None
    ema_update_decay: float | None = None
    frontier_contract_sha256: str | None = None
    frontier_stage: dict[str, Any] | None = None
    init_checkpoint: dict[str, Any] | None = None
    out_dir: Path | None = None
    thin_high: float | None = None
    thin_low: float | None = None
    update_rows: int | None = None


@define
class DataState:
    """Data state owned by one training run; tensors are never copied here."""

    campaign_split: CampaignSplit | None = None
    distill_cache_validation: dict[str, Any] | None = None
    eligible_valid_count: int | None = None
    extra_valid: np.ndarray | None = None
    final_fit_membership: FinalFitMembership | None = None
    fit_subset: dict[str, Any] | None = None
    frontier_cache_binding: dict[str, Any] | None = None
    frontier_contract: dict[str, Any] | None = None
    frontier_order: dict[str, Any] | None = None
    high: float | None = None
    inner_crossfit: dict[str, Any] | None = None
    low: float | None = None
    n_valid: int | None = None
    prefetch: int | None = None
    prop_cache: dict[str, Path] | None = None
    prop_model: torch.nn.Module | None = None
    provenance: dict[str, Any] | None = None
    quote_cache_provenance: dict[str, Any] | None = None
    sampler: torch.utils.data.Sampler[int] | None = None
    saved_training: dict[str, Any] | None = None
    shifts: list[int] | None = None
    spread_provenance: dict[str, Any] | None = None
    teachers: list[tuple[str, float]] | None = None
    thin: tuple[float, float, float] | tuple[float, float, float, str] | None = None
    train_ds: torch.utils.data.Dataset | None = None
    train_eval_loader: DataLoader | None = None
    train_eval_members_sha256: str | None = None
    train_eval_row_groups: list[int] | None = None
    train_idx: np.ndarray | None = None
    train_loader: DataLoader | None = None
    train_split: str | None = None
    valid_idx: np.ndarray | None = None
    valid_loader: DataLoader | None = None
    workers: int | None = None


@define
class NetworkState:
    """Network state owned by one training run; tensors are never copied here."""

    family_at: int | None = None
    init: dict[str, Any] | None = None
    initializer_teacher: torch.nn.Module | None = None
    ks: list[int] | None = None
    mask_at: int | None = None
    model: torch.nn.Module | None = None
    moments: ReadoutMoments | None = None
    n_in: int | None = None
    n_out: int | None = None
    next_at: int | None = None
    quote_at: int | None = None
    sign_at: int | None = None
    struct_at: int | None = None
    tier_at: int | None = None
    total: int | None = None
