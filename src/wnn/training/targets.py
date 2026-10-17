from __future__ import annotations

import argparse

import torch

from wnn.data import N_FEATURES
from wnn.features import FEATURE_NAMES, N_EVENT_FEATURES
from wnn.outcomes import sequence_class
from wnn.training.environment import TrainingEnvironment


def next_change_labels(
    y: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Training-only labels from the future of the target stream: log1p(rows until the next change
    of D = t0 - t1) and the sign of that change. Rows without a later change are masked out."""
    d = y[..., 0] - y[..., 1]
    b, t = d.shape
    idx = torch.arange(t, device=y.device).expand(b, t)
    changed = torch.zeros_like(d, dtype=torch.bool)
    changed[:, 1:] = d[:, 1:] != d[:, :-1]
    marker = torch.where(changed, idx, torch.full_like(idx, t))
    nxt = torch.flip(torch.cummin(torch.flip(marker, [1]), 1).values, [1])
    nxt = torch.cat([nxt[:, 1:], torch.full_like(nxt[:, :1], t)], 1)
    has = nxt < t
    j = nxt.clamp(max=t - 1)
    jump = torch.gather(d, 1, j) - torch.gather(d, 1, (j - 1).clamp(min=0))
    rows_to = torch.log1p((nxt - idx).float())
    return (rows_to, (jump > 0).float(), valid & has)


def next_label_change(y: torch.Tensor) -> torch.Tensor:
    """(B, T) index of the first row strictly after t where the target pair changes (T if none): the label is
    identical on (t, next - 1], so a causal estimate made at row next - 1 is a valid target for row t."""
    b, t, _ = y.shape
    idx = torch.arange(t, device=y.device).expand(b, t)
    changed = torch.zeros((b, t), dtype=torch.bool, device=y.device)
    changed[:, 1:] = (y[:, 1:] != y[:, :-1]).any(-1)
    marker = torch.where(changed, idx, torch.full_like(idx, t))
    at_or_after = torch.flip(torch.cummin(torch.flip(marker, [1]), 1).values, [1])
    return torch.cat([at_or_after[:, 1:], torch.full_like(at_or_after[:, :1], t)], 1)


def kth_change_labels(
    y: torch.Tensor, valid: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """log1p(rows until the k-th later change of D = t0 - t1) and sign of D(k-th change) - D(now)."""
    d = y[..., 0] - y[..., 1]
    b, t = d.shape
    changed = torch.zeros_like(d, dtype=torch.long)
    changed[:, 1:] = (d[:, 1:] != d[:, :-1]).long()
    cnt = torch.cumsum(changed, 1)
    pos = torch.searchsorted(cnt.contiguous(), (cnt + k).contiguous())
    has = pos < t
    j = pos.clamp(max=t - 1)
    idx = torch.arange(t, device=y.device).expand(b, t)
    rows_to = torch.log1p((j - idx).float())
    move = torch.gather(d, 1, j) - d
    return (rows_to, (move > 0).float(), valid & has)


def shifted_targets(
    y: torch.Tensor, valid: torch.Tensor, shifts: list[int]
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Training-only labels: the target ``sh`` rows later. Never used as an input."""
    out = []
    for sh in shifts:
        ys = torch.cat([y[:, sh:], y[:, -1:].expand(-1, sh, -1)], dim=1)
        keep = valid.clone()
        keep[:, -sh:] = False
        out.append((ys, keep))
    return out


def true_mask_weights(
    main_valid: torch.Tensor,
    valid: torch.Tensor,
    scored: torch.Tensor,
    eps: float,
    branchless: bool,
) -> torch.Tensor:
    """Rows of sequences that carry a true scoring mask get c * valid * (is_scored + eps), with c chosen so
    those sequences keep the total weight their propensity weights had; other sequences keep main_valid.

    branchless=True computes the same rows and the same c without boolean indexing or a host-side branch
    (no GPU->CPU sync); tests/test_perf_parity.py checks both against each other."""
    true_w = scored.float() + eps
    if branchless:
        hm = scored.any(dim=1)[:, None]
        c = (main_valid * hm).sum() / (valid.float() * true_w * hm).sum().clamp_min(1e-08)
        return torch.where(hm, valid.float() * true_w * c, main_valid)
    has_mask = scored.any(dim=1)
    if has_mask.any():
        c = main_valid[has_mask].sum() / (
            valid.float()[has_mask] * true_w[has_mask]
        ).sum().clamp_min(1e-08)
        return torch.where(has_mask[:, None], valid.float() * true_w * c, main_valid)
    return main_valid


def episode_redundancy_weight(y: torch.Tensor, scale: float, floor: float) -> torch.Tensor:
    """Reduce duplicate class-3 targets in long offline label episodes.

    Episode length uses future labels only to set training weights. It is never
    added to model inputs or computed during inference.
    """
    batch, steps, _ = y.shape
    row = torch.arange(steps, device=y.device)[None].expand(batch, -1)
    changed = torch.ones((batch, steps), dtype=torch.bool, device=y.device)
    changed[:, 1:] = (y[:, 1:] != y[:, :-1]).any(-1)
    start = torch.cummax(torch.where(changed, row, 0), 1).values
    marker = torch.where(changed, row, steps)
    next_change = torch.flip(torch.cummin(torch.flip(marker, (1,)), 1).values, (1,))
    end = torch.cat((next_change[:, 1:], torch.full_like(next_change[:, :1], steps)), 1)
    length = (end - start).float()
    weight = torch.rsqrt(1.0 + length / scale).clamp_min(floor)
    return torch.where(sequence_class(y)[:, None] == 3, weight, 1.0)


def main_row_weights(
    inputs: torch.Tensor,
    y: torch.Tensor,
    valid: torch.Tensor,
    scored: torch.Tensor,
    prop_model: torch.nn.Module | None,
    prop_cached: torch.Tensor | None,
    args: argparse.Namespace,
    raw: torch.Tensor | None = None,
    *,
    environment: TrainingEnvironment | None = None,
) -> torch.Tensor:
    """Shared fit/calibration weights, including the batch-level true-mask scale."""
    environment = environment if environment is not None else TrainingEnvironment()
    main_valid = valid & scored if args.mask_weight else valid
    if prop_model is not None:
        if prop_cached is not None:
            prob = prop_cached
        elif args.prop_cache:
            raise RuntimeError("--prop-cache set but the batch carries no cached propensity")
        else:
            prop_inputs = inputs[..., : N_FEATURES + N_EVENT_FEATURES]
            if environment.inputs.sentinel_preprocessor is not None:
                if raw is None:
                    raise ValueError("sentinel weighting requires original raw propensity inputs")
                prop_inputs = torch.cat((raw[..., :N_FEATURES], prop_inputs[..., N_FEATURES:]), -1)
            prob = environment.propensity(prop_model, prop_inputs)
        main_valid = valid.float() * (prob**args.prop_power + args.prop_eps)
        if args.class_weight:
            cw = torch.tensor([float(v) for v in args.class_weight.split(",")], device=y.device)
            main_valid = main_valid * cw[sequence_class(y)][:, None]
        if args.valid_true_mask >= 0:
            main_valid = true_mask_weights(
                main_valid, valid, scored, args.valid_true_mask, args.sync_free
            )
    if args.since_weight > 0:
        since = torch.expm1(inputs[..., N_FEATURES + FEATURE_NAMES.index("since_i0px")].float())
        sw_rows = torch.clamp(1.0 / (1.0 + since / args.since_weight), min=args.since_floor)
        main_valid = main_valid.float() * sw_rows
    if args.episode_weight > 0:
        main_valid = main_valid.float() * episode_redundancy_weight(
            y, args.episode_weight, args.episode_floor
        )
    return main_valid
