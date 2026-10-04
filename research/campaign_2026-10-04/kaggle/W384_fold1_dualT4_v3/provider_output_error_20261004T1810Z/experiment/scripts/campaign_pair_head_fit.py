"""Fit frozen-procedure interaction and matched controls on one clean fold.

This command reads only ``*_fit`` state artifacts. Evaluation states are never
opened. All heads use the same selected rows, shuffled blocks, optimizer
updates, loss and fixed final epoch; scoring runs as a separate later command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from wnn.resources import configure_torch
from wnn.splits import load_campaign_split

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("bilinear", "unary", "output_only")


class ResidualHead(nn.Module):
    """Zero-initialized rank-8 correction with one of three fixed mechanisms."""

    def __init__(self, kind: str, hidden: int, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        if kind not in KINDS:
            raise ValueError(kind)
        self.kind = kind
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())
        if kind == "output_only":
            self.readout = nn.Linear(4, 8)
        else:
            self.left = nn.Linear(hidden, 8)
            self.right = nn.Linear(hidden, 8)
        self.out = nn.Linear(8, 2, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, states: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        if self.kind == "output_only":
            features = torch.nn.functional.gelu(self.readout(predictions.reshape(-1, 4)))
        else:
            normalized = (states - self.mean) / self.std
            left = self.left(normalized[:, 0])
            right = self.right(normalized[:, 1])
            features = (
                left * right
                if self.kind == "bilinear"
                else torch.nn.functional.gelu(left) + torch.nn.functional.gelu(right)
            )
        return self.out(features)


def state_normalization(states: np.memmap, block: int = 32_768) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute train-only member/channel scaling in a bounded streaming pass."""
    total = np.zeros(states.shape[1:], dtype=np.float64)
    square = np.zeros_like(total)
    for start in range(0, len(states), block):
        x = np.asarray(states[start : start + block], dtype=np.float64)
        total += x.sum(axis=0)
        square += np.square(x).sum(axis=0)
    mean = total / len(states)
    variance = np.maximum(square / len(states) - mean * mean, 1e-4)
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(np.sqrt(variance), dtype=torch.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-prefix", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs, batch and learning rate must be positive")
    prefix = args.fit_prefix.resolve()
    meta_path = prefix.with_suffix(".json")
    metadata = json.loads(meta_path.read_text())
    if metadata["mode"] != "fit" or metadata["selection"] != "need_prediction AND is_scored":
        raise ValueError("head fitting requires selected fitting rows from one clean outer fold")
    split = load_campaign_split(
        ROOT / "research" / "campaign_2026-09-29" / "artifacts" / "splits" / f"fold{metadata['fold']}.json", ROOT
    )
    if split.sha256 != metadata["manifest_sha256"] or list(split.valid_groups) != metadata["groups"]:
        raise ValueError("fitting groups differ from frozen outer-fold manifest")
    for key, expected in metadata["arrays_sha256"].items():
        actual = sha256(prefix.with_name(f"{prefix.name}_{key}.npy"))
        if actual != expected:
            raise ValueError(f"{key} artifact changed after export")
    states = np.load(prefix.with_name(f"{prefix.name}_states.npy"), mmap_mode="r")
    predictions = np.load(prefix.with_name(f"{prefix.name}_predictions.npy"), mmap_mode="r")
    targets = np.load(prefix.with_name(f"{prefix.name}_targets.npy"), mmap_mode="r")
    masks = np.load(prefix.with_name(f"{prefix.name}_mask.npy"), mmap_mode="r")
    row_groups = np.load(prefix.with_name(f"{prefix.name}_row_group.npy"), mmap_mode="r")
    if states.shape != (metadata["n_rows"], 2, metadata["hidden"]) or not masks.all():
        raise ValueError("fitting state shape or official mask is invalid")
    if targets.shape != (len(states), 2) or predictions.shape != (len(states), 2, 2):
        raise ValueError("fitting target/prediction shapes are invalid")
    if row_groups.shape != (len(states),) or not np.isin(row_groups, split.valid_groups).all():
        raise ValueError("head fitting contains a row outside the disjoint fitting groups")
    if len(metadata["members"]) != 2 or len(metadata["member_checkpoint_sha256"]) != 2:
        raise ValueError("head fitting requires exactly two frozen backbone checkpoints")

    configure_torch()
    device = torch.device("cuda")
    mean, std = state_normalization(states)
    output = ROOT / "research" / "campaign_2026-09-29" / "artifacts" / "pair_heads" / args.name
    output.mkdir(parents=True, exist_ok=False)
    blocks = (len(states) + args.batch - 1) // args.batch
    rng = np.random.default_rng(args.seed)
    orders = [rng.permutation(blocks) for _ in range(args.epochs)]
    history: dict[str, list[float]] = {}
    checkpoints: dict[str, str] = {}
    for kind in KINDS:
        torch.manual_seed(args.seed)
        head = ResidualHead(kind, metadata["hidden"], mean, std).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
        losses: list[float] = []
        for order in orders:
            total_loss = 0.0
            for block_index in order:
                start = int(block_index) * args.batch
                stop = min(start + args.batch, len(states))
                state = torch.as_tensor(np.asarray(states[start:stop], dtype=np.float32), device=device)
                base = torch.as_tensor(np.asarray(predictions[start:stop]), device=device)
                target = torch.as_tensor(np.asarray(targets[start:stop]), device=device).clamp(-2, 2)
                raw_prediction = base.mean(dim=1) + head(state, base)
                weights = target.abs()
                loss = (weights * (raw_prediction - target).square()).sum() / weights.sum().clamp_min(1e-8)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())
            losses.append(total_loss / blocks)
            print(json.dumps({"kind": kind, "epoch": len(losses) - 1, "wmse": losses[-1]}), flush=True)
        checkpoint_path = output / f"{kind}.pt"
        torch.save(
            {
                "model": head.cpu().state_dict(),
                "kind": kind,
                "hidden": metadata["hidden"],
                "epoch": args.epochs - 1,
                "fit_manifest_sha256": metadata["manifest_sha256"],
                "fit_export_sha256": sha256(meta_path),
                "member_checkpoint_sha256": metadata["member_checkpoint_sha256"],
                "fixed_training_args": vars(args) | {"fit_prefix": str(prefix)},
            },
            checkpoint_path,
        )
        checkpoints[kind] = sha256(checkpoint_path)
        history[kind] = losses
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "fit_export": str(prefix.relative_to(ROOT)),
                "fit_export_sha256": sha256(meta_path),
                "n_fit_rows": len(states),
                "members": metadata["members"],
                "member_checkpoint_sha256": metadata["member_checkpoint_sha256"],
                "manifest_sha256": metadata["manifest_sha256"],
                "fixed_epoch": args.epochs - 1,
                "n_updates_per_head": args.epochs * blocks,
                "checkpoint_sha256": checkpoints,
                "history": history,
                "evaluation_access": "none",
            },
            indent=2,
        ) + "\n"
    )


if __name__ == "__main__":
    main()
