"""Fixed-procedure frozen-state falsifier; never reads evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from codebook import fit_codebooks, nearest_clipped_labels
from model import JointTargetHead, arm_loss, require_finite_training_state

ROOT = Path(__file__).resolve().parents[3]
ARMS = (
    "joint_tilted",
    "joint_ordinary",
    "shuffled_tilted",
    "independent_tilted",
    "regression_anchor",
)


class DutyCycleLike(Protocol):
    """The subset of the shared resource controller used by these runners."""

    def tick(self) -> None: ...


def file_sha256(path: Path) -> str:
    """Hash a source artifact in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_resources() -> DutyCycleLike | None:
    """Apply repository CPU/CUDA allocator limits before creating tensors."""
    sys.path.insert(0, str(ROOT / "src"))
    from wnn.resources import configure_torch

    return configure_torch()


def tick_gpu_duty(duty: DutyCycleLike | None, device: torch.device) -> None:
    """Honor the shared GPU duty cycle after a completed accelerator step."""
    if duty is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
        duty.tick()


def verify_fit_prefix(prefix: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Validate fit-only provenance and load read-only memory maps."""
    metadata_path = prefix.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("mode") != "fit"
        or metadata.get("selection") != "need_prediction AND is_scored"
    ):
        raise ValueError("requires a fit-only pair-state export with the official fit-row mask")
    split_path = (
        ROOT / "research/campaign_2026-09-29/artifacts/splits" / f"fold{metadata['fold']}.json"
    )
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from wnn.splits import load_campaign_split

    split = load_campaign_split(split_path, ROOT)
    if split.sha256 != metadata.get("manifest_sha256"):
        raise ValueError("fit-state manifest hash differs from the frozen split")
    if list(split.valid_groups) != metadata.get("groups"):
        raise ValueError("fit-state groups differ from the frozen fitting pool")
    members = metadata.get("members", [])
    member_hashes = metadata.get("member_checkpoint_sha256", [])
    if len(members) != 2 or len(member_hashes) != 2:
        raise ValueError("paired fit export must identify exactly two parent checkpoints")
    for member, expected_hash in zip(members, member_hashes, strict=True):
        checkpoint = ROOT / "artifacts/runs" / member / "model_ep0.pt"
        if file_sha256(checkpoint) != expected_hash:
            raise ValueError(f"parent checkpoint changed since paired-state export: {member}")
    arrays: dict[str, np.ndarray] = {}
    for key, expected_hash in metadata["arrays_sha256"].items():
        path = prefix.with_name(f"{prefix.name}_{key}.npy")
        if file_sha256(path) != expected_hash:
            raise ValueError(f"fit array changed after export: {key}")
        arrays[key] = np.load(path, mmap_mode="r")
    n_rows, hidden = metadata["n_rows"], metadata["hidden"]
    if arrays["states"].shape != (n_rows, 2, hidden):
        raise ValueError("unexpected paired-state shape")
    if arrays["predictions"].shape != (n_rows, 2, 2) or arrays["targets"].shape != (n_rows, 2):
        raise ValueError("unexpected prediction/target shape")
    if arrays["mask"].shape != (n_rows,) or not arrays["mask"].all():
        raise ValueError("fit export must contain only official selected rows")
    if not np.isfinite(arrays["predictions"]).all() or not np.isfinite(arrays["targets"]).all():
        raise ValueError("selected fit rows contain nonfinite values")
    if not np.isin(arrays["row_group"], split.valid_groups).all():
        raise ValueError("a row escaped the frozen fit groups")
    return metadata, arrays


def run_smoke(device_name: str, duty: DutyCycleLike | None = None) -> None:
    """Run a realistic batch/width synthetic optimizer smoke, never real data."""
    device = torch.device(device_name)
    torch.manual_seed(271828)
    rng = np.random.default_rng(271828)
    target = np.clip(rng.normal(size=(8192, 2)), -2, 2).astype(np.float32)
    paired, independent, shuffled, _ = fit_codebooks(target, joint_atoms=32, marginal_atoms=16)
    states = torch.randn(8192, 2, 128, device=device)
    parent = torch.rand(8192, 2, 2, device=device) * 2 - 1
    target_device = torch.as_tensor(target, device=device)
    paired_labels = torch.as_tensor(nearest_clipped_labels(target, paired), device=device)
    shuffled_labels = torch.as_tensor(nearest_clipped_labels(target, shuffled), device=device)
    marginal_labels = torch.stack(
        [
            torch.as_tensor(
                nearest_clipped_labels(target[:, k : k + 1], independent[k, :, None]),
                device=device,
            )
            for k in range(2)
        ],
        dim=-1,
    )
    expected = parent.mean(dim=1)
    for arm in ARMS:
        torch.manual_seed(271828)
        head = JointTargetHead(
            state_width=128,
            paired_codebook=torch.from_numpy(paired),
            independent_codebook=torch.from_numpy(independent),
            shuffled_codebook=torch.from_numpy(shuffled),
        ).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.01)
        output, joint_logits, independent_logits = head(states, parent, arm)
        if not torch.equal(output, expected):
            raise AssertionError(f"{arm} does not initialize exactly at the parent prediction")
        joint_y = shuffled_labels if arm == "shuffled_tilted" else paired_labels
        loss, ce, wmse = arm_loss(
            output,
            joint_logits,
            independent_logits,
            joint_y,
            marginal_labels,
            arm,
            target_device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        require_finite_training_state(loss, (ce, wmse), head.parameters(), "gradients")
        optimizer.step()
        require_finite_training_state(loss, (ce, wmse), head.parameters(), "parameters")
        tick_gpu_duty(duty, device)
    print(
        json.dumps(
            {
                "smoke": "passed",
                "device": str(device),
                "batch": 8192,
                "state_shape": [8192, 2, 128],
                "head_width": 128,
                "adamw_step_per_arm": True,
                "arms": list(ARMS),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-prefix", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=271828)
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    duty = configure_resources() if args.device == "cuda" else None
    if args.device == "cpu":
        torch.set_num_threads(1)
    if args.smoke:
        if args.fit_prefix or args.output_dir:
            parser.error("synthetic smoke cannot be combined with real artifacts")
        run_smoke(args.device, duty)
        return
    if not args.fit_prefix or not args.output_dir:
        parser.error("--fit-prefix and --output-dir are required outside synthetic smoke")
    if args.epochs != 1 or args.batch < 1 or args.lr <= 0:
        parser.error("this preregistered falsifier uses exactly one epoch and positive batch/lr")
    torch.manual_seed(args.seed)
    prefix = args.fit_prefix.resolve()
    metadata, arrays = verify_fit_prefix(prefix)
    targets = arrays["targets"]
    # Sampling is deterministic and draws only from this frozen training pool.
    fit_sample = np.asarray(
        targets[:: max(1, int(np.ceil(len(targets) / 65_536)))], dtype=np.float32
    )
    paired, independent, shuffled, n_codebook_rows = fit_codebooks(
        fit_sample, seed=args.seed, max_fit_rows=65_536
    )
    joint_labels = nearest_clipped_labels(targets, paired)
    shuffled_labels = nearest_clipped_labels(targets, shuffled)
    marginal_labels = np.column_stack(
        [
            nearest_clipped_labels(np.asarray(targets[:, k : k + 1]), independent[k, :, None])
            for k in range(2)
        ]
    )
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    hidden = int(metadata["hidden"])
    block_count = (len(targets) + args.batch - 1) // args.batch
    order = np.random.default_rng(args.seed).permutation(block_count)
    export_meta_sha = file_sha256(prefix.with_suffix(".json"))
    history: dict[str, dict[str, float]] = {}
    checkpoint_hashes: dict[str, str] = {}
    for arm in ARMS:
        torch.manual_seed(args.seed)
        head = JointTargetHead(
            hidden,
            torch.from_numpy(paired),
            torch.from_numpy(independent),
            torch.from_numpy(shuffled),
        ).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
        totals = np.zeros(3, dtype=np.float64)
        for batch_index in order:
            start = int(batch_index) * args.batch
            stop = min(start + args.batch, len(targets))
            state = torch.as_tensor(
                np.asarray(arrays["states"][start:stop], dtype=np.float32), device=device
            )
            parent = torch.as_tensor(np.asarray(arrays["predictions"][start:stop]), device=device)
            target = torch.as_tensor(
                np.asarray(targets[start:stop], dtype=np.float32), device=device
            ).clamp(-2, 2)
            pred, joint_logits, independent_logits = head(state, parent, arm)
            joint_y = shuffled_labels if arm == "shuffled_tilted" else joint_labels
            loss, ce, wmse = arm_loss(
                pred,
                joint_logits,
                independent_logits,
                torch.as_tensor(joint_y[start:stop], device=device),
                torch.as_tensor(marginal_labels[start:stop], device=device),
                arm,
                target,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            require_finite_training_state(loss, (ce, wmse), head.parameters(), "gradients")
            optimizer.step()
            require_finite_training_state(loss, (ce, wmse), head.parameters(), "parameters")
            tick_gpu_duty(duty, device)
            totals += np.asarray([loss.item(), ce.item(), wmse.item()])
        epoch_metrics = (totals / block_count).tolist()
        history[arm] = {
            "loss": epoch_metrics[0],
            "quantized_ce": epoch_metrics[1],
            "training_wmse_surrogate": epoch_metrics[2],
        }
        torch.save(
            {
                "state_dict": head.cpu().state_dict(),
                "arm": arm,
                "paired_codebook": paired,
                "independent_codebook": independent,
                "shuffled_codebook": shuffled,
                "fixed_epoch": 0,
                "fit_export_sha256": export_meta_sha,
                "fit_manifest_sha256": metadata["manifest_sha256"],
                "member_checkpoint_sha256": metadata["member_checkpoint_sha256"],
                "fit_array_sha256": metadata["arrays_sha256"],
                "training_args": {
                    "epochs": 1,
                    "batch": args.batch,
                    "lr": args.lr,
                    "seed": args.seed,
                },
            },
            destination / f"{arm}.pt",
        )
        checkpoint_hashes[arm] = file_sha256(destination / f"{arm}.pt")
    report = {
        "scope": "fit-only frozen-state falsifier; no evaluation artifact opened",
        "evaluation_access": "none",
        "fit_export": str(prefix),
        "fit_export_metadata_sha256": export_meta_sha,
        "fit_manifest_sha256": metadata["manifest_sha256"],
        "member_checkpoint_sha256": metadata["member_checkpoint_sha256"],
        "n_fit_rows": len(targets),
        "n_codebook_fit_rows": n_codebook_rows,
        "codebook_source": "same-origin clipped targets in frozen fit export only",
        "paired_atoms": len(paired),
        "marginal_atoms_per_target": independent.shape[1],
        "trainable_parameter_counts_equal": True,
        "arms": list(ARMS),
        "fixed_epoch": 0,
        "updates_per_arm": block_count,
        "checkpoint_sha256": checkpoint_hashes,
        "metrics_are_fit_diagnostics_only": True,
        "history": history,
    }
    (destination / "fit_receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"output_dir": str(destination), "updates_per_arm": block_count, "arms": list(ARMS)}
        )
    )


if __name__ == "__main__":
    main()
