from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wnn.splits import file_sha256
from wnn.training.masked_group_pretraining import (
    MASKED_INPUT_WIDTH,
    FeatureOnlySequences,
    build_masked_model,
    install_input_statistics,
    pretraining_source_paths,
)
from wnn.training.masked_pretraining_loop import fit_masked_groups
from wnn.training.masked_pretraining_statistics import (
    load_pretraining_membership,
    load_statistics,
    prepare_statistics,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the feature-only statistics and pretraining commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stats = commands.add_parser("stats", help="prepare fixed fold-clean input statistics")
    _add_data_arguments(stats)
    stats.add_argument("--output", type=Path, required=True)
    stats.add_argument("--seed", type=int, default=880301)
    stats.add_argument("--sample-sequences", type=int, default=64)
    stats.add_argument("--row-stride", type=int, default=64)
    stats.add_argument("--cpu-threads", type=int, default=1)
    stats.add_argument("--verify-only", action="store_true")
    fit = commands.add_parser("pretrain", help="fit a causal masked-group GRU initializer")
    _add_data_arguments(fit)
    fit.add_argument("--stats", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--seed", type=int, required=True)
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--hidden", type=int, default=512)
    fit.add_argument("--batch-size", type=int, default=4)
    fit.add_argument("--epochs", type=int, default=2)
    fit.add_argument("--chunk-rows", type=int, default=2000)
    fit.add_argument("--mask-block-rows", type=int, default=32)
    fit.add_argument("--learning-rate", type=float, default=1e-4)
    fit.add_argument("--weight-decay", type=float, default=0.01)
    fit.add_argument("--max-grad-norm", type=float, default=1.0)
    fit.add_argument("--profile-updates", type=int, default=0)
    fit.add_argument("--resume", type=Path)
    fit.add_argument("--cpu-threads", type=int, default=1)
    return parser


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    """Add project-root and frozen split bindings."""
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)


def run_pretraining(args: argparse.Namespace) -> None:
    """Run a strict-FP32 fit or a bounded real-shape B300 profile."""
    if args.batch_size <= 0 or args.cpu_threads <= 0:
        raise ValueError("batch size and CPU thread count must be positive")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    environment, split, members = load_pretraining_membership(args.root, args.split_manifest)
    mean, std, stats_metadata = load_statistics(args.stats, split, environment)
    model = build_masked_model(hidden=args.hidden, device=device, environment=environment)
    install_input_statistics(model, mean.numpy(), std.numpy())
    dataset = FeatureOnlySequences(members, environment.root, environment)
    order_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=order_generator,
    )
    source_paths = pretraining_source_paths(environment.source_root)
    recipe = {
        "schema": "causal_group_mask_pretraining_v1",
        "seed": args.seed,
        "fold": split.fold,
        "split_manifest_sha256": split.sha256,
        "fit_members": {
            "train": list(split.train_groups),
            "valid": list(split.valid_groups),
        },
        "evaluation_members": list(split.evaluation_groups),
        "stats_sha256": stats_metadata["statistics_sha256"],
        "stats_metadata_sha256": file_sha256(args.stats.with_suffix(args.stats.suffix + ".json")),
        "feature_only": True,
        "source_sha256": {name: file_sha256(path) for name, path in source_paths.items()},
        "model": {
            "family": "plain_gru",
            "hidden": args.hidden,
            "layers": 1,
            "input_width": MASKED_INPUT_WIDTH,
            "raw_group_count": 13,
            "mask_block_rows": args.mask_block_rows,
        },
        "optimization": {
            "precision": "fp32",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "chunk_rows": args.chunk_rows,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "optimizer_betas": [0.9, 0.999],
            "max_grad_norm": args.max_grad_norm,
            "scheduler": "OneCycleLR_cosine_pct_start_0.05",
            "scheduler_div_factor": 10.0,
            "scheduler_final_div_factor": 100.0,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = fit_masked_groups(
        model,
        loader,
        mean,
        std,
        environment=environment,
        device=device,
        seed=args.seed,
        epochs=args.epochs,
        chunk_rows=args.chunk_rows,
        mask_block_rows=args.mask_block_rows,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        progress_path=args.output_dir / "progress.jsonl",
        checkpoint_dir=args.output_dir / "checkpoints",
        recipe=recipe,
        profile_updates=args.profile_updates,
        resume_checkpoint=args.resume,
    )
    summary.update(
        {
            "recipe": recipe,
            "device": str(device),
            "feature_only_fit_members": len(members),
            "statistics_metadata": stats_metadata,
        }
    )
    summary_path = args.output_dir / (
        "profile.json" if args.profile_updates else "pretrain_result.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if summary["complete"]:
        checkpoint = {
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "recipe": recipe,
            "normalization_mean": mean,
            "normalization_std": std,
            "statistics_metadata": stats_metadata,
            "epoch": args.epochs - 1,
        }
        final_checkpoint = args.output_dir / "pretrained.pt"
        temporary_checkpoint = final_checkpoint.with_suffix(f".tmp-{args.seed}")
        torch.save(checkpoint, temporary_checkpoint)
        temporary_checkpoint.replace(final_checkpoint)
    print(json.dumps(summary, indent=2))


def main() -> None:
    """Run the selected bounded pretraining command."""
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "stats":
        prepare_statistics(args)
    else:
        run_pretraining(args)


if __name__ == "__main__":
    main()
