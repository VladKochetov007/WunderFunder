from __future__ import annotations

import argparse

from wnn.training_dynamics import add_arguments as add_dynamics_arguments


def add_performance_arguments(ap: argparse.ArgumentParser) -> None:
    """Register performance options in their historical order."""
    perf = ap.add_argument_group("throughput (see research/perf-2026-09-27.md)")
    perf.add_argument(
        "--fast",
        action="store_true",
        help="all verified throughput options: --sync-free --fused-adam --pin-memory --prop-cache",
    )
    perf.add_argument(
        "--sync-free", action="store_true", help="no per-step host<->GPU syncs in the train step"
    )
    perf.add_argument(
        "--fused-adam",
        action="store_true",
        help="fused AdamW (GradScaler skips inf steps on device)",
    )
    perf.add_argument(
        "--pin-memory", action="store_true", help="pinned persistent loaders + CUDA-stream prefetch"
    )
    perf.add_argument(
        "--prop-cache",
        action="store_true",
        help="read propensity from cache/propcache_<name>_<split>.npy (scripts/cache_propensity.py)",
    )
    perf.add_argument(
        "--precision",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="autocast dtype; bf16 trains without GradScaler",
    )
    perf.add_argument(
        "--amp-init-scale",
        type=float,
        default=65536.0,
        help="initial GradScaler scale for fp16; default preserves PyTorch's current training scale",
    )
    perf.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="stop each epoch after this many batches (harness)",
    )
    perf.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help="time this many batches per stage (after --profile-warmup), print a table and exit",
    )
    perf.add_argument("--profile-warmup", type=int, default=5)
    perf.add_argument(
        "--parity-dump", default="", help="save per-step losses and final weights here, then exit"
    )
    perf.add_argument(
        "--deterministic",
        action="store_true",
        help="cudnn deterministic, no benchmark (parity runs)",
    )
    add_dynamics_arguments(ap)
