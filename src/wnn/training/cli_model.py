from __future__ import annotations

import argparse

from wnn.specialist import PrefixAdapterGRUModel


def add_shape_arguments(ap: argparse.ArgumentParser) -> None:
    """Register shape options in their historical order."""
    ap.add_argument("--name", required=True)
    ap.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="CPU is available for bounded implementation checks; production default remains CUDA",
    )
    ap.add_argument("--train-seqs", type=int, default=10607)
    ap.add_argument("--valid-seqs", type=int, default=600, help="dev sequences used for selection")
    ap.add_argument("--full-valid", action="store_true", help="all 1505 dev sequences")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument(
        "--encoder-width",
        type=int,
        default=None,
        help="plain GRU input-projection width; defaults to --hidden",
    )
    ap.add_argument(
        "--recurrent-width",
        type=int,
        default=None,
        help="plain GRU state width; defaults to --hidden",
    )
    ap.add_argument(
        "--readout-width",
        type=int,
        default=None,
        help="plain GRU readout hidden width; defaults to --hidden",
    )
    ap.add_argument(
        "--head-layers", type=int, choices=(1, 2), default=1, help="number of hidden readout layers"
    )
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument(
        "--ensemble-k",
        type=int,
        default=1,
        help="TabM-mini GRU ensemble members (plain GRU only); 1 keeps the ordinary model",
    )


def add_architecture_arguments(ap: argparse.ArgumentParser) -> None:
    """Register architecture options in their historical order."""
    ap.add_argument(
        "--arch",
        choices=[
            "gru",
            "dual",
            "gated",
            "structured",
            "event-content",
            "tcn",
            "twoclock",
            "specialist",
            "prefix-adapter",
            "block-memory",
        ],
        default="gru",
        help="tcn: causal dilated conv stack, --hidden channels, --layers blocks (receptive field 2^layers); twoclock: --layers x GRU(--hidden) on rows + GRU(--event-hidden) stepping on a6 endpoint rows; specialist: shared GRU plus always-updated expert GRUs; prefix-adapter: matched low-rank arms; block-memory: causal block attention; width is --hidden; layers default 4; --block-heads 8; --block-rows 256",
    )
    ap.add_argument(
        "--block-heads",
        type=int,
        default=8,
        help="--arch block-memory: attention heads; --hidden must be divisible by this",
    )
    ap.add_argument(
        "--block-rows",
        type=int,
        default=256,
        help="--arch block-memory: rows in one completed causal block",
    )
    ap.add_argument(
        "--tcn-mix", type=int, default=0, help="--arch tcn: GELU + 1x1 conv after each block's conv"
    )
    ap.add_argument(
        "--decay-init",
        choices=["multi", "single"],
        default="multi",
        help="structured-state initial half-lives; identical learned bounds and phases",
    )
    ap.add_argument(
        "--memory-arm",
        choices=["endpoint-content", "endpoint-position", "raw-code-content"],
        default="endpoint-content",
    )
    ap.add_argument(
        "--expert-hidden",
        type=int,
        default=32,
        help="--arch specialist: hidden width of each always-updated expert GRU",
    )
    ap.add_argument(
        "--num-experts",
        type=int,
        default=4,
        help="--arch specialist: number of specialist GRU states",
    )
    ap.add_argument(
        "--specialist-route",
        choices=["learned", "hash", "universal"],
        default="learned",
        help="--arch specialist: learned causal top-one, deterministic prefix hash, or universal mean",
    )
    ap.add_argument(
        "--hash-seed",
        type=int,
        default=1729,
        help="--arch specialist: salt for the fixed hash-routing control",
    )
    ap.add_argument(
        "--adapter-rank",
        type=int,
        default=8,
        help="--arch prefix-adapter: rank shared by all adapter controls",
    )
    ap.add_argument(
        "--adapter-mode",
        choices=PrefixAdapterGRUModel.MODES,
        default="conditional",
        help="--arch prefix-adapter: conditional, unconditional, or diagonal control",
    )
    ap.add_argument(
        "--prefix-rows",
        type=int,
        default=99,
        help="maximum causal prefix rows used for routing and input adaptation",
    )
    ap.add_argument(
        "--gate-ce",
        type=float,
        default=0.0,
        help="BCE weight for the gated model's target-difference sign head",
    )
    ap.add_argument("--cell", choices=["gru", "lstm"], default="gru")
    ap.add_argument(
        "--enc",
        choices=["linear", "mlp", "tokens", "conv"],
        default="linear",
        help="per-row encoder before the recurrent cell (residual on the linear projection)",
    )
    ap.add_argument("--tok-dim", type=int, default=32)
    ap.add_argument("--att-layers", type=int, default=1)
    ap.add_argument("--conv-k", type=int, default=16)
    ap.add_argument("--event-hidden", type=int, default=128)
