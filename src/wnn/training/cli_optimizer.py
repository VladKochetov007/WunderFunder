from __future__ import annotations

import argparse


def add_optimizer_arguments(ap: argparse.ArgumentParser) -> None:
    """Register optimizer options in their historical order."""
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--wd", type=float, default=0.01)
    research = ap.add_argument_group(
        "optimizer research (src/wnn/optim.py; defaults reproduce AdamW + OneCycle)"
    )
    research.add_argument(
        "--optim",
        choices=["adamw", "muon", "sgd", "cautious_adamw", "ademamix"],
        default="adamw",
        help="muon: orthogonalised per-gate updates on the recurrent matrices, AdamW elsewhere; "
        "cautious_adamw/ademamix: src/wnn/optim_variants.py",
    )
    research.add_argument("--ademamix-alpha", type=float, default=4.0, help="slow-EMA mix weight")
    research.add_argument("--ademamix-beta3", type=float, default=0.999, help="slow-EMA decay")
    research.add_argument(
        "--ademamix-warmup", type=int, default=0, help="alpha/beta3 warmup steps (0: none)"
    )
    research.add_argument(
        "--muon-scope",
        choices=["rnn", "hidden"],
        default="rnn",
        help="hidden: Muon also on head.0.weight",
    )
    research.add_argument("--beta2", type=float, default=0.999, help="Adam second-moment decay")
    research.add_argument(
        "--sched",
        choices=["onecycle", "wsd", "sgdr"],
        default="onecycle",
        help="wsd: 5%% linear warmup, constant lr, (1 - sqrt) cooldown over the last --wsd-decay",
    )
    research.add_argument(
        "--wsd-decay", type=float, default=0.2, help="cooldown share of all optimizer steps"
    )
    research.add_argument(
        "--wsd-final", type=float, default=0.01, help="final lr as a fraction of --lr"
    )
    research.add_argument(
        "--wsd-shape",
        choices=["sqrt", "linear"],
        default="sqrt",
        help="WSD cooldown shape; linear with --wsd-decay 1 --wsd-final 0 is decay-to-zero",
    )
    research.add_argument(
        "--rnn-init",
        choices=["default", "chrono", "orth", "chrono_orth"],
        default="default",
        help="chrono gate biases (memory timescales U[2, --chrono-tmax] rows) and/or orthogonal W_hh",
    )
    research.add_argument("--chrono-tmax", type=float, default=2000.0)
    research.add_argument("--clip", type=float, default=1.0, help="global gradient-norm clip")
    research.add_argument(
        "--grad-stats",
        action="store_true",
        help="per-epoch pre-clip gradient-norm quantiles, clip rate and family shares",
    )
