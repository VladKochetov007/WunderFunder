from __future__ import annotations

import argparse


def add_targets_arguments(ap: argparse.ArgumentParser) -> None:
    """Register targets options in their historical order."""
    ap.add_argument(
        "--loss",
        choices=[
            "corr",
            "wmse",
            "mix",
            "huber",
            "ce_joint",
            "mse",
            "gcorr",
            "wmse_gcorr",
            "gcorr_pool",
        ],
        default="mix",
    )
    ap.add_argument(
        "--thin-prob",
        type=float,
        default=0.0,
        help="training-view row thinning: probability per sequence (0 disables)",
    )
    ap.add_argument(
        "--thin-range", default="0.1,0.4", help="low,high of the dropped share of quiet rows"
    )
    ap.add_argument(
        "--thin-mode",
        choices=["compress", "retained-mass"],
        default="compress",
        help="retained-mass preserves the observed stream and masks the same sampled quiet rows only in losses",
    )
    ap.add_argument(
        "--pool-decay",
        type=float,
        default=0.98,
        help="--loss gcorr_pool: per-chunk decay of the pooled weighted sums kept from earlier chunks",
    )
    ap.add_argument("--huber-delta", type=float, default=0.5)
    ap.add_argument(
        "--ce-tilt", action="store_true", help="weight code cross-entropy rows by mean |clip(y)|"
    )
    ap.add_argument(
        "--ce-mse",
        type=float,
        default=0.0,
        help="add this weight x wMSE on the differentiable readout",
    )
    ap.add_argument(
        "--distill",
        default="",
        help="teacher runs name[*scale],...; cached pred_<name>_<split>.npy",
    )
    ap.add_argument(
        "--distill-cache-root",
        default="",
        help="validated preweighted two-column teacher cache for independent distillation",
    )
    ap.add_argument(
        "--distill-cache-variant",
        choices=["source", "normalized", "homogeneous"],
        default="source",
        help="main target column set in the preweighted cache",
    )
    ap.add_argument(
        "--distill-cache-scale",
        type=float,
        default=1.0,
        help="scale cached main teacher targets before the training-time ±2 clip",
    )
    ap.add_argument(
        "--distill-family-weight",
        type=float,
        default=0.0,
        help="training-only weighted-MSE coefficient for the four-family eight-output head",
    )
    ap.add_argument(
        "--distill-alpha", type=float, default=0.5, help="target = (1-a) y + a mean(teachers)"
    )
    ap.add_argument(
        "--distill-alpha-far",
        type=float,
        default=None,
        help="noise-adaptive distillation: a(u) = a + (far - a)(1 - exp(-u/tau)), u = rows until the label changes (training-time only; inputs stay causal)",
    )
    ap.add_argument(
        "--distill-alpha-tau",
        type=float,
        default=32.0,
        help="rows scale of the noise-adaptive weight",
    )
    ap.add_argument(
        "--distill-boot-h",
        type=int,
        default=0,
        help="stopped bootstrap: in the distillation blend replace y by the teacher at row t+h while the label run outlives t+h, else keep y (training-time only; min(t+h, run end) is a stopping time)",
    )
    ap.add_argument(
        "--distill-reanchor",
        type=float,
        default=0.0,
        help="original-label fraction of final distilled target on bootstrap-surviving rows only",
    )
    ap.add_argument(
        "--readout-activation",
        choices=["gelu", "fixed_mix"],
        default="gelu",
        help="readout-only fixed equal GELU/LeakyReLU(.01) mixture; research export required",
    )
    ap.add_argument(
        "--aux-next",
        type=float,
        default=0.0,
        help="weight of auxiliary heads: log rows to next target change (MSE) and its sign (BCE)",
    )
    ap.add_argument(
        "--aux-next-input-head",
        action="store_true",
        help="append a training-only head that predicts the next canonical model-input row",
    )
    ap.add_argument(
        "--aux-next-input-weight",
        type=float,
        default=0.0,
        help="SmoothL1 weight for the next-input auxiliary objective (requires its head)",
    )
    ap.add_argument(
        "--aux-tier",
        type=float,
        default=0.0,
        help="class-3 winning-target tick-tier auxiliary CE; head discarded for prediction",
    )
    ap.add_argument(
        "--aux-mask",
        type=float,
        default=0.0,
        help="validation-source scoring-mask auxiliary BCE; head discarded for prediction",
    )
    ap.add_argument(
        "--aux-sign",
        type=float,
        default=0.0,
        help="current target-contrast direction BCE; head discarded for prediction",
    )
    ap.add_argument(
        "--aux-quote",
        type=float,
        default=0.0,
        help="training-only class-3 future 8/24-book-event mean quote displacement MSE",
    )
    ap.add_argument(
        "--quote-cache-root",
        default="",
        help="optional paired auxiliary cache directory; historical cache remains the default",
    )
    ap.add_argument(
        "--aux-next-k",
        default="",
        help="extra event horizons for the next-change aux heads, e.g. 2,4",
    )
    ap.add_argument(
        "--spread-aux-mode",
        choices=["original", "proxy_known", "current_known"],
        default="original",
    )
    ap.add_argument(
        "--spread-cache-root",
        default="",
        help="supervision-only cache bound to the clean fitting population",
    )
    ap.add_argument(
        "--aux-struct",
        type=float,
        default=0.0,
        help="weight of auxiliary cross-entropy heads for spread (ticks), branch and class",
    )
    ap.add_argument("--aux", default="", help="future row shifts as auxiliary targets, e.g. 20,200")
    ap.add_argument("--aux-weight", type=float, default=0.5)
