from __future__ import annotations

import argparse


def representation_configuration_error(args: argparse.Namespace) -> str | None:
    """Reject input variants whose columns would make the fixed slots ambiguous."""
    if args.representation_mode is None:
        return None
    rejected = []
    if args.arch != "gru" or args.cell != "gru" or args.enc != "linear":
        rejected.append("the plain GRU with linear input encoding is required")
    if args.precision != "fp32" or args.qat != "none" or args.recurrent_rank:
        rejected.append("dense FP32 without QAT is required")
    if args.raw_only:
        rejected.append("--raw-only")
    if args.research_features != "none":
        rejected.append("--research-features")
    if args.privileged_mode != "none":
        rejected.append("--privileged-mode")
    if args.heuristics:
        rejected.append("--heuristics")
    if args.online_norm:
        rejected.append("--online-norm")
    if args.prop_input:
        rejected.append("--prop-input")
    if args.numeric_map != "none":
        rejected.append("--numeric-map")
    if args.sentinel_base_width or args.sentinel_indicators != "none":
        rejected.append("sentinel preprocessing")
    if args.sentinel_values != "unchanged":
        rejected.append("--sentinel-values")
    if not rejected:
        return None
    return "--representation-mode does not support " + ", ".join(rejected)
