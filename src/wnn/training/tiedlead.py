from __future__ import annotations

from collections.abc import Mapping

from wnn.tiedlead import TIEDLEAD_HIDDEN


def tiedlead_configuration_error(args: Mapping[str, object]) -> str | None:
    """Reject options that change the fixed clean TL3 teacher architecture or inputs."""
    fixed = {
        "arch": "gru",
        "cell": "gru",
        "enc": "tiedlead",
        "hidden": TIEDLEAD_HIDDEN,
        "layers": 2,
        "dropout": 0.0,
        "precision": "fp32",
        "numeric_map": "none",
        "readout_activation": "gelu",
        "research_features": "tl3_lead",
        "research_feature_control": "none",
        "privileged_mode": "none",
        "prop_input": "",
        "init": "",
        "rnn_init": "default",
        "qat": "none",
        "activation_backend": "none",
    }
    invalid = [
        f"--{name.replace('_', '-')}={args.get(name)!r}"
        for name, value in fixed.items()
        if args.get(name, value) != value
    ]
    disabled = {
        "raw_only": "--raw-only",
        "heuristics": "--heuristics",
        "online_norm": "--online-norm",
        "init_qmc": "--init-qmc",
        "recurrent_rank": "--recurrent-rank",
        "sentinel_base_width": "sentinel preprocessing",
        "distill_family_weight": "--distill-family-weight",
        "aux_struct": "auxiliary output heads",
        "aux_next": "auxiliary output heads",
        "aux_tier": "auxiliary output heads",
        "aux_mask": "auxiliary output heads",
        "aux_sign": "auxiliary output heads",
        "aux_quote": "auxiliary output heads",
    }
    invalid.extend(option for name, option in disabled.items() if args.get(name, 0))
    if args.get("aux", ""):
        invalid.append("--aux")
    if args.get("distill", "") or args.get("distill_cache_root", ""):
        invalid.append("labels-only teachers cannot use distillation targets")
    if args.get("ema_teacher", 0.0) or args.get("td_alpha", 0.0):
        invalid.append("labels-only teachers cannot use teacher-consistency objectives")
    if args.get("sentinel_indicators", "none") != "none":
        invalid.append("sentinel preprocessing")
    if args.get("sentinel_values", "unchanged") != "unchanged":
        invalid.append("sentinel preprocessing")
    if any(
        args.get(name) is not None for name in ("encoder_width", "recurrent_width", "readout_width")
    ):
        invalid.append("independent encoder/recurrent/readout widths")
    if args.get("head_layers", 1) != 1:
        invalid.append("--head-layers")
    return None if not invalid else "tiedlead profile rejects " + ", ".join(dict.fromkeys(invalid))
