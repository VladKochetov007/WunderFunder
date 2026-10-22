from __future__ import annotations

from pathlib import Path
from typing import cast

from warm_i1_preflight_parity import compare_parent_parity
from warm_i1_preflight_setup import setup_arm


def _cpu_argv(argv: list[str], run_name: str) -> list[str]:
    """Retain the fit setup while removing save-state and selecting CPU setup."""
    result: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == "--save-training-state":
            index += 1
            continue
        if value == "--fast":
            # Keep the fitted propensity cache while disabling GPU-only loader/optimizer flags.
            result.append("--prop-cache")
            index += 1
            continue
        if value == "--name":
            result.extend((value, run_name))
            index += 2
            continue
        if value == "--device":
            result.extend((value, "cpu"))
            index += 2
            continue
        result.append(value)
        index += 1
    if "--init" not in result or "--i1-warm-normalization" not in result:
        raise ValueError("preflight command lost its parent or appended-only normalization")
    if "--save-training-state" in result or "--resume-state" in result:
        raise ValueError("setup-only preflight cannot save or resume training state")
    return result


def run_fold_preflight(
    plan: dict[str, object], fold: int, project_root: Path, source_root: Path
) -> dict[str, object]:
    """Require paired setup identities, exact initialization, and parent parity."""
    cells = cast(list[dict[str, object]], plan["cells"])
    cell = next(item for item in cells if item["fold"] == fold)
    runs = cast(list[dict[str, object]], plan["runs"])
    prepared: list[dict[str, object]] = []
    parity: dict[str, list[dict[str, object]]] = {}
    for arm, control in (("control", "all"), ("history", "none")):
        record = next(item for item in runs if item["fold"] == fold and item["arm"] == arm)
        run_name = f"preflight_{record['run_id']}"
        argv = _cpu_argv(cast(list[str], record["fit_argv"]), run_name)
        setup, run = setup_arm(argv, run_name, project_root, source_root, plan, cell)
        if setup["arm"] != control:
            raise ValueError(f"feature control mismatch for {arm}: {setup['arm']}")
        prepared.append(setup)
        parity[arm] = compare_parent_parity(run, project_root, source_root, 1e-5)
        del run

    left, right = prepared
    identity_fields = (
        "initial_checkpoint_sha256",
        "split_manifest_sha256",
        "fit_membership_identity",
        "initial_parameter_sha256",
        "normalization_identity",
        "appended_mean_identity",
        "appended_std_identity",
        "feature_prior_sha256",
        "torch_rng_before_normalization_sha256",
        "torch_rng_after_normalization_sha256",
        "sample_order_before_normalization",
        "sample_order_after_normalization",
    )
    for key in identity_fields:
        if left[key] != right[key]:
            raise ValueError(f"matched control/history setup identity differs: {key}")
    return {
        "fold": fold,
        "parent_checkpoint_sha256": left["initial_checkpoint_sha256"],
        "matched_setup_identity_sha256": _identity(
            {key: left[key] for key in left if key not in ("run_name", "arm")}
        ),
        "runs": prepared,
        "parity": parity,
        "parity_tolerance": 1e-5,
    }


def _identity(value: object) -> str:
    """Hash a deterministic JSON representation of paired setup identities."""
    import hashlib
    import json

    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
