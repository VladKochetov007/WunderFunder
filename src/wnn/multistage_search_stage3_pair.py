from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_search import (
    append_event,
    atomic_write_json,
    file_sha256,
    read_json,
)
from wnn.multistage_search_budget import utc_now


def finalize_pair(run_dir: Path, request: Mapping[str, object]) -> dict[str, object]:
    """Write a source-bound paired control/KD comparison once both fits have scores."""
    pair_id = request.get("pair_id")
    if not isinstance(pair_id, str):
        return {"status": "not_a_pair"}
    pair_dir = run_dir / "stage3" / "pairs" / pair_id
    pair_path = pair_dir / "pair.json"
    pair = read_json(pair_path)
    variants = pair.get("variants")
    if not isinstance(variants, list) or len(variants) != 2:
        raise ValueError("stage3 pair record must contain one control and one KD student")
    results: dict[str, tuple[Path, dict[str, object]]] = {}
    for variant in variants:
        if not isinstance(variant, Mapping):
            raise ValueError("stage3 pair variant record is malformed")
        number = variant.get("trial_number")
        role = variant.get("pair_role")
        if type(number) is not int or role not in {"label_only_control", "kd_student"}:
            raise ValueError("stage3 pair variant identity is malformed")
        result_path = run_dir / "stage3" / "trials" / f"{number:06d}" / "trial_result.json"
        if not result_path.is_file():
            return {"status": "awaiting_mate", "pair_id": pair_id}
        results[str(role)] = (result_path, read_json(result_path))
    if set(results) != {"label_only_control", "kd_student"}:
        raise ValueError("stage3 pair must include exactly one matched label-only and KD result")
    control_path, control = results["label_only_control"]
    kd_path, kd = results["kd_student"]
    control_config = read_json(control_path.parent / "config.json")
    kd_config = read_json(kd_path.parent / "config.json")
    control_without_weight = dict(control_config)
    kd_without_weight = dict(kd_config)
    control_weight, kd_weight = (
        control_without_weight.pop("kd_weight"),
        kd_without_weight.pop("kd_weight"),
    )
    if (
        control_weight != 0
        or not isinstance(kd_weight, (int, float))
        or kd_weight <= 0
        or control_without_weight != kd_without_weight
    ):
        raise ValueError("stage3 pair differs in more than its declared KD treatment")
    control_wp = _wp(control)
    kd_wp = _wp(kd)
    if (
        control.get("status") != "COMPLETE"
        or kd.get("status") != "COMPLETE"
        or control.get("stage") != 3
        or kd.get("stage") != 3
        or control.get("pair_id") != pair_id
        or kd.get("pair_id") != pair_id
        or control.get("pair_role") != "label_only_control"
        or kd.get("pair_role") != "kd_student"
        or not _is_inner_proxy(control)
        or not _is_inner_proxy(kd)
        or control.get("objective", {}).get("split_sha256")
        != kd.get("objective", {}).get("split_sha256")
        or control.get("objective", {}).get("primary_split_sha256")
        != kd.get("objective", {}).get("primary_split_sha256")
    ):
        raise ValueError("stage3 pair results do not share qualified control-only ancestry")
    if control.get("objective", {}).get("evaluation") != kd.get("objective", {}).get("evaluation"):
        raise ValueError("stage3 pair score masks or scope differ")
    result = {
        "schema": "wnn.multistage-stage3-pair-result.v1",
        "status": "complete",
        "pair_id": pair_id,
        "scope": "inner_train_proxy",
        "primary_valid_used": False,
        "matched_dimension": "kd_weight",
        "kd_weight_control": control_weight,
        "kd_weight_treatment": kd_weight,
        "label_only_control": _result_ref(control_path, control, control_wp),
        "kd_student": _result_ref(kd_path, kd, kd_wp),
        "delta_wp_kd_minus_control": kd_wp - control_wp,
        "teacher_pool_id": pair.get("teacher_pool_id"),
        "teacher_cache_manifest_sha256": pair.get("teacher_cache_manifest_sha256"),
        "pair_manifest_path": str(pair_path.resolve()),
        "pair_manifest_sha256": file_sha256(pair_path),
        "completed_at_utc": utc_now(),
    }
    output = pair_dir / "pair_result.json"
    if output.exists():
        old = read_json(output)
        if any(
            old.get(key) != result.get(key)
            for key in (
                "pair_id",
                "label_only_control",
                "kd_student",
                "delta_wp_kd_minus_control",
                "pair_manifest_sha256",
            )
        ):
            raise FileExistsError("stage3 pair result already exists with different contents")
        return {"status": "already_complete", "result_path": str(output.resolve())}
    atomic_write_json(output, result, replace=False)
    plots = _plot_pair(output, control_wp, kd_wp)
    append_event(
        run_dir / "events.jsonl",
        {
            "event": "stage3_pair_complete",
            "at_utc": utc_now(),
            "stage": 3,
            "pair_id": pair_id,
            "result_path": str(output.resolve()),
            "result_sha256": file_sha256(output),
            "delta_wp": kd_wp - control_wp,
            "status": "COMPLETE",
        },
    )
    return {
        "status": "complete",
        "result_path": str(output.resolve()),
        "result_sha256": file_sha256(output),
        "plots": plots,
    }


def _wp(result: Mapping[str, object]) -> float:
    """Extract only the verified pooled clipped WP from an inner objective."""
    objective = result.get("objective")
    nested = objective.get("objective") if isinstance(objective, Mapping) else None
    value = nested.get("value") if isinstance(nested, Mapping) else None
    if not isinstance(value, (int, float)):
        raise ValueError("paired trial result is missing its numeric inner WP")
    return float(value)


def _is_inner_proxy(result: Mapping[str, object]) -> bool:
    """Require the unscored train-source all-after-warmup proxy context."""
    objective = result.get("objective")
    evaluation = objective.get("evaluation") if isinstance(objective, Mapping) else None
    return (
        isinstance(evaluation, Mapping)
        and evaluation.get("scope") == "inner_train_proxy"
        and evaluation.get("source") == "train"
        and evaluation.get("is_scored_used") is False
        and evaluation.get("row_mask") == "all_after_warmup AND row_index>=99"
    )


def _result_ref(path: Path, result: Mapping[str, object], value: float) -> dict[str, object]:
    return {
        "trial_number": result.get("trial_number"),
        "trial_result_path": str(path.resolve()),
        "trial_result_sha256": file_sha256(path),
        "checkpoint_sha256": result.get("fit_model_sha256"),
        "config_sha256": result.get("config_sha256"),
        "objective_value": value,
    }


def _plot_pair(path: Path, control_wp: float, kd_wp: float) -> list[dict[str, str]]:
    """Save actual paired inner-WP values as PNG and SVG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.5, 4.0))
    bars = axis.bar(["Label-only", "KD student"], [control_wp, kd_wp], color=["#4c78a8", "#f58518"])
    axis.set_ylabel("Pooled clipped WP")
    axis.set_title("Matched stage3 inner train proxy")
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, fmt="%.4f")
    axis.text(
        0.5,
        0.02,
        f"KD − control = {kd_wp - control_wp:+.5f}",
        transform=axis.transAxes,
        ha="center",
    )
    figure.tight_layout()
    out_dir = path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    exports: list[dict[str, str]] = []
    for suffix in ("png", "svg"):
        target = out_dir / f"{path.stem}.{suffix}"
        figure.savefig(target, dpi=160 if suffix == "png" else None, bbox_inches="tight")
        exports.append({"path": str(target.resolve()), "sha256": file_sha256(target)})
    plt.close(figure)
    return exports
