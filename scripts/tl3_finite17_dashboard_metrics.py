from __future__ import annotations

import hashlib
import json
import math
from typing import Any

try:
    from scripts.tl3_finite17_dashboard_core import (
        VALID_GROUPS,
        VALID_ROWS,
        BridgeError,
        _context,
    )
except ModuleNotFoundError:
    from tl3_finite17_dashboard_core import (
        VALID_GROUPS,
        VALID_ROWS,
        BridgeError,
        _context,
    )


def _cohort_event(
    plan: dict[str, Any],
    node: dict[str, Any],
    row: dict[str, Any],
    train_groups: list[int],
    checkpoint_sha: str,
) -> tuple[int, dict[str, int | float], dict[str, Any]]:
    """Validate and map one complete paired VALID record to dashboard aliases."""
    epoch = row.get("epoch")
    if type(epoch) is not int or not 0 <= epoch < int(node["epochs"]):
        raise BridgeError(f"{node['id']}: VALID epoch is outside frozen schedule")
    if row.get("split") != f"vfold{VALID_GROUPS}" or row.get("selected_rows") != VALID_ROWS:
        raise BridgeError(f"{node['id']}: VALID does not cover official scored rows")
    prefix = sorted(train_groups)[:32]
    members_sha = hashlib.sha256(
        json.dumps([["train", group] for group in prefix], separators=(",", ":")).encode("ascii")
    ).hexdigest()
    if (
        row.get("train_eval_mask") != "all_after_warmup"
        or row.get("train_eval_sequences") != 32
        or row.get("train_eval_row_groups") != prefix
        or row.get("train_eval_members_sha256") != members_sha
        or type(row.get("train_eval_selected_rows")) is not int
        or row["train_eval_selected_rows"] <= 0
    ):
        raise BridgeError(f"{node['id']}: train-eval membership or mask differs")
    update = row.get("update_diagnostics", {}).get("cumulative", {})
    step = update.get("optimizer_attempts") if isinstance(update, dict) else None
    if type(step) is not int or step <= 0:
        raise BridgeError(f"{node['id']}: VALID lacks cumulative optimizer count")
    required_pairs = (
        ("train_eval_wp", "train_eval/wp"),
        ("train_eval_wp_t0", "train_eval/t0"),
        ("train_eval_wp_t1", "train_eval/t1"),
        ("train_eval_wmse", "train_eval/wmse"),
        ("weighted_pearson", "valid/wp"),
        ("global_t0", "valid/t0"),
        ("global_t1", "valid/t1"),
        ("valid_wmse", "valid/wmse"),
    )
    metrics: dict[str, int | float] = {}
    for source, target in required_pairs:
        value = row.get(source)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise BridgeError(f"{node['id']}: VALID lacks finite {source}")
        metrics[target] = value
    for source, target in (
        ("train_eval_wmse_t0", "train_eval/wmse_t0"),
        ("train_eval_wmse_t1", "train_eval/wmse_t1"),
        ("valid_wmse_t0", "valid/wmse_t0"),
        ("valid_wmse_t1", "valid/wmse_t1"),
    ):
        value = row.get(source)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            metrics[target] = value
    metrics.update(
        {
            "train_eval/loss": metrics["train_eval/wmse"],
            "valid/loss": metrics["valid/wmse"],
            "train_eval/score": metrics["train_eval/wp"],
            "train_eval/score/t0": metrics["train_eval/t0"],
            "train_eval/score/t1": metrics["train_eval/t1"],
            "valid/score": metrics["valid/wp"],
            "valid/score/t0": metrics["valid/t0"],
            "valid/score/t1": metrics["valid/t1"],
        }
    )
    summary = row.get("fit_summary")
    if isinstance(summary, dict):
        objective = summary.get("objective_mean")
        if (
            isinstance(objective, (int, float))
            and not isinstance(objective, bool)
            and math.isfinite(float(objective))
        ):
            metrics["train/objective_window_mean"] = objective
        rates = summary.get("learning_rate_by_group", [])
        for item in rates if isinstance(rates, list) else []:
            if isinstance(item, dict) and type(item.get("group")) is int:
                rate = item.get("mean")
                if (
                    isinstance(rate, (int, float))
                    and not isinstance(rate, bool)
                    and math.isfinite(float(rate))
                ):
                    metrics[f"optimizer/lr/group_{item['group']}/mean"] = rate
    context = {
        **_context(plan, node, checkpoint_sha),
        "completed_epoch": epoch + 1,
        "source_epoch_index": epoch,
        "valid": {
            "row_mask": "is_scored",
            "cohort_count": VALID_GROUPS,
            "selected_rows": VALID_ROWS,
        },
        "train_eval": {
            "row_mask": "all_after_warmup",
            "source": "train",
            "cohort_count": 32,
            "selected_rows": row["train_eval_selected_rows"],
            "row_groups": prefix,
            "members_sha256": members_sha,
        },
    }
    return step, metrics, context
