from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from wnn.multistage_search import read_json


def _score(result: dict[str, Any]) -> float | None:
    objective = result.get("objective")
    evaluation = objective.get("evaluation") if isinstance(objective, dict) else None
    nested = objective.get("objective") if isinstance(objective, dict) else None
    value = nested.get("value") if isinstance(nested, dict) else None
    endpoint = result.get("selected_endpoint")
    role = "fixed_final" if result.get("stage") == 1 else "configured_final"
    if (
        result.get("status") != "COMPLETE"
        or result.get("primary_valid_used", False) is not False
        or not isinstance(endpoint, dict)
        or endpoint.get("selection_role") != role
        or not isinstance(evaluation, dict)
        or evaluation.get("scope") != "inner_train_proxy"
        or evaluation.get("source") != "train"
        or evaluation.get("row_mask") != "all_after_warmup AND row_index>=99"
        or evaluation.get("is_scored_used") is not False
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def select_stage4_candidates(run_dir: Path) -> tuple[list[str], list[str]]:
    """Choose the reviewed direct, fine-tuned, matched KD, and fallback finalists."""
    paths = set((run_dir / "finalized").rglob("trial_result.json"))
    for stage in (2, 3):
        paths.update((run_dir / f"stage{stage}" / "trials").glob("*/trial_result.json"))
    rows = []
    for path in sorted(paths):
        if path.is_file() and not path.is_symlink():
            try:
                result = read_json(path)
                score = _score(result)
                if score is not None and result.get("stage") in (1, 2, 3):
                    rows.append((path.resolve(), result, score))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
    chosen: list[tuple[str, Path, dict[str, Any]]] = []
    hashes: set[str] = set()

    def add(name: str, row: tuple[Path, dict[str, Any], float] | None) -> None:
        if row is None:
            return
        path, result, _ = row
        digest = result.get("fit_model_sha256")
        if isinstance(digest, str) and digest not in hashes:
            hashes.add(digest)
            chosen.append((name, path, result))

    def best(test: Any) -> tuple[Path, dict[str, Any], float] | None:
        options = [row for row in rows if test(row[1])]
        return max(options, key=lambda row: row[2]) if options else None

    def family(result: dict[str, Any]) -> str:
        return str(result.get("family", "")).lower()

    add(
        "direct_gru",
        best(lambda r: r.get("stage") == 1 and "gru" in family(r) and "galam" not in family(r)),
    )
    add(
        "direct_lstm",
        best(lambda r: r.get("stage") == 1 and "lstm" in family(r) and "galam" not in family(r)),
    )
    add("transformer", best(lambda r: r.get("stage") == 1 and "transformer" in family(r)))
    add("finetune_gru", best(lambda r: r.get("stage") == 2 and "gru" in family(r)))
    add("finetune_lstm", best(lambda r: r.get("stage") == 2 and "lstm" in family(r)))
    student = best(lambda r: r.get("stage") == 3 and r.get("pair_role") == "kd_student")
    if student is not None:
        pair = student[1].get("pair_id")
        add(
            "kd_label_control",
            best(
                lambda r: (
                    r.get("stage") == 3
                    and r.get("pair_role") == "label_only_control"
                    and r.get("pair_id") == pair
                )
            ),
        )
        add("kd_student", student)
    fallback = sorted(
        (row for row in rows if row[1].get("stage") == 1 and "galam" not in family(row[1])),
        key=lambda row: row[2],
        reverse=True,
    )
    add(
        "direct_other",
        next((row for row in fallback if row[1].get("fit_model_sha256") not in hashes), None),
    )
    required = {
        "direct_gru",
        "direct_lstm",
        "transformer",
        "finetune_gru",
        "finetune_lstm",
        "kd_student",
        "kd_label_control",
        "direct_other",
    }
    names = {item[0] for item in chosen}
    return [f"{name}={path}" for name, path, _ in chosen[:8]], sorted(required - names)
