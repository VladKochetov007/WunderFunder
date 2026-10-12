#!/usr/bin/env python3
"""Run the frozen small retrospective public-score comparisons."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "endpoint_coverage_r1.csv"
CONTRACT = ROOT / "MODEL_CONTRACT_R1.md"
OUTPUT = ROOT / "MODEL_RESULTS_R1.json"
ALPHA = 10.0
LOCAL_SUBSETS = {
    "fold1_receipt_bound_float": {"2ebpj2er", "0srim121", "0hikjxpr"},
    "fold1_manifest_reported_float": {"19w7a96y"},
    "historical_368_source_reported": {"1bmxngta", "21svxfgi", "05ytuk8y", "1jdh5rtz", "1x8nyiwi"},
    "historical_368_ema_member0_proxy": {"1ykwb95u", "246401vj"},
}
FAMILY_CATEGORIES = (
    "fold1_student",
    "h388_h384_teacher",
    "h230_hft_student",
    "h230_hetero_c1_student",
    "oct04_combo",
    "d256_d07",
    "legacy_since_ema",
    "chunked_ensemble",
    "other_source_qualified",
    "unknown_source",
)
PRECISION_CATEGORIES = ("q8vnni", "i16_declared", "other_declared", "unknown")
FIT_CATEGORIES = ("all_data", "fold0", "fold1", "unknown")


class DisjointSet:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def collect_sources(node: Any) -> list[str]:
    if isinstance(node, dict):
        source = node.get("source")
        if isinstance(source, str) and source:
            return [source]
        return [source for value in node.values() for source in collect_sources(value)]
    if isinstance(node, list):
        if len(node) == 2 and isinstance(node[0], str) and isinstance(node[1], (int, float)):
            return [node[0]]
        return [source for value in node for source in collect_sources(value)]
    return []


def count_logical_specs(node: Any) -> int:
    if isinstance(node, dict):
        if isinstance(node.get("source"), str):
            return 1
        if "a" in node or "b" in node:
            return max(count_logical_specs(node.get("a")), count_logical_specs(node.get("b")))
        return sum(count_logical_specs(value) for value in node.values())
    if isinstance(node, list):
        if len(node) == 2 and isinstance(node[0], str) and isinstance(node[1], (int, float)):
            return 1
        return sum(count_logical_specs(value) for value in node)
    return 0


def parse_members(text: str) -> tuple[list[str], int | None, bool]:
    if not text:
        return [], None, False
    value = json.loads(text)
    sources = collect_sources(value)
    routed = isinstance(value, dict) and ("a" in value or "b" in value)
    count = count_logical_specs(value)
    return sources, count if count > 0 else None, routed


def family_for(row: dict[str, str], sources: list[str]) -> str:
    text = " ".join([row["package_alias"], *sources]).lower()
    if "chunk500" in text:
        return "chunked_ensemble"
    if "vt_f1" in text or "vt_f0" in text:
        return "fold1_student"
    if "vt_all_h388" in text:
        return "h388_h384_teacher"
    if "c1v2" in text or "hetero" in text:
        return "h230_hetero_c1_student"
    if "c20261004_combo" in text:
        return "oct04_combo"
    if "d256" in text or "d07" in text or "alpha07" in text:
        return "d256_d07"
    if "bd26_h230" in text or any(token in text for token in ("z1c", "z2c", "z4", "z5")):
        return "h230_hft_student"
    if any(token in text for token in ("ema", "since", "ab6", "ab7", "ab8")):
        return "legacy_since_ema"
    if sources:
        return "other_source_qualified"
    return "unknown_source"


def precision_for(row: dict[str, str]) -> str:
    value = row["precision_backend"].lower()
    if "q8vnni" in value:
        return "q8vnni"
    if "i16 declared by" in value:
        return "i16_declared"
    if "other declared" in value or "fp32 declared" in value:
        return "other_declared"
    return "unknown"


def fit_scope_for(row: dict[str, str]) -> str:
    value = row["fit_membership_and_exposure"].lower()
    if "all-data" in value or "all data" in value:
        return "all_data"
    if "fold0" in value:
        return "fold0"
    if "fold1" in value:
        return "fold1"
    return "unknown"


def source_tokens(sources: list[str]) -> list[str]:
    tokens: set[str] = set()
    for source in sources:
        tokens.update(part.strip() for part in source.split("|") if part.strip())
    return sorted(tokens)


def ancestry_groups(
    rows: list[dict[str, str]], row_sources: dict[str, list[str]]
) -> dict[str, str]:
    ids = [row["submission_id"] for row in rows]
    dsu = DisjointSet(ids)
    token_owner: dict[str, str] = {}
    unknown_ids: list[str] = []
    cluster_owner: dict[str, str] = {}
    for row in rows:
        sid = row["submission_id"]
        tokens = source_tokens(row_sources[sid])
        if tokens:
            for token in tokens:
                if token in token_owner:
                    dsu.union(sid, token_owner[token])
                else:
                    token_owner[token] = sid
        else:
            unknown_ids.append(sid)
        cluster = row["ancestry_cluster_for_validation"]
        if cluster and not cluster.startswith("unresolved/"):
            if cluster in cluster_owner:
                dsu.union(sid, cluster_owner[cluster])
            else:
                cluster_owner[cluster] = sid

    # Unknown endpoints are one conservative group: no independence is
    # assumed for artifacts whose members cannot be compared from receipts.
    for sid in unknown_ids[1:]:
        dsu.union(unknown_ids[0], sid)

    roots: dict[str, list[str]] = defaultdict(list)
    for sid in ids:
        roots[dsu.find(sid)].append(sid)
    ordered = sorted(roots.values(), key=lambda members: min(ids.index(sid) for sid in members))
    return {
        sid: f"source_group_{index:02d}"
        for index, members in enumerate(ordered, 1)
        for sid in members
    }


def make_features(
    rows: list[dict[str, str]], parsed: dict[str, tuple[list[str], int | None, bool]]
) -> tuple[list[str], np.ndarray]:
    timestamps = [datetime.strptime(row["display_timestamp"], "%Y-%m-%d %H:%M") for row in rows]
    age_days = np.array(
        [value.toordinal() + (value.hour * 60 + value.minute) / 1440.0 for value in timestamps],
        dtype=float,
    )
    counts = [parsed[row["submission_id"]][1] for row in rows]
    numeric = np.column_stack(
        [
            age_days,
            np.array([np.nan if count is None else float(count) for count in counts]),
            np.array([count is None for count in counts], dtype=float),
            np.array([parsed[row["submission_id"]][2] for row in rows], dtype=float),
        ]
    )
    names = [
        "display_age_days",
        "declared_logical_specs",
        "logical_spec_count_missing",
        "conditional_route",
    ]
    categorical_values = [
        [precision_for(row) for row in rows],
        [fit_scope_for(row) for row in rows],
        [family_for(row, parsed[row["submission_id"]][0]) for row in rows],
    ]
    categories = [PRECISION_CATEGORIES, FIT_CATEGORIES, FAMILY_CATEGORIES]
    for prefix, values, allowed in zip(
        ("precision", "fit_scope", "source_family"), categorical_values, categories, strict=True
    ):
        for category in allowed:
            numeric = np.column_stack(
                [numeric, np.array([value == category for value in values], dtype=float)]
            )
            names.append(f"{prefix}={category}")
    return names, numeric


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    train_imputed = x_train.copy()
    test_imputed = x_test.copy()
    for column in range(train_imputed.shape[1]):
        observed = train_imputed[:, column][np.isfinite(train_imputed[:, column])]
        fill = float(observed.mean()) if len(observed) else 0.0
        train_imputed[~np.isfinite(train_imputed[:, column]), column] = fill
        test_imputed[~np.isfinite(test_imputed[:, column]), column] = fill
    means = train_imputed.mean(axis=0)
    scales = train_imputed.std(axis=0)
    scales[scales < 1e-12] = 1.0
    train = (train_imputed - means) / scales
    test = (test_imputed - means) / scales
    intercept = float(y_train.mean())
    beta = np.linalg.solve(
        train.T @ train + ALPHA * np.eye(train.shape[1]), train.T @ (y_train - intercept)
    )
    return intercept + test @ beta


def metrics(observations: list[tuple[dict[str, str], float, float]]) -> dict[str, float | int]:
    errors = np.array(
        [prediction - float(row["public_score_outcome"]) for row, prediction, _ in observations]
    )
    group_errors: dict[str, list[float]] = defaultdict(list)
    for row, prediction, _ in observations:
        group_errors[row["validation_group"]].append(
            abs(prediction - float(row["public_score_outcome"]))
        )
    return {
        "n_rows": len(observations),
        "n_source_groups": len(group_errors),
        "mae": float(np.mean(np.abs(errors))) if len(errors) else math.nan,
        "rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else math.nan,
        "group_equal_mae": float(np.mean([np.mean(values) for values in group_errors.values()]))
        if group_errors
        else math.nan,
    }


def recent_mean(train_rows: list[dict[str, str]]) -> float:
    ordered = sorted(train_rows, key=lambda row: (row["display_timestamp"], row["submission_id"]))
    return float(np.mean([float(row["public_score_outcome"]) for row in ordered[-5:]]))


def run_grouped_cv(
    rows: list[dict[str, str]], x: np.ndarray, groups: dict[str, str]
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[groups[row["submission_id"]]].append(index)
    model_results: list[tuple[dict[str, str], float, float]] = []
    mean_results: list[tuple[dict[str, str], float, float]] = []
    split_notes = []
    for group, test_indices in grouped.items():
        test_set = set(test_indices)
        train_indices = [index for index in range(len(rows)) if index not in test_set]
        train_groups = {groups[rows[index]["submission_id"]] for index in train_indices}
        if len(train_indices) < 10 or len(train_groups) < 5:
            split_notes.append(
                {
                    "held_group": group,
                    "status": "insufficient_training_support",
                    "n_train": len(train_indices),
                    "n_test": len(test_indices),
                }
            )
            continue
        y_train = np.array([float(rows[index]["public_score_outcome"]) for index in train_indices])
        predictions = ridge_predict(x[train_indices], y_train, x[test_indices])
        mean_prediction = float(y_train.mean())
        for local_index, pred in zip(test_indices, predictions, strict=True):
            model_results.append((rows[local_index], float(pred), 0.0))
            mean_results.append((rows[local_index], mean_prediction, 0.0))
        split_notes.append(
            {
                "held_group": group,
                "status": "scored",
                "n_train": len(train_indices),
                "n_train_groups": len(train_groups),
                "n_test": len(test_indices),
            }
        )
    return {
        "model": metrics(model_results),
        "mean_baseline": metrics(mean_results),
        "split_details": split_notes,
    }


def partition(values: list[str], count: int) -> list[list[str]]:
    bins: list[list[str]] = [[] for _ in range(count)]
    for index, value in enumerate(values):
        bins[min(count - 1, index * count // max(1, len(values)))].append(value)
    return [items for items in bins if items]


def run_chronological_cv(
    rows: list[dict[str, str]], x: np.ndarray, groups: dict[str, str]
) -> dict[str, Any]:
    unique_times = sorted({row["display_timestamp"] for row in rows})
    warmup_count = max(10, math.ceil(len(unique_times) / 2))
    test_blocks = partition(unique_times[warmup_count:], 3)
    model_results: list[tuple[dict[str, str], float, float]] = []
    mean_results: list[tuple[dict[str, str], float, float]] = []
    recent_results: list[tuple[dict[str, str], float, float]] = []
    split_notes = []
    for block_index, block in enumerate(test_blocks, 1):
        test_indices = [
            index for index, row in enumerate(rows) if row["display_timestamp"] in set(block)
        ]
        test_groups = {groups[rows[index]["submission_id"]] for index in test_indices}
        first_test = min(block)
        train_indices = [
            index
            for index, row in enumerate(rows)
            if row["display_timestamp"] < first_test
            and groups[row["submission_id"]] not in test_groups
        ]
        train_groups = {groups[rows[index]["submission_id"]] for index in train_indices}
        if len(train_indices) < 10 or len(train_groups) < 5:
            split_notes.append(
                {
                    "block": block_index,
                    "timestamps": block,
                    "status": "insufficient_training_support",
                    "n_train": len(train_indices),
                    "n_test": len(test_indices),
                }
            )
            continue
        y_train = np.array([float(rows[index]["public_score_outcome"]) for index in train_indices])
        model_predictions = ridge_predict(x[train_indices], y_train, x[test_indices])
        mean_prediction = float(y_train.mean())
        recent_prediction = recent_mean([rows[index] for index in train_indices])
        for local_index, pred in zip(test_indices, model_predictions, strict=True):
            model_results.append((rows[local_index], float(pred), 0.0))
            mean_results.append((rows[local_index], mean_prediction, 0.0))
            recent_results.append((rows[local_index], recent_prediction, 0.0))
        split_notes.append(
            {
                "block": block_index,
                "timestamps": block,
                "status": "scored",
                "n_train": len(train_indices),
                "n_train_groups": len(train_groups),
                "n_test": len(test_indices),
                "n_test_groups": len(test_groups),
                "recent_baseline_artifact_count": min(5, len(train_indices)),
            }
        )
    return {
        "model": metrics(model_results),
        "mean_baseline": metrics(mean_results),
        "recent_five_baseline": metrics(recent_results),
        "warmup_unique_timestamps": warmup_count,
        "total_unique_timestamps": len(unique_times),
        "split_details": split_notes,
    }


def local_diagnostics(rows: list[dict[str, str]], groups: dict[str, str]) -> dict[str, Any]:
    by_id = {row["submission_id"]: row for row in rows}
    result: dict[str, Any] = {}
    for name, ids in LOCAL_SUBSETS.items():
        selected = [by_id[sid] for sid in ids if sid in by_id and by_id[sid]["local_metric"]]
        group_ids = sorted({groups[row["submission_id"]] for row in selected})
        offsets = [
            float(row["public_score_outcome"]) - float(row["local_metric"]) for row in selected
        ]
        enough_groups = len(group_ids) >= 4
        result[name] = {
            "n_rows": len(selected),
            "n_source_groups": len(group_ids),
            "submission_ids": [row["submission_id"] for row in selected],
            "mean_local_metric": float(np.mean([float(row["local_metric"]) for row in selected]))
            if selected
            else None,
            "mean_public_score": float(
                np.mean([float(row["public_score_outcome"]) for row in selected])
            )
            if selected
            else None,
            "mean_public_minus_local_offset_descriptive_only": float(np.mean(offsets))
            if offsets
            else None,
            "grouped_ridge_alpha10_status": "insufficient independent groups for useful grouped validation"
            if not enough_groups
            else "eligible",
            "ridge_grouped_cv_performed": False,
            "qualification": "retrospective/exposed; not a prospective transfer correction",
        }
    return result


def main() -> None:
    with INPUT.open(newline="", encoding="utf-8") as stream:
        all_rows = list(csv.DictReader(stream))
    rows = [row for row in all_rows if row["model_target_usable"] == "True"]
    for row in rows:
        if row["portal_status"] != "SCORE" or not row["public_score_outcome"]:
            raise ValueError("score-target filter admitted a censored or blank row")
        row["_timestamp"] = datetime.strptime(
            row["display_timestamp"], "%Y-%m-%d %H:%M"
        ).isoformat()

    parsed = {row["submission_id"]: parse_members(row["member_specs_and_weights"]) for row in rows}
    groups = ancestry_groups(rows, {sid: value[0] for sid, value in parsed.items()})
    for row in rows:
        row["validation_group"] = groups[row["submission_id"]]
    feature_names, x = make_features(rows, parsed)
    scores = np.array([float(row["public_score_outcome"]) for row in rows])
    grouped = run_grouped_cv(rows, x, groups)
    chronological = run_chronological_cv(rows, x, groups)
    local = local_diagnostics(all_rows, groups)
    all_ordered = sorted(rows, key=lambda row: (row["display_timestamp"], row["submission_id"]))
    result = {
        "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "input_coverage_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "target_rows": len(rows),
        "excluded_non_score_rows": len(all_rows) - len(rows),
        "alpha": ALPHA,
        "loss": "unweighted squared error plus alpha * squared L2 coefficients; intercept unpenalized",
        "features": feature_names,
        "source_component_count": len(set(groups.values())),
        "source_component_sizes": dict(
            sorted(
                {
                    group: list(groups.values()).count(group) for group in set(groups.values())
                }.items()
            )
        ),
        "source_component_by_submission": groups,
        "full_history_score_mean_descriptive": float(scores.mean()),
        "last_five_artifact_score_mean_descriptive": float(
            np.mean([float(row["public_score_outcome"]) for row in all_ordered[-5:]])
        ),
        "grouped_component_cv": grouped,
        "chronological_purged_cv": chronological,
        "local_metric_subsets": local,
        "interpretation_limit": "All feature reconstruction and backtests are retrospective. Results are diagnostics only; no current ZIP public score is forecast or promoted.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "target_rows",
                    "source_component_count",
                    "full_history_score_mean_descriptive",
                    "last_five_artifact_score_mean_descriptive",
                    "grouped_component_cv",
                    "chronological_purged_cv",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
