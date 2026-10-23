from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sequence_oracle_sources import (
    endpoint_moments,
    one_target_wp,
    read_json,
    sha256_file,
    source_inputs,
    write_json,
)

from wnn.evaluation import paired_component_bootstrap
from wnn.gmetric import METRIC_ID, GlobalAccumulator


def grade(
    root: Path, recipe_path: Path, output_path: Path, limit_groups: int | None
) -> dict[str, Any]:
    """Run either a bounded shape profile or the frozen full-panel diagnostic."""
    started_at = time.monotonic()
    recipe = read_json(recipe_path)
    recipe_bytes = recipe_path.read_bytes()
    sources, group_records = source_inputs(root, recipe)
    if limit_groups is not None:
        if not 1 <= limit_groups <= len(group_records):
            raise ValueError("profile group limit is outside the frozen panel")
        group_records = group_records[:limit_groups]
    elif len(group_records) != int(recipe["panel"]["groups"]):
        raise ValueError("full grade requires the complete frozen evaluation panel")

    target_spec = recipe["targets"]
    target_path = root / target_spec["path"]
    target_stat = target_path.stat()
    if target_stat.st_size != int(target_spec["size_bytes"]):
        raise ValueError("target cache size changed after recipe freeze")
    if target_stat.st_mtime_ns != int(target_spec["mtime_ns"]):
        raise ValueError("target cache mtime changed after recipe freeze")
    target_sha256 = sha256_file(target_path)
    with np.load(target_path, allow_pickle=False) as archive:
        target_groups = np.asarray(archive["row_group"], dtype=np.int64)
        target_seq_ix = np.asarray(archive["seq_ix"], dtype=np.int64)
        if len(set(map(int, target_groups))) != len(target_groups):
            raise ValueError("target cache contains duplicate row groups")
        target_position = {int(group): index for index, group in enumerate(target_groups)}
        y_all = np.asarray(archive["y"], dtype=np.float32)
        mask_all = np.asarray(archive["mask"], dtype=bool)
    expected_y_shape = (int(target_spec["sequences"]), int(recipe["panel"]["steps"]), 2)
    expected_mask_shape = expected_y_shape[:2]
    if y_all.shape != expected_y_shape or mask_all.shape != expected_mask_shape:
        raise ValueError(f"target cache shape changed: {y_all.shape}/{mask_all.shape}")
    target_seq_by_group = {
        int(group): int(seq_ix) for group, seq_ix in zip(target_groups, target_seq_ix, strict=True)
    }

    steps = int(recipe["panel"]["steps"])
    early_start, early_stop = map(int, recipe["early_window"])
    late_start, late_stop = map(int, recipe["late_window"])
    score_start = int(recipe["scoring_start"])
    positions = np.arange(steps)
    early_accumulators = {name: GlobalAccumulator() for name in recipe["endpoints"]}
    late_accumulators = {name: GlobalAccumulator() for name in recipe["endpoints"]}
    endpoint_stats = {
        name: np.empty((len(group_records), 2, 6), dtype=np.float64) for name in recipe["endpoints"]
    }
    early_choices: list[list[bool]] = [[], []]
    late_choices: list[list[bool]] = [[], []]
    early_margins: list[list[float]] = [[], []]
    selector_agreements: list[list[bool]] = [[], []]
    selected_rows = {"early": 0, "late": 0}

    for group_index, group in enumerate(group_records):
        row_group = int(group["row_group"])
        if row_group not in target_position:
            raise ValueError(f"target cache is missing row group {row_group}")
        target_index = target_position[row_group]
        if target_seq_by_group[row_group] != int(group["seq_ix"]):
            raise ValueError(f"target sequence identity mismatch for row group {row_group}")
        target = y_all[target_index]
        scoring_mask = mask_all[target_index] & (positions >= score_start)
        early_mask = scoring_mask[early_start:early_stop]
        late_mask = scoring_mask[late_start:late_stop]
        if not early_mask.any() or not late_mask.any():
            raise ValueError(f"row group {row_group} has no scored rows in a frozen window")
        selected_rows["early"] += int(early_mask.sum())
        selected_rows["late"] += int(late_mask.sum())

        tl3 = np.asarray(sources["tl3"][int(group["tl3_index"])], dtype=np.float32)
        r3 = np.asarray(sources["r3"][int(group["r3_index"])], dtype=np.float32)
        early_tl3 = tl3[early_start:early_stop]
        early_r3 = r3[early_start:early_stop]
        late_tl3 = tl3[late_start:late_stop]
        late_r3 = r3[late_start:late_stop]

        early_choice = np.zeros(2, dtype=bool)
        late_choice = np.zeros(2, dtype=bool)
        for target_index_local in range(2):
            early_tl3_wp = one_target_wp(
                target[early_start:early_stop, target_index_local],
                early_tl3[:, target_index_local],
                early_mask,
            )
            early_r3_wp = one_target_wp(
                target[early_start:early_stop, target_index_local],
                early_r3[:, target_index_local],
                early_mask,
            )
            late_tl3_wp = one_target_wp(
                target[late_start:late_stop, target_index_local],
                late_tl3[:, target_index_local],
                late_mask,
            )
            late_r3_wp = one_target_wp(
                target[late_start:late_stop, target_index_local],
                late_r3[:, target_index_local],
                late_mask,
            )
            early_choice[target_index_local] = early_r3_wp > early_tl3_wp
            late_choice[target_index_local] = late_r3_wp > late_tl3_wp
            early_choices[target_index_local].append(bool(early_choice[target_index_local]))
            late_choices[target_index_local].append(bool(late_choice[target_index_local]))
            early_margins[target_index_local].append(float(early_r3_wp - early_tl3_wp))
            selector_agreements[target_index_local].append(
                bool(early_choice[target_index_local] == late_choice[target_index_local])
            )

        early_route = np.where(early_choice[None, :], late_r3, late_tl3).astype(np.float32)
        late_route = np.where(late_choice[None, :], late_r3, late_tl3).astype(np.float32)
        early_endpoints = {
            "tl3": early_tl3,
            "r3": early_r3,
            "fixed_75_tl3_25_r3": 0.75 * early_tl3 + 0.25 * early_r3,
            "early_label_selector": np.where(early_choice[None, :], early_r3, early_tl3).astype(
                np.float32
            ),
            "late_hindsight_selector": np.where(late_choice[None, :], early_r3, early_tl3).astype(
                np.float32
            ),
        }
        late_endpoints = {
            "tl3": late_tl3,
            "r3": late_r3,
            "fixed_75_tl3_25_r3": 0.75 * late_tl3 + 0.25 * late_r3,
            "early_label_selector": early_route,
            "late_hindsight_selector": late_route,
        }
        for name in recipe["endpoints"]:
            early_accumulators[name].add(
                target[early_start:early_stop],
                early_endpoints[name],
                early_mask,
            )
            late_accumulators[name].add(
                target[late_start:late_stop], late_endpoints[name], late_mask
            )
            endpoint_stats[name][group_index] = endpoint_moments(
                target[late_start:late_stop],
                late_endpoints[name],
                late_mask,
            )

    if limit_groups is not None:
        profile = {
            "schema": "wnn.sequence_oracle.profile.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
            "target_cache_sha256": target_sha256,
            "processed_groups": len(group_records),
            "panel_groups": int(recipe["panel"]["groups"]),
            "selected_rows": selected_rows,
            "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * 1024,
            "elapsed_seconds": time.monotonic() - started_at,
            "profile_only": True,
            "scientific_scores": None,
        }
        write_json(output_path, profile)
        return profile

    components = np.asarray([int(group["component"]) for group in group_records], dtype=np.int64)
    early_scores = {name: accumulator.result() for name, accumulator in early_accumulators.items()}
    late_scores = {name: accumulator.result() for name, accumulator in late_accumulators.items()}
    comparisons = {}
    for reference in ("tl3", "fixed_75_tl3_25_r3"):
        comparisons[f"early_label_selector_vs_{reference}"] = paired_component_bootstrap(
            endpoint_stats["early_label_selector"],
            endpoint_stats[reference],
            components,
            n_boot=int(recipe["bootstrap"]["replicates"]),
            seed=int(recipe["bootstrap"]["seed"]),
            centered=True,
        )
    denominator = len(group_records)
    selector = {
        f"target_{target_index}": {
            "early_r3_choice_rate": float(np.mean(early_choices[target_index])),
            "late_r3_choice_rate": float(np.mean(late_choices[target_index])),
            "early_late_choice_agreement": float(np.mean(selector_agreements[target_index])),
            "early_r3_minus_tl3_local_wp_median": float(np.median(early_margins[target_index])),
            "early_r3_minus_tl3_local_wp_mean": float(np.mean(early_margins[target_index])),
            "sequence_count": denominator,
        }
        for target_index in range(2)
    }
    result = {
        "schema": "wnn.sequence_oracle.result.v1",
        "id": recipe["id"],
        "hypothesis_id": recipe["hypothesis_id"],
        "created_utc": datetime.now(UTC).isoformat(),
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "grader_sha256": sha256_file(Path(__file__).resolve()),
        "target_cache_sha256": target_sha256,
        "metric_id": METRIC_ID,
        "panel": {
            "groups": len(group_records),
            "components": int(len(np.unique(components))),
            "early_scored_rows": selected_rows["early"],
            "late_scored_rows": selected_rows["late"],
            "exposure": recipe["panel"]["exposure"],
        },
        "early_window": {"start_inclusive": early_start, "stop_exclusive": early_stop},
        "late_window": {"start_inclusive": late_start, "stop_exclusive": late_stop},
        "scores": {"early": early_scores, "late": late_scores},
        "selector_diagnostics": selector,
        "paired_component_bootstrap": comparisons,
        "interpretation_limits": recipe["interpretation_limits"],
        "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "elapsed_seconds": time.monotonic() - started_at,
    }
    write_json(output_path, result)
    return result


def main() -> int:
    """Run a frozen source profile or the full exposed-panel analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-groups", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    recipe_path = args.recipe if args.recipe.is_absolute() else root / args.recipe
    output_path = args.output if args.output.is_absolute() else root / args.output
    result = grade(root, recipe_path, output_path, args.profile_groups)
    print(
        json.dumps(
            {
                key: result[key]
                for key in result
                if key
                in {
                    "schema",
                    "panel",
                    "scores",
                    "elapsed_seconds",
                    "peak_process_rss_bytes",
                    "processed_groups",
                }
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
