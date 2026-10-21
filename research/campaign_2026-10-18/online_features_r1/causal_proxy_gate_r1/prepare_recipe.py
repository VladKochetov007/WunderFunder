from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def prepare(root: Path, output: Path, proxy_predictions: Path) -> dict[str, Any]:
    proxy_root = root / "research/campaign_2026-10-18/online_features_r1/causal_label_proxy_r1"
    proxy_recipe_path = proxy_root / "recipe.json"
    proxy_grade_path = proxy_root / "independent_grade_r1.json"
    proxy_execution_path = proxy_root / "execution_receipt_r1.json"
    proxy_fit_path = proxy_root / "fit_receipt_r1.json"
    proxy_recipe = _read_json(proxy_recipe_path)
    proxy_grade = _read_json(proxy_grade_path)
    groups = proxy_recipe["selection"]["evaluation_groups"]
    group_ids = [int(item["row_group"]) for item in groups]
    components = [int(item["component"]) for item in groups]
    sequence_ids = [int(item["seq_ix"]) for item in groups]
    if len(set(group_ids)) != len(group_ids) or len(set(components)) != len(components):
        raise ValueError("proxy evaluation groups/components must be unique")
    if proxy_grade["status"] != "independent_metric_regrade_passed":
        raise ValueError("proxy independent grade has not passed")
    if proxy_grade["recipe_sha256"] != sha256_file(proxy_recipe_path):
        raise ValueError("proxy recipe hash differs from its independent grade")
    if proxy_grade["prediction_sha256"] != sha256_file(proxy_predictions):
        raise ValueError("proxy prediction file differs from its independent grade")

    split_hash = proxy_recipe["selection"]["split_manifest_sha256"]
    members = {
        "tl3": {
            "prediction": root / "cache/pred_tl3_unknown_ancestry_f0_valid.npy",
            "sidecar": root / "cache/pred_tl3_unknown_ancestry_f0_valid.json",
        },
        "r3": {
            "prediction": root / "cache/pred_geometry_r3_reverse_half_q16x_20261018_v2_valid.npy",
            "sidecar": root / "cache/pred_geometry_r3_reverse_half_q16x_20261018_v2_valid.json",
        },
    }
    source_members: dict[str, dict[str, Any]] = {}
    cache_indices: dict[int, int] | None = None
    for name, paths in members.items():
        prediction_path = paths["prediction"]
        sidecar_path = paths["sidecar"]
        sidecar = _read_json(sidecar_path)
        mapped = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
        if mapped.shape != (468, 20_000, 2) or mapped.dtype != np.float32:
            raise ValueError(
                f"{name} prediction shape or dtype changed: {mapped.shape}/{mapped.dtype}"
            )
        if sidecar["split_manifest_sha256"] != split_hash:
            raise ValueError(f"{name} split hash differs from the proxy recipe")
        mapping = {int(group): index for index, group in enumerate(sidecar["row_groups"])}
        seq_by_group = {
            int(group): int(seq_ix)
            for group, seq_ix in zip(sidecar["row_groups"], sidecar["sequence_ids"], strict=True)
        }
        if len(mapping) != 468 or len(seq_by_group) != 468:
            raise ValueError(f"{name} sidecar has duplicate or incomplete source groups")
        if any(
            seq_by_group[group] != seq for group, seq in zip(group_ids, sequence_ids, strict=True)
        ):
            raise ValueError(f"{name} row-group/sequence identity differs from the proxy recipe")
        current_indices = {group: mapping[group] for group in group_ids}
        if cache_indices is not None and current_indices != cache_indices:
            raise ValueError("expert caches use different row-group indices")
        cache_indices = current_indices
        source_members[name] = {
            "prediction_path": prediction_path.relative_to(root).as_posix(),
            "prediction_sha256": sha256_file(prediction_path),
            "prediction_shape": list(mapped.shape),
            "prediction_dtype": str(mapped.dtype),
            "sidecar_path": sidecar_path.relative_to(root).as_posix(),
            "sidecar_sha256": sha256_file(sidecar_path),
            "zip_sha256": sidecar["zip_sha256"],
            "prediction_sha256_in_sidecar": sidecar["prediction_sha256"],
            "ancestry": sidecar.get("ancestry", sidecar.get("scope", "not declared")),
        }
        if source_members[name]["prediction_sha256"] != sidecar["prediction_sha256"]:
            raise ValueError(f"{name} prediction cache hash differs from its sidecar")

    assert cache_indices is not None
    gate_configs = [
        {"id": "proxy_gate_h256", "half_life_steps": 256},
        {"id": "proxy_gate_h2048", "half_life_steps": 2048},
        {"id": "proxy_gate_h8192", "half_life_steps": 8192},
    ]
    recipe = {
        "schema": "wnn.causal_proxy_gate_recipe.v1",
        "id": "E-ONLINE-CAUSAL-PROXY-GATE-20261018-R1",
        "hypothesis_id": "H-ONLINE-DELAYED-PROXY-20261018-R1",
        "code": {
            "recipe_builder_sha256": sha256_file(Path(__file__).resolve()),
            "prediction_builder_sha256": sha256_file(
                Path(__file__).with_name("build_predictions.py")
            ),
            "independent_grader_sha256": sha256_file(Path(__file__).with_name("grade_gate.py")),
        },
        "question": "Does a feature-derived previous-row-label pseudo-target provide useful causal online weighting between frozen TL3 and R3 experts? Target maturity is unknown.",
        "metric_id": "global_wp_clip2_v1",
        "alignment": {
            "proxy_prediction": "At row t, proxy output estimates clipped y[t-1] from x[t] and x[t]-x[t-1].",
            "update": "Compare proxy output at t with each expert prediction at t-1; update before mixing expert predictions for row t.",
            "first_scored_row": 99,
            "first_row_rule": "Use fixed 75% TL3 / 25% R3 because no following feature row is available yet.",
            "last_scored_row": 19_999,
            "state_reset": "Reset every gate state at each sequence boundary.",
            "labels_and_masks": "Never read by recipe builder or prediction builder; grader-only.",
        },
        "proxy": {
            "recipe_path": proxy_recipe_path.relative_to(root).as_posix(),
            "recipe_sha256": sha256_file(proxy_recipe_path),
            "grade_path": proxy_root.joinpath("independent_grade_r1.json")
            .relative_to(root)
            .as_posix(),
            "grade_sha256": sha256_file(proxy_grade_path),
            "execution_receipt_path": proxy_execution_path.relative_to(root).as_posix(),
            "execution_receipt_sha256": sha256_file(proxy_execution_path),
            "fit_receipt_path": proxy_fit_path.relative_to(root).as_posix(),
            "fit_receipt_sha256": sha256_file(proxy_fit_path),
            "runner_sha256": _read_json(proxy_execution_path)["code"]["runner_sha256"],
            "predictions_path": str(proxy_predictions.resolve()),
            "predictions_sha256": sha256_file(proxy_predictions),
            "rows_per_sequence": 19_900,
            "fit_components": sorted(
                {int(item["component"]) for item in proxy_recipe["selection"]["fit_groups"]}
            ),
            "fit_membership_sha256": hashlib.sha256(
                json.dumps(proxy_recipe["selection"]["fit_groups"], sort_keys=True).encode()
            ).hexdigest(),
        },
        "expert_sources": source_members,
        "selection": {
            "split_manifest_path": proxy_recipe["selection"]["split_manifest"],
            "split_manifest_sha256": split_hash,
            "groups": [
                {
                    **item,
                    "tl3_cache_index": cache_indices[int(item["row_group"])],
                    "r3_cache_index": cache_indices[int(item["row_group"])],
                    "proxy_offset": index * 19_900,
                    "proxy_rows": 19_900,
                }
                for index, item in enumerate(groups)
            ],
            "components": components,
            "exposure": "32 component-disjoint F0 groups; all have been repeatedly exposed to prior model and feature selection. TL3 fitting ancestry is unknown.",
        },
        "endpoints": ["tl3", "r3", "fixed_75_tl3_25_r3", *[item["id"] for item in gate_configs]],
        "controls": {"fixed_75_tl3_25_r3": {"tl3_weight": 0.75, "r3_weight": 0.25}},
        "gate": {
            "initial_r3_weight": 0.25,
            "prior_min_r3_weight": 0.05,
            "prior_max_r3_weight": 0.50,
            "eta": 2.0,
            "warmup_observations": 128,
            "target_count": 2,
            "feedback_target": "clip(proxy_prediction[t], -2, 2)",
            "feedback_prediction": "clip(expert_prediction[t-1], -2, 2)",
            "pseudo_weight": "abs(clipped_proxy_target)",
            "loss": "EMA of pseudo_weight * squared_error; divide by EMA(pseudo_weight * max(proxy_target^2, 0.01))",
            "weight_rule": "sigmoid(logit(0.25) - eta * (normalized_R3_loss - normalized_TL3_loss)); clip result to [0.05, 0.50]",
            "gate_configs": gate_configs,
        },
        "interpretation": {
            "positive": "A useful conditional endpoint improvement justifies cross-fold/source-qualified replication and a causal serving prototype; it is not clean-CV or public-transfer evidence.",
            "negative_or_uncertain": "Retain the broader online-learning direction. This screen tests one linear proxy, two fixed experts, three predeclared timescales, one exposed component panel, and one inherited prediction cache.",
            "b300_draft_gate": "Prepare an additional B300 draft only if the fixed endpoint gate shows meaningful causal complementarity and an implementable no-label callback contract. Resolve target maturity and TL3 ancestry before claiming realized online learning.",
        },
        "resources": {
            "gpu": False,
            "cpu_worker_cores": 1,
            "memory_reservation_bytes": 2_000_000_000,
            "swap_bytes": 0,
            "scratch_root": "/home/vlad/development/wunder/research_tmp",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
    return recipe


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a label-blind causal expert-gate recipe.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--proxy-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(
        args.root.resolve(strict=True),
        args.output.resolve(),
        args.proxy_predictions.resolve(strict=True),
    )


if __name__ == "__main__":
    main()
