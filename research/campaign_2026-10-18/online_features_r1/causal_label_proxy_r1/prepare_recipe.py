from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _select_groups(
    sequences: list[dict[str, int]], allowed: list[int], seed: int, count: int
) -> tuple[list[dict[str, int]], int]:
    allowed_set = set(map(int, allowed))
    by_component: dict[int, list[dict[str, int]]] = {}
    for item in sequences:
        group = int(item["row_group"])
        if group in allowed_set:
            by_component.setdefault(int(item["component"]), []).append(
                {key: int(item[key]) for key in ("component", "row_group", "seq_ix")}
            )
    components = np.asarray(sorted(by_component), dtype=np.int64)
    if components.size < count:
        raise ValueError(f"only {components.size} eligible components; need {count}")
    rng = np.random.default_rng(seed)
    selected_components = sorted(map(int, rng.choice(components, count, replace=False)))
    selected = [
        min(by_component[component], key=lambda item: item["row_group"])
        for component in selected_components
    ]
    return selected, int(components.size)


def prepare_recipe(manifest_path: Path, output_path: Path) -> None:
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    train_groups, train_components = _select_groups(
        manifest["sequences"]["train"], manifest["fit"]["train"], 20261018, 32
    )
    eval_groups, eval_components = _select_groups(
        manifest["sequences"]["valid"], manifest["evaluation"]["valid"], 20261019, 32
    )
    train_component_ids = {item["component"] for item in train_groups}
    eval_component_ids = {item["component"] for item in eval_groups}
    if train_component_ids & eval_component_ids:
        raise ValueError("selected fit and evaluation components overlap")
    sources = manifest["sources"]
    recipe = {
        "schema_version": 1,
        "id": "E-ONLINE-LAGGED-TARGET-PROXY-20261018-R1",
        "hypothesis_id": "H-ONLINE-DELAYED-PROXY-20261018-R1",
        "question": "Can current observable features predict the previous-row label as a feature-only lagged pseudo-label? Target maturity is unknown, and no expert reweighting is tested here.",
        "metric_id": "global_wp_clip2_v1",
        "target_alignment": "At current step t >= 100, predict clipped previous-row label y[t-1] from x[t] and x[t]-x[t-1]. This is a one-row proxy alignment only: the target horizon and maturity at t are undocumented. The predictor never reads target or scoring mask at inference.",
        "features": "All 112 current raw features followed by their one-step differences; no sequence ID, future row, target, or mask is an input.",
        "model": {
            "kind": "two-output weighted ridge regression",
            "target": "clip(y[t-1], -2, 2)",
            "sample_weight": "abs(clip(y[t-1], -2, 2)) separately per target",
            "standardization": "weighted fit-sample mean and standard deviation, fit groups only; use stable centered weighted batch moments",
            "ridge_alpha": 0.1,
            "fit_step_stride": 10,
            "evaluation_step_stride": 1,
        },
        "selection": {
            "split_manifest": "research/campaign_2026-09-29/artifacts/splits/fold0.json",
            "split_manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "fit_seed": 20261018,
            "evaluation_seed": 20261019,
            "selection_rule": "Uniformly sample 32 distinct eligible component IDs with NumPy default_rng, then use the lowest row-group ID within each sampled component.",
            "eligible_fit_component_count": train_components,
            "eligible_eval_component_count": eval_components,
            "fit_groups": train_groups,
            "evaluation_groups": eval_groups,
            "fit_data_identity": sources["datasets/train.parquet"],
            "evaluation_data_identity": sources["datasets/valid.parquet"],
            "exposure": "F0 fit groups are train-only; F0 evaluation components have been repeatedly inspected and are exposed development data.",
        },
        "resources": {
            "cpu_cores": 1,
            "memory_reservation_bytes": 1500000000,
            "gpu": False,
            "swap_bytes": 0,
            "physical_output_root": "/home/vlad/development/wunder/research_tmp/online_features_r1/causal_label_proxy_r1",
            "fit_endpoints": 1,
        },
        "interpretation": {
            "success": "A useful positive lagged-proxy WP would justify a separate frozen test of a proxy-driven expert-weight update; it would not establish mature feedback or a competition score gain.",
            "weak_result": "Defer this exact proxy-weighting route, but keep covariate-only online adaptation and learned feature extraction open.",
            "uncertainty": "Component bootstrap is conditional on one proxy fit and 32 exposed evaluation components; it does not cover training variation or public transfer.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze source-qualified causal proxy membership.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_recipe(args.manifest.resolve(strict=True), args.output)


if __name__ == "__main__":
    main()
