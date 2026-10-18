"""Grade a frozen recurrent-feature pair on the fixed F0 evaluation panel."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from grade_prediction_bank import SCORED_FROM_ROW, Lazy, aligned, score, sha256  # noqa: E402

from wnn.evaluation import paired_component_bootstrap, paired_component_moments  # noqa: E402

PACKET = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
TARGETS_PATH = ROOT / "cache/valid_targets.npz"
ACTIVE = "kalman_groupfast_active_f0_s3_r1"
CONTROL = "kalman_groupfast_zero_f0_s3_r1"
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 20261018
TIME_BINS = ((99, 500), (500, 2_000), (2_000, 10_000), (10_000, 20_000))


def _load_prediction(
    name: str, groups: list[int], split_sha256: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a row-aligned prediction only after validating its feature-only sidecar."""
    path = ROOT / "cache" / f"pred_{name}_valid.npy"
    sidecar_path = path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("campaign_export_scope") != "evaluation" or not sidecar.get("feature_only"):
        raise ValueError(f"{name}: prediction export was not evaluation-only feature inference")
    result = aligned({"prediction": path.relative_to(ROOT).as_posix()}, groups, split_sha256)
    if isinstance(result, str):
        raise ValueError(f"{name}: {result}")
    array, order = result
    prediction = Lazy(array, order).load()
    if prediction.shape != (len(groups), 20_000, 2) or not np.isfinite(prediction).all():
        raise ValueError(f"{name}: invalid prediction shape or non-finite output")
    return prediction, {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "sidecar_sha256": sha256(sidecar_path),
        "membership_sha256": sidecar.get("evaluation_membership_sha256"),
    }


def _score_pair(
    targets: np.ndarray,
    mask: np.ndarray,
    active: np.ndarray,
    control: np.ndarray,
    components: np.ndarray,
) -> dict[str, Any]:
    """Compute matched official WP, component uncertainty, and fixed position bins."""
    active_score = score(targets, mask, active)
    control_score = score(targets, mask, control)
    active_moments, control_moments = paired_component_moments(targets, active, control, mask)
    bootstrap = paired_component_bootstrap(
        active_moments,
        control_moments,
        components,
        n_boot=BOOTSTRAP_DRAWS,
        seed=BOOTSTRAP_SEED,
        centered=True,
    )
    bins: dict[str, Any] = {}
    for start, end in TIME_BINS:
        bin_mask = mask.copy()
        bin_mask[:, :start] = False
        bin_mask[:, end:] = False
        active_bin = score(targets, bin_mask, active)
        control_bin = score(targets, bin_mask, control)
        bins[f"{start}:{end}"] = {
            "active": active_bin,
            "zero_control": control_bin,
            "delta": active_bin["wp"] - control_bin["wp"],
            "scored_rows": int(bin_mask.sum()),
        }
    return {
        "active": active_score,
        "zero_control": control_score,
        "delta": active_score["wp"] - control_score["wp"],
        "paired_component_bootstrap": bootstrap,
        "time_bins": bins,
    }


def main() -> None:
    """Grade both frozen predictions and save the immutable result receipt."""
    split_bytes = SPLIT_PATH.read_bytes()
    split = json.loads(split_bytes)
    split_sha256 = hashlib.sha256(split_bytes).hexdigest()
    groups = [int(group) for group in split["evaluation"]["valid"]]
    if len(groups) != 468:
        raise ValueError(f"expected 468 F0 evaluation groups, found {len(groups)}")
    component_by_group = {
        int(row["row_group"]): int(row["component"]) for row in split["sequences"]["valid"]
    }
    components = np.asarray([component_by_group[group] for group in groups], dtype=np.int64)
    if len(np.unique(components)) < 100:
        raise ValueError("unexpectedly few evaluation components for the fixed F0 panel")

    active, active_identity = _load_prediction(ACTIVE, groups, split_sha256)
    control, control_identity = _load_prediction(CONTROL, groups, split_sha256)
    with np.load(TARGETS_PATH, allow_pickle=False) as cache:
        position = {int(group): index for index, group in enumerate(cache["row_group"])}
        index = [position[group] for group in groups]
        targets = np.asarray(cache["y"][index], dtype=np.float32)
        mask = np.asarray(cache["mask"][index], dtype=bool)
    mask &= np.arange(targets.shape[1])[None, :] >= SCORED_FROM_ROW

    result = {
        "schema": "wnn.kalman_group_recurrent_interaction_grade.v1",
        "experiment_id": "E-KALMAN-GROUP-RECURRENT-INTERACTION-20261018-R1",
        "hypothesis_id": "H-KALMAN-GROUP-RECURRENT-INTERACTION-20261018-R1",
        "metric_id": "global_wp_clip2_v1",
        "split_manifest_sha256": split_sha256,
        "groups": len(groups),
        "unique_components": int(len(np.unique(components))),
        "mask": f"cached is_scored AND position >= {SCORED_FROM_ROW}; predictions frozen before target/mask access",
        "prediction_identity": {"active": active_identity, "zero_control": control_identity},
        "comparison": _score_pair(targets, mask, active, control, components),
        "score_json_identity": {
            name: sha256(ROOT / "artifacts/runs" / name / "score.json")
            for name in (ACTIVE, CONTROL)
        },
        "bootstrap": {
            "unit": "source-qualified connected evaluation component",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "scope": "conditional on these two fixed checkpoints and the repeatedly inspected F0 panel; excludes fit-seed, fold, and public-transfer uncertainty",
            "timing": "descriptive post-grade uncertainty; the pooled scores were already observed before this component analysis was written",
        },
        "exposure": "F0 is repeatedly inspected development evidence, not clean validation",
    }
    result["source_sha256"] = {
        "script": sha256(Path(__file__)),
        "split": sha256(SPLIT_PATH),
        "targets_cache": sha256(TARGETS_PATH),
    }
    output = PACKET / "grade_result_r1.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
