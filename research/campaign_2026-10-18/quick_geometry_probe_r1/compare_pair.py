"""Grade the frozen quick pair with existing official component uncertainty tools."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from grade_prediction_bank import aligned  # noqa: E402

from wnn.evaluation import paired_component_bootstrap, paired_component_moments  # noqa: E402

PACKET = Path(__file__).resolve().parent
SPLIT = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"


def digest(path: Path) -> str:
    """Hash artifacts in bounded chunks."""
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def prediction(name: str, groups: list[int], split_sha: str) -> np.ndarray:
    """Require source-qualified producer alignment before reading predictions."""
    result = aligned({"prediction": f"cache/pred_{name}_valid.npy"}, groups, split_sha)
    if isinstance(result, str):
        raise ValueError(result)
    array, positions = result
    return np.asarray(array[positions], dtype=np.float32)


def main() -> None:
    """Report one exposed-fold paired contrast without a significance claim."""
    split = json.loads(SPLIT.read_text())
    groups = split["evaluation"]["valid"]
    components_by_group = {
        row["row_group"]: row["component"] for row in split["sequences"]["valid"]
    }
    components = np.asarray([components_by_group[group] for group in groups])
    with np.load(ROOT / "cache/valid_targets.npz", allow_pickle=False) as source:
        positions = {int(group): index for index, group in enumerate(source["row_group"])}
        selected = [positions[group] for group in groups]
        targets = np.asarray(source["y"][selected], dtype=np.float32)
        mask = np.asarray(source["mask"][selected], dtype=bool)
    mask[:, :99] = False
    names = {arm: f"geometry_quick_f0_s3_{arm}" for arm in ("feature", "control")}
    provenance = {
        arm: json.loads((ROOT / "artifacts/runs" / name / "provenance.json").read_text())
        for arm, name in names.items()
    }
    matched = (
        "initial_checkpoint_sha256",
        "initial_parameter_sha256",
        "fit_membership_identity",
        "normalization_identity",
        "causal_feature_prior",
        "split_manifest_sha256",
    )
    for field in matched:
        if provenance["feature"].get(field) != provenance["control"].get(field):
            raise ValueError(f"paired identity mismatch: {field}")
    histories = {
        arm: json.loads((ROOT / "artifacts/runs" / name / "history.json").read_text())
        for arm, name in names.items()
    }
    for field in ("epoch", "cumulative", "sample_order_identity"):
        if (
            histories["feature"][0]["update_diagnostics"][field]
            != histories["control"][0]["update_diagnostics"][field]
        ):
            raise ValueError(f"paired update/sampler mismatch: {field}")
    predictions = {arm: prediction(name, groups, digest(SPLIT)) for arm, name in names.items()}
    predictions["parent"] = prediction("warm_mechanism_f0_s3_c_r3", groups, digest(SPLIT))
    result = {
        "experiment_id": "E-GEOMETRY-QUICK-PROBE-20261018-R1",
        "metric": "global_wp_clip2_v1",
        "sequences": len(groups),
        "components": int(len(np.unique(components))),
        "selected_rows": int(mask.sum()),
        "matched_fields": list(matched),
        "comparisons": {},
        "limitations": "One exposed F0, fitting seed3 and fixed parent. Component intervals condition on fitted endpoints; they do not cover fitting-seed, selection or public-transfer uncertainty. No public score forecast.",
    }
    for candidate, reference in (
        ("feature", "control"),
        ("feature", "parent"),
        ("control", "parent"),
    ):
        left, right = paired_component_moments(
            targets, predictions[candidate], predictions[reference], mask
        )
        result["comparisons"][f"{candidate}_vs_{reference}"] = paired_component_bootstrap(
            left, right, components, n_boot=2000, seed=20261018, centered=True
        )
    (PACKET / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
