"""Evaluate the fixed adapter against its parent using the existing pair tools."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[2]
sys.path.insert(0, str(PACKET.parent / "quick_geometry_probe_r1"))
from compare_pair import SPLIT, digest, prediction  # noqa: E402

from wnn.evaluation import paired_component_bootstrap, paired_component_moments  # noqa: E402


def main() -> None:
    """Keep source-qualified membership and fixed-fit uncertainty explicit."""
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
    candidate = prediction("geometry_adapter_f0_s3_r2", groups, digest(SPLIT))
    parent = prediction("warm_mechanism_f0_s3_c_r3", groups, digest(SPLIT))
    left, right = paired_component_moments(targets, candidate, parent, mask)
    result = {
        "experiment_id": "E-GEOMETRY-ADAPTER-PROBE-20261018-R2",
        "metric": "global_wp_clip2_v1",
        "sequences": len(groups),
        "components": int(len(np.unique(components))),
        "adapter_vs_parent": paired_component_bootstrap(
            left, right, components, n_boot=2000, seed=20261018, centered=True
        ),
        "limitations": "One exposed F0 and fitting seed, adaptive followup to R1. Component intervals condition on fitted endpoints; not seed, selection or public-transfer uncertainty. No public forecast.",
    }
    (PACKET / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
