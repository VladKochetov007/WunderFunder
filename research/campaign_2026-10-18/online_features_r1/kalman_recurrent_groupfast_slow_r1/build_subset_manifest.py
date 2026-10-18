"""Select complete non-evaluation components for the recurrent interaction screen."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from wnn.splits import load_campaign_split  # noqa: E402
from wnn.training.membership import load_component_fit_subset  # noqa: E402

SPLIT_PATH = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
OUTPUT_PATH = Path(__file__).with_name("fit_subset_manifest_r1.json")
EXPERIMENT_ID = "E-KALMAN-GROUP-RECURRENT-INTERACTION-20261018-R1"
SEED = 20261018
COMPONENT_LIMIT = 32


def main() -> None:
    split = load_campaign_split(SPLIT_PATH, ROOT)
    by_component: dict[int, dict[str, list[int]]] = defaultdict(lambda: {"train": [], "valid": []})
    group_components: dict[tuple[str, int], int] = {}
    for source, group, component in split.components:
        by_component[int(component)][source].append(int(group))
        group_components[source, int(group)] = int(component)

    evaluation_components = {group_components["valid", group] for group in split.evaluation_groups}
    available = sorted(
        component
        for component, groups in by_component.items()
        if component not in evaluation_components and groups["train"] and groups["valid"]
    )
    if len(available) < COMPONENT_LIMIT:
        raise RuntimeError("not enough source-complete non-evaluation components")
    rng = np.random.default_rng(SEED)
    selected_components = sorted(
        int(value) for value in rng.choice(available, size=COMPONENT_LIMIT, replace=False)
    )
    selected_groups = {
        source: sorted(
            group for component in selected_components for group in by_component[component][source]
        )
        for source in ("train", "valid")
    }
    members = sorted(
        (source, group) for source, groups in selected_groups.items() for group in groups
    )
    evaluation_groups = sorted(int(group) for group in split.evaluation_groups)
    manifest = {
        "version": 1,
        "experiment_id": EXPERIMENT_ID,
        "split_manifest_sha256": split.sha256,
        "selection_seed": SEED,
        "selection_policy": "uniform_without_replacement_from_fit_components_present_in_train_and_valid_v1",
        "requested_component_count": COMPONENT_LIMIT,
        "selected_components": selected_components,
        "selected_component_sha256": hashlib.sha256(
            np.asarray(selected_components, dtype="<i8").tobytes()
        ).hexdigest(),
        "selected_groups": selected_groups,
        "selected_members_sha256": hashlib.sha256(
            json.dumps(members, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        ).hexdigest(),
        "evaluation_groups": evaluation_groups,
        "evaluation_group_count": len(evaluation_groups),
        "evaluation_membership_sha256": hashlib.sha256(
            np.asarray(evaluation_groups, dtype="<i8").tobytes()
        ).hexdigest(),
    }
    OUTPUT_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validated = load_component_fit_subset(OUTPUT_PATH, split, expected_experiment_id=EXPERIMENT_ID)
    print(
        json.dumps(
            {
                "manifest": str(OUTPUT_PATH.relative_to(ROOT)),
                "manifest_sha256": hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest(),
                "selected_component_count": len(validated["manifest"]["selected_components"]),
                "selected_group_counts": {
                    source: len(groups) for source, groups in selected_groups.items()
                },
                "evaluation_groups": len(evaluation_groups),
                "split_manifest_sha256": split.sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
