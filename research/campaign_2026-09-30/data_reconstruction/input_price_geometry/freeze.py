"""Freeze target-free price panel using existing component metadata only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[3]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    foldpath = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    fold = json.loads(foldpath.read_text())
    component_of = {int(n["row_group"]): int(n["component"]) for n in fold["sequences"]["train"]}
    prior = OWN.parent / "selection.json"
    records = json.loads(prior.read_text())
    excluded = {component_of[g] for g in records["fit_train_groups"] + records["test_train_groups"]}
    provenance = {str(prior.relative_to(ROOT)): digest(prior)}
    for path in (
        ROOT / "research/campaign_2026-09-30/event_content/exposure/selection.json",
        OWN.parent / "additional_feature_relations/selection.json",
        OWN.parent / "event_rate_features/selection.json",
        OWN.parent / "joint_origin_clock/selection.json",
    ):
        data = json.loads(path.read_text())
        for key in ("fit", "evaluation", "selected"):
            excluded.update(component_of[int(node["row_group"])] for node in data.get(key, []))
        provenance[str(path.relative_to(ROOT))] = digest(path)
    seed = "input-price-geometry-v1"
    allowed = [
        (int(g), component_of[int(g)])
        for g in fold["fit"]["train"]
        if int(g) != 528 and component_of[int(g)] not in excluded
    ]
    allowed.sort(key=lambda item: hashlib.sha256(f"{seed}|{item[0]}|{item[1]}".encode()).digest())
    chosen = []
    seen = set()
    for group, component in allowed:
        if component in seen:
            continue
        chosen.append({"row_group": group, "component": component})
        seen.add(component)
        if len(chosen) == 16:
            break
    if len(chosen) != 16:
        raise ValueError("insufficient fresh components")
    result = {
        "fit": chosen[:8],
        "test": chosen[8:],
        "excluded_components": sorted(excluded),
        "manifest_sha256": digest(foldpath),
        "prior_selection_hashes": provenance,
        "preregistration_sha256": digest(OWN / "preregistration.json"),
        "source_train_identity": fold["sources"]["datasets/train.parquet"],
        "selected_without_features_targets_masks": True,
        "scope": "within-probe unseen canonical components; campaign exploratory and prior broad feature fits exist",
    }
    with (OWN / "selection.json").open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
