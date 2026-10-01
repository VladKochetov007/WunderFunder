"""Freeze metadata-only TRAIN selection; never open competition data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[3]
FOLD = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
PRIOR = ROOT / "research/campaign_2026-09-30/data_reconstruction/selection.json"
MAP = ROOT / "research/campaign_2026-09-30/data_reconstruction/interior_code_table.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    contract = json.loads((OWN / "preregistration.json").read_text())
    fold = json.loads(FOLD.read_text())
    prior = json.loads(PRIOR.read_text())
    component_of = {int(n["row_group"]): int(n["component"]) for n in fold["sequences"]["train"]}
    inspected = prior["fit_train_groups"] + prior["test_train_groups"]
    excluded = {component_of[g] for g in inspected}
    seed = contract["selection"]["seed"]
    candidates = [
        (int(g), component_of[int(g)])
        for g in fold["fit"]["train"]
        if int(g) != 528 and component_of[int(g)] not in excluded
    ]
    candidates.sort(
        key=lambda pair: hashlib.sha256(f"{seed}:{pair[0]}:{pair[1]}".encode()).digest()
    )
    seen = set()
    selected = []
    for group, component in candidates:
        if component not in seen:
            selected.append(
                {
                    "row_group": group,
                    "component": component,
                    "selection_hash": hashlib.sha256(
                        f"{seed}:{group}:{component}".encode()
                    ).hexdigest(),
                }
            )
            seen.add(component)
        if len(selected) == 16:
            break
    if len(selected) != 16:
        raise ValueError("insufficient fresh allowed-fit TRAIN components")
    data = ROOT / "datasets/train.parquet"
    stat = data.stat()
    source = fold["sources"]["datasets/train.parquet"]
    if stat.st_size != source["bytes"] or stat.st_mtime_ns != source["mtime_ns"]:
        raise ValueError("TRAIN file identity changed")
    result = {
        "selected": selected,
        "eligible_group_count": len(candidates),
        "excluded_prior_canonical_components": sorted(excluded),
        "prior_component_grammar": "prior chain_folds chain IDs differ from canonical connectedcomponent IDs; exclusion converted using prior group memberships in frozen fold0 manifest",
        "disjoint_from_prior_groups": True,
        "disjoint_from_prior_canonical_components": True,
        "source_train_identity": source,
        "fold_manifest_sha256": digest(FOLD),
        "prior_selection_sha256": digest(PRIOR),
        "class_table_sha256": digest(MAP),
        "preregistration_sha256": digest(OWN / "preregistration.json"),
        "selected_without_features_targets_masks": True,
        "class_map_fit_groups_not_fold0_fit": sorted(
            set(prior["fit_train_groups"]) - set(fold["fit"]["train"])
        ),
        "class_map_note": "existing TRAIN-only table diagnostics only; not fitted for this fold and never used by D1 model",
    }
    out = OWN / "selection.json"
    if out.exists() and json.loads(out.read_text()) != result:
        raise ValueError("immutable selection differs")
    out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
