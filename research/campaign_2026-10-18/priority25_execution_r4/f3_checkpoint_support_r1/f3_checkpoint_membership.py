from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from f3_checkpoint_common import (
    DATASET_LENGTH,
    FOLD3_SHA256,
    mapping,
    read_json,
    require,
    sha256_file,
)


def validate_manifest_membership(
    manifest_path: Path, contract: Mapping[str, object], provenance: Mapping[str, object]
) -> dict[str, object]:
    """Recompute exact F3 fit and evaluation row orders from the sealed manifest."""
    import numpy as np

    require(manifest_path.stat().st_size <= 2_000_000, "fold manifest size bound")
    require(sha256_file(manifest_path) == FOLD3_SHA256, "fold3 manifest SHA256")
    manifest = read_json(manifest_path)
    require(
        (manifest["version"], manifest["fold"], manifest["n_folds"]) == (1, 3, 4), "fold identity"
    )
    fit = mapping(manifest["fit"], "manifest fit membership")
    evaluation = mapping(manifest["evaluation"], "manifest evaluation membership")
    train = cast(list[int], fit["train"])
    valid = cast(list[int], fit["valid"])
    eval_valid = cast(list[int], evaluation["valid"])
    require((len(train), len(valid), len(eval_valid)) == (10_539, 1_368, 505), "fold3 group counts")
    require(
        all(len(set(groups)) == len(groups) for groups in (train, valid, eval_valid)),
        "duplicate manifest group",
    )
    train_order = np.random.default_rng(1001).permutation(np.asarray(train, dtype="<i8"))[:10607]
    valid_order = np.asarray(valid, dtype="<i8")
    eval_order = np.random.default_rng(12345).permutation(np.asarray(eval_valid, dtype="<i8"))
    fit_rows = mapping(provenance["fit_row_groups"], "provenance fit row groups")
    require(fit_rows["train"] == train_order.tolist(), "ordered train membership")
    require(fit_rows["valid"] == valid_order.tolist(), "ordered fit-valid membership")
    require(
        provenance["evaluation_row_groups"] == eval_order.tolist(), "ordered evaluation membership"
    )
    components = mapping(manifest["sequences"], "manifest source-qualified sequences")
    component_of: dict[tuple[str, int], int] = {}
    for split in ("train", "valid"):
        rows = cast(list[dict[str, object]], components[split])
        for row in rows:
            key = (split, int(row["row_group"]))
            require(key not in component_of, "duplicate source-qualified sequence")
            component_of[key] = int(row["component"])
    fit_components = {component_of[("train", int(group))] for group in train}
    fit_components.update(component_of[("valid", int(group))] for group in valid)
    eval_components = {component_of[("valid", int(group))] for group in eval_valid}
    require(not fit_components.intersection(eval_components), "fit/evaluation component overlap")

    def identity(groups: object) -> dict[str, object]:
        values = np.asarray(groups, dtype="<i8")
        return {
            "count": len(values),
            "ordered_int64_le_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        }

    expected_membership = {
        "train": identity(train_order),
        "valid": identity(valid_order),
        "valid_repeat": 1,
        "dataset_length": DATASET_LENGTH,
    }
    require(contract["fit_membership"] == expected_membership, "contract fit membership identity")
    require(
        provenance["fit_membership_identity"] == expected_membership,
        "provenance fit membership identity",
    )
    require(provenance.get("split_manifest_sha256") == FOLD3_SHA256, "provenance fold manifest SHA")
    return {
        "manifest_sha256": FOLD3_SHA256,
        "fit_train_groups": len(train),
        "fit_valid_groups": len(valid),
        "evaluation_groups": len(eval_valid),
        "evaluation_components": len(eval_components),
        "fit_membership_identity": expected_membership,
        "ordered_evaluation_membership_sha256": identity(eval_order)["ordered_int64_le_sha256"],
    }
