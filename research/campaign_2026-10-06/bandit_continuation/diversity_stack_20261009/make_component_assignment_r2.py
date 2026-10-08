from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
SEED = "LUNA-STACK-R2-20261009"
N_COMPONENTS = 141
N_FOLDS = 3
COMPONENTS_PER_FOLD = 47


def _repo_root() -> Path:
    """Return the repository root containing this isolated research artifact."""
    return Path(__file__).resolve().parents[4]


def _parse_args() -> argparse.Namespace:
    """Parse the frozen source and in-folder output locations."""
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=_repo_root() / "research/campaign_2026-09-29/artifacts/splits/fold0.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=folder / "LUNA_STACKING_COMPONENT_FOLDS_R2.json",
    )
    return parser.parse_args()


def _build_assignment(split_bytes: bytes) -> dict[str, object]:
    """Build fixed three-way folds from evaluation components without labels."""
    split_sha256 = hashlib.sha256(split_bytes).hexdigest()
    if split_sha256 != SPLIT_SHA256:
        raise ValueError(f"split manifest hash mismatch: {split_sha256}")

    manifest = json.loads(split_bytes)
    eval_groups = manifest["evaluation"]["valid"]
    component_by_group = {
        row["row_group"]: row["component"]
        for row in manifest["sequences"]["valid"]
    }
    if len(eval_groups) != 468 or len(set(eval_groups)) != 468:
        raise ValueError("expected 468 unique evaluation groups")
    if not set(eval_groups).issubset(component_by_group):
        raise ValueError("evaluation group missing from component map")

    groups_by_component: dict[int, list[int]] = {}
    for group_id in eval_groups:
        component_id = component_by_group[group_id]
        groups_by_component.setdefault(component_id, []).append(group_id)
    component_ids = sorted(groups_by_component)
    if len(component_ids) != N_COMPONENTS:
        raise ValueError(f"expected 141 components, found {len(component_ids)}")

    ordered = sorted(
        (
            hashlib.sha256(f"{SEED}\0{component_id}".encode("ascii")).hexdigest(),
            component_id,
        )
        for component_id in component_ids
    )
    components = []
    for rank, (digest, component_id) in enumerate(ordered):
        fold_id = rank // COMPONENTS_PER_FOLD
        components.append(
            {
                "component_id": component_id,
                "assignment_digest": digest,
                "outer_fold": fold_id,
                "evaluation_groups": sorted(groups_by_component[component_id]),
            }
        )
    if len(components) != N_COMPONENTS or any(
        row["outer_fold"] >= N_FOLDS for row in components
    ):
        raise ValueError("component assignment does not form three 47-component folds")

    body: dict[str, object] = {
        "schema": "wnn.luna-stacking-component-folds.v1",
        "experiment_id": "LUNA-STACK-R2-20261009",
        "source_split_path": "research/campaign_2026-09-29/artifacts/splits/fold0.json",
        "source_split_sha256": split_sha256,
        "source_population": {
            "evaluation_groups": 468,
            "evaluation_components": N_COMPONENTS,
            "component_method": manifest["component_method"],
        },
        "assignment": {
            "seed": SEED,
            "digest_rule": "sha256(UTF8(seed) + NUL + ASCII(decimal(component_id)))",
            "order": "ascending (assignment_digest, component_id)",
            "fold_rule": "rank//47",
            "components_per_fold": [COMPONENTS_PER_FOLD] * N_FOLDS,
            "components": components,
        },
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    body["assignment_sha256"] = hashlib.sha256(canonical).hexdigest()
    return body


def main() -> int:
    """Verify and write the frozen component-fold membership artifact."""
    args = _parse_args()
    folder = Path(__file__).resolve().parent
    if args.output.resolve().parent != folder:
        raise ValueError("output must remain in the isolated proposal folder")
    result = _build_assignment(args.split_manifest.read_bytes())
    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(args.output)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info(
        "wrote %s with assignment_sha256=%s",
        args.output,
        result["assignment_sha256"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
