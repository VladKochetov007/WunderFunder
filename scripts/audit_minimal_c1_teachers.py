"""Audit the exact existing strong-teacher pair used by minimal C1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_distill_cache import (
    _audit_ancestor_closure,
    _inspect_teacher,
    _target_order,
)

from wnn.distillation_cache import file_sha256, row_groups_sha256
from wnn.splits import load_campaign_split

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "research/campaign_2026-10-06/bandit_continuation"
SOURCE_FAMILIES = CAMPAIGN / "SOURCE_FAMILIES.json"
SPLIT_MANIFEST = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
EXPECTED_TEACHERS = {
    3: {
        "run": "c20261003_combo_s3_f0",
        "epoch": 3,
        "checkpoint_sha256": "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e",
        "provenance_sha256": "9a9d17eeb4b844d85bb5ca132e0085d0cf6dd1387d3d3de4c56d8e3bb8be7a6f",
    },
    4: {
        "run": "c20261003_combo_s4_f0",
        "epoch": 3,
        "checkpoint_sha256": "0203b109a1f25f3c82794ad3684b83fe4a4bc815f4b5c1bc4bc1ea450dba0a8d",
        "provenance_sha256": "c9fb263175f9b13c5b897f8c24d8f2dbe1164415bfd5adcb22f59dc95a72b332",
    },
}


def audit_teachers() -> dict[str, Any]:
    """Verify pinned checkpoints, clean-fold ancestry, and prediction bytes."""
    source_manifest = json.loads(SOURCE_FAMILIES.read_text())
    strong_sources = {
        int(source["seed"]): source
        for source in source_manifest["families"]
        if source.get("family") == "strong"
    }
    if set(strong_sources) != set(EXPECTED_TEACHERS):
        raise ValueError("SOURCE_FAMILIES strong teacher seeds differ from minimal C1")

    split = load_campaign_split(SPLIT_MANIFEST, ROOT)
    if split.sha256 != SPLIT_SHA256:
        raise ValueError("fold0 split manifest hash differs from the frozen C1 recipe")
    target_orders = {name: _target_order(ROOT, name) for name in ("train", "valid")}
    teachers: list[dict[str, Any]] = []
    for seed, expected in EXPECTED_TEACHERS.items():
        source = strong_sources[seed]
        for field in ("run", "epoch_zero_based", "checkpoint_sha256", "provenance_sha256"):
            expected_field = "epoch" if field == "epoch_zero_based" else field
            if source.get(field) != expected[expected_field]:
                raise ValueError(f"SOURCE_FAMILIES seed {seed} has a changed {field}")
        run = str(expected["run"])
        checkpoint_root = ROOT / "artifacts/runs"
        checkpoint_path = checkpoint_root / run / f"model_ep{expected['epoch']}.pt"
        provenance_path = checkpoint_root / run / "provenance.json"
        checkpoint_sha256 = file_sha256(checkpoint_path)
        provenance_sha256 = file_sha256(provenance_path)
        if checkpoint_sha256 != expected["checkpoint_sha256"]:
            raise ValueError(f"teacher {run} checkpoint differs from the pinned source hash")
        if provenance_sha256 != expected["provenance_sha256"]:
            raise ValueError(f"teacher {run} provenance differs from the pinned source hash")

        ancestor_closure = _audit_ancestor_closure(
            run,
            split,
            checkpoint_root,
            f"model_ep{expected['epoch']}.pt",
            str(expected["checkpoint_sha256"]),
        )
        record, arrays, _ = _inspect_teacher(
            run,
            split,
            ROOT / "cache",
            checkpoint_root,
            target_orders,
            int(expected["epoch"]),
        )
        if (
            record["architecture"] != "gru"
            or record["cell"] != "gru"
            or record["hidden"] != 256
            or record["layers"] != 2
            or record["seed"] != seed
        ):
            raise ValueError(f"teacher {run} architecture or seed differs from the C1 source")
        for split_name, array in arrays.items():
            record["predictions"][split_name].update(
                {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "finite_scan": "passed in 64-sequence blocks",
                }
            )
        record["provenance_sha256"] = provenance_sha256
        record["ancestor_closure"] = ancestor_closure
        teachers.append(record)

    return {
        "format": "minimal-c1-teacher-audit-v1",
        "status": "passed",
        "split_manifest_sha256": split.sha256,
        "source_families_sha256": file_sha256(SOURCE_FAMILIES),
        "source_families_path": str(SOURCE_FAMILIES.relative_to(ROOT)),
        "target_orders": {
            name: {
                "row_count": len(groups),
                "row_groups_sha256": row_groups_sha256(groups),
            }
            for name, groups in target_orders.items()
        },
        "teachers": teachers,
        "target_recipe": {
            "teacher_runs": [teacher["run"] for teacher in teachers],
            "cli_scales": [0.8125, 0.8125],
            "legacy_loader_divisor": 2,
            "effective_coefficients": [0.40625, 0.40625],
            "coefficient_sum": 0.8125,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_teachers()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
        return
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(payload)
    print(output)


if __name__ == "__main__":
    main()
