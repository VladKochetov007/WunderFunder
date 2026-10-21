"""Create the deterministic, label-free eight-group precision-probe plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
SPLIT = Path("research/campaign_2026-09-29/artifacts/splits/fold0.json")
GRADE = Path("research/campaign_2026-10-21/next_cycle_r1/NATIVE_GRADE_R3.json")
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
GRADE_SHA256 = "872ae52f0c83611d390e778d9666f910773e8d24e6494c12cb56fd8384788e35"
SEED = 271828


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(value).hexdigest()


def make_plan(repo_root: Path) -> dict[str, Any]:
    """Select one valid sequence per component without opening target data."""
    split_path, grade_path = repo_root / SPLIT, repo_root / GRADE
    split_raw, grade_raw = split_path.read_bytes(), grade_path.read_bytes()
    if sha256_bytes(split_raw) != SPLIT_SHA256 or sha256_bytes(grade_raw) != GRADE_SHA256:
        raise ValueError("frozen F0 split or native grade identity changed")
    split = json.loads(split_raw)
    grade = json.loads(grade_raw)
    if split.get("fold") != 0 or grade.get("schema") != "wnn.h512-tl3-endpoint-screen-grade.v1":
        raise ValueError("probe requires the frozen F0 native-grade source")
    sequence_by_group = {int(item["row_group"]): item for item in split["sequences"]["valid"]}
    candidates: list[tuple[str, str, dict[str, int]]] = []
    for row_group in map(int, split["evaluation"]["valid"]):
        item = sequence_by_group.get(row_group)
        if item is None:
            raise ValueError(f"F0 evaluation row group {row_group} lacks sequence metadata")
        qualified = f"valid:{row_group}"
        rank = sha256_bytes(f"{SEED}:{qualified}".encode("ascii"))
        candidates.append(
            (
                rank,
                qualified,
                {
                    "row_group": row_group,
                    "seq_ix": int(item["seq_ix"]),
                    "component": int(item["component"]),
                },
            )
        )
    selected: list[dict[str, Any]] = []
    seen_components: set[int] = set()
    for rank, qualified, item in sorted(candidates):
        if item["component"] in seen_components:
            continue
        selected.append({"source_qualified_group": qualified, "selection_sha256": rank, **item})
        seen_components.add(item["component"])
        if len(selected) == 8:
            break
    if len(selected) != 8:
        raise ValueError("F0 selection did not contain eight distinct components")
    selection_hash = sha256_bytes(
        "\n".join(item["source_qualified_group"] for item in selected).encode("ascii")
    )
    return {
        "schema": "wnn.native-h512-precision-probe-selection.v1",
        "status": "PREPARED_LABEL_FREE_SELECTION",
        "selection_seed": SEED,
        "selection_rule": "sort SHA256('271828:' + 'valid:<row_group>'); take first row group per distinct connected component",
        "labels_or_predictions_read": False,
        "split_path": SPLIT.as_posix(),
        "split_sha256": SPLIT_SHA256,
        "native_grade_path": GRADE.as_posix(),
        "native_grade_sha256": GRADE_SHA256,
        "evaluation_manifest_sha256": grade["target_identity"]["evaluation_manifest_sha256"],
        "selected_membership_sha256": selection_hash,
        "members": selected,
    }


def main() -> None:
    """Create a fresh output directory containing only the reviewed plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/home/vlad/development/wunder/research_tmp/native-h512-precision-probe-output-20261021"
        ),
    )
    args = parser.parse_args()
    plan = make_plan(args.repo_root.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "selection_plan.json").open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "selection_plan": str(args.output_dir / "selection_plan.json"),
                "membership_sha256": plan["selected_membership_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
