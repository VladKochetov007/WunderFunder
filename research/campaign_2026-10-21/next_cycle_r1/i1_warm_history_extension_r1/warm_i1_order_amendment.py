from __future__ import annotations

from pathlib import Path
from typing import cast

from warm_i1_preflight_amendment import (
    PLAN_RELATIVE,
    PLAN_SHA256,
    SOURCE_COMMIT,
    SOURCE_ROOT,
    _read_object,
    _sha256,
)

ORDER_EXPECTATION_RELATIVE = Path(
    "research/campaign_2026-10-21/next_cycle_r1/i1_warm_history_extension_r1/"
    "WARM_F1_ORDER_EXPECTATION_AMENDMENT_R1.json"
)
ORDER_EXPECTATION_SHA256 = "54810ff226495470562423919ccd5e0c62de5f71ddb2fcac3192b37839dcd0f5"
ORDER_SOURCE_RELATIVE = "src/wnn/training/data_membership.py"
ORDER_SOURCE_SHA256 = "7516c374cbfd57cbb395eac8d47de6a8eafbc6da47baf9bd29e8270bbb1f6daf"
ORDER_SPLIT_SHA256 = "31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae"
ORDER_PARENT_TRAIN_SHA256 = "0d0791b38ab83c3afd025fd8423b6f14d2e2f29a1cbec4aa6c80341d1b82af4e"
ORDER_CHILD_TRAIN_SHA256 = "35e122415b780c10092b215af19108ddc224726bee33970520636cd24d85bd5f"
ORDER_VALID_SHA256 = "087c48ccdd86649d56923b0755a9f909d6ee5270158f2027fc38f0ddae81c6a8"


def verify_f1_order_expectation(
    order_path: Path,
    source_root: Path,
    project_root: Path,
    plan_path: Path,
    plan: dict[str, object],
) -> dict[str, object]:
    """Bind the seed-4 child order while retaining the inherited R1 parent order."""
    order_path = order_path.resolve()
    source_root = source_root.resolve()
    project_root = project_root.resolve()
    if order_path != (project_root / ORDER_EXPECTATION_RELATIVE).resolve():
        raise ValueError("F1 order amendment path differs from its canonical packet path")
    if source_root != SOURCE_ROOT:
        raise ValueError("F1 order amendment requires the exact repaired source root")
    if plan_path.resolve() != (project_root / PLAN_RELATIVE).resolve():
        raise ValueError("F1 order amendment does not use the immutable R1 plan path")
    if _sha256(plan_path) != PLAN_SHA256:
        raise ValueError("frozen warm-history plan bytes differ from R1")
    if _sha256(order_path) != ORDER_EXPECTATION_SHA256:
        raise ValueError("F1 order-amendment bytes differ from the reviewed artifact")

    amendment = _read_object(order_path)
    if amendment.get("schema_version") != 1:
        raise ValueError("unsupported F1 order-amendment schema")
    if amendment.get("amendment_id") != "I1_WARM_F1_CANONICAL_ORDER_EXPECTATION_R1":
        raise ValueError("F1 order-amendment identity differs from R1")
    if amendment.get("status") != "PREPARED_ROOT_REVIEW_NOT_RUN":
        raise ValueError("F1 order-amendment status differs from the prepared artifact")

    frozen_plan = cast(dict[str, object], amendment["frozen_plan"])
    if frozen_plan != {"path": str(PLAN_RELATIVE), "sha256": PLAN_SHA256, "modified": False}:
        raise ValueError("F1 order amendment does not bind the immutable R1 plan")
    source = cast(dict[str, object], amendment["source"])
    source_spec = cast(dict[str, object], plan["source"])
    source_pins = cast(dict[str, str], source_spec["current_source_sha256"])
    if source != {
        "root": str(SOURCE_ROOT),
        "commit": SOURCE_COMMIT,
        "path": ORDER_SOURCE_RELATIVE,
        "sha256": ORDER_SOURCE_SHA256,
        "role": "Pinned canonical select_membership implementation used by the F1 fit.",
    }:
        raise ValueError("F1 order amendment source identity differs from its frozen contract")
    if source_pins.get(ORDER_SOURCE_RELATIVE) != ORDER_SOURCE_SHA256:
        raise ValueError("R1 plan does not pin the order-selection source bytes")
    if _sha256(source_root / ORDER_SOURCE_RELATIVE) != ORDER_SOURCE_SHA256:
        raise ValueError("repaired source order-selection file differs from R1")

    cells = cast(list[dict[str, object]], plan["cells"])
    cell = next((item for item in cells if item.get("fold") == 1), None)
    if cell is None:
        raise ValueError("frozen plan has no F1 cell")
    split_spec = cast(dict[str, object], cell["split"])
    expected_split = {
        "path": "research/campaign_2026-09-29/artifacts/splits/fold1.json",
        "sha256": ORDER_SPLIT_SHA256,
        "fold": 1,
        "fit_train_groups": 10533,
        "fit_valid_groups": 1416,
        "evaluation_groups_excluded": 457,
    }
    if amendment["split"] != expected_split:
        raise ValueError("F1 order amendment split metadata differs from R1")
    if (
        split_spec.get("path") != expected_split["path"]
        or split_spec.get("sha256") != ORDER_SPLIT_SHA256
        or split_spec.get("fit_train_order_sha256") != ORDER_PARENT_TRAIN_SHA256
        or split_spec.get("fit_valid_order_sha256") != ORDER_VALID_SHA256
        or split_spec.get("train_fit_groups") != expected_split["fit_train_groups"]
        or split_spec.get("valid_fit_groups") != expected_split["fit_valid_groups"]
        or split_spec.get("evaluation_groups") != expected_split["evaluation_groups_excluded"]
    ):
        raise ValueError("F1 split or historical parent-order identity differs from R1")

    run_ids = [
        str(run["run_id"])
        for run in cast(list[dict[str, object]], plan["runs"])
        if run.get("fold") == 1
    ]
    expected_run_ids = [
        "i1warm_f1_control_seed4_e2_cp1_r1",
        "i1warm_f1_history_seed4_e2_cp1_r1",
    ]
    if (
        amendment["cell"]
        != {
            "fold": 1,
            "seed": 4,
            "run_ids": expected_run_ids,
            "train_seqs_argument": 10607,
            "valid_repeat": 1,
        }
        or run_ids != expected_run_ids
        or cell.get("fit_seed") != 4
    ):
        raise ValueError("F1 order amendment run identity differs from the frozen R1 cell")

    membership_order = cast(dict[str, object], amendment["membership_order"])
    train = cast(dict[str, object], membership_order["train"])
    valid = cast(dict[str, object], membership_order["valid"])
    if train != {
        "R1_parent_order_sha256": ORDER_PARENT_TRAIN_SHA256,
        "child_expected_order_sha256": ORDER_CHILD_TRAIN_SHA256,
        "child_expected_count": 10533,
        "derivation": (
            "np.random.default_rng(seed=4).permutation(campaign_split.train_groups)[:train_seqs], "
            "where train_seqs=10607 and the source split contains 10533 eligible train groups; "
            "hash the resulting values as contiguous little-endian int64 bytes using group_identity."
        ),
        "interpretation": (
            "The R1 hash is retained as the actual parent historical order. It is not the expected "
            "order of either new F1 child, because the frozen fit argv seed is 4."
        ),
    }:
        raise ValueError("F1 child train-order amendment differs from the reviewed derivation")
    if (
        valid
        != {
            "R1_expected_order_sha256": ORDER_VALID_SHA256,
            "child_expected_order_sha256": ORDER_VALID_SHA256,
            "expected_count": 1416,
            "derivation": "campaign_split.valid_groups in frozen split order; valid_repeat=1.",
        }
        or membership_order.get("dataset_length") != 11949
    ):
        raise ValueError("F1 valid-order amendment differs from the frozen fit membership")

    return {
        "path": str(order_path),
        "sha256": ORDER_EXPECTATION_SHA256,
        "amendment_id": amendment["amendment_id"],
        "fold": 1,
        "seed": 4,
        "R1_parent_order_sha256": ORDER_PARENT_TRAIN_SHA256,
        "child_expected_order_sha256": ORDER_CHILD_TRAIN_SHA256,
        "child_expected_valid_order_sha256": ORDER_VALID_SHA256,
        "source_path": ORDER_SOURCE_RELATIVE,
        "source_sha256": ORDER_SOURCE_SHA256,
        "split_sha256": ORDER_SPLIT_SHA256,
    }
