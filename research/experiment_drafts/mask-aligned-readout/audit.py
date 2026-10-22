"""Exact receiver-side audits for the four mask-aligned head checkpoints."""

from __future__ import annotations

# Any represents serialized checkpoint and provenance records validated by helpers.
import argparse
import sys
from pathlib import Path
from typing import Any

PACKET = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKET))
from audit_model import load_model_state, non_head_hashes  # noqa: E402
from audit_prefit import prefit  # noqa: E402
from audit_receipts import write_receipt  # noqa: E402
from audit_runtime import require_b300_runtime, require_same_device  # noqa: E402
from audit_validation import (  # noqa: E402
    compare_inherited_identity,
    validate_child_args,
    validate_child_membership,
    validate_embedded_provenance,
    validate_fixed_endpoint,
    validate_trainer_provenance,
    validate_training_runtime,
)
from receiver_assets import (  # noqa: E402
    RECIPE_PATH,
    REPO_ROOT,
    Cell,
    file_sha256,
    load_json,
    load_receiver_assets,
    output_root,
    recipe_cells,
)


def compare_pair(cell: Cell, assets: Any, treatment: dict[str, Any]) -> None:
    """Require the paired fit records to differ only by name and mask weighting."""
    import torch

    root = assets.root
    control_path = root / "artifacts" / "runs" / cell.run_name("all_valid") / "model_ep0.pt"
    control_prov_path = control_path.with_name("provenance.json")
    if not control_path.is_file() or not control_prov_path.is_file():
        raise FileNotFoundError("paired all-valid control checkpoint/provenance is missing")
    control = torch.load(control_path, map_location="cpu", weights_only=False)
    control_prov = load_json(control_prov_path)
    recipe = load_json(RECIPE_PATH)
    cell_recipe = next(value for value in recipe["cells"] if value["fold"] == cell.fold)
    validate_child_args(
        cell, "all_valid", str(cell_recipe["split_manifest_sha256"]), control, control_prov
    )
    validate_child_membership(cell, assets, control_prov)
    validate_embedded_provenance(
        control,
        control_prov,
        (
            "split_manifest_sha256",
            "fit_row_groups",
            "evaluation_row_groups",
            "initial_checkpoint_sha256",
            "normalization_identity",
            "trainer_implementation",
            "execution_runtime",
        ),
    )
    validate_trainer_provenance(cell, assets, control_prov)
    validate_training_runtime(cell, control_prov)
    validate_fixed_endpoint(cell, "all_valid", assets, control, control_prov)
    treatment_args = dict(treatment["args"])
    control_args = dict(control.get("args", {}))
    for args in (control_args, treatment_args):
        args.pop("name", None)
        args.pop("mask_weight", None)
    if control_args != treatment_args:
        raise ValueError("paired control and treatment fit arguments differ beyond the mask arm")
    if control_prov.get("initial_checkpoint_sha256") != treatment.get("provenance", {}).get(
        "initial_checkpoint_sha256"
    ):
        raise ValueError("paired arms loaded different frozen parent checkpoints")


def after_fit(cell: Cell, arm: str, assets_path: Path, receipt_path: Path) -> None:
    """Verify checkpoint lineage, FP32/B300 metadata, and unchanged frozen tensors."""

    assets = load_receiver_assets(assets_path)
    recipe = load_json(RECIPE_PATH)
    cell_recipe = next(value for value in recipe["cells"] if value["fold"] == cell.fold)
    child_path = assets.root / "artifacts" / "runs" / cell.run_name(arm) / "model_ep0.pt"
    provenance_path = child_path.with_name("provenance.json")
    if not child_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(
            f"completed child checkpoint/provenance missing for {cell.fold}/{arm}"
        )
    if file_sha256(child_path) != load_json(provenance_path).get("checkpoint_sha256", {}).get(
        "model_ep0.pt"
    ):
        raise ValueError("child checkpoint bytes differ from its provenance record")
    parent_path = assets.parent_checkpoints[cell.fold]
    parent_sha = file_sha256(parent_path)
    if load_json(provenance_path).get("initial_checkpoint_sha256") != parent_sha:
        raise ValueError("child provenance does not bind the pinned clean parent checkpoint")
    child, child_model = load_model_state(child_path)
    parent, _ = load_model_state(parent_path)
    provenance = load_json(provenance_path)
    split_sha = str(cell_recipe["split_manifest_sha256"])
    validate_child_args(cell, arm, split_sha, child, provenance)
    validate_child_membership(cell, assets, provenance)
    validate_embedded_provenance(
        child,
        provenance,
        (
            "split_manifest_sha256",
            "fit_row_groups",
            "evaluation_row_groups",
            "initial_checkpoint_sha256",
            "normalization_identity",
            "trainer_implementation",
            "execution_runtime",
        ),
    )
    validate_trainer_provenance(cell, assets, provenance)
    training_runtime = validate_training_runtime(cell, provenance)
    fit_endpoint = validate_fixed_endpoint(cell, arm, assets, child, provenance)
    compare_inherited_identity(cell, assets, parent, child, provenance)
    if child.get("epoch") != 0:
        raise ValueError("child checkpoint is not fixed epoch zero")
    if child.get("args", {}).get("device") != "cuda":
        raise ValueError("child checkpoint was not configured to train on CUDA")
    hardware = require_b300_runtime()
    require_same_device(training_runtime, hardware)
    if child.get("args", {}).get("precision") != "fp32":
        raise ValueError("child checkpoint was not configured for canonical strict FP32")
    before = non_head_hashes(parent["model"])
    after = non_head_hashes(child["model"])
    if before != after:
        changed = sorted(
            set(before) ^ set(after)
            | {key for key in before.keys() & after.keys() if before[key] != after[key]}
        )
        raise ValueError("frozen trunk/preprocessing tensors changed: " + ", ".join(changed))
    prefit_path = output_root(assets.root) / "audit" / f"prefit_{cell.fold}.json"
    prefit_record = load_json(prefit_path)
    expected_trainable = sorted(
        name for name, _ in child_model.named_parameters() if name.startswith("head.")
    )
    if (
        prefit_record.get("parent_checkpoint_sha256") != parent_sha
        or prefit_record.get("non_head_tensor_sha256") != before
        or prefit_record.get("trainable_parameter_names") != expected_trainable
        or prefit_record.get("normalization_identity") != provenance["normalization_identity"]
    ):
        raise ValueError("child audit does not match the immutable prefit parameter receipt")
    require_same_device(training_runtime, prefit_record["device"])
    record = {
        "schema_version": 1,
        "stage": "after_fit",
        "fold": cell.fold,
        "arm": arm,
        "child_run": cell.run_name(arm),
        "child_checkpoint_sha256": file_sha256(child_path),
        "child_provenance_sha256": file_sha256(provenance_path),
        "prefit_receipt_sha256": file_sha256(prefit_path),
        "parent_checkpoint_sha256": parent_sha,
        "parent_provenance_sha256": file_sha256(assets.parent_provenance[cell.fold]),
        "split_manifest_sha256": file_sha256(assets.split_paths[cell.fold]),
        "recipe_sha256": file_sha256(RECIPE_PATH),
        "profile_receipt_sha256": prefit_record["profile_receipt_sha256"],
        "profile_memory_evidence": prefit_record["profile_memory_evidence"],
        "receiver_hardware_observed_after_fit": hardware,
        "training_process_runtime": training_runtime,
        "fixed_fit_endpoint": fit_endpoint,
        "training_device_bound_by_trainer_runtime": "B300",
        "training_precision_bound_by_trainer_runtime": "strict fp32",
        "trainer_implementation_sha256": provenance["trainer_implementation"]["sha256"],
        "non_head_tensor_sha256": after,
        "normalization_identity": provenance["normalization_identity"],
        "expected_trainable_parameter_names_by_canonical_freeze": expected_trainable,
        "frozen_tensor_unchanged": True,
        "validated_fields": [
            "child_checkpoint_sha256",
            "child_provenance_sha256",
            "parent_checkpoint_sha256",
            "parent_provenance_sha256",
            "split_manifest_sha256",
            "recipe_sha256",
            "profile_receipt_sha256",
            "profile_memory_evidence",
            "trainer_implementation_sha256",
            "prefit_receipt_sha256",
            "receiver_hardware_observed_after_fit",
            "training_process_runtime",
            "fixed_fit_endpoint",
            "training_precision_bound_by_trainer_runtime",
            "non_head_tensor_sha256",
            "normalization_identity",
            "frozen_tensor_unchanged",
        ],
    }
    if arm == "is_scored_weighted":
        compare_pair(cell, assets, {"args": child["args"], "provenance": provenance})
        record["paired_fit_args_match_except_mask_weight"] = True
    write_receipt(receipt_path, record)


def parse_args() -> argparse.Namespace:
    """Parse explicit stage and receiver paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prefit", "after"))
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--fold", choices=("F0", "F2"), required=True)
    parser.add_argument("--arm", choices=("all_valid", "is_scored_weighted"))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--profile-receipt", type=Path)
    return parser.parse_args()


def main() -> int:
    """Run the requested parameter or frozen-tensor audit."""
    args = parse_args()
    recipe = load_json(RECIPE_PATH)
    cell = next(cell for cell in recipe_cells(recipe) if cell.fold == args.fold)
    sys.path.insert(0, str(REPO_ROOT / "src"))
    if args.stage == "prefit":
        if args.profile_receipt is None:
            raise SystemExit("--profile-receipt is required for prefit audit")
        prefit(cell, args.assets, args.receipt, args.profile_receipt)
    else:
        if args.arm is None or args.profile_receipt is not None:
            raise SystemExit("--arm is required and --profile-receipt is prefit-only")
        after_fit(cell, args.arm, args.assets, args.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
