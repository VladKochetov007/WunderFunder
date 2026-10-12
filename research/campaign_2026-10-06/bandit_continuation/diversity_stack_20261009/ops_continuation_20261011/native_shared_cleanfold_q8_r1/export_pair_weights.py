#!/usr/bin/env python3
"""Export four source-pinned checkpoints as NumPy weights for the Py311 Q8 consumer."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PACKET = Path(__file__).resolve().parent


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=PACKET / "CONTRACT_R1.json")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.weights.exists() or args.receipt.exists():
        raise FileExistsError("refusing to overwrite a weights archive or receipt")
    contract_path = args.contract.resolve(strict=True)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != "wnn.native.shared-cleanfold-q8-quality-contract.r1":
        raise ValueError("unexpected contract schema")
    if contract.get("status") != "prepared_pending_root_execution_admission":
        raise ValueError("contract is not the pending frozen evaluation contract")
    for source in contract["source_bindings"].values():
        source_path = Path(source["path"])
        if file_sha256(source_path) != source["sha256"]:
            raise ValueError(f"frozen source changed: {source_path}")
    loader_path = Path(contract["source_bindings"]["checkpoint_loader"]["path"])
    loader_root = str(loader_path.parent)
    sys.path.insert(0, loader_root)
    helper = importlib.import_module("benchmark_shared_f0")
    import torch

    arrays: dict[str, np.ndarray] = {}
    array_map: dict[str, dict[str, dict[str, str]]] = {}
    fold_checks: dict[str, Any] = {}
    head_keys = {"head.0.weight", "head.0.bias", "head.2.weight", "head.2.bias"}
    for fold_text, binding in contract["folds"].items():
        fold = int(fold_text)
        loaded: dict[str, tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]] = {}
        for role in ("student", "hft"):
            model = binding["models"][role]
            checkpoint_path = Path(model["checkpoint_path"])
            if file_sha256(checkpoint_path) != model["checkpoint_sha256"]:
                raise ValueError(f"fold {fold} {role} checkpoint SHA changed")
            provenance_path = Path(model["provenance_path"])
            if file_sha256(provenance_path) != model["provenance_sha256"]:
                raise ValueError(f"fold {fold} {role} provenance SHA changed")
            weights, checkpoint = helper.load_checkpoint_weights(
                checkpoint_path, model["checkpoint_sha256"]
            )
            loaded[role] = (weights, checkpoint, model)

        student_raw = loaded["student"][1]["model"]
        hft_raw = loaded["hft"][1]["model"]
        if set(student_raw) != set(hft_raw):
            raise ValueError(f"fold {fold}: checkpoint tensor names differ")
        changed = [
            key
            for key in sorted(student_raw)
            if not torch.equal(student_raw[key].detach().cpu(), hft_raw[key].detach().cpu())
        ]
        if set(changed) != head_keys:
            raise ValueError(f"fold {fold}: shared trunks differ outside four heads: {changed}")
        student_head_rows = int(student_raw["head.2.weight"].shape[0])
        hft_head_rows = int(hft_raw["head.2.weight"].shape[0])
        if student_head_rows != 31 or hft_head_rows != 23:
            raise ValueError(
                f"fold {fold}: raw output head geometry differs: {student_head_rows}/{hft_head_rows}"
            )
        left = helper.canonical_inference_config(loaded["student"][1])
        right = helper.canonical_inference_config(loaded["hft"][1])
        if left != right or left["n_in"] != 211 or left["readout_present"]:
            raise ValueError(f"fold {fold}: serving configuration differs or is unsupported")
        member_shapes: dict[str, dict[str, list[int]]] = {}
        array_map[str(fold)] = {}
        for role in ("student", "hft"):
            member_weights = loaded[role][0]
            mapping: dict[str, str] = {}
            shapes: dict[str, list[int]] = {}
            for index, key in enumerate(sorted(member_weights)):
                value = np.ascontiguousarray(member_weights[key], dtype=np.float32)
                if not np.isfinite(value).all():
                    raise ValueError(f"fold {fold} {role} has nonfinite weight {key}")
                archive_key = f"f{fold}_{role}_{index:02d}"
                arrays[archive_key] = value
                mapping[key] = archive_key
                shapes[key] = list(value.shape)
            array_map[str(fold)][role] = mapping
            member_shapes[role] = shapes
        fold_checks[str(fold)] = {
            "changed_raw_tensors": changed,
            "raw_head_output_rows": {"student": student_head_rows, "hft": hft_head_rows},
            "served_output_rows": 2,
            "shared_backbone_verified_bitwise": True,
            "canonical_inference_config": left,
            "member_shapes": member_shapes,
        }

    args.weights.parent.mkdir(parents=True, exist_ok=True)
    with args.weights.open("xb") as stream:
        np.savez(stream, **arrays)
    archive_sha = file_sha256(args.weights)
    receipt = {
        "schema": "wnn.native.shared-cleanfold-q8-weights.r1",
        "status": "PASS_SOURCE_PINNED_WEIGHTS_EXPORTED",
        "contract_path": str(contract_path),
        "contract_sha256": file_sha256(contract_path),
        "weights_path": str(args.weights.resolve()),
        "weights_sha256": archive_sha,
        "weights_bytes": args.weights.stat().st_size,
        "member_order": ["student", "hft"],
        "coefficients": [0.5, 0.5],
        "array_map": array_map,
        "folds": fold_checks,
        "training_performed": False,
        "predictions_generated": False,
        "quality_evaluation_performed": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "weights_sha256": archive_sha,
                "weights_bytes": receipt["weights_bytes"],
                "output": str(args.weights),
            }
        )
    )


if __name__ == "__main__":
    main()
