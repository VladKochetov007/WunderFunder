"""Small synthetic export and compact-source inputs for the R4 collector test."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pins import METRIC_ID
from source_alignment import sha256_array, sha256_file

SCORING_MASK = "need_prediction AND is_scored; compact rows already selected"


@dataclass(frozen=True)
class TinyCollectionFixture:
    """Files and metadata used to exercise one tiny end-to-end collection."""

    args: argparse.Namespace
    diagnostic: dict[str, object]
    bundle_spec: dict[str, object]
    bundle_path: Path
    q8_receipt: dict[str, object]
    q8_receipt_path: Path
    expected_bundle: dict[str, np.ndarray]
    split_result: tuple[
        dict[int, dict[str, int]],
        list[int],
        dict[str, object],
        dict[str, object],
        Path,
        Path,
        Path,
    ]
    export_receipt: dict[str, object]
    capture_bindings: dict[str, object]
    raw_guard_path: Path
    captured_sidecar_path: Path
    preinference_path: Path
    extracted: dict[str, np.ndarray]
    group_audit: list[dict[str, object]]
    control: np.ndarray


def _digest(path: Path) -> str:
    """Return a fixture file's SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_tiny_fixture(root: Path) -> TinyCollectionFixture:
    """Create small, producer-shaped files for the positive collector test."""
    diagnostic_dir = root / "repo/research/campaign_2026-10-18/quality_gap_r1/diagnostics"
    diagnostic_dir.mkdir(parents=True)
    (diagnostic_dir / "INPUTS.json").write_text("{}\n", encoding="utf-8")
    (root / "outputs").mkdir()

    target = np.asarray([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6], [0.7, -0.8]], dtype=np.float32)
    control = np.asarray([[-0.1, 0.2], [0.2, -0.3], [0.4, 0.5], [-0.6, 0.7]], dtype=np.float32)
    row_group = np.asarray([1, 1, 2, 2], dtype=np.int64)
    component = np.asarray([10, 10, 11, 11], dtype=np.int64)
    expected_bundle = {"target": target, "row_group": row_group, "component": component}
    array_hashes = {name: sha256_array(value) for name, value in expected_bundle.items()}

    bundle_path = diagnostic_dir / "bundle.npz"
    np.savez(bundle_path, **expected_bundle)
    bundle_spec: dict[str, object] = {
        "path": "bundle.npz",
        "sha256": sha256_file(bundle_path),
        "bytes": bundle_path.stat().st_size,
        "array_sha256": array_hashes,
        "keys": {"target": "target", "row_group": "row_group", "component": "component"},
        "scalar_bindings": {},
    }
    fold_path = diagnostic_dir / "fold.json"
    fold_path.write_text('{"fixture":true}\n', encoding="utf-8")
    q8_receipt_path = diagnostic_dir / "q8.json"
    q8_receipt: dict[str, object] = {"panel_sha256": "f" * 64}
    q8_receipt_path.write_text(json.dumps(q8_receipt), encoding="utf-8")
    diagnostic: dict[str, object] = {
        "schema": "wnn.quality_gap.inputs.v1",
        "source": {
            "metric_id": METRIC_ID,
            "mask_predicate": SCORING_MASK,
            "selected_rows": 4,
            "row_groups": 2,
            "components": 2,
            "fold_manifest": {"path": "fold.json", "sha256": _digest(fold_path)},
        },
        "bundle": bundle_spec,
        "receipts": {},
        "models": {
            name: {
                "kind": "bundle_key",
                "key": "target",
                "array_sha256": array_hashes["target"],
            }
            for name in ("wide", "deep", "q8")
        },
        "comparisons": [],
    }

    valid_parquet = root / "valid.parquet"
    valid_parquet.write_bytes(b"tiny source identity fixture")
    old_split, canonical_split, membership = (
        root / "old-split.json",
        root / "canonical-split.json",
        root / "membership.json",
    )
    for path in (old_split, canonical_split, membership):
        path.write_text("{}\n", encoding="utf-8")
    split_result = (
        {1: {"seq_ix": 1001, "component": 10}, 2: {"seq_ix": 1002, "component": 11}},
        [1, 2],
        {},
        {"status": "passed"},
        old_split,
        canonical_split,
        membership,
    )

    prediction_path = root / "full-valid.npy"
    with prediction_path.open("wb") as stream:
        np.save(stream, np.arange(24, dtype=np.float32).reshape(4, 3, 2), allow_pickle=False)
    prediction_sha256 = sha256_file(prediction_path)
    sidecar_path = root / "full-valid.json"
    sidecar = {
        "prediction_sha256": prediction_sha256,
        "row_groups": [0, 1, 2, 3],
        "campaign_fit_scope": "full-valid-feature-only",
        "evaluation_membership_sha256": None,
        "campaign_evaluation_membership_sha256": None,
    }
    sidecar_path.write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")
    contract_path = root / "contract.json"
    contract_path.write_text('{"fixture":"contract"}\n', encoding="utf-8")
    export_receipt_path = root / "export-receipt.json"
    export_receipt: dict[str, object] = {
        "schema": "quality_gap.f0_strong_control_export_receipt.r1",
        "source_pins": {"checkpoint_sha256": "a" * 64},
        "execution": {"command": ["fixture"]},
        "pre_inference_runtime": {"strict_fp32": True},
    }
    export_receipt_path.write_text(json.dumps(export_receipt, sort_keys=True), encoding="utf-8")
    raw_guard_path = root / "captured-guard.json"
    raw_guard_path.write_text('{"status":"finished"}\n', encoding="utf-8")
    captured_sidecar_path = root / "captured-sidecar.json"
    captured_sidecar_path.write_bytes(sidecar_path.read_bytes())
    preinference_path = root / "captured-pre-inference.json"
    preinference_path.write_text('{"strict_fp32":true}\n', encoding="utf-8")
    capture_bindings: dict[str, object] = {
        "contract_sha256": _digest(contract_path),
        "export_receipt_sha256": _digest(export_receipt_path),
        "attempt_id": "tiny-export-attempt",
        "raw_guard_attempt_sha256": sha256_file(raw_guard_path),
        "ordered_row_groups_sha256": hashlib.sha256(
            np.arange(4, dtype="<i8").tobytes()
        ).hexdigest(),
        "captured_sidecar_sha256": sha256_file(captured_sidecar_path),
        "raw_guard_attempt_path": str(raw_guard_path),
        "captured_sidecar_path": str(captured_sidecar_path),
        "raw_preinference_sha256": sha256_file(preinference_path),
        "captured_preinference_path": str(preinference_path),
    }
    extracted = {
        "target": target.copy(),
        "prediction": control.copy(),
        "row_group": row_group.copy(),
        "component": component.copy(),
        "position": np.asarray([100, 101, 200, 201], dtype=np.int64),
    }
    group_audit: list[dict[str, object]] = [
        {"row_group": 1, "selected_rows": 2},
        {"row_group": 2, "selected_rows": 2},
    ]
    args = argparse.Namespace(
        repo_root=root / "repo",
        valid_parquet=valid_parquet,
        prediction=prediction_path,
        sidecar=sidecar_path,
        export_contract=contract_path,
        expected_export_contract_sha256=_digest(contract_path),
        export_receipt=export_receipt_path,
        expected_sidecar_sha256=_digest(sidecar_path),
        expected_export_receipt_sha256=_digest(export_receipt_path),
        output=root / "outputs/aligned",
    )
    return TinyCollectionFixture(
        args=args,
        diagnostic=diagnostic,
        bundle_spec=bundle_spec,
        bundle_path=bundle_path,
        q8_receipt=q8_receipt,
        q8_receipt_path=q8_receipt_path,
        expected_bundle=expected_bundle,
        split_result=split_result,
        export_receipt=export_receipt,
        capture_bindings=capture_bindings,
        raw_guard_path=raw_guard_path,
        captured_sidecar_path=captured_sidecar_path,
        preinference_path=preinference_path,
        extracted=extracted,
        group_audit=group_audit,
        control=control,
    )
