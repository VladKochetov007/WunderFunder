from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from f3_checkpoint_common import (
    ARCHIVE_MANIFEST_SHA256,
    FOLD3_SHA256,
    PLAN_SHA256,
    REGISTRY_SHA256,
    REMOTE_ROOT,
    SOURCE_REVISION,
    arm_spec,
    expected_counts,
    validate_plan,
)
from f3_checkpoint_lineage import verify_lineage

PLAN = (
    Path(__file__).resolve().parents[1] / "f3_beleriand_preparation_r1" / "F3_BELERIAND_PLAN.json"
)


def _history_fixture() -> list[dict[str, object]]:
    """Build two rows in the captured producer history schema."""
    rows = []
    for epoch in range(2):
        counts = expected_counts(1)
        rows.append(
            {
                "epoch": epoch,
                "train_seconds": 1.0,
                "evaluation": "withheld",
                "update_diagnostics": {
                    "epoch": counts,
                    "cumulative": expected_counts(epoch + 1),
                    "grad_stats": {"epoch": epoch, "steps": 14_880, "nonfinite_steps": 0},
                },
                "fit_summary": {
                    "objective_mean": 1.0,
                    "optimizer_windows": 14_880,
                    "optimizer_updates": dict(counts),
                    "learning_rate_by_group": [
                        {"group": 0, "first": 0.1, "last": 0.01, "mean": 0.05}
                    ],
                },
            }
        )
    return rows


class F3LineageTests(unittest.TestCase):
    def test_terminal_lineage_rejects_wrong_parent_attempt_and_history(self) -> None:
        plan_path = (
            Path(__file__).resolve().parents[1]
            / "f3_beleriand_preparation_r1"
            / "F3_BELERIAND_PLAN.json"
        )
        plan = validate_plan(plan_path)
        wide = arm_spec(plan, "wide", "stage1")
        deep = arm_spec(plan, "deep", "stage1")
        history = _history_fixture()[:2]
        common = {"fit_membership": {"fold": 3}, "normalization": {"mean": "same"}}
        wide_contract = {**common, "options": {"hidden": 512, "layers": 1, "valid_fold": 3}}
        deep_contract = {**common, "options": {"hidden": 384, "layers": 2, "valid_fold": 3}}
        helper_hash = "c" * 64
        dependencies = {"f3_checkpoint_common.py": "d" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wide_dir = root / "wide"
            deep_dir = root / "deep"
            wide_dir.mkdir()
            deep_dir.mkdir()
            wide_state_path = wide_dir / "training_state.pt"
            deep_state_path = deep_dir / "training_state.pt"
            wide_state_path.write_bytes(b"wide stage1")
            deep_state_path.write_bytes(b"deep stage1")
            wide_state = {
                "next_epoch": 2,
                "contract": wide_contract,
                "progress": {"history": history, "cumulative_updates": expected_counts(2)},
            }
            deep_state = {
                "next_epoch": 2,
                "contract": deep_contract,
                "progress": {"history": history, "cumulative_updates": expected_counts(2)},
            }

            def receipt(arm: str, spec: dict[str, object], path: Path) -> dict[str, object]:
                return {
                    "schema": "priority25.f3_checkpoint_verification.v1",
                    "status": "passed",
                    "phase": "stage1",
                    "arm": arm,
                    "run": spec["run"],
                    "next_epoch": 2,
                    "plan_sha256": PLAN_SHA256,
                    "fold3_manifest_sha256": FOLD3_SHA256,
                    "portable_registry_sha256": REGISTRY_SHA256,
                    "source_archive_manifest_sha256": ARCHIVE_MANIFEST_SHA256,
                    "source_revision": SOURCE_REVISION,
                    "verification_helper_sha256": helper_hash,
                    "verification_dependencies": dependencies,
                    "completed_update_counts": expected_counts(2),
                    "attempt": {
                        "id": spec["attempt"],
                        "command": spec["command"],
                        "status": "finished",
                        "exit_status": 0,
                    },
                    "artifacts": {
                        "training_state.pt": {
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                        }
                    },
                }

            wide_receipt = root / "wide.json"
            deep_receipt = root / "deep.json"
            wide_receipt.write_text(json.dumps(receipt("wide", wide, wide_state_path)))
            deep_receipt.write_text(json.dumps(receipt("deep", deep, deep_state_path)))
            args = Namespace(
                phase="terminal",
                arm="wide",
                stage1_snapshot_dir=wide_dir,
                stage1_verification=wide_receipt,
                peer_stage1_snapshot_dir=deep_dir,
                peer_stage1_verification=deep_receipt,
            )
            spec = arm_spec(plan, "wide", "terminal")
            parent_path = (
                REMOTE_ROOT / "data/artifacts/runs" / str(wide["run"]) / "training_state.pt"
            )
            resumed = {
                "path": str(parent_path),
                "sha256": hashlib.sha256(wide_state_path.read_bytes()).hexdigest(),
                "next_epoch": 2,
            }
            provenance = {"resumed_from": resumed}
            collection = {"resumed_from": resumed}
            states = {str(wide_state_path): wide_state, str(deep_state_path): deep_state}

            def load_state(path: Path) -> object:
                return states[str(path)]

            self.assertEqual(
                verify_lineage(
                    args,
                    plan,
                    spec,
                    wide_contract,
                    history,
                    provenance,
                    collection,
                    load_state,
                    helper_hash,
                    dependencies,
                ),
                resumed["sha256"],
            )

            bad_receipt = receipt("wide", wide, wide_state_path)
            bad_receipt["attempt"]["id"] = "wrong-attempt"
            wide_receipt.write_text(json.dumps(bad_receipt))
            with self.assertRaisesRegex(ValueError, "verified stage1 attempt identity"):
                verify_lineage(
                    args,
                    plan,
                    spec,
                    wide_contract,
                    history,
                    provenance,
                    collection,
                    load_state,
                    helper_hash,
                    dependencies,
                )

            wide_receipt.write_text(json.dumps(receipt("wide", wide, wide_state_path)))
            changed_history = [dict(history[0]), history[1]]
            changed_history[0]["fit_summary"] = {"objective_mean": 9.0}
            states[str(wide_state_path)]["progress"]["history"] = changed_history
            with self.assertRaisesRegex(ValueError, "stage1 history changed"):
                verify_lineage(
                    args,
                    plan,
                    spec,
                    wide_contract,
                    history,
                    provenance,
                    collection,
                    load_state,
                    helper_hash,
                    dependencies,
                )
