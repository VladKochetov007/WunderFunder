from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from f1_checkpoint_common import (
    ORIGINAL_PLAN_PATH,
    RECOVERY_PACKET_DIR,
    REMOTE_ROOT,
    V2_STAGE1_VERIFIER_SHA256,
    CheckpointMismatch,
    arm_spec,
    expected_counts,
    read_json,
    sha256_file,
    validate_plan,
)
from f1_checkpoint_lineage import verify_lineage


class PeerStage1LineageTests(unittest.TestCase):
    """Exercise recovery verification against the real V2 stage1 receipt schema."""

    def setUp(self) -> None:
        self.plan = validate_plan(ORIGINAL_PLAN_PATH)
        self.spec = arm_spec(self.plan, "wide", "terminal")
        receipts = RECOVERY_PACKET_DIR.parent / "receipts"
        self.wide_receipt_path = receipts / "RANK01_WIDE_F1_S1001_STAGE1_VERIFICATION.json"
        self.deep_receipt_path = receipts / "RANK01_DEEP_F1_S1001_STAGE1_VERIFICATION.json"
        self.wide_receipt = read_json(self.wide_receipt_path)
        self.deep_receipt = read_json(self.deep_receipt_path)
        self.current_verifier_sha256 = sha256_file(
            Path(__file__).parent / "verify_f1_checkpoint.py"
        )
        self.assertNotEqual(self.current_verifier_sha256, V2_STAGE1_VERIFIER_SHA256)

    def _verify_lineage(self, root: Path, peer_receipt_path: Path) -> str | None:
        wide_dir = root / "wide-stage1"
        deep_dir = root / "deep-stage1"
        wide_dir.mkdir()
        deep_dir.mkdir()
        wide_state_path = wide_dir / "training_state.pt"
        deep_state_path = deep_dir / "training_state.pt"
        wide_state_path.touch()
        deep_state_path.touch()
        history = [{"epoch": 0}, {"epoch": 1}]
        fit_membership = {"identity": "same-pair"}
        normalization = {"identity": "same-normalization"}
        wide_contract = {
            "options": {"hidden": 512, "layers": 1},
            "fit_membership": fit_membership,
            "normalization": normalization,
        }
        deep_contract = {
            "options": {"hidden": 384, "layers": 2},
            "fit_membership": fit_membership,
            "normalization": normalization,
        }
        wide_state = {
            "next_epoch": 2,
            "contract": wide_contract,
            "progress": {"history": history},
        }
        deep_state = {
            "next_epoch": 2,
            "contract": deep_contract,
            "progress": {"cumulative_updates": expected_counts(2)},
        }
        args = Namespace(
            arm="wide",
            phase="terminal",
            stage1_snapshot_dir=wide_dir,
            stage1_verification=self.wide_receipt_path,
            peer_stage1_snapshot_dir=deep_dir,
            peer_stage1_verification=peer_receipt_path,
        )
        expected_parent = {
            "path": str(
                REMOTE_ROOT
                / "data/artifacts/runs"
                / str(self.spec["stage1_run"])
                / "training_state.pt"
            ),
            "sha256": self.wide_receipt["artifacts"]["training_state.pt"]["sha256"],
            "next_epoch": 2,
        }
        provenance = {"resumed_from": expected_parent}
        collection = {"resumed_from": expected_parent}
        state_by_path = {wide_state_path: wide_state, deep_state_path: deep_state}
        hash_by_path = {
            wide_state_path: str(self.wide_receipt["artifacts"]["training_state.pt"]["sha256"]),
            deep_state_path: str(self.deep_receipt["artifacts"]["training_state.pt"]["sha256"]),
        }

        def load_training_state(path: Path) -> object:
            return state_by_path[path]

        def observed_sha256(path: Path) -> str:
            return hash_by_path[path]

        with patch("f1_checkpoint_lineage.sha256_file", side_effect=observed_sha256):
            return verify_lineage(
                args,
                self.plan,
                self.spec,
                wide_contract,
                history,
                provenance,
                collection,
                load_training_state,
            )

    def test_v2_peer_receipt_passes_with_distinct_recovery_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            parent_sha = self._verify_lineage(Path(temporary_directory), self.deep_receipt_path)

        self.assertEqual(parent_sha, self.wide_receipt["artifacts"]["training_state.pt"]["sha256"])
        self.assertEqual(self.deep_receipt["verification_helper_sha256"], V2_STAGE1_VERIFIER_SHA256)

    def test_v2_peer_receipt_rejects_wrong_stage1_helper_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            deep_receipt_path = root / "wrong-peer-receipt.json"
            mutated_receipt = dict(self.deep_receipt)
            mutated_receipt["verification_helper_sha256"] = self.current_verifier_sha256
            deep_receipt_path.write_text(json.dumps(mutated_receipt))
            self.assertNotEqual(
                mutated_receipt["verification_helper_sha256"], V2_STAGE1_VERIFIER_SHA256
            )
            with self.assertRaisesRegex(CheckpointMismatch, "V2 stage1 verification helper SHA"):
                self._verify_lineage(root, deep_receipt_path)


if __name__ == "__main__":
    unittest.main()
