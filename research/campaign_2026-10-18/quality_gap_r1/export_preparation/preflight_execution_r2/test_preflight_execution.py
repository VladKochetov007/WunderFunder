"""Bounded tests for local receipt and create-only evidence helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bindings import (
    EXECUTION_DIR,
    build_stage_binding,
    expected_receipt_bindings,
    sha256_file,
    validate_preflight_receipt,
)
from local_common import write_bytes_exclusive
from remote_stage import create_link


class PreflightExecutionTests(unittest.TestCase):
    """Exercise local-only safeguards without SSH or large data access."""

    def test_review_bound_receipt_passes_without_a_predetermined_digest(self) -> None:
        receipt = expected_receipt_bindings()
        receipt["created_utc"] = "2026-10-18T15:00:00+00:00"
        validate_preflight_receipt(receipt)

    def test_receipt_with_wrong_overlay_digest_is_rejected(self) -> None:
        receipt = expected_receipt_bindings()
        receipt["created_utc"] = "2026-10-18T15:00:00+00:00"
        receipt["overlay_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "overlay_manifest_sha256"):
            validate_preflight_receipt(receipt)

    def test_stage_plan_keeps_old_registry_identity_and_current_cache_exception_separate(
        self,
    ) -> None:
        binding = build_stage_binding("preflight-r2-test-20261018")
        cache = next(
            item
            for item in binding["remote"]["portable_sources"]
            if item["project_path"] == "cache/chain_folds.npz"
        )
        self.assertEqual(cache["registry_entry"]["original_identity"]["bytes"], 550072)
        self.assertEqual(binding["remote"]["cache_folds"]["bytes"], 462719)
        self.assertEqual(
            binding["remote"]["cache_folds"]["sha256"],
            "a112f809259f631c2ee7c12480e4ec1d4eef197a980e7ce23c707d842d092567",
        )

    def test_stage_plan_pins_remote_helper_bytes(self) -> None:
        helper = EXECUTION_DIR / "remote_stage.py"
        binding = build_stage_binding("preflight-r2-test-20261018")
        self.assertEqual(
            binding["remote"]["stage_helper"],
            {
                "path": "research/campaign_2026-10-18/quality_gap_r1/export_preparation/preflight_execution_r2/remote_stage.py",
                "bytes": helper.stat().st_size,
                "sha256": sha256_file(helper),
            },
        )

    def test_raw_capture_write_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "receipt.json"
            write_bytes_exclusive(target, b"original")
            with self.assertRaises(FileExistsError):
                write_bytes_exclusive(target, b"replacement")
            self.assertEqual(target.read_bytes(), b"original")

    def test_staged_source_link_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            root = temp / "root"
            source = temp / "source.npz"
            root.mkdir()
            source.write_bytes(b"small-source")
            result = create_link(root, "cache/source.npz", str(source))
            link = root / "cache/source.npz"
            self.assertEqual(result["literal_target"], str(source))
            self.assertEqual(link.resolve(), source.resolve())
            with self.assertRaises(FileExistsError):
                create_link(root, "cache/source.npz", str(source))


if __name__ == "__main__":
    unittest.main()
