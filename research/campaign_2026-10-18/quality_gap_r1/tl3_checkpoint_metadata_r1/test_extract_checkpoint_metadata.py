"""Bounded tests for the TL3 metadata-only checkpoint extractor."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import extract_checkpoint_metadata as extractor


class FakeTensor:
    """Small tensor-shaped metadata fixture with no numerical payload."""

    def __init__(self, shape: tuple[int, ...], dtype: str = "torch.float32") -> None:
        self.shape = shape
        self.dtype = dtype

    def numel(self) -> int:
        result = 1
        for value in self.shape:
            result *= value
        return result


def fake_manifest() -> dict[str, object]:
    """Return the exact allowlist with deterministic fixture digests."""
    inputs = []
    for logical_id, (path, size) in extractor.ALLOWLIST.items():
        inputs.append(
            {
                "id": logical_id,
                "path": path,
                "expected_bytes": size,
                "expected_sha256": hashlib.sha256(logical_id.encode()).hexdigest(),
            }
        )
    return {
        "schema": extractor.SCHEMA,
        "status": "ROOT_HASH_BOUND",
        "root_admission_receipt_sha256": "a" * 64,
        "resource_contract": extractor.RESOURCE_CONTRACT,
        "inputs": inputs,
    }


class ManifestTests(unittest.TestCase):
    def test_exact_root_bound_inventory_passes(self) -> None:
        self.assertEqual(len(extractor.validate_manifest(fake_manifest())), 8)

    def test_unbound_template_fails_closed(self) -> None:
        value = fake_manifest()
        value["status"] = "NEEDS_ROOT_FRESH_SHA_BINDING"
        with self.assertRaisesRegex(extractor.MetadataError, "fresh root-bound"):
            extractor.validate_manifest(value)

    def test_unexpected_source_path_fails_closed(self) -> None:
        value = fake_manifest()
        value["inputs"][0]["path"] = "/tmp/model.pt"  # type: ignore[index]
        with self.assertRaisesRegex(extractor.MetadataError, "unexpected path"):
            extractor.validate_manifest(value)

    def test_missing_source_hash_fails_closed(self) -> None:
        value = fake_manifest()
        value["inputs"][0]["expected_sha256"] = None  # type: ignore[index]
        with self.assertRaisesRegex(extractor.MetadataError, "fresh SHA-256 missing"):
            extractor.validate_manifest(value)

    def test_duplicate_id_rejected(self) -> None:
        value = fake_manifest()
        value["inputs"][1]["id"] = value["inputs"][0]["id"]  # type: ignore[index]
        with self.assertRaisesRegex(extractor.MetadataError, "duplicate input id"):
            extractor.validate_manifest(value)


class PayloadTests(unittest.TestCase):
    def test_extracts_tensor_schema_without_values_and_redacts_secret_keys(self) -> None:
        payload = {
            "args": {"seed": 5, "init": "", "api_token": "do-not-copy"},
            "model": {"layer.weight": FakeTensor((2, 3)), "layer.bias": FakeTensor((2,))},
            "epoch": 3,
            "optimizer": {"state": FakeTensor((4,))},
        }
        result = extractor.summarize_payload(payload, lambda value: isinstance(value, FakeTensor))
        self.assertEqual(result["metadata"]["args"], {"seed": 5, "init": ""})
        self.assertEqual(result["metadata"]["epoch"], 3)
        self.assertEqual(
            result["tensor_schema"],
            [
                {
                    "path": "model.layer.weight",
                    "shape": [2, 3],
                    "dtype": "torch.float32",
                },
                {"path": "model.layer.bias", "shape": [2], "dtype": "torch.float32"},
                {"path": "optimizer.state", "shape": [4], "dtype": "torch.float32"},
            ],
        )
        serialized = json.dumps(result)
        self.assertNotIn("do-not-copy", serialized)
        self.assertNotIn("0.0", serialized)

    def test_tensor_schema_handles_nested_sequences(self) -> None:
        result = extractor.tensor_metadata(
            {"readout": [FakeTensor((1, 2), "torch.float16")]},
            lambda value: isinstance(value, FakeTensor),
        )
        self.assertEqual(result[0]["path"], "readout[0]")
        self.assertEqual(result[0]["dtype"], "torch.float16")

    def test_sha256_streams_expected_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "small.bin"
            path.write_bytes(b"checkpoint metadata test")
            self.assertEqual(
                extractor.sha256_file(path), hashlib.sha256(path.read_bytes()).hexdigest()
            )

    def test_existing_output_is_rejected_before_manifest_or_import(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "already-present.json"
            output.write_text("keep", encoding="utf-8")
            patcher = patch.object(
                extractor, "_read_manifest", side_effect=AssertionError("read too early")
            )
            patcher.start()
            self.addCleanup(patcher.stop)
            with self.assertRaisesRegex(extractor.MetadataError, "output receipt already exists"):
                extractor.extract(Path("missing.json"), output)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")

    def test_tiny_torch_checkpoint_capture_is_metadata_only(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable in local test environment")
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            checkpoint = base / "fixture.pt"
            torch.save(
                {
                    "args": {"name": "fixture", "seed": 5, "init": ""},
                    "model": {"layer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
                    "epoch": 2,
                },
                checkpoint,
            )
            expected_bytes = checkpoint.stat().st_size
            expected_sha = extractor.sha256_file(checkpoint)
            manifest = {
                "schema": extractor.SCHEMA,
                "status": "ROOT_HASH_BOUND",
                "root_admission_receipt_sha256": "b" * 64,
                "resource_contract": extractor.RESOURCE_CONTRACT,
                "inputs": [
                    {
                        "id": "fixture",
                        "path": str(checkpoint),
                        "expected_bytes": expected_bytes,
                        "expected_sha256": expected_sha,
                    }
                ],
            }
            manifest_path = base / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output = base / "receipt.json"
            with (
                patch.object(
                    extractor, "ALLOWLIST", {"fixture": (str(checkpoint), expected_bytes)}
                ),
                patch.object(
                    extractor, "_resource_observation", return_value={"cpu_quota_cores": 0.5}
                ),
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}),
            ):
                result = extractor.extract(manifest_path, output)
            record = result["inputs"][0]
            self.assertEqual(record["sha256_before"], expected_sha)
            self.assertEqual(record["sha256_after"], expected_sha)
            self.assertEqual(record["metadata"]["args"]["seed"], 5)
            self.assertEqual(record["tensor_schema"][0]["shape"], [2, 2])
            self.assertNotIn("model", record["metadata"])
            self.assertNotIn("1.0", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
