"""Synthetic-only tests for the bounded W18 checkpoint metadata extractor."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import extract_w18_checkpoint_metadata as extractor
import w18_metadata_core as core


def test_entry(logical_path: Path, identifier: str, size: int) -> dict[str, object]:
    """Build a root-bound synthetic input record without production paths or hashes."""
    resolved = extractor._path_resolution(logical_path)
    return {
        "id": identifier,
        "logical_path": str(logical_path),
        "source_resolution": resolved,
        "expected_bytes": size,
        "expected_sha256": "a" * 64,
        "expected_generation": extractor._stat_signature(Path(resolved["resolved_path"]).stat()),
    }


def test_manifest(inputs: list[dict[str, object]], admission_path: Path) -> dict[str, object]:
    """Build an exact synthetic manifest for validator and loader tests."""
    admission_raw = admission_path.read_bytes()
    return {
        "schema": extractor.SCHEMA,
        "status": "ROOT_HASH_BOUND",
        "resource_contract": extractor.RESOURCE_CONTRACT,
        "source_observation": extractor.SOURCE_OBSERVATION,
        "root_admission_receipt": {
            "path": str(admission_path),
            "sha256": hashlib.sha256(admission_raw).hexdigest(),
        },
        "inputs": inputs,
    }


class ManifestTests(unittest.TestCase):
    """Check the exact eight-target contract and fail-closed placeholders."""

    def test_allowlist_is_only_the_eight_preflighted_endpoints(self) -> None:
        base = "/home/grinchenkoeg/WunderFunder-creative/artifacts/runs"
        endpoints = (
            ("ffull_s3_ep1", "ffull_s3", "model_ep1.pt", 1_053_479),
            ("ffull_s4_ep1", "ffull_s4", "model_ep1.pt", 1_053_479),
            ("frawW_s3_ep3", "frawW_s3", "model_ep3.pt", 1_054_943),
            ("frawW_s4_ep3", "frawW_s4", "model_ep3.pt", 1_054_943),
            ("frawW_s3ft_ep1", "frawW_s3ft", "model_ep1.pt", 1_053_607),
            ("frawW_s4ft_ep1", "frawW_s4ft", "model_ep1.pt", 1_053_607),
            ("fmu128_s3_ep3", "fmu128_s3", "model_ep3.pt", 1_054_815),
            ("fmu128_s4_ep3", "fmu128_s4", "model_ep3.pt", 1_054_815),
        )
        expected = {
            identifier: (f"{base}/{run}/{filename}", size)
            for identifier, run, filename, size in endpoints
        }
        self.assertEqual(core.ALLOWLIST, expected)
        self.assertNotIn("fbc1k_s5_ep3", core.ALLOWLIST)
        self.assertNotIn("orgbase_model", core.ALLOWLIST)
        self.assertEqual(sum(size for _, size in core.ALLOWLIST.values()), 8_433_688)

    def test_validation_requires_exact_ordered_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            admission = base / "admission.json"
            admission.write_bytes(b'{"status":"synthetic"}')
            entries: list[dict[str, object]] = []
            with patch.object(core, "ALLOWLIST", {"fixture": (str(base / "model.pt"), 4)}):
                model = base / "model.pt"
                model.write_bytes(b"test")
                entries.append(test_entry(model, "fixture", 4))
                manifest = test_manifest(entries, admission)
                self.assertEqual(len(extractor.validate_manifest(manifest)[0]), 1)
                manifest["inputs"] = []
                with self.assertRaisesRegex(core.MetadataError, "exact eight inputs"):
                    extractor.validate_manifest(manifest)

    def test_unobserved_parent_id_is_not_allowlisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            admission = base / "admission.json"
            admission.write_bytes(b"admitted")
            manifest = {
                "schema": extractor.SCHEMA,
                "status": "ROOT_HASH_BOUND",
                "resource_contract": extractor.RESOURCE_CONTRACT,
                "source_observation": extractor.SOURCE_OBSERVATION,
                "root_admission_receipt": {
                    "path": str(admission),
                    "sha256": hashlib.sha256(admission.read_bytes()).hexdigest(),
                },
                "inputs": [{"id": "orgbase_model"}],
            }
            with self.assertRaisesRegex(core.MetadataError, "exact eight inputs"):
                extractor.validate_manifest(manifest)


class PathResolutionTests(unittest.TestCase):
    """Ensure symlink changes are represented and detected explicitly."""

    def test_resolution_records_symlink_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            target = base / "target.pt"
            logical = base / "model.pt"
            target.write_bytes(b"checkpoint")
            logical.symlink_to(target)
            binding = extractor._path_resolution(logical)
            self.assertEqual(binding["resolved_path"], str(target.resolve()))
            logical_link = next(
                entry for entry in binding["symlinks"] if entry["link_path"] == str(logical)
            )
            self.assertEqual(logical_link["link_text"], str(target))

    def test_manifest_rejects_symlinked_source_not_in_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            target = base / "target.pt"
            logical = base / "model.pt"
            target.write_bytes(b"test")
            logical.symlink_to(target)
            admission = base / "admission.json"
            admission.write_bytes(b"synthetic admission")
            entry = test_entry(logical, "fixture", 4)
            manifest = test_manifest([entry], admission)
            with (
                patch.object(core, "ALLOWLIST", {"fixture": (str(logical), 4)}),
                self.assertRaisesRegex(core.MetadataError, "resolved source path differs"),
            ):
                extractor.validate_manifest(manifest)

    def test_binding_hash_refuses_symlink_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            target = base / "target.json"
            link = base / "link.json"
            target.write_text("{}", encoding="utf-8")
            link.symlink_to(target)
            with self.assertRaisesRegex(core.MetadataError, "not a regular file"):
                core._sha256_file(link, 1024)

    def test_source_generation_replacement_during_load_is_rejected(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable in local test environment")
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            target = base / "source.pt"
            replacement = base / "replacement.pt"
            logical = target
            torch.save({"args": {"seed": 3}, "model": {"w": torch.ones((1,))}}, target)
            torch.save({"args": {"seed": 4}, "model": {"w": torch.zeros((1,))}}, replacement)
            admission = base / "admission.json"
            admission.write_bytes(b"synthetic admission")
            entry = test_entry(logical, "fixture", target.stat().st_size)
            fd = os.open(target, os.O_RDONLY)
            try:
                entry["expected_sha256"] = extractor._fd_sha256(fd)
            finally:
                os.close(fd)
            manifest = test_manifest([entry], admission)
            manifest_path = base / "inputs.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output_path = base / "result.json"
            original_load = torch.load

            def replace_source(*args: object, **kwargs: object) -> object:
                payload = original_load(*args, **kwargs)
                os.replace(replacement, target)
                return payload

            patch_allowlist = patch.object(
                core, "ALLOWLIST", {"fixture": (str(logical), target.stat().st_size)}
            )
            with (
                patch_allowlist,
                patch.object(
                    extractor, "_resource_observation", return_value={"cpu_quota_cores": 0.5}
                ),
                patch.object(torch, "load", side_effect=replace_source),
                self.assertRaisesRegex(core.MetadataError, "generation changed"),
            ):
                extractor.extract(manifest_path, output_path)
            self.assertFalse(output_path.exists())


class PayloadTests(unittest.TestCase):
    """Verify metadata summaries contain no tensor values or secret config values."""

    def test_tensor_schema_walk_is_depth_bounded(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable in local test environment")
        value: object = torch.ones((1,))
        for _ in range(core.MAX_DEPTH + 1):
            value = [value]
        with self.assertRaisesRegex(core.MetadataError, "tensor-schema nesting exceeds limit"):
            extractor.summarize_payload({"model": value}, torch.is_tensor)

    def test_summary_keeps_shapes_and_safe_config_only(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable in local test environment")
        result = extractor.summarize_payload(
            {
                "args": {"seed": 5, "api_token": "never-export"},
                "model": {"layer.weight": torch.tensor([[1.2345, 9.8765]])},
                "epoch": 2,
            },
            torch.is_tensor,
        )
        self.assertEqual(result["metadata"]["args"], {"seed": 5})
        self.assertEqual(result["tensor_schema"][0]["shape"], [1, 2])
        encoded = json.dumps(result)
        self.assertNotIn("never-export", encoded)
        self.assertNotIn("1.2345", encoded)

    def test_tiny_checkpoint_receipt_binds_inputs_and_admission(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable in local test environment")
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            source = base / "tiny-source.pt"
            logical = source
            torch.save(
                {
                    "args": {"seed": 5, "password": "never-export"},
                    "model": {"layer.weight": torch.tensor([[1.2345, 9.8765]])},
                    "epoch": 2,
                },
                source,
            )
            admission = base / "admission.json"
            admission.write_bytes(b'{"decision":"synthetic-only"}')
            entry = test_entry(logical, "fixture", source.stat().st_size)
            fd = os.open(source, os.O_RDONLY)
            try:
                entry["expected_sha256"] = extractor._fd_sha256(fd)
            finally:
                os.close(fd)
            manifest = test_manifest([entry], admission)
            manifest_path = base / "inputs.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output_path = base / "receipt.json"
            with (
                patch.object(core, "ALLOWLIST", {"fixture": (str(logical), source.stat().st_size)}),
                patch.object(
                    extractor, "_resource_observation", return_value={"cpu_quota_cores": 0.5}
                ),
            ):
                receipt = extractor.extract(manifest_path, output_path)
            stored = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                receipt["input_manifest_sha256"],
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                stored["root_admission_receipt_sha256"],
                manifest["root_admission_receipt"]["sha256"],
            )
            record = stored["inputs"][0]
            self.assertEqual(record["sha256_before"], record["sha256_after"])
            self.assertEqual(record["source_resolution"]["resolved_path"], str(source))
            self.assertEqual(record["tensor_schema"][0]["shape"], [1, 2])
            encoded = output_path.read_text(encoding="utf-8")
            self.assertNotIn("1.2345", encoded)
            self.assertNotIn("never-export", encoded)

    def test_existing_output_rejected_before_manifest_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            output = base / "existing.json"
            output.write_text("preserve", encoding="utf-8")
            with (
                patch.object(
                    extractor, "_read_manifest", side_effect=AssertionError("read too soon")
                ),
                self.assertRaisesRegex(core.MetadataError, "output receipt already exists"),
            ):
                extractor.extract(base / "missing.json", output)
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
