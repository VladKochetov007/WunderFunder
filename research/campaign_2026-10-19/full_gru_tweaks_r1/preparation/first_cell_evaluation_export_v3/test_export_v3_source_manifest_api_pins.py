"""Metadata-only checks for V3's API-to-source-manifest pin join."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[4]
PREPARATION = ROOT / "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation"
V2 = PREPARATION / "first_cell_evaluation_export_v2"
V2_BINDING = V2 / "EXPORT_BINDING_V1.json"
V3_BINDING = PACKET / "EXPORT_BINDING_V3.json"
MANIFEST = PACKET / "EXPORT_SOURCE_MANIFEST.json"
V3_EXPORTER = PACKET / "export_first_cell_v3.py"
RECORDS = ROOT / "research/campaign_2026-10-19/full_gru_tweaks_r1/records"
EXPECTED_API_PATHS = (
    "src/wnn/data.py",
    "src/wnn/resources.py",
    "src/wnn/splits.py",
    "src/wnn/training/environment.py",
    "src/wnn/training/evaluation.py",
    "src/wnn/training/input_representation.py",
    "src/wnn/training/inputs.py",
    "src/wnn/training/models.py",
)


def _load_exporter():
    """Load only the stdlib-only exporter module for direct gate exercise."""
    spec = importlib.util.spec_from_file_location("export_first_cell_v3_fixture", V3_EXPORTER)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load V3 source module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata() -> tuple[dict[str, object], dict[str, object], bytes, bytes]:
    """Read only the pinned V3 binding and 167-file source manifest."""
    binding_raw = V3_BINDING.read_bytes()
    binding = json.loads(binding_raw)
    manifest_raw = MANIFEST.read_bytes()
    manifest = json.loads(manifest_raw)
    if not isinstance(binding, dict) or not isinstance(manifest, dict):
        raise AssertionError("binding and source manifest must be JSON objects")
    return binding, manifest, binding_raw, manifest_raw


def _source_tree_digest(files: list[dict[str, object]]) -> str:
    """Recompute the declared metadata aggregate without reading source files."""
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: value["path"]):
        digest.update(str(item["path"]).encode("utf-8") + b"\0")
        digest.update(str(item["sha256"]).encode("ascii") + b"\n")
    return digest.hexdigest()


def _actual_receipt_paths() -> dict[str, tuple[Path, Path]]:
    """Locate the ten exact small local accepted-fit and exact-stop receipts."""
    return {
        "control": (
            RECORDS / "child_control_fit_v1/TERMINAL_ACCEPTANCE.json",
            RECORDS / "child_control_fit_v1/exact_stop/STDOUT.raw.json",
        ),
        "ema": (
            RECORDS / "child_ema_fit_v1/TERMINAL_ACCEPTANCE.json",
            RECORDS / "child_ema_fit_v1/exact_stop/EXACT_STOP.json",
        ),
        "sparse": (
            RECORDS / "child_dictionary_fit_v1/TERMINAL_ACCEPTANCE.json",
            RECORDS / "child_dictionary_fit_v1/exact_stop/EXACT_STOP.json",
        ),
        "nll": (
            RECORDS / "child_nll_fit_v1/TERMINAL_ACCEPTANCE.json",
            RECORDS / "child_nll_fit_v1/exact_stop/EXACT_STOP.json",
        ),
        "allthree": (
            RECORDS / "child_allthree_fit_v1/TERMINAL_ACCEPTANCE.json",
            RECORDS / "child_allthree_fit_v1/exact_stop/EXACT_STOP.json",
        ),
    }


class ExportV3SourceManifestApiTests(unittest.TestCase):
    def test_v3_binding_is_only_the_new_private_evidence_root(self) -> None:
        old = V2_BINDING.read_bytes()
        new = V3_BINDING.read_bytes()
        old_root = b"fg19_first_cell_f0_s1001_evaluation_v1"
        new_root = b"fg19_first_cell_f0_s1001_evaluation_v2"
        self.assertEqual(old.count(old_root), 4)
        self.assertEqual(new, old.replace(old_root, new_root))
        self.assertEqual(len(new), 15_798)
        self.assertEqual(
            hashlib.sha256(new).hexdigest(),
            "372ec16016517baa510cc1899f140cd0978a35b83f96a4a5fa1ff4bda8b39730",
        )

    def test_all_eight_api_pins_match_complete_manifest_rows(self) -> None:
        binding, manifest, _, manifest_raw = _metadata()
        source = binding["source"]
        files = manifest["files"]
        self.assertEqual(len(manifest_raw), source["manifest_bytes"])
        self.assertEqual(hashlib.sha256(manifest_raw).hexdigest(), source["manifest_sha256"])
        self.assertEqual(manifest["schema"], "fg19.teacher-cache-source-manifest.v1")
        self.assertEqual(manifest["source_commit"], source["commit"])
        self.assertEqual(manifest["source_root"], source["source_root"])
        self.assertEqual(len(files), 167)
        self.assertEqual(source["file_count"], 167)
        self.assertEqual(_source_tree_digest(files), source["source_set_sha256"])
        self.assertEqual(
            source["source_set_sha256"],
            "9c461bcc5883572a071d41a86e18b9a91136099cf8fa87371ce7c3ee41445bd8",
        )
        self.assertEqual(set(source["api_files"]), set(EXPECTED_API_PATHS))
        by_path = {item["path"]: item for item in files}
        self.assertEqual(len(by_path), 167)
        for relative in EXPECTED_API_PATHS:
            api_pin = source["api_files"][relative]
            self.assertEqual(set(api_pin), {"bytes", "sha256"})
            self.assertEqual(by_path[relative], {"path": relative, **api_pin})

    def test_full_source_tree_gate_uses_full_manifest_row_shape(self) -> None:
        exporter = _load_exporter()
        binding = exporter._load_binding(PACKET)
        manifest = json.loads(MANIFEST.read_bytes())
        files = manifest["files"]

        with tempfile.TemporaryDirectory(prefix="export-v3-source-manifest-") as temp:
            source_root = (Path(temp) / "source").resolve()
            source_root.mkdir()
            binding["source"]["source_root"] = str(source_root)
            for item in files:
                (source_root / item["path"]).parent.mkdir(parents=True, exist_ok=True)

            file_pins = {item["path"]: item for item in files}

            def fake_digest_file(path: Path, max_bytes: int) -> tuple[int, str]:
                del max_bytes
                relative = path.relative_to(source_root).as_posix()
                item = file_pins[relative]
                return item["bytes"], item["sha256"]

            with patch.object(exporter, "_digest_file", side_effect=fake_digest_file):
                observed = exporter._verify_source_tree(source_root, binding, PACKET)
            self.assertEqual(len(observed), 167)
            self.assertEqual(observed, file_pins)

    def test_full_source_tree_gate_rejects_wrong_api_bytes_hash_and_path(self) -> None:
        exporter = _load_exporter()
        binding = exporter._load_binding(PACKET)
        manifest = json.loads(MANIFEST.read_bytes())
        files = manifest["files"]

        with tempfile.TemporaryDirectory(prefix="export-v3-source-manifest-negative-") as temp:
            source_root = (Path(temp) / "source").resolve()
            source_root.mkdir()
            for item in files:
                (source_root / item["path"]).parent.mkdir(parents=True, exist_ok=True)
            file_pins = {item["path"]: item for item in files}

            def fake_digest_file(path: Path, max_bytes: int) -> tuple[int, str]:
                del max_bytes
                item = file_pins[path.relative_to(source_root).as_posix()]
                return item["bytes"], item["sha256"]

            with patch.object(exporter, "_digest_file", side_effect=fake_digest_file):
                for field, invalid_value in (
                    ("bytes", binding["source"]["api_files"]["src/wnn/data.py"]["bytes"] + 1),
                    ("sha256", "0" * 64),
                ):
                    invalid = json.loads(json.dumps(binding))
                    invalid["source"]["source_root"] = str(source_root)
                    invalid["source"]["api_files"]["src/wnn/data.py"][field] = invalid_value
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        exporter._verify_source_tree(source_root, invalid, PACKET)

                invalid = json.loads(json.dumps(binding))
                invalid["source"]["source_root"] = str(source_root)
                invalid["source"]["api_files"]["src/wnn/missing.py"] = invalid["source"][
                    "api_files"
                ].pop("src/wnn/data.py")
                with self.assertRaises(ValueError):
                    exporter._verify_source_tree(source_root, invalid, PACKET)

    def test_full_receipt_model_gate_matches_ten_actual_local_receipts(self) -> None:
        exporter = _load_exporter()
        binding = exporter._load_binding(PACKET)
        receipts_by_remote_path: dict[str, dict[str, object]] = {}
        actual_paths = _actual_receipt_paths()
        self.assertEqual(set(actual_paths), {arm["arm"] for arm in binding["arms"]})
        receipt_count = 0
        for arm in binding["arms"]:
            for pin, path in zip(
                (arm["terminal_acceptance"], arm["exact_stop"]),
                actual_paths[arm["arm"]],
                strict=True,
            ):
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                self.assertEqual((len(raw), digest), (pin["bytes"], pin["sha256"]))
                value = json.loads(raw)
                self.assertIsInstance(value, dict)
                receipts_by_remote_path[pin["absolute_path"]] = value
                receipt_count += 1
        self.assertEqual(receipt_count, 10)
        control_acceptance_path = binding["arms"][0]["terminal_acceptance"]["absolute_path"]
        self.assertNotIn(
            "mode", receipts_by_remote_path[control_acceptance_path]["input_representation"]
        )

        with tempfile.TemporaryDirectory(prefix="export-v3-receipt-model-gate-") as temp:
            project_base = Path(temp).resolve()
            data_root = project_base / "data"
            data_root.mkdir()
            split_path = project_base / "splits/fold0.json"
            split_path.parent.mkdir(parents=True)
            split_path.write_bytes(b"synthetic path placeholder; digest is faked")
            binding["project_base"] = str(project_base)
            binding["data_root"] = str(data_root)
            binding["split"]["absolute_path"] = str(split_path)
            model_pins: dict[Path, tuple[int, str]] = {}
            for arm in binding["arms"]:
                model_path = data_root / arm["model"]["relative_path"]
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model_pins[model_path] = (arm["model"]["bytes"], arm["model"]["sha256"])

            def fake_digest_file(path: Path, max_bytes: int) -> tuple[int, str]:
                if path == split_path:
                    self.assertLessEqual(1, max_bytes)
                    return 1, binding["split"]["sha256"]
                if path not in model_pins:
                    raise AssertionError(f"unexpected payload digest request: {path}")
                expected = model_pins[path]
                self.assertLessEqual(expected[0], max_bytes)
                return expected

            def read_exact_local_receipt(
                pin: dict[str, object], candidate_binding: dict[str, object]
            ) -> dict[str, object]:
                del candidate_binding
                return receipts_by_remote_path[pin["absolute_path"]]

            with (
                patch.object(exporter, "_read_json_pin", side_effect=read_exact_local_receipt),
                patch.object(exporter, "_digest_file", side_effect=fake_digest_file),
            ):
                exporter._verify_receipts_and_models(data_root, binding)


if __name__ == "__main__":
    unittest.main()
