"""Exercise the real export metadata gate with bounded synthetic metadata."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PACKET = Path(__file__).resolve().parent
PREPARATION = PACKET.parent
EXPORT_V4 = PREPARATION / "first_cell_evaluation_export_v4"
SCORE_V4 = PREPARATION / "first_cell_evaluation_score_v4"
sys.path.insert(0, str(PACKET))

import first_cell_score_contract as contract  # noqa: E402
import first_cell_score_core as core  # noqa: E402


def _read_small_json(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise AssertionError(f"expected a JSON object at {path}")
    return value


class ScoreV5ApiManifestGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binding = _read_small_json(EXPORT_V4 / "EXPORT_BINDING_V4.json")
        cls.source_manifest = _read_small_json(EXPORT_V4 / "EXPORT_SOURCE_MANIFEST.json")
        cls.binding_raw = (EXPORT_V4 / "EXPORT_BINDING_V4.json").read_bytes()
        cls.source_raw = (EXPORT_V4 / "EXPORT_SOURCE_MANIFEST.json").read_bytes()
        cls.evidence_root = Path(str(cls.binding["evidence_root"]))
        cls.export_root = cls.evidence_root / "exports"
        cls.binding_path = cls.evidence_root / "EXPORT_BINDING_V4.json"
        cls.source_manifest_path = cls.evidence_root / "EXPORT_SOURCE_MANIFEST.json"
        cls.groups = list(range(core.N_GROUPS))
        cls.completion_manifest = cls._completion_manifest()

    @classmethod
    def _completion_manifest(cls) -> dict[str, object]:
        registered = cls.binding["registered_data"]
        sources = copy.deepcopy(registered["source_identities"])
        source_observations = {
            relative: {
                key: entry[key]
                for key in ("path", "resolved", "bytes", "device", "inode", "islink", "mtime_ns")
            }
            for relative, entry in sources.items()
        }
        valid = registered["valid_parquet"]
        valid_observation = {
            key: valid[key] for key in ("path", "resolved", "bytes", "device", "inode", "mtime_ns")
        }
        valid_observation["islink"] = False
        valid_observation["parent"] = {
            "path": str(Path(valid["path"]).parent),
            "resolved": core.EXPECTED_PARENT_RESOLUTIONS["datasets"],
            "device": 64512,
            "inode": 987654321,
        }
        data_identity = {
            "schema": "fg19.registered-valid-data-identity.v1",
            "admission": copy.deepcopy(registered["admission"]),
            "portable_manifest": {
                **copy.deepcopy(registered["portable_manifest"]),
                "source_names": list(core.EXPECTED_PORTABLE_SOURCE_NAMES),
                "split_manifest_sha256": core.EXPECTED_SPLIT_SHA256,
                "version": 1,
            },
            "registered_source_identities": sources,
            "registered_valid_parquet": copy.deepcopy(valid),
            "observed_sources_before_split": source_observations,
            "observed_sources_after_split": copy.deepcopy(source_observations),
            "observed_before_sequence_open": valid_observation,
            "observed_after_export": copy.deepcopy(valid_observation),
        }
        return {
            "schema": "fg19.first-cell-evaluation-export-manifest.v1",
            "status": "COMPLETE_PREDICTION_EXPORTS_NOT_SCORED",
            "created_utc": "2026-10-21T00:00:00Z",
            "binding_sha256": core.EXPECTED_BINDING_SHA256,
            "source": copy.deepcopy(cls.binding["source"]),
            "split": {
                **copy.deepcopy(cls.binding["split"]),
                "role": "fit-excluded-exposed-development-evaluation",
            },
            "data_identity": data_identity,
            "ordered_groups": cls.groups,
            "membership_sha256": core.membership_sha256(cls.groups),
            "arms": [],
            "quality_or_metric_claim": False,
        }

    def _run_gate(self, binding: dict[str, object]) -> tuple[dict, dict, dict, dict, tuple]:
        binding_pin = {
            "path": str(self.binding_path),
            "bytes": len(self.binding_raw),
            "sha256": hashlib.sha256(self.binding_raw).hexdigest(),
        }
        source_pin = {
            "path": str(self.source_manifest_path),
            "bytes": len(self.source_raw),
            "sha256": hashlib.sha256(self.source_raw).hexdigest(),
        }
        manifest_pin = {
            "path": str(self.export_root / "EXPORT_MANIFEST_V1.json"),
            "bytes": 1,
            "sha256": "0" * 64,
        }

        def read_json(path: Path, _cap: int) -> tuple[dict, dict, tuple[int, int, int, int]]:
            if path == self.binding_path:
                return copy.deepcopy(binding), binding_pin, (1, 2, 3, 4)
            if path == self.source_manifest_path:
                return copy.deepcopy(self.source_manifest), source_pin, (1, 2, 3, 4)
            if path == self.export_root / "EXPORT_MANIFEST_V1.json":
                return copy.deepcopy(self.completion_manifest), manifest_pin, (1, 2, 3, 4)
            raise AssertionError(f"unexpected metadata path: {path}")

        original_is_dir = Path.is_dir
        original_resolve = Path.resolve

        def is_dir(path: Path) -> bool:
            if path in (self.evidence_root, self.export_root):
                return True
            return original_is_dir(path)

        def is_symlink(_path: Path) -> bool:
            return False

        def resolve(path: Path, strict: bool = False) -> Path:
            if path in (self.evidence_root, self.export_root):
                return path
            return original_resolve(path, strict=strict)

        with (
            patch.object(contract, "_read_json", side_effect=read_json),
            patch.object(contract, "_safe_child", side_effect=lambda root, name: root / name),
            patch.object(Path, "is_dir", is_dir),
            patch.object(Path, "is_symlink", is_symlink),
            patch.object(Path, "resolve", resolve),
        ):
            return contract._read_export_bundle(
                self.evidence_root,
                self.export_root,
                self.binding_path,
                self.source_manifest_path,
            )

    def test_actual_v4_binding_and_source_manifest_pass_full_bundle_gate(self) -> None:
        binding, source_manifest, manifest, _pin, _identity = self._run_gate(self.binding)
        self.assertEqual(binding, self.binding)
        self.assertEqual(source_manifest, self.source_manifest)
        self.assertEqual(manifest["binding_sha256"], core.EXPECTED_BINDING_SHA256)
        self.assertEqual(len(binding["source"]["api_files"]), 8)
        self.assertEqual(binding["source"]["file_count"], 167)
        self.assertEqual(len(source_manifest["files"]), 167)

    def test_only_the_api_row_join_changed_from_score_v4(self) -> None:
        old_block = (
            '    api_files = source.get("api_files")\n'
            "    if not isinstance(api_files, dict) or any(\n"
            "        item != source_file_pins.get(relative) for relative, item in api_files.items()\n"
            "    ):\n"
            '        raise ValueError("export binding canonical API pins differ from the source manifest")\n'
        )
        new_block = (
            '    api_files = source.get("api_files")\n'
            "    if not isinstance(api_files, dict):\n"
            '        raise ValueError("export binding canonical API pins differ from the source manifest")\n'
            "    for relative, item in api_files.items():\n"
            "        source_pin = source_file_pins.get(relative)\n"
            "        if (\n"
            "            not isinstance(relative, str)\n"
            "            or not isinstance(source_pin, dict)\n"
            "            or not isinstance(item, dict)\n"
            '            or set(item) != {"bytes", "sha256"}\n'
            '            or type(item.get("bytes")) is not int\n'
            '            or item["bytes"] <= 0\n'
            '            or not isinstance(item.get("sha256"), str)\n'
            '            or len(item["sha256"]) != 64\n'
            '            or any(character not in "0123456789abcdef" for character in item["sha256"])\n'
            '            or item["bytes"] != source_pin.get("bytes")\n'
            '            or item["sha256"] != source_pin.get("sha256")\n'
            "        ):\n"
            '            raise ValueError("export binding canonical API pins differ from the source manifest")\n'
        )
        prior = (SCORE_V4 / "first_cell_score_contract.py").read_text(encoding="utf-8")
        current = (PACKET / "first_cell_score_contract.py").read_text(encoding="utf-8")
        self.assertEqual(prior.count(old_block), 1)
        self.assertEqual(current.count(new_block), 1)
        self.assertEqual(prior.replace(old_block, new_block), current)
        for name in (
            "first_cell_score_core.py",
            "first_cell_score_predictions.py",
            "score_first_cell.py",
        ):
            self.assertEqual((PACKET / name).read_bytes(), (SCORE_V4 / name).read_bytes())

    def test_api_row_mutations_fail_the_actual_full_bundle_gate(self) -> None:
        mutations = {
            "wrong byte count": lambda api: api["src/wnn/data.py"].__setitem__("bytes", 1),
            "wrong digest": lambda api: api["src/wnn/data.py"].__setitem__("sha256", "0" * 64),
            "wrong path": lambda api: api.__setitem__(
                "src/wnn/not-data.py", api.pop("src/wnn/data.py")
            ),
            "unexpected row field": lambda api: api["src/wnn/data.py"].__setitem__(
                "path", "src/wnn/data.py"
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                binding = copy.deepcopy(self.binding)
                mutate(binding["source"]["api_files"])
                # Keep the exact on-disk pin to isolate the later structural guard;
                # these injected objects do not represent altered binding bytes.
                with self.assertRaisesRegex(
                    ValueError, "export binding canonical API pins differ from the source manifest"
                ):
                    self._run_gate(binding)


if __name__ == "__main__":
    unittest.main()
