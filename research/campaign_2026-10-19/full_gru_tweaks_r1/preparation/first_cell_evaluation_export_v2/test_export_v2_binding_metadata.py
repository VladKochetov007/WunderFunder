"""Focused metadata-only checks for the flat-support export binding correction."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[4]
V1 = ROOT / "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation"
V1 /= "first_cell_evaluation_export_v1"
DIAGNOSTIC = ROOT / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/first_cell_evaluation_v1/"
    "export_v1/preflight_outcome_v1/STDOUT.raw.json"
)


def _read_json(path: Path) -> tuple[bytes, dict[str, object]]:
    """Read a local source or metadata JSON document without importing it."""
    raw = path.read_bytes()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise AssertionError(f"expected JSON object at {path}")
    return raw, parsed


class ExportV2BindingMetadataTests(unittest.TestCase):
    def test_binding_delta_is_exactly_two_flat_support_paths(self) -> None:
        old_raw, old = _read_json(V1 / "EXPORT_BINDING_V1.json")
        new_raw, new = _read_json(PACKET / "EXPORT_BINDING_V1.json")
        old_loader = old["source"]["restricted_loader"]
        old_support = old["source"]["restricted_loader_support"]
        new_loader = new["source"]["restricted_loader"]
        new_support = new["source"]["restricted_loader_support"]
        self.assertEqual(
            (old_loader["relative_path"], new_loader["relative_path"]),
            ("preparation/parent_fit_terminal_check_v3.py", "parent_fit_terminal_check_v3.py"),
        )
        self.assertEqual(
            (old_support["relative_path"], new_support["relative_path"]),
            ("preparation/propensity_terminal_support.py", "propensity_terminal_support.py"),
        )
        self.assertEqual(new_loader["bytes"], old_loader["bytes"])
        self.assertEqual(new_loader["sha256"], old_loader["sha256"])
        self.assertEqual(new_support["bytes"], old_support["bytes"])
        self.assertEqual(new_support["sha256"], old_support["sha256"])
        inverse = new_raw.replace(
            b"parent_fit_terminal_check_v3.py", b"preparation/parent_fit_terminal_check_v3.py", 1
        ).replace(
            b"propensity_terminal_support.py",
            b"preparation/propensity_terminal_support.py",
            1,
        )
        self.assertEqual(inverse, old_raw)
        self.assertEqual(len(old_raw), 15_822)
        self.assertEqual(
            hashlib.sha256(old_raw).hexdigest(),
            "fcb3ee7934a19d023f29090e527be1d151dde23ae9fae3e0e3a911cea1bd809c",
        )
        self.assertEqual(len(new_raw), 15_798)
        self.assertEqual(
            hashlib.sha256(new_raw).hexdigest(),
            "6cd8c46f31214623d97e54c32bac9395608bfc918b9b44e879e44ad2548f0eb9",
        )

    def test_exporter_diff_is_only_the_exact_binding_pin_pair(self) -> None:
        old = (V1 / "export_first_cell_v1.py").read_bytes()
        new = (PACKET / "export_first_cell_v1.py").read_bytes()
        old_bytes = b"BINDING_BYTES = 15_822"
        new_bytes = b"BINDING_BYTES = 15_798"
        old_sha = (
            b'BINDING_SHA256 = "fcb3ee7934a19d023f29090e527be1d151dde23ae9fae3e0e3a911cea1bd809c"'
        )
        new_sha = (
            b'BINDING_SHA256 = "6cd8c46f31214623d97e54c32bac9395608bfc918b9b44e879e44ad2548f0eb9"'
        )
        self.assertEqual(new.count(new_bytes), 1)
        self.assertEqual(new.count(new_sha), 1)
        self.assertEqual(new.replace(new_bytes, old_bytes, 1).replace(new_sha, old_sha, 1), old)
        self.assertEqual(
            (PACKET / "EXPORT_SOURCE_MANIFEST.json").read_bytes(),
            (V1 / "EXPORT_SOURCE_MANIFEST.json").read_bytes(),
        )

    def test_failed_v2_preflight_records_the_flat_helper_layout(self) -> None:
        raw, diagnostic = _read_json(DIAGNOSTIC)
        self.assertEqual(len(raw), 1_509)
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            "40a55d9e1c2d251627e6208775c0736699e3f1aa0378cdbc2ba7b126acd126c2",
        )
        self.assertEqual(diagnostic["schema"], "fg19.first-cell-export-preflight-outcome.v1")
        self.assertTrue(diagnostic["no_mutations"])
        self.assertFalse(diagnostic["evidence_lexists"])
        self.assertFalse(diagnostic["nested_preparation_lexists"])
        self.assertEqual(
            diagnostic["unit_properties"],
            {
                "ActiveState": "inactive",
                "Id": "computer-vision-homework-fg19-first-cell-export-v1.service",
                "InvocationID": "",
                "LoadState": "not-found",
                "MainPID": "0",
                "SubState": "dead",
            },
        )
        binding = json.loads((PACKET / "EXPORT_BINDING_V1.json").read_bytes())
        support_root = binding["support_root"]
        helper_entries = {
            "parent_fit_terminal_check_v3.py": binding["source"]["restricted_loader"],
            "propensity_terminal_support.py": binding["source"]["restricted_loader_support"],
        }
        observed = diagnostic["support_flat_code_stat_only"]
        expected = {
            "parent_fit_terminal_check_v3.py": (64_550, 64512, 739549298),
            "propensity_terminal_support.py": (13_107, 64512, 739549299),
            "teacher_s5_terminal_check_v1.py": (37_448, 64512, 739549300),
        }
        for filename, (byte_count, device, inode) in expected.items():
            item = observed[filename]
            self.assertEqual(item["path"], f"{support_root}/{filename}")
            self.assertEqual(
                (item["bytes"], item["device"], item["inode"]), (byte_count, device, inode)
            )
            self.assertFalse(item["islink"])
        self.assertEqual(
            (
                helper_entries["parent_fit_terminal_check_v3.py"]["relative_path"],
                helper_entries["parent_fit_terminal_check_v3.py"]["bytes"],
                helper_entries["parent_fit_terminal_check_v3.py"]["sha256"],
            ),
            (
                "parent_fit_terminal_check_v3.py",
                64_550,
                "f2b819c1e105171f0093fae433aec301347645f8f64717effa16917005dbfe41",
            ),
        )
        self.assertEqual(
            (
                helper_entries["propensity_terminal_support.py"]["relative_path"],
                helper_entries["propensity_terminal_support.py"]["bytes"],
                helper_entries["propensity_terminal_support.py"]["sha256"],
            ),
            (
                "propensity_terminal_support.py",
                13_107,
                "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409",
            ),
        )


if __name__ == "__main__":
    unittest.main()
