"""Focused offline fixtures for the frozen CONTROL terminal verifier adapter."""

from __future__ import annotations

import ast
import base64
import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

PACKET = Path(__file__).resolve().parent
CAMPAIGN = PACKET.parents[1]
RECORDS = CAMPAIGN / "records"
TEMPLATE = PACKET / "REMOTE_SUBMISSION_TEMPLATE.py"
NATIVE_CAPTURE = RECORDS / "child_control_fit_v1/terminal_native_capture_v1/STDOUT.raw.json"


def load_adapter() -> dict[str, object]:
    """Load only adapter definitions; omit the template's top-level main call."""
    tree = ast.parse(TEMPLATE.read_text(encoding="utf-8"), filename=str(TEMPLATE))
    final = tree.body[-1]
    if not isinstance(final, ast.Expr) or ast.unparse(final.value) != "main()":
        raise AssertionError("template must end in exactly one main call")
    tree.body.pop()
    namespace: dict[str, object] = {"__name__": "offline_adapter_fixture"}
    exec(compile(ast.fix_missing_locations(tree), str(TEMPLATE), "exec"), namespace)
    return namespace


class ControlVerifierFixtureTests(unittest.TestCase):
    """Exercise native capture schema and stat-only gates with synthetic files."""

    @classmethod
    def setUpClass(cls) -> None:
        """Read the bounded native capture metadata, not any model or state files."""
        raw = NATIVE_CAPTURE.read_bytes()
        if len(raw) > 75_000:
            raise AssertionError("native capture exceeds its fixed metadata cap")
        cls.capture = json.loads(raw)

    def test_actual_native_receipts_and_run_inventory_shape(self) -> None:
        """Validate receipt and run-stat fields from the actual stored terminal capture."""
        self.assertEqual(self.capture["schema"], "fg19-child-control-startup-capture.v2")
        receipts = self.capture["receipts"]
        for name in ("ADMISSION.json", "LAUNCH.json", "unit.log"):
            with self.subTest(receipt=name):
                record = receipts[name]
                self.assertTrue(record["present"])
                self.assertTrue(record["regular_file"])
                decoded = base64.b64decode(record["base64"], validate=True)
                self.assertEqual(len(decoded), record["bytes"])
                self.assertEqual(hashlib.sha256(decoded).hexdigest(), record["sha256"])

        inventory = self.capture["run_directory_stat_only"]
        self.assertTrue(inventory["exists"])
        self.assertFalse(inventory["symlink"])
        names = {row["name"] for row in inventory["entries"]}
        self.assertEqual(
            names,
            {
                "model.pt",
                *(f"model_ep{epoch}.pt" for epoch in range(6)),
                "training_state.pt",
                "provenance.json",
                "history.json",
            },
        )
        self.assertTrue(
            {"bytes", "device", "inode", "mtime_ns", "mode", "is_symlink"}.issubset(
                inventory["entries"][0]
            )
        )

        namespace = load_adapter()
        with tempfile.TemporaryDirectory(dir=PACKET) as temporary:
            run = Path(temporary) / "run"
            run.mkdir()
            namespace["RUN"] = run
            synthetic_entries = []
            for name in sorted(names):
                path = run / name
                path.write_bytes(b"")
                info = path.lstat()
                synthetic_entries.append(
                    {
                        "name": name,
                        "bytes": info.st_size,
                        "device": info.st_dev,
                        "inode": info.st_ino,
                        "mtime_ns": info.st_mtime_ns,
                        "mode": info.st_mode,
                        "is_symlink": False,
                    }
                )
            fixture = {
                "run_directory_stat_only": {
                    "exists": True,
                    "symlink": False,
                    "entries": synthetic_entries,
                }
            }
            result = namespace["check_run_stats"](fixture)
            self.assertEqual(result["payload_bytes_read"], 0)
            broken = copy.deepcopy(fixture)
            broken["run_directory_stat_only"]["entries"][0]["inode"] += 1
            with self.assertRaisesRegex(RuntimeError, "terminal stat differs"):
                namespace["check_run_stats"](broken)

    def test_source_gate_accepts_167_and_rejects_count_or_digest_drift(self) -> None:
        """Use a tiny synthetic tree to exercise the exact 167-file streaming gate."""
        namespace = load_adapter()
        with tempfile.TemporaryDirectory(dir=PACKET) as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            namespace["SOURCE"] = source
            rows = []
            for index in range(167):
                content = f"synthetic source {index}\n".encode()
                relative = f"src/file_{index:03}.py"
                path = source / relative
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(content)
                rows.append(
                    {
                        "path": relative,
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
            manifest = json.dumps(
                {"source_commit": namespace["COMMIT"], "files": rows},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            admission = {
                "source_files_verified": 167,
                "source_manifest_sha256": namespace["MANIFEST_SHA"],
            }
            result = namespace["check_source"]({"SOURCE_MANIFEST.json": manifest}, admission)
            self.assertEqual(result["file_count"], 167)

            short_manifest = json.dumps(
                {"source_commit": namespace["COMMIT"], "files": rows[:-1]},
                separators=(",", ":"),
            ).encode("utf-8")
            with self.assertRaisesRegex(RuntimeError, "source manifest identity differs"):
                namespace["check_source"]({"SOURCE_MANIFEST.json": short_manifest}, admission)

            drifted = copy.deepcopy(rows)
            drifted[0]["sha256"] = "0" * 64
            drift_manifest = json.dumps(
                {"source_commit": namespace["COMMIT"], "files": drifted},
                separators=(",", ":"),
            ).encode("utf-8")
            with self.assertRaisesRegex(RuntimeError, "source pin differs"):
                namespace["check_source"]({"SOURCE_MANIFEST.json": drift_manifest}, admission)

    def test_predecessor_receipts_and_accepted_artifact_stats(self) -> None:
        """Hash small retained receipts and verify ancestor artifact stats only."""
        namespace = load_adapter()
        receipt_sources = {
            "PARENT_TERMINAL_ACCEPTANCE.json": RECORDS
            / "parent_fit_r2_v1/TERMINAL_ACCEPTANCE.json",
            "PARENT_EXACT_STOP.json": RECORDS / "parent_fit_r2_v1/exact_stop/STDOUT.raw.json",
            "DICTIONARY_ACCEPTANCE.json": RECORDS / "dictionary_f0_v1/ACCEPTANCE.json",
            "DICTIONARY_EXACT_STOP.json": RECORDS / "dictionary_f0_v1/exact_stop/STDOUT.raw.json",
            "PROFILE_TERMINAL_ACCEPTANCE.json": RECORDS
            / "child_allthree_profile_v1/TERMINAL_ACCEPTANCE.json",
            "PROFILE_EXACT_STOP.json": RECORDS
            / "child_allthree_profile_v1/exact_stop/STDOUT.raw.json",
        }
        profile_sources = {
            "ADMISSION.json": RECORDS / "child_allthree_profile_v1/ADMISSION.json",
            "LAUNCH.json": RECORDS / "child_allthree_profile_v1/LAUNCH.json",
            "unit.log": RECORDS / "child_allthree_profile_v1/capture/unit.log",
        }
        with tempfile.TemporaryDirectory(dir=PACKET) as temporary:
            root = Path(temporary)
            evidence = root / "evidence"
            profile = root / "profile"
            evidence.mkdir()
            profile.mkdir()
            for name, source in receipt_sources.items():
                raw = source.read_bytes()
                if len(raw) > 20_000:
                    self.fail(f"unexpectedly large metadata receipt: {name}")
                shutil.copyfile(source, evidence / name)
            for name, source in profile_sources.items():
                raw = source.read_bytes()
                if len(raw) > 20_000:
                    self.fail(f"unexpectedly large profile proof: {name}")
                shutil.copyfile(source, profile / name)

            parent = root / "accepted-parent.pt"
            dictionary = root / "accepted-dictionary.pt"
            parent.write_bytes(b"synthetic parent stat fixture")
            dictionary.write_bytes(b"synthetic dictionary stat fixture")
            namespace["EVIDENCE"] = evidence
            namespace["PROFILE_EVIDENCE"] = profile
            namespace["PARENT_CHECKPOINT"] = parent
            namespace["DICT"] = dictionary
            parent_stat = parent.stat()
            dictionary_stat = dictionary.stat()
            admission = {
                "staged_receipt_sha256": namespace["READONLY_RECEIPTS"],
                "parent_checkpoint_sha256": namespace["PARENT_CHECKPOINT_SHA"],
                "parent_checkpoint_identity": {
                    "path": str(parent),
                    "sha256": namespace["PARENT_CHECKPOINT_SHA"],
                    "bytes": parent_stat.st_size,
                    "device": parent_stat.st_dev,
                    "inode": parent_stat.st_ino,
                    "mtime_ns": parent_stat.st_mtime_ns,
                },
                "dictionary": {
                    "path": str(dictionary),
                    "sha256": namespace["DICTIONARY_SHA"],
                    "bytes": dictionary_stat.st_size,
                    "device": dictionary_stat.st_dev,
                    "inode": dictionary_stat.st_ino,
                },
            }
            result = namespace["check_predecessors"](admission)
            self.assertFalse(result["payload_bytes_hashed"])
            self.assertEqual(result["parent_checkpoint_stat_only"]["inode"], parent_stat.st_ino)
            self.assertEqual(
                result["dictionary_artifact_stat_only"]["inode"], dictionary_stat.st_ino
            )

            wrong_parent = copy.deepcopy(admission)
            wrong_parent["parent_checkpoint_identity"]["inode"] += 1
            with self.assertRaisesRegex(RuntimeError, "parent checkpoint stat differs"):
                namespace["check_predecessors"](wrong_parent)
            wrong_dictionary = copy.deepcopy(admission)
            wrong_dictionary["dictionary"]["inode"] += 1
            with self.assertRaisesRegex(RuntimeError, "dictionary artifact stat differs"):
                namespace["check_predecessors"](wrong_dictionary)


if __name__ == "__main__":
    unittest.main()
