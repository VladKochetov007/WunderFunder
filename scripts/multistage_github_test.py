"""Small CPU-only integrity tests for the multistage GitHub snapshot path."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from multistage_github_common import (  # noqa: E402
    canonical_json,
    make_manifest,
    make_snapshot,
    safe_run_id,
    sha256_bytes,
    verify_snapshot,
)
from multistage_github_local_files import (  # noqa: E402
    ensure_local_mirror_root,
    safe_run_directory,
)
from multistage_github_source import (  # noqa: E402
    _journal,
    _preserve_metadata,
    _sanitize_event_context,
)


def _event(
    kind: str, step: int, metrics: dict[str, int | float] | None = None
) -> dict[str, object]:
    """Build one minimal real-schema monitor event for parser boundary tests."""
    return {
        "schema": "wnn.monitor.v1",
        "kind": kind,
        "step": step,
        "epoch": 1.0,
        "wall_time": "2026-10-31T00:00:00Z",
        "metrics": metrics or {},
        "context": {"status": "completed"} if kind == "status" else {},
    }


class GitHubSnapshotTests(unittest.TestCase):
    """Check traversal, stride disclosure, exact digests, and mirror root ownership."""

    def test_run_id_rejects_traversal(self) -> None:
        self.assertTrue(safe_run_id("stage1-gru-0123456789-abcd1234"))
        for value in ("../outside", "..", "stage/one", r"stage\one", "é-run"):
            self.assertFalse(safe_run_id(value))

    def test_full_ancestry_is_preserved_and_credentials_are_rejected_or_redacted(self) -> None:
        ancestry = [f"producer-{index}" for index in range(900)]
        metadata = {"provenance": {"identity": {"parents": ancestry}}}
        self.assertEqual(_preserve_metadata(metadata), metadata)
        context = _sanitize_event_context({"members": ancestry, "api_token": "secret-value"})
        self.assertEqual(len(context["members"]), 900)
        self.assertEqual(context["api_token"], "[redacted]")
        with self.assertRaises(ValueError):
            _preserve_metadata({"credential_token": "secret-value"})

    def test_train_step_sampling_preserves_other_event_kinds_and_full_source_hash(self) -> None:
        records = [_event("train_step", index, {"train/loss": float(index)}) for index in range(21)]
        records.extend((_event("status", 21), _event("diagnostics", 22), _event("resources", 23)))
        raw = b"".join(canonical_json(record) + b"\n" for record in records)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "monitor_events.jsonl"
            path.write_bytes(raw)
            result = _journal(path, evaluations_only=False)
        events = result["events"]
        self.assertEqual(result["source_sha256"], sha256_bytes(raw))
        self.assertEqual(result["source_complete_lines"], 24)
        self.assertEqual(result["sampling"]["stride"], 10)
        self.assertEqual(result["sampling"]["train_step_source_count"], 21)
        self.assertEqual(result["sampling"]["train_step_retained_count"], 3)
        self.assertEqual(
            [event["step"] for event in events if event["kind"] == "train_step"], [0, 10, 20]
        )
        self.assertEqual(
            {event["kind"] for event in events if event["kind"] != "train_step"},
            {"status", "diagnostics", "resources"},
        )

    def test_manifest_hashes_and_local_root_guard(self) -> None:
        source_sha = "a" * 64
        split_sha = "b" * 64
        primary_sha = "c" * 64
        config = {"model": {"architecture": "gru"}, "kd_weight": 0.0}
        metadata = {
            "source": {"sha256": source_sha, "files": {}},
            "config": config,
            "provenance": {
                "data": {
                    "evaluation_role": "control",
                    "split_sha256": split_sha,
                    "primary_split_sha256": primary_sha,
                    "teacher": None,
                    "initializer": None,
                }
            },
        }
        journal = {
            "source_sha256": "d" * 64,
            "source_bytes": 0,
            "source_complete_lines": 0,
            "source_partial_tail_bytes": 0,
            "retained_event_count": 0,
            "events": [],
        }
        run = {
            "run_id": "stage1-gru-0123456789-abcd1234",
            "stage": 1,
            "family": "gru",
            "config_sha256": sha256_bytes(canonical_json(config)),
            "source_sha256": source_sha,
            "split_sha256": split_sha,
            "primary_split_sha256": primary_sha,
            "metadata": metadata,
            "event_journal": journal,
            "evaluation_journal": journal,
        }
        payload, compressed, _ = make_snapshot([run])
        manifest = make_manifest(payload, compressed, "2026-10-31T00:00:00+00:00")
        body, restored = verify_snapshot(compressed, manifest)
        self.assertEqual(restored, payload)
        self.assertEqual(body["runs"][0]["run_id"], run["run_id"])
        with self.assertRaises(ValueError):
            verify_snapshot(compressed + b"x", manifest)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "mirror"
            ensure_local_mirror_root(root)
            self.assertTrue(safe_run_directory(root, run["run_id"]).is_dir())
            with self.assertRaises(ValueError):
                safe_run_directory(root, "../outside")


if __name__ == "__main__":
    unittest.main()
