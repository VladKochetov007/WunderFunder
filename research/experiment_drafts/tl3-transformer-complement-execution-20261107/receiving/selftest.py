from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import owned_phase
from controller import Controller
from receiving_state import (
    COMMIT,
    SHORT_RUN_ROOT,
    SOURCE_MANIFEST_SHA,
    SPLIT_SHA,
    approval_ready,
    digest,
    process_family_records,
    reference_ready,
)


def write(path: Path, value: object, *, replace: bool = True) -> None:
    """Write tiny synthetic receipts only; never invoke an actual receiving command."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    path.write_text(json.dumps(value))


def stub(root: Path) -> Controller:
    """Build a CPU-only controller seam without real source/data/ownership prerequisites."""
    value = Controller.__new__(Controller)
    value.runtime, value.control = root, root / "control"
    value.source, value.run_root = root / "source-r3", root / "b300"
    value.lock, value.stop = threading.RLock(), threading.Event()
    value.h = SimpleNamespace(write=write)
    value.records, value.resource_identities = {}, {}
    value.owner = {"pid": 10, "birth": "100", "pgid": 10}
    value.state = {"cells": {"labels_s101001": {}}, "fit_attempts": [], "profile_attempts": []}
    value.amendment_sha, value.error = None, None
    value.args = SimpleNamespace(python=Path("/provided/python"), support_src=Path("/support/src"))
    return value


class ReceivingTests(unittest.TestCase):
    """Exercise the narrow receiver gates without staging, models, GPU or process launches."""

    def test_child_environment_uses_actual_source_short_alias_and_compiler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            value = stub(Path(directory))
            env = value.environment("2,3")
            self.assertEqual(env["WNN_RUN_ROOT"], str(SHORT_RUN_ROOT))
            self.assertEqual(env["PYTHONPATH"], str(value.source / "src"))
            self.assertEqual(env["WNN_EXPECTED_COMMIT"], COMMIT)
            self.assertEqual(env["WNN_SOURCE_MANIFEST_SHA256"], SOURCE_MANIFEST_SHA)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,3")
            self.assertEqual((env["CC"], env["CXX"]), ("/usr/bin/gcc", "/usr/bin/g++"))
            self.assertEqual(env["TORCHINDUCTOR_COMPILE_THREADS"], "4")

    def test_approval_preserves_native_measurements_and_requires_ROOT(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, cell = Path(directory), "labels_s101001"
            self.assertFalse(approval_ready(root, cell))
            native = {
                "status": "profile_complete",
                "updates": 12,
                "errors": [],
                "scientific_checkpoint_written": False,
                "identity": {"world_size": 2},
                "approved": False,
                "elapsed_seconds": 1.2,
            }
            write(root / cell / "profile-two/profile_receipt.json", native)
            approved = native | {"approved": True, "reviewer": "ROOT"}
            path = root / cell / "approved_profile.json"
            write(path, approved)
            self.assertTrue(approval_ready(root, cell))
            approved["elapsed_seconds"] = 0.1
            write(path, approved)
            with self.assertRaises(ValueError):
                approval_ready(root, cell)

    def test_reference_requires_bound_baseline_and_same_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(reference_ready(root))
            for membership in ("fit", "evaluation"):
                write(
                    root / "reference" / membership / "manifest.json",
                    {
                        "complete": True,
                        "membership": membership,
                        "split_sha256": SPLIT_SHA,
                        "reference_sha256": "fixed-reference",
                    },
                )
            baseline_path = root / "b300/baseline-score.json"
            baseline = {
                "schema": "wnn.tl3-reference-score.v1",
                "split_sha256": SPLIT_SHA,
                "sequences": 1873,
                "producer": {"reference_sha256": "fixed-reference"},
                "predictions_manifest_sha256": digest(root / "reference/evaluation/manifest.json"),
                "score": {"metric_id": "global_wp_clip2_v1", "wp": 0.6923774245652725},
            }
            write(baseline_path, baseline)
            receipt_path = root / "receipts/reference-preparation-complete-r1.json"
            receipt = {
                "status": "completed",
                "split_sha256": SPLIT_SHA,
                "baseline_score_sha256": digest(baseline_path),
            }
            write(receipt_path, receipt)
            self.assertTrue(reference_ready(root))
            baseline["score"]["wp"] += 0.0001
            write(baseline_path, baseline)
            receipt["baseline_score_sha256"] = digest(baseline_path)
            write(receipt_path, receipt)
            with self.assertRaises(ValueError):
                reference_ready(root)

    def test_authority_starts_once_and_amendment_precedes_fits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            value = stub(Path(directory))
            self.assertIsNone(value.authority())
            base = value.authority(start=True)
            self.assertEqual(base["deadline_epoch"] - base["first_profile_epoch"], 86400)
            amendment = base | {
                "approved_by": "ROOT",
                "reason": "measured profile duration",
                "previous_authority_sha256": digest(value.control / "authority.json"),
                "deadline_epoch": base["deadline_epoch"] + 86400,
                "fit_deadline_epoch": base["fit_deadline_epoch"] + 86400,
            }
            write(value.control / "ROOT-authority-amendment.json", amendment)
            self.assertEqual(value.authority()["deadline_epoch"], amendment["deadline_epoch"])
            value.amendment_sha = None
            value.state["fit_attempts"].append({"cell": "already-admitted"})
            with self.assertRaises(ValueError):
                value.authority()

    def test_phase_does_not_retry_and_does_not_duplicate_observer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            value = stub(Path(directory))
            value.guard = lambda **_kwargs: None
            calls = []

            def run(command: list[str], **kwargs: object) -> tuple[int, bool, dict[str, object]]:
                calls.append(command)
                write(kwargs["receipt_path"], {"status": "completed"})
                return 0, False, {}

            value.h.run = run
            value.execute("profile-one", "labels_s101001", "0,1")
            self.assertEqual(len(calls), 1)
            self.assertTrue(any("owned_phase.py" in item for item in calls[0]))
            self.assertTrue(any("run.sh" in item for item in calls[0]))
            self.assertFalse(any("observe_sequence.py" in item for item in calls[0]))
            with self.assertRaises(FileExistsError):
                value.execute("profile-one", "labels_s101001", "0,1")
            self.assertEqual(len(calls), 1)

    def test_reserve_closes_fit_admission_but_retains_collection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            value = stub(Path(directory))
            value.authority = lambda **_kwargs: {"fit_deadline_epoch": 1000, "deadline_epoch": 1100}
            value.last_resource_check = time.monotonic()
            value.terminate = Mock()
            with patch("controller.time.time", return_value=1001):
                with self.assertRaises(RuntimeError):
                    value.guard(admission=True)
                value.guard()
            with patch("controller.time.time", return_value=1101), self.assertRaises(RuntimeError):
                value.guard()
            value.terminate.assert_called_once()

    def test_relay_catches_TERM_before_spawn_and_reaches_owned_cleanup(self) -> None:
        calls = []
        process = Mock()
        process.wait.return_value = process.poll.return_value = 2
        owner = {"pid": 13, "pgid": 13, "birth": "100"}
        helper = SimpleNamespace(
            owner=lambda: owner,
            write=lambda *_args: None,
            terminate=lambda *_args, **_kwargs: calls.append("kill"),
        )
        with (
            patch.object(owned_phase.os, "name", "posix"),
            patch.object(owned_phase.os, "getpid", return_value=13),
            patch.object(owned_phase.os, "getpgrp", return_value=13, create=True),
            patch("sys.argv", ["owned_phase.py", "--support-src", "/support", "--", "bash"]),
            patch.object(owned_phase, "load_helpers", return_value=helper),
            patch.object(
                owned_phase, "become_subreaper", side_effect=lambda: calls.append("reaper")
            ),
            patch.object(owned_phase, "process_family_records", return_value={13: owner}),
            patch.dict(
                "os.environ", {"WNN_CONTROLLER_DESCENDANTS_RECEIPT": "/synthetic/proof.json"}
            ),
            patch.object(
                owned_phase.subprocess,
                "Popen",
                side_effect=lambda _args: calls.append("spawn") or process,
            ),
            patch.object(
                owned_phase.signal, "signal", side_effect=lambda *_args: calls.append("handler")
            ),
        ):
            self.assertEqual(owned_phase.main(), 2)
        self.assertEqual(calls, ["reaper", "handler", "spawn", "kill"])

    def test_descendants_include_separate_sessions_and_exclude_reused_foreign_pid(self) -> None:
        def proc(pid: int, parent: int, birth: int, group: int) -> Mock:
            fields = ["0"] * 22
            fields[0], fields[1], fields[2], fields[19], fields[21] = (
                "S",
                str(parent),
                str(group),
                str(birth),
                "2",
            )
            path = Mock()
            path.parent.name = str(pid)
            path.read_text.return_value = f"{pid} (worker) " + " ".join(fields)
            return path

        identities = {13: 100}
        with patch(
            "receiving_state.Path.glob",
            return_value=[proc(13, 1, 100, 13), proc(21, 13, 200, 21), proc(99, 1, 900, 99)],
        ):
            records = process_family_records(identities)
        self.assertEqual(set(records), {13, 21})
        self.assertEqual(records[21]["pgid"], 21)
        with patch(
            "receiving_state.Path.glob", return_value=[proc(13, 1, 100, 13), proc(21, 13, 999, 21)]
        ):
            self.assertEqual(set(process_family_records(identities)), {13})

    def test_cleanup_stops_owned_rank_groups_before_leader(self) -> None:
        owner = {"pid": 13, "pgid": 13, "birth": "100"}
        rank = {"pid": 21, "pgid": 21, "birth": "200", "parent_pid": 13}
        calls = []
        helper = SimpleNamespace(
            write=lambda *_args: None,
            same=lambda _identity: True,
            terminate=lambda identity, **_kwargs: calls.append(identity["pid"]),
        )
        with (
            patch.object(
                owned_phase,
                "process_family_records",
                side_effect=[{13: owner, 21: rank}, {13: owner}, {13: owner}],
            ),
            patch.object(owned_phase.time, "sleep"),
        ):
            owned_phase.clean_descendant_groups(
                helper, owner, {13: 100, 21: 200}, Path("proof.json"), include_leader=True
            )
        self.assertEqual(calls, [21, 13])


if __name__ == "__main__":
    unittest.main()
