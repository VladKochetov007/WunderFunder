from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import wave_end_auditor as auditor


class FakeProcfs:
    def __init__(self, rows: list[auditor.ProcessRow]) -> None:
        self.rows = {row.pid: row for row in rows}

    def all(self) -> list[auditor.ProcessRow]:
        return list(self.rows.values())

    def read(self, pid: int) -> auditor.ProcessRow | None:
        return self.rows.get(pid)


class ReceiptAndBoundaryTests(unittest.TestCase):
    def test_extracts_nested_supported_pid_birth_shapes(self) -> None:
        data = {
            "owner": {"pid": 101, "birth_ticks": 9001},
            "child": {"child_pid": 102, "child_birth": 9002},
            "owned_pid_births": {"103": 9003, "104": {"birth_ticks": 9004}},
        }
        found = auditor.extract_identities(data, "owner.json")
        self.assertEqual(
            {(item.pid, item.birth_ticks) for item in found},
            {(101, 9001), (102, 9002), (103, 9003), (104, 9004)},
        )

    def test_role_root_allowlists_include_partial_e3_and_r4_readout_controls(
        self,
    ) -> None:
        self.assertTrue(
            auditor.allowed_root(
                "/data/epanov_s/tabdata-all-20261112-r1/control/partial-e3-readout-r1",
                "b300",
            )
        )
        self.assertTrue(
            auditor.allowed_root(
                "/data2/lc-epanov_s/tabdata-followup-readouts-20261112-r4/control/canonical-readouts-r3",
                "l4",
            )
        )
        self.assertFalse(
            auditor.allowed_root(
                "/data/otheruser/tabdata-all-20261112-r1/control/impact81-gru-queue-r1",
                "b300",
            )
        )
        self.assertFalse(
            auditor.allowed_root(
                "/data/epanov_s/otherproject/control/impact81-gru-queue-r1", "b300"
            )
        )
        self.assertFalse(
            auditor.allowed_root(
                "/data/epanov_s/tabdata-all-20261112-r1/control/../otherproject",
                "b300",
            )
        )
        self.assertFalse(
            auditor.allowed_root(
                "/data/epanov_s/tabdata-all-20261112-r1/control/score-attacher",
                "b300",
            )
        )

    def test_receipt_path_must_be_lexically_contained(self) -> None:
        root = "/data/epanov_s/tabdata-all-20261112-r1/control/impact81-gru-queue-r1"
        self.assertTrue(auditor.path_is_within(root + "/owner.json", root))
        self.assertFalse(auditor.path_is_within(root + "/../other/owner.json", root))


class OwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            auditor.ProcessRow(10, 100, 500, 1000, "python", "python3"),
            auditor.ProcessRow(11, 110, 10, 1000, "worker", "worker"),
            auditor.ProcessRow(12, 120, 11, 1000, "child", "child"),
            auditor.ProcessRow(13, 130, 10, 2000, "foreign", "foreign"),
            auditor.ProcessRow(14, 140, 10, 1000, "sshd", "sshd"),
            auditor.ProcessRow(15, 150, 10, 1000, "agent", "rpcbind"),
            auditor.ProcessRow(16, 160, 500, 1000, "unrelated", "unrelated"),
            auditor.ProcessRow(17, 170, 11, 1000, "zombie", "worker", "Z"),
            auditor.ProcessRow(999, 1999, 10, 1000, "auditor", "python3"),
            auditor.ProcessRow(500, 5000, 1, 1000, "sshd", "sshd"),
        ]
        self.procfs = FakeProcfs(self.rows)

    def test_snapshot_only_verified_same_uid_descendants(self) -> None:
        owners = [auditor.ReceiptIdentity(10, 100, "owner.json")]
        owned, excluded = auditor.snapshot_owned_tree(
            owners, [], self.procfs, auditor_pid=999
        )
        self.assertEqual({item.pid for item in owned}, {10, 11, 12})
        self.assertFalse({item.pid for item in owned} & {13, 14, 15, 16, 500, 999})
        self.assertIn(500, {row.pid for row in self.procfs.all()})
        self.assertTrue(any(item["reason"] == "auditor_excluded" for item in excluded))
        self.assertTrue(
            any(item["reason"] == "descendant_uid_mismatch" for item in excluded)
        )
        self.assertTrue(
            any(item["reason"] == "zombie_or_dead_process" for item in excluded)
        )
        self.assertTrue(all(len(item.comm_sha256) == 64 for item in owned))

    def test_dead_owner_does_not_hide_live_explicit_owned_process(self) -> None:
        dead_owner_procfs = FakeProcfs(
            [
                auditor.ProcessRow(
                    row.pid,
                    row.birth_ticks,
                    row.ppid,
                    row.uid,
                    row.comm,
                    row.exe_name,
                    "Z" if row.pid == 10 else row.state,
                )
                for row in self.rows
            ]
        )
        owned, excluded = auditor.snapshot_owned_tree(
            [auditor.ReceiptIdentity(10, 100, "controller-owner.json")],
            [auditor.ReceiptIdentity(11, 110, "owned/worker.json")],
            dead_owner_procfs,
            auditor_pid=999,
            expected_uid=1000,
        )
        self.assertIn(11, {item.pid for item in owned})
        self.assertIn(12, {item.pid for item in owned})
        self.assertTrue(
            any(
                item["pid"] == 10 and item["reason"] == "zombie_or_dead_process"
                for item in excluded
            )
        )

    def test_orphaned_owned_receipt_rejects_mixed_uid(self) -> None:
        owned, excluded = auditor.snapshot_owned_tree(
            [auditor.ReceiptIdentity(10, 100, "controller-owner.json")],
            [auditor.ReceiptIdentity(13, 130, "owned/foreign.json")],
            self.procfs,
            auditor_pid=999,
            expected_uid=1000,
        )
        self.assertNotIn(13, {item.pid for item in owned})
        self.assertTrue(
            any(
                item["pid"] == 13 and item["reason"] == "owned_uid_mismatch"
                for item in excluded
            )
        )

    def test_birth_mismatch_fails_closed(self) -> None:
        owned, excluded = auditor.snapshot_owned_tree(
            [auditor.ReceiptIdentity(10, 999, "stale-owner.json")],
            [],
            self.procfs,
            auditor_pid=999,
        )
        self.assertEqual(owned, [])
        self.assertEqual(excluded[0]["reason"], "owner_missing_or_birth_mismatch")

    def test_drain_signals_only_snapshot_with_fake_sender(self) -> None:
        owned, _ = auditor.snapshot_owned_tree(
            [auditor.ReceiptIdentity(10, 100, "owner.json")],
            [],
            self.procfs,
            auditor_pid=999,
        )
        clock = [20.0]
        sent: list[tuple[int, int, int]] = []

        def fake_now() -> float:
            return clock[0]

        def fake_sleep(seconds: float) -> None:
            clock[0] += seconds

        def fake_signal(
            ref: auditor.OwnedProcess, sig: int, _procfs: FakeProcfs
        ) -> str:
            sent.append((ref.pid, ref.birth_ticks, sig))
            return "test_recorded_no_signal_sent"

        events = auditor.drain_snapshot(
            owned, self.procfs, 120.0, fake_signal, fake_now, fake_sleep
        )
        self.assertEqual({pid for pid, _birth, _sig in sent}, {10, 11, 12})
        self.assertEqual(
            sum(sig == auditor.TERM_SIGNAL for _pid, _birth, sig in sent), 3
        )
        self.assertEqual(
            sum(sig == auditor.KILL_SIGNAL for _pid, _birth, sig in sent), 3
        )
        self.assertTrue(
            all(event["result"] == "test_recorded_no_signal_sent" for event in events)
        )

    def test_late_drain_still_kills_before_end_after_t10(self) -> None:
        owned = [auditor.OwnedProcess(10, 100, 500, 1000, "a" * 64, 0, "owner")]
        sent: list[int] = []

        def fake_signal(
            _ref: auditor.OwnedProcess, sig: int, _procfs: FakeProcfs
        ) -> str:
            sent.append(sig)
            return "test_recorded_no_signal_sent"

        events = auditor.drain_snapshot(
            owned,
            self.procfs,
            110.0,
            fake_signal,
            lambda: 105.0,
            lambda _seconds: None,
        )
        self.assertEqual(sent, [auditor.TERM_SIGNAL, auditor.KILL_SIGNAL])
        self.assertTrue(events[-1]["missed_t10_deadline"])

    def test_post_deadline_cleanup_signals_only_verified_owned_process(self) -> None:
        owned = [auditor.OwnedProcess(10, 100, 500, 1000, "a" * 64, 0, "owner")]
        sent: list[int] = []

        def fake_signal(
            _ref: auditor.OwnedProcess, sig: int, _procfs: FakeProcfs
        ) -> str:
            sent.append(sig)
            return "test_recorded_no_signal_sent"

        events = auditor.drain_snapshot(
            owned,
            self.procfs,
            100.0,
            fake_signal,
            lambda: 101.0,
            lambda _seconds: None,
        )
        self.assertEqual(sent, [auditor.TERM_SIGNAL, auditor.KILL_SIGNAL])
        self.assertTrue(all(event["release_after_deadline"] for event in events))
        self.assertTrue(events[-1]["missed_t10_deadline"])

    def test_gpu_report_exposes_only_pid_uuid_memory_for_foreign_processes(
        self,
    ) -> None:
        owned = [auditor.OwnedProcess(10, 100, 500, 1000, "a" * 64, 0, "owner.json")]
        records = [
            {"pid": 10, "gpu_uuid": "GPU-ours", "used_memory_mib": 12},
            {"pid": 70, "gpu_uuid": "GPU-foreign", "used_memory_mib": 321},
        ]
        report = auditor.gpu_ownership_report(records, owned, self.procfs)
        self.assertEqual(report["owned_gpu_processes"][0]["birth_ticks"], 100)
        self.assertEqual(report["foreign_gpu_processes_untouched"], [records[1]])
        self.assertEqual(
            set(report["foreign_gpu_processes_untouched"][0]),
            {"pid", "gpu_uuid", "used_memory_mib"},
        )


class OutputAndClockTests(unittest.TestCase):
    def test_wait_for_epoch_uses_short_bounded_sleeps(self) -> None:
        clock = [0.0]
        intervals: list[float] = []

        def fake_now() -> float:
            return clock[0]

        def fake_sleep(seconds: float) -> None:
            intervals.append(seconds)
            clock[0] += seconds

        auditor.wait_for_epoch(2.5, fake_now, fake_sleep)
        self.assertAlmostEqual(clock[0], 2.5)
        self.assertLessEqual(max(intervals), 1.0)

    def test_final_json_is_exclusive_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.json"
            auditor.write_immutable_json(
                str(path), {"schema": "test", "status": "clear"}
            )
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["status"], "clear"
            )
            with self.assertRaises(FileExistsError):
                auditor.write_immutable_json(str(path), {"status": "replacement"})
            os.chmod(path, 0o600)


if __name__ == "__main__":
    unittest.main()
