from __future__ import annotations

import ast
import json
import mmap
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from bounded_loader import BoundedSequenceFile, release_auxiliary_mappings
from build_packets import ENTRY, gpu_payload
from derive_source import derive_trainer, require_production_sequence_length
from host_guard import (
    cgroup_ancestors,
    host_sample,
    oom_counts,
    sample_v1_hierarchy,
    supervised,
    tree_sample,
)
from phase import FIT_REQUIRED_OUTPUTS, collect_fit_artifacts, output_run_name, worker_plan

import wnn.data as data
from wnn.data import FEATURE_COLUMNS, SEQUENCE_LENGTH, SequenceFile

ROOT = Path(__file__).resolve().parents[4]


class LoaderParity(unittest.TestCase):
    """Synthetic complete-sequence parity, with no competition rows or GPU."""

    @staticmethod
    def write_v1_level(path: Path, limit: int | str, current: int) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "memory.limit_in_bytes").write_text(f"{limit}\n")
        (path / "memory.usage_in_bytes").write_text(f"{current}\n")

    def test_complete_sequence_and_optional_columns(self) -> None:
        """Preserve masks, signed zeros, float conversion and dense row order."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.parquet"
            row = np.arange(SEQUENCE_LENGTH, dtype=np.float32)
            values = {"seq_ix": np.full(SEQUENCE_LENGTH, 71, dtype=np.int64)}
            for column, name in enumerate(FEATURE_COLUMNS):
                values[name] = np.sin(row / (column + 1)).astype(np.float64)
                values[name][0] = -0.0
            values.update(t0=row / 31, t1=-row / 41, is_scored=row % 7 == 0)
            pq.write_table(pa.table(values), path, row_group_size=SEQUENCE_LENGTH)
            old, new = SequenceFile(path), BoundedSequenceFile(path)
            for want_targets, want_mask in [(True, True), (False, False), (True, False)]:
                expected = old.read(0, targets=want_targets, mask=want_mask)
                actual = new.read(0, targets=want_targets, mask=want_mask)
                self.assertEqual(expected.seq_ix, actual.seq_ix)
                for field in ("features", "targets", "scored"):
                    a, b = getattr(expected, field), getattr(actual, field)
                    if a is None:
                        self.assertIsNone(b)
                    else:
                        self.assertEqual(a.dtype, b.dtype)
                        self.assertEqual(a.shape, b.shape)
                        self.assertEqual(a.tobytes(), b.tobytes())
                self.assertTrue(actual.features.flags.c_contiguous)

    def test_small_fixture_follows_patched_data_contract(self) -> None:
        """A fixture row count follows the same active contract as SequenceFile."""
        rows = 8
        with tempfile.TemporaryDirectory() as directory, patch.object(
            data, "SEQUENCE_LENGTH", rows
        ):
            path = Path(directory) / "fixture.parquet"
            values = {
                name: np.arange(rows, dtype=np.float32) + column
                for column, name in enumerate(FEATURE_COLUMNS)
            }
            values.update(
                seq_ix=np.full(rows, 71, dtype=np.int64),
                t0=np.linspace(0.0, 1.0, rows, dtype=np.float32),
                t1=np.linspace(1.0, 0.0, rows, dtype=np.float32),
                is_scored=np.arange(rows) % 2 == 1,
            )
            pq.write_table(pa.table(values), path, row_group_size=rows)
            old, new = SequenceFile(path), BoundedSequenceFile(path)
            expected = old.read(0)
            actual = new.read(0)
            self.assertEqual(new.expected_rows, rows)
            for field in ("features", "targets", "scored"):
                self.assertEqual(
                    getattr(expected, field).tobytes(), getattr(actual, field).tobytes()
                )

    def test_row_group_rejects_mixed_sequence_ids(self) -> None:
        """Matching row counts do not permit a row group to cross sequence ids."""
        rows = 8
        with tempfile.TemporaryDirectory() as directory, patch.object(
            data, "SEQUENCE_LENGTH", rows
        ):
            path = Path(directory) / "fixture.parquet"
            values = {
                name: np.zeros(rows, dtype=np.float32) for name in FEATURE_COLUMNS
            }
            values.update(
                seq_ix=np.array([71] * (rows - 1) + [72], dtype=np.int64),
                t0=np.zeros(rows, dtype=np.float32),
                t1=np.zeros(rows, dtype=np.float32),
                is_scored=np.ones(rows, dtype=bool),
            )
            pq.write_table(pa.table(values), path, row_group_size=rows)
            with self.assertRaisesRegex(ValueError, "multiple sequence ids"):
                BoundedSequenceFile(path).read(0)

    def test_production_contract_rejects_short_row_group(self) -> None:
        """Without the fixture patch, an eight-row group cannot pass as production."""
        self.assertEqual(data.SEQUENCE_LENGTH, 20_000)
        rows = 8
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short.parquet"
            values = {
                name: np.zeros(rows, dtype=np.float32) for name in FEATURE_COLUMNS
            }
            values.update(
                seq_ix=np.full(rows, 71, dtype=np.int64),
                t0=np.zeros(rows, dtype=np.float32),
                t1=np.zeros(rows, dtype=np.float32),
                is_scored=np.ones(rows, dtype=bool),
            )
            pq.write_table(pa.table(values), path, row_group_size=rows)
            with self.assertRaisesRegex(ValueError, "has 8 rows"):
                BoundedSequenceFile(path).read(0)

    def test_mappings_remain_valid_after_release(self) -> None:
        """Reclaim resident cache pages only after returned arrays own copies."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.npy"
            original = np.arange(128 * 2, dtype=np.float32).reshape(128, 2)
            np.save(path, original)
            cached = np.load(path, mmap_mode="r")
            copied = np.array(cached[31], dtype=np.float32)
            owner = SimpleNamespace(_teach=[(cached, 1.0)], _prop=None, _quote=None)
            release_auxiliary_mappings(owner)
            np.testing.assert_array_equal(copied, original[31])
            np.testing.assert_array_equal(cached[127], original[127])
            self.assertIsInstance(cached._mmap, mmap.mmap)
            self.assertFalse(cached._mmap.closed)

    def test_source_patch_is_loader_only(self) -> None:
        """The derivation leaves scientific statements exactly unchanged."""
        original = (
            ROOT / "research/campaign_2026-10-05/w384_v7/bundle_source/scripts/train_rnn.py"
        ).read_text()
        derived = derive_trainer(original)
        original_ast, derived_ast = ast.parse(original), ast.parse(derived)
        # Remove exactly the reviewed import and return-boundary release call.
        for module in (original_ast, derived_ast):
            for node in ast.walk(module):
                if isinstance(node, ast.ImportFrom) and node.module == "wnn.data":
                    node.names = [name for name in node.names if name.name != "SequenceFile"]
                if isinstance(node, ast.FunctionDef) and node.name == "__getitem__":
                    node.body = [
                        statement
                        for statement in node.body
                        if not (
                            isinstance(statement, ast.Expr)
                            and isinstance(statement.value, ast.Call)
                            and isinstance(statement.value.func, ast.Name)
                            and statement.value.func.id == "release_auxiliary_mappings"
                        )
                    ]
            module.body = [
                node
                for node in module.body
                if not (isinstance(node, ast.ImportFrom) and node.module == "wnn.bounded_loader")
            ]
        self.assertEqual(ast.dump(original_ast), ast.dump(derived_ast))
        with self.assertRaises(ValueError):
            derive_trainer(derived)

    def test_production_row_count_is_pinned(self) -> None:
        """Private staging rejects altered or ambiguous production row contracts."""
        source = (ROOT / "src/wnn/data.py").read_text()
        require_production_sequence_length(source)
        with self.assertRaisesRegex(ValueError, "20_000"):
            require_production_sequence_length(source.replace("SEQUENCE_LENGTH = 20_000", "SEQUENCE_LENGTH = 8"))
        with self.assertRaisesRegex(ValueError, "20_000"):
            require_production_sequence_length(
                source + "\nSEQUENCE_LENGTH = 8\n"
            )

    def test_payloads_keep_existing_science_and_single_launcher(self) -> None:
        """Both width kernels are syntactically valid and source-pinned."""
        for width in (256, 384):
            source = gpu_payload(width).decode()
            ast.parse(source)
            self.assertIn('"--width", "' + str(width) + '"', source)
            self.assertIn("Original manifest identity mismatch", source)
            self.assertIn("timeout_seconds=17_900", source)
        ast.parse(ENTRY)

    def test_recovery_plan_changes_output_identity_only(self) -> None:
        """The recovery writes fresh artifacts while retaining scientific argv."""
        old = "c20261004_w384_adamw_s3_f1_t4v4_probe"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "kaggle_specs").mkdir()
            (root / "plans").mkdir()
            (root / "continuation_width").mkdir()
            (root / "kaggle_specs" / f"{old}.json").write_text(
                json.dumps(
                    {
                        "argv": ["python", "scripts/train_rnn.py", "--plan", "plans/old.json"]
                    }
                )
            )
            source_plan = {
                "steps": [
                    {
                        "argv": [
                            "python",
                            "scripts/train_rnn.py",
                            "--seed",
                            "3",
                            "--width",
                            "384",
                            "--name",
                            old,
                        ]
                    }
                ],
                "report": "reports/old.json",
            }
            (root / "plans" / "old.json").write_text(json.dumps(source_plan))
            command, name = worker_plan(root, 384, True)
            self.assertEqual(name, "c20261006night_w384_adamw_s3_f1_serialv3r1_probe")
            self.assertIn(str(root / "continuation_width" / f"{name}_plan.json"), command)
            plan = json.loads((root / "continuation_width" / f"{name}_plan.json").read_text())
            expected = source_plan["steps"][0]["argv"].copy()
            expected[expected.index(old)] = name
            self.assertEqual(plan["steps"][0]["argv"], expected)
            self.assertEqual(plan["report"], f"continuation_width/{name}_steps.json")

    def test_fit_manifest_requires_each_fresh_checkpoint_and_record(self) -> None:
        """The v3r1 manifest binds checkpoint, provenance, state, and history."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fit_name = output_run_name(384, "fit")
            run_directory = root / "artifacts/runs" / fit_name
            run_directory.mkdir(parents=True)
            for name in FIT_REQUIRED_OUTPUTS:
                (run_directory / name).write_bytes(name.encode())

            records = collect_fit_artifacts(root, fit_name)
            record_names = {Path(row["path"]).name for row in records}
            self.assertEqual(record_names, set(FIT_REQUIRED_OUTPUTS))
            for missing in FIT_REQUIRED_OUTPUTS:
                path = run_directory / missing
                path.unlink()
                with self.assertRaisesRegex(FileNotFoundError, missing):
                    collect_fit_artifacts(root, fit_name)
                path.write_bytes(missing.encode())

    def test_live_read_only_inventory(self) -> None:
        """Guard observes this tiny fixture and the actual cgroup ceiling."""
        import os

        host = host_sample()
        self.assertGreater(host["capacity_bytes"], 0)
        self.assertLessEqual(host["available_bytes"], host["capacity_bytes"])
        self.assertGreater(tree_sample(os.getpid())[0]["rss_bytes"], 0)
        self.assertEqual(
            cgroup_ancestors(Path("/sys/fs/cgroup"), "/parent/child"),
            [
                Path("/sys/fs/cgroup/parent/child"),
                Path("/sys/fs/cgroup/parent"),
                Path("/sys/fs/cgroup"),
            ],
        )

    def test_v1_intermediate_limit_constrains_capacity_and_headroom(self) -> None:
        """A tighter parent cgroup constrains a leaf with a looser limit."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "memory"
            parent = root / "parent"
            leaf = parent / "child"
            self.write_v1_level(root, 16 * 1024**3, 1 * 1024**3)
            self.write_v1_level(parent, 2 * 1024**3, 1 * 1024**3)
            self.write_v1_level(leaf, 8 * 1024**3, 1 * 1024**3)
            (leaf / "memory.oom_control").write_text("oom_kill 7\nunder_oom 0\n")
            capacity, available, records = sample_v1_hierarchy(
                [leaf, parent, root], 32 * 1024**3, 30 * 1024**3
            )
        self.assertEqual(capacity, 2 * 1024**3)
        self.assertEqual(available, 1 * 1024**3)
        self.assertEqual(len(records), 3)
        self.assertEqual(
            oom_counts({"cgroups": records}),
            {f"{leaf}/memory.oom_control/oom_kill": 7},
        )

    def test_v1_missing_intermediate_fails_closed(self) -> None:
        """Readable root and leaf do not hide an unreadable parent level."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "memory"
            parent = root / "parent"
            leaf = parent / "child"
            self.write_v1_level(root, 16 * 1024**3, 1 * 1024**3)
            self.write_v1_level(leaf, 8 * 1024**3, 1 * 1024**3)
            with self.assertRaisesRegex(RuntimeError, "not fully observable"):
                sample_v1_hierarchy([leaf, parent, root], 32 * 1024**3, 30 * 1024**3)

    def test_v1_malformed_intermediate_fails_closed(self) -> None:
        """A malformed parent limit cannot silently fall back to the root."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "memory"
            parent = root / "parent"
            leaf = parent / "child"
            self.write_v1_level(root, 16 * 1024**3, 1 * 1024**3)
            self.write_v1_level(parent, "unknown", 1 * 1024**3)
            self.write_v1_level(leaf, 8 * 1024**3, 1 * 1024**3)
            with self.assertRaisesRegex(RuntimeError, "not fully observable"):
                sample_v1_hierarchy([leaf, parent, root], 32 * 1024**3, 30 * 1024**3)

    def test_failed_child_preserves_receipt(self) -> None:
        """A child import failure remains a failed attempt with durable evidence."""
        import json
        import os
        import sys

        available = {"capacity_bytes": 32 * 1024**3, "available_bytes": 32 * 1024**3, "cgroups": {}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "attempt"
            with (
                patch("host_guard.host_sample", return_value=available),
                self.assertRaises(RuntimeError),
            ):
                supervised(
                    [sys.executable, "-c", "raise ImportError('fixture')"],
                    Path(directory),
                    output,
                    os.environ.copy(),
                    timeout=20,
                    rss_limit=128 * 1024**2,
                )
            receipt = json.loads((output / "result.json").read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertNotEqual(receipt["returncode"], 0)
            self.assertIn("fixture", (output / "worker.log").read_text())

    def test_timeout_terminates_nested_session(self) -> None:
        """The outer transport also cleans up a worker in its own session."""
        import json
        import os
        import sys

        available = {"capacity_bytes": 32 * 1024**3, "available_bytes": 32 * 1024**3, "cgroups": {}}
        source = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); print(p.pid,flush=True); time.sleep(30)"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "attempt"
            with (
                patch("host_guard.host_sample", return_value=available),
                self.assertRaisesRegex(RuntimeError, "timeout"),
            ):
                supervised(
                    [sys.executable, "-c", source],
                    Path(directory),
                    output,
                    os.environ.copy(),
                    timeout=1,
                    rss_limit=128 * 1024**2,
                )
            receipt = json.loads((output / "result.json").read_text())
            self.assertEqual(receipt["reason"], "timeout")
            pid = int((output / "worker.log").read_text().strip())
            time.sleep(0.1)
            rows = tree_sample(pid)
            self.assertTrue(not rows or rows[0]["rss_bytes"] == 0)

    def test_fast_parent_exit_adopts_and_reaps_new_session(self) -> None:
        """A child spawned after sampling cannot escape by orphaning itself."""
        import json
        import os
        import sys

        available = {"capacity_bytes": 32 * 1024**3, "available_bytes": 32 * 1024**3, "cgroups": {}}
        source = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); print(p.pid,flush=True)"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "attempt"
            with (
                patch("host_guard.host_sample", return_value=available),
                self.assertRaisesRegex(RuntimeError, "surviving descendants"),
            ):
                supervised(
                    [sys.executable, "-c", source],
                    Path(directory),
                    output,
                    os.environ.copy(),
                    timeout=20,
                    rss_limit=128 * 1024**2,
                )
            receipt = json.loads((output / "result.json").read_text())
            self.assertEqual(receipt["reason"], "descendants_after_root_exit")
            self.assertEqual(receipt["teardown"]["status"], "complete")
            pid = int((output / "worker.log").read_text().strip())
            self.assertEqual(tree_sample(pid), [])


if __name__ == "__main__":
    unittest.main()
