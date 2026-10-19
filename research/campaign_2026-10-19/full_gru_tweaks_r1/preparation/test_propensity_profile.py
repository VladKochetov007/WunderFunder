"""Small CPU-only invariants for the actual-data propensity profile plan."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from propensity_profile import ProfilePlan, _load_rows, _write_create_only, build_profile_plan
from propensity_profile_core import ProfileRowsData


class _Journal:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def append(self, attempt_id: str, event: dict[str, object], *, durable: bool = False) -> int:
        self.events.append({"attempt_id": attempt_id, **event, "durable": durable})
        return len(self.events)


class _ShapeOnlyArray:
    def __init__(self, shape: tuple[int, ...], dtype: object) -> None:
        self.shape = shape
        self.dtype = dtype


class PropensityProfilePlanTests(unittest.TestCase):
    def test_cold_entry_and_core_import_do_not_load_torch(self) -> None:
        preparation = Path(__file__).resolve().parent
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(preparation)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import propensity_profile; import propensity_profile_core; "
                "import sys; assert 'torch' not in sys.modules",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_load_rows_constructs_the_lazy_data_container(self) -> None:
        class FakeSplit:
            valid_groups = tuple(range(16))
            evaluation_groups = (100,)
            sha256 = "split-sha256"

            def assert_fit_groups(self, train_groups, valid_groups) -> None:
                self.assertion = set(train_groups).isdisjoint(valid_groups)
                if not self.assertion:
                    raise AssertionError("profile membership overlaps train groups")

        split = FakeSplit()

        class FakeReader:
            def __init__(self, path: Path) -> None:
                self.path = path

            def __len__(self) -> int:
                return 16

            def read(self, group: int, *, targets: bool, mask: bool):
                if targets or not mask:
                    raise AssertionError("profile must read features and mask only")
                return types.SimpleNamespace(
                    targets=None,
                    scored=_ShapeOnlyArray((20_000,), np.bool_),
                    features=_ShapeOnlyArray((20_000, 112), np.float32),
                    seq_ix=1_000 + group,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            valid_path = data_root / "datasets/valid.parquet"
            valid_path.parent.mkdir(parents=True)
            valid_path.write_bytes(b"synthetic")
            split_path = root / "fold0.json"
            identity = {"bytes": len(b"synthetic"), "sha256": "fixture"}
            split_path.write_text(json.dumps({"sources": {"datasets/valid.parquet": identity}}))
            package = types.ModuleType("wnn")
            package.__path__ = []
            data_module = types.ModuleType("wnn.data")
            data_module.N_FEATURES = 112
            data_module.SequenceFile = FakeReader
            split_module = types.ModuleType("wnn.splits")
            split_module.load_campaign_split = lambda _path, _root: split
            journal = _Journal()
            with patch.dict(
                sys.modules,
                {
                    "wnn": package,
                    "wnn.data": data_module,
                    "wnn.splits": split_module,
                },
            ):
                rows = _load_rows(data_root, split_path, journal, "fixture-attempt")

        self.assertIsInstance(rows, ProfileRowsData)
        self.assertEqual(rows.plan.batch_groups, tuple(range(16)))
        self.assertEqual(rows.sequence_ids, tuple(range(1_000, 1_016)))
        self.assertEqual(len(rows.feature_batches), 16)
        self.assertEqual(len(rows.masks), 16)
        self.assertTrue(split.assertion)
        self.assertTrue(all(event["durable"] for event in journal.events))

    def test_plan_uses_first_sixteen_actual_groups_twice(self) -> None:
        valid = tuple(range(30, 58))
        plan = build_profile_plan(valid, (100, 101))

        self.assertEqual(
            plan,
            ProfilePlan(
                first_four_groups=(30, 31, 32, 33),
                batch_groups=tuple(range(30, 46)),
                repeat_batch_groups=(tuple(range(30, 46)),) * 2,
                chunk_starts=tuple(range(0, 20_000, 2_000)),
            ),
        )

    def test_plan_rejects_small_or_duplicate_membership(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires 16"):
            build_profile_plan(tuple(range(15)), ())
        with self.assertRaisesRegex(ValueError, "not unique"):
            build_profile_plan(tuple(range(15)) + (14,), ())

    def test_plan_rejects_direct_evaluation_group_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_profile_plan(tuple(range(16)), (15, 100))

    def test_result_writer_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "PROFILE_RESULT.json"
            _write_create_only(path, {"status": "ok"})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                _write_create_only(path, {"status": "replacement"})
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
