from __future__ import annotations

import importlib.util
import json
import struct
import tempfile
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[6]
PACKET = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_single_member_cleanfold_q8_r6"
)
R6_PATH = ROOT / PACKET / "run_single_controls_r6.py"
R5_PATH = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_single_member_cleanfold_q8_r5/callback_runtime_r5.py"
)
SPEC = importlib.util.spec_from_file_location("r6_test_subject", R6_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import R6 test subject")
R6 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R6)
R5_SPEC = importlib.util.spec_from_file_location("r5_helpers_test", R5_PATH)
if R5_SPEC is None or R5_SPEC.loader is None:
    raise RuntimeError("cannot import pinned R5 helpers")
R5 = importlib.util.module_from_spec(R5_SPEC)
R5_SPEC.loader.exec_module(R5)


class FirstReadReached(Exception):
    pass


class FakeArray:
    def flush(self) -> None:
        return None

    def __setitem__(self, key: Any, value: Any) -> None:
        return None


class FakeParquet:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.metadata = SimpleNamespace(
            num_rows=20_000, row_group=lambda index: SimpleNamespace(num_rows=20_000)
        )
        self.num_row_groups = 1
        self.schema = SimpleNamespace(
            names=["seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1"]
        )

    def read_row_group(self, group: int, *, columns: list[str], use_threads: bool) -> Any:
        self.events.append("first_row_group_read")
        raise FirstReadReached


class GateTests(unittest.TestCase):
    def test_wrapper_requires_exact_scope_and_one_gigabyte_no_swap(self) -> None:
        budget = {
            "cpu_cores": 1,
            "ram_max_bytes": 1_000_000_000,
            "wall_seconds": 1800,
            "gpu": False,
            "new_spend_usd": 0,
            "thread_count": 1,
            "provider_actions": False,
        }
        env = {
            "WNN_SLICE": "wnn-campaign.slice",
            "WNN_CPU_QUOTA": "100%",
            "WNN_MEM_MAX": "1000000000",
            "WNN_RUNTIME_MAX": "1800s",
            "WNN_SCOPE_UNIT": "wnn-native-q8-single-controls-r6.scope",
            "CUDA_VISIBLE_DEVICES": "",
            **{key: "1" for key in R6.THREAD_ENV},
        }
        cg = {
            "path": "/user.slice/wnn-campaign.slice/wnn-native-q8-single-controls-r6.scope",
            "cpu_quota": 100_000,
            "cpu_period": 100_000,
            "memory_max_bytes": 1_000_000_000,
            "memory_current_bytes": 200_000_000,
            "memory_swap_max_bytes": 0,
        }
        R6.verify_wrapper_limits(budget, env, cg)
        with self.assertRaisesRegex(R6.GateError, "scope"):
            R6.verify_wrapper_limits(
                budget, env, {**cg, "path": "/user.slice/wnn-campaign.slice/other.scope"}
            )
        with self.assertRaisesRegex(R6.GateError, "memory"):
            R6.verify_wrapper_limits(budget, env, {**cg, "memory_max_bytes": 1_000_000_001})
        with self.assertRaisesRegex(R6.GateError, "swap"):
            R6.verify_wrapper_limits(budget, env, {**cg, "memory_swap_max_bytes": 1})

    def test_root_release_must_bind_launch_and_output(self) -> None:
        models = {
            fold: {
                "evaluation_group_list_sha256": f"m{fold}",
                "models": {
                    role: {
                        "checkpoint_sha256": f"{role}{fold}",
                        "provenance_sha256": f"p{role}{fold}",
                    }
                    for role in ("student", "hft")
                },
            }
            for fold in ("0", "1")
        }
        contract = {
            "root_admission": {"schema": "test.r6", "decision": "admit"},
            "resource_budget": {"cpu_cores": 1},
            "launch_command": {"argv": ["scripts/limited", "python", "runner.py"]},
            "r4_binding": {"contract_sha256": "a", "consumer_sha256": "b"},
            "r5_binding": {"callback_fixture_sha256": "c", "runtime_sha256": "d"},
            "r3_binding": {
                "contract_sha256": "e",
                "input_preflight_sha256": "f",
                "valid_dataset_identity": {
                    "bytes": 1,
                    "footer_bytes": 2,
                    "footer_sha256_including_8_byte_trailer": "x",
                },
            },
            "models": models,
            "weights": {"archive_sha256": "g", "receipt_sha256": "h"},
            "runtime_assets": {"native_elf": {"sha256": "i"}},
            "output": {"root_repo_path": "out"},
        }
        contract["source_bindings"] = {"r6_consumer": {"sha256": "j"}}
        expected = R6.admission_expected_bindings(contract, "k", "j", "f", "c")
        release = {
            "schema": "test.r6",
            "decision": "admit",
            "status": "admitted",
            "authorized": True,
            "bindings": expected,
            "budget": contract["resource_budget"],
        }
        R6.verify_admission_object(release, contract, "k", "j", "f", "c")
        bad = dict(release, bindings={**expected, "launch_command": {"argv": ["different"]}})
        with self.assertRaisesRegex(R6.GateError, "binding"):
            R6.verify_admission_object(bad, contract, "k", "j", "f", "c")

    def test_footer_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic.parquet"
            footer = b"one"

            def write(value: bytes) -> None:
                path.write_bytes(b"PAR1body" + value + struct.pack("<I", len(value)) + b"PAR1")

            write(footer)
            with path.open("rb") as stream:
                original = R6.file_footer_identity(stream, path, len(footer))
            r3 = {"valid_dataset": {key: original[key] for key in R6.IDENTITY_KEYS}}
            preflight = {
                "valid_dataset_identity": {key: original[key] for key in ("inode", "mtime_ns")}
            }
            R6.validate_live_identity(original, r3, preflight)
            write(b"two")
            with path.open("rb") as stream:
                changed = R6.file_footer_identity(stream, path, len(footer))
            with self.assertRaises(R6.GateError):
                R6.validate_live_identity(changed, r3, preflight)

    def test_r5_schema_name_mutation_fails_before_schema_repr(self) -> None:
        class Schema:
            names = ["seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1"]

            def __str__(self) -> str:
                raise AssertionError("schema body must not be consulted after name mismatch")

        parquet = SimpleNamespace(
            metadata=SimpleNamespace(
                num_rows=20_000, row_group=lambda index: SimpleNamespace(num_rows=20_000)
            ),
            num_row_groups=1,
            schema=Schema(),
        )
        r3 = {"valid_dataset": {"rows": 20_000, "row_groups": 1}}
        preflight = {
            "metadata_only_validation": {"schema_names_count": 6, "schema_names_sha256": "0" * 64}
        }
        with self.assertRaisesRegex(ValueError, "column names"):
            R5.check_stable_metadata(parquet, r3, preflight, [])

    def test_actual_full_consumer_stops_at_first_read_without_callbacks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            data_path = temp_path / "synthetic.parquet"
            data_path.write_bytes(b"fixture only")
            output_root = temp_path / "output"
            r4_contract_path = (
                ROOT
                / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_single_member_cleanfold_q8_r4/CONTRACT_SINGLE_R4.json"
            )
            r4_contract = json.loads(r4_contract_path.read_text(encoding="utf-8"))
            r3_path = ROOT / r4_contract["r3_binding"]["contract_repo_path"]
            r3 = json.loads(r3_path.read_text(encoding="utf-8"))
            identity_stat = data_path.stat()
            identity = {
                "bytes": identity_stat.st_size,
                "footer_bytes": 0,
                "footer_sha256_including_8_byte_trailer": "0" * 64,
                "device": identity_stat.st_dev,
                "inode": identity_stat.st_ino,
                "mtime_ns": identity_stat.st_mtime_ns,
            }
            events: list[str] = []
            parquet = FakeParquet(events)
            pq = SimpleNamespace(
                ParquetFile=lambda stream: (events.append("parquet_metadata"), parquet)[1]
            )

            class FakeMaps:
                lib = SimpleNamespace(
                    format=SimpleNamespace(open_memmap=lambda *args, **kwargs: FakeArray())
                )
                float32 = "float32"

            class FakeRunner:
                callbacks = 0

                def run_callbacks(self, *args: Any, **kwargs: Any) -> Any:
                    self.callbacks += 1
                    raise AssertionError(
                        "callback must not run before first row-group read returns"
                    )

            runner = FakeRunner()

            def guard() -> None:
                events.append("resource_guard")

            def metadata_check(*args: Any) -> None:
                events.append("stable_schema_gate")

            contract = {
                "resource_budget": {"wall_seconds": 1800},
                "repository": r4_contract["repository"],
                "r3_producer_receipts": r4_contract["r3_producer_receipts"],
                "models": r4_contract["models"],
            }
            preflight: dict[str, Any] = {}
            fake_pyarrow = ModuleType("pyarrow")
            fake_pyarrow.parquet = pq  # type: ignore[attr-defined]
            with (
                patch.dict("sys.modules", {"pyarrow": fake_pyarrow, "pyarrow.parquet": pq}),
                patch.object(
                    R6,
                    "file_footer_identity",
                    side_effect=lambda stream, path, expected: identity,
                ),
                patch.object(R6, "validate_live_identity", return_value=None),
                self.assertRaises(FirstReadReached),
            ):
                R6.run_full_controls(
                    contract,
                    r3,
                    preflight,
                    data_path,
                    output_root,
                    {},
                    runner,
                    SimpleNamespace(GlobalAccumulator=lambda: object()),
                    SimpleNamespace(FEATURE_COLUMNS=[]),
                    FakeMaps(),
                    R5,
                    guard,
                    time.monotonic(),
                    metadata_check,
                )
            self.assertIn("first_row_group_read", events)
            self.assertEqual(runner.callbacks, 0)
            self.assertGreaterEqual(events.count("resource_guard"), 3)
            self.assertLess(
                events.index("stable_schema_gate"), events.index("first_row_group_read")
            )


if __name__ == "__main__":
    unittest.main()
