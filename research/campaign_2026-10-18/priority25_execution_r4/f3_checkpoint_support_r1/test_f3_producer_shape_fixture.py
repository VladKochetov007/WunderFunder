from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

from f3_checkpoint_common import (
    DATASET_LENGTH,
    F3_PLAN_REMOTE_PATH,
    PLAN_SHA256,
    SCHEDULE_UPDATES,
    CheckpointMismatch,
    arm_spec,
    validate_attempt,
    validate_plan,
)
from f3_checkpoint_evidence import _verify_contract

HERE = Path(__file__).resolve().parent
PLAN_PATH = HERE.parent / "f3_beleriand_preparation_r1" / "F3_BELERIAND_PLAN.json"
FIXTURE_PATH = HERE / "fixtures/f3_wide_stage1_contract.json"


def _fixture() -> dict[str, object]:
    return cast(dict[str, object], json.loads(FIXTURE_PATH.read_text()))


def _source_map() -> dict[str, str]:
    return {
        str(F3_PLAN_REMOTE_PATH): PLAN_SHA256,
        "/home/grinchenkoeg/WunderFunder/scripts/bandit_resources.py": "b" * 64,
    }


def _source_manifest_sha(source_map: dict[str, str]) -> str:
    payload = json.dumps(source_map, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _contract(fixture: dict[str, object]) -> dict[str, object]:
    return {
        "epochs": 6,
        "total_steps": SCHEDULE_UPDATES,
        "options": dict(cast(dict[str, object], fixture["options"])),
        "loader": {"batch_size": 8, "drop_last": True, "dataset_length": DATASET_LENGTH},
        "runtime": {"matmul_tf32": False, "cudnn_tf32": False},
        "teacher_predictions": {},
        "consumed_caches": {},
        "ancestors": {"initial_checkpoint_sha256": None, "propensity_checkpoint_sha256": None},
    }


class ProducerShapeFixtureTests(unittest.TestCase):
    """Exercise persisted producer options separately from the captured CLI command."""

    def setUp(self) -> None:
        self.fixture = _fixture()
        self.plan = validate_plan(PLAN_PATH)
        self.spec = arm_spec(self.plan, "wide", "stage1")

    def test_source_derived_stage1_contract_omits_operational_save_flag(self) -> None:
        options = cast(dict[str, object], self.fixture["options"])
        counts = cast(dict[str, object], self.fixture["cumulative_updates"])
        self.assertNotIn("save_training_state", options)
        self.assertFalse(self.fixture["save_training_state_present"])
        self.assertEqual(self.fixture["next_epoch"], 2)
        self.assertEqual(counts["optimizer_completed_updates"], 29760)
        _verify_contract(_contract(self.fixture), self.plan, self.spec)

    def test_source_derived_command_binds_save_state_and_stage1_stop(self) -> None:
        command = cast(list[str], self.fixture["attempt_command"])
        self.assertEqual(command, self.spec["command"])
        self.assertIn("--save-training-state", command)
        stop_index = command.index("--stop-after-epoch")
        self.assertEqual(command[stop_index + 1], "1")
        source_map = _source_map()
        record = {
            "id": self.spec["attempt"],
            "command": command,
            "unit": f"wnn-bandit-{self.spec['attempt']}.service",
            "working_directory": "/home/grinchenkoeg/computer-vision-homework-20261018/source-a266b036",
            "contract": str(F3_PLAN_REMOTE_PATH),
            "contract_sha256": PLAN_SHA256,
            "source_sha256": source_map,
            "source_manifest_sha256": _source_manifest_sha(source_map),
            "status": "finished",
            "exit_status": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.json"
            path.write_text(json.dumps(record))
            self.assertEqual(validate_attempt(path, self.plan, self.spec)["command"], command)

    def test_attempt_command_cannot_drop_save_state_flag(self) -> None:
        command = cast(list[str], self.fixture["attempt_command"])
        source_map = _source_map()
        record = {
            "id": self.spec["attempt"],
            "command": [value for value in command if value != "--save-training-state"],
            "unit": f"wnn-bandit-{self.spec['attempt']}.service",
            "working_directory": "/home/grinchenkoeg/computer-vision-homework-20261018/source-a266b036",
            "contract": str(F3_PLAN_REMOTE_PATH),
            "contract_sha256": PLAN_SHA256,
            "source_sha256": source_map,
            "source_manifest_sha256": _source_manifest_sha(source_map),
            "status": "finished",
            "exit_status": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.json"
            path.write_text(json.dumps(record))
            with self.assertRaises(CheckpointMismatch):
                validate_attempt(path, self.plan, self.spec)

    def test_attempt_command_cannot_change_stage1_epoch_cap(self) -> None:
        command = cast(list[str], self.fixture["attempt_command"])
        stop_index = command.index("--stop-after-epoch")
        command[stop_index + 1] = "2"
        source_map = _source_map()
        record = {
            "id": self.spec["attempt"],
            "command": command,
            "unit": f"wnn-bandit-{self.spec['attempt']}.service",
            "working_directory": "/home/grinchenkoeg/computer-vision-homework-20261018/source-a266b036",
            "contract": str(F3_PLAN_REMOTE_PATH),
            "contract_sha256": PLAN_SHA256,
            "source_sha256": source_map,
            "source_manifest_sha256": _source_manifest_sha(source_map),
            "status": "finished",
            "exit_status": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.json"
            path.write_text(json.dumps(record))
            with self.assertRaises(CheckpointMismatch):
                validate_attempt(path, self.plan, self.spec)

    def test_attempt_contract_and_source_manifest_bindings_are_checked(self) -> None:
        command = cast(list[str], self.fixture["attempt_command"])
        source_map = _source_map()
        base = {
            "id": self.spec["attempt"],
            "command": command,
            "unit": f"wnn-bandit-{self.spec['attempt']}.service",
            "working_directory": "/home/grinchenkoeg/computer-vision-homework-20261018/source-a266b036",
            "contract": str(F3_PLAN_REMOTE_PATH),
            "contract_sha256": PLAN_SHA256,
            "source_sha256": source_map,
            "source_manifest_sha256": _source_manifest_sha(source_map),
            "status": "finished",
            "exit_status": 0,
        }
        for field, invalid, error in (
            ("contract", str(F3_PLAN_REMOTE_PATH) + ".other", "attempt contract path"),
            ("contract_sha256", "0" * 64, "attempt contract SHA256"),
            (
                "source_sha256",
                {**source_map, str(F3_PLAN_REMOTE_PATH): "0" * 64},
                "attempt source map F3 plan SHA256",
            ),
            ("source_manifest_sha256", "not-a-digest", "attempt source manifest SHA256"),
            ("source_manifest_sha256", "0" * 64, "attempt source manifest/map consistency"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                record = {**base, field: invalid}
                path = Path(directory) / "attempt.json"
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(CheckpointMismatch, error):
                    validate_attempt(path, self.plan, self.spec)

    def test_scientific_fold_guard_remains_strict(self) -> None:
        contract = _contract(self.fixture)
        cast(dict[str, object], contract["options"])["valid_fold"] = 0
        with self.assertRaises(CheckpointMismatch):
            _verify_contract(contract, self.plan, self.spec)

    def test_labels_only_guard_remains_strict(self) -> None:
        contract = _contract(self.fixture)
        cast(dict[str, object], contract["options"])["distill"] = "teacher"
        with self.assertRaises(CheckpointMismatch):
            _verify_contract(contract, self.plan, self.spec)


if __name__ == "__main__":
    unittest.main()
