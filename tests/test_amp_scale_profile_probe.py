from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/profile_amp_scale_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("profile_amp_scale_diagnostic", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load AMP profile diagnostic module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeScaler:
    def __init__(self) -> None:
        self.scale = 65536.0
        self.update_calls = 0
        self.enabled = True

    def is_enabled(self) -> bool:
        return self.enabled

    def get_scale(self) -> float:
        return self.scale

    def update(self, *args: object, **kwargs: object) -> str:
        self.update_calls += 1
        if self.update_calls == 2:
            self.scale *= 0.5
        return f"delegated-{self.update_calls}"


def test_probe_delegates_updates_and_records_only_scale_transitions() -> None:
    original_update = FakeScaler.update
    records: list[dict[str, object]] = []
    restore, update_count = MODULE.install_update_probe(FakeScaler, records.append)
    scaler = FakeScaler()

    assert scaler.update("first", value=1) == "delegated-1"
    assert scaler.update("second") == "delegated-2"
    assert scaler.update() == "delegated-3"

    assert update_count() == 3
    assert [record["update_ordinal"] for record in records] == [1, 2, 3]
    assert [record["scale_reduced"] for record in records] == [False, True, False]
    assert [(record["scale_before"], record["scale_after"]) for record in records] == [
        (65536.0, 65536.0),
        (65536.0, 32768.0),
        (32768.0, 32768.0),
    ]
    assert scaler.update_calls == 3

    assert restore() == 3
    assert FakeScaler.update is original_update


def test_probe_preserves_disabled_scaler_and_original_result() -> None:
    records: list[dict[str, object]] = []
    restore, _ = MODULE.install_update_probe(FakeScaler, records.append)
    scaler = FakeScaler()
    scaler.enabled = False

    assert scaler.update() == "delegated-1"
    assert records[0]["enabled"] is False
    assert records[0]["scale_reduced"] is False
    restore()


def test_trainer_argv_preserves_recipe_arguments() -> None:
    trainer = Path("scripts/train_rnn.py")
    args = ["--name", "profile-r2", "--profile-steps", "4", "--grad-stats"]

    assert MODULE.build_trainer_argv(trainer, args) == [str(trainer), *args]
    assert args == ["--name", "profile-r2", "--profile-steps", "4", "--grad-stats"]


def test_source_guard_matches_the_frozen_trainer_tree() -> None:
    root = MODULE_PATH.parents[1]

    actual = MODULE.verify_source(root)

    assert actual == MODULE.EXPECTED_SOURCE_HASHES


def test_source_guard_rejects_a_mutated_trainer(tmp_path: Path) -> None:
    for relative_path in MODULE.EXPECTED_SOURCE_HASHES:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"different source")

    with pytest.raises(RuntimeError, match="source pin mismatch"):
        MODULE.verify_source(tmp_path)


def test_recipe_binds_wrapper_source_and_exact_trainer_arguments() -> None:
    recipe_path = (
        MODULE_PATH.parents[1]
        / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/recipes/E529_S4_SERVED_HEAD_CLEANFOLD_CONTROL_AMP_DIAGNOSTIC_R2/diagnostic_recipe.json"
    )
    recipe = json.loads(recipe_path.read_text())

    assert recipe["diagnostic_source"]["wrapper_sha256"] == MODULE.sha256_file(MODULE_PATH)
    assert recipe["trainer_source_pins"] == MODULE.EXPECTED_SOURCE_HASHES
    assert recipe["execution"]["underlying_trainer_argv"][1:] == (MODULE.EXPECTED_TRAINER_ARGS)
    assert recipe["execution"]["entrypoint_argv"][2:] == MODULE.EXPECTED_TRAINER_ARGS
