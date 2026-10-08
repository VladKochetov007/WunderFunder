from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent
PACKET = BASE / "h217_packet_v5r2"
REVIEWED_V5_RECIPE_SHA256 = "c18b96fda798ead5cffbb089bd79589c5ce3aa5149cc81a49d2b4eb8985c9eac"
SPEC = importlib.util.spec_from_file_location("h217_v5r2_worker", PACKET / "worker.py")
assert SPEC is not None and SPEC.loader is not None
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


def _original_recipe() -> bytes:
    return (PACKET / "recipe.json").read_bytes()


def _runtime() -> dict[str, object]:
    return json.loads((PACKET / "runtime_resources.json").read_text())


def _with_extra_argv(extra: list[str]) -> bytes:
    recipe = json.loads(_original_recipe())
    recipe["profile_sequence"][0]["argv"].extend(extra)
    return (json.dumps(recipe, indent=2) + "\n").encode()


def test_original_recipe_bytes_pass_guard_and_semantic_validator() -> None:
    raw = _original_recipe()
    assert hashlib.sha256(raw).hexdigest() == REVIEWED_V5_RECIPE_SHA256
    assert WORKER.EXPECTED_RECIPE_SHA256 == REVIEWED_V5_RECIPE_SHA256
    recipe = WORKER.load_bound_recipe(PACKET / "recipe.json")
    WORKER.validate_recipe(recipe, _runtime())
    assert recipe["recipe_id"] == "E529-H217-TCN-FULLFEATURE-CAPACITY-V5-AMP4096"


@pytest.mark.parametrize(
    "extra",
    [
        ["--class-balance"],
        ["--valid-repeat", "2"],
        ["--epoch-samples", "1"],
        ["--lr", "0.1"],
        ["--seed", "8"],
    ],
)
def test_sampler_and_duplicate_flag_mutations_fail_before_parse(
    tmp_path: Path, extra: list[str]
) -> None:
    mutated = _with_extra_argv(extra)
    parsed = json.loads(mutated)
    WORKER.validate_recipe(parsed, _runtime())
    path = tmp_path / "recipe.json"
    path.write_bytes(mutated)
    with pytest.raises(ValueError, match="refusing to parse or execute"):
        WORKER.load_bound_recipe(path)


def test_malformed_recipe_is_rejected_before_parse(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    path.write_bytes(b"{")
    with pytest.raises(ValueError, match="refusing to parse or execute"):
        WORKER.load_bound_recipe(path)


def test_missing_recipe_has_no_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WNN_RECIPE_HASH_SKIP", "1")
    monkeypatch.setenv("WNN_EXPECTED_RECIPE_SHA256", REVIEWED_V5_RECIPE_SHA256)
    with pytest.raises(FileNotFoundError, match="sealed recipe is missing"):
        WORKER.load_bound_recipe(tmp_path / "absent.json")


def test_pin_is_a_source_literal() -> None:
    text = (PACKET / "worker.py").read_text()
    assert f'EXPECTED_RECIPE_SHA256 = "{REVIEWED_V5_RECIPE_SHA256}"' in text
    assert "EXPECTED_RECIPE_SHA256 = hashlib" not in text
