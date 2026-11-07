"""Explicit regularization recipes retain the legacy zero-dropout contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wnn.training.sequence_config import load_config


def test_legacy_dropout_guard_is_preserved(tmp_path: Path) -> None:
    """Historical recipes still reject nonzero dropout without an explicit policy."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": {"dropout": 0.1}}))
    with pytest.raises(ValueError, match="freezes dropout"):
        load_config(path)


@pytest.mark.parametrize("dropout", [0.0, 0.1, 0.3])
def test_explicit_dropout_is_loaded(tmp_path: Path, dropout: float) -> None:
    """The new opt-in is serialized in the recipe identity and retains its value."""
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"dropout_policy": "explicit_regularization_v1", "model": {"dropout": dropout}})
    )
    result = load_config(path)
    assert result["model"]["dropout"] == dropout
    assert result["dropout_policy"] == "explicit_regularization_v1"


@pytest.mark.parametrize("dropout", [-0.1, 1.0, True, float("nan")])
def test_invalid_explicit_dropout_is_rejected(tmp_path: Path, dropout: float) -> None:
    """Invalid probabilities cannot bypass the historical guard via the new policy."""
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"dropout_policy": "explicit_regularization_v1", "model": {"dropout": dropout}})
    )
    with pytest.raises(ValueError, match="explicit dropout"):
        load_config(path)
