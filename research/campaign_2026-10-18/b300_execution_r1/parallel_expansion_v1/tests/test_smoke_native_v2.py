"""Verify the versioned smoke uses its separate, pinned subset helper."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

SMOKE_DIR = Path(__file__).resolve().parents[1] / "smoke_v2"
sys.path.insert(0, str(SMOKE_DIR))

from smoke_native_validation import options, set_option  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "smoke_native_subset_v2", SMOKE_DIR / "smoke_native_subset.py"
)
assert spec is not None and spec.loader is not None
subset_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = subset_module
spec.loader.exec_module(subset_module)


def test_subset_builder_is_separately_pinned_from_trainer_source() -> None:
    builder_path = subset_module.SUBSET_BUILDER_PATH
    assert builder_path.is_file()
    assert not builder_path.is_symlink()
    digest = hashlib.sha256(builder_path.read_bytes()).hexdigest()
    assert digest == subset_module.SUBSET_BUILDER_SHA256
    assert "source-a266-lf-subset-v1" not in str(builder_path)


def test_smoke_v2_keeps_recipe_override_parser_strict() -> None:
    argv = ["--hidden", "512", "--precision", "fp32"]
    set_option(argv, "--hidden", "16")
    assert options(argv) == {"hidden": "16", "precision": "fp32"}
    with pytest.raises(ValueError, match="exactly one"):
        set_option(["--epochs", "1", "--epochs", "6"], "--epochs", "1")
