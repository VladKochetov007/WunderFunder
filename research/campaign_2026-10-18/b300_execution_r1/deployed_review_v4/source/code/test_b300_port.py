"""B300 port integrity and ownership tests, without GPU execution."""

from __future__ import annotations
import json, shutil
from pathlib import Path
import pytest
from adapter_common import verify_source_manifest
from source_package_verify import verify_source_package
from fit_recipe import build_training_argv

CODE = Path(__file__).parent
SOURCE = CODE.parents[1] / "source-a266-lf-subset-v1"


def test_source_package_rejects_mutation_and_extra_dependency(tmp_path: Path) -> None:
    copy = tmp_path / "source"
    shutil.copytree(SOURCE, copy)
    manifest = json.loads((CODE / "source_package_manifest.json").read_text())
    check = lambda: verify_source_package(copy, manifest, manifest_sha256="test")
    assert check()["file_count"] == 147
    extra = copy / "src/extra.py"
    extra.write_text("raise RuntimeError('extra')")
    with pytest.raises(ValueError, match="membership differ"):
        check()
    extra.unlink()
    target = copy / "src/wnn/training/runner.py"
    original = target.read_bytes()
    target.write_bytes(original + b"\n# mutation\n")
    with pytest.raises(ValueError, match="membership differ"):
        check()
    target.write_bytes(original)
    assert check()["file_count"] == 147


def test_canonical_trainer_includes_readme() -> None:
    manifest = json.loads((CODE / "a266_source_manifest.json").read_text())
    actual = verify_source_manifest(SOURCE, manifest)
    assert actual["file_count"] == 61
    assert "src/wnn/training/README.md" in actual["files"]


def test_external_seeds_rejected_before_launch(tmp_path: Path) -> None:
    split = tmp_path / "fold0.json"
    split.write_text("{}")
    recipe = json.loads((CODE / "command_builder.json").read_text())
    for seed in (1001, 1002):
        with pytest.raises(ValueError, match="excluded"):
            build_training_argv(
                recipe,
                arm="wide",
                fold=0,
                seed=seed,
                run_name="test",
                split_manifest=split,
                mode="fit",
            )
    argv = build_training_argv(
        recipe,
        arm="deep",
        fold=0,
        seed=1003,
        run_name="test",
        split_manifest=split,
        mode="fit",
    )
    assert argv[argv.index("--epochs") + 1] == "6"
    assert argv[argv.index("--checkpoint-epoch") + 1] == "5"
    assert "--no-eval-during-fit" in argv
