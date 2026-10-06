from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from derive_source import derive_trainer

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "research/campaign_2026-10-05/w384_v7/bundle_source"
SOURCE_MANIFEST = ROOT / "research/campaign_2026-10-05/w384_v7/bundle/bundle-manifest.json"
EXPECTED_DERIVED_TRAINER_SHA256 = (
    "8b691b00a0727bc5081c96973ff3e2cb4cb557b5a65775501513443e781bf4de"
)


def _ignore_generated(_: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in {"__pycache__", ".pytest_cache"} or name.endswith(".pyc")
    }


def main() -> int:
    """Run the legacy CUDA-preflight tests against an isolated derived loader."""
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    descriptors = {row["path"]: row for row in manifest["files"]}
    for relative_path in (
        "scripts/train_rnn.py",
        "src/wnn/data.py",
        "tests/test_trainer_resume_integration.py",
        "tests/test_training_state.py",
    ):
        source = SOURCE_ROOT / relative_path
        descriptor = descriptors[relative_path]
        content = source.read_bytes()
        if (
            len(content) != descriptor["size"]
            or hashlib.sha256(content).hexdigest() != descriptor["sha256"]
        ):
            raise ValueError(f"W384 source differs from bound manifest: {relative_path}")
    with tempfile.TemporaryDirectory(prefix="wnn-width-cpu-preflight-") as directory:
        staged = Path(directory)
        shutil.copytree(SOURCE_ROOT / "src", staged / "src", ignore=_ignore_generated)
        shutil.copytree(SOURCE_ROOT / "scripts", staged / "scripts", ignore=_ignore_generated)
        tests = staged / "tests"
        tests.mkdir()
        for name in (
            "__init__.py",
            "test_training_state.py",
            "test_trainer_resume_integration.py",
        ):
            source = SOURCE_ROOT / "tests" / name
            if source.is_file():
                shutil.copy2(source, tests / name)

        trainer = staged / "scripts/train_rnn.py"
        derived = derive_trainer(trainer.read_text())
        if hashlib.sha256(derived.encode()).hexdigest() != EXPECTED_DERIVED_TRAINER_SHA256:
            raise ValueError("derived trainer differs from the pinned W384 transport source")
        trainer.write_text(derived)
        shutil.copy2(HERE / "bounded_loader.py", staged / "src/wnn/bounded_loader.py")

        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "WNN_INTEGRATION_DEVICE": "cpu",
                "WNN_LOADER_WORKERS": "0",
                "PYTHONPATH": os.pathsep.join(
                    (str(staged / "src"), str(staged / "scripts"), str(staged))
                ),
            }
        )
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "WNN_THREADS",
        ):
            environment[name] = "1"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                f"--junitxml={HERE / 'legacy_cpu_tests.xml'}",
                "tests/test_trainer_resume_integration.py",
                "tests/test_training_state.py",
            ],
            cwd=staged,
            env=environment,
            check=False,
        )
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
