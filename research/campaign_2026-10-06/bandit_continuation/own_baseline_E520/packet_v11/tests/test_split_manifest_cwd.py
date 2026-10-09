"""Exercise trainer split-manifest path resolution in a child process."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from profile_worker import SPLIT_RELATIVE, staged_split_manifest_argument


def test_absolute_staged_manifest_survives_trainer_working_directory(
    tmp_path: Path,
) -> None:
    working_root = tmp_path / "kaggle-working"
    source_root = working_root / "e520_source_v11"
    manifest = source_root / SPLIT_RELATIVE
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"version": 1}\n', encoding="utf-8")

    trainer_probe = (
        "from pathlib import Path; import sys; "
        "path = Path(sys.argv[1]).resolve(); "
        "print(path.read_text(encoding='utf-8'), end='')"
    )
    legacy = subprocess.run(
        [sys.executable, "-c", trainer_probe, SPLIT_RELATIVE.as_posix()],
        cwd=working_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert legacy.returncode != 0
    assert "FileNotFoundError" in legacy.stderr

    absolute_argument = staged_split_manifest_argument(source_root)
    fixed = subprocess.run(
        [sys.executable, "-c", trainer_probe, absolute_argument],
        cwd=working_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert fixed.returncode == 0, fixed.stderr
    assert fixed.stdout == '{"version": 1}\n'
    assert Path(absolute_argument).is_absolute()
