"""Run the isolated E425 trainer against repository data and record its closure."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[5]
STAGING_ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = STAGING_ROOT / "scripts" / "train_rnn.py"


def _put_import_roots_first() -> None:
    """Prioritize staged WNN overrides and preserve root fallbacks."""
    paths = [STAGING_ROOT / "src", REPO_ROOT / "src", REPO_ROOT / "scripts"]
    normalized = {str(path) for path in paths}
    sys.path[:] = [path for path in sys.path if path not in normalized]
    sys.path[0:0] = [str(path) for path in paths]


def _load_trainer() -> ModuleType:
    """Import the staged trainer after installing its source overlay."""
    _put_import_roots_first()
    spec = importlib.util.spec_from_file_location("e425_staged_train_rnn", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load staged trainer at {TRAINER_PATH}")
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    trainer.ROOT = REPO_ROOT
    return trainer


def _source_closure() -> dict[str, object]:
    """Hash all frozen WNN roots and staged E425 overrides used by this entrypoint."""
    paths = set((REPO_ROOT / "src" / "wnn").rglob("*.py"))
    paths.update((STAGING_ROOT / "src" / "wnn").rglob("*.py"))
    paths.update((STAGING_ROOT / "scripts").glob("*.py"))
    entries = [
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(paths)
    ]
    payload = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {
        "schema": "e425-python-source-closure-v1",
        "files": entries,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> None:
    """Set runtime provenance and invoke the actual staged training loop."""
    trainer = _load_trainer()
    trainer.E425_SOURCE_CLOSURE = _source_closure()
    trainer.main()


if __name__ == "__main__":
    main()
