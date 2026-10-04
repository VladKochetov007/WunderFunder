"""Run the staged E426 trainer against the repository's real data root."""

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
    """Prioritize the staged WNN overrides while retaining frozen root modules."""
    paths = [
        STAGING_ROOT / "src",
        REPO_ROOT / "src",
        REPO_ROOT / "scripts",
    ]
    normalized = {str(path) for path in paths}
    sys.path[:] = [path for path in sys.path if path not in normalized]
    sys.path[0:0] = [str(path) for path in paths]


def _load_trainer() -> ModuleType:
    """Load the staged trainer as a module after installing its import overlay."""
    _put_import_roots_first()
    spec = importlib.util.spec_from_file_location("e426_staged_train_rnn", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load staged trainer at {TRAINER_PATH}")
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    trainer.ROOT = REPO_ROOT
    return trainer


def _source_closure() -> dict[str, object]:
    """Hash every WNN source and staged override used by this executable overlay."""
    paths = set((REPO_ROOT / "src" / "wnn").rglob("*.py"))
    paths.update((STAGING_ROOT / "src" / "wnn").rglob("*.py"))
    paths.update((STAGING_ROOT / "scripts").glob("train_rnn.py"))
    paths.add(Path(__file__).resolve())
    entries = []
    for path in sorted(paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({"path": path.relative_to(REPO_ROOT).as_posix(), "sha256": digest})
    payload = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {
        "schema": "e426-python-source-closure-v1",
        "files": entries,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> None:
    """Run E426 with repository data paths and include the exact source closure."""
    trainer = _load_trainer()
    trainer.E426_SOURCE_CLOSURE = _source_closure()
    trainer.main()


if __name__ == "__main__":
    main()
