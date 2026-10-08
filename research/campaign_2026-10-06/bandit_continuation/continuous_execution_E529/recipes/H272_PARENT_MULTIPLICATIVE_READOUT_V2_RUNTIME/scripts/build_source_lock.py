"""Regenerate the reviewed V2 source/recipe lock after an identified source edit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[5]
LOCK_PATH = PACKAGE_ROOT / "H272_PARENT_READOUT_V2_SOURCE_LOCK_V1_20261008.json"
RECIPE_PATHS = {
    "cleanfold": "recipes/H272_PARENT_READOUT_V2_CLEANFOLD_20261008.json",
    "full_data": "recipes/H272_PARENT_READOUT_V2_FULL_DATA_20261008.json",
}
NATIVE_PATHS = (
    "research/campaign_2026-10-04/E416_native_q8_build/native/src/lib.rs",
    "research/campaign_2026-10-04/E416_native_q8_build/native/src/python.rs",
    "research/campaign_2026-10-04/E416_native_q8_build/native/src/rnn_fast.rs",
    "research/campaign_2026-10-04/E416_native_q8_build/native/src/readout.rs",
    "research/campaign_2026-10-04/E416_native_q8_build/native/src/stream.rs",
    "research/campaign_2026-10-04/E416_native_q8_build/native/Cargo.toml",
    "research/campaign_2026-10-04/E416_native_q8_build/native/Cargo.lock",
)
BASELINE_PATHS = (
    "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/recipes/"
    "H272_DIRECT_LABEL_READOUT_V1_RUNTIME/scripts/train_conditional_readout.py",
    "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/recipes/"
    "H272_DIRECT_LABEL_READOUT_V1_RUNTIME/src/wnn/conditional_readout.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    source_files = {
        path.relative_to(PACKAGE_ROOT).as_posix(): sha256(path)
        for path in sorted(PACKAGE_ROOT.rglob("*"))
        if path.is_file()
        and path != LOCK_PATH
        and "__pycache__" not in path.parts
        and "target" not in path.parts
    }
    payload = {
        "schema": "h272-parent-readout-source-lock-v1",
        "source_sha256": source_files,
        "recipe_sha256": {
            key: sha256(PACKAGE_ROOT / relative) for key, relative in RECIPE_PATHS.items()
        },
        "native_source_sha256": {path: sha256(REPO_ROOT / path) for path in NATIVE_PATHS},
        "baseline_h272_source_sha256": {path: sha256(REPO_ROOT / path) for path in BASELINE_PATHS},
    }
    LOCK_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(LOCK_PATH), "sha256": sha256(LOCK_PATH)}, sort_keys=True))


if __name__ == "__main__":
    main()
