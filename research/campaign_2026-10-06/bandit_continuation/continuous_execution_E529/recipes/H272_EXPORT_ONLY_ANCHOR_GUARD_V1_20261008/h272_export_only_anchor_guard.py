"""Run frozen H272 export with the C1 producer-checkpoint guard correction.

The parent sidecar names the ``model_ep1.pt`` bytes that produced its
prediction. The frozen H272 exporter compared that field to the distinct
``model.pt`` alias file hash. This adapter changes only that comparison;
H272's original alias/tensor, prediction-byte, and row-order gates remain in
the unchanged exporter.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any  # Sidecar and checkpoint JSON values have heterogeneous schemas.

CAMPAIGN_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CAMPAIGN_ROOT.parents[3]
RUNTIME_ROOT = CAMPAIGN_ROOT / "recipes" / "H272_DIRECT_LABEL_READOUT_V1_RUNTIME"
RUNTIME_SCRIPTS = RUNTIME_ROOT / "scripts"
RUNTIME_SOURCE = RUNTIME_ROOT / "src"
SOURCE_LOCK_PATH = (
    CAMPAIGN_ROOT / "recipes" / "H272_DIRECT_LABEL_FROZEN_RUNTIME_SOURCE_LOCK_V1_20261008.json"
)
SOURCE_LOCK_SHA256 = "a8d56811e97783bcaa354a28beb992f8ce8acebc1b62f70f945c6697426aaa76"
RUNTIME_RELATIVE_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/"
    "recipes/H272_DIRECT_LABEL_READOUT_V1_RUNTIME"
)
RUNTIME_MEMBER_COUNT = 55
EXPORTER_PATH = RUNTIME_SCRIPTS / "train_conditional_readout.py"


def _sha256(path: Path) -> str:
    """Hash a pinned file in fixed-size chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_h272_source() -> dict[str, str]:
    """Verify H272's exact frozen runtime, recipe, and source-lock identity."""
    if _sha256(SOURCE_LOCK_PATH) != SOURCE_LOCK_SHA256:
        raise RuntimeError("the frozen H272 source lock changed")
    source_lock = json.loads(SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
    if (
        source_lock.get("schema") != "h272-frozen-runtime-source-lock-v1"
        or source_lock.get("runtime_root") != RUNTIME_RELATIVE_PATH
        or source_lock.get("member_count") != RUNTIME_MEMBER_COUNT
    ):
        raise RuntimeError("the H272 source lock does not name the frozen V1 runtime")
    members = source_lock.get("members_sha256")
    if not isinstance(members, dict) or len(members) != RUNTIME_MEMBER_COUNT:
        raise RuntimeError("the H272 source lock member map is malformed")

    observed: dict[str, str] = {}
    for relative_path, expected_sha256 in sorted(members.items()):
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe H272 source-lock path: {relative_path}")
        path = RUNTIME_ROOT / relative
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"frozen H272 runtime member changed: {relative_path}")
        observed[relative_path] = actual_sha256

    recipe_path = source_lock.get("recipe_path")
    recipe_sha256 = source_lock.get("recipe_sha256")
    if not isinstance(recipe_path, str) or not isinstance(recipe_sha256, str):
        raise RuntimeError("the H272 source lock has no frozen recipe binding")
    relative_recipe = Path(recipe_path)
    if relative_recipe.is_absolute() or ".." in relative_recipe.parts:
        raise RuntimeError("the H272 source lock contains an unsafe recipe path")
    actual_recipe_sha256 = _sha256(PROJECT_ROOT / relative_recipe)
    if actual_recipe_sha256 != recipe_sha256:
        raise RuntimeError("the frozen H272 recipe changed")
    observed[recipe_path] = actual_recipe_sha256
    return observed


def _load_frozen_exporter() -> ModuleType:
    """Import the pinned H272 exporter and its matching local modules."""
    for path in (RUNTIME_SOURCE, RUNTIME_SCRIPTS):
        value = str(path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    spec = importlib.util.spec_from_file_location("h272_frozen_v1_exporter", EXPORTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the frozen H272 exporter")
    exporter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = exporter
    spec.loader.exec_module(exporter)
    if Path(exporter.__file__).resolve() != EXPORTER_PATH.resolve():
        raise RuntimeError("the imported H272 exporter resolved to an unexpected path")
    if Path(exporter.trainer.__file__).resolve() != (RUNTIME_SCRIPTS / "train_rnn.py"):
        raise RuntimeError("the H272 exporter imported an unexpected train_rnn module")
    return exporter


def install_corrected_parent_anchor_guard(exporter: ModuleType) -> None:
    """Correct only the sidecar producer-checkpoint comparison for export."""

    def validate_anchor_reference(split: str, row_groups: tuple[int, ...]) -> dict[str, Any] | None:
        if split != "valid":
            return None
        prediction = exporter.ROOT / "cache" / f"pred_{exporter.PARENT_RUN}_valid.npy"
        metadata_path = prediction.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("checkpoint_sha256") != exporter.CHECKPOINT_SHA256:
            raise ValueError(
                "the existing C1 parent prediction does not bind to its frozen model_ep1.pt"
            )
        if metadata.get("prediction_sha256") != exporter.PARENT_VALID_PREDICTION_SHA256:
            raise ValueError("the existing C1 anchor prediction differs from its frozen bytes")
        if tuple(map(int, metadata.get("row_groups", []))) != row_groups:
            raise ValueError("the C1 anchor prediction row order differs from the export split")
        if exporter.file_sha256(prediction) != metadata.get("prediction_sha256"):
            raise ValueError("the existing C1 parent prediction bytes changed")
        return {
            "prediction_path": str(prediction.relative_to(exporter.ROOT)),
            "prediction_sha256": metadata["prediction_sha256"],
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "identity_check": (
                "the unchanged exporter separately verifies model.pt alias bytes and "
                "tensor equality with the pinned model_ep1.pt checkpoint"
            ),
        }

    exporter._validate_anchor_reference = validate_anchor_reference


def _require_valid_split(argv: list[str]) -> None:
    """Reject train export requests while preserving the frozen valid default."""
    split_values: list[str] = []
    for index, argument in enumerate(argv):
        if argument == "--split":
            if index + 1 >= len(argv):
                raise SystemExit("H272 export adapter requires --split valid")
            split_values.append(argv[index + 1])
        elif argument.startswith("--split="):
            split_values.append(argument.partition("=")[2])
    if split_values and (len(split_values) != 1 or split_values[0] != "valid"):
        raise SystemExit("this H272 adapter supports only the valid export split")


def main() -> None:
    """Run the frozen H272 exporter only for its valid-export subcommand."""
    if len(sys.argv) < 2 or sys.argv[1] != "export":
        raise SystemExit("this H272 adapter supports only the export subcommand")
    _require_valid_split(sys.argv[2:])
    source_hashes = verify_frozen_h272_source()
    exporter = _load_frozen_exporter()
    install_corrected_parent_anchor_guard(exporter)
    print(
        json.dumps(
            {
                "event": "h272_export_only_parent_anchor_guard_installed",
                "adapter_id": "H272_C1S3_EXPORT_ONLY_ANCHOR_GUARD_V1_20261008",
                "guard_change": ("sidecar checkpoint_sha256 compares to frozen model_ep1.pt hash"),
                "unchanged_gates": [
                    "model.pt alias hash and tensor equality",
                    "parent prediction hash",
                    "parent prediction row order",
                    "fixed H272 export destination and epoch",
                ],
                "source_lock_sha256": SOURCE_LOCK_SHA256,
                "verified_source_members": len(source_hashes) - 1,
                "source_hashes": source_hashes,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    exporter.main()


if __name__ == "__main__":
    main()
