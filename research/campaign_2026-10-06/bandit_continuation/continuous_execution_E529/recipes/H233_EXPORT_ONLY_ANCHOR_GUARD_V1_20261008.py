"""Run the frozen H233 exporter with one corrected anchor metadata guard.

The frozen V2 exporter checks the C1 prediction sidecar's producer checkpoint
hash. That field identifies ``model_ep1.pt``. V2 compared it to the distinct
``model.pt`` alias file hash. This export-only wrapper replaces that one guard
with the same checks against the producer checkpoint hash, while the original
exporter still performs its separate alias hash and tensor-equality checks.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

CAMPAIGN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CAMPAIGN_ROOT.parents[3]
RUNTIME_ROOT = CAMPAIGN_ROOT / "recipes" / "H233_EXECUTED_SOURCES_V2_RUNTIME"
RUNTIME_SCRIPTS = RUNTIME_ROOT / "scripts"
RUNTIME_SOURCE = RUNTIME_ROOT / "src"
SOURCE_LOCK_PATH = (
    CAMPAIGN_ROOT / "recipes" / "H233_C1S3_FROZEN_RUNTIME_SOURCE_LOCK_V2_20261008.json"
)
SOURCE_LOCK_SHA256 = "831c3de1ea5ae48a9330f82a2cc829fb93021c8ac82f5dff9e1da40ea8bf1b3b"
SOURCE_LOCK_MEMBER_COUNT = 11
EXPORTER_PATH = RUNTIME_SCRIPTS / "train_conditional_readout.py"


def _sha256(path: Path) -> str:
    """Hash a pinned source or prediction file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_original_source_closure() -> dict[str, str]:
    """Fail closed unless the original, reviewed H233 source closure is intact."""
    if _sha256(SOURCE_LOCK_PATH) != SOURCE_LOCK_SHA256:
        raise RuntimeError("the original H233 V2 runtime source lock changed")
    lock = json.loads(SOURCE_LOCK_PATH.read_text())
    if lock.get("runtime_root") != str(RUNTIME_ROOT.relative_to(PROJECT_ROOT)):
        raise RuntimeError("the H233 source lock names a different frozen runtime root")
    pinned = lock.get("pinned_source_validation")
    if not isinstance(pinned, dict) or len(pinned) != SOURCE_LOCK_MEMBER_COUNT:
        raise RuntimeError("the H233 V2 source lock member set is malformed")
    observed: dict[str, str] = {}
    for relative, expected in sorted(pinned.items()):
        path = RUNTIME_ROOT / relative
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen H233 source changed: {relative}")
        observed[relative] = actual
    return observed


def _load_frozen_exporter(source_hashes: dict[str, str] | None = None) -> ModuleType:
    """Import the reviewed exporter and its matching frozen training modules."""
    if source_hashes is None:
        _verify_original_source_closure()
    for path in (RUNTIME_SCRIPTS, RUNTIME_SOURCE):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    spec = importlib.util.spec_from_file_location("h233_frozen_v2_exporter", EXPORTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the frozen H233 V2 exporter")
    exporter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = exporter
    spec.loader.exec_module(exporter)
    return exporter


def install_corrected_anchor_guard(exporter: ModuleType) -> None:
    """Replace only the sidecar producer-checkpoint comparison.

    The prediction sidecar records the hash of the ``model_ep1.pt`` file that
    produced it. The original exporter separately verifies that ``model.pt``
    is the expected alias and that its model tensors equal the pinned epoch-1
    checkpoint before this guard is called. Prediction bytes and row order are
    still checked here before any export rows are emitted.
    """

    def validate_anchor_reference(
        split: str, row_groups: tuple[int, ...]
    ) -> dict[str, Any] | None:
        if split != "valid":
            return None
        prediction = exporter.ROOT / "cache" / f"pred_{exporter.PARENT_RUN}_valid.npy"
        metadata_path = prediction.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("checkpoint_sha256") != exporter.CHECKPOINT_SHA256:
            raise ValueError(
                "the existing C1 parent prediction does not bind to its frozen model_ep1.pt"
            )
        if metadata.get("prediction_sha256") != exporter.PARENT_VALID_PREDICTION_SHA256:
            raise ValueError("the existing C1 anchor prediction differs from its frozen E511 bytes")
        if tuple(map(int, metadata.get("row_groups", []))) != row_groups:
            raise ValueError("the C1 anchor prediction row order differs from the export split")
        if exporter.file_sha256(prediction) != metadata.get("prediction_sha256"):
            raise ValueError("the existing C1 anchor prediction bytes changed")
        return {
            "prediction_path": str(prediction.relative_to(exporter.ROOT)),
            "prediction_sha256": metadata["prediction_sha256"],
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "identity_check": "E511 verified model.pt tensors equal the pinned model_ep1.pt tensors",
        }

    exporter._validate_anchor_reference = validate_anchor_reference


def main() -> None:
    """Accept only the frozen export subcommand, then run it with the narrow fix."""
    if len(sys.argv) < 2 or sys.argv[1] != "export":
        raise SystemExit("this H233 adapter supports only the export subcommand")
    source_hashes = _verify_original_source_closure()
    exporter = _load_frozen_exporter(source_hashes)
    install_corrected_anchor_guard(exporter)
    print(
        json.dumps(
            {
                "event": "h233_export_only_anchor_guard_installed",
                "guard_change": "sidecar checkpoint_sha256 compares to frozen model_ep1.pt hash",
                "alias_guard": "original frozen model.pt hash and tensor-equality checks retained",
                "source_lock_sha256": SOURCE_LOCK_SHA256,
                "verified_source_members": len(source_hashes),
                "source_hashes": source_hashes,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    exporter.main()


if __name__ == "__main__":
    main()
