"""Build the H-ADH-R history target cache from frozen full-order predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "src"))
from wnn.history_swap_cache import (  # noqa: E402
    BASE_CACHE_PATH,
    BASE_METADATA_SHA256,
    PARENT_ATTEMPT,
    PARENT_CONTRACT_PATH,
    PARENT_PROVENANCE_PATH,
    PARENT_RUN,
    PARENT_TERMINAL_PATH,
    SOURCE_MANIFEST_PATH,
    SOURCE_MANIFEST_SHA256,
    build_history_cache,
)

from wnn.splits import file_sha256  # noqa: E402


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve(strict=True)
    else:
        resolved = (project_root / path).resolve(strict=True)
    if not resolved.is_relative_to(project_root):
        raise ValueError(f"path escapes project root: {value}")
    return resolved


def _output_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() and ".." in path.parts:
        raise ValueError(f"output path must not contain parent traversal: {value}")
    resolved = (path if path.is_absolute() else project_root / path).resolve()
    if not resolved.is_relative_to(project_root):
        raise ValueError(f"output path escapes project root: {value}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--base-cache", default=BASE_CACHE_PATH)
    parser.add_argument("--expected-base-metadata-sha256", default=BASE_METADATA_SHA256)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-run", default=PARENT_RUN)
    parser.add_argument("--parent-attempt", default=PARENT_ATTEMPT)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--parent-checkpoint-sha256", required=True)
    parser.add_argument("--parent-contract", default=PARENT_CONTRACT_PATH)
    parser.add_argument("--parent-contract-sha256", required=True)
    parser.add_argument("--parent-terminal-receipt", default=PARENT_TERMINAL_PATH)
    parser.add_argument("--parent-terminal-receipt-sha256", required=True)
    parser.add_argument("--parent-provenance", default=PARENT_PROVENANCE_PATH)
    parser.add_argument("--parent-provenance-sha256", required=True)
    parser.add_argument("--parent-prediction-train", required=True)
    parser.add_argument("--parent-prediction-valid", required=True)
    parser.add_argument("--chunk-sequences", type=int, default=16)
    parser.add_argument("--min-free-gb", type=float, default=25.0)
    args = parser.parse_args()
    project_root = args.project_root.resolve(strict=True)
    source_manifest = _project_path(project_root, SOURCE_MANIFEST_PATH)
    if file_sha256(source_manifest) != SOURCE_MANIFEST_SHA256:
        raise ValueError("sealed V3 source manifest changed")
    parent_run = args.parent_run
    if parent_run != PARENT_RUN or args.parent_attempt != PARENT_ATTEMPT:
        raise ValueError("history target builder accepts only the frozen H230 parent ancestry")
    parent = {
        "run": parent_run,
        "attempt_id": args.parent_attempt,
        "checkpoint_path": str(
            _project_path(project_root, args.parent_checkpoint).relative_to(project_root)
        ),
        "checkpoint_sha256": args.parent_checkpoint_sha256,
        "contract_path": str(
            _project_path(project_root, args.parent_contract).relative_to(project_root)
        ),
        "contract_sha256": args.parent_contract_sha256,
        "terminal_receipt_path": str(
            _project_path(project_root, args.parent_terminal_receipt).relative_to(project_root)
        ),
        "terminal_receipt_sha256": args.parent_terminal_receipt_sha256,
        "provenance_path": str(
            _project_path(project_root, args.parent_provenance).relative_to(project_root)
        ),
        "provenance_sha256": args.parent_provenance_sha256,
    }
    output = _output_path(project_root, args.output)
    result = build_history_cache(
        data_root=project_root,
        base_cache=_project_path(project_root, args.base_cache),
        output=output,
        parent=parent,
        parent_prediction_paths={
            "train": _project_path(project_root, args.parent_prediction_train),
            "valid": _project_path(project_root, args.parent_prediction_valid),
        },
        expected_base_metadata_sha256=args.expected_base_metadata_sha256,
        chunk_sequences=args.chunk_sequences,
        min_free_gb=args.min_free_gb,
    )
    print(
        json.dumps(
            {"output": str(result), "metadata_sha256": file_sha256(result / "metadata.json")},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
