"""Replay one accepted fixed H512-in-TL3 package on source-qualified F0."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# F0Source is imported dynamically from the frozen replay packet.
from endpoint_contract import (
    ENDPOINTS,
    SCRATCH_ROOT,
    physical_scratch,
    validate_candidate,
)

SPLIT_PATH = Path("research/campaign_2026-09-29/artifacts/splits/fold0.json")
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
VALID_FOOTER_SHA256 = "4e0a6ad91c87449cc74940e83195ec51c8900d31a25774537bc8283164a7d93b"
REPLAY_SOURCE = Path("research/campaign_2026-10-18/online_features_r1/tl3_replay")
EXPECTED_GROUPS = 468
EXPECTED_ROWS = 9_360_000
EXPECTED_WARMUP_ROWS = 46_332
EXPECTED_PREDICTED_ROWS = 9_313_668
EXPECTED_CACHE_PAYLOAD_BYTES = 74_880_000


def validate_prediction_encoding(path: Path) -> None:
    """Require the exact NPY shape, dtype and payload, accounting for its header."""
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        values.shape != (EXPECTED_GROUPS, 20_000, 2)
        or values.dtype != np.dtype("float32")
        or not values.flags.c_contiguous
        or values.nbytes != EXPECTED_CACHE_PAYLOAD_BYTES
        or path.stat().st_size != values.offset + values.nbytes
    ):
        raise ValueError("native prediction NPY encoding or coverage differs")


def _parse_args() -> argparse.Namespace:
    """Parse explicit candidate and create-only output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("endpoint", choices=tuple(ENDPOINTS))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--output-sidecar", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    return parser.parse_args()


def _scratch_output(path: Path) -> Path:
    """Resolve a fresh output directly under approved physical scratch."""
    output = physical_scratch(path)
    if output.parent != SCRATCH_ROOT.resolve(strict=True):
        raise ValueError(f"output must be a direct child of approved scratch: {SCRATCH_ROOT}")
    if output.exists():
        raise FileExistsError(output)
    return output


def _source_paths(root: Path) -> None:
    """Import the shared replay producer from the exact project checkout."""
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / REPLAY_SOURCE))


def _validated_source(root: Path) -> Any:
    """Require the frozen F0 evaluation groups and source footer."""
    from f0_source import validate_f0_source

    split_path = root / SPLIT_PATH
    source = validate_f0_source(root, split_path, None)
    if (
        source.split.sha256 != SPLIT_SHA256
        or len(source.groups) != EXPECTED_GROUPS
        or source.data_identity.get("footer_sha256_from_split") != VALID_FOOTER_SHA256
    ):
        raise ValueError("replay source differs from the frozen F0 evaluation manifest")
    return source


def replay(args: argparse.Namespace) -> dict[str, Any]:
    """Run one fixed package and bind its outputs to accepted source receipts."""
    root = args.root.resolve(strict=True)
    archive = args.zip.resolve(strict=True)
    build_receipt = args.build_receipt.resolve(strict=True)
    predictions = _scratch_output(args.output_predictions)
    sidecar = _scratch_output(args.output_sidecar)
    receipt_path = _scratch_output(args.output_receipt)
    if len({predictions, sidecar, receipt_path}) != 3:
        raise ValueError("prediction, sidecar, and receipt paths must be distinct")
    binding_before = validate_candidate(root, args.endpoint, archive, build_receipt)
    _source_paths(root)
    source = _validated_source(root)
    from f0_replay import replay_groups
    from native_runtime import sha256_file as replay_sha256
    from native_runtime import write_json_create_only

    result = replay_groups(source, archive, SCRATCH_ROOT, predictions, reset_check=True)
    try:
        binding_after = validate_candidate(root, args.endpoint, archive, build_receipt)
        if binding_after != binding_before:
            raise ValueError("candidate source binding changed during replay")
        prediction_sha = replay_sha256(predictions)
        validate_prediction_encoding(predictions)
        if (
            result.rows != EXPECTED_ROWS
            or result.warmup_rows != EXPECTED_WARMUP_ROWS
            or result.predicted_rows != EXPECTED_PREDICTED_ROWS
            or result.reset_aba_equal is not True
        ):
            raise ValueError("native replay did not cover the frozen full-F0 endpoint")
        source_path = root / SPLIT_PATH
        prediction_record: dict[str, Any] = {
            "schema": "wnn.h512-tl3-f0-endpoint-predictions.v1",
            "candidate": f"h512_{args.endpoint}_tl3_fixed034",
            "endpoint": args.endpoint,
            "zip_sha256": binding_after["zip_sha256"],
            "build_receipt_path": binding_after["build_receipt_path"],
            "build_receipt_sha256": binding_after["build_receipt_sha256"],
            "fit_terminal_receipt_path": binding_after["fit_terminal_path"],
            "fit_terminal_receipt_sha256": binding_after["fit_terminal_sha256"],
            "derivation_receipt_path": binding_after["derivation_receipt_path"],
            "derivation_receipt_sha256": binding_after["derivation_receipt_sha256"],
            "prediction_path": str(predictions),
            "prediction_sha256": prediction_sha,
            "shape": [EXPECTED_GROUPS, 20_000, 2],
            "dtype": "float32",
            "split_manifest_sha256": source.split.sha256,
            "row_groups": list(source.groups),
            "sequence_ids": [source.sequence_by_group[group] for group in source.groups],
            "replay_receipt_path": str(receipt_path),
            "callback_contract": "Feature-only sequence replay; targets and score mask were not read.",
            "ancestry": binding_after["whole_endpoint_ancestry"],
            "evidence_classification": binding_after["evidence_classification"],
        }
        replay_record: dict[str, Any] = {
            "schema": "wnn.h512-tl3-f0-endpoint-replay.v1",
            "status": "REPLAY_COMPLETE",
            "endpoint": args.endpoint,
            "package_binding": binding_after,
            "split_manifest": str(source_path),
            "split_manifest_sha256": source.split.sha256,
            "source_data_identity": source.data_identity,
            "prediction_path": str(predictions),
            "prediction_sha256": prediction_sha,
            "prediction_shape": [EXPECTED_GROUPS, 20_000, 2],
            "prediction_dtype": "float32",
            "replay": result.__dict__,
            "f0_source_sha256": replay_sha256(root / REPLAY_SOURCE / "f0_source.py"),
            "f0_replay_sha256": replay_sha256(root / REPLAY_SOURCE / "f0_replay.py"),
            "native_runtime_sha256": replay_sha256(root / REPLAY_SOURCE / "native_runtime.py"),
            "replay_script_sha256": replay_sha256(Path(__file__).resolve()),
            "endpoint_contract_sha256": replay_sha256(
                Path(__file__).with_name("endpoint_contract.py")
            ),
            "scored_targets_read": False,
            "quality_score_computed": False,
        }
        write_json_create_only(receipt_path, replay_record)
        prediction_record["replay_receipt_sha256"] = replay_sha256(receipt_path)
        write_json_create_only(sidecar, prediction_record)
        return replay_record
    except Exception:
        predictions.unlink(missing_ok=True)
        raise


def main() -> None:
    """Run the endpoint replay without invoking the legacy hard-coded receipt writer."""
    args = _parse_args()
    result = replay(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "endpoint": args.endpoint,
                "rows": result["replay"]["rows"],
            }
        )
    )


if __name__ == "__main__":
    main()
