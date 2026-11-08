from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

SOURCE6_SHA256 = "8d97910b523931bb115c4a979c32ec1c37c7353a666bf0dcf6f22e5de246ed35"
SCORE_SCRIPT_SHA256 = "56502eb04c4cbe2495e78d6236ef354cb0820b1ecc63ab820610a804cfac89a3"
PRIMARY_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
AUXILIARY_SHA256 = "26b448bad7f60612cea482f4abaf5f184e5a724228eef227d7e78a8155190075"
PRIMARY_GROUPS = 1_873
PRIMARY_ROWS = 3_528_273
TL3_WP = 0.6923774245652725


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _source_identity(source_root: Path) -> dict[str, object]:
    entrypoint = source_root / "scripts/predict_sequence.py"
    paths = sorted((source_root / "src").rglob("*.py"))
    if not entrypoint.is_file() or entrypoint.is_symlink():
        raise ValueError("pinned Source6 scorer entrypoint is missing or unsafe")
    if entrypoint not in paths:
        paths.append(entrypoint)
    if any(path.is_symlink() for path in paths):
        raise ValueError("pinned Source6 tree contains a Python symlink")
    files = {path.relative_to(source_root).as_posix(): _sha256(path) for path in paths}
    return {"sha256": _canonical_sha256(files), "files": files}


def _verify_source6(source_root: Path, expected: Mapping[str, object]) -> None:
    identity = _source_identity(source_root)
    script_sha = _sha256(source_root / "scripts/score_sequence.py")
    if (
        identity["sha256"] != SOURCE6_SHA256
        or identity["sha256"] != expected.get("source_sha256")
        or script_sha != SCORE_SCRIPT_SHA256
        or script_sha != expected.get("score_script_sha256")
    ):
        raise ValueError("Source6 files differ from the frozen scorer/source identity")


def _regular(path: Path, label: str, limit: int = 16_000_000) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"{label} must be a bounded regular file")
    return path.resolve(strict=True)


def _read_json(path: Path, label: str, limit: int = 16_000_000) -> dict[str, object]:
    raw = _regular(path, label, limit).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _load_source6(source_root: Path) -> tuple[object, ...]:
    sys.path.insert(0, str(source_root / "src"))
    import wnn

    package_path = Path(str(wnn.__file__)).resolve()
    if not package_path.is_relative_to(source_root.resolve()):
        raise ValueError("Source6 scorer import resolved outside its verified tree")
    from wnn.multistage_primary_report import (  # noqa: PLC0415
        PRIMARY_SCORED_ROW_COUNT,
        _score_primary_valid,
        _StreamingMetrics,
    )
    from wnn.splits import load_campaign_split  # noqa: PLC0415
    from wnn.training.sequence_config import load_config  # noqa: PLC0415
    from wnn.training.sequence_data import SequenceRecords  # noqa: PLC0415
    from wnn.training.sequence_fit_utils import collate_sequences  # noqa: PLC0415
    from wnn.training.sequence_monitor_identity import membership_sha256  # noqa: PLC0415
    from wnn.training.sequence_precision import configure_precision_mode  # noqa: PLC0415
    from wnn.training.sequence_predict import load_endpoint  # noqa: PLC0415

    return (
        PRIMARY_SCORED_ROW_COUNT,
        _StreamingMetrics,
        _score_primary_valid,
        load_campaign_split,
        load_config,
        SequenceRecords,
        collate_sequences,
        membership_sha256,
        load_endpoint,
        configure_precision_mode,
    )


def _load_request(path: Path, expected_sha: str) -> dict[str, object]:
    request_path = _regular(path, "score request", 32_000_000)
    if _sha256(request_path) != expected_sha:
        raise ValueError("score request bytes changed after Stage4 validation")
    request = _read_json(request_path, "score request", 32_000_000)
    if request.get("schema") != "wnn.multistage-primary-ensemble-score-request.v1":
        raise ValueError("unsupported primary ensemble score request")
    return request


def _verify_splits(request: Mapping[str, object], load_campaign_split: object):
    data_root = Path(str(request["data_root"]))
    primary_path = Path(str(request["primary_split"]["path"]))
    auxiliary_path = Path(str(request["auxiliary_split"]["path"]))
    registry = _regular(
        Path(str(request["portable_data_registry"]["path"])), "portable data registry"
    )
    if _sha256(registry) != request["portable_data_registry"].get("sha256"):
        raise ValueError("portable data registry changed after Stage4 ancestry validation")
    if _sha256(_regular(primary_path, "primary split manifest")) != request["primary_split"].get(
        "sha256"
    ) or _sha256(_regular(auxiliary_path, "auxiliary split manifest")) != request[
        "auxiliary_split"
    ].get("sha256"):
        raise ValueError("split manifest bytes changed after Stage4 ancestry validation")
    os.environ["WNN_PORTABLE_DATA_MANIFEST"] = str(registry)
    primary = load_campaign_split(primary_path, data_root)
    auxiliary = load_campaign_split(auxiliary_path, data_root)
    if (
        primary.sha256 != PRIMARY_SHA256
        or auxiliary.sha256 != AUXILIARY_SHA256
        or len(primary.train_groups) != 10_325
        or primary.valid_groups
        or len(primary.evaluation_groups) != PRIMARY_GROUPS
        or len(auxiliary.train_groups) != 8_260
        or len(auxiliary.evaluation_groups) != 2_065
    ):
        raise ValueError("Source6 worker split membership differs from the frozen protocol")
    return primary, auxiliary


def _verify_candidate_artifact_hashes(candidate: Mapping[str, object]) -> None:
    for key, path_key, limit in (
        ("trial_result_sha256", "trial_result_path", 64_000_000),
        ("trial_request_sha256", "trial_request_path", 16_000_000),
        ("config_sha256", "config_path", 16_000_000),
    ):
        path = _regular(Path(str(candidate[path_key])), path_key, limit)
        if _sha256(path) != candidate.get(key):
            raise ValueError(f"candidate {path_key} changed after validation")


def score_member(args: argparse.Namespace) -> dict[str, object]:
    """Score one validated candidate with the exact Source6 primary scorer."""
    request_path = _regular(args.request, "score request", 32_000_000)
    request = _load_request(request_path, args.request_sha256)
    source_root = Path(str(request["source6"]["root"])).resolve(strict=True)
    _verify_source6(source_root, request["source6"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise ValueError("worker visibility differs from its registered GPU UUID")
    (
        primary_rows,
        _,
        score_primary_valid,
        load_campaign_split,
        load_config,
        sequence_records,
        _,
        membership_sha256,
        load_endpoint,
        configure_precision_mode,
    ) = _load_source6(source_root)
    primary, auxiliary = _verify_splits(request, load_campaign_split)
    if primary_rows != PRIMARY_ROWS:
        raise ValueError("pinned Source6 scored-row contract changed")
    candidates = request.get("candidates")
    if not isinstance(candidates, list) or not 0 <= args.member_index < len(candidates):
        raise ValueError("candidate index is outside the validated Stage4 ensemble")
    candidate = candidates[args.member_index]
    if not isinstance(candidate, Mapping) or candidate.get("gpu_uuid") != args.gpu_uuid:
        raise ValueError("worker candidate/GPU binding differs from the parent request")
    checkpoint = _regular(
        Path(str(candidate["checkpoint_path"])), "fixed-final checkpoint", 256 * 1024 * 1024
    )
    if _sha256(checkpoint) != candidate.get("checkpoint_sha256"):
        raise ValueError("fixed-final checkpoint changed after Stage4 ancestry validation")
    _verify_candidate_artifact_hashes(candidate)
    config = load_config(Path(str(candidate["config_path"])))
    configure_precision_mode("fp32")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("each Source6 score worker requires its one visible assigned GPU")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    if "B300" not in device_name.upper():
        raise ValueError(f"assigned primary scorer device is not a B300: {device_name}")
    model, payload = load_endpoint(checkpoint, device)
    provenance = payload.get("provenance")
    identity = provenance.get("identity") if isinstance(provenance, Mapping) else None
    data = payload.get("data")
    expected_fit = [["train", group] for group in auxiliary.train_groups]
    expected_control = [["train", group] for group in auxiliary.evaluation_groups]
    if (
        not isinstance(identity, Mapping)
        or not isinstance(data, Mapping)
        or _canonical_sha256(identity) != candidate.get("producer_identity_sha256")
        or payload.get("config") != config
        or payload.get("completed_epochs") != config.get("epochs")
        or payload.get("epoch") != config.get("epochs") - 1
        or payload.get("checkpoint_schema") != "wnn.sequence-fit.v1"
        or data.get("split_sha256") != auxiliary.sha256
        or data.get("primary_split_sha256") != primary.sha256
        or data.get("fit_members") != expected_fit
        or data.get("evaluation_members") != expected_control
        or data.get("evaluation_source") != "train"
        or data.get("evaluation_role") != "control"
        or data.get("evaluation_mask") != "all_after_warmup"
    ):
        raise ValueError("Source6-loaded model identity differs from its validated candidate")
    members = [("valid", group) for group in primary.evaluation_groups]
    dataset = sequence_records(
        Path(str(request["data_root"])),
        members,
        0,
        training_mask="is_scored",
        input_feature_mode=config["model"].get("input_feature_mode", "raw112"),
    )
    dataset.validate_source_schema()
    output_path = _regular_parent_path(Path(str(candidate["predictions_path"])))
    receipt_path = _regular_parent_path(Path(str(candidate["receipt_path"])))
    temporary = output_path.with_name(output_path.name + ".partial")
    if (
        output_path.exists()
        or output_path.is_symlink()
        or temporary.exists()
        or temporary.is_symlink()
    ):
        raise FileExistsError("candidate prediction output must be fresh")
    metrics, prediction_receipt = score_primary_valid(
        model, dataset, int(config["chunk_rows"]), 64, device, temporary
    )
    if metrics.get("selected_rows") != PRIMARY_ROWS:
        raise ValueError("Source6 candidate score did not cover all official scored rows")
    temporary.replace(output_path)
    result = {
        "schema": "wnn.multistage-primary-ensemble-member-score.v1",
        "status": "complete",
        "member_index": args.member_index,
        "name": candidate["name"],
        "gpu_uuid": args.gpu_uuid,
        "device_name": device_name,
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "trial_result_sha256": candidate["trial_result_sha256"],
        "trial_request_sha256": candidate["trial_request_sha256"],
        "config_sha256": candidate["config_sha256"],
        "config_identity_sha256": candidate["config_identity_sha256"],
        "prediction_manifest_sha256": candidate["prediction_manifest_sha256"],
        "inner_objective_sha256": candidate["inner_objective_sha256"],
        "source_sha256": candidate["training_source_sha256"],
        "producer_identity_sha256": candidate["producer_identity_sha256"],
        "primary_split_sha256": primary.sha256,
        "auxiliary_split_sha256": auxiliary.sha256,
        "metrics": metrics,
        "predictions": {
            "path": str(output_path.resolve()),
            "sha256": _sha256(output_path),
            "dtype": "float32",
            "shape": [PRIMARY_ROWS, 2],
            **prediction_receipt,
        },
        "scorer": {
            "source6_sha256": SOURCE6_SHA256,
            "score_script_sha256": SCORE_SCRIPT_SHA256,
            "batch_size": 64,
            "device": "cuda",
        },
    }
    _write_new(receipt_path, result)
    result["receipt_sha256"] = _sha256(receipt_path)
    return result


def _regular_parent_path(path: Path) -> Path:
    parent = path.parent.resolve(strict=True)
    if path.parent != parent or path.is_symlink():
        raise ValueError("worker output parent/path is unsafe")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("member",), required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--request-sha256")
    parser.add_argument("--member-index", type=int)
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    try:
        result = score_member(args)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f"Source6 primary ensemble worker refused: {error}") from error
    sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
