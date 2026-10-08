"""Export only the frozen H218 fold-0 evaluation groups as FP32 predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import struct
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SPLIT_PATH = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
VALID_PATH = ROOT / "datasets/valid.parquet"
RUN_NAME = "h218_block_memory_b128_l2_h4_r256_b4_c2000_fold0_seed7_4ep_fixedep3_v1_20261008"
FIT_ATTEMPT = "h218-block-memory-b128-l2-h4-r256-b4-c2000-fold0-seed7-4ep-fixedep3-v1-20261008"
EPOCH = 3
EXPECTED_TRAIN_GROUPS = 10_535
EXPECTED_FIT_VALID_GROUPS = 1_405
EXPECTED_EVALUATION_GROUPS = 468
OUTPUT_NAME = "h218_block_memory_cleanfold0_seed7_ep3_fp32_eval468_v1"
OUTPUT_PATH = ROOT / "cache" / f"pred_{OUTPUT_NAME}_valid.npy"
SIDECAR_PATH = OUTPUT_PATH.with_suffix(".json")
TEMPORARY_PATH = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.name}.partial")
TEMPORARY_SIDECAR_PATH = SIDECAR_PATH.with_name(f"{SIDECAR_PATH.name}.partial")
SOURCE_LOCK_PATH = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/attention_training_H218/H218_FOLD0_EXPORT_SOURCE_LOCK_V1_20261008.json"
)


def file_sha256(path: Path) -> str:
    """Hash a file in bounded chunks for source and artifact provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_lock(path: Path, expected_sha256: str) -> dict[str, str]:
    """Verify every runtime source against an externally frozen source lock."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("export source-lock hash differs from the frozen contract")
    lock_obj = json.loads(raw)
    if not isinstance(lock_obj, dict) or not isinstance(lock_obj.get("files"), dict):
        raise ValueError("export source lock has an invalid schema")
    file_hashes_obj = lock_obj["files"]
    file_hashes: dict[str, str] = {}
    for relative_path, expected in file_hashes_obj.items():
        if not isinstance(relative_path, str) or not isinstance(expected, str):
            raise ValueError("export source lock file entry is malformed")
        if len(expected) != 64 or any(
            character not in "0123456789abcdef" for character in expected
        ):
            raise ValueError(f"export source lock has an invalid digest for {relative_path}")
        resolved = (ROOT / relative_path).resolve()
        if not resolved.is_relative_to(ROOT):
            raise ValueError(f"export source path escapes the repository: {relative_path}")
        actual = file_sha256(resolved)
        if actual != expected:
            raise ValueError(f"frozen export source changed: {relative_path}")
        file_hashes[relative_path] = actual
    exporter_relative = str(Path(__file__).resolve().relative_to(ROOT))
    if exporter_relative not in file_hashes:
        raise ValueError("export source lock does not bind this H218 exporter")
    return file_hashes


def evaluation_group_digest(groups: list[int]) -> str:
    """Return the canonical SHA256 for sorted little-endian int64 row groups."""
    return hashlib.sha256(struct.pack(f"<{len(groups)}q", *groups)).hexdigest()


def load_fold0_evaluation_groups(
    path: Path, expected_manifest_sha256: str, expected_groups_sha256: str
) -> tuple[dict[str, object], list[int], str]:
    """Validate frozen fold-0 membership and return only its excluded valid groups.

    Raises:
        ValueError: If the manifest identity, split, groups, or group digest differs.
    """
    raw = path.read_bytes()
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError("fold-0 manifest hash differs from the frozen export contract")
    manifest_obj = json.loads(raw)
    if not isinstance(manifest_obj, dict):
        raise ValueError("fold-0 manifest must be a JSON object")
    manifest: dict[str, object] = manifest_obj
    if (
        type(manifest.get("version")) is not int
        or manifest.get("version") != 1
        or type(manifest.get("fold")) is not int
        or manifest.get("fold") != 0
    ):
        raise ValueError("export requires campaign split manifest version 1, fold 0")
    if type(manifest.get("n_folds")) is not int or manifest.get("n_folds") != 4:
        raise ValueError("fold-0 manifest has an unexpected fold count")

    fit_obj = manifest.get("fit")
    evaluation_obj = manifest.get("evaluation")
    if not isinstance(fit_obj, dict) or not isinstance(evaluation_obj, dict):
        raise ValueError("fold-0 manifest is missing fit/evaluation membership")
    fit_train_obj = fit_obj.get("train")
    fit_valid_obj = fit_obj.get("valid")
    groups_obj = evaluation_obj.get("valid")
    if not all(isinstance(value, list) for value in (fit_train_obj, fit_valid_obj, groups_obj)):
        raise ValueError("fold-0 manifest group memberships must be lists")
    fit_train = fit_train_obj
    fit_valid = fit_valid_obj
    groups = groups_obj
    for name, membership in (
        ("fit train", fit_train),
        ("fit valid", fit_valid),
        ("evaluation", groups),
    ):
        if any(type(group) is not int or group < 0 for group in membership):
            raise ValueError(f"{name} groups must be nonnegative integer row-group IDs")
        if len(membership) != len(set(membership)):
            raise ValueError(f"{name} groups contain duplicate row-group IDs")
    if groups != sorted(set(groups)) or len(groups) != EXPECTED_EVALUATION_GROUPS:
        raise ValueError("evaluation group membership is not the exact sorted 468-row fold")
    if len(fit_train) != EXPECTED_TRAIN_GROUPS or len(fit_valid) != EXPECTED_FIT_VALID_GROUPS:
        raise ValueError("fold-0 fit membership counts differ from the frozen recipe")
    if set(groups).intersection(fit_valid):
        raise ValueError("an evaluation row group appears in the fit-valid membership")
    groups_sha256 = evaluation_group_digest(groups)
    if groups_sha256 != expected_groups_sha256:
        raise ValueError("evaluation group digest differs from the frozen export contract")
    return manifest, groups, groups_sha256


def validate_valid_parquet_identity(manifest: dict[str, object], path: Path) -> None:
    """Check the valid parquet file identity recorded in the fold manifest."""
    sources_obj = manifest.get("sources")
    if not isinstance(sources_obj, dict):
        raise ValueError("fold-0 manifest has no source identities")
    identity_obj = sources_obj.get("datasets/valid.parquet")
    if not isinstance(identity_obj, dict):
        raise ValueError("fold-0 manifest has no valid parquet identity")
    stat = path.stat()
    if (
        identity_obj.get("bytes") != stat.st_size
        or identity_obj.get("mtime_ns") != stat.st_mtime_ns
    ):
        raise ValueError("valid parquet file identity differs from the frozen fold manifest")


def validate_checkpoint(
    checkpoint: object, manifest: dict[str, object], checkpoint_sha256: str
) -> dict[str, object]:
    """Reject checkpoints with an unexpected recipe or overlapping fit membership."""
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a mapping")
    args_obj = checkpoint.get("args")
    provenance_obj = checkpoint.get("provenance")
    if not isinstance(args_obj, dict) or not isinstance(provenance_obj, dict):
        raise ValueError("checkpoint is missing training arguments or provenance")
    expected_args: dict[str, object] = {
        "arch": "block-memory",
        "hidden": 128,
        "layers": 2,
        "block_heads": 4,
        "block_rows": 256,
        "batch": 4,
        "chunk": 2000,
        "seed": 7,
        "train_source": "train+valid",
        "valid_fold": 0,
        "epochs": 4,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": EPOCH,
        "no_eval_during_fit": True,
        "privileged_mode": "none",
    }
    if any(args_obj.get(name) != expected for name, expected in expected_args.items()):
        raise ValueError("checkpoint arguments differ from the frozen H218 recipe")
    if checkpoint.get("epoch") != EPOCH:
        raise ValueError("checkpoint is not the fixed zero-based epoch-3 checkpoint")
    if args_obj.get("init") not in (None, "", "none"):
        raise ValueError("H218 clean-fold export checkpoint is not a fresh scratch model")
    if (
        provenance_obj.get("split_manifest_sha256")
        != hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest()
    ):
        raise ValueError("checkpoint split provenance differs from the frozen fold-0 manifest")
    if file_sha256(ROOT / "artifacts/runs" / RUN_NAME / f"model_ep{EPOCH}.pt") != checkpoint_sha256:
        raise ValueError("checkpoint SHA256 differs from the terminal fit receipt")

    fit_rows_obj = provenance_obj.get("fit_row_groups")
    if not isinstance(fit_rows_obj, dict):
        raise ValueError("checkpoint fit-row provenance is missing")
    train_rows = fit_rows_obj.get("train")
    valid_rows = fit_rows_obj.get("valid")
    fit_obj = manifest.get("fit")
    if not isinstance(fit_obj, dict):
        raise ValueError("fold-0 manifest fit membership is missing")
    expected_train = fit_obj.get("train")
    expected_valid = fit_obj.get("valid")
    if not isinstance(train_rows, list) or not isinstance(valid_rows, list):
        raise ValueError("checkpoint fit-row membership is malformed")
    if any(type(group) is not int or group < 0 for group in train_rows + valid_rows):
        raise ValueError("checkpoint fit-row groups must be nonnegative integer IDs")
    if (
        len(train_rows) != len(set(train_rows))
        or len(valid_rows) != len(set(valid_rows))
        or set(train_rows) != set(expected_train)
        or set(valid_rows) != set(expected_valid)
    ):
        raise ValueError("checkpoint fit membership differs from the clean fold-0 recipe")
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict) or set(valid_rows).intersection(
        evaluation.get("valid", [])
    ):
        raise ValueError("checkpoint fit membership overlaps excluded evaluation groups")
    return args_obj


def configure_checkpoint_state(
    train_rnn: object, model: object, checkpoint: dict[str, object], device: object
) -> object:
    """Configure project inference state from the checkpoint, matching native prediction."""
    from wnn.propensity import load_propensity
    from wnn.sentinel import SentinelPreprocessor

    args = checkpoint["args"]
    train_rnn.SENTINEL_PREPROCESSOR = (
        SentinelPreprocessor(checkpoint["sentinel_preprocessing"])
        if checkpoint.get("sentinel_preprocessing")
        else None
    )
    train_rnn.RAW_ONLY = bool(args.get("raw_only"))
    train_rnn.ONLINE_NORM = bool(args.get("online_norm"))
    train_rnn.HEURISTICS = bool(args.get("heuristics"))
    train_rnn.RESEARCH_FEATURES = args.get("research_features", "none")
    train_rnn.PRIVILEGED_MODE = args.get("privileged_mode", "none")
    train_rnn.PRIVILEGED_HORIZON = int(args.get("privileged_horizon", 8))
    input_propensity = args.get("prop_input")
    train_rnn.INPUT_PROPENSITY = (
        load_propensity(ROOT / "artifacts/runs" / input_propensity / "model.pt", device)
        if input_propensity
        else None
    )
    if checkpoint.get("readout") is not None:
        train_rnn.READOUT = [table.to(device) for table in checkpoint["readout"]]
        train_rnn.READOUT_TILTED = bool(checkpoint.get("readout_tilted", False))
    else:
        train_rnn.READOUT = None
        train_rnn.READOUT_TILTED = False
    if args.get("arch") == "gated":
        model.gate_mode = "soft"
    return args


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-source-lock-sha256", required=True)
    parser.add_argument("--expected-split-manifest-sha256", required=True)
    parser.add_argument("--expected-evaluation-groups-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    """Validate the frozen H218 sources and export the excluded 468 groups only."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    started = time.perf_counter()
    source_hashes = validate_source_lock(SOURCE_LOCK_PATH, args.expected_source_lock_sha256)
    manifest, groups, groups_sha256 = load_fold0_evaluation_groups(
        SPLIT_PATH,
        args.expected_split_manifest_sha256,
        args.expected_evaluation_groups_sha256,
    )
    validate_valid_parquet_identity(manifest, VALID_PATH)
    checkpoint_path = ROOT / "artifacts/runs" / RUN_NAME / f"model_ep{EPOCH}.pt"
    checkpoint_sha256 = file_sha256(checkpoint_path)
    if checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise ValueError("fixed epoch-3 checkpoint SHA256 differs from the fit terminal receipt")
    if any(
        path.exists()
        for path in (OUTPUT_PATH, TEMPORARY_PATH, SIDECAR_PATH, TEMPORARY_SIDECAR_PATH)
    ):
        raise FileExistsError("H218 evaluation export destination already exists")
    if shutil.disk_usage(OUTPUT_PATH.parent).free < 1_000_000_000:
        raise OSError("insufficient free space for the bounded H218 prediction export")
    input_validation_seconds = time.perf_counter() - started

    model_setup_started = time.perf_counter()
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from scripts import predict_run, train_rnn
    from wnn.resources import configure_torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = validate_checkpoint(checkpoint, manifest, checkpoint_sha256)
    if checkpoint.get("n_in") != 211:
        raise ValueError("checkpoint feature width differs from the reviewed 211-column profile")
    state = checkpoint.get("model")
    if not isinstance(state, dict) or "head.2.weight" not in state:
        raise ValueError("checkpoint has no expected two-target model head")
    head_rows = int(state["head.2.weight"].shape[0])
    if head_rows < 2:
        raise ValueError("checkpoint head has fewer than two competition targets")
    configure_torch()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the admitted prediction export")
    model = train_rnn.make_model(checkpoint_args, checkpoint["n_in"], head_rows).to(device)
    model.load_state_dict(state)
    model._inference_fp32 = True
    if checkpoint_args.get("arch") == "gated":
        model.gate_mode = "soft"
    configure_checkpoint_state(train_rnn, model, checkpoint, device)
    dataset = predict_run.FeatureOnlySequences("valid", groups)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    model_setup_seconds = time.perf_counter() - model_setup_started
    inference_start = time.perf_counter()
    output = np.lib.format.open_memmap(
        TEMPORARY_PATH,
        mode="w+",
        dtype=np.float32,
        shape=(len(groups), 20_000, 2),
    )
    row = 0
    first_100_seconds: float | None = None
    for batch in loader:
        prediction = np.asarray(train_rnn.predict(model, batch, 4_000, device))
        if (
            prediction.ndim != 3
            or prediction.shape[:2] != (len(batch), 20_000)
            or prediction.shape[2] < 2
            or prediction.dtype != np.float32
        ):
            raise ValueError("inference output has an unexpected shape or dtype")
        two_targets = prediction[..., :2]
        if two_targets.shape != (len(batch), 20_000, 2) or not np.isfinite(two_targets).all():
            raise ValueError("inference output is nonfinite or lacks the two competition targets")
        output[row : row + len(batch)] = two_targets
        row += len(batch)
        if row == 100:
            first_100_seconds = time.perf_counter() - inference_start
            logging.info(
                json.dumps(
                    {
                        "event": "H218_EXPORT_FIRST_100_ROWS",
                        "rows": row,
                        "elapsed_seconds": first_100_seconds,
                        "rows_per_second": row / first_100_seconds,
                        "projected_export_seconds": len(groups) * first_100_seconds / row,
                    },
                    sort_keys=True,
                )
            )
        if row % 100 == 0:
            logging.info(json.dumps({"event": "H218_EXPORT_PROGRESS", "rows": row}, sort_keys=True))
    if row != len(groups):
        raise RuntimeError(f"export wrote {row} of {len(groups)} evaluation groups")
    output.flush()
    del output
    with TEMPORARY_PATH.open("rb") as stream:
        os.fsync(stream.fileno())
    TEMPORARY_PATH.replace(OUTPUT_PATH)
    prediction_sha256 = file_sha256(OUTPUT_PATH)
    inference_seconds = time.perf_counter() - inference_start
    metadata = {
        "schema": "h218-fold0-feature-only-prediction-v1",
        "run": RUN_NAME,
        "fit_attempt_id": FIT_ATTEMPT,
        "epoch": EPOCH,
        "checkpoint_sha256": checkpoint_sha256,
        "prediction_sha256": prediction_sha256,
        "prediction_path": str(OUTPUT_PATH.relative_to(ROOT)),
        "prediction_shape": [len(groups), 20_000, 2],
        "prediction_dtype": "float32",
        "prediction_bytes": OUTPUT_PATH.stat().st_size,
        "split_manifest_path": str(SPLIT_PATH.relative_to(ROOT)),
        "split_manifest_sha256": args.expected_split_manifest_sha256,
        "evaluation_group_count": len(groups),
        "evaluation_row_groups": groups,
        "evaluation_group_ids_sha256": groups_sha256,
        "labels_read": False,
        "targets_read": False,
        "inference_precision": "fp32",
        "sequence_state_reset": "batch_size=1 starts state=None for every sequence; state carries only across that sequence's chunks",
        "independent_grading_only": True,
        "official_score_computed": False,
        "source_sha256": {
            **source_hashes,
            "source_lock": args.expected_source_lock_sha256,
        },
        "timing": {
            "input_validation_seconds": input_validation_seconds,
            "model_setup_seconds": model_setup_seconds,
            "prediction_and_feature_io_seconds": inference_seconds,
            "first_100_rows_seconds": first_100_seconds,
            "prediction_rows_per_second": len(groups) / inference_seconds,
        },
    }
    with TEMPORARY_SIDECAR_PATH.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    TEMPORARY_SIDECAR_PATH.replace(SIDECAR_PATH)
    logging.info(json.dumps({"event": "H218_EXPORT_TERMINAL", **metadata}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
