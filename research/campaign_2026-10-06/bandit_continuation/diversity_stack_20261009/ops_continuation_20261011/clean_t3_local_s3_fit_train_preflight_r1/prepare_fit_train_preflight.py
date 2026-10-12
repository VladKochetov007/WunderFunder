"""Validate the frozen s3 fit-train export selector before CUDA, without export."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import runpy
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

import numpy as np

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
OPS = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011"
)
R3 = OPS / "clean_t3_fold0_h390_r3_teacher_cohort"
R5 = OPS / "clean_t3_fold0_h390_r5_teacher_cohort"
RUN = "clean_t3_f0_muonall_h384_e3_s3_r5"
OUTPUT_NAME = "clean_t3_f0_muonall_h384_e3_s3_r5_local_fit_train_r1"
PREDICTOR = R3 / "source_snapshot/scripts/predict_run.py"
TRAINER = R3 / "source_snapshot/scripts/train_rnn.py"
SPLIT = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
TRAIN_PARQUET = ROOT / "datasets/train.parquet"
TRAIN_TARGETS = ROOT / "cache/train_targets.npz"
CHECKPOINT = ROOT / "artifacts/runs" / RUN / "model_ep2.pt"
PROVENANCE = ROOT / "artifacts/runs" / RUN / "provenance.json"
INPUT_BINDING = R3 / "INPUT_BINDING_R3.json"
SOURCE_BINDING = R3 / "SOURCE_BINDING_R3.json"
R5_SPEC = (
    R5
    / "terminal_collection_r1/provider_output/attempt_001/outputs/experiment/kaggle_specs/export_clean_t3_f0_muonall_h384_e3_s3_r5_fit_train.json"
)
ACTUAL_CP_REVIEW = OPS / "clean_t3_local_s3_export_r1/ROOT_ACTUAL_CONSUMER_REVIEW_R1.json"
S3_TERMINAL = OPS / "clean_t3_local_s3_terminal_r1/LOCAL_TERMINAL_HANDOFF_R1.json"
OUT = ROOT / "cache" / f"pred_{OUTPUT_NAME}_train.npy"
OUT_JSON = OUT.with_suffix(".json")
OUT_PARTIAL = OUT.with_name(OUT.name + ".partial")

EXPECTED = {
    "checkpoint": "201221c6b4140dda70b078db2a9d7a7246ec40b813f054570d195cecefc61582",
    "provenance": "740b0d96ee311f215153db62272262e0153ce11f8f91560b03efae8e739f9ed0",
    "source_manifest": "41a5452c752bb10bee28af2eb2e06e7a224da8d72f307b4d5bec6fb5b3b236fa",
    "predictor": "30647bb74da64ddab4ccaa5e2916338bbb4826466b47cd68ec61593269554dd0",
    "trainer": "a2b5440ff4b8e5179db01cc047102ef634522327374fced31ca5fdf8afcd17a6",
    "split": "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d",
    "input_binding": "2d74435d33df9c24f5c7d5656ee82ff2bbbe0ede095a8c6d6f82f8aa9234ef48",
    "source_binding": "1895d15df875763921f9bf0ccfbf671a07029261fe7ffaad125275e23708d2e4",
    "r5_spec": "6ee3ba26cd3c02dac88fd9512a47e030d14c37e0dc729b2090d8f123aca906ea",
    "cp_review": "5f5ea45496db354ab47839250528930d525e4734059fe25b7e61029af637d8d9",
    "s3_terminal": "0eabad07a2bea44f0d4691575ccc7105526c39049c56dbf32ba948d7efb717c8",
    "train_parquet": "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
    "train_targets": "ead1a05e7aa5ac3d1c10cde65986be9f48fd342cadb95330611e58c3a8f817e9",
}
EXPECTED_BYTES = {"train_parquet": 29_240_219_011, "train_targets": 1_909_335_790}
EXPECTED_MTIME_NS = 1_577_826_000_000_000_000
EXPECTED_TRAIN_FOOTER_BYTES = 60_437_285
EXPECTED_TRAIN_FOOTER_SHA256 = "071a60cc9e72a95b013fe6784318b92a6fb82b7c95b6371d1f21a0e5d04e3bfb"
R5_SPEC_SHA256 = "6ee3ba26cd3c02dac88fd9512a47e030d14c37e0dc729b2090d8f123aca906ea"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tail(path: Path, footer_bytes: int) -> str:
    digest = hashlib.sha256()
    remaining = footer_bytes + 8  # Parquet footer length plus trailing PAR1.
    with path.open("rb") as stream:
        stream.seek(-remaining, os.SEEK_END)
        while remaining:
            block = stream.read(min(1 << 20, remaining))
            if not block:
                raise EOFError("train.parquet footer range was truncated")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


class BeforeCuda(RuntimeError):
    """Sentinel raised at the real predictor's CUDA configuration boundary."""


BOUNDARY_CAPTURE: dict[str, object] = {}


def fail_before_cuda(*args: object, **kwargs: object) -> NoReturn:
    del args, kwargs
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        raise RuntimeError("could not inspect actual predictor boundary")
    caller = frame.f_back
    local = caller.f_locals
    parsed = local["args"]
    checkpoint = local["ck"]
    campaign_split = local["campaign_split"]
    groups = np.asarray(local["groups"], dtype=np.int64)
    BOUNDARY_CAPTURE.clear()
    BOUNDARY_CAPTURE.update(
        {
            "args": {
                "name": parsed.name,
                "split": parsed.split,
                "epoch": parsed.epoch,
                "output_name": parsed.output_name,
                "campaign_split_manifest": str(parsed.campaign_split_manifest),
                "campaign_evaluation_only": parsed.campaign_evaluation_only,
                "campaign_full_valid_feature_only": parsed.campaign_full_valid_feature_only,
                "fit_subset_manifest": str(parsed.fit_subset_manifest)
                if parsed.fit_subset_manifest
                else None,
                "device": parsed.device,
                "inference_precision": parsed.inference_precision,
            },
            "checkpoint": {
                "epoch": checkpoint.get("epoch"),
                "name": checkpoint["args"].get("name"),
                "seed": checkpoint["args"].get("seed"),
                "valid_fold": checkpoint["args"].get("valid_fold"),
                "muon_scope": checkpoint["args"].get("muon_scope"),
                "propensity": checkpoint["args"].get("propensity"),
                "train_source": checkpoint["args"].get("train_source"),
                "include_heldout_valid": checkpoint["args"].get("include_heldout_valid"),
                "split_manifest_sha256": checkpoint["args"].get("split_manifest_sha256"),
            },
            "campaign_split_sha256": campaign_split.sha256,
            "campaign_fit_train_count": len(campaign_split.train_groups),
            "campaign_fit_valid_count": len(campaign_split.valid_groups),
            "campaign_evaluation_valid_count": len(campaign_split.evaluation_groups),
            "groups": groups.tolist(),
            "feature_only": bool(local["feature_only"]),
            "cuda_initialized_at_boundary": bool(__import__("torch").cuda.is_initialized()),
        }
    )
    raise BeforeCuda("intercepted actual predictor immediately before configure_torch")


def main() -> None:
    root = ROOT.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    expected_files = {
        "checkpoint": CHECKPOINT,
        "provenance": PROVENANCE,
        "predictor": PREDICTOR,
        "trainer": TRAINER,
        "split": SPLIT,
        "input_binding": INPUT_BINDING,
        "source_binding": SOURCE_BINDING,
        "r5_spec": R5_SPEC,
        "cp_review": ACTUAL_CP_REVIEW,
        "s3_terminal": S3_TERMINAL,
        "source_manifest": R3 / "source_snapshot/SOURCE_MANIFEST_H390.json",
    }
    observed = {
        key: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for key, path in expected_files.items()
    }
    for key, row in observed.items():
        expected = EXPECTED.get(key)
        if expected is not None and row["sha256"] != expected:
            raise ValueError(f"frozen input SHA changed for {key}")

    input_binding = json.loads(INPUT_BINDING.read_text())
    train_input = next(
        (
            row
            for row in input_binding["external_inputs"]
            if row.get("path") == "datasets/train.parquet"
        ),
        None,
    )
    if (
        train_input is None
        or train_input.get("sha256") != EXPECTED["train_parquet"]
        or train_input.get("size") != EXPECTED_BYTES["train_parquet"]
    ):
        raise ValueError("R3 immutable input binding does not identify the expected train.parquet")
    train_targets_input = next(
        (
            row
            for row in input_binding["external_inputs"]
            if row.get("path") == "cache/train_targets.npz"
        ),
        None,
    )
    if (
        train_targets_input is None
        or train_targets_input.get("sha256") != EXPECTED["train_targets"]
        or train_targets_input.get("size") != EXPECTED_BYTES["train_targets"]
    ):
        raise ValueError("R3 immutable input binding does not identify the train-target archive")

    cp_review = json.loads(ACTUAL_CP_REVIEW.read_text())
    if (
        cp_review.get("status")
        != "PASS_ACTUAL_FROZEN_PREDICTOR_CP_SPLIT_AND_FULL_VALID_SELECTOR_BEFORE_CUDA"
        or cp_review.get("checkpoint_sha256") != EXPECTED["checkpoint"]
        or cp_review.get("predictor_sha256") != EXPECTED["predictor"]
        or cp_review.get("trainer_sha256") != EXPECTED["trainer"]
        or cp_review.get("cuda_initialized") is not False
    ):
        raise ValueError("root actual CP2 consumer review does not bind this frozen endpoint")
    s3_terminal = json.loads(S3_TERMINAL.read_text())
    if (
        s3_terminal.get("status") != "terminal_export_validated_grade_pending_root"
        or s3_terminal.get("source_and_model", {}).get("checkpoint_sha256")
        != EXPECTED["checkpoint"]
    ):
        raise ValueError("s3 CP2 terminal evidence does not bind this checkpoint")

    split = json.loads(SPLIT.read_text())
    train_targets_stat = TRAIN_TARGETS.stat()
    train_targets_split_source = split["sources"]["cache/train_targets.npz"]
    if (
        train_targets_stat.st_size != EXPECTED_BYTES["train_targets"]
        or train_targets_stat.st_size != train_targets_split_source.get("bytes")
        or train_targets_stat.st_mtime_ns != train_targets_split_source.get("mtime_ns")
    ):
        raise ValueError("local train_targets size/mtime differs from the frozen split source")
    observed["train_targets"] = {
        "path": str(TRAIN_TARGETS),
        "bytes": train_targets_stat.st_size,
        "mtime_ns": train_targets_stat.st_mtime_ns,
        "expected_full_sha256_from_R3_input_binding_and_R5_spec": EXPECTED["train_targets"],
        "full_content_rehashed_locally": False,
        "runtime_consumer_index_sha256": train_targets_split_source.get("index_sha256"),
    }
    split_source = split["sources"]["datasets/train.parquet"]
    train_stat = TRAIN_PARQUET.stat()
    if (
        train_stat.st_size != EXPECTED_BYTES["train_parquet"]
        or train_stat.st_mtime_ns != EXPECTED_MTIME_NS
    ):
        raise ValueError("local train.parquet size/mtime differs from the frozen split source")
    if (
        split_source.get("bytes") != train_stat.st_size
        or split_source.get("mtime_ns") != train_stat.st_mtime_ns
    ):
        raise ValueError("fold split train source metadata differs from the actual file")
    if (
        split_source.get("footer_bytes") != EXPECTED_TRAIN_FOOTER_BYTES
        or split_source.get("footer_sha256") != EXPECTED_TRAIN_FOOTER_SHA256
    ):
        raise ValueError("fold split train footer identity differs")
    footer_sha = sha256_tail(TRAIN_PARQUET, EXPECTED_TRAIN_FOOTER_BYTES)
    if footer_sha != EXPECTED_TRAIN_FOOTER_SHA256:
        raise ValueError("local train.parquet trailing footer+length+PAR1 identity differs")

    spec = json.loads(R5_SPEC.read_text())
    if sha256_file(R5_SPEC) != R5_SPEC_SHA256:
        raise ValueError("historical R5 fit-train export spec changed")
    expected_r5_tail = [
        "scripts/predict_run.py",
        RUN,
        "train",
        "--epoch",
        "2",
        "--output-name",
        f"{RUN}_fit_train",
        "--campaign-split-manifest",
        "research/campaign_2026-09-29/artifacts/splits/fold0.json",
        "--device",
        "cuda",
        "--inference-precision",
        "fp32",
    ]
    if (
        spec.get("name") != f"export_{RUN}_fit_train"
        or spec.get("argv", [])[2:] != expected_r5_tail
    ):
        raise ValueError("R5 train export recipe differs from the actual frozen predictor CLI")
    if spec.get("artifacts") != [
        f"cache/pred_{RUN}_fit_train_train.npy",
        f"cache/pred_{RUN}_fit_train_train.json",
    ]:
        raise ValueError("R5 expected output artifact names differ")
    spec_inputs = {row["path"]: row for row in spec.get("inputs", [])}
    if (
        spec_inputs.get("datasets/train.parquet", {}).get("sha256") != EXPECTED["train_parquet"]
        or spec_inputs.get("datasets/train.parquet", {}).get("size")
        != EXPECTED_BYTES["train_parquet"]
    ):
        raise ValueError("R5 train spec does not bind the same raw train parquet")
    if (
        spec_inputs.get("cache/train_targets.npz", {}).get("sha256") != EXPECTED["train_targets"]
        or spec_inputs.get("cache/train_targets.npz", {}).get("size")
        != EXPECTED_BYTES["train_targets"]
    ):
        raise ValueError("R5 train spec does not bind the same train-target index archive")

    split_sha = sha256_file(SPLIT)
    if split_sha != EXPECTED["split"] or split.get("fold") != 0:
        raise ValueError("fold-0 source-qualified split differs")
    expected_groups = list(map(int, split["fit"]["train"]))
    if len(expected_groups) != 10_535 or len(set(expected_groups)) != 10_535:
        raise ValueError("frozen train selector does not contain 10,535 unique fit groups")
    component_by_source_group = {
        (source, int(row["row_group"])): int(row["component"])
        for source, rows in split["sequences"].items()
        for row in rows
    }
    fit_components = {component_by_source_group[("train", group)] for group in expected_groups}
    fit_components.update(
        component_by_source_group[("valid", int(group))] for group in split["fit"]["valid"]
    )
    evaluation_components = {
        component_by_source_group[("valid", int(group))] for group in split["evaluation"]["valid"]
    }
    if fit_components & evaluation_components:
        raise ValueError("source-qualified fit/evaluation components overlap")
    group_digest = hashlib.sha256(np.asarray(expected_groups, dtype="<i8").tobytes()).hexdigest()

    output_paths = (OUT, OUT_JSON, OUT_PARTIAL)
    collisions = [str(path) for path in output_paths if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing existing fit-train output paths: {collisions}")

    os.environ["WNN_PROJECT_ROOT"] = str(root)
    os.environ["WNN_LOADER_WORKERS"] = "0"
    os.environ.pop("WNN_PORTABLE_DATA_MANIFEST", None)
    sys.path.insert(0, str(PREDICTOR.parent))
    sys.path.insert(0, str(R3 / "source_snapshot/src"))
    import torch

    import wnn.resources as resources

    cuda_seen = {"initialized_before": bool(torch.cuda.is_initialized())}
    resources.configure_torch = fail_before_cuda
    argv = [
        str(PREDICTOR),
        RUN,
        "train",
        "--epoch",
        "2",
        "--output-name",
        OUTPUT_NAME,
        "--campaign-split-manifest",
        "research/campaign_2026-09-29/artifacts/splits/fold0.json",
        "--device",
        "cuda",
        "--inference-precision",
        "fp32",
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        namespace = runpy.run_path(str(PREDICTOR), run_name="wnn_s3_fit_train_preflight")
        try:
            namespace["main"]()
        except BeforeCuda:
            pass
        else:
            raise RuntimeError("actual predictor did not reach the intercepted CUDA boundary")
    finally:
        sys.argv = old_argv

    trainer_module = namespace["train_rnn"]
    if Path(namespace["__file__"]).resolve() != PREDICTOR.resolve():
        raise ValueError("actual runpy consumer path differs from the frozen predictor")
    if Path(trainer_module.__file__).resolve() != TRAINER.resolve():
        raise ValueError("predictor imported a different trainer implementation")
    if Path(namespace["ROOT"]).resolve() != root or Path(trainer_module.ROOT).resolve() != root:
        raise ValueError("predictor/trainer ROOT does not resolve to MAIN project data")
    if cuda_seen["initialized_before"] or torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the pre-export boundary")
    capture = BOUNDARY_CAPTURE
    if not capture:
        raise RuntimeError("actual predictor did not expose its parsed selector state")
    parsed = capture["args"]
    checkpoint = capture["checkpoint"]
    observed_groups = capture["groups"]
    if parsed != {
        "name": RUN,
        "split": "train",
        "epoch": 2,
        "output_name": OUTPUT_NAME,
        "campaign_split_manifest": "research/campaign_2026-09-29/artifacts/splits/fold0.json",
        "campaign_evaluation_only": False,
        "campaign_full_valid_feature_only": False,
        "fit_subset_manifest": None,
        "device": "cuda",
        "inference_precision": "fp32",
    }:
        raise ValueError("actual predictor parsed arguments differ from the prepared train export")
    expected_checkpoint = {
        "epoch": 2,
        "name": RUN,
        "seed": 3,
        "valid_fold": 0,
        "muon_scope": "all",
        "propensity": "c20260929_prop_f0",
        "train_source": "train+valid",
        "include_heldout_valid": False,
        "split_manifest_sha256": EXPECTED["split"],
    }
    if checkpoint != expected_checkpoint:
        mismatched_checkpoint_fields = {
            key: {"expected": value, "observed": checkpoint.get(key)}
            for key, value in expected_checkpoint.items()
            if checkpoint.get(key) != value
        }
        raise ValueError(
            "actual CP2 recipe/provenance differs from the source-qualified teacher: "
            f"{mismatched_checkpoint_fields}"
        )
    if capture["campaign_split_sha256"] != EXPECTED["split"]:
        raise ValueError("actual predictor loaded a different fold-0 selector")
    if (
        capture["campaign_fit_train_count"] != 10_535
        or capture["campaign_fit_valid_count"] != 1_405
        or capture["campaign_evaluation_valid_count"] != 468
        or capture["feature_only"] is not True
        or capture["cuda_initialized_at_boundary"] is not False
        or observed_groups != expected_groups
    ):
        raise ValueError("actual predictor selected the wrong fit-train rows or read mode")
    dataset = namespace["FeatureOnlySequences"]("train", observed_groups)
    if dataset.path.resolve() != TRAIN_PARQUET.resolve() or dataset.groups != expected_groups:
        raise ValueError("actual feature-only dataset does not point at MAIN train.parquet/groups")
    if any(path.exists() for path in output_paths):
        raise FileExistsError("preflight unexpectedly created a prediction output")

    array_shape = [10_535, 20_000, 2]
    payload_bytes = int(np.prod(array_shape)) * np.dtype(np.float32).itemsize
    expected_npy_bytes = payload_bytes + 128
    if payload_bytes != 1_685_600_000 or expected_npy_bytes != 1_685_600_128:
        raise AssertionError("fit-train NPY size arithmetic changed")
    argv = [
        str(ROOT / ".venv/bin/python"),
        str(PREDICTOR),
        RUN,
        "train",
        "--epoch",
        "2",
        "--output-name",
        OUTPUT_NAME,
        "--campaign-split-manifest",
        "research/campaign_2026-09-29/artifacts/splits/fold0.json",
        "--device",
        "cuda",
        "--inference-precision",
        "fp32",
    ]
    receipt = {
        "schema": "wnn.local-clean-t3-f0-s3-fit-train-export-preflight.r1",
        "status": "PASS_ACTUAL_FIT_TRAIN_SELECTOR_BEFORE_CUDA; PREPARED_NOT_LAUNCHED",
        "created_utc": datetime.now(UTC).isoformat(),
        "scope": "Read-only source/input/CP2/selector consumer preflight; no prediction rows exported, no CUDA initialized, no model fit or provider action.",
        "run": RUN,
        "seed": 3,
        "checkpoint_epoch": 2,
        "actual_checkpoint_sha256": sha256_file(CHECKPOINT),
        "actual_provenance_sha256": sha256_file(PROVENANCE),
        "root_actual_cp_and_predictor_review_sha256": sha256_file(ACTUAL_CP_REVIEW),
        "s3_terminal_handoff_sha256": sha256_file(S3_TERMINAL),
        "source_bindings": {
            "source_manifest_sha256": EXPECTED["source_manifest"],
            "source_binding_sha256": observed["source_binding"]["sha256"],
            "input_binding_sha256": observed["input_binding"]["sha256"],
            "predictor_sha256": observed["predictor"]["sha256"],
            "trainer_sha256": observed["trainer"]["sha256"],
            "r5_fit_train_spec_sha256": observed["r5_spec"]["sha256"],
        },
        "raw_train_source_identity": {
            "path": str(TRAIN_PARQUET),
            "expected_full_content_sha256_from_immutable_R3_input_binding_and_R5_spec": EXPECTED[
                "train_parquet"
            ],
            "full_content_rehashed_locally": False,
            "verified_size_bytes": train_stat.st_size,
            "verified_mtime_ns": train_stat.st_mtime_ns,
            "verified_split_footer_bytes": EXPECTED_TRAIN_FOOTER_BYTES,
            "verified_footer_plus_length_plus_PAR1_sha256": footer_sha,
            "local_identity_scope": "matches source-qualified split stat/footer identity and bound producer full-content SHA; no redundant 29.24GB full-file hash",
        },
        "actual_consumer_boundary": {
            "predictor_file": str(PREDICTOR),
            "predictor_sha256": observed["predictor"]["sha256"],
            "imported_trainer_file": str(Path(trainer_module.__file__).resolve()),
            "trainer_sha256": observed["trainer"]["sha256"],
            "predictor_root": str(Path(namespace["ROOT"]).resolve()),
            "trainer_root": str(Path(trainer_module.ROOT).resolve()),
            "actual_parsed_cli": parsed,
            "actual_cp_metadata_before_cuda": checkpoint,
            "actual_campaign_split_sha256": capture["campaign_split_sha256"],
            "actual_fit_train_count": capture["campaign_fit_train_count"],
            "actual_fit_valid_count": capture["campaign_fit_valid_count"],
            "actual_evaluation_valid_count": capture["campaign_evaluation_valid_count"],
            "selected_fit_train_groups_ordered_int64_le_sha256": group_digest,
            "selected_groups_equal_frozen_split_fit_train_list": observed_groups == expected_groups,
            "source_qualified_fit_evaluation_component_overlap": len(
                fit_components & evaluation_components
            ),
            "feature_only": capture["feature_only"],
            "actual_dataset_path": str(dataset.path.resolve()),
            "cuda_initialized_before_and_at_boundary": False,
        },
        "planned_export": {
            "status": "pending root live-resource/profile admission",
            "argv": argv,
            "environment": {
                "WNN_PROJECT_ROOT": str(ROOT),
                "PYTHONPATH": str(R3 / "source_snapshot/src"),
                "WNN_LOADER_WORKERS": "0",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
            "output_npy": str(OUT),
            "output_sidecar": str(OUT_JSON),
            "partial_output": str(OUT_PARTIAL),
            "expected_shape": array_shape,
            "expected_dtype": "float32",
            "expected_payload_bytes": payload_bytes,
            "expected_npy_bytes_including_header": expected_npy_bytes,
            "output_paths_absent": True,
            "source_selected_group_count": len(observed_groups),
            "metric_scope": "in-sample fit-train feature predictions; not a validation/test score",
        },
        "remaining_gates": [
            "fresh local aggregate CPU/RAM/GPU admission",
            "actual local-device fit-train export resource/profile gate; the earlier 1,873-row valid export does not qualify this 10,535-row shape",
            "root-controlled export admission and checkpoint/output collision recheck",
        ],
    }
    output = Path(__file__).resolve().parent / "LOCAL_S3_FIT_TRAIN_PREFLIGHT_R1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(output),
                "selected_groups": len(observed_groups),
                "cuda_initialized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
