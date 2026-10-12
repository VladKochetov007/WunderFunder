"""Run the frozen s3 predictor directly or profile its first complete FIT_TRAIN batch.

The profile mode adapts only the imported campaign selector in this isolated runner.
It keeps the frozen predictor, checkpoint, feature reader, batch size, recurrent
chunking, precision, and output writer intact while limiting membership to one
normal batch of selected FIT_TRAIN groups.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a bound file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Load one JSON object and reject other top-level shapes."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return cast(dict[str, Any], value)


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically persist an observation receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def digest_group_ids(groups: list[int], numpy: ModuleType) -> str:
    """Hash source-qualified group IDs in the frozen little-endian encoding."""
    values = numpy.asarray(groups, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def load_pinned_sources(
    source_root: Path, source_manifest_path: Path, expected_manifest_sha256: str
) -> dict[str, str]:
    """Verify every file named by the immutable producer source manifest."""
    if sha256_file(source_manifest_path) != expected_manifest_sha256:
        raise ValueError("frozen source manifest hash changed")
    manifest = read_json(source_manifest_path)
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("frozen source manifest has no source hash map")
    result: dict[str, str] = {}
    for relative, expected in sources.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("frozen source manifest contains a malformed entry")
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"pinned source is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"pinned source hash changed: {relative}")
        result[relative] = actual
    return result


def load_module(path: Path, module_name: str) -> ModuleType:
    """Import an immutable predictor from its bound source snapshot."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen predictor at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    """Parse the exact frozen bindings for a profile or a later full export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main() -> int:
    """Run one contract-bound profile or the frozen full export."""
    args = build_parser().parse_args()
    started = time.monotonic()
    contract_path = args.contract.resolve(strict=True)
    contract = read_json(contract_path)
    bindings = contract["bindings"]
    paths = contract["paths"]
    output_spec = contract["output"]
    mode = contract["mode"]
    if mode not in {"profile_first_batch", "full_export"}:
        raise ValueError(f"unsupported profile mode: {mode}")
    if args.receipt.exists():
        raise FileExistsError(f"refusing to overwrite profile receipt: {args.receipt}")
    if args.preflight.resolve(strict=True) != Path(paths["preflight"]).resolve():
        raise ValueError("preflight path differs from the frozen contract")
    if args.predictor.resolve(strict=True) != Path(paths["predictor"]).resolve():
        raise ValueError("predictor path differs from the frozen contract")
    if args.trainer.resolve(strict=True) != Path(paths["trainer"]).resolve():
        raise ValueError("trainer path differs from the frozen contract")
    if args.source_manifest.resolve(strict=True) != Path(paths["source_manifest"]).resolve():
        raise ValueError("source manifest path differs from the frozen contract")
    project_root = args.project_root.resolve(strict=True)
    if project_root != Path(paths["project_root"]).resolve():
        raise ValueError("project root differs from the frozen contract")
    if os.environ.get("WNN_LOADER_WORKERS") != "0":
        raise RuntimeError("the profile requires zero data-loader workers")
    if os.environ.get("WNN_GPU_MEMORY_BYTES") != "4300000000":
        raise RuntimeError("the profile requires the 4.3 GB CUDA allocator cap")
    if os.environ.get("WNN_THREADS") != "1":
        raise RuntimeError("the profile requires a one-CPU worker")
    if os.environ.get("WNN_MEM_MAX") != "5500000000":
        raise RuntimeError("the profile requires the 5.5 GB process RAM cap")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("the profile requires the admitted local GPU")

    preflight_path = args.preflight.resolve()
    preflight = read_json(preflight_path)
    if sha256_file(preflight_path) != bindings["preflight_sha256"]:
        raise ValueError("selector preflight hash changed")
    if preflight.get("status") != (
        "PASS_ACTUAL_FIT_TRAIN_SELECTOR_BEFORE_CUDA; PREPARED_NOT_LAUNCHED"
    ):
        raise ValueError("actual FIT_TRAIN selector preflight did not pass")
    boundary = preflight["actual_consumer_boundary"]
    source_identity = preflight["raw_train_source_identity"]
    if boundary["actual_fit_train_count"] != contract["selector"]["full_count"]:
        raise ValueError("FIT_TRAIN group count differs from the preflight")
    if (
        boundary["selected_fit_train_groups_ordered_int64_le_sha256"]
        != contract["selector"]["full_membership_sha256"]
    ):
        raise ValueError("FIT_TRAIN group membership differs from the preflight")
    if boundary["actual_campaign_split_sha256"] != bindings["campaign_split_sha256"]:
        raise ValueError("campaign split differs from the frozen checkpoint")
    checkpoint_path = Path(paths["checkpoint"]).resolve(strict=True)
    if sha256_file(checkpoint_path) != bindings["checkpoint_sha256"]:
        raise ValueError("CP2 checkpoint hash changed")
    if sha256_file(args.predictor.resolve()) != bindings["predictor_sha256"]:
        raise ValueError("frozen predictor hash changed")
    if sha256_file(args.trainer.resolve()) != bindings["trainer_sha256"]:
        raise ValueError("frozen trainer hash changed")
    if (
        source_identity["expected_full_content_sha256_from_immutable_R3_input_binding_and_R5_spec"]
        != (bindings["train_content_sha256"])
    ):
        raise ValueError("train content binding differs from the frozen contract")
    dataset_path = Path(source_identity["path"]).resolve(strict=True)
    dataset_stat = dataset_path.stat()
    if dataset_stat.st_size != source_identity["verified_size_bytes"]:
        raise ValueError("train parquet size differs from the selector preflight")
    if dataset_stat.st_mtime_ns != source_identity["verified_mtime_ns"]:
        raise ValueError("train parquet mtime differs from the selector preflight")
    if dataset_stat.st_size != contract["input"]["size_bytes"]:
        raise ValueError("train parquet size differs from the frozen contract")
    if dataset_stat.st_mtime_ns != contract["input"]["mtime_ns"]:
        raise ValueError("train parquet mtime differs from the frozen contract")

    source_root = args.predictor.resolve().parents[1]
    pinned_sources = load_pinned_sources(
        source_root,
        args.source_manifest.resolve(),
        bindings["source_manifest_sha256"],
    )
    if pinned_sources.get("scripts/predict_run.py") != bindings["predictor_sha256"]:
        raise ValueError("source manifest does not bind the selected predictor")
    if pinned_sources.get("scripts/train_rnn.py") != bindings["trainer_sha256"]:
        raise ValueError("source manifest does not bind the selected trainer")
    if contract["selector"]["source_qualified_overlap_count"] != 0:
        raise ValueError("FIT_TRAIN selector overlaps evaluated ancestry")
    if contract["selector"]["feature_only"] is not True:
        raise ValueError("FIT_TRAIN profile must use feature-only decoding")

    output_name = str(output_spec["output_name"])
    output_path = project_root / "cache" / f"pred_{output_name}_train.npy"
    sidecar_path = output_path.with_suffix(".json")
    partial_path = output_path.with_name(output_path.name + ".partial")
    if any(path.exists() for path in (output_path, sidecar_path, partial_path)):
        raise FileExistsError("profile output path already exists")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"cache directory is missing: {output_path.parent}")
    free_bytes = os.statvfs(output_path.parent).f_bavail * os.statvfs(output_path.parent).f_frsize
    if free_bytes < int(output_spec["minimum_free_bytes"]):
        raise OSError(f"insufficient free disk for this profile: {free_bytes} bytes")

    snapshot_scripts = source_root / "scripts"
    snapshot_src = source_root / "src"
    sys.path[:0] = [str(snapshot_scripts), str(snapshot_src)]
    os.environ["WNN_PROJECT_ROOT"] = str(project_root)
    import numpy as np

    import wnn.splits as splits_api

    expected_split_path = (project_root / contract["paths"]["split_manifest"]).resolve()
    if sha256_file(expected_split_path) != bindings["campaign_split_manifest_file_sha256"]:
        raise ValueError("campaign split file hash changed")
    original_split_loader = splits_api.load_campaign_split
    observations: dict[str, Any] = {
        "schema": "wnn.clean-t3-fit-train-profile-observation.r1",
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "mode": mode,
        "status": "running",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_sha256": bindings["checkpoint_sha256"],
        "predictor_sha256": bindings["predictor_sha256"],
        "trainer_sha256": bindings["trainer_sha256"],
        "preflight_sha256": bindings["preflight_sha256"],
        "source_manifest_sha256": bindings["source_manifest_sha256"],
        "dataset_path": str(dataset_path),
        "dataset_size_bytes": dataset_stat.st_size,
        "dataset_mtime_ns": dataset_stat.st_mtime_ns,
        "dataset_identity_scope": (
            "frozen R3/R5 content binding and selector preflight; "
            "point-of-use size/mtime checked; full 29.24 GB rehash omitted"
        ),
        "verified_footer_plus_length_plus_PAR1_sha256": (
            source_identity["verified_footer_plus_length_plus_PAR1_sha256"]
        ),
        "campaign_split_sha256": bindings["campaign_split_sha256"],
        "output_path": str(output_path),
        "output_expected_shape": output_spec["shape"],
        "output_dtype": output_spec["dtype"],
        "loader_workers": 0,
        "batch_size": 16,
        "recurrent_chunk_length": 4000,
        "inference_precision": "fp32",
        "metrics": {},
    }

    def profiled_split_loader(path: Path, root: Path) -> Any:
        """Verify full membership, then bound profile mode to one real batch."""
        selection_started = time.monotonic()
        split = original_split_loader(path, root)
        full_groups = [int(group) for group in split.train_groups]
        full_digest = digest_group_ids(full_groups, np)
        if split.sha256 != bindings["campaign_split_sha256"]:
            raise ValueError("runtime selector split hash differs from preflight")
        if len(full_groups) != contract["selector"]["full_count"]:
            raise ValueError("runtime FIT_TRAIN group count differs from preflight")
        if full_digest != contract["selector"]["full_membership_sha256"]:
            raise ValueError("runtime FIT_TRAIN group membership differs from preflight")
        if mode == "profile_first_batch":
            limit = int(contract["selector"]["profile_group_limit"])
            if limit != 16 or len(full_groups) < limit:
                raise ValueError("the bounded profile must use exactly one 16-group batch")
            selected_groups = full_groups[:limit]
        else:
            selected_groups = full_groups
        observations.update(
            full_selected_group_count=len(full_groups),
            full_selected_group_sha256=full_digest,
            selected_group_count=len(selected_groups),
            selected_group_sha256=digest_group_ids(selected_groups, np),
            selected_group_ids=selected_groups if mode == "profile_first_batch" else None,
            selector_seconds=time.monotonic() - selection_started,
        )
        write_json(args.receipt, observations)
        if mode == "profile_first_batch":
            # This adapter limits only membership. All frozen decode and inference
            # functions still process one regular DataLoader batch unchanged.
            return SimpleNamespace(
                sha256=split.sha256,
                train_groups=np.asarray(selected_groups, dtype=np.int64),
                valid_groups=split.valid_groups,
                evaluation_groups=split.evaluation_groups,
            )
        return split

    splits_api.load_campaign_split = profiled_split_loader
    predictor = load_module(
        args.predictor.resolve(),
        "_clean_t3_frozen_predict_run_profiled",
    )
    if Path(predictor.train_rnn.__file__).resolve() != args.trainer.resolve():
        raise ValueError("imported trainer path is outside the frozen source snapshot")
    if Path(predictor.ROOT).resolve() != project_root:
        raise ValueError("predictor selected an unexpected project root")

    metrics: dict[str, Any] = observations["metrics"]
    original_torch_load = predictor.torch.load

    def timed_torch_load(*load_args: Any, **load_kwargs: Any) -> Any:
        """Measure the immutable CP2 checkpoint load."""
        load_started = time.monotonic()
        try:
            return original_torch_load(*load_args, **load_kwargs)
        finally:
            metrics["checkpoint_load_seconds"] = (
                metrics.get("checkpoint_load_seconds", 0.0) + time.monotonic() - load_started
            )

    predictor.torch.load = timed_torch_load
    original_configure_torch = predictor.configure_torch

    def timed_configure_torch(*configure_args: Any, **configure_kwargs: Any) -> Any:
        """Measure the original CUDA setup call."""
        configure_started = time.monotonic()
        try:
            return original_configure_torch(*configure_args, **configure_kwargs)
        finally:
            metrics["cuda_setup_seconds"] = time.monotonic() - configure_started

    predictor.configure_torch = timed_configure_torch
    original_load_model = predictor.load_model

    def timed_load_model(*model_args: Any, **model_kwargs: Any) -> Any:
        """Measure model reconstruction and device transfer."""
        model_started = time.monotonic()
        try:
            return original_load_model(*model_args, **model_kwargs)
        finally:
            metrics["model_build_and_transfer_seconds"] = time.monotonic() - model_started

    predictor.load_model = timed_load_model
    original_predict = predictor.predict

    def timed_predict(model: Any, features: Any, chunk: int, device: Any) -> Any:
        """Record actual input/chunk shape and frozen recurrent prediction time."""
        predict_started = time.monotonic()
        metrics["predict_input_shape"] = [int(value) for value in features.shape]
        metrics["predict_chunk_length"] = int(chunk)
        try:
            return original_predict(model, features, chunk, device)
        finally:
            metrics["predict_seconds"] = time.monotonic() - predict_started

    predictor.predict = timed_predict
    original_getitem = predictor.FeatureOnlySequences.__getitem__

    def timed_getitem(dataset: Any, index: int) -> Any:
        """Measure the frozen feature-only sequence reader."""
        decode_started = time.monotonic()
        try:
            return original_getitem(dataset, index)
        finally:
            metrics["feature_decode_seconds"] = (
                metrics.get("feature_decode_seconds", 0.0) + time.monotonic() - decode_started
            )
            metrics["feature_decode_count"] = int(metrics.get("feature_decode_count", 0)) + 1

    predictor.FeatureOnlySequences.__getitem__ = timed_getitem
    predictor_argv = [
        str(args.predictor.resolve()),
        str(contract["run"]["name"]),
        "train",
        "--epoch",
        str(contract["run"]["epoch"]),
        "--output-name",
        output_name,
        "--campaign-split-manifest",
        str(contract["paths"]["split_manifest"]),
        "--device",
        "cuda",
        "--inference-precision",
        "fp32",
    ]
    sys.argv = predictor_argv
    observations["predictor_argv"] = predictor_argv
    observations["python_version"] = sys.version
    observations["source_snapshot_manifest_path"] = str(args.source_manifest.resolve())
    observations["verified_source_count"] = len(pinned_sources)
    observations["started_monotonic"] = started
    write_json(args.receipt, observations)
    logging.info(
        "running %s with %s selected sequence(s)",
        mode,
        contract["selector"]["profile_group_limit"]
        if mode == "profile_first_batch"
        else contract["selector"]["full_count"],
    )

    try:
        predictor.main()
        metadata = read_json(sidecar_path)
        expected_shape = [int(value) for value in output_spec["shape"]]
        if mode == "profile_first_batch" and metadata.get("row_groups") != observations.get(
            "selected_group_ids"
        ):
            raise ValueError("profile output membership differs from selected FIT_TRAIN IDs")
        if mode == "full_export" and metadata.get("campaign_fit_scope") != "train":
            raise ValueError("full successor did not produce FIT_TRAIN predictions")
        if mode == "profile_first_batch" and len(metadata.get("row_groups", [])) != 16:
            raise ValueError("profile output did not stop at one complete 16-group batch")
        if not output_path.is_file():
            raise FileNotFoundError("frozen predictor did not finalize its output")
        output_array = np.load(output_path, mmap_mode="r")
        actual_shape = [int(value) for value in output_array.shape]
        actual_dtype = str(output_array.dtype)
        del output_array
        if actual_shape != expected_shape or actual_dtype != output_spec["dtype"]:
            raise ValueError("output shape or dtype differs from the frozen contract")
        observations.update(
            output_shape=actual_shape,
            output_dtype=actual_dtype,
            status="PROFILE_COMPLETE" if mode == "profile_first_batch" else "FULL_EXPORT_COMPLETE",
            finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            elapsed_seconds=time.monotonic() - started,
            output_size_bytes=output_path.stat().st_size,
            output_sha256=metadata.get("prediction_sha256"),
            sidecar_path=str(sidecar_path),
            sidecar_sha256=sha256_file(sidecar_path),
            output_metadata=metadata,
        )
        if predictor.torch.cuda.is_initialized():
            predictor.torch.cuda.synchronize()
            observations["cuda_peak_allocated_bytes"] = int(
                predictor.torch.cuda.max_memory_allocated()
            )
            observations["cuda_peak_reserved_bytes"] = int(
                predictor.torch.cuda.max_memory_reserved()
            )
        write_json(args.receipt, observations)
        logging.info(
            "%s completed in %.2f seconds; output is %d bytes",
            mode,
            observations["elapsed_seconds"],
            observations["output_size_bytes"],
        )
    except Exception as error:
        observations.update(
            status="FAILED",
            finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            elapsed_seconds=time.monotonic() - started,
            failure_type=type(error).__name__,
            failure=str(error),
            traceback=traceback.format_exc(),
        )
        write_json(args.receipt, observations)
        raise
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    raise SystemExit(main())
