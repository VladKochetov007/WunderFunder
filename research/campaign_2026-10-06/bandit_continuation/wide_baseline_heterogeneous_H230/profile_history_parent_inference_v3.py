"""Profile small batches from the frozen all-data H230 parent on real inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.dont_write_bytecode = True

ROOT_DEFAULT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
PROFILE_BATCH_SIZE = 16
PROFILE_CHUNK_ROWS = 4000
PROFILE_TIMED_BATCHES = 3
PROFILE_WARMUP_BATCHES = 1
PROFILE_SEQUENCE_LENGTH = 20000
PROFILE_GROUPS_PER_SPLIT = PROFILE_BATCH_SIZE * (
    PROFILE_TIMED_BATCHES + PROFILE_WARMUP_BATCHES
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def require_hash(path: Path, expected: str, label: str) -> str:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: {actual} != {expected}")
    return actual


def validate_raw_feature_batch(
    batch: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    raw_feature_width: int,
) -> None:
    """Require the SequenceFile raw-feature layout before trainer preprocessing."""
    expected = (batch_size, sequence_length, raw_feature_width)
    if batch.dtype != torch.float32 or tuple(batch.shape) != expected:
        raise ValueError(
            f"unexpected raw feature batch: dtype={batch.dtype}, shape={tuple(batch.shape)}, "
            f"expected=float32 {expected}"
        )


def verify_source_closure(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    source = contract["source"]
    manifest_path = root / source["manifest_path"]
    require_hash(manifest_path, source["manifest_sha256"], "source manifest")
    manifest = read_json(manifest_path)
    source_root = root / manifest["source_root"]
    verified = 0
    for entry in manifest["files"]:
        path = source_root / entry["path"]
        require_hash(path, entry["sha256"], f"source member {entry['path']}")
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"source member size mismatch: {entry['path']}")
        verified += 1
    if verified != source["file_count"] or verified != manifest["file_count"]:
        raise ValueError(f"source member count mismatch: {verified}")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": source["manifest_sha256"],
        "source_root": str(source_root),
        "verified_file_count": verified,
    }


def verify_frozen_inputs(root: Path, contract: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_evidence = verify_source_closure(root, contract)
    review = contract["root_review"]
    review_path = root / review["path"]
    require_hash(review_path, review["sha256"], "root source review")
    review_json = read_json(review_path)
    if review_json.get("status") != "PASS_scientific_source_only_future_phase_admission_required":
        raise ValueError("root source review does not authorize this profile basis")

    recipe_ref = contract["recipe"]
    recipe_path = root / recipe_ref["path"]
    require_hash(recipe_path, recipe_ref["sha256"], "history-swap recipe")
    recipe = read_json(recipe_path)
    parent = recipe["parent_dependency"]

    parent_contract_path = root / parent["fit_contract_path"]
    require_hash(parent_contract_path, parent["fit_contract_sha256"], "parent fit contract")
    terminal_path = root / parent["terminal_receipt_path"]
    require_hash(terminal_path, parent["terminal_receipt_sha256"], "parent terminal receipt")
    terminal = read_json(terminal_path)
    if (
        terminal.get("status") != "finished"
        or terminal.get("exit_status") != 0
        or terminal.get("contract_sha256") != parent["fit_contract_sha256"]
    ):
        raise ValueError("parent terminal receipt is not a successful matching fit")
    provenance_path = root / parent["provenance_path"]
    require_hash(provenance_path, parent["provenance_sha256"], "parent provenance")
    checkpoint_path = root / parent["checkpoint_path"]
    require_hash(checkpoint_path, parent["checkpoint_sha256"], "parent checkpoint")

    return (
        source_evidence,
        {
            "review_path": str(review_path),
            "review_sha256": review["sha256"],
            "recipe_path": str(recipe_path),
            "recipe_sha256": recipe_ref["sha256"],
            "run": parent["run"],
            "attempt_id": parent["attempt_id"],
            "terminal_path": str(terminal_path),
            "terminal_sha256": parent["terminal_receipt_sha256"],
            "fit_contract_path": str(parent_contract_path),
            "fit_contract_sha256": parent["fit_contract_sha256"],
            "provenance_path": str(provenance_path),
            "provenance_sha256": parent["provenance_sha256"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": parent["checkpoint_sha256"],
            "checkpoint_epoch_zero_based": parent["checkpoint_epoch_zero_based"],
        },
    )


def profile_split(
    root: Path,
    module: Any,
    model: torch.nn.Module,
    device: torch.device,
    split: str,
    raw_feature_width: int,
    model_input_width: int,
) -> dict[str, Any]:
    from wnn.distillation_cache import row_groups_sha256

    groups = module._target_groups(root, split)
    sampled_groups = groups[:PROFILE_GROUPS_PER_SPLIT]
    if len(sampled_groups) != PROFILE_GROUPS_PER_SPLIT:
        raise ValueError(f"not enough groups for the frozen {split} profile")
    dataset_path = root / "datasets" / f"{split}.parquet"
    target_index_path = root / "cache" / f"{split}_targets.npz"
    dataset_identity = {
        "path": str(dataset_path),
        "bytes": dataset_path.stat().st_size,
        "mtime_ns": dataset_path.stat().st_mtime_ns,
    }
    index_identity = {
        "path": str(target_index_path),
        "sha256": file_sha256(target_index_path),
        "bytes": target_index_path.stat().st_size,
        "mtime_ns": target_index_path.stat().st_mtime_ns,
    }
    dataset = module.FeatureOnlyGroups(dataset_path, sampled_groups)
    loader = DataLoader(dataset, batch_size=PROFILE_BATCH_SIZE, shuffle=False, num_workers=0)
    iterator = iter(loader)
    batch_seconds: list[float] = []
    output_digest = hashlib.sha256()
    sequences_checked = 0
    torch.cuda.synchronize(device)
    warmup = next(iterator)
    validate_raw_feature_batch(
        warmup,
        batch_size=PROFILE_BATCH_SIZE,
        sequence_length=PROFILE_SEQUENCE_LENGTH,
        raw_feature_width=raw_feature_width,
    )
    warm_prediction = module.train_rnn.predict(model, warmup, PROFILE_CHUNK_ROWS, device)
    warm_values = np.asarray(warm_prediction[..., :2], dtype=np.float32)
    if warm_values.shape != (PROFILE_BATCH_SIZE, 20000, 2) or not np.isfinite(warm_values).all():
        raise ValueError("warm-up prediction has an invalid shape or nonfinite values")
    del warmup, warm_prediction, warm_values
    torch.cuda.synchronize(device)

    for _ in range(PROFILE_TIMED_BATCHES):
        batch = next(iterator)
        validate_raw_feature_batch(
            batch,
            batch_size=PROFILE_BATCH_SIZE,
            sequence_length=PROFILE_SEQUENCE_LENGTH,
            raw_feature_width=raw_feature_width,
        )
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        prediction = module.train_rnn.predict(model, batch, PROFILE_CHUNK_ROWS, device)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        values = np.asarray(prediction[..., :2], dtype=np.float32)
        if values.shape != (PROFILE_BATCH_SIZE, 20000, 2) or not np.isfinite(values).all():
            raise ValueError("timed prediction has an invalid shape or nonfinite values")
        output_digest.update(np.ascontiguousarray(values).tobytes())
        sequences_checked += len(values)
        batch_seconds.append(elapsed)
        del batch, prediction, values

    return {
        "split": split,
        "dataset_identity_stat_only": dataset_identity,
        "target_index_identity": index_identity,
        "target_index_group_count": len(groups),
        "sampled_groups": PROFILE_GROUPS_PER_SPLIT,
        "sampled_row_groups_sha256": row_groups_sha256(sampled_groups),
        "batch_size": PROFILE_BATCH_SIZE,
        "chunk_rows": PROFILE_CHUNK_ROWS,
        "raw_feature_width_from_sealed_data_module": raw_feature_width,
        "model_input_width_after_sealed_preprocessing": model_input_width,
        "raw_feature_shape_per_batch": [PROFILE_BATCH_SIZE, PROFILE_SEQUENCE_LENGTH, raw_feature_width],
        "model_input_shape_per_sequence": [PROFILE_SEQUENCE_LENGTH, model_input_width],
        "warmup_batches": PROFILE_WARMUP_BATCHES,
        "timed_batches": PROFILE_TIMED_BATCHES,
        "timed_sequences": sequences_checked,
        "timed_batch_wall_seconds": batch_seconds,
        "timed_sequence_per_second": sequences_checked / sum(batch_seconds),
        "timed_prediction_sha256": output_digest.hexdigest(),
        "prediction_dtype": "float32",
        "prediction_shape_per_batch": [PROFILE_BATCH_SIZE, 20000, 2],
        "features_only": True,
        "labels_or_masks_read": False,
        "prediction_arrays_persisted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    contract_path = args.contract.resolve(strict=True)
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite profile output: {output_path}")
    contract = read_json(contract_path)
    runner_sha = contract["profile_runner_sha256"]
    require_hash(Path(__file__).resolve(), runner_sha, "profile runner")
    source_evidence, parent_evidence = verify_frozen_inputs(root, contract)

    source_root = root / contract["source"]["source_root"]
    sys.path.insert(0, str(source_root / "scripts"))
    sys.path.insert(0, str(source_root / "src"))
    import export_all_data_parent as exporter  # noqa: E402

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the admitted inference profile")
    device = torch.device("cuda")
    torch.set_num_threads(2)
    os.environ["WNN_PROJECT_ROOT"] = str(root)
    exporter.train_rnn.ROOT = root
    checkpoint_path = Path(parent_evidence["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    exporter._check_parent(checkpoint, argparse.Namespace(run=parent_evidence["run"]))
    if checkpoint.get("n_in") != 211:
        raise ValueError("parent checkpoint input width is not the actual 211-feature shape")
    if checkpoint.get("epoch") != 3:
        raise ValueError("parent checkpoint is not fixed zero-based epoch 3")
    exporter.train_rnn.configure_torch()
    output_width = int(checkpoint["model"]["head.2.weight"].shape[0])
    model = exporter.train_rnn.make_model(checkpoint["args"], checkpoint["n_in"], output_width).to(
        device
    )
    model.load_state_dict(checkpoint["model"])
    exporter._configure_checkpoint(checkpoint, model, device)
    exporter.train_rnn.configure_factorized(
        model, int(checkpoint["args"].get("recurrent_rank", 0))
    )
    exporter.train_rnn.configure_qat(model, str(checkpoint["args"].get("qat", "none")))
    model._inference_fp32 = True
    model.eval()
    raw_feature_width = int(exporter.train_rnn.N_FEATURES)
    model_input_width = int(checkpoint["n_in"])
    probe = torch.zeros((1, 3, raw_feature_width), dtype=torch.float32, device=device)
    derived_input_width = int(exporter.train_rnn.build_inputs(probe).shape[-1])
    if derived_input_width != model_input_width:
        raise ValueError(
            "sealed trainer preprocessing width differs from the frozen parent checkpoint: "
            f"{derived_input_width} != {model_input_width}"
        )
    del probe
    del checkpoint
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    profile_started = time.perf_counter()
    splits = [
        profile_split(
            root,
            exporter,
            model,
            device,
            split,
            raw_feature_width,
            model_input_width,
        )
        for split in ("train", "valid")
    ]
    torch.cuda.synchronize(device)
    receipt = {
        "schema": "h230-history-parent-inference-profile-result-v1",
        "status": "profile_complete_no_export_no_fit",
        "contract_path": str(contract_path),
        "contract_sha256": file_sha256(contract_path),
        "profile_runner_sha256": runner_sha,
        "observed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_closure": source_evidence,
        "parent": parent_evidence,
        "device": {
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": total_bytes,
            "free_memory_bytes_before_profile": free_bytes,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "feature_widths": {
            "raw_sequencefile_width": raw_feature_width,
            "event_features_from_sealed_trainer": int(exporter.train_rnn.N_EVENT_FEATURES),
            "derived_model_input_width": derived_input_width,
            "checkpoint_n_in": model_input_width,
        },
        "profile": {
            "model_setup_and_preflight_seconds_excluded": True,
            "total_profile_seconds": time.perf_counter() - profile_started,
            "splits": splits,
            "prediction_state_reset": "each predict() call starts a fresh state; state spans only 4000-row chunks in that sequence",
            "no_training_or_optimizer_step": True,
            "no_targets_or_masks_read": True,
            "no_prediction_arrays_written": True,
            "no_cache_mutation": True,
            "full_dataset_export_estimate": None,
        },
        "process_max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "phase_boundary": "This profile is execution evidence only; each full split export, CPU cache build, and student fit requires a fresh phase admission.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    partial.write_text(json.dumps(receipt, indent=2) + "\n")
    partial.replace(output_path)
    print(json.dumps({"status": receipt["status"], "output": str(output_path), "timed_sequences": 96}, sort_keys=True))


if __name__ == "__main__":
    main()
