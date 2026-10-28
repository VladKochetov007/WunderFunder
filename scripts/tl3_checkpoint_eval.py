from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

# Frozen checkpoints expose model-specific state dictionaries and tensor batches.
from typing import Any


def _sha256(path: Path) -> str:
    """Hash one stable checkpoint or manifest file."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular non-symlink file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
        after = os.fstat(source.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("checkpoint changed while it was being read")
    return digest.hexdigest()


def _read_json(path: Path, maximum: int = 16 << 20) -> dict[str, Any]:
    """Read one bounded JSON object without following a final symlink."""
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError(f"unsafe or oversized JSON input: {path}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_sha256(value: object) -> str:
    """Hash a canonical JSON value for deterministic cohort identity."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _physical_gpu_uuid(value: object) -> str:
    """Normalize Torch's raw UUID spelling to nvidia-smi's GPU-prefixed form."""
    normalized = str(value)
    return normalized if normalized.startswith("GPU-") else f"GPU-{normalized}"


def _accumulate_wmse(
    target: Any,
    prediction: Any,
    mask: Any,
    totals: list[float],
) -> None:
    """Accumulate canonical clipped-target, absolute-target-weighted MSE."""
    import numpy as np

    values = np.asarray(target, dtype=np.float32)
    predicted = np.asarray(prediction, dtype=np.float32)
    selected = np.asarray(mask, dtype=bool)
    if values.shape != predicted.shape or values.ndim != 3 or selected.shape != values.shape[:2]:
        raise ValueError("evaluation target, prediction, and mask shapes disagree")
    if not np.isfinite(values[selected]).all() or not np.isfinite(predicted[selected]).all():
        raise ValueError("nonfinite selected target or prediction")
    clipped = np.clip(values, -2.0, 2.0)
    weights = np.abs(clipped) * selected[..., None]
    totals[0] += float(np.sum(weights * np.square(predicted - clipped), dtype=np.float64))
    totals[1] += float(np.sum(weights, dtype=np.float64))


def _evaluate_cohort(
    model: Any,
    loader: Any,
    device: Any,
    environment: Any,
    *,
    row_mask: str,
    group_order_sha256: str,
) -> dict[str, Any]:
    """Stream official pooled WP and clipped-target WMSE for one fixed cohort."""
    import numpy as np

    from wnn.gmetric import GlobalAccumulator
    from wnn.training.evaluation import predict

    accumulator = GlobalAccumulator()
    wmse_totals = [0.0, 0.0]
    selected_rows = 0
    for features, target, scored in loader:
        prediction = predict(model, features, 4000, device, environment=environment)
        targets = target.numpy()
        scored_rows = scored.numpy().astype(bool, copy=False)
        positions = np.arange(targets.shape[1])[None, :]
        if row_mask == "all_after_warmup":
            mask = np.broadcast_to(positions >= environment.warmup, scored_rows.shape).copy()
        elif row_mask == "is_scored":
            mask = scored_rows.copy()
            mask[:, : environment.warmup] = False
        else:
            raise ValueError("unsupported evaluation row mask")
        accumulator.add(targets, prediction, mask)
        _accumulate_wmse(targets, prediction, mask, wmse_totals)
        selected_rows += int(mask.sum())
    if wmse_totals[1] <= 0 or selected_rows <= 0:
        raise ValueError("evaluation cohort has no positive target weight after masking")
    pooled = accumulator.result()
    metrics = {
        "wp": float(pooled["wp"]),
        "t0": float(pooled["t0"]),
        "t1": float(pooled["t1"]),
        "wmse": wmse_totals[0] / wmse_totals[1],
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("evaluation returned a nonfinite metric")
    return {
        "row_mask": row_mask,
        "cohort_count": len(loader.dataset),
        "cohort_group_order_sha256": group_order_sha256,
        "selected_rows": selected_rows,
        "metrics": metrics,
    }


def evaluate_checkpoint(
    run_dir: Path,
    data_root: Path,
    source_root: Path,
    split_manifest: Path,
    epoch_index: int,
    run_identity_sha256: str,
    source_identity_sha256: str,
    batch_size: int,
) -> dict[str, Any]:
    """Evaluate one frozen checkpoint on 32 train groups and full official valid."""
    source_root = source_root.resolve(strict=True)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source_root / "src"))
    import torch
    from torch.utils.data import DataLoader

    from wnn.data import SequenceFile
    from wnn.resources import configure_strict_fp32
    from wnn.training.datasets import Sequences
    from wnn.training.environment import TrainingEnvironment
    from wnn.training.models import make_model
    from wnn.training.source_identity import training_source_identity

    data_root = data_root.resolve(strict=True)
    run_dir = run_dir.resolve(strict=True)
    if run_dir != data_root / "artifacts" / "runs" / run_dir.name:
        raise ValueError("run directory does not match the data-root artifact layout")
    provenance = _read_json(run_dir / "provenance.json")
    implementation = provenance.get("trainer_implementation")
    if (
        not isinstance(implementation, dict)
        or implementation.get("sha256") != source_identity_sha256
    ):
        raise ValueError("run source identity does not match the launch receipt")
    environment = TrainingEnvironment(source_root=source_root, root=data_root)
    current_identity = training_source_identity(environment)
    if current_identity.get("sha256") != source_identity_sha256:
        raise ValueError("evaluator source tree differs from the fitted trainer source")
    if _sha256(split_manifest) != provenance.get("split_manifest_sha256"):
        raise ValueError("official split manifest does not match run provenance")
    manifest = _read_json(split_manifest)
    fit = manifest.get("fit", {})
    evaluation = manifest.get("evaluation", {})
    train_groups = provenance.get("fit_row_groups", {}).get("train")
    fit_valid = provenance.get("fit_row_groups", {}).get("valid")
    valid_groups = provenance.get("evaluation_row_groups")
    if not all(isinstance(values, list) for values in (train_groups, fit_valid, valid_groups)):
        raise ValueError("run provenance omits actual fit/evaluation memberships")
    if set(train_groups) != set(fit.get("train", [])) or fit_valid:
        raise ValueError("train evaluation groups do not come from the official train-only fit")
    if set(valid_groups) != set(evaluation.get("valid", [])):
        raise ValueError("validation evaluation is not full official validation")
    train_prefix = [int(group) for group in train_groups[:32]]
    valid_order = [int(group) for group in valid_groups]
    if len(train_prefix) != 32 or not valid_order:
        raise ValueError("fixed evaluation cohorts are incomplete")
    checkpoint_name = f"model_ep{epoch_index}.pt"
    checkpoint_path = run_dir / checkpoint_name
    checkpoint_sha256 = _sha256(checkpoint_path)
    expected_sha = provenance.get("checkpoint_sha256", {}).get(checkpoint_name)
    if checkpoint_sha256 != expected_sha:
        raise ValueError("frozen checkpoint bytes do not match run provenance")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("epoch") != epoch_index:
        raise ValueError("frozen checkpoint epoch metadata does not match its filename")
    args = checkpoint.get("args")
    if not isinstance(args, dict) or args != provenance.get("args"):
        raise ValueError("checkpoint and run provenance trainer arguments differ")
    if (
        args.get("arch", "gru") != "gru"
        or args.get("cell", "gru") != "gru"
        or args.get("enc", "linear") not in {"linear", "tiedlead"}
        or args.get("precision") != "fp32"
        or args.get("raw_only")
        or args.get("numeric_map", "none") != "none"
        or args.get("online_norm")
        or args.get("heuristics")
        or args.get("research_features", "none") != "none"
        or args.get("privileged_mode", "none") != "none"
        or args.get("prop_input")
        or args.get("sentinel_indicators", "none") != "none"
    ):
        raise ValueError("frozen evaluator supports only the declared plain GRU input schemas")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("frozen checkpoint worker requires exactly its receipt-visible GPU")
    visible_uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_uuid.startswith("GPU-"):
        raise ValueError("frozen worker is not bound to the receipt GPU UUID")
    runtime_uuid = _physical_gpu_uuid(torch.cuda.get_device_properties(0).uuid)
    if runtime_uuid != visible_uuid:
        raise ValueError("frozen worker CUDA device UUID differs from the launch receipt")
    configure_strict_fp32()
    output_width = int(checkpoint["model"]["head.2.weight"].shape[0])
    model = make_model(args, int(checkpoint["n_in"]), output_width).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    model._inference_fp32 = True
    environment.inputs.tiedlead = args.get("enc") == "tiedlead"
    if checkpoint.get("readout") is not None:
        environment.readout.tables = tuple(
            torch.as_tensor(table, device="cuda") for table in checkpoint["readout"]
        )
        environment.readout.tilted = bool(checkpoint.get("readout_tilted", False))
    if SequenceFile(data_root / "datasets" / "train.parquet").has_mask:
        raise ValueError("train evaluation source unexpectedly contains is_scored")
    if not SequenceFile(data_root / "datasets" / "valid.parquet").has_mask:
        raise ValueError("official validation source is missing is_scored")
    train_dataset = Sequences("train", train_prefix, environment=environment)
    valid_dataset = Sequences("valid", valid_order, environment=environment)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda:0")
    train_eval = _evaluate_cohort(
        model,
        train_loader,
        device,
        environment,
        row_mask="all_after_warmup",
        group_order_sha256=_canonical_sha256(train_prefix),
    )
    official_valid = _evaluate_cohort(
        model,
        valid_loader,
        device,
        environment,
        row_mask="is_scored",
        group_order_sha256=_canonical_sha256(valid_order),
    )
    if _sha256(checkpoint_path) != checkpoint_sha256:
        raise ValueError("frozen checkpoint changed during evaluation")
    return {
        "schema": "wnn.tl3-frozen-checkpoint-evaluation.v1",
        "run_identity_sha256": run_identity_sha256,
        "source_identity_sha256": source_identity_sha256,
        "completed_epoch": epoch_index + 1,
        "source_epoch_index": epoch_index,
        "checkpoint_sha256": checkpoint_sha256,
        "train_eval": train_eval,
        "valid": official_valid,
    }


def _parser() -> argparse.ArgumentParser:
    """Build the bounded worker interface used by the bridge subprocess."""
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen TL3 checkpoint on fixed cohorts."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--epoch-index", type=int, required=True)
    parser.add_argument("--run-identity-sha256", required=True)
    parser.add_argument("--source-identity-sha256", required=True)
    parser.add_argument("--batch-size", type=int, choices=range(1, 17), required=True)
    return parser


def main() -> None:
    """Run one read-only evaluation and print a finite JSON receipt."""
    args = _parser().parse_args()
    if args.epoch_index < 0:
        raise SystemExit("epoch index must be nonnegative")
    result = evaluate_checkpoint(
        args.run_dir,
        args.data_root,
        args.source_root,
        args.split_manifest,
        args.epoch_index,
        args.run_identity_sha256,
        args.source_identity_sha256,
        args.batch_size,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
