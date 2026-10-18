"""Replay one frozen parent with causal Kalman posterior-level input replacements.

Default preflight is CPU-only and imports no Torch. ``--mode execute`` is for
the root controller after explicit GPU and resource admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from kalman_features import RAW_COLUMNS, KalmanConfig, OnlineKalmanFeatures
from kalman_normalization import prior_from_parent_normalization
from probe_contract import (
    ARMS,
    CHECKPOINT,
    CHECKPOINT_SHA256,
    DATA,
    GROUPS,
    NORMALIZATION,
    OUTPUT,
    PREFLIGHT,
    ROOT,
    SEQUENCE_IDS,
    cgroup_receipt,
    sha256_file,
    verify_preflight,
    write_preflight,
)


class InputReplacementForward:
    """Inject one causal level after original raw input produced event features."""

    def __init__(self, base_forward: Callable[..., tuple[Any, Any]], levels: Any, columns: Any):
        self.base_forward = base_forward
        self.levels = levels
        self.columns = columns
        self.next_row = 0

    def __call__(self, model: Any, inputs: Any, sl: slice, state: Any) -> tuple[Any, Any]:
        """Replace four raw slots, preserve event features, then call the parent forward."""
        start = 0 if sl.start is None else int(sl.start)
        stop = inputs.shape[1] if sl.stop is None else int(sl.stop)
        if start != self.next_row or not start < stop or stop > len(self.levels):
            raise ValueError("forward chunks must cover the complete sequence in order")
        chunk = inputs[:, sl].clone()
        chunk[..., self.columns] = self.levels[start:stop].unsqueeze(0)
        output, next_state = self.base_forward(model, chunk, slice(0, stop - start), state)
        self.next_row = stop
        return output, next_state


def predict_one(
    torch: Any,
    predict: Callable[..., np.ndarray],
    model: Any,
    feature_array: np.ndarray,
    device: Any,
    environment: Any,
) -> np.ndarray:
    """Call the canonical predictor and enforce finite (T,2) output."""
    features = torch.from_numpy(np.ascontiguousarray(feature_array)).unsqueeze(0)
    result = predict(model, features, 4_000, device, environment=environment)
    if result.shape != (1, 20_000, 2) or not np.isfinite(result).all():
        raise FloatingPointError("parent prediction shape or finite-value check failed")
    return result[0].astype(np.float32, copy=False)


def save_arm(
    output_dir: Path,
    arm: str,
    predictions: np.ndarray,
    preflight: dict[str, Any],
    checkpoint_sha: str,
) -> dict[str, Any]:
    """Write one prediction-only cache and its lineage sidecar atomically."""
    cache = output_dir / f"{arm}.npz"
    temporary = cache.with_suffix(".npz.partial")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            group_ids=np.asarray(GROUPS, dtype=np.int64),
            sequence_ids=np.asarray(SEQUENCE_IDS, dtype=np.int64),
            predictions=np.asarray(predictions, dtype=np.float32),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, cache)
    receipt = {
        "schema": "kalman-frozen-gru-prediction-cache.v1",
        "arm": arm,
        "cache_path": str(cache),
        "cache_sha256": sha256_file(cache),
        "cache_bytes": cache.stat().st_size,
        "group_ids": preflight["membership"]["groups"],
        "sequence_ids": preflight["membership"]["sequence_ids"],
        "rows_per_sequence": 20_000,
        "prediction_shape": list(predictions.shape),
        "prediction_dtype": "float32",
        "checkpoint_sha256": checkpoint_sha,
        "preflight_sha256": sha256_file(PREFLIGHT),
        "normalization_sha256": preflight["normalization"]["sha256"],
        "source_hashes": preflight["source_hashes"],
        "targets_or_masks_loaded": False,
        "official_metric_called": False,
        "selection_exposure": "F0 exposed; bounded diagnostic only",
    }
    (output_dir / f"{arm}.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def load_runtime(preflight: dict[str, Any]) -> dict[str, Any]:
    """Load the pinned parent and canonical input/prediction dependencies."""
    checkpoint_sha = sha256_file(CHECKPOINT)
    if checkpoint_sha != CHECKPOINT_SHA256:
        raise ValueError("parent checkpoint bytes differ from the frozen SHA-256")
    sys.path.insert(0, str(ROOT / "src"))
    import torch

    from wnn.data import SequenceFile
    from wnn.training.environment import InputConfiguration, TrainingEnvironment
    from wnn.training.evaluation import predict
    from wnn.training.forward import forward_chunk
    from wnn.training.models import make_model

    if not torch.cuda.is_available():
        raise RuntimeError("GPU replay requires the root-admitted CUDA worker")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    expected = {
        "name": "warm_mechanism_f0_s3_c_r3",
        "arch": "gru",
        "hidden": 256,
        "layers": 2,
        "dropout": 0.0,
        "raw_only": False,
        "online_norm": False,
        "heuristics": False,
        "research_features": "none",
        "research_feature_control": "none",
        "privileged_mode": "none",
    }
    if any(args.get(key) != value for key, value in expected.items()):
        raise ValueError("parent architecture or input recipe differs from the frozen card")
    if int(checkpoint["n_in"]) != 211:
        raise ValueError("parent checkpoint input width must be raw112 plus event99")
    n_out = 2 + 2 * len([value for value in args.get("aux", "").split(",") if value])
    if n_out != int(checkpoint["model"]["head.2.weight"].shape[0]):
        raise ValueError("checkpoint output shape disagrees with its head metadata")
    model = make_model(args, int(checkpoint["n_in"]), n_out)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device=device, dtype=torch.float32).eval()
    model._inference_fp32 = True
    mean_sha = hashlib.sha256(model.mean.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    std_sha = hashlib.sha256(model.std.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    norm = preflight["normalization"]
    if mean_sha != norm["source_full_mean_sha256"] or std_sha != norm["source_full_std_sha256"]:
        raise ValueError("model normalization buffers differ from the trusted extraction")
    normalization = json.loads(NORMALIZATION.read_text(encoding="utf-8"))
    return {
        "torch": torch,
        "device": device,
        "model": model,
        "predict": predict,
        "forward_chunk": forward_chunk,
        "sequence_file": SequenceFile(DATA),
        "environment_type": TrainingEnvironment,
        "input_configuration": InputConfiguration(
            online_norm=False,
            heuristics=False,
            raw_only=False,
            research_features="none",
            research_feature_control="none",
            privileged_mode="none",
        ),
        "prior": prior_from_parent_normalization(normalization),
        "checkpoint_sha256": checkpoint_sha,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device),
    }


def predict_group(runtime: dict[str, Any], group: int, sequence_id: int) -> dict[str, np.ndarray]:
    """Read one feature-only group and emit its three matched parent predictions."""
    item = runtime["sequence_file"].read(group, targets=False, mask=False)
    if item.seq_ix != sequence_id or item.features.shape != (20_000, 112):
        raise ValueError(f"feature-only row-group identity or shape mismatch at {group}")
    if item.targets is not None or item.scored is not None:
        raise AssertionError("feature-only SequenceFile read returned labels or masks")
    result: dict[str, np.ndarray] = {}
    for arm in ARMS:
        environment = runtime["environment_type"](
            root=ROOT,
            source_root=ROOT,
            inputs=runtime["input_configuration"],
        )
        if arm != "raw":
            model_name = "random_walk" if arm == "random_walk_posterior" else "local_trend"
            kalman = OnlineKalmanFeatures(runtime["prior"], KalmanConfig())
            levels = kalman.filter_selected_sequence(item.features[:, RAW_COLUMNS], model_name)
            torch = runtime["torch"]
            levels_tensor = torch.as_tensor(levels, device=runtime["device"], dtype=torch.float32)
            columns = torch.as_tensor(RAW_COLUMNS, device=runtime["device"], dtype=torch.long)
            replacement = InputReplacementForward(runtime["forward_chunk"], levels_tensor, columns)
            environment.forward_chunk = replacement
        result[arm] = predict_one(
            runtime["torch"],
            runtime["predict"],
            runtime["model"],
            item.features,
            runtime["device"],
            environment,
        )
        if arm != "raw" and replacement.next_row != 20_000:
            raise AssertionError("input replacement did not consume every causal row")
    del item
    return result


def execute(output_dir: Path) -> dict[str, Any]:
    """Run all frozen input arms after explicit root-controller admission."""
    preflight = verify_preflight()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite probe artifacts: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cpu_started = time.process_time()
    runtime = load_runtime(preflight)
    predictions = {arm: np.empty((len(GROUPS), 20_000, 2), dtype=np.float32) for arm in ARMS}
    for index, (group, sequence_id) in enumerate(zip(GROUPS, SEQUENCE_IDS, strict=True)):
        current = predict_group(runtime, group, sequence_id)
        for arm in ARMS:
            predictions[arm][index] = current[arm]
    receipts = [
        save_arm(output_dir, arm, predictions[arm], preflight, runtime["checkpoint_sha256"])
        for arm in ARMS
    ]
    return {
        "schema": "kalman-frozen-gru-replay.v1",
        "status": "prediction-only replay complete; independent grading required",
        "arms": receipts,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "elapsed_cpu_seconds": time.process_time() - cpu_started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "torch_version": runtime["torch_version"],
        "cuda_device": runtime["cuda_device"],
        "cgroup": cgroup_receipt(),
        "torch_imported": True,
        "gpu_used": True,
        "official_metric_called": False,
        "targets_or_masks_loaded": False,
    }


def main() -> None:
    """Run CPU-only source preflight by default or the admitted replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "execute"), default="preflight")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = write_preflight() if args.mode == "preflight" else execute(args.output_dir)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
