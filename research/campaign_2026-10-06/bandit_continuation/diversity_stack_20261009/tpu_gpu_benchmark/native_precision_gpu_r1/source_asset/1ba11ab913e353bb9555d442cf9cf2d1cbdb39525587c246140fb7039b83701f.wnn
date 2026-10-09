"""Run finite/reset/reload/prefix checks on a real CUDA device and R5 fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_STARTED = time.perf_counter()


class LifecycleFailure(RuntimeError):
    """The actual-device FP16 lifecycle failed a declared correctness check."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_model_state(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as arrays:
        state = {key: torch.from_numpy(np.array(arrays[key], copy=True)) for key in arrays.files}
    if not state or any(tensor.dtype != torch.float32 for tensor in state.values()):
        raise LifecycleFailure("profile initial state must contain only FP32 model tensors")
    return state


def run(initial_state_path: Path, profile_inputs_path: Path, output_path: Path) -> dict[str, Any]:
    """Check actual CUDA autocast lifecycle against the producer's first real batch."""
    if not torch.cuda.is_available():
        raise LifecycleFailure("CUDA is unavailable; actual-device evidence is required")
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    free_before, total = torch.cuda.mem_get_info(device)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from wnn.nn import RNNModel

    with np.load(profile_inputs_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"inputs", "targets", "row_mask"}:
            raise LifecycleFailure("profile fixture has an unexpected array inventory")
        inputs_array = arrays["inputs"]
        targets_array = arrays["targets"]
        row_mask = arrays["row_mask"]
        if (
            inputs_array.shape != (32, 2000, 211)
            or inputs_array.dtype != np.dtype("<f4")
            or targets_array.shape != (32, 2000, 2)
            or targets_array.dtype != np.dtype("<f4")
            or row_mask.shape != (32, 2000)
            or row_mask.dtype != np.dtype("|b1")
        ):
            raise LifecycleFailure(
                "profile fixture differs from the actual D01 first-batch contract"
            )
        if not np.array_equal(row_mask, np.broadcast_to(np.arange(2000) >= 99, row_mask.shape)):
            raise LifecycleFailure("profile fixture row mask differs from WARMUP=99")
        if not np.isfinite(inputs_array).all() or not np.isfinite(targets_array).all():
            raise LifecycleFailure("profile fixture contains nonfinite inputs or targets")
        inputs = torch.from_numpy(np.array(inputs_array, copy=True)).to(device)

    state = _load_model_state(initial_state_path)

    def build() -> torch.nn.Module:
        model = RNNModel(211, hidden=256, layers=2, dropout=0.0, n_out=2, cell="gru", enc="linear")
        model.load_state_dict(state, strict=True)
        return model.to(device).eval()

    model = build()
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise LifecycleFailure("FP16 autocast changed an FP32 master parameter dtype")

    def forward(
        values: torch.Tensor,
        hidden: torch.Tensor | None = None,
        *,
        instance: torch.nn.Module = model,
    ):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            return instance(values, hidden)

    torch.cuda.synchronize(device)
    first_start = torch.cuda.Event(enable_timing=True)
    first_end = torch.cuda.Event(enable_timing=True)
    first_start.record()
    full_output, full_state = forward(inputs)
    first_end.record()
    torch.cuda.synchronize(device)
    first_forward_ms = float(first_start.elapsed_time(first_end))

    prefix_output, prefix_state = forward(inputs[:, :1000], None)
    suffix_output, suffix_state = forward(inputs[:, 1000:], prefix_state)
    reset_output, reset_state = forward(inputs, None)

    with tempfile.TemporaryDirectory(prefix="wnn-fp16-reload-") as temporary:
        checkpoint = Path(temporary) / "model-state.pt"
        torch.save(model.state_dict(), checkpoint)
        restored = build()
        restored.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=False), strict=True
        )
        restored.to(device).eval()
        reloaded_output, reloaded_state = forward(inputs, instance=restored)

    prefix_error = float((full_output[:, :1000].float() - prefix_output.float()).abs().max().item())
    carried_error = float(
        (full_output[:, 1000:].float() - suffix_output.float()).abs().max().item()
    )
    exact_reset = torch.equal(full_output, reset_output) and torch.equal(full_state, reset_state)
    exact_reload = torch.equal(full_output, reloaded_output) and torch.equal(
        full_state, reloaded_state
    )
    all_tensors = (
        full_output,
        full_state,
        prefix_output,
        prefix_state,
        suffix_output,
        suffix_state,
    )
    finite = all(bool(torch.isfinite(tensor).all().item()) for tensor in all_tensors)

    warm_times_ms = []
    for _ in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        forward(inputs)
        end.record()
        torch.cuda.synchronize(device)
        warm_times_ms.append(float(start.elapsed_time(end)))
    free_after, _ = torch.cuda.mem_get_info(device)
    max_abs_tolerance = 0.02
    passed = (
        finite
        and exact_reset
        and exact_reload
        and prefix_error <= max_abs_tolerance
        and carried_error <= max_abs_tolerance
    )
    receipt = {
        "schema": "wnn.native-gpu-fp16-lifecycle-gate.v1",
        "state": "passed" if passed else "failed",
        "device": {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": int(total),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "fixture": {
            "initial_state_sha256": _sha256(initial_state_path),
            "profile_inputs_sha256": _sha256(profile_inputs_path),
            "shape": [32, 2000, 211],
            "precision": "torch.autocast CUDA float16 with FP32 model parameters",
        },
        "lifecycle": {
            "all_outputs_and_states_finite": finite,
            "same_model_reset_is_bitwise_repeatable": exact_reset,
            "serialized_state_reload_is_bitwise_repeatable": exact_reload,
            "prefix_max_absolute_error": prefix_error,
            "carried_prefix_suffix_max_absolute_error": carried_error,
            "prefix_max_absolute_tolerance": max_abs_tolerance,
            "first_forward_cuda_event_ms_including_lazy_setup": first_forward_ms,
            "warmed_forward_cuda_event_ms": warm_times_ms,
            "warmed_forward_median_cuda_event_ms": float(np.median(warm_times_ms)),
        },
        "memory": {
            "free_before_bytes": int(free_before),
            "free_after_bytes": int(free_after),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "process_wall_seconds_after_module_imports": time.perf_counter() - _STARTED,
    }
    _atomic_json(output_path, receipt)
    if not passed:
        raise LifecycleFailure("actual CUDA FP16 finite/reset/reload/prefix gate failed")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--profile-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.initial_state, args.profile_inputs, args.output)
    except BaseException as error:
        failure = {
            "schema": "wnn.native-gpu-fp16-lifecycle-gate.v1",
            "state": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "process_wall_seconds_after_module_imports": time.perf_counter() - _STARTED,
        }
        if not args.output.exists():
            _atomic_json(args.output, failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
