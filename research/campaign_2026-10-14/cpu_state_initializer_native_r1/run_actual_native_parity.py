"""Compare initialized native callbacks with the Python wrapper on a pinned fit sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import wnn_native

from wnn.data import SequenceFile
from wnn.features import event_features_torch
from wnn.nn import RNNModel
from wnn.state_initialization import InitializedGRU, export_native_initializer

CHECKPOINT_REL = Path("artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/model_ep5.pt")
CHECKPOINT_SHA256 = "e30577ffb39d061bc40043f68a90285965bd2cf790f6ba5947ba78c60b776eda"
REFERENCE_SCRIPT_REL = Path("scripts/diagnose_state_decay.py")
REFERENCE_SCRIPT_SHA256 = "374e5f7b39fd7dcd01f5f1a1ca0aae152d5ad24c6685ddb03702a825779b12c2"
EXPECTED_FEATURE_SHA256 = "fe24c974e6d6e3e720d36937be06d222d00f743efd65e4a0a3ec178db891f57a"
EXPECTED_SEQ_IX = 7_598_721
ROW_GROUP = 37
ROW_COUNT = 20_000
MODES = ("zero", "constant", "first_row", "prefix")


def sha256(path: Path) -> str:
    """Hash a file with bounded reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_arrays(wrapper: InitializedGRU) -> dict[str, np.ndarray]:
    """Export weights, projecting the existing first two readout rows for callback ABI."""
    arrays = {}
    for name, tensor in wrapper.backbone.state_dict().items():
        values = tensor.detach().cpu().numpy()
        if name == "head.2.weight" or name == "head.2.bias":
            values = values[:2]
        arrays[name] = np.ascontiguousarray(values, dtype=np.float32)
    return arrays


def make_wrapper(state_dict: dict[str, torch.Tensor], mode: str) -> InitializedGRU:
    """Recreate deterministic nonzero initializer parameters over the pinned backbone."""
    torch.manual_seed(197)
    output_count = int(state_dict["head.2.weight"].shape[0])
    backbone = RNNModel(211, hidden=256, layers=2, n_out=output_count)
    backbone.load_state_dict(state_dict, strict=True)
    wrapper = InitializedGRU(backbone, mode, residual_bound=0.7)
    with torch.no_grad():
        if wrapper.constant is not None:
            wrapper.constant.fill_(0.4)
        if wrapper.predictor is not None:
            wrapper.predictor[0].weight.normal_(std=0.025)
            wrapper.predictor[0].bias.normal_(std=0.02)
            wrapper.predictor[-1].weight.normal_(std=0.035)
            wrapper.predictor[-1].bias.fill_(0.2)
    wrapper.eval()
    return wrapper


def callback_predictions(model: object, raw: np.ndarray, seq_ix: int) -> np.ndarray:
    """Run a complete sequence through the shipped one-row callback interface."""
    outputs = []
    for row in raw:
        point = SimpleNamespace(seq_ix=seq_ix, state=row, need_prediction=True)
        prediction = model.predict(point)
        if prediction is None:
            raise AssertionError("callback omitted a required prediction")
        outputs.append(np.asarray(prediction, dtype=np.float32))
    return np.stack(outputs)


def error_summary(actual: np.ndarray, expected: np.ndarray) -> dict[str, float | bool]:
    """Summarize numerical differences without claiming exact equivalence."""
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "rms": float(np.sqrt(np.mean(difference**2))),
        "allclose_rtol_3e-4_atol_3e-5": bool(np.allclose(actual, expected, rtol=3e-4, atol=3e-5)),
    }


def chunked_reference(
    wrapper: InitializedGRU, inputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the same full sequence in cuts bracketing the row-99 boundary."""
    cuts = sorted({0, 1, 37, 98, 99, 100, inputs.shape[1]})
    cuts = [cut for cut in cuts if 0 <= cut <= inputs.shape[1]]
    state = None
    outputs = []
    with torch.inference_mode():
        for begin, end in zip(cuts[:-1], cuts[1:], strict=True):
            if end <= begin:
                continue
            output, state = wrapper(inputs[:, begin:end], state)
            outputs.append(output)
    if state is None:
        raise AssertionError("sequence fixture is empty")
    return torch.cat(outputs, dim=1), state.hidden


def main() -> None:
    """Run bounded actual-shape native and Python comparisons on the frozen train row."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    torch.set_num_threads(1)
    checkpoint_path = args.root / CHECKPOINT_REL
    reference_script = args.root / REFERENCE_SCRIPT_REL
    if sha256(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint differs from the pinned clean F0 control")
    if sha256(reference_script) != REFERENCE_SCRIPT_SHA256:
        raise ValueError("reference diagnostic source differs from its pinned receipt")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]
    raw_sequence = SequenceFile(args.root / "datasets/train.parquet").read(ROW_GROUP)
    if raw_sequence.seq_ix != EXPECTED_SEQ_IX:
        raise ValueError("source-qualified fitting sequence identity changed")
    feature_hash = hashlib.sha256(
        np.asarray(raw_sequence.features, dtype="<f4").tobytes()
    ).hexdigest()
    if feature_hash != EXPECTED_FEATURE_SHA256:
        raise ValueError("fitting-sequence feature bytes differ from the pinned receipt")
    if raw_sequence.features.shape[1] != 112 or len(raw_sequence.features) < ROW_COUNT:
        raise ValueError("fitting sequence has an unexpected raw shape or length")
    raw = np.ascontiguousarray(raw_sequence.features[:ROW_COUNT], dtype=np.float32)
    raw_tensor = torch.from_numpy(raw)[None]
    inputs = torch.cat((raw_tensor, event_features_torch(raw_tensor)), dim=-1)
    if inputs.shape != (1, ROW_COUNT, 211):
        raise ValueError(f"unexpected model input shape {tuple(inputs.shape)}")

    results: dict[str, dict[str, object]] = {}
    predictions_by_policy: dict[str, dict[str, np.ndarray]] = {"exact": {}, "fast": {}}
    chunk_errors: dict[str, dict[str, float | bool]] = {}
    for mode_index, mode in enumerate(MODES):
        wrapper = make_wrapper(state_dict, mode)
        with torch.inference_mode():
            reference, reference_state = wrapper(inputs)
        chunked, chunked_hidden = chunked_reference(wrapper, inputs)
        chunk_errors[mode] = error_summary(chunked.cpu().numpy()[0], reference.cpu().numpy()[0])
        chunk_hidden_error = error_summary(
            chunked_hidden.cpu().numpy()[:, 0], reference_state.hidden.cpu().numpy()[:, 0]
        )
        chunk_errors[mode]["final_hidden_max_abs"] = chunk_hidden_error["max_abs"]
        reference_predictions = reference.cpu().numpy()[0, :, :2]
        reference_hidden = reference_state.hidden.cpu().numpy()[:, 0]
        weights = tensor_arrays(wrapper)
        config = export_native_initializer(wrapper)
        policy_results: dict[str, dict[str, object]] = {}
        for policy, fast_activations in (("exact", False), ("fast", True)):
            native_model = wnn_native.EnsemblePredictor(
                [weights],
                [1.0],
                fast_activations=fast_activations,
                precision="f32",
                state_initializers=[config],
            )
            actual = callback_predictions(native_model, raw, 10_000 + mode_index)
            prediction_error = error_summary(actual, reference_predictions)
            native_hidden = native_model.hidden_states[0]
            hidden_error = error_summary(native_hidden, reference_hidden)
            final_hidden = native_hidden.copy()
            replay = callback_predictions(native_model, raw, 20_000 + mode_index)
            prediction_replay = bool(np.array_equal(replay, actual))
            hidden_replay = bool(np.array_equal(native_model.hidden_states[0], final_hidden))
            policy_results[policy] = {
                "prediction_vs_python": prediction_error,
                "final_hidden_vs_python": hidden_error,
                "reset_prediction_bitwise_replay": prediction_replay,
                "reset_hidden_bitwise_replay": hidden_replay,
                "finite_predictions": bool(np.isfinite(actual).all()),
                "finite_hidden": bool(np.isfinite(native_hidden).all()),
            }
            predictions_by_policy[policy][mode] = actual
        results[mode] = policy_results

    for mode in MODES:
        exact = predictions_by_policy["exact"][mode]
        fast = predictions_by_policy["fast"][mode]
        results[mode]["fast_vs_exact"] = error_summary(fast, exact)
        if mode == "zero":
            for policy in ("exact", "fast"):
                weights = tensor_arrays(make_wrapper(state_dict, "zero"))
                baseline = wnn_native.EnsemblePredictor(
                    [weights], [1.0], fast_activations=(policy == "fast"), precision="f32"
                )
                baseline_prediction = callback_predictions(baseline, raw, 10_000)
                results[mode][f"{policy}_zero_vs_omitted_bitwise"] = bool(
                    np.array_equal(baseline_prediction, predictions_by_policy[policy][mode])
                )

    native_path = Path(wnn_native.__file__).resolve()
    runner_script = Path(__file__).resolve()
    result = {
        "schema": "wnn.state-initializer-actual-native-parity.v1",
        "source_commit": "d3686dfd8d9dd5e1860129d3aa1fa892f83dca74",
        "runner_script": str(runner_script),
        "runner_script_sha256": sha256(runner_script),
        "execution": {
            "argv": sys.argv,
            "root": str(args.root.resolve()),
            "output": str(args.output.resolve()),
            "cpu_quota": os.environ.get("WNN_CPU_QUOTA"),
            "threads": os.environ.get("WNN_THREADS"),
            "memory_max": os.environ.get("WNN_MEM_MAX"),
            "runtime_max": os.environ.get("WNN_RUNTIME_MAX"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "prior_failed_cli_attempt": {
                "command": "PYTHONPATH=src:native/target/release CUDA_VISIBLE_DEVICES= WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s scripts/limited .venv/bin/python research/campaign_2026-10-14/cpu_state_initializer_native_r1/run_actual_native_parity.py --help",
                "exit_code": 127,
                "error": "ionice: failed to execute .venv/bin/python: No such file or directory",
                "diagnosis": "The sparse worktree has no local .venv. The same bounded --help preflight passed with MAIN's absolute Python 3.13 environment path; no source or input changed.",
            },
        },
        "checkpoint": str(CHECKPOINT_REL),
        "checkpoint_sha256": sha256(checkpoint_path),
        "reference_script": str(REFERENCE_SCRIPT_REL),
        "reference_script_sha256": sha256(reference_script),
        "data": {
            "source": "train",
            "row_group": ROW_GROUP,
            "seq_ix": raw_sequence.seq_ix,
            "rows_evaluated": ROW_COUNT,
            "full_sequence_raw_shape": list(raw_sequence.features.shape),
            "feature_sha256": feature_hash,
            "model_input_shape": list(inputs.shape),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "native_path": str(native_path),
            "native_sha256": sha256(native_path),
            "torch_threads": torch.get_num_threads(),
            "visible_cuda_devices": "CUDA_VISIBLE_DEVICES is empty",
        },
        "comparison": {
            "policies": ["native exact activation f32", "native fast activation f32"],
            "initializer_parameters": "deterministic synthetic parameters over the pinned unchanged checkpoint; no fitting",
            "native_output_adapter": "native callback requires two outputs; uses exactly the first two existing 23-row checkpoint head outputs, matching diagnose_state_decay.py's [:2] comparison",
            "wrapper_chunk_cuts": [0, 1, 37, 98, 99, 100, ROW_COUNT],
            "all_modes_finite": all(
                mode_result[policy]["finite_predictions"] and mode_result[policy]["finite_hidden"]
                for mode_result in results.values()
                for policy in ("exact", "fast")
            ),
            "native_callback_results": results,
            "python_wrapper_chunk_vs_full": chunk_errors,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "host_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "quality_claim": False,
        "limitations": "One clean-F0 checkpoint, one source-qualified fitting sequence and deterministic synthetic initializer tensors. Exact/fast deltas are callback numerics only; no initializer fit, score, population stability, runtime ranking or submission claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
