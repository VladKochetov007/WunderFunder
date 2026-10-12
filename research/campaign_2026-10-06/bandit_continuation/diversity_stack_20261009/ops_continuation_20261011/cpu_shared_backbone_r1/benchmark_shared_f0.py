#!/usr/bin/env python3
"""Compare separate and exact shared-backbone Q8VNNI F0 inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import sys
import sysconfig
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import numpy as np

ROOT = Path(__file__).resolve().parents[6]
PROJECT_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
PACKET_ROOT = Path(__file__).resolve().parent
RUNNER_PATH = Path(
    "/home/vlad/development/wunder/wnn-agent-worktrees/"
    "zen4-native-cpu-optimization-20261012/scripts/benchmark_submission_callbacks.py"
)
RUNNER_SHA256 = "8bc57f520b9b284ca863fb8a798be3a0e923ad9242c0fb7ffa58abc2e2a0af53"
UTILS_SHA256 = "e8e1034d61be359ebe943015e4e6bd2f76716303eff520857b7d07eeeb6144a6"
PANEL_PATH = Path("/tmp/wnn_submission_exploit_runtime_r1/valid_first10_complete_groups.parquet")
PANEL_SHA256 = "e5e6472cedfcb20ecf01195777ab986b3560379c5e81caea71973632dde0d6a3"
STUDENT_CHECKPOINT = PROJECT_ROOT / "artifacts/runs/oldz4_h230_student_f0_s3_r3/model_ep1.pt"
HFT_CHECKPOINT = PROJECT_ROOT / "artifacts/runs/oldz4_h230_hft_f0_s3_r5/model_ep0.pt"
STUDENT_SHA256 = "664643b2653a85d192f74a976d76afa933ae56c02b2621387d2f576fd61695eb"
HFT_SHA256 = "2bde556119ededc3693733280444a42f0f64138a2e3759bed6b068cb72f18ec8"
GRADE_ROOT = ROOT / "research/campaign_2026-10-06/bandit_continuation/runtime/receipts"
STUDENT_GRADE = GRADE_ROOT / "oldz4-r5-f0-s3-student-export/F0_GRADE_SINGLE_R5.json"
HFT_GRADE = GRADE_ROOT / "oldz4-r5-f0-s3-hft-export/F0_HFT_GRADE_PAIRED_R5.json"
FIXED_BLEND_GRADE = GRADE_ROOT / "oldz4-r5-f0-s3-hft-export/F0_STUDENT_HFT_FIXED_HALF_GRADE_R5.json"
HEAD_KEYS = {"head.0.weight", "head.0.bias", "head.2.weight", "head.2.bias"}
PRACTICAL_GAIN_FRACTION = 0.05


class NativePredictor(Protocol):
    """Subset of the native callback interface used by this benchmark."""

    @property
    def precision(self) -> str: ...

    @property
    def storage_bytes(self) -> list[dict[str, int]]: ...

    def predict(self, data_point: object) -> np.ndarray | None: ...

    def reset(self) -> None: ...


@dataclass
class WrappedPredictor:
    """Match the existing callback benchmark's model wrapper contract."""

    model: NativePredictor

    def predict(self, data_point: object) -> np.ndarray | None:
        """Forward one callback to the native predictor."""
        return self.model.predict(data_point)


@dataclass(frozen=True)
class CallbackPoint:
    """Small input object for native API invariant checks."""

    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


def sha256_file(path: Path) -> str:
    """Hash one file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Hash an in-memory contiguous prediction or feature buffer."""
    return hashlib.sha256(value).hexdigest()


def load_callback_runner() -> ModuleType:
    """Load the frozen project callback and panel helpers by source hash."""
    if sha256_file(RUNNER_PATH) != RUNNER_SHA256:
        raise ValueError("callback benchmark source differs from the pinned runner")
    utils_path = RUNNER_PATH.parent.parent / "utils.py"
    if sha256_file(utils_path) != UTILS_SHA256:
        raise ValueError("callback benchmark utilities differ from the pinned source")
    spec = importlib.util.spec_from_file_location("shared_f0_callback_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load callback benchmark runner: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_native_module() -> ModuleType:
    """Import the isolated abi3 build supplied by PYTHONPATH."""
    return importlib.import_module("wnn_native")


def canonical_inference_config(checkpoint: dict[str, Any]) -> dict[str, object]:
    """Extract only checkpoint options that can change served input geometry."""
    args = checkpoint["args"]
    return {
        "n_in": int(checkpoint["n_in"]),
        "arch": args.get("arch") or "gru",
        "cell": args.get("cell") or "gru",
        "enc": args.get("enc") or "linear",
        "numeric_map": args.get("numeric_map") or "none",
        "online_norm": bool(args.get("online_norm", False)),
        "heuristics": bool(args.get("heuristics", False)),
        "raw_only": bool(args.get("raw_only", False)),
        "research_features": args.get("research_features") or "none",
        "privileged_mode": args.get("privileged_mode") or "none",
        "privileged_horizon": int(args.get("privileged_horizon") or 8),
        "prop_input": args.get("prop_input"),
        "sentinel_values": args.get("sentinel_values") or "unchanged",
        "sentinel_indicators": args.get("sentinel_indicators") or "none",
        "sentinel_base_width": int(args.get("sentinel_base_width") or 0),
        "readout_present": checkpoint.get("readout") is not None,
        "readout_tilted": bool(checkpoint.get("readout_tilted", False)),
    }


def load_checkpoint_weights(
    path: Path, expected_sha256: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load the exact own checkpoint and select the predictor's first two output rows."""
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"checkpoint hash mismatch: {path}")
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    tensors = checkpoint["model"]
    weights = {
        name: np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
        for name, tensor in tensors.items()
    }
    if weights["head.2.weight"].shape[0] < 2 or weights["head.2.bias"].shape[0] < 2:
        raise ValueError("checkpoint lacks the two official first output rows")
    weights["head.2.weight"] = weights["head.2.weight"][:2].copy()
    weights["head.2.bias"] = weights["head.2.bias"][:2].copy()
    if any(not np.isfinite(value).all() for value in weights.values()):
        raise ValueError(f"checkpoint has a non-finite tensor: {path}")
    return weights, checkpoint


def checkpoint_pair_evidence() -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    """Verify exact backbone identity and the source-grade feature contract."""
    student_weights, student = load_checkpoint_weights(STUDENT_CHECKPOINT, STUDENT_SHA256)
    hft_weights, hft = load_checkpoint_weights(HFT_CHECKPOINT, HFT_SHA256)
    student_raw = student["model"]
    hft_raw = hft["model"]
    if set(student_raw) != set(hft_raw):
        raise ValueError("producer checkpoint tensor names differ")
    changed = [
        key
        for key in sorted(student_raw)
        if not np.array_equal(
            student_raw[key].detach().cpu().numpy(), hft_raw[key].detach().cpu().numpy()
        )
    ]
    if set(changed) != HEAD_KEYS:
        raise ValueError(f"unexpected changed tensors outside four head tensors: {changed}")
    student_config = canonical_inference_config(student)
    hft_config = canonical_inference_config(hft)
    if student_config != hft_config:
        raise ValueError(f"checkpoint input configurations differ: {student_config} / {hft_config}")
    if student_config["n_in"] != 211 or student_config["readout_present"]:
        raise ValueError("only the common native 211-input, no-extra-readout path is supported")
    if student["model"]["head.2.weight"].shape[0] != 31:
        raise ValueError("student producer geometry no longer matches the pinned 31-row model")
    if hft["model"]["head.2.weight"].shape[0] != 23:
        raise ValueError("HFT producer geometry no longer matches the pinned 23-row model")

    normalization = {}
    for name in ("mean", "std"):
        left = student_raw[name].detach().cpu().numpy()
        right = hft_raw[name].detach().cpu().numpy()
        if not np.array_equal(left, right):
            raise ValueError(f"normalization tensor differs: {name}")
        normalization[name] = sha256_bytes(np.ascontiguousarray(left).tobytes())

    evidence = {
        "student_checkpoint_sha256": STUDENT_SHA256,
        "hft_checkpoint_sha256": HFT_SHA256,
        "raw_changed_tensors": changed,
        "student_raw_output_shape": list(student_raw["head.2.weight"].shape),
        "hft_raw_output_shape": list(hft_raw["head.2.weight"].shape),
        "served_output_crop": "first two rows, matching frozen predict_run.py p[..., :2]",
        "served_output_shape_each": [2, 256],
        "hidden_width": int(student["args"]["hidden"]),
        "layers": int(student["args"]["layers"]),
        "normalization_sha256": normalization,
        "canonical_inference_config": student_config,
        "student_epoch": int(student["epoch"]),
        "hft_epoch": int(hft["epoch"]),
        "grade_precision": "fp16",
        "benchmark_precision_requested": "q8vnni",
        "grade_receipt_sha256": {
            "student": sha256_file(STUDENT_GRADE),
            "hft": sha256_file(HFT_GRADE),
            "fixed_half_blend": sha256_file(FIXED_BLEND_GRADE),
        },
    }
    return [student_weights, hft_weights], evidence


def make_fixture_weights(seed: int) -> dict[str, np.ndarray]:
    """Create small two-layer models for API default and rejection checks."""
    rng = np.random.default_rng(seed)
    hidden = 16
    weights: dict[str, np.ndarray] = {
        "mean": np.zeros(211, dtype=np.float32),
        "std": np.ones(211, dtype=np.float32),
        "inp.weight": rng.normal(0, 0.02, (hidden, 211)).astype(np.float32),
        "inp.bias": rng.normal(0, 0.02, hidden).astype(np.float32),
    }
    for layer in range(2):
        weights.update(
            {
                f"gru.weight_ih_l{layer}": rng.normal(0, 0.02, (3 * hidden, hidden)).astype(
                    np.float32
                ),
                f"gru.weight_hh_l{layer}": rng.normal(0, 0.02, (3 * hidden, hidden)).astype(
                    np.float32
                ),
                f"gru.bias_ih_l{layer}": rng.normal(0, 0.02, 3 * hidden).astype(np.float32),
                f"gru.bias_hh_l{layer}": rng.normal(0, 0.02, 3 * hidden).astype(np.float32),
            }
        )
    weights["head.0.weight"] = rng.normal(0, 0.02, (hidden, 2 * hidden)).astype(np.float32)
    weights["head.0.bias"] = rng.normal(0, 0.02, hidden).astype(np.float32)
    weights["head.2.weight"] = rng.normal(0, 0.02, (2, hidden)).astype(np.float32)
    weights["head.2.bias"] = rng.normal(0, 0.02, 2).astype(np.float32)
    return weights


def api_self_test(native: ModuleType) -> dict[str, object]:
    """Test default-disabled behavior, shared parity, and incompatible rejection."""
    first = make_fixture_weights(73)
    second = make_fixture_weights(74)
    head_keys = HEAD_KEYS
    for key in first.keys() - head_keys:
        second[key] = first[key].copy()
    defaults = WrappedPredictor(
        native.EnsemblePredictor([first, second], [0.5, 0.5], precision="q8vnni")
    )
    disabled = WrappedPredictor(
        native.EnsemblePredictor(
            [first, second], [0.5, 0.5], precision="q8vnni", shared_backbone=False
        )
    )
    shared = WrappedPredictor(
        native.EnsemblePredictor(
            [first, second], [0.5, 0.5], precision="q8vnni", shared_backbone=True
        )
    )
    rng = np.random.default_rng(7182)
    expected: list[np.ndarray] = []
    explicit: list[np.ndarray] = []
    actual: list[np.ndarray] = []
    for predictor in (defaults, disabled, shared):
        predictor.model.reset()
    for step in range(256):
        point = CallbackPoint(810_027, step, True, rng.normal(0, 0.1, 112).astype(np.float32))
        outputs = [predictor.predict(point) for predictor in (defaults, disabled, shared)]
        if any(output is None for output in outputs):
            raise ValueError("native self-test unexpectedly skipped a prediction")
        expected.append(np.asarray(outputs[0], dtype=np.float32))
        explicit.append(np.asarray(outputs[1], dtype=np.float32))
        actual.append(np.asarray(outputs[2], dtype=np.float32))
    default_array = np.stack(expected)
    disabled_array = np.stack(explicit)
    shared_array = np.stack(actual)
    default_disabled_equal = np.array_equal(default_array, disabled_array)
    shared_equal = np.array_equal(default_array, shared_array)
    if not default_disabled_equal or not shared_equal:
        raise ValueError("native self-test output mismatch")

    incompatible = {key: value.copy() for key, value in second.items()}
    incompatible["gru.weight_hh_l0"][0, 0] += np.float32(0.125)
    try:
        native.EnsemblePredictor(
            [first, incompatible], [0.5, 0.5], precision="q8vnni", shared_backbone=True
        )
    except ValueError as error:
        if "non-identical" not in str(error):
            raise
        rejected = True
    else:
        rejected = False
    if not rejected:
        raise ValueError("incompatible shared-backbone pair was accepted")
    return {
        "default_and_explicit_disabled_equal": default_disabled_equal,
        "shared_q8_outputs_equal": shared_equal,
        "incompatible_backbone_rejected": rejected,
        "callbacks": len(default_array),
        "precision": defaults.model.precision,
        "output_sha256": sha256_bytes(default_array.tobytes(order="C")),
    }


def run_method(
    runner: ModuleType,
    model: WrappedPredictor,
    sequences: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run two reset passes, retaining only selected outputs and callback timings."""
    expected: dict[int, np.ndarray] = {}
    hashes: dict[int, str] = {}
    per_sequence_ns: dict[int, list[int]] = {int(item["sequence_id"]): [] for item in sequences}
    trials: list[dict[str, int | float]] = []
    for repeat in range(2):
        repeat_ns = 0
        for sequence in sequences:
            elapsed, predictions = runner.run_callbacks(model, sequence)
            sequence_id = int(sequence["sequence_id"])
            if repeat == 0:
                expected[sequence_id] = predictions
                hashes[sequence_id] = runner.sha256_bytes(
                    predictions[sequence["need"]].tobytes(order="C")
                )
            elif not np.array_equal(predictions, expected[sequence_id], equal_nan=True):
                raise ValueError(f"repeat reset changed predictions for sequence {sequence_id}")
            per_sequence_ns[sequence_id].append(int(elapsed))
            repeat_ns += int(elapsed)
        trials.append(
            {
                "repeat": repeat,
                "rows": len(sequences) * 20_000,
                "callbacks": len(sequences) * 20_000,
                "callback_ns": repeat_ns,
                "callback_us_per_row": repeat_ns / (len(sequences) * 20_000) / 1000,
            }
        )
    callback_ns = sum(int(trial["callback_ns"]) for trial in trials)
    callbacks = sum(int(trial["callbacks"]) for trial in trials)
    return {
        "precision": model.model.precision,
        "storage_bytes_by_member": model.model.storage_bytes,
        "callback_ns": callback_ns,
        "callbacks": callbacks,
        "mean_callback_us_per_row": callback_ns / callbacks / 1000,
        "trials": trials,
        "per_sequence_callback_ns": per_sequence_ns,
        "prediction_sha256_by_sequence": hashes,
        "reset_replay_bitwise_identical": True,
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one frozen separate-versus-shared callback pair."""
    native = load_native_module()
    self_test = api_self_test(native)
    if sha256_file(PANEL_PATH) != PANEL_SHA256:
        raise ValueError("features-only callback panel does not match the frozen pin")
    weights, producer = checkpoint_pair_evidence()
    runner = load_callback_runner()

    allowed_cpus = sorted(os.sched_getaffinity(0))
    if not allowed_cpus:
        raise RuntimeError("process has no allowed CPU affinity")
    selected_cpu = args.cpu if args.cpu is not None else allowed_cpus[0]
    if selected_cpu not in allowed_cpus:
        raise ValueError(f"CPU {selected_cpu} is outside allowed affinity {allowed_cpus}")
    os.sched_setaffinity(0, {selected_cpu})
    sequences = runner.load_sequences(PANEL_PATH, 0, 10)
    if len(sequences) != 10 or any(item["features"].shape != (20_000, 112) for item in sequences):
        raise ValueError("panel must contain ten complete 20,000-row raw-feature sequences")
    requested_order = args.order.split("-")
    if set(requested_order) != {"separate", "shared"} or len(requested_order) != 2:
        raise ValueError("benchmark order must contain separate and shared exactly once")

    predictors: dict[str, WrappedPredictor] = {}
    for name, shared in (("separate", False), ("shared", True)):
        predictors[name] = WrappedPredictor(
            native.EnsemblePredictor(
                weights,
                [0.5, 0.5],
                fast_activations=True,
                precision="q8vnni",
                shared_backbone=shared,
            )
        )
    if any(not predictor.model.precision.startswith("q8vnni") for predictor in predictors.values()):
        raise ValueError("runtime did not select a dynamic Q8VNNI mode")

    runtime_module = Path(native.__file__).resolve()
    cgroup_before = runner.read_cgroup_snapshot()
    loadavg_before = Path("/proc/loadavg").read_text(encoding="utf-8").strip()
    start_utc = datetime.now(UTC).isoformat()
    wall_start = time.perf_counter()
    methods = {name: run_method(runner, predictors[name], sequences) for name in requested_order}
    elapsed_wall_s = time.perf_counter() - wall_start
    finish_utc = datetime.now(UTC).isoformat()
    cgroup_after = runner.read_cgroup_snapshot()
    loadavg_after = Path("/proc/loadavg").read_text(encoding="utf-8").strip()

    separate_hashes = methods["separate"]["prediction_sha256_by_sequence"]
    shared_hashes = methods["shared"]["prediction_sha256_by_sequence"]
    parity = separate_hashes == shared_hashes
    if parity:
        baseline_us = float(methods["separate"]["mean_callback_us_per_row"])
        shared_us = float(methods["shared"]["mean_callback_us_per_row"])
        runtime_gain_fraction = 1.0 - shared_us / baseline_us
    else:
        baseline_us = float(methods["separate"]["mean_callback_us_per_row"])
        shared_us = float(methods["shared"]["mean_callback_us_per_row"])
        runtime_gain_fraction = None

    separate_storage = methods["separate"]["storage_bytes_by_member"]
    shared_storage = methods["shared"]["storage_bytes_by_member"]
    separate_payload = sum(
        item["shared_backbone_step_payload"] + item["independent_head_step_payload"]
        for item in separate_storage
    )
    shared_payload = (
        shared_storage[0]["shared_backbone_step_payload"]
        + shared_storage[0]["independent_head_step_payload"]
        + shared_storage[1]["independent_head_step_payload"]
    )
    result: dict[str, Any] = {
        "schema": "wnn.shared-f0-q8vnni-callback-benchmark.r1",
        "status": "PASS_PARITY" if parity else "FAIL_PARITY",
        "pair": args.pair,
        "order": requested_order,
        "benchmark_only": True,
        "promotion": "not_requested; target-quality/native-package gates remain",
        "objective": "exact fixed 0.5/0.5 F0 output blend with one shared Q8VNNI backbone",
        "methodology": {
            "panel": str(PANEL_PATH),
            "panel_sha256": PANEL_SHA256,
            "sequences": len(sequences),
            "rows_per_sequence": 20_000,
            "callbacks_per_method": 400_000,
            "reset_passes_per_method": 2,
            "parquet_decode_and_reset_excluded_from_callback_time": True,
            "labels_loaded": False,
            "material_runtime_gain_fraction": PRACTICAL_GAIN_FRACTION,
        },
        "producer": producer,
        "self_test": self_test,
        "methods": methods,
        "comparison": {
            "prediction_hashes_equal_by_sequence": parity,
            "separate_mean_callback_us": baseline_us,
            "shared_mean_callback_us": shared_us,
            "runtime_gain_fraction": runtime_gain_fraction,
            "logical_step_weight_payload_bytes_per_callback": {
                "separate": separate_payload,
                "shared": shared_payload,
                "saved": separate_payload - shared_payload,
            },
            "logical_step_weight_payload_bytes_per_panel": {
                "separate": separate_payload * 400_000,
                "shared": shared_payload * 400_000,
                "saved": (separate_payload - shared_payload) * 400_000,
            },
            "resident_weight_bytes_by_method": {
                name: sum(
                    item["total_streamed_weights"] for item in method["storage_bytes_by_member"]
                )
                for name, method in methods.items()
            },
            "recurrent_state_bytes_by_method": {
                name: sum(item["recurrent_state"] for item in method["storage_bytes_by_member"])
                for name, method in methods.items()
            },
            "scratch_bytes_by_method": {
                name: sum(item["scratch"] for item in method["storage_bytes_by_member"])
                for name, method in methods.items()
            },
            "second_model_recurrent_state_is_still_allocated": True,
            "eligible_for_one_alternating_pair": bool(
                parity
                and runtime_gain_fraction is not None
                and runtime_gain_fraction >= PRACTICAL_GAIN_FRACTION
            ),
        },
        "runtime": {
            "start_utc": start_utc,
            "finish_utc": finish_utc,
            "wall_seconds_after_model_setup": elapsed_wall_s,
            "cpu_model": runner.cpu_model(),
            "cpu_flags": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("flags")
                ),
                "",
            ),
            "affinity_used": [selected_cpu],
            "affinity_allowed": allowed_cpus,
            "python": sys.version,
            "python_implementation": platform.python_implementation(),
            "python_soabi": sysconfig.get_config_var("SOABI"),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "native_module_path": str(runtime_module),
            "native_module_sha256": sha256_file(runtime_module),
            "native_precision_by_method": {
                name: method["precision"] for name, method in methods.items()
            },
            "cgroup_before": cgroup_before,
            "cgroup_after": cgroup_after,
            "loadavg_before": loadavg_before,
            "loadavg_after": loadavg_after,
        },
        "source": {
            "worktree": str(ROOT),
            "branch": "codex/shared-hft-backbone-20261012",
            "base_commit": "759c1450cd4852899ac518c2db1fdbe997af6ba1",
            "source_commit": os.environ.get("WNN_SOURCE_COMMIT", "unbound"),
            "runner_path": str(RUNNER_PATH),
            "runner_sha256": RUNNER_SHA256,
            "runner_utils_sha256": UTILS_SHA256,
            "benchmark_source_sha256": sha256_file(Path(__file__).resolve()),
        },
        "elapsed_wall_seconds_after_setup": elapsed_wall_s,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    """Parse one self-test or one immutable benchmark pair."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--pair", choices=("pair1", "pair2"), default="pair1")
    parser.add_argument(
        "--order", choices=("separate-shared", "shared-separate"), default="separate-shared"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    if not args.self_test and not args.preflight_only and args.output is None:
        parser.error("--output is required for a benchmark pair")
    return args


def main() -> None:
    """Run the isolated native API test or one fixed callback pair."""
    args = parse_args()
    native = load_native_module()
    if args.self_test:
        print(json.dumps(api_self_test(native), indent=2, sort_keys=True))
        return
    if args.preflight_only:
        _, producer = checkpoint_pair_evidence()
        if sha256_file(PANEL_PATH) != PANEL_SHA256:
            raise ValueError("features-only callback panel does not match the frozen pin")
        load_callback_runner()
        print(
            json.dumps(
                {
                    "status": "PASS_PREFLIGHT",
                    "producer": producer,
                    "panel": str(PANEL_PATH),
                    "panel_sha256": PANEL_SHA256,
                    "runner_sha256": RUNNER_SHA256,
                    "runner_utils_sha256": UTILS_SHA256,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    result = benchmark(args)
    print(json.dumps(result["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
