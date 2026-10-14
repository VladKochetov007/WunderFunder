#!/usr/bin/env python3
"""Bounded, cost-only full-callback screen for the frozen student shape ladder."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
DEFAULT_POLICY_WORKTREE = Path("/home/vlad/development/wunder/worktrees/cpu-policy-20261014")
SHAPES = ((256, 1), (192, 2), (256, 2), (512, 1), (384, 2), (640, 1), (512, 2), (448, 3))
PRECISIONS = ("f32", "q8vnni")
SEED = 20261014
ROWS = 20_000
WARMUP = 99
INPUT_WIDTH = 211
RAW_WIDTH = 112
NATIVE_SHA256 = "cd8b9692d102e5173fbe3b89a15373dcccb63382892f7621c0b2a1382f0ba362"
VALIDATION_SHA256 = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
ARCHIVE_SHA256 = "f1635d14c0e1b47848b5caeb2ec4ad9362a85800f1b2e2df80702127d3867b27"
MEMBER_ZERO_SHA256 = "45ac50f7324b8e26f37eee97b975e2643134e54ab2399077de8778f931abc2bf"
FIXTURE_SHA256 = "bfec7e22df30437429f794156f75ce6f6843c882149ec6e6e7d0c57ead588f3e"
NEED_SHA256 = "838c9b675ba71c3e713fb0279acad0eb1b09e89a84efa3bc6f84ecf91dafb76b"
SEQUENCE_ID = 256678


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha256_bytes(value: bytes | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


def array_digest(array: np.ndarray) -> str:
    return sha256_bytes(memoryview(np.ascontiguousarray(array)).cast("B"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--policy-worktree", type=Path, default=DEFAULT_POLICY_WORKTREE)
    parser.add_argument("--output-dir", type=Path, default=HERE / "runs_r1")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--hidden", type=int, choices=sorted({h for h, _ in SHAPES}))
    parser.add_argument("--layers", type=int, choices=(1, 2, 3))
    parser.add_argument("--precision", choices=PRECISIONS)
    parser.add_argument("--source-commit", default="unsealed")
    parser.add_argument("--result-path", type=Path)
    return parser.parse_args()


def standard_paths(main_root: Path, policy_worktree: Path) -> dict[str, Path]:
    campaign = main_root / "research/campaign_2026-10-13/team_models_r1"
    policy = (
        policy_worktree / "research/campaign_2026-10-14/cpu_student_frontier_r1/inference_policy_r1"
    )
    fixture = policy / "probes/input_r1"
    return {
        "inventory": campaign / "INVENTORY.json",
        "original_archive": main_root / "artifacts/team_models_20261013/xteam_h4.zip",
        "member_zero": main_root / "artifacts/team_models_20261013/xteam_h4/weights_0.npz",
        "fixture_manifest": fixture / "EXPORT_MANIFEST_R1.json",
        "fixture_features": fixture / "rows/group_0.f32",
        "native_library": policy / "packages/extracted/A_rebuilt/v3/wnn_native.abi3.so",
    }


def verify_inputs(paths: dict[str, Path]) -> dict[str, Any]:
    inventory = json.loads(paths["inventory"].read_text())
    fixture_manifest = json.loads(paths["fixture_manifest"].read_text())
    if inventory["packages"]["xteam_h4"]["sha256"] != ARCHIVE_SHA256:
        raise ValueError("xteam_h4 archive identity changed in the frozen inventory")
    if inventory["data"]["sha256"] != VALIDATION_SHA256:
        raise ValueError("validation dataset identity changed in the frozen inventory")
    if fixture_manifest["source_archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("raw feature fixture is bound to a different source archive")
    if fixture_manifest["validation_file_sha256"] != VALIDATION_SHA256:
        raise ValueError("raw feature fixture is bound to a different validation dataset")
    if fixture_manifest["rows"][0]["row_group"] != 0:
        raise ValueError("raw feature fixture is not the frozen row group 0")
    if fixture_manifest["rows"][0]["sequence_id"] != SEQUENCE_ID:
        raise ValueError("raw feature fixture sequence ID changed")
    if fixture_manifest["rows"][0]["features_sha256"] != FIXTURE_SHA256:
        raise ValueError("raw feature fixture hash changed in its export manifest")
    if fixture_manifest["rows"][0]["output_bytes"] != ROWS * RAW_WIDTH * 4:
        raise ValueError("raw feature fixture size changed")
    if sha256_file(paths["original_archive"]) != ARCHIVE_SHA256:
        raise ValueError("original xteam_h4 archive hash mismatch")
    if sha256_file(paths["member_zero"]) != MEMBER_ZERO_SHA256:
        raise ValueError("original member-0 weights hash mismatch")
    if sha256_file(paths["fixture_features"]) != FIXTURE_SHA256:
        raise ValueError("raw feature fixture hash mismatch")
    if sha256_file(paths["native_library"]) != NATIVE_SHA256:
        raise ValueError("rebuilt-A native binary hash mismatch")
    return {
        "archive_sha256": ARCHIVE_SHA256,
        "archive_bytes": paths["original_archive"].stat().st_size,
        "validation_file_sha256": VALIDATION_SHA256,
        "validation_file_bytes": inventory["data"]["bytes"],
        "validation_file_sha256_source": "campaign_2026-10-13/team_models_r1/INVENTORY.json",
        "validation_features": {
            "source_file": str(paths["fixture_features"]),
            "row_group": 0,
            "sequence_id": SEQUENCE_ID,
            "rows": ROWS,
            "features": RAW_WIDTH,
            "sha256": FIXTURE_SHA256,
            "need_mask_sha256": NEED_SHA256,
            "fixture_manifest_sha256": sha256_file(paths["fixture_manifest"]),
        },
        "normalization_tensors": {
            "source_file": str(paths["member_zero"]),
            "sha256": MEMBER_ZERO_SHA256,
        },
        "native_binary": {
            "source_file": str(paths["native_library"]),
            "sha256": NATIVE_SHA256,
        },
    }


def cgroup_snapshot() -> dict[str, Any]:
    path = None
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            path = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
            break
    if path is None or not path.is_dir():
        return {"available": False}
    fields = ("memory.current", "memory.peak", "memory.max", "memory.swap.max", "cpu.max")
    result: dict[str, Any] = {"available": True, "path": str(path)}
    for field in fields:
        target = path / field
        if target.exists():
            result[field] = target.read_text().strip()
    events = path / "memory.events"
    if events.exists():
        result["memory.events"] = {
            key: int(value)
            for key, value in (line.split() for line in events.read_text().splitlines())
        }
    cpu_stat = path / "cpu.stat"
    if cpu_stat.exists():
        result["cpu.stat"] = {
            key: int(value)
            for key, value in (line.split() for line in cpu_stat.read_text().splitlines())
        }
    return result


def synthesize_weights(
    main_root: Path, hidden: int, layers: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = main_root / "artifacts/team_models_20261013/xteam_h4/weights_0.npz"
    with np.load(source, allow_pickle=False) as archive:
        mean = np.ascontiguousarray(archive["mean"], dtype=np.float32)
        std = np.ascontiguousarray(archive["std"], dtype=np.float32)
    if mean.shape != (INPUT_WIDTH,) or std.shape != (INPUT_WIDTH,):
        raise ValueError("frozen xteam_h4 normalization tensors are not 211-wide")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("frozen normalization tensors contain invalid values")

    rng = np.random.default_rng(np.random.SeedSequence([SEED, hidden, layers]))

    def matrix(name: str, shape: tuple[int, int], fan_in: int) -> np.ndarray:
        del name  # name is kept at call sites to make the frozen architecture explicit.
        bound = 1.0 / np.sqrt(fan_in)
        return np.ascontiguousarray(rng.uniform(-bound, bound, size=shape).astype(np.float32))

    weights: dict[str, np.ndarray] = {
        "mean": mean,
        "std": std,
        "inp.weight": matrix("inp.weight", (hidden, INPUT_WIDTH), INPUT_WIDTH),
        "inp.bias": np.zeros(hidden, dtype=np.float32),
    }
    for layer in range(layers):
        for part in ("ih", "hh"):
            weights[f"gru.weight_{part}_l{layer}"] = matrix(
                f"gru.weight_{part}_l{layer}", (3 * hidden, hidden), hidden
            )
        weights[f"gru.bias_ih_l{layer}"] = np.zeros(3 * hidden, dtype=np.float32)
        weights[f"gru.bias_hh_l{layer}"] = np.zeros(3 * hidden, dtype=np.float32)
    weights["head.0.weight"] = matrix("head.0.weight", (hidden, 2 * hidden), 2 * hidden)
    weights["head.0.bias"] = np.zeros(hidden, dtype=np.float32)
    weights["head.2.weight"] = matrix("head.2.weight", (2, hidden), hidden)
    weights["head.2.bias"] = np.zeros(2, dtype=np.float32)
    weights = {key: np.ascontiguousarray(value, dtype=np.float32) for key, value in weights.items()}

    weight_hash = hashlib.sha256()
    tensor_hashes: dict[str, str] = {}
    trainable_parameters = 0
    for name in sorted(weights):
        tensor = weights[name]
        metadata = json.dumps(
            {"name": name, "dtype": tensor.dtype.str, "shape": tensor.shape},
            separators=(",", ":"),
        ).encode()
        payload = memoryview(tensor).cast("B")
        weight_hash.update(len(metadata).to_bytes(4, "little"))
        weight_hash.update(metadata)
        weight_hash.update(payload)
        tensor_hashes[name] = sha256_bytes(payload)
        if name not in {"mean", "std"}:
            trainable_parameters += tensor.size
    return weights, {
        "seed": SEED,
        "generator": "numpy.PCG64 via default_rng(SeedSequence([20261014, hidden, layers]))",
        "initialization": "independent float32 uniform matrices with +/-1/sqrt(fan_in); zero biases",
        "normalization": "exact original xteam_h4 member-0 mean/std, reused across shapes",
        "trainable_parameter_count": trainable_parameters,
        "trainable_float32_bytes": trainable_parameters * 4,
        "tensor_sha256": tensor_hashes,
        "all_tensor_sha256": weight_hash.hexdigest(),
    }


def load_fixture(path: Path) -> tuple[np.ndarray, np.ndarray, int, dict[str, str]]:
    feature_sha = sha256_file(path)
    if feature_sha != FIXTURE_SHA256:
        raise ValueError("raw feature fixture changed")
    features = np.fromfile(path, dtype="<f4").reshape(ROWS, RAW_WIDTH)
    features = np.ascontiguousarray(features, dtype=np.float32)
    need = np.arange(ROWS) >= WARMUP
    if array_digest(features) != FIXTURE_SHA256:
        raise ValueError("decoded raw features differ from the frozen export hash")
    need_sha = array_digest(need.astype(np.uint8))
    if need_sha != NEED_SHA256:
        raise ValueError("warm-up mask differs from the frozen export hash")
    return features, need, SEQUENCE_ID, {"features": feature_sha, "need": need_sha}


def replay(model: Any, sequence: int, features: np.ndarray, need: np.ndarray) -> np.ndarray:
    from utils import DataPoint

    prediction = np.zeros((ROWS, 2), dtype=np.float32)
    for step, row in enumerate(features):
        value = model.predict(DataPoint(sequence, step, bool(need[step]), row))
        if not need[step]:
            if value is not None:
                raise ValueError(f"warm-up step {step} unexpectedly returned a prediction")
        elif value is None or np.shape(value) != (2,):
            raise ValueError(f"scored step {step} did not return two values")
        else:
            prediction[step] = value
    if not np.isfinite(prediction).all():
        raise ValueError("full-sequence replay produced a nonfinite output")
    return prediction


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    if args.hidden is None or args.layers is None or args.precision is None:
        raise ValueError("worker requires --hidden, --layers and --precision")
    if args.source_commit != git_head(args.main_root):
        raise ValueError("worker source-commit argument does not match its checked-out worktree")
    paths = standard_paths(args.main_root, args.policy_worktree)
    provenance = verify_inputs(paths)
    sys.path.insert(0, str(args.main_root))
    sys.path.insert(0, str(paths["native_library"].parent))
    import wnn_native

    if sha256_file(paths["native_library"]) != NATIVE_SHA256:
        raise ValueError("native binary changed between preflight and import")
    start_synthesis = time.perf_counter()
    weights, weight_provenance = synthesize_weights(args.main_root, args.hidden, args.layers)
    synthesis_seconds = time.perf_counter() - start_synthesis
    features, need, sequence, input_hashes = load_fixture(paths["fixture_features"])

    constructor_start = time.perf_counter()
    model = wnn_native.EnsemblePredictor(
        [weights],
        [np.float32(1.0)],
        fast_activations=True,
        precision=args.precision,
        budget_s=None,
    )
    constructor_seconds = time.perf_counter() - constructor_start
    if model.precision != args.precision:
        raise AssertionError(f"selected precision is {model.precision}, expected {args.precision}")
    if model.active_members != 1 or model.coefficients != [1.0]:
        raise AssertionError("native profile silently selected a fallback or changed coefficient")
    if len(model.weight_bytes) != 1 or len(model.storage_bytes) != 1:
        raise AssertionError("native model storage report has an unexpected member count")
    storage_bytes = dict(model.storage_bytes[0])
    native_weight_bytes = int(model.weight_bytes[0])
    if native_weight_bytes <= 0 or storage_bytes.get("total_streamed_weights", 0) <= 0:
        raise AssertionError("native model reports no resident streamed weights")
    if args.precision == "q8vnni":
        if storage_bytes.get("shared_backbone_step_payload", 0) <= 0:
            raise AssertionError("selected dynamic-Q8 policy has no backbone payload")
        if storage_bytes.get("independent_head_step_payload", 0) <= 0:
            raise AssertionError("selected dynamic-Q8 policy has no readout payload")

    # Match the shipped Python callback's lifetime: native construction copies the
    # supplied arrays, after which the temporary NumPy weight dictionaries are released.
    del weights
    gc.collect()
    replay_start = time.perf_counter()
    prediction = replay(model, sequence, features, need)
    first_replay_seconds = time.perf_counter() - replay_start
    prediction_sha256 = array_digest(prediction)
    reset_start = time.perf_counter()
    model.reset()
    reset_seconds = time.perf_counter() - reset_start
    reset_replay_start = time.perf_counter()
    reset_prediction = replay(model, sequence, features, need)
    reset_replay_seconds = time.perf_counter() - reset_replay_start
    reset_prediction_sha256 = array_digest(reset_prediction)
    reset_bitwise = bool(np.array_equal(prediction, reset_prediction))
    if not reset_bitwise:
        raise AssertionError("reset replay changed complete-sequence prediction bits")

    result = {
        "kind": "cost_only_random_weight_runtime_proxy",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_commit": args.source_commit,
        "source_script": str(Path(__file__).resolve()),
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "source_tree_base_commit": "b0331300b939f08877cb526a91da66e1082a025c",
        "native_api": "EnsemblePredictor.predict(DataPoint) via rebuilt A q8vnni binary",
        "native_selected_precision": model.precision,
        "active_members": model.active_members,
        "coefficients": model.coefficients,
        "native_binary_sha256": NATIVE_SHA256,
        "shape": {
            "hidden": args.hidden,
            "layers": args.layers,
            "input_width": INPUT_WIDTH,
            "encoder": "211 -> H, GELU",
            "recurrent": "L stacked GRU layers, H each",
            "skip_readout": "concatenate final GRU state and encoder activation, 2H -> H GELU -> 2",
            "activations": "fast native GELU, project GRU gates/tanh",
        },
        "weights": weight_provenance,
        "input": {
            **provenance["validation_features"],
            "sequence_id": sequence,
            "rows": ROWS,
            "warmup_rows": WARMUP,
            "need_prediction_rows": int(need.sum()),
            "features_sha256": input_hashes["features"],
            "need_mask_sha256": input_hashes["need"],
            "labels_targets_masks_read": False,
            "validation_full_file_sha256": VALIDATION_SHA256,
        },
        "weights_source": provenance["normalization_tensors"],
        "timings_seconds": {
            "synthetic_weight_generation": synthesis_seconds,
            "constructor_quantize_pack_and_native_model_init": constructor_seconds,
            "complete_callback_replay": first_replay_seconds,
            "explicit_reset": reset_seconds,
            "complete_callback_replay_after_reset": reset_replay_seconds,
            "two_replay_mean": (first_replay_seconds + reset_replay_seconds) / 2.0,
        },
        "reset_replay_bitwise": reset_bitwise,
        "complete_prediction_sha256": prediction_sha256,
        "post_reset_prediction_sha256": reset_prediction_sha256,
        "resident_model_storage_bytes": storage_bytes,
        "native_weight_bytes": native_weight_bytes,
        "native_q8_packed_weight_bytes": None,
        "native_q8_packed_weight_bytes_note": (
            "The sealed Python getter q8_weight_bytes counts only static PackedQ8 matrices; "
            "q8vnni uses dynamic DynQ8 matrices. Their storage is included in native_weight_bytes "
            "and total_streamed_weights, while shared_backbone_step_payload and "
            "independent_head_step_payload expose logical dynamic-Q8 step payloads."
        ),
        "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "cpu": {
            "platform": platform.platform(),
            "processor": next(
                (
                    line.partition(":")[2].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                platform.processor(),
            ),
            "affinity_cpus": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "thread_environment": {
                key: os.environ.get(key)
                for key in (
                    "WNN_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "ARROW_NUM_THREADS",
                )
            },
        },
        "cgroup": cgroup_snapshot(),
        "provenance": provenance,
        "interpretation": (
            "Runtime/memory only. Synthetic deterministic weights preserve tensor shapes and "
            "use original member-0 normalization; no fitting, labels, quality score, or trained-"
            "weight package-size claim. F32 and Q8 are distinct arithmetic policies."
        ),
    }
    return result


def git_head(main_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=HERE.parents[3],
        check=True,
        capture_output=True,
        text=True,
    )
    del main_root
    return result.stdout.strip()


def run_ladder(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    paths = standard_paths(args.main_root, args.policy_worktree)
    provenance = verify_inputs(paths)
    source_commit = git_head(args.main_root)
    outputs: list[dict[str, Any]] = []
    for hidden, layers in SHAPES:
        shape_results: dict[str, Any] = {"hidden": hidden, "layers": layers, "policies": {}}
        for precision in PRECISIONS:
            name = f"h{hidden}l{layers}_{precision}"
            result_path = args.output_dir / f"{name}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--main-root",
                str(args.main_root),
                "--policy-worktree",
                str(args.policy_worktree),
                "--hidden",
                str(hidden),
                "--layers",
                str(layers),
                "--precision",
                precision,
                "--source-commit",
                source_commit,
                "--result-path",
                str(result_path),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            receipt = {
                "name": name,
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "created_utc": datetime.now(UTC).isoformat(),
            }
            receipt_path = args.output_dir / f"{name}_command.json"
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            if completed.returncode != 0:
                failure_path = args.output_dir / f"{name}_FAILURE.json"
                failure_path.write_text(json.dumps(receipt, indent=2) + "\n")
                raise subprocess.CalledProcessError(
                    completed.returncode, command, completed.stdout, completed.stderr
                )
            result = json.loads(result_path.read_text())
            if result["native_selected_precision"] != precision:
                raise AssertionError(f"{name}: requested precision was not selected")
            if not result["reset_replay_bitwise"]:
                raise AssertionError(f"{name}: reset replay failed")
            shape_results["policies"][precision] = {
                "result_file": result_path.name,
                "complete_callback_seconds": result["timings_seconds"]["complete_callback_replay"],
                "reset_replay_seconds": result["timings_seconds"][
                    "complete_callback_replay_after_reset"
                ],
                "two_replay_mean_seconds": result["timings_seconds"]["two_replay_mean"],
                "constructor_seconds": result["timings_seconds"][
                    "constructor_quantize_pack_and_native_model_init"
                ],
                "native_weight_bytes": result["native_weight_bytes"],
                "native_q8_packed_weight_bytes": result["native_q8_packed_weight_bytes"],
                "process_max_rss_kib": result["process_max_rss_kib"],
                "reset_replay_bitwise": result["reset_replay_bitwise"],
                "weight_hash": result["weights"]["all_tensor_sha256"],
                "input_hash": result["input"]["features_sha256"],
            }
            print(
                json.dumps({"completed": name, "timings_seconds": result["timings_seconds"]}),
                flush=True,
            )
        policies = shape_results["policies"]
        if policies["f32"]["weight_hash"] != policies["q8vnni"]["weight_hash"]:
            raise AssertionError(f"H{hidden}L{layers}: precision controls used different weights")
        if policies["f32"]["input_hash"] != policies["q8vnni"]["input_hash"]:
            raise AssertionError(f"H{hidden}L{layers}: precision controls used different inputs")
        outputs.append(shape_results)
    result = {
        "kind": "cost_only_student_shape_ladder_screen",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "base_commit": "b0331300b939f08877cb526a91da66e1082a025c",
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "controller_cgroup_after_all_workers": cgroup_snapshot(),
        "admission": {
            "execution_order": "shape order from frozen plan; F32 then Q8 per shape; one worker at a time",
            "CPU": "scripts/limited WNN_CPU_QUOTA=100%, WNN_THREADS=1",
            "memory_max_bytes": 1_500_000_000,
            "swap_max_bytes": 0,
            "GPU": "not used",
            "fitting": "not performed",
            "precision_policies": list(PRECISIONS),
            "W4": "not profiled; no straightforward existing precision entry point",
        },
        "frozen_provenance": provenance,
        "results": outputs,
        "limits": [
            "Synthetic random-weight runtime proxy, not trained-weight quality or candidate ranking.",
            "One fixed complete supplied-validation raw-feature sequence; no labels or score used.",
            "One Intel host and one CPU quota; results are local cost evidence, not portal runtime.",
            "No ZIP was created for synthetic weights; trained package size is not inferred.",
            "The native library is the unchanged rebuilt-A q8vnni library from the sealed R1 policy package.",
        ],
    }
    result_path = args.output_dir / "LADDER_RESULTS_R1.json"
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"result": str(result_path), "shapes": len(outputs)}, flush=True))
    return result


def main() -> None:
    args = parse_args()
    args.main_root = args.main_root.resolve()
    args.policy_worktree = args.policy_worktree.resolve()
    if args.worker:
        if args.result_path is None:
            raise ValueError("worker requires --result-path")
        result = run_worker(args)
        args.result_path.parent.mkdir(parents=True, exist_ok=True)
        args.result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {"result": str(args.result_path), "precision": result["native_selected_precision"]}
            )
        )
        return
    run_ladder(args)


if __name__ == "__main__":
    main()
