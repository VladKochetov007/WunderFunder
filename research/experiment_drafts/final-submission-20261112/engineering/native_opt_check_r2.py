"""Compare two isolated native libraries on the exact four-member callback."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

EXPECTED_MANIFEST_SHA256 = "e344d98a1c5277c64693d955c0e90a557f83773947a386fb865f53383a379045"
EXPECTED_SOLUTION_SHA256 = "e194efa82a83d1ebf745ec0f8027a73344c5be6e7ed2e81e4594b4cf17501c23"
ROLES = ("w18", "tiedlead_s5", "tiedlead_s6", "source4_lstm")
WIDTHS = (211, 263, 263, 112)
WEIGHT_HASHES = {
    "weights_w18.npz": "e94a02edbc51fc442ff38107f18a3a3990d07b78a46fb9924c9b5fadcc93e73d",
    "weights_tiedlead_s5.npz": "444bd6b8807e710542d0de4789a80c98659bb6862811ae0d62d696182e1d4abe",
    "weights_tiedlead_s6.npz": "76c3c9d128d0b2d341c4fe80f28d40dce2deb072743cc241df6a97279aee0d73",
    "weights_source4_lstm.npz": "49209e73995a80b1f0ed3135a7876fc0db79c6eddad89bd93b73988f2a1921d8",
}
SEQUENCE_ROWS = 20_000
WARMUP_ROWS = 99
PREFIX_ROWS = 1_000
CHILD_TIMEOUT_SECONDS = 180
WHOLE_TIMEOUT_SECONDS = 600
MEMBER_OUTPUTS = ("step_members_0.npy", "step_members_1.npy")
CALLBACK_OUTPUTS = ("callback_0.npy", "callback_1.npy")


def sha256_file(path: Path) -> str:
    """Hash one regular, non-symlink file incrementally."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_runtime(cpu: int, child: bool) -> None:
    """Require Linux Python 3.11, one requested CPU, one BLAS thread, and no GPU."""
    if sys.version_info[:2] != (3, 11) or not sys.platform.startswith("linux"):
        raise RuntimeError("native optimization check requires Linux Python 3.11")
    import resource

    available = os.sched_getaffinity(0)
    if cpu not in available:
        raise RuntimeError(f"requested CPU {cpu} is outside current affinity {sorted(available)}")
    os.sched_setaffinity(0, {cpu})
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if child:
        resource.setrlimit(resource.RLIMIT_CPU, (CHILD_TIMEOUT_SECONDS, CHILD_TIMEOUT_SECONDS))
        memory_limit = 16 * (1 << 30)
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))


def package_identity(package: Path) -> dict[str, object]:
    """Validate and hash the frozen R2 callback, manifest, and four weight files."""
    manifest_path = package / "members.json"
    solution_path = package / "solution.py"
    manifest_hash = sha256_file(manifest_path)
    solution_hash = sha256_file(solution_path)
    if manifest_hash != EXPECTED_MANIFEST_SHA256 or solution_hash != EXPECTED_SOLUTION_SHA256:
        raise ValueError("package callback or member manifest differs from frozen R2 bytes")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("members")
    if not isinstance(entries, list) or tuple(item.get("role") for item in entries) != ROLES:
        raise ValueError("package does not contain the frozen four roles in order")
    if tuple(item.get("input_width") for item in entries) != WIDTHS:
        raise ValueError("package member input widths differ from the frozen contract")
    if (
        manifest.get("precision") != "q8vnni"
        or manifest.get("fast_activations") is not True
        or manifest.get("serving_policy") != "full_mix"
        or manifest.get("active_member_indices") != [0, 1, 2, 3]
    ):
        raise ValueError("package serving policy or precision differs from full-mix R2")
    weight_records: dict[str, dict[str, str]] = {}
    for entry in entries:
        name = entry.get("weights")
        expected = WEIGHT_HASHES.get(name)
        actual = sha256_file(package / str(name)) if isinstance(name, str) else None
        if expected is None or actual != expected or entry.get("weights_sha256") != expected:
            raise ValueError(f"R2 weight bytes or manifest binding differ: {name!r}")
        weight_records[name] = {"sha256": actual, "role": str(entry["role"])}
    return {
        "manifest_sha256": manifest_hash,
        "solution_sha256": solution_hash,
        "roles": list(ROLES),
        "input_widths": list(WIDTHS),
        "precision": manifest["precision"],
        "serving_policy": manifest["serving_policy"],
        "weights": weight_records,
        "manifest": manifest,
    }


def feature_columns() -> list[str]:
    """Return the exact raw112 column order used by SequenceFile."""
    columns = [
        f"i{index}_{group}{offset}"
        for index in range(2)
        for group, count in (("p", 22), ("v", 22), ("dp", 4), ("dv", 4))
        for offset in range(count)
    ]
    return columns + [f"a{index}" for index in range(8)]


def prepare_inputs(data_path: Path, package_info: dict[str, object], output: Path) -> dict[str, object]:
    """Read only raw features and sequence IDs from validation row groups 0 and 1."""
    import numpy as np
    import pyarrow.parquet as pq

    data_hash = sha256_file(data_path.resolve(strict=True))
    manifest = package_info["manifest"]
    final_fit = manifest.get("final_fit")
    source_hashes = final_fit.get("source_sha256") if isinstance(final_fit, dict) else None
    if not isinstance(source_hashes, dict) or data_hash != source_hashes.get("valid"):
        raise ValueError("data file hash is not the exact R2 official validation file")
    reader = pq.ParquetFile(data_path)
    columns = feature_columns()
    requested = ["seq_ix", *columns]
    if any(name not in reader.schema_arrow.names for name in requested):
        raise ValueError("validation parquet lacks canonical sequence or raw112 feature columns")
    if any(name in requested for name in ("t0", "t1", "is_scored")):
        raise AssertionError("feature-only input column list unexpectedly includes labels/mask")
    if reader.num_row_groups < 2:
        raise ValueError("validation parquet must contain row groups 0 and 1")
    ids: list[int] = []
    entries: list[dict[str, object]] = []
    for group in (0, 1):
        table = reader.read_row_group(group, columns=requested, use_threads=False)
        if table.num_rows != SEQUENCE_ROWS:
            raise ValueError(f"validation row group {group} has {table.num_rows} rows")
        sequence_ids = table.column("seq_ix").to_numpy(zero_copy_only=False)
        if len(sequence_ids) != SEQUENCE_ROWS or not np.all(sequence_ids == sequence_ids[0]):
            raise ValueError(f"validation row group {group} does not have one stable seq_ix")
        raw = np.ascontiguousarray(
            np.stack(
                [table.column(name).to_numpy(zero_copy_only=False) for name in columns], axis=0
            ).astype(np.float32, copy=False).T,
            dtype=np.float32,
        )
        if raw.shape != (SEQUENCE_ROWS, 112) or not np.isfinite(raw).all():
            raise ValueError(f"validation feature group {group} is not finite raw112")
        path = output / f"raw112_{group}.npy"
        np.save(path, raw, allow_pickle=False)
        ids.append(int(sequence_ids[0]))
        entries.append({"path": path.name, "sha256": sha256_file(path), "shape": list(raw.shape)})
    if ids[0] == ids[1]:
        raise ValueError("the two validation row groups must have distinct sequence IDs")
    (output / "input_manifest.json").write_text(
        json.dumps(
            {
                "schema": "wnn.native-opt-inputs.r1",
                "data_sha256": data_hash,
                "data_path": str(data_path.resolve()),
                "sequence_ids": ids,
                "row_groups": [0, 1],
                "rows_per_sequence": SEQUENCE_ROWS,
                "feature_columns": requested,
                "targets_and_mask_read": False,
                "features": entries,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"sha256": data_hash, "sequence_ids": ids, "feature_column_count": len(columns)}


def safe_child_environment() -> dict[str, str]:
    """Inherit runtime paths needed by Python/native loading without credentials."""
    allowed = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LD_LIBRARY_PATH")
    environment = {key: os.environ[key] for key in allowed if key in os.environ}
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONNOUSERSITE": "1",
            "WNN_NATIVE_LEVEL": "v3",
        }
    )
    return environment


def run_child(args: argparse.Namespace) -> int:
    """Run one library in a fresh process and save member/callback outputs plus timings."""
    configure_runtime(args.cpu, child=True)
    import numpy as np

    package = args.package_dir.resolve(strict=True)
    library = args.library.resolve(strict=True)
    output = args.child_out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    package_info = package_identity(package)
    input_manifest = json.loads((args.inputs_dir / "input_manifest.json").read_text())
    raw_sequences = [
        np.load(args.inputs_dir / f"raw112_{index}.npy", allow_pickle=False)
        for index in range(2)
    ]
    sequence_ids = input_manifest["sequence_ids"]
    for raw in raw_sequences:
        if raw.shape != (SEQUENCE_ROWS, 112) or raw.dtype != np.float32 or not np.isfinite(raw).all():
            raise ValueError("prepared feature stream has invalid shape, dtype, or values")

    library_hash = sha256_file(library)
    if args.child_label == "candidate" and args.expected_candidate_sha and library_hash != args.expected_candidate_sha:
        raise ValueError("candidate native library differs from the requested pinned build")
    native_spec = importlib.util.spec_from_file_location("wnn_native", library)
    if native_spec is None or native_spec.loader is None:
        raise ImportError(f"cannot load native extension from {library}")
    native = importlib.util.module_from_spec(native_spec)
    sys.modules["wnn_native"] = native
    native_spec.loader.exec_module(native)
    loaded_library = Path(native.__file__).resolve()
    if loaded_library != library:
        raise RuntimeError("extension import did not resolve to the exact requested native library")

    solution_spec = importlib.util.spec_from_file_location("native_opt_solution", package / "solution.py")
    if solution_spec is None or solution_spec.loader is None:
        raise ImportError("package solution.py cannot be imported")
    solution = importlib.util.module_from_spec(solution_spec)
    sys.modules[solution_spec.name] = solution
    solution_spec.loader.exec_module(solution)
    callback = solution.PredictionModel()
    model = callback.model
    if (
        tuple(model.member_widths) != WIDTHS
        or tuple(model.active_member_indices) != (0, 1, 2, 3)
        or model.serving_policy != "full_mix"
        or model.fast_activations is not True
        or not all(str(value).startswith("q8vnni") for value in model.effective_precisions)
    ):
        raise RuntimeError("native runtime did not activate the exact full-mix Q8 member roster")

    class DataPoint:
        """Minimal callback record accepted by FinalTl3Predictor."""

        def __init__(self, seq_ix: int, step: int, row: np.ndarray) -> None:
            self.seq_ix = seq_ix
            self.step_in_seq = step
            self.need_prediction = True
            self.state = row

    points = [
        [DataPoint(sequence_ids[index], step, row) for step, row in enumerate(raw)]
        for index, raw in enumerate(raw_sequences)
    ]
    member_outputs: list[np.ndarray] = []
    member_seconds: list[float] = []
    for sequence_points in points:
        model.reset()
        output_members = np.empty((SEQUENCE_ROWS, 4, 2), dtype=np.float32)
        started = time.perf_counter()
        for step, point in enumerate(sequence_points):
            output_members[step] = np.asarray(model.step_members(point), dtype=np.float32)
        member_seconds.append(time.perf_counter() - started)
        if not np.isfinite(output_members).all():
            raise ValueError("step_members returned NaN or infinity")
        member_outputs.append(output_members)

    callback_outputs: list[np.ndarray] = []
    callback_seconds: list[float] = []
    callback_profile: list[dict[str, object]] = []
    profile_enabled = getattr(model, "q8_profile_enabled", None)
    for sequence_points in points:
        for _repeat in range(2):
            model.reset()
            reset_profile = getattr(model, "reset_q8_profile", None)
            if callable(reset_profile):
                reset_profile()
            outputs = np.empty((SEQUENCE_ROWS - WARMUP_ROWS, 2), dtype=np.float32)
            started = time.perf_counter()
            for step, point in enumerate(sequence_points):
                prediction = callback.predict(point)
                if step >= WARMUP_ROWS:
                    if prediction is None:
                        raise ValueError(f"callback omitted post-warmup prediction at row {step}")
                    outputs[step - WARMUP_ROWS] = np.asarray(prediction, dtype=np.float32)
            callback_seconds.append(time.perf_counter() - started)
            if not np.isfinite(outputs).all():
                raise ValueError("callback returned NaN or infinity after warmup")
            callback_outputs.append(outputs)
            cycle_getter = getattr(model, "q8_profile_cycles_by_member", None)
            callback_profile.append(
                {
                    "sequence_id": sequence_points[0].seq_ix,
                    "repeat": len(callback_profile) % 2,
                    "q8_profile_cycles_by_member": cycle_getter if isinstance(cycle_getter, list) else (cycle_getter() if callable(cycle_getter) else None),
                }
            )

    for index, (members, served) in enumerate(zip(member_outputs, callback_outputs[::2], strict=True)):
        blend = np.stack(
            [np.asarray(native.FinalTl3Predictor.blend_members(row), dtype=np.float32) for row in members[WARMUP_ROWS:]]
        )
        if blend.tobytes() != served.tobytes():
            raise AssertionError(f"callback blend differs bitwise from step_members outputs for sequence {index}")

    model.reset()
    replay = np.empty_like(member_outputs[0])
    for step, point in enumerate(points[0]):
        replay[step] = np.asarray(model.step_members(point), dtype=np.float32)
    replay_identical = replay.tobytes() == member_outputs[0].tobytes()
    if not replay_identical:
        raise AssertionError("reset replay changed step_members outputs")

    model.reset()
    future = np.empty((PREFIX_ROWS, 4, 2), dtype=np.float32)
    for step, point in enumerate(points[0][:PREFIX_ROWS]):
        future[step] = np.asarray(model.step_members(point), dtype=np.float32)
    changed = raw_sequences[0].copy()
    changed[PREFIX_ROWS:, 0] += np.float32(0.25)
    changed[PREFIX_ROWS:, 11] -= np.float32(0.5)
    changed[PREFIX_ROWS:, 52] += np.float32(0.3)
    model.reset()
    future_points = [DataPoint(sequence_ids[0], step, row) for step, row in enumerate(changed)]
    changed_prefix = np.empty_like(future)
    for step, point in enumerate(future_points[:PREFIX_ROWS]):
        changed_prefix[step] = np.asarray(model.step_members(point), dtype=np.float32)
    prefix_invariant = future.tobytes() == changed_prefix.tobytes()
    if not prefix_invariant:
        raise AssertionError("future-row perturbation changed the first 1,000 member outputs")

    model.reset()
    for point in points[0]:
        model.step_members(point)
    boundary = np.empty_like(member_outputs[1])
    for step, point in enumerate(points[1]):
        boundary[step] = np.asarray(model.step_members(point), dtype=np.float32)
    sequence_boundary_identical = boundary.tobytes() == member_outputs[1].tobytes()
    if not sequence_boundary_identical:
        raise AssertionError("automatic seq_ix boundary differs from independent reset outputs")

    for index in range(2):
        np.save(output / f"step_members_{index}.npy", member_outputs[index], allow_pickle=False)
        np.save(output / f"callback_{index}.npy", callback_outputs[index * 2], allow_pickle=False)
    result = {
        "schema": "wnn.native-opt-child.r1",
        "label": args.child_label,
        "python": platform.python_version(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "native_library_path": str(library),
        "native_library_sha256": library_hash,
        "package_manifest_sha256": package_info["manifest_sha256"],
        "package_solution_sha256": package_info["solution_sha256"],
        "weight_hashes": package_info["weights"],
        "effective_precisions": list(model.effective_precisions),
        "profile_enabled": bool(profile_enabled) if profile_enabled is not None else None,
        "profile_cycles_by_member_per_callback_repeat": callback_profile,
        "sequence_ids": sequence_ids,
        "step_member_shape_per_sequence": [SEQUENCE_ROWS, 4, 2],
        "callback_shape_per_sequence": [SEQUENCE_ROWS - WARMUP_ROWS, 2],
        "member_seconds_per_sequence": member_seconds,
        "callback_seconds_per_sequence_per_repeat": callback_seconds,
        "callback_median_seconds_20000_rows": statistics.median(callback_seconds),
        "callback_microseconds_per_input_row": statistics.median(callback_seconds) * 1e6 / SEQUENCE_ROWS,
        "callback_output_rows_per_sequence": SEQUENCE_ROWS - WARMUP_ROWS,
        "outputs_finite": True,
        "callback_equals_member_blend_bitwise": True,
        "reset_replay_bitwise": replay_identical,
        "future_prefix_1000_bitwise": prefix_invariant,
        "automatic_sequence_boundary_bitwise": sequence_boundary_identical,
        "files": {
            name: {"sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size}
            for name in (*MEMBER_OUTPUTS, *CALLBACK_OUTPUTS)
        },
    }
    (output / "result.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return 0


def main() -> int:
    """Prepare feature-only inputs, launch isolated libraries, and require exact parity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--baseline-library", type=Path)
    parser.add_argument("--candidate-library", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--expected-candidate-sha", default="af81c997ec734d85f90d3dff83013d35785ee543f5f1eacbf9409d07f63f60f2")
    parser.add_argument("--child-label", choices=("baseline", "candidate"))
    parser.add_argument("--library", type=Path)
    parser.add_argument("--inputs-dir", type=Path)
    parser.add_argument("--child-out", type=Path)
    args = parser.parse_args()
    if args.child_label:
        return run_child(args)
    if any(value is None for value in (args.package_dir, args.baseline_library, args.candidate_library, args.data, args.out)):
        parser.error("parent mode requires --package-dir, both libraries, --data, and --out")
    configure_runtime(args.cpu, child=False)
    started = time.monotonic()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        package = args.package_dir.resolve(strict=True)
        package_info = package_identity(package)
        inputs_dir = out / "inputs"
        inputs_dir.mkdir()
        data_info = prepare_inputs(args.data.resolve(strict=True), package_info, inputs_dir)
        common = ["--package-dir", str(package), "--inputs-dir", str(inputs_dir), "--cpu", str(args.cpu)]
        environment = safe_child_environment()
        child_receipts: dict[str, dict[str, object]] = {}
        for label, library in (("baseline", args.baseline_library), ("candidate", args.candidate_library)):
            if time.monotonic() - started >= WHOLE_TIMEOUT_SECONDS:
                raise TimeoutError("whole receiver exceeded its 600-second wall-clock bound")
            child_out = out / label
            command = [sys.executable, str(Path(__file__).resolve()), *common,
                       "--child-label", label, "--library", str(library.resolve(strict=True)),
                       "--child-out", str(child_out)]
            if label == "candidate":
                command.extend(["--expected-candidate-sha", args.expected_candidate_sha])
            completed = subprocess.run(command, check=False, capture_output=True, text=True,
                                        env=environment, timeout=CHILD_TIMEOUT_SECONDS)
            if completed.returncode != 0:
                raise RuntimeError(f"{label} child failed ({completed.returncode}): {completed.stderr[-4000:]}")
            child_receipts[label] = json.loads((child_out / "result.json").read_text())

        import numpy as np

        comparisons: dict[str, dict[str, object]] = {}
        for name in (*MEMBER_OUTPUTS, *CALLBACK_OUTPUTS):
            baseline = np.load(out / "baseline" / name, allow_pickle=False)
            candidate = np.load(out / "candidate" / name, allow_pickle=False)
            comparisons[name] = {
                "shape": list(baseline.shape),
                "dtype": str(baseline.dtype),
                "bitwise_equal": baseline.tobytes() == candidate.tobytes(),
                "baseline_sha256": sha256_file(out / "baseline" / name),
                "candidate_sha256": sha256_file(out / "candidate" / name),
            }
        bitwise = all(bool(entry["bitwise_equal"]) for entry in comparisons.values())
        baseline_time = float(child_receipts["baseline"]["callback_median_seconds_20000_rows"])
        candidate_time = float(child_receipts["candidate"]["callback_median_seconds_20000_rows"])
        time_useful = candidate_time < baseline_time
        result = {
            "schema": "wnn.native-opt-comparison.r1",
            "status": "passed" if bitwise and time_useful else "rejected",
            "bitwise_parity_passed": bitwise,
            "time_useful": time_useful,
            "candidate_speedup_vs_baseline": baseline_time / candidate_time,
            "package": package_info,
            "data": data_info,
            "input_manifest_sha256": sha256_file(inputs_dir / "input_manifest.json"),
            "library_results": child_receipts,
            "output_comparisons": comparisons,
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "elapsed_wall_seconds": time.monotonic() - started,
        }
        (out / "comparison.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
        return 0 if bitwise and time_useful else 2
    except (OSError, RuntimeError, TimeoutError, ValueError, ImportError, AssertionError) as error:
        failure = {"status": "failed", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
        (out / "failure.json").write_text(json.dumps(failure, sort_keys=True, indent=2) + "\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

