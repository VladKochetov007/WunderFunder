"""Package and verify the prescribed E457 parent/head endpoints for manual upload."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import (  # noqa: E402
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    TARGET_COLUMNS,
    DataPoint,
    validate_sequence,
)
from wnn.gmetric import reference_global_wp  # noqa: E402

CAMPAIGN = ROOT / "research/campaign_2026-10-04"
DATA = ROOT / "cache/valid_subset20.parquet"
CASES = {
    "parent": ("c20261004_combo_parent_s3_i16", "c20261004_combo_s3_final_r1@ep3"),
    "head": ("c20261004_combo_head_s3_i16", "c20261004_combo_head_s3_final@ep0"),
}
REFERENCE_SHA256 = "21f9b7368eddd64ad8ae19ae0702990a57a1806531a85148b657e5aed719dafb"


def digest(path: Path) -> str:
    """Hash artifacts without retaining their bytes."""
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def command(arguments: list[str], log: Path, environment: dict[str, str]) -> dict:
    """Run one bounded child and preserve stdout before interpreting its result."""
    with log.open("x") as stream:
        subprocess.run(
            [sys.executable, *arguments],
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=300,
        )
    for line in reversed(log.read_text().splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def source_checkpoints() -> dict[str, str]:
    """Bind both prescribed source checkpoints to hashes without loading Torch."""
    return {
        run: digest(
            ROOT / "artifacts/runs" / run.split("@ep")[0] / f"model_ep{run.split('@ep')[1]}.pt"
        )
        for _name, run in CASES.values()
    }


def check_trunks(output: Path) -> None:
    """Verify head-only provenance in a separate, short-lived Torch process."""
    from package_submission import load_state

    states = [load_state(run) for _name, run in CASES.values()]
    if set(states[0]) != set(states[1]):
        raise ValueError("Parent/head state dictionaries have different tensor names")
    for name in states[0]:
        if not name.startswith("head.") and not np.array_equal(states[0][name], states[1][name]):
            raise ValueError(f"Head-only checkpoint changed trunk tensor {name}")
    result = {"non_head_tensors_bitwise_equal": True, "checkpoint_sha256": source_checkpoints()}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def dataset_identity() -> dict:
    """Require exactly twenty canonical complete callback streams."""
    parquet = pq.ParquetFile(DATA)
    if parquet.num_row_groups != 20:
        raise ValueError("E457 requires the fixed twenty-stream validation subset")
    sequence_ids = []
    for group in range(20):
        table = parquet.read_row_group(
            group, columns=["seq_ix", "step_in_seq", "need_prediction"], use_threads=False
        )
        sequence_id, need = validate_sequence(table)
        if not np.array_equal(need, np.arange(SEQUENCE_LENGTH) >= 99):
            raise ValueError("Dataset has a noncanonical warm-up mask")
        sequence_ids.append(sequence_id)
    if len(set(sequence_ids)) != 20:
        raise ValueError("Dataset repeats sequence identifiers")
    return {"sha256": digest(DATA), "sequence_ids": sequence_ids, "rows": 20 * SEQUENCE_LENGTH}


def timing_environment() -> dict:
    """Record actual competing CUDA processes instead of assuming an idle host."""
    probe = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    return {
        "loadavg": Path("/proc/loadavg").read_text().strip(),
        "cuda_compute_processes": probe.stdout.strip(),
    }


def reset_check(package: Path, output: Path, expected: Path) -> None:
    """Replay a whole stream under A,B,A identities and a changed future suffix."""
    spec = importlib.util.spec_from_file_location("e457_callback", package / "solution.py")
    if spec is None or spec.loader is None:
        raise ImportError("Missing packaged solution loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.PredictionModel()
    table = pq.ParquetFile(DATA).read_row_group(0, use_threads=False)
    raw = np.column_stack([table[name].to_numpy() for name in FEATURE_COLUMNS]).astype(np.float32)
    need = table["need_prediction"].to_numpy().astype(bool)
    with np.load(expected, allow_pickle=False) as cache:
        reference = cache["pred"][0]
        labels = np.column_stack([table[name].to_numpy() for name in TARGET_COLUMNS]).astype(
            np.float32
        )
        scored = table["is_scored"].to_numpy().astype(bool) & need
        if not np.array_equal(cache["y"][0], labels) or not np.array_equal(
            cache["mask"][0], scored
        ):
            raise ValueError("Reset-reference cache does not match the fixed first dataset stream")
    for sequence_id in (101, 102, 101):
        values = np.zeros((SEQUENCE_LENGTH, 2), np.float32)
        for row in range(SEQUENCE_LENGTH):
            prediction = model.predict(DataPoint(sequence_id, row, bool(need[row]), raw[row]))
            if need[row]:
                values[row] = prediction
            elif prediction is not None:
                raise ValueError("Reset replay returned a warmup prediction")
        if not np.array_equal(values, reference):
            raise ValueError("Sequence reset changed required predictions")
    changed = raw.copy()
    changed[10_000:] = np.random.default_rng(457).normal(size=changed[10_000:].shape)
    for row in range(SEQUENCE_LENGTH):
        prediction = model.predict(DataPoint(103, row, bool(need[row]), changed[row]))
        if row < 10_000 and need[row] and not np.array_equal(prediction, reference[row]):
            raise ValueError("A changed future suffix altered an earlier prediction")
    output.write_text(
        json.dumps(
            {"reset_bitwise": True, "past_only_bitwise": True, "dataset_sha256": digest(DATA)}
        )
        + "\n"
    )


def prepare(output: Path) -> None:
    """Build both fixed endpoints, run callback gates, and emit readiness reports."""
    if output.exists():
        raise FileExistsError(output)
    for name, _run in CASES.values():
        if (ROOT / "submissions" / name).exists():
            raise FileExistsError(f"Refusing to overwrite submission {name}")
    output.mkdir(parents=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("WNN_PRECISION", "WNN_NATIVE_LEVEL"):
        environment.pop(name, None)
    results = {}
    reference_zip = ROOT / "submissions/c20261004_combo_s3_i16/submission.zip"
    if digest(reference_zip) != REFERENCE_SHA256:
        raise ValueError("The public-confirmed E412 timing reference archive changed")
    data_identity = dataset_identity()
    checkpoints = source_checkpoints()
    command(
        [
            "scripts/head_transfer_packages.py",
            "--check-trunks",
            "--output",
            str(output / "trunks.json"),
        ],
        output / "trunks.log",
        environment,
    )
    for case, (name, run) in CASES.items():
        directory = output / case
        directory.mkdir()
        command(
            [
                "scripts/package_submission.py",
                name,
                run + ":1.0",
                "--native-from",
                "research/campaign_2026-10-02/engine/native_build_l2/b2",
                "--solution",
                "submission/solution_native.py",
                "--precision",
                "i16",
            ],
            directory / "package.log",
            environment,
        )
        package_zip = ROOT / "submissions" / name / "submission.zip"
        arrays, callback_reports = {}, {}
        for precision, level in (("i16", "v3"), ("i16", "v2"), ("f32", "v3")):
            key = precision + "_" + level
            env = dict(environment, WNN_PRECISION=precision, WNN_NATIVE_LEVEL=level)
            destination = directory / (key + ".npz")
            callback_reports[key] = command(
                [
                    "scripts/verify_submission.py",
                    str(package_zip),
                    str(DATA),
                    str(destination),
                ],
                directory / (key + ".log"),
                env,
            )
            if (
                callback_reports[key].get("sequences") != 20
                or callback_reports[key].get("rows") != 400_000
            ):
                raise ValueError("Callback report did not cover twenty complete streams")
            with np.load(destination, allow_pickle=False) as stored:
                arrays[key] = {item: stored[item] for item in stored.files}
        quantized, portable, floating = (arrays[key] for key in ("i16_v3", "i16_v2", "f32_v3"))
        if any(
            not np.array_equal(quantized[key], other[key])
            for other in (portable, floating)
            for key in ("y", "mask")
        ):
            raise ValueError("Verification datasets or scored rows differ")
        if not np.array_equal(quantized["pred"], portable["pred"]):
            raise ValueError("i16 v2/v3 callbacks differ")
        required = np.broadcast_to(np.arange(SEQUENCE_LENGTH) >= 99, quantized["mask"].shape)
        if quantized["pred"].shape != (20, SEQUENCE_LENGTH, 2):
            raise ValueError("Callback prediction cache has an unexpected shape")
        max_error = float(np.abs(quantized["pred"][required] - floating["pred"][required]).max())
        wp_error = float(
            reference_global_wp(quantized["y"], quantized["pred"], quantized["mask"])
            - reference_global_wp(floating["y"], floating["pred"], floating["mask"])
        )
        if max_error > 3e-4 or abs(wp_error) > 5e-6:
            raise ValueError(f"{case}: i16 error gate failed: {max_error}, WP{wp_error}")
        command(
            [
                "scripts/head_transfer_packages.py",
                "--reset",
                "--package",
                str(package_zip.parent / "pkg"),
                "--expected",
                str(directory / "i16_v3.npz"),
                "--output",
                str(directory / "reset.json"),
            ],
            directory / "reset.log",
            dict(environment, WNN_PRECISION="i16", WNN_NATIVE_LEVEL="v3"),
        )
        timings = []
        for repeat in range(3):
            order = [(case, package_zip), ("equal_mixture", reference_zip)]
            if repeat % 2:
                order.reverse()
            for label, path in order:
                before = timing_environment()
                timing = command(
                    [
                        "scripts/engine_fp16_check.py",
                        "calib",
                        str(path),
                        "--data",
                        str(DATA),
                        "--seqs",
                        "3",
                    ],
                    directory / f"timing_{repeat}_{label}.log",
                    dict(environment, WNN_NATIVE_LEVEL="v3"),
                )
                timings.append(
                    dict(
                        case=label,
                        repeat=repeat,
                        environment_before=before,
                        environment_after=timing_environment(),
                        **timing,
                    )
                )
        package_bytes = package_zip.stat().st_size
        if package_bytes > 20_000_000:
            raise ValueError("Submission exceeds20MB")
        if source_checkpoints() != checkpoints or digest(DATA) != data_identity["sha256"]:
            raise ValueError("Source checkpoints or callback dataset changed during verification")
        if digest(reference_zip) != REFERENCE_SHA256:
            raise ValueError("Public timing reference changed during verification")
        results[case] = {
            "zip": str(package_zip.relative_to(ROOT)),
            "sha256": digest(package_zip),
            "bytes": package_bytes,
            "source": run,
            "checkpoint_sha256": checkpoints[run],
            "export_sha256": digest(package_zip.parent / "pkg/weights_0.npz"),
            "dataset_identity": data_identity,
            "reference_zip_sha256": REFERENCE_SHA256,
            "reference_public": 0.6655,
            "reference_public_runtime_seconds": 2267,
            "callback_reports": callback_reports,
            "v2_v3_bitwise": True,
            "max_i16_f32_error": max_error,
            "wp_i16_f32_delta": wp_error,
            "reset_and_past_only": json.loads((directory / "reset.json").read_text()),
            "timing_trials": timings,
            "ready_for_manual_upload": True,
            "score_scope": "Callback-verification subset only; public score unmeasured; no selection.",
        }
        (directory / "readiness.json").write_text(json.dumps(results[case], indent=2) + "\n")
        print(
            json.dumps(
                {key: results[case][key] for key in ("zip", "sha256", "ready_for_manual_upload")}
            ),
            flush=True,
        )
    (output / "readiness.json").write_text(json.dumps(results, indent=2) + "\n")


def main() -> None:
    """Run the bounded package experiment or its isolated reset subprocess."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--check-trunks", action="store_true")
    parser.add_argument("--package", type=Path)
    parser.add_argument("--expected", type=Path)
    arguments = parser.parse_args()
    if arguments.check_trunks:
        check_trunks(arguments.output)
    elif arguments.reset:
        if arguments.package is None or arguments.expected is None:
            parser.error("Reset stage requires --package and --expected")
        reset_check(arguments.package, arguments.output, arguments.expected)
    else:
        if 15 not in os.sched_getaffinity(0):
            raise RuntimeError("E457 requires the established CPU15 timing affinity")
        os.sched_setaffinity(0, {15})
        prepare(arguments.output)


if __name__ == "__main__":
    main()
