"""Verify the sealed seed-4 i16 package against native callbacks and sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / "submissions/c20261004_combo_s4_i16/submission.zip"
PACKAGE_DIR = PACKAGE.parent / "pkg"
PACKAGE_SHA256 = "c4dc258b529a52267a5e108ef0936f7919221cbdd17913270c8eb3482557f94a"
DATA = ROOT / "cache/valid_subset20.parquet"
DATA_SHA256 = "cd6a349b3655bb6f98314e5db391e69b1e1ea7f12ab18f1d15c61cefd28daa37"
PARENT_RUN = "c20261004_combo_s4_final"
HEAD_RUN = "c20261004_combo_head_s4_final"
PARENT_CHECKPOINT_SHA256 = "87bafd489d7d6e5212589154f07fef96c447777f75f949ab4dadc1a6efe4caaf"
HEAD_CHECKPOINT_SHA256 = "f331b218b5d08626f01c56fae484b895b2b57df3ca90592ad0d71a9fd53985df"
PARENT_FP32_CACHE = ROOT / "cache/pred_c20261004_combo_s4_final_fp32_valid.npy"
PARENT_FP32_SHA256 = "5917763efafa5a5e3efae725b8b70ff14765696e5cd234d2500b1036604706dc"
HEAD_FP32_CACHE = ROOT / "cache/pred_c20261004_combo_head_s4_final_fp32_valid.npy"
HEAD_FP32_SHA256 = "4f7f90a34713eb56d11abbdb98cb23cfbe8faf464650589846f22e2f6e5ce941"
PARENT_FP32_SIDECAR = PARENT_FP32_CACHE.with_suffix(".json")
HEAD_FP32_SIDECAR = HEAD_FP32_CACHE.with_suffix(".json")
PARENT_FP32_SIDECAR_SHA256 = "062e18fc2058bc2b3470c12e562a4ced177eecf82cf2bdf749b94150b9b00ae7"
HEAD_FP32_SIDECAR_SHA256 = "1fcf1c5c68112c459deea3cff48e7b45dc174be7a3fb3ef81a50512f1ede4fce"
VALID_TARGET_CACHE = ROOT / "cache/valid_targets.npz"
VALID_TARGET_CACHE_SHA256 = "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955"
EXPECTED_SEQUENCE_IDS = [
    256678,
    1990332,
    8348957,
    9579986,
    9831922,
    1200028,
    4490992,
    28417254,
    29941155,
    30273600,
    751005,
    6746551,
    9147005,
    16979262,
    23433965,
    1892371,
    2308749,
    19569293,
    24654613,
    28399078,
]
MAX_I16_F32_ERROR = 3e-4
MAX_WP_DELTA = 5e-6


def digest(path: Path) -> str:
    """Hash a file incrementally without retaining its contents."""
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check_fixed_inputs() -> dict[str, str]:
    """Validate and return immutable package, checkpoint, dataset, and cache hashes."""
    expected = {
        "package": (PACKAGE, PACKAGE_SHA256),
        "callback_dataset": (DATA, DATA_SHA256),
        "valid_target_cache": (VALID_TARGET_CACHE, VALID_TARGET_CACHE_SHA256),
        "parent_checkpoint": (
            ROOT / f"artifacts/runs/{PARENT_RUN}/model_ep3.pt",
            PARENT_CHECKPOINT_SHA256,
        ),
        "head_checkpoint": (
            ROOT / f"artifacts/runs/{HEAD_RUN}/model_ep0.pt",
            HEAD_CHECKPOINT_SHA256,
        ),
        "parent_fp32_predictions": (PARENT_FP32_CACHE, PARENT_FP32_SHA256),
        "head_fp32_predictions": (HEAD_FP32_CACHE, HEAD_FP32_SHA256),
        "parent_fp32_sidecar": (PARENT_FP32_SIDECAR, PARENT_FP32_SIDECAR_SHA256),
        "head_fp32_sidecar": (HEAD_FP32_SIDECAR, HEAD_FP32_SIDECAR_SHA256),
    }
    hashes = {}
    for name, (path, expected_hash) in expected.items():
        actual = digest(path)
        if actual != expected_hash:
            raise ValueError(f"{name} hash changed: {actual} != {expected_hash}")
        hashes[name] = actual
    return hashes


def source_hashes() -> dict[str, str]:
    """Record the checker, callback, packaging, metric, and native source closure."""
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/verify_submission.py",
        ROOT / "scripts/head_transfer_packages.py",
        ROOT / "scripts/package_submission.py",
        ROOT / "submission/solution_native.py",
        ROOT / "utils.py",
        ROOT / "src/wnn/gmetric.py",
        ROOT / "research/campaign_2026-10-02/engine/native_build_l2/b2/v2/wnn_native.abi3.so",
        ROOT / "research/campaign_2026-10-02/engine/native_build_l2/b2/avx2/wnn_native.abi3.so",
        ROOT / "research/campaign_2026-10-02/engine/native_build_l2/b2/v3/wnn_native.abi3.so",
    )
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}


def check_package_identity() -> dict[str, object]:
    """Bind ZIP contents to the expected seed-4 parent/head merge and native code."""
    with zipfile.ZipFile(PACKAGE) as archive:
        file_names = set()
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe package member path: {entry.filename}")
            if not entry.is_dir():
                file_names.add(entry.filename)
                package_file = PACKAGE_DIR.joinpath(*path.parts)
                if not package_file.is_file():
                    raise FileNotFoundError(package_file)
                if digest(package_file) != hashlib.sha256(archive.read(entry)).hexdigest():
                    raise ValueError(f"package directory differs from pinned ZIP: {entry.filename}")
        directory_names = {
            path.relative_to(PACKAGE_DIR).as_posix()
            for path in PACKAGE_DIR.rglob("*")
            if path.is_file()
        }
        if file_names != directory_names:
            raise ValueError("package ZIP and extracted package have different member sets")
        members = json.loads(archive.read("members.json"))
        precision = archive.read("precision.txt").decode("ascii").strip()
        solution_sha256 = hashlib.sha256(archive.read("solution.py")).hexdigest()
        native_hashes = {
            level: hashlib.sha256(archive.read(f"{level}/wnn_native.abi3.so")).hexdigest()
            for level in ("v2", "avx2", "v3")
        }
        weight_hash = hashlib.sha256(archive.read("weights_0.npz")).hexdigest()

    expected_source = f"{PARENT_RUN}@ep3|{HEAD_RUN}@ep0"
    if len(members) != 1 or members[0] != {
        "weights": "weights_0.npz",
        "coef": 1.0,
        "source": expected_source,
    }:
        raise ValueError(f"package members do not match the prescribed seed-4 merge: {members}")
    if precision != "i16":
        raise ValueError(f"expected i16 package precision, got {precision!r}")
    if solution_sha256 != digest(ROOT / "submission/solution_native.py"):
        raise ValueError("packaged callback differs from the reviewed native callback source")
    if PACKAGE.stat().st_size > 20_000_000:
        raise ValueError("seed-4 package exceeds the 20-MB submission cap")
    canonical_native = ROOT / "research/campaign_2026-10-02/engine/native_build_l2/b2"
    for level, actual in native_hashes.items():
        expected = digest(canonical_native / level / "wnn_native.abi3.so")
        if actual != expected:
            raise ValueError(f"packaged {level} native callback differs from canonical build")
    return {
        "members": members,
        "precision": precision,
        "weights_sha256": weight_hash,
        "solution_sha256": solution_sha256,
        "native_sha256": native_hashes,
        "zip_bytes": PACKAGE.stat().st_size,
    }


def child_json(
    arguments: list[str], log_path: Path, environment: dict[str, str], timeout: int
) -> dict[str, object]:
    """Run one bounded subprocess, preserve its log, and parse its final JSON line."""
    with log_path.open("x", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, *arguments],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )
    for line in reversed(log_path.read_text(encoding="utf-8").splitlines()):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            return record
    raise ValueError(f"no JSON result in subprocess log: {log_path}")


def check_seed4_source_merge(output: Path) -> None:
    """Check bitwise trunk identity and exact merged-head weights in a short child."""
    if output.exists():
        raise FileExistsError(output)
    import numpy as np

    sys.path.insert(0, str(ROOT / "scripts"))
    from package_submission import load_state, merged_heads

    runs = (f"{PARENT_RUN}@ep3", f"{HEAD_RUN}@ep0")
    states = [load_state(run) for run in runs]
    if set(states[0]) != set(states[1]):
        raise ValueError("seed-4 parent/head checkpoint tensor names differ")
    for name in states[0]:
        if not name.startswith("head.") and not np.array_equal(states[0][name], states[1][name]):
            raise ValueError(f"head-only source changed recurrent trunk tensor {name}")

    expected = merged_heads(list(runs))
    with np.load(PACKAGE_DIR / "weights_0.npz", allow_pickle=False) as packaged:
        if set(packaged.files) != set(expected):
            raise ValueError("packaged seed-4 merge tensor names differ from source procedure")
        for name in expected:
            if not np.array_equal(packaged[name], expected[name]):
                raise ValueError(f"packaged seed-4 merged tensor differs from source: {name}")
    result = {
        "non_head_trunk_bitwise_equal": True,
        "package_weights_match_actual_parent_head_merge_bitwise": True,
        "parent_checkpoint_sha256": digest(ROOT / f"artifacts/runs/{PARENT_RUN}/model_ep3.pt"),
        "head_checkpoint_sha256": digest(ROOT / f"artifacts/runs/{HEAD_RUN}/model_ep0.pt"),
        "weights_sha256": digest(PACKAGE_DIR / "weights_0.npz"),
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


def dataset_identity() -> dict[str, object]:
    """Check the 20 canonical full streams and return their fixed identity."""
    import numpy as np
    import pyarrow.parquet as pq

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    from utils import SEQUENCE_LENGTH, validate_sequence

    parquet = pq.ParquetFile(DATA)
    if parquet.num_row_groups != 20:
        raise ValueError("seed-4 parity requires exactly twenty full callback streams")
    sequence_ids = []
    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(
            group,
            columns=["seq_ix", "step_in_seq", "need_prediction"],
            use_threads=False,
        )
        sequence_id, need = validate_sequence(table)
        if not np.array_equal(need, np.arange(SEQUENCE_LENGTH) >= 99):
            raise ValueError(f"sequence {sequence_id} has a noncanonical first-99-row warmup")
        sequence_ids.append(sequence_id)
    if sequence_ids != EXPECTED_SEQUENCE_IDS or len(set(sequence_ids)) != 20:
        raise ValueError("seed-4 parity dataset differs from the pinned E457 stream order")
    return {
        "sha256": digest(DATA),
        "sequence_ids": sequence_ids,
        "sequences": len(sequence_ids),
        "rows_per_sequence": SEQUENCE_LENGTH,
        "warmup_rows": 99,
    }


def check_fp32_cache_sidecar(
    path: Path, run: str, epoch: int, checkpoint_sha: str, cache_sha: str
) -> dict[str, object]:
    """Validate one immutable all-VALID CUDA FP32 cache without loading it in RAM."""
    import numpy as np

    sidecar_path = path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if (
        sidecar.get("run") != run
        or sidecar.get("split") != "valid"
        or sidecar.get("epoch") != epoch
        or sidecar.get("checkpoint_sha256") != checkpoint_sha
        or sidecar.get("prediction_sha256") != cache_sha
        or sidecar.get("inference_precision") != "fp32"
        or len(sidecar.get("row_groups", [])) != 1873
        or sidecar.get("row_groups") != list(range(1873))
    ):
        raise ValueError(f"FP32 cache sidecar identity differs: {sidecar_path}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != (1873, 20_000, 2) or values.dtype != np.float32:
        raise ValueError(f"FP32 cache has wrong shape or dtype: {path}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": digest(path),
        "sidecar_sha256": digest(sidecar_path),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
    }


def load_callback_arrays(directory: Path) -> dict[str, dict[str, object]]:
    """Load the compact full-stream callback outputs for direct parity checks."""
    import numpy as np

    arrays = {}
    for key in ("i16_v3", "i16_v2", "f32_v3"):
        with np.load(directory / f"{key}.npz", allow_pickle=False) as stored:
            arrays[key] = {name: stored[name] for name in stored.files}
    return arrays


def compare_callbacks(arrays: dict[str, dict[str, object]]) -> dict[str, object]:
    """Apply exact target/mask and native precision gates on all scored rows."""
    import numpy as np

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    from utils import SEQUENCE_LENGTH
    from wnn.gmetric import reference_global_wp

    native, portable, floating = (arrays[key] for key in ("i16_v3", "i16_v2", "f32_v3"))
    expected_shape = (20, SEQUENCE_LENGTH, 2)
    for key, row in (("i16_v3", native), ("i16_v2", portable), ("f32_v3", floating)):
        if row["pred"].shape != expected_shape or row["y"].shape != expected_shape:
            raise ValueError(f"{key} callback arrays have an unexpected shape")
        if row["mask"].shape != expected_shape[:2]:
            raise ValueError(f"{key} callback mask has an unexpected shape")
    if any(
        not np.array_equal(native[key], other[key])
        for other in (portable, floating)
        for key in ("y", "mask")
    ):
        raise ValueError("callback targets or scored-row masks differ across precision runs")
    if not np.array_equal(native["pred"], portable["pred"]):
        raise ValueError("i16 v2 and v3 callbacks are not bitwise identical")
    required = np.broadcast_to(np.arange(SEQUENCE_LENGTH) >= 99, expected_shape)
    if (
        not np.isfinite(native["pred"][required]).all()
        or not np.isfinite(floating["pred"][required]).all()
    ):
        raise ValueError("a precision callback produced nonfinite required-row outputs")
    max_error = float(np.abs(native["pred"][required] - floating["pred"][required]).max())
    score_i16 = reference_global_wp(native["y"], native["pred"], native["mask"])
    score_f32 = reference_global_wp(floating["y"], floating["pred"], floating["mask"])
    wp_delta = float(score_i16 - score_f32)
    passed = max_error <= MAX_I16_F32_ERROR and abs(wp_delta) <= MAX_WP_DELTA
    if not passed:
        raise ValueError(f"i16/native-f32 parity gate failed: max={max_error}, WP={wp_delta}")
    return {
        "sequences": 20,
        "rows": 400_000,
        "v2_v3_bitwise": True,
        "targets_masks_exact": True,
        "max_i16_f32_error": max_error,
        "max_i16_f32_error_limit": MAX_I16_F32_ERROR,
        "wp_i16_f32_delta": wp_delta,
        "wp_delta_limit": MAX_WP_DELTA,
        "official_metric_id": "global_wp_clip2_v1",
        "passed": True,
    }


def fp32_cache_diagnostic(
    arrays: dict[str, dict[str, object]], dataset: dict[str, object]
) -> dict[str, object]:
    """Compare native f32 callback outputs with the actual source FP32 caches."""
    import numpy as np
    import pyarrow.parquet as pq

    from utils import SEQUENCE_LENGTH
    from wnn.gmetric import reference_global_wp

    with np.load(ROOT / "cache/valid_targets.npz", allow_pickle=False) as targets:
        seq_ids = targets["seq_ix"]
    positions = {int(sequence_id): index for index, sequence_id in enumerate(seq_ids)}
    if len(positions) != len(seq_ids):
        raise ValueError("VALID target cache has duplicate sequence IDs")
    parent = np.load(PARENT_FP32_CACHE, mmap_mode="r", allow_pickle=False)
    head = np.load(HEAD_FP32_CACHE, mmap_mode="r", allow_pickle=False)
    parquet = pq.ParquetFile(DATA)
    cache_mixture = []
    for group, sequence_id in enumerate(dataset["sequence_ids"]):
        actual_id = int(
            parquet.read_row_group(group, columns=["seq_ix"], use_threads=False)
            .column(0)[0]
            .as_py()
        )
        if actual_id != sequence_id or sequence_id not in positions:
            raise ValueError("FP32 cache alignment to the fixed callback streams failed")
        position = positions[sequence_id]
        cache_mixture.append(
            (np.asarray(parent[position], np.float64) + np.asarray(head[position], np.float64))
            / 2.0
        )
    cache_mixture_array = np.stack(cache_mixture)
    native = arrays["f32_v3"]
    required = np.broadcast_to(np.arange(SEQUENCE_LENGTH) >= 99, native["pred"].shape)
    error = np.abs(native["pred"][required] - cache_mixture_array[required])
    wp_delta = float(
        reference_global_wp(native["y"], native["pred"], native["mask"])
        - reference_global_wp(native["y"], cache_mixture_array, native["mask"])
    )
    return {
        "diagnostic_only": True,
        "source_runs": [f"{PARENT_RUN}@ep3", f"{HEAD_RUN}@ep0"],
        "max_abs_error": float(error.max()),
        "p999_abs_error": float(np.quantile(error, 0.999)),
        "wp_delta_native_f32_minus_cache_average": wp_delta,
    }


def run_trunk_check(output: Path, environment: dict[str, str]) -> dict[str, object]:
    """Run the exact source-head merge check in a separate bounded process."""
    log_path = output.with_suffix(".log")
    return child_json(
        [
            "research/campaign_2026-10-04/staging/e450/verify_seed4_package.py",
            "--check-source-merge",
            str(output),
        ],
        log_path,
        environment,
        timeout=180,
    )


def run(output: Path) -> dict[str, object]:
    """Run sealed seed-4 parity/reset checks without timing or package creation."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    fixed_hashes = check_fixed_inputs()
    package_identity = check_package_identity()
    data_identity = dataset_identity()
    parent_sidecar = check_fp32_cache_sidecar(
        PARENT_FP32_CACHE,
        PARENT_RUN,
        3,
        PARENT_CHECKPOINT_SHA256,
        PARENT_FP32_SHA256,
    )
    head_sidecar = check_fp32_cache_sidecar(
        HEAD_FP32_CACHE,
        HEAD_RUN,
        0,
        HEAD_CHECKPOINT_SHA256,
        HEAD_FP32_SHA256,
    )
    before_sources = source_hashes()

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["NUMEXPR_NUM_THREADS"] = "1"
    callback_reports = {}
    for name, precision, level in (
        ("i16_v3", "i16", "v3"),
        ("i16_v2", "i16", "v2"),
        ("f32_v3", "f32", "v3"),
    ):
        env = dict(environment, WNN_PRECISION=precision, WNN_NATIVE_LEVEL=level)
        report = child_json(
            [
                "scripts/verify_submission.py",
                str(PACKAGE),
                str(DATA),
                str(output / f"{name}.npz"),
            ],
            output / f"{name}.log",
            env,
            timeout=360,
        )
        callback_reports[name] = {
            key: value
            for key, value in report.items()
            if key not in {"load_s", "us_per_row", "eta_min_39M_rows"}
        }
        if (
            report.get("sequences") != 20
            or report.get("rows") != 400_000
            or report.get("metric_id") != "global_wp_clip2_v1"
        ):
            raise ValueError(f"{name} callback did not verify all twenty full streams")

    arrays = load_callback_arrays(output)
    parity = compare_callbacks(arrays)
    reset = child_json(
        [
            "scripts/head_transfer_packages.py",
            "--reset",
            "--package",
            str(PACKAGE_DIR),
            "--expected",
            str(output / "i16_v3.npz"),
            "--output",
            str(output / "reset_and_causality.json"),
        ],
        output / "reset_and_causality.log",
        dict(environment, WNN_PRECISION="i16", WNN_NATIVE_LEVEL="v3"),
        timeout=240,
    )
    if reset.get("reset_bitwise") is not True or reset.get("past_only_bitwise") is not True:
        raise ValueError("A-B-A reset or future-suffix causality check failed")

    trunks = run_trunk_check(output / "source_merge.json", environment)
    if (
        trunks.get("non_head_trunk_bitwise_equal") is not True
        or trunks.get("package_weights_match_actual_parent_head_merge_bitwise") is not True
    ):
        raise ValueError("seed-4 package did not preserve the actual head-only merge procedure")
    fp32_diagnostic = fp32_cache_diagnostic(arrays, data_identity)
    if check_fixed_inputs() != fixed_hashes or source_hashes() != before_sources:
        raise ValueError(
            "a pinned source, package, checkpoint, dataset, or cache changed in flight"
        )

    result = {
        "status": "parity_passed_runtime_pending",
        "package": package_identity | {"sha256": fixed_hashes["package"]},
        "fixed_input_sha256": fixed_hashes,
        "source_sha256": before_sources,
        "callback_dataset": data_identity,
        "fp32_source_caches": {"parent": parent_sidecar, "head": head_sidecar},
        "callback_reports": callback_reports,
        "callback_parity": parity,
        "source_merge": trunks,
        "reset_and_causality": reset,
        "fp32_cache_diagnostic": fp32_diagnostic,
        "public_score_measured": False,
        "runtime_measured": False,
        "manual_upload_readiness": "pending separate quiet-window runtime review",
    }
    result_path = output / "readiness.json"
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    """Run the CPU-only verifier, or its isolated actual-source merge check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-source-merge", type=Path)
    arguments = parser.parse_args()
    if (arguments.output is None) == (arguments.check_source_merge is None):
        parser.error("choose exactly one of --output or --check-source-merge")
    if arguments.check_source_merge is not None:
        check_seed4_source_merge(arguments.check_source_merge.resolve())
        return
    print(json.dumps(run(arguments.output.resolve()), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
