from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

OWNER_RUN = "clean_t3_f3_muonall_h384_e3_s3_r5"
CHECKPOINT_SHA256 = "d67bed97eaa16f73338ea275c316bf76c785b37f8451aa5cee8fe5daadf2f1b5"
PROVENANCE_SHA256 = "c60719d1db7d44d17e04e0379f576616a43c6ad2564fd58e192ced5453bc8da3"
SPLIT_SHA256 = "78157b3347b9e9ba6d2c075d0c53df612e67af9ecd5ee8af7fd7957e406b8343"
WRAPPER_SHA256 = "246586bfd4be3a4e3d95ac026860bb835611f09958193b75798124c9500455ab"
PREDICTOR_SHA256 = "30647bb74da64ddab4ccaa5e2916338bbb4826466b47cd68ec61593269554dd0"
SOURCE_PREFIX = Path(
    "research/campaign_2026-10-13/strategy_execution_r1/first_cohort_f3_teachers_r1/source_snapshot"
)
SPLIT_PATH = Path("research/campaign_2026-09-29/artifacts/splits/fold3.json")
PIPELINE_PATH = Path("results/f3_teacher_export_pipeline.json")
MIN_FREE_DISK_BYTES = 6_000_000_000
CHUNK_SEQUENCES = 32

STAGES = (
    {
        "name": "fit_train",
        "split": "train",
        "rows": 10539,
        "scope": "train",
        "output_name": f"{OWNER_RUN}_fit_train",
        "full_valid": False,
    },
    {
        "name": "fit_valid",
        "split": "valid",
        "rows": 1368,
        "scope": "valid",
        "output_name": f"{OWNER_RUN}_fit_valid",
        "full_valid": False,
    },
    {
        "name": "full_valid",
        "split": "valid",
        "rows": 1873,
        "scope": "full-valid-feature-only",
        "output_name": f"{OWNER_RUN}_full_valid",
        "full_valid": True,
    },
)


def sha256_file(path: Path) -> str:
    """Hash a file through bounded reads for provenance and output checks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_small_json(path: Path, max_bytes: int) -> dict[str, object]:
    """Read a bounded JSON sidecar only after checking its declared length."""
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"JSON metadata exceeds its {max_bytes}-byte limit: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_pipeline(path: Path, value: dict[str, object]) -> None:
    """Atomically replace the small per-attempt phase receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def require_frozen_source(path: Path, expected_sha256: str) -> None:
    """Reject any source drift before importing the historical predictor."""
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"frozen source hash mismatch: {path}")


def smoke_test_cleanfold_wrapper(root: Path, source_root: Path) -> None:
    """Exercise the unchanged R1 wrapper with predictor help before CUDA work."""
    wrapper = source_root / "scripts/cleanfold_exec.py"
    env = os.environ.copy()
    env["WNN_PROJECT_ROOT"] = str(root)
    env["PYTHONPATH"] = str(source_root / "src")
    completed = subprocess.run(
        [sys.executable, str(wrapper), "scripts/predict_run.py", "--help"],
        cwd=root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "frozen cleanfold_exec.py help preflight failed: " + completed.stderr[-4000:]
        )


def load_frozen_predictor(root: Path) -> ModuleType:
    """Import the exact R1 predictor once so its source-hash cache spans exports."""
    source_root = root / SOURCE_PREFIX
    wrapper = source_root / "scripts/cleanfold_exec.py"
    predictor_path = source_root / "scripts/predict_run.py"
    require_frozen_source(wrapper, WRAPPER_SHA256)
    require_frozen_source(predictor_path, PREDICTOR_SHA256)
    smoke_test_cleanfold_wrapper(root, source_root)

    os.environ["WNN_PROJECT_ROOT"] = str(root)
    os.environ["PYTHONPATH"] = str(source_root / "src")
    sys.path.insert(0, str(source_root / "scripts"))
    sys.path.insert(0, str(source_root / "src"))

    sitecustomize_path = source_root / "src/sitecustomize.py"
    site_spec = importlib.util.spec_from_file_location(
        "_frozen_r1_sitecustomize", sitecustomize_path
    )
    if site_spec is None or site_spec.loader is None:
        raise ImportError("cannot load the frozen R1 sitecustomize module")
    site_module = importlib.util.module_from_spec(site_spec)
    site_spec.loader.exec_module(site_module)

    predictor_spec = importlib.util.spec_from_file_location(
        "_frozen_r1_predict_run", predictor_path
    )
    if predictor_spec is None or predictor_spec.loader is None:
        raise ImportError("cannot load the frozen R1 predict_run module")
    predictor = importlib.util.module_from_spec(predictor_spec)
    predictor_spec.loader.exec_module(predictor)
    if root != predictor.ROOT or source_root != predictor.SOURCE_ROOT:
        raise ValueError("frozen predictor resolved an unexpected project/source root")
    return predictor


def finite_float32_array(path: Path, rows: int) -> int:
    """Check a prediction NPY's geometry and finiteness with bounded row windows."""
    import numpy as np

    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (rows, 20_000, 2) or array.dtype != np.dtype("float32"):
        raise ValueError(f"unexpected prediction geometry or dtype: {path}")
    count = 0
    for first in range(0, rows, CHUNK_SEQUENCES):
        block = array[first : first + CHUNK_SEQUENCES]
        finite = np.isfinite(block)
        if not bool(finite.all()):
            raise ValueError(f"non-finite prediction value in {path} at row {first}")
        count += int(finite.size)
    del array
    return count


def stage_argv(stage: dict[str, object]) -> list[str]:
    """Build exact causal feature-only arguments for one frozen R1 prediction."""
    args = [
        "predict_run.py",
        OWNER_RUN,
        stage["split"],
        "--epoch",
        "2",
        "--output-name",
        stage["output_name"],
        "--campaign-split-manifest",
        SPLIT_PATH.as_posix(),
        "--device",
        "cuda",
        "--inference-precision",
        "fp32",
    ]
    if stage["full_valid"]:
        args.append("--campaign-full-valid-feature-only")
    return args


def validate_stage_output(
    root: Path, stage: dict[str, object], expected_groups: list[int]
) -> dict[str, object]:
    """Bind one prediction to CP2, exact source-qualified membership and FP32."""
    array_path = root / "cache" / f"pred_{stage['output_name']}_{stage['split']}.npy"
    sidecar_path = array_path.with_suffix(".json")
    metadata = read_small_json(sidecar_path, 4_000_000)
    expected_metadata = {
        "run": OWNER_RUN,
        "split": stage["split"],
        "epoch": 2,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "split_manifest_sha256": SPLIT_SHA256,
        "inference_precision": "fp32",
        "campaign_fit_scope": stage["scope"],
        "row_groups": expected_groups,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"prediction sidecar {key} differs for {stage['name']}")
    expected_payload = stage["rows"] * 20_000 * 2 * 4
    file_bytes = array_path.stat().st_size
    if not expected_payload <= file_bytes <= expected_payload + 4096:
        raise ValueError(f"prediction NPY size differs for {stage['name']}")
    if metadata.get("prediction_sha256") != sha256_file(array_path):
        raise ValueError(f"prediction hash differs from its sidecar for {stage['name']}")
    finite_values = finite_float32_array(array_path, stage["rows"])
    return {
        "array_path": array_path.relative_to(root).as_posix(),
        "array_bytes": file_bytes,
        "array_sha256": metadata["prediction_sha256"],
        "sidecar_path": sidecar_path.relative_to(root).as_posix(),
        "sidecar_bytes": sidecar_path.stat().st_size,
        "shape": [stage["rows"], 20_000, 2],
        "dtype": "float32",
        "finite_values": finite_values,
        "row_group_count": len(expected_groups),
    }


def run_exports() -> dict:
    """Export and verify the fixed CP2 triplet, without training or scoring."""
    root = Path(os.environ.get("WNN_PROJECT_ROOT", "/kaggle/working/experiment")).resolve()
    output_receipt = root / PIPELINE_PATH
    if output_receipt.exists():
        raise FileExistsError(f"attempt output already exists: {output_receipt}")
    if shutil.disk_usage(root.parent).free < MIN_FREE_DISK_BYTES:
        raise OSError("less than six GB free for prediction arrays and runner copies")

    checkpoint_path = root / "artifacts/runs" / OWNER_RUN / "model_ep2.pt"
    if sha256_file(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError("worker CP2 bytes differ from the root-verified checkpoint")
    provenance_path = root / "artifacts/runs" / OWNER_RUN / "provenance.json"
    if sha256_file(provenance_path) != PROVENANCE_SHA256:
        raise ValueError("worker provenance bytes differ from the root-verified sidecar")
    provenance = read_small_json(provenance_path, 1_000_000)
    provenance_args = provenance.get("args")
    if (
        provenance.get("seed") != 3
        or provenance.get("selected_epoch") != 2
        or provenance.get("split_manifest_sha256") != SPLIT_SHA256
        or provenance.get("checkpoint_sha256", {}).get("model_ep2.pt") != CHECKPOINT_SHA256
        or not isinstance(provenance_args, dict)
        or provenance_args.get("name") != OWNER_RUN
        or provenance_args.get("seed") != 3
        or provenance_args.get("valid_fold") != 3
    ):
        raise ValueError("checkpoint provenance does not match the frozen F3 CP2 identity")
    predictor = load_frozen_predictor(root)
    split = predictor.load_campaign_split(root / SPLIT_PATH, root)
    if split.sha256 != SPLIT_SHA256 or (len(split.train_groups), len(split.valid_groups)) != (
        10539,
        1368,
    ):
        raise ValueError("loaded campaign split differs from the frozen F3 membership")
    provenance_fit = provenance.get("fit_row_groups")
    if (
        not isinstance(provenance_fit, dict)
        or set(provenance_fit.get("train", [])) != set(split.train_groups)
        or set(provenance_fit.get("valid", [])) != set(split.valid_groups)
        or len(provenance_fit.get("train", [])) != 10539
        or len(provenance_fit.get("valid", [])) != 1368
        or set(provenance.get("evaluation_row_groups", [])) != set(split.evaluation_groups)
        or len(provenance.get("evaluation_row_groups", [])) != 505
    ):
        raise ValueError("CP2 provenance source-qualified group membership differs from F3 split")
    import numpy as np

    with np.load(root / "cache/valid_targets.npz", allow_pickle=False) as valid_index:
        full_valid_groups = np.asarray(valid_index["row_group"], dtype=np.int64).tolist()
    expected_groups = {
        "fit_train": list(split.train_groups),
        "fit_valid": list(split.valid_groups),
        "full_valid": full_valid_groups,
    }
    if (
        len(full_valid_groups) != 1873
        or len(set(full_valid_groups)) != 1873
        or set(full_valid_groups) != set(split.valid_groups) | set(split.evaluation_groups)
        or set(split.valid_groups) & set(split.evaluation_groups)
    ):
        raise ValueError("full-valid row-group index has an unexpected row count")

    receipt = {
        "schema": "wnn.f3-teacher-cp2-export-pipeline.r1",
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "run": OWNER_RUN,
        "checkpoint_epoch": 2,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "provenance_sha256": PROVENANCE_SHA256,
        "split_manifest_sha256": SPLIT_SHA256,
        "source_wrapper_sha256": WRAPPER_SHA256,
        "source_predictor_sha256": PREDICTOR_SHA256,
        "evaluation": "prediction export only; no fitting, labels, score, or grade",
        "stages": [],
    }
    write_pipeline(output_receipt, receipt)
    try:
        for stage in STAGES:
            array_path = root / "cache" / f"pred_{stage['output_name']}_{stage['split']}.npy"
            if array_path.exists() or array_path.with_suffix(".json").exists():
                raise FileExistsError(f"refusing to overwrite {array_path}")
            stage_receipt = {
                "name": stage["name"],
                "status": "running",
                "argv": stage_argv(stage),
                "started_utc": datetime.now(UTC).isoformat(),
                "started_monotonic": time.monotonic(),
            }
            receipt["stages"].append(stage_receipt)
            write_pipeline(output_receipt, receipt)
            previous_argv = sys.argv
            try:
                sys.argv = stage_argv(stage)
                predictor.main()
            finally:
                sys.argv = previous_argv
            validated = validate_stage_output(root, stage, expected_groups[stage["name"]])
            stage_receipt.update(validated)
            stage_receipt["status"] = "complete"
            stage_receipt["completed_utc"] = datetime.now(UTC).isoformat()
            stage_receipt["wall_seconds"] = round(
                time.monotonic() - stage_receipt.pop("started_monotonic"), 3
            )
            write_pipeline(output_receipt, receipt)
            gc.collect()
            predictor.torch.cuda.empty_cache()
        receipt["status"] = "complete"
        receipt["completed_utc"] = datetime.now(UTC).isoformat()
        receipt["prediction_payload_bytes"] = sum(
            int(stage["array_bytes"]) for stage in receipt["stages"]
        )
        receipt["scientific_quality_result"] = None
        write_pipeline(output_receipt, receipt)
        return receipt
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["failed_utc"] = datetime.now(UTC).isoformat()
        receipt["failure"] = f"{type(error).__name__}: {error}"
        write_pipeline(output_receipt, receipt)
        raise


def main() -> None:
    """Run the sealed, sequential teacher export worker."""
    result = run_exports()
    print(
        json.dumps(
            {
                "status": result["status"],
                "run": result["run"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "prediction_payload_bytes": result["prediction_payload_bytes"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
