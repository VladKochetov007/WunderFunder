from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import tarfile
import time
from pathlib import Path

PACKET_ROOT = Path(__file__).resolve().parent
SPEC_PATH = PACKET_ROOT / "worker_spec.json"


def _write_json(path: Path, value: object) -> None:
    """Atomically save a bounded worker receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    """Hash one file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(project: Path, spec: dict[str, object]) -> dict[str, str]:
    """Verify the sealed V4 source closure and execution-only adapters."""
    pins = spec["source_pins"]
    if not isinstance(pins, dict):
        raise ValueError("worker source pin set is malformed")
    observed: dict[str, str] = {}
    for relative, expected in pins.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("worker source pin has an invalid path or digest")
        path = project / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"required source is absent or linked: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"source digest mismatch: {relative}")
        observed[relative] = actual
    manifest_path = project / str(spec["source_manifest_path"])
    if _sha256(manifest_path) != spec["source_manifest_sha256"]:
        raise ValueError("source manifest digest differs from the frozen V4 source")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("file_count") != 57 or len(manifest.get("files", [])) != 57:
        raise ValueError("V4 manifest does not contain the reviewed 57-file closure")
    for item in manifest["files"]:
        relative = str(item["path"])
        path = project / relative
        if observed.get(relative) != item.get("sha256") or path.stat().st_size != item.get("bytes"):
            raise ValueError(f"V4 source member failed manifest verification: {relative}")
    return observed


def expected_inputs(spec: dict[str, object]) -> dict[str, dict[str, object]]:
    """Return the unique frozen ten-file worker input set."""
    records = spec.get("inputs")
    if not isinstance(records, list):
        raise ValueError("worker input contract is malformed")
    expected: dict[str, dict[str, object]] = {}
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("worker input record is not an object")
        path = item.get("path")
        if not isinstance(path, str) or path in expected:
            raise ValueError("worker input path is missing or duplicated")
        expected[path] = item
    if len(expected) != 10:
        raise ValueError("seed54354 requires the exact ten-file input set")
    return expected


def validate_staged_manifest(
    raw_manifest: str,
    input_root: Path,
    expected: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """Check identities using the supervisor's remote SHA verification.

    The durable Vast supervisor hashes uploads before atomic rename. The V4
    portability gate revalidates raw Parquet inputs, and the trainer performs
    its mandatory full distillation-cache validation. This worker avoids a
    redundant 39 GB checksum pass.
    """
    items = json.loads(raw_manifest)
    if not isinstance(items, list):
        raise ValueError("WNN_INPUT_MANIFEST must be a JSON list")
    received: dict[str, dict[str, object]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("staged input record is not an object")
        path = item.get("path")
        if not isinstance(path, str) or path in received:
            raise ValueError("staged input path is missing or duplicated")
        received[path] = item
    if set(received) != set(expected):
        raise ValueError("staged input set differs from the frozen ten-file contract")

    observed: list[dict[str, object]] = []
    for relative in sorted(expected):
        frozen = expected[relative]
        staged = received[relative]
        expected_hash = frozen.get("sha256")
        expected_size = frozen.get("size_bytes")
        if staged.get("sha256") != expected_hash or staged.get("size_bytes") != expected_size:
            raise ValueError(f"supervisor identity differs from frozen input: {relative}")
        source = input_root / relative
        if not source.is_file() or source.is_symlink() or source.stat().st_size != expected_size:
            raise ValueError(f"staged input is absent or has wrong size: {relative}")
        observed.append(
            {
                "path": relative,
                "size_bytes": expected_size,
                "sha256": expected_hash,
                "verification": "vast_campaign_remote_sha256_and_local_size",
            }
        )
    return observed


def link_staged_inputs(
    project: Path, input_root: Path, expected: dict[str, dict[str, object]]
) -> None:
    """Attach staged inputs at V4 project-relative paths without replacement."""
    for relative in sorted(expected):
        source = input_root / relative
        destination = project / relative
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to replace a packet path: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source)


def stop_process(child: subprocess.Popen[bytes]) -> None:
    """Terminate only the process group created for a bounded phase."""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def gpu_bytes() -> int:
    """Read total visible GPU memory use from the owned one-GPU worker."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if not values:
        raise RuntimeError("nvidia-smi returned no GPU memory measurements")
    return sum(value * 1024**2 for value in values)


def process_tree_rss(pid: int) -> int:
    """Measure the worker phase process tree without exposing argv."""
    import psutil

    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
        return sum(process.memory_info().rss for process in processes if process.is_running())
    except psutil.NoSuchProcess:
        return 0


def run_phase(
    argv: list[str],
    phase: str,
    timeout_seconds: int,
    project: Path,
    output: Path,
    environment: dict[str, str],
    limits: dict[str, int],
    phase_records: list[dict[str, object]],
) -> dict[str, object]:
    """Run one phase with wall, process RSS, host RAM, and VRAM guards."""
    import psutil

    log_path = output / f"{phase}.log"
    started = time.monotonic()
    peak_rss = 0
    peak_vram = 0
    child: subprocess.Popen[bytes] | None = None
    try:
        with log_path.open("wb") as log:
            child = subprocess.Popen(
                argv,
                cwd=project,
                env=environment,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            while child.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= timeout_seconds:
                    raise TimeoutError(f"{phase} exceeded its {timeout_seconds}s phase cap")
                rss = process_tree_rss(child.pid)
                vram = gpu_bytes()
                peak_rss = max(peak_rss, rss)
                peak_vram = max(peak_vram, vram)
                available = int(psutil.virtual_memory().available)
                heartbeat = {
                    "phase": phase,
                    "elapsed_seconds": elapsed,
                    "rss_bytes": rss,
                    "peak_rss_bytes": peak_rss,
                    "vram_bytes": vram,
                    "peak_vram_bytes": peak_vram,
                    "host_available_bytes": available,
                }
                _write_json(output / "heartbeat.json", heartbeat)
                if rss > limits["worker_ram_bytes"]:
                    raise MemoryError(f"{phase} exceeded worker RSS cap")
                if vram > limits["worker_vram_bytes"]:
                    raise MemoryError(f"{phase} exceeded observed VRAM cap")
                if available < limits["minimum_available_ram_bytes"]:
                    raise MemoryError(f"{phase} exhausted admitted host headroom")
                time.sleep(3)
            exit_code = child.wait()
        record = {
            "phase": phase,
            "started_monotonic": started,
            "elapsed_seconds": time.monotonic() - started,
            "timeout_seconds": timeout_seconds,
            "exit_code": exit_code,
            "peak_process_tree_rss_bytes": peak_rss,
            "peak_observed_total_vram_bytes": peak_vram,
            "log_path": str(log_path.relative_to(output)),
            "log_sha256": _sha256(log_path),
        }
        if exit_code:
            raise subprocess.CalledProcessError(exit_code, argv)
        phase_records.append(record)
        return record
    except (Exception, KeyboardInterrupt) as error:
        if child is not None:
            stop_process(child)
        phase_records.append(
            {
                "phase": phase,
                "started_monotonic": started,
                "elapsed_seconds": time.monotonic() - started,
                "timeout_seconds": timeout_seconds,
                "exit_code": child.returncode if child is not None else None,
                "peak_process_tree_rss_bytes": peak_rss,
                "peak_observed_total_vram_bytes": peak_vram,
                "log_path": str(log_path.relative_to(output)),
                "log_sha256": _sha256(log_path) if log_path.is_file() else None,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        raise


def python_entry(script: str, arguments: list[str], python: str) -> list[str]:
    """Set Arrow's thread counts before importing the frozen trainer or predictor."""
    wrapper = (
        "import pyarrow,runpy,sys;"
        "pyarrow.set_cpu_count(1);pyarrow.set_io_thread_count(1);"
        "sys.path.insert(0,'scripts');"
        "script=sys.argv[1];sys.argv=[script,*sys.argv[2:]];"
        "runpy.run_path(script,run_name='__main__')"
    )
    return [python, "-c", wrapper, script, *arguments]


def verify_device(expected: dict[str, object]) -> dict[str, object]:
    """Fail closed unless the measured device matches the reviewed 3090 profile."""
    import torch

    if torch.__version__ != "2.14.0+cu130":
        raise RuntimeError(f"unexpected pinned image Torch version: {torch.__version__}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the frozen packet requires exactly one visible CUDA GPU")
    name = torch.cuda.get_device_name(0)
    capability = list(torch.cuda.get_device_capability(0))
    bf16 = bool(torch.cuda.is_bf16_supported())
    if expected["device_name_substring"].lower() not in name.lower():
        raise RuntimeError(f"unexpected GPU for the frozen profile: {name}")
    if capability != expected["device_capability"] or bf16 is not expected["cuda_bf16_supported"]:
        raise RuntimeError("GPU capability or BF16 support differs from the frozen profile")
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": name,
        "device_capability": capability,
        "cuda_bf16_supported": bf16,
        "orthogonalization_dtype": "bfloat16" if bf16 and capability[0] >= 8 else "float32",
        "total_vram_bytes": int(torch.cuda.get_device_properties(0).total_memory),
    }


def validate_prediction(
    project: Path,
    split: str,
    output_name: str,
    run_name: str,
    expected_rows: int,
    checkpoint_sha256: str,
    target_sha256: str,
) -> dict[str, object]:
    """Check full prediction shape, finiteness, target order, and fixed ancestry."""
    import numpy as np

    target_path = project / "cache" / f"{split}_targets.npz"
    prediction_path = project / "cache" / f"pred_{output_name}_{split}.npy"
    sidecar_path = prediction_path.with_suffix(".json")
    if not target_path.is_file() or not prediction_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"{split} prediction, target index, or sidecar is missing")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    with np.load(target_path, allow_pickle=False) as target_archive:
        expected_groups = np.asarray(target_archive["row_group"], dtype=np.int64)
    groups = np.asarray(sidecar.get("row_groups", []), dtype=np.int64)
    if expected_groups.shape != (expected_rows,) or not np.array_equal(groups, expected_groups):
        raise ValueError(f"{split} prediction row order differs from its target index")
    if len(np.unique(expected_groups)) != expected_rows:
        raise ValueError(f"{split} target index contains duplicate row groups")
    predictions = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    if predictions.shape != (expected_rows, 20_000, 2) or predictions.dtype != np.float32:
        raise ValueError(f"{split} prediction shape or dtype is invalid")
    for start in range(0, expected_rows, 16):
        if not np.isfinite(predictions[start : start + 16]).all():
            raise ValueError(f"{split} prediction contains nonfinite values")
    expected_file_bytes = int(predictions.offset) + int(predictions.nbytes)
    if prediction_path.stat().st_size != expected_file_bytes:
        raise ValueError(f"{split} prediction byte size is invalid")
    if (
        sidecar.get("run") != run_name
        or sidecar.get("epoch") != 3
        or sidecar.get("inference_precision") != "fp32"
        or sidecar.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError(f"{split} prediction sidecar ancestry or precision is invalid")
    prediction_sha256 = sidecar.get("prediction_sha256")
    if (
        not isinstance(prediction_sha256, str)
        or len(prediction_sha256) != 64
        or any(character not in "0123456789abcdef" for character in prediction_sha256)
    ):
        raise ValueError(f"{split} prediction sidecar hash is malformed")
    observed_prediction_sha256 = _sha256(prediction_path)
    if observed_prediction_sha256 != prediction_sha256:
        raise ValueError(f"{split} prediction file hash differs from its sidecar")
    return {
        "split": split,
        "run_name": run_name,
        "epoch": 3,
        "inference_precision": "fp32",
        "path": str(prediction_path.relative_to(project)),
        "sidecar_path": str(sidecar_path.relative_to(project)),
        "rows": expected_rows,
        "row_groups_sha256": hashlib.sha256(expected_groups.astype("<i8").tobytes()).hexdigest(),
        "target_archive_sha256": target_sha256,
        "shape": list(predictions.shape),
        "dtype": str(predictions.dtype),
        "size_bytes": prediction_path.stat().st_size,
        "prediction_sha256": observed_prediction_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "finite": True,
    }


def write_artifact_tar(project: Path, output: Path, spec: dict[str, object]) -> None:
    """Package only source-identified worker outputs, logs, and fixed checkpoints."""
    selected: list[Path] = []
    for relative in spec["artifact_project_paths"]:
        path = project / str(relative)
        if path.is_file():
            selected.append(path)
        elif path.is_dir():
            selected.extend(item for item in path.rglob("*") if item.is_file())
    for name in (
        "handoff.json",
        "profile_fit.log",
        "export_train.log",
        "export_valid.log",
        "heartbeat.json",
        "image_preflight.json",
        "dependencies.log",
    ):
        path = output / name
        if path.is_file():
            selected.append(path)
    unique = sorted(set(selected))
    for path in unique:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"refusing unsafe artifact member: {path}")
    with tarfile.open(output / "artifacts.tar", "w") as archive:
        for path in unique:
            try:
                relative = path.relative_to(project)
                arcname = f"project/{relative.as_posix()}"
            except ValueError:
                arcname = f"worker/{path.relative_to(output).as_posix()}"
            archive.add(path, arcname=arcname, recursive=False)


def prepare_environment(project: Path, spec: dict[str, object]) -> dict[str, str]:
    """Set the resource and import paths declared by the frozen worker contract."""
    environment = os.environ.copy()
    resources = spec["resources"]
    for key, value in resources["environment"].items():
        environment[str(key)] = str(value)
    environment["WNN_PROJECT_ROOT"] = str(project)
    environment["WNN_PORTABLE_DATA_MANIFEST"] = str(
        project / "research/campaign_2026-10-04/portable_inputs.json"
    )
    environment["PYTHONPATH"] = os.pathsep.join([str(project / "src"), str(project / "scripts")])
    return environment


def run_worker(spec: dict[str, object], output: Path) -> dict[str, object]:
    """Execute the real-data profile gate, one conditional fit, and two exports."""
    if not output.is_dir():
        raise FileNotFoundError("WNN_OUTPUT_DIR does not exist")
    for name in ("result.json", "artifacts.tar", "handoff.json"):
        if (output / name).exists():
            raise FileExistsError(f"refusing to overwrite prior attempt output: {name}")
    project = PACKET_ROOT / "project"
    input_root = Path(os.environ["WNN_INPUT_DIR"])
    frozen_inputs = expected_inputs(spec)
    input_receipt = validate_staged_manifest(
        os.environ["WNN_INPUT_MANIFEST"], input_root, frozen_inputs
    )
    preflight_path = output / "image_preflight.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("pinned image interpreter preflight receipt is missing")
    image_preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        image_preflight.get("status") != "passed"
        or image_preflight.get("expected_image") != spec.get("expected_image")
        or image_preflight.get("python_executable") != "/venv/main/bin/python"
        or image_preflight.get("torch_version") != "2.14.0+cu130"
        or image_preflight.get("cuda_runtime") != "13.0"
        or image_preflight.get("device_count") != 1
        or image_preflight.get("device_capability") != [8, 6]
        or image_preflight.get("cuda_bf16_supported") is not True
    ):
        raise RuntimeError("pinned image interpreter preflight did not match the V4 runtime gate")
    link_staged_inputs(project, input_root, frozen_inputs)
    source_hashes = verify_source(project, spec)
    device = verify_device(spec["expected_device"])
    environment = prepare_environment(project, spec)
    resources = spec["resources"]
    limits = {
        "worker_ram_bytes": int(resources["worker_ram_bytes"]),
        "worker_vram_bytes": int(resources["worker_vram_bytes"]),
        "minimum_available_ram_bytes": int(resources["minimum_available_ram_bytes"]),
    }
    phase_records: list[dict[str, object]] = []
    started = time.monotonic()
    result: dict[str, object] = {
        "attempt_id": os.environ.get("WNN_ATTEMPT_ID"),
        "packet_sha256": os.environ.get("WNN_PACKET_SHA256"),
        "recipe_id": spec["recipe_id"],
        "source_manifest_sha256": spec["source_manifest_sha256"],
        "source_sha256": source_hashes,
        "input_manifest": input_receipt,
        "device": device,
        "image_preflight": image_preflight,
        "image_preflight_sha256": _sha256(preflight_path),
        "grade_performed": False,
        "scientific_score": None,
        "heldout_claim": False,
        "all_supplied_data_used_for_fit": True,
        "previously_exposed_groups_included": True,
        "teacher_cache_training_scope": "clean_fold_fit_only",
        "teacher_cache_prediction_coverage": "all supplied train and valid target-index rows",
        "phase_records": phase_records,
    }
    plan_path = str(spec["profile_plan_path"])
    python = "/venv/main/bin/python"
    try:
        fit_record = run_phase(
            python_entry("scripts/h230_profile_then_fit.py", ["--plan", plan_path], python),
            "profile_fit",
            int(resources["profile_fit_timeout_seconds"]),
            project,
            output,
            environment,
            limits,
            phase_records,
        )
        gate_path = project / str(spec["profile_gate_path"])
        if not gate_path.is_file():
            raise FileNotFoundError("V4 profile-fit helper did not write its gate receipt")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        result["profile_fit_gate"] = gate
        result["profile_fit_phase"] = fit_record
        if gate.get("status") != "fit_complete":
            if gate.get("status") == "fit_not_started_projected_runtime_exceeds_cap":
                result["status"] = "deferred_insufficient_runtime"
                result["deferred_reason"] = "V4 profile projection exceeded the frozen fit cap"
                return result
            raise RuntimeError(f"V4 profile/fit did not complete: {gate.get('status')}")

        run_name = str(spec["fit_run_name"])
        output_name = str(spec["prediction_output_name"])
        checkpoint = project / "artifacts/runs" / run_name / "model_ep3.pt"
        checkpoint_hash = (
            gate.get("fit_output", {}).get("artifacts", {}).get("model_ep3.pt", {}).get("sha256")
        )
        if not checkpoint.is_file() or _sha256(checkpoint) != checkpoint_hash:
            raise ValueError("fixed epoch-3 checkpoint differs from the fit terminal receipt")
        result["fixed_epoch3_checkpoint"] = {
            "path": str(checkpoint.relative_to(project)),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_hash,
            "selected_epoch": gate.get("fit_output", {}).get("selected_epoch"),
        }

        predictions: list[dict[str, object]] = []
        for split, rows in (("train", 10_606), ("valid", 1_873)):
            command = python_entry(
                "scripts/predict_run.py",
                [
                    run_name,
                    split,
                    "--epoch",
                    "3",
                    "--output-name",
                    output_name,
                    "--inference-precision",
                    "fp32",
                ],
                python,
            )
            export_record = run_phase(
                command,
                f"export_{split}",
                int(resources["prediction_export_timeout_seconds"]),
                project,
                output,
                environment,
                limits,
                phase_records,
            )
            target_record = frozen_inputs[f"cache/{split}_targets.npz"]
            prediction = validate_prediction(
                project,
                split,
                output_name,
                run_name,
                rows,
                str(checkpoint_hash),
                str(target_record["sha256"]),
            )
            prediction["phase"] = export_record
            predictions.append(prediction)
        result["predictions"] = predictions
        result["status"] = "delivered_ungraded"
        result["elapsed_seconds"] = time.monotonic() - started
        return result
    finally:
        write_artifact_tar(project, output, spec)


def main() -> int:
    """Run the frozen H230 packet and persist terminal evidence."""
    output = Path(os.environ["WNN_OUTPUT_DIR"])
    try:
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        result = run_worker(spec, output)
        _write_json(output / "handoff.json", result)
        _write_json(output / "result.json", result)
        return (
            0
            if result.get("status")
            in {
                "delivered_ungraded",
                "deferred_insufficient_runtime",
            }
            else 1
        )
    except (Exception, KeyboardInterrupt) as error:
        result = {
            "status": "failed",
            "attempt_id": os.environ.get("WNN_ATTEMPT_ID"),
            "packet_sha256": os.environ.get("WNN_PACKET_SHA256"),
            "error": f"{type(error).__name__}: {error}",
            "grade_performed": False,
            "scientific_score": None,
        }
        _write_json(output / "handoff.json", result)
        _write_json(output / "result.json", result)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
