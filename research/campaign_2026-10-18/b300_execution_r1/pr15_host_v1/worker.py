"""Bounded host adapter for one exact PR15 geometry fit or profile."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from validation import (
    SOURCE_COMMIT,
    parse_profile_events,
    read_json,
    sha256_file,
    validate_fit_artifacts,
    validate_prediction_and_score,
    validate_profile_events,
    verify_score_support,
    verify_source_tree,
)
from workspace import prepare_workspace

CONTROL_VALUES = {"all", "moments_only", "dictionary_only", "none"}
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def parse_args() -> argparse.Namespace:
    """Parse one cell request from the root-owned B300 controller."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--control", required=True, choices=sorted(CONTROL_VALUES))
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--cpus", required=True, help="comma-separated IDs or ranges")
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("profile", "fit"))
    return parser.parse_args()


def _cpu_ids(spec: str) -> set[int]:
    """Parse a nonempty Linux CPU list such as ``0-1,8``."""
    result: set[int] = set()
    for item in spec.split(","):
        parts = item.strip().split("-", 1)
        if not parts[0].isdigit() or (len(parts) == 2 and not parts[1].isdigit()):
            raise ValueError(f"invalid CPU affinity list: {spec}")
        low, high = int(parts[0]), int(parts[-1])
        if low < 0 or high < low:
            raise ValueError(f"invalid CPU range: {item}")
        result.update(range(low, high + 1))
    if not result:
        raise ValueError("CPU affinity list is empty")
    return result


def _apply_affinity(cpus: set[int]) -> list[int]:
    """Apply and report this worker's declared CPU affinity before child imports."""
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise OSError("PR15 host worker requires Linux CPU affinity controls")
    allowed = set(os.sched_getaffinity(0))
    if not cpus <= allowed:
        raise ValueError(
            f"requested CPUs are outside controller affinity: {sorted(cpus - allowed)}"
        )
    os.sched_setaffinity(0, cpus)
    observed = sorted(os.sched_getaffinity(0))
    if observed != sorted(cpus):
        raise OSError("kernel did not apply the requested PR15 CPU affinity")
    return observed


def _gpu_row(gpu_uuid: str) -> dict[str, str]:
    """Require the named host UUID to identify one NVIDIA B300 device."""
    command = [
        "nvidia-smi",
        "--query-gpu=uuid,name,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    rows = list(csv.reader(output.splitlines(), skipinitialspace=True))
    matches = [row for row in rows if len(row) >= 4 and row[0].strip() == gpu_uuid]
    if len(matches) != 1 or "B300" not in matches[0][1]:
        raise ValueError(
            f"GPU UUID does not identify exactly one NVIDIA B300: {gpu_uuid}"
        )
    return {
        "uuid": matches[0][0].strip(),
        "name": matches[0][1].strip(),
        "memory_free_mib": matches[0][2].strip(),
        "memory_total_mib": matches[0][3].strip(),
    }


def _load_common(source_root: Path) -> Any:
    """Import only PR15's frozen command builder, without running its host launcher."""
    path = source_root / "scripts" / "geometry_b300" / "common.py"
    sys.path.insert(0, str(path.parent))
    module_spec = importlib.util.spec_from_file_location("pr15_geometry_common", path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot import frozen command builder: {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def _replace_flag(argv: list[str], flag: str, value: str) -> list[str]:
    """Replace one exact flag value while preserving all other recipe arguments."""
    result = list(argv)
    positions = [index for index, token in enumerate(result) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(result):
        raise ValueError(f"frozen command has no unique {flag} value")
    result[positions[0] + 1] = value
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically persist one receipt without replacing an existing attempt."""
    if path.exists():
        raise FileExistsError(f"refusing to replace receipt: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _tee_process(
    command: list[str], env: dict[str, str], cwd: Path, log_path: Path
) -> dict[str, Any]:
    """Run one phase in the inherited worker group and retain/stream its full log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("x", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
        log.flush()
        os.fsync(log.fileno())
    return {
        "command": command,
        "exit_code": return_code,
        "started_epoch": started,
        "finished_epoch": time.time(),
        "elapsed_seconds": time.time() - started,
        "log": str(log_path),
        "log_sha256": sha256_file(log_path),
    }


def _phase_environment(
    base: dict[str, str], *, work: Path, gpu_uuid: str, phase: str
) -> dict[str, str]:
    """Set bounded disk, loader, CUDA and per-phase envelope locations."""
    phase_root = work / "envelope" / phase
    scratch = work / "scratch"
    scratch.mkdir(exist_ok=True)
    env = base.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu_uuid,
            "WNN_GPU_ID": gpu_uuid,
            "WNN_PROJECT_ROOT": str(work),
            "WNN_RUN_DIR": str(phase_root),
            "WNN_LOADER_WORKERS": "0",
            "WNN_LOADER_RAM_GB": "1",
            "LOADER_RAM_GB": "1",
            "WORKERS": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "NUMBA_CACHE_DIR": str(scratch / "numba"),
            "TORCHINDUCTOR_CACHE_DIR": str(scratch / "torchinductor"),
            "TMPDIR": str(scratch),
            "TMP": str(scratch),
            "TEMP": str(scratch),
            "NVIDIA_TF32_OVERRIDE": "0",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    (scratch / "numba").mkdir(exist_ok=True)
    (scratch / "torchinductor").mkdir(exist_ok=True)
    return env


def _run_gpu_actor(
    *,
    package: Path,
    work: Path,
    request: Path,
    actor: str,
    phase: str,
    base_env: dict[str, str],
    gpu_uuid: str,
    source_root: Path,
) -> dict[str, Any]:
    """Run one whitelisted actor through the frozen six-GiB B300 wrapper."""
    if actor not in {"train_cell.py", "export_cell.py"}:
        raise ValueError(f"unsupported PR15 actor: {actor}")
    command = [
        sys.executable,
        "-u",
        str(package / "gpu_entrypoint.py"),
        str(package / actor),
        "--request",
        str(request),
    ]
    env = _phase_environment(base_env, work=work, gpu_uuid=gpu_uuid, phase=phase)
    log = work / "logs" / f"{phase}.log"
    record = _tee_process(command, env, source_root, log)
    if record["exit_code"] != 0:
        raise subprocess.CalledProcessError(record["exit_code"], command)
    envelope_path = (
        work / "envelope" / phase / f"{Path(actor).stem}_resource_receipt.json"
    )
    envelope = read_json(envelope_path)
    if (
        envelope.get("status") != "actor_returned"
        or envelope.get("gpu") is None
        or "B300" not in envelope.get("gpu", "")
        or envelope.get("visible_devices") != gpu_uuid
        or envelope.get("allocator_budget_bytes") != 6 * 1024**3
    ):
        raise ValueError(
            f"GPU process envelope failed identity/resource validation: {envelope_path}"
        )
    record["resource_receipt"] = str(envelope_path)
    record["resource_receipt_sha256"] = sha256_file(envelope_path)
    return record


def _request_path(work: Path) -> Path:
    """Return the one fresh request path shared with the phase actors."""
    path = work / "worker_request.json"
    if path.exists():
        raise FileExistsError(path)
    return path


def _make_request(
    path: Path,
    *,
    mode: str,
    name: str,
    source_root: Path,
    work: Path,
    train_argv: list[str],
    predict_argv: list[str] | None,
) -> None:
    """Persist exact canonical trainer and predictor arguments for this worker."""
    payload: dict[str, Any] = {
        "schema": "pr15-host-worker-request.v1",
        "mode": mode,
        "name": name,
        "source_root": str(source_root),
        "work": str(work),
        "train_argv": train_argv,
    }
    if predict_argv is not None:
        payload["predict_argv"] = predict_argv
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _run_score(
    work: Path, name: str, split_path: Path, base: dict[str, str]
) -> dict[str, Any]:
    """Run the frozen PR15 global_wp_clip2_v1 scorer on CPU-only workspace outputs."""
    command = [
        sys.executable,
        "-u",
        str(work / "scripts" / "score_run.py"),
        "--run",
        name,
        "--split",
        str(split_path),
        "--epoch",
        "2",
    ]
    env = base.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "WNN_PROJECT_ROOT": str(work),
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TMPDIR": str(work / "scratch"),
        }
    )
    record = _tee_process(command, env, work, work / "logs" / "score.log")
    if record["exit_code"] != 0:
        raise subprocess.CalledProcessError(record["exit_code"], command)
    if not (work / "artifacts" / "runs" / name / "score.json").is_file():
        raise FileNotFoundError("PR15 scorer returned without its bound score.json")
    return record


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the frozen package, then execute one profile or one complete PR15 cell."""
    if not NAME_PATTERN.fullmatch(args.name):
        raise ValueError("name must contain only safe filename characters")
    if not 0 <= args.fold < 4:
        raise ValueError("invalid PR15 fold")
    maximum_seed = 1008 if args.fold < 2 else 1007
    if not 1001 <= args.seed <= maximum_seed:
        raise ValueError("seed is outside the original PR15 CP30 fold matrix")
    if args.mode == "profile" and args.control != "none":
        raise ValueError(
            "PR15 throughput profile must exercise the full 271-wide geometry input"
        )
    root = args.root.resolve(strict=True)
    source_root = root / "runtime" / "pr15-source-e238-v1"
    source_identity = verify_source_tree(
        source_root, Path(__file__).with_name("source_manifest.json")
    )
    package = Path(__file__).resolve().parent
    scorer_support = verify_score_support(package) if args.mode == "fit" else None
    cpus = _cpu_ids(args.cpus)
    affinity = _apply_affinity(cpus)
    gpu = _gpu_row(args.gpu)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "", args.gpu):
        raise ValueError(
            f"controller CUDA_VISIBLE_DEVICES differs from --gpu: {visible}"
        )
    work = args.work.absolute()
    workspace = prepare_workspace(
        root,
        work,
        args.fold,
        package_root=package,
        include_scorer=args.mode == "fit",
    )
    split_path = Path(workspace["split_manifest"])
    split = read_json(split_path)
    split_hash = str(workspace["split_manifest_sha256"])
    common = _load_common(source_root)
    train_command = common.train_command(args.name, args.fold, args.seed, args.control)
    if train_command[2] != "scripts/train_rnn.py":
        raise ValueError("frozen PR15 builder no longer points to scripts/train_rnn.py")
    train_argv = _replace_flag(train_command[3:], "--split-manifest", str(split_path))
    if args.mode == "profile":
        train_argv.remove("--save-training-state")
        train_argv.extend(("--profile-steps", "8", "--profile-warmup", "2"))
    predict_command = common.prediction_command(args.name, args.fold)
    predict_argv = _replace_flag(
        predict_command[3:], "--campaign-split-manifest", str(split_path)
    )
    request_path = _request_path(work)
    _make_request(
        request_path,
        mode=args.mode,
        name=args.name,
        source_root=source_root,
        work=work,
        train_argv=train_argv,
        predict_argv=predict_argv,
    )
    base_env = os.environ.copy()
    base_env.update(
        {
            "WNN_PROJECT_ROOT": str(work),
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "WNN_GPU_ID": args.gpu,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    identity = {
        "name": args.name,
        "mode": args.mode,
        "fold": args.fold,
        "seed": args.seed,
        "control": args.control,
        "gpu": gpu,
        "cpu_affinity": affinity,
        "source": source_identity,
        "workspace": workspace,
        "request_path": str(request_path),
        "request_sha256": sha256_file(request_path),
        "split_hash": split_hash,
    }
    if scorer_support is not None:
        identity["scorer_support"] = scorer_support
    phase = _run_gpu_actor(
        package=package,
        work=work,
        request=request_path,
        actor="train_cell.py",
        phase="profile" if args.mode == "profile" else "fit",
        base_env=base_env,
        gpu_uuid=args.gpu,
        source_root=source_root,
    )
    if args.mode == "profile":
        events = parse_profile_events(
            Path(phase["log"]).read_text(encoding="utf-8", errors="replace")
        )
        validate_profile_events(events)
        profile = {
            "schema": "pr15-host-profile-receipt.v1",
            "status": "profile_passed",
            "source_commit": SOURCE_COMMIT,
            "source_manifest_sha256": source_identity["manifest_sha256"],
            "name": args.name,
            "fold": args.fold,
            "seed": args.seed,
            "control": "none",
            "shape": {
                "hidden": 512,
                "layers": 1,
                "batch": 8,
                "chunk_rows": 2000,
                "input_width": 271,
                "profile_warmup_batches": 2,
                "profile_measured_batches": 8,
                "expected_optimizer_updates": 100,
            },
            "gpu": gpu,
            "cpu_affinity": affinity,
            "profile_events": events,
            "phase": phase,
            "completed_epoch": time.time(),
        }
        _write_json(work / "profile_receipt.json", profile)
        return {
            **identity,
            "status": "profile_passed",
            "profile_receipt": str(work / "profile_receipt.json"),
            "profile_receipt_sha256": sha256_file(work / "profile_receipt.json"),
        }

    fit = validate_fit_artifacts(work, args.name)
    _write_json(
        work / "fit_receipt.json",
        {
            "schema": "pr15-host-fit-receipt.v1",
            "status": "fit_completed",
            **identity,
            "phase": phase,
            "fit_artifacts": fit,
            "training_recipe": {
                "family": "causal_geometry_discrete",
                "feature_control": args.control,
                "model": "H512x1 GRU",
                "input_width": 271,
                "batch": 8,
                "chunk_rows": 2000,
                "epochs": 3,
                "fixed_checkpoint_epoch": 2,
                "precision": "fp32",
                "optimizer": "AdamW",
                "learning_rate": 0.001,
                "weight_decay": 0.01,
                "scheduler": "OneCycle",
                "gradient_clip": 1.0,
            },
        },
    )
    export = _run_gpu_actor(
        package=package,
        work=work,
        request=request_path,
        actor="export_cell.py",
        phase="export",
        base_env=base_env,
        gpu_uuid=args.gpu,
        source_root=source_root,
    )
    prediction = work / "cache" / f"pred_{args.name}_valid.npy"
    metadata = prediction.with_suffix(".json")
    if not prediction.is_file() or not metadata.is_file():
        raise FileNotFoundError(
            "PR15 evaluation export did not produce prediction and sidecar"
        )
    prediction_meta = read_json(metadata)
    if (
        prediction_meta.get("epoch") != 2
        or prediction_meta.get("campaign_export_scope") != "evaluation"
        or prediction_meta.get("campaign_split_manifest_sha256") != split_hash
        or prediction_meta.get("row_groups") != split["evaluation"]["valid"]
    ):
        raise ValueError(
            "PR15 evaluation export does not bind fixed epoch-2 fold membership"
        )
    if prediction_meta.get("prediction_sha256") != sha256_file(prediction):
        raise ValueError("PR15 prediction sidecar hash does not match the output bytes")
    _write_json(
        work / "export_receipt.json",
        {
            "schema": "pr15-host-export-receipt.v1",
            "status": "export_completed",
            "phase": export,
            "prediction": str(prediction),
            "prediction_sha256": sha256_file(prediction),
            "sidecar": str(metadata),
            "sidecar_sha256": sha256_file(metadata),
            "evaluation_groups": prediction_meta["row_groups"],
            "epoch": 2,
        },
    )
    score_phase = _run_score(work, args.name, split_path, base_env)
    score = validate_prediction_and_score(work, args.name, split, split_hash)
    _write_json(
        work / "score_receipt.json",
        {
            "schema": "pr15-host-score-receipt.v1",
            "status": "graded_source_native",
            "scorer": "frozen PR15 scripts/score_run.py",
            "metric_id": "global_wp_clip2_v1",
            "phase": score_phase,
            "score": score,
        },
    )
    return {
        **identity,
        "status": "fit_export_grade_completed",
        "fit_receipt": str(work / "fit_receipt.json"),
        "export_receipt": str(work / "export_receipt.json"),
        "score_receipt": str(work / "score_receipt.json"),
        "score": score,
    }


def main() -> int:
    """Write a durable terminal receipt while preserving any failed-phase outputs."""
    args = parse_args()
    receipt_path = args.work.absolute() / "worker_receipt.json"
    try:
        result = _execute(args)
        receipt = {"schema": "pr15-host-worker-receipt.v1", **result}
        _write_json(receipt_path, receipt)
        print("PR15_WORKER_RESULT " + json.dumps(receipt, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        failed = {
            "schema": "pr15-host-worker-receipt.v1",
            "status": "failed",
            "name": args.name,
            "mode": args.mode,
            "fold": args.fold,
            "seed": args.seed,
            "control": args.control,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if args.work.exists() and args.work.is_dir() and not receipt_path.exists():
            _write_json(receipt_path, failed)
        print("PR15_WORKER_RESULT " + json.dumps(failed, sort_keys=True), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
