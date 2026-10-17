"""Prepare hash-pinned four-arm architecture specs on the eventual Linux GPU host.

This command validates CUDA devices, the frozen F0 data/split and the packet's
source closure. It writes worker specs and exact SHA256 allowlists, but never
starts a trainer, predictor, grader, pytest process, or model import.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CAMPAIGN = Path("research/campaign_2026-10-17/architecture_gpu_wave_r1")
RECIPE_PATH = CAMPAIGN / "recipe.json"
SPLIT_DEFAULT = Path("research/campaign_2026-09-29/artifacts/splits/fold0.json")
REQUIRED_GPU_TESTS = (
    "tests/test_causal_transformer_gpu.py",
    "tests/test_mamba2_gpu.py",
    "tests/test_architecture_wave_integration_gpu.py",
)
GPU_QUERY = "index,uuid,name,memory.total"
HASH_BLOCK_BYTES = 1 << 20
RUN_ID_RE = re.compile(r"[A-Za-z0-9_.-]{1,96}\Z")


class PreparationError(RuntimeError):
    """A host, source, or data validation gate failed."""


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    """Serialize deterministic JSON bytes for explicit identities."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_json_write(path: Path, value: object) -> None:
    """Create a JSON file once through an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("xb") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
    temporary.replace(path)


def parse_gpu_ids(value: str) -> list[str]:
    """Parse one to four distinct physical GPU indices or UUIDs."""
    items = [part.strip() for part in value.split(",")]
    if not 1 <= len(items) <= 4 or any(not part for part in items):
        raise PreparationError("--gpu-ids must name one to four distinct GPUs")
    if len(items) != len(set(items)):
        raise PreparationError("--gpu-ids contains duplicates")
    if any(not re.fullmatch(r"[A-Za-z0-9_.:-]{1,96}", item) for item in items):
        raise PreparationError(
            "GPU ids may contain only letters, digits, dot, underscore, colon, and dash"
        )
    return items


def git_value(project_root: Path, *args: str) -> str | None:
    """Read one Git value without changing the checkout."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=project_root, text=True, capture_output=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def gpu_inventory(python: Path, gpu_ids: list[str]) -> list[dict[str, Any]]:
    """Require each selected physical NVIDIA device to initialize CUDA alone."""
    try:
        query = subprocess.run(
            ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        detail = getattr(error, "stderr", "")
        raise PreparationError(f"nvidia-smi inventory failed: {detail or error}") from error

    devices: dict[str, dict[str, Any]] = {}
    for line in query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            raise PreparationError(f"unexpected nvidia-smi inventory row: {line!r}")
        index, uuid, name, memory = parts
        try:
            memory_mib = int(memory)
        except ValueError as error:
            raise PreparationError(f"invalid GPU memory field in row: {line!r}") from error
        record = {"index": index, "uuid": uuid, "name": name, "memory_mib": memory_mib}
        devices[index] = record
        devices[uuid] = record

    selected: list[dict[str, Any]] = []
    seen_uuids: set[str] = set()
    probe = (
        "import json, torch; "
        "assert torch.cuda.is_available(); "
        "assert torch.cuda.device_count() == 1; "
        "d=torch.cuda.get_device_properties(0); "
        "print(json.dumps({'name': d.name, 'memory_bytes': int(d.total_memory), "
        "'torch_version': torch.__version__, 'cuda_version': torch.version.cuda}))"
    )
    for gpu_id in gpu_ids:
        device = devices.get(gpu_id)
        if device is None:
            raise PreparationError(f"GPU id {gpu_id!r} is absent from nvidia-smi inventory")
        if device["uuid"] in seen_uuids:
            raise PreparationError("--gpu-ids resolves to the same physical GPU more than once")
        seen_uuids.add(device["uuid"])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        try:
            checked = subprocess.run(
                [str(python), "-c", probe],
                cwd=Path.cwd(),
                env=env,
                text=True,
                capture_output=True,
                check=True,
                timeout=60,
            )
            torch_record = json.loads(checked.stdout.strip().splitlines()[-1])
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            ValueError,
            IndexError,
        ) as error:
            detail = getattr(error, "stderr", "")
            raise PreparationError(
                f"CUDA initialization failed for physical GPU {gpu_id}: {detail or error}"
            ) from error
        selected.append({**device, "cuda_probe": torch_record})
    return selected


def source_paths(project_root: Path, recipe: dict[str, Any]) -> list[Path]:
    """Expand the declared runner, tests, dependency, and WNN source closure."""
    closure = recipe["source_closure"]
    paths = [project_root / path for path in closure["scripts"]]
    paths.extend(project_root / path for path in closure["tests"])
    paths.extend(project_root / path for path in closure["dependency_files"])
    paths.append(project_root / closure["recipe"])
    for source_root in closure["source_roots"]:
        root = project_root / source_root
        if not root.is_dir():
            raise PreparationError(f"source root is missing: {root}")
        paths.extend(sorted(root.rglob("*.py")))
    unique = sorted({path.resolve() for path in paths})
    missing = [path for path in unique if not path.is_file()]
    if missing:
        listed = ", ".join(str(path) for path in missing[:12])
        raise PreparationError(f"source closure has missing files: {listed}")
    return unique


def source_lock(project_root: Path, recipe: dict[str, Any]) -> dict[str, Any]:
    """Capture exact source-file digests and the visible Git revision."""
    files: list[dict[str, Any]] = []
    for path in source_paths(project_root, recipe):
        relative = path.relative_to(project_root)
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    canonical = canonical_bytes(files)
    return {
        "schema": "wnn.architecture-gpu-wave-source-closure.v1",
        "project_root": str(project_root),
        "git_head": git_value(project_root, "rev-parse", "HEAD"),
        "git_branch": git_value(project_root, "branch", "--show-current"),
        "git_status": git_value(
            project_root, "status", "--short", "--", *[item["path"] for item in files]
        ),
        "files": files,
        "closure_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def validate_data(
    project_root: Path,
    data_root: Path,
    split_path: Path,
    portable_manifest: Path | None,
) -> dict[str, Any]:
    """Validate the immutable F0 split, exact full-valid size and data registry."""
    if not data_root.is_dir():
        raise PreparationError(f"data root does not exist: {data_root}")
    for relative in (
        "datasets/train.parquet",
        "datasets/valid.parquet",
        "cache/train_targets.npz",
        "cache/valid_targets.npz",
        "cache/chains.npz",
        "cache/chain_folds.npz",
    ):
        if not (data_root / relative).is_file():
            raise PreparationError(f"required data input is missing: {data_root / relative}")
    if not split_path.is_file():
        raise PreparationError(f"frozen fold manifest is missing: {split_path}")

    split_bytes = split_path.read_bytes()
    split_sha256 = hashlib.sha256(split_bytes).hexdigest()
    split_value = json.loads(split_bytes)
    if split_value.get("fold") != 0 or split_value.get("n_folds") != 4:
        raise PreparationError("the selected split must be the frozen four-fold fold-0 manifest")
    if portable_manifest is not None:
        if not portable_manifest.is_file():
            raise PreparationError(f"portable data registry is missing: {portable_manifest}")
        registry = json.loads(portable_manifest.read_text(encoding="utf-8"))
        if registry.get("version") != 1 or registry.get("split_manifest_sha256") != split_sha256:
            raise PreparationError(
                "portable data registry is not bound to the selected fold-0 split"
            )
        for name, identity in split_value.get("sources", {}).items():
            entry = registry.get("sources", {}).get(name)
            if not isinstance(entry, dict) or entry.get("original_identity") != identity:
                raise PreparationError(f"portable data registry source identity differs: {name}")
            if not re.fullmatch(r"[a-f0-9]{64}", str(entry.get("sha256", ""))):
                raise PreparationError(f"portable data registry lacks a full SHA256: {name}")

    os.environ["WNN_PROJECT_ROOT"] = str(data_root)
    if portable_manifest is not None:
        os.environ["WNN_PORTABLE_DATA_MANIFEST"] = str(portable_manifest)
    else:
        os.environ.pop("WNN_PORTABLE_DATA_MANIFEST", None)
    src = str(project_root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        import numpy as np

        from wnn.splits import load_campaign_split
    except Exception as error:
        raise PreparationError(f"cannot load the project split/data validators: {error}") from error
    try:
        split = load_campaign_split(split_path, data_root)
    except Exception as error:
        raise PreparationError(f"fold-0 data and membership validation failed: {error}") from error
    if split.fold != 0 or len(split.evaluation_groups) != 468:
        raise PreparationError(
            f"expected fold 0 with 468 evaluation groups; got fold={split.fold}, "
            f"groups={len(split.evaluation_groups)}"
        )
    with np.load(data_root / "cache/valid_targets.npz", allow_pickle=False) as targets:
        full_valid_groups = int(len(targets["row_group"]))
    if full_valid_groups != 1873:
        raise PreparationError(f"expected all 1873 validation groups; found {full_valid_groups}")

    sources: dict[str, Any] = {}
    registry_value = None
    if portable_manifest is not None:
        registry_value = json.loads(portable_manifest.read_text(encoding="utf-8"))
    for name, identity in split_value["sources"].items():
        path = data_root / name
        stat = path.stat()
        source = {
            "path": name,
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "frozen_identity": identity,
        }
        if registry_value is not None:
            source["portable_sha256"] = registry_value["sources"][name]["sha256"]
        elif "sha256" in identity:
            actual = sha256_file(path)
            if actual != identity["sha256"]:
                raise PreparationError(f"frozen source SHA256 mismatch: {name}")
            source["sha256"] = actual
        sources[name] = source
    return {
        "schema": "wnn.architecture-gpu-wave-data-identity.v1",
        "data_root": str(data_root),
        "split_path": str(split_path),
        "split_sha256": split_sha256,
        "fold": split.fold,
        "fit_train_groups": len(split.train_groups),
        "fit_valid_groups": len(split.valid_groups),
        "evaluation_groups": len(split.evaluation_groups),
        "full_valid_groups": full_valid_groups,
        "portable_manifest_path": None if portable_manifest is None else str(portable_manifest),
        "portable_manifest_sha256": (
            None if portable_manifest is None else sha256_file(portable_manifest)
        ),
        "sources": sources,
    }


def worker_environment(
    data_root: Path,
    project_root: Path,
    portable_manifest: Path | None,
    cpu_threads: int,
    loader_ram_gb: float,
    gpu_id: str | None,
) -> dict[str, str]:
    """Build the exact environment overlay captured in each queue step."""
    environment = {
        "WNN_PROJECT_ROOT": str(data_root),
        "WNN_THREADS": str(cpu_threads),
        "WNN_LOADER_RAM_GB": str(loader_ram_gb),
        "PYTHONPATH": str(project_root / "src")
        + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
    }
    if portable_manifest is not None:
        environment["WNN_PORTABLE_DATA_MANIFEST"] = str(portable_manifest)
    if gpu_id is not None:
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    return environment


def train_argv(
    python: Path,
    project_root: Path,
    split_path: Path,
    recipe: dict[str, Any],
    arm: dict[str, Any],
    *,
    preflight: bool,
) -> list[str]:
    """Construct the explicitly matched scratch trainer command."""
    common = recipe["common"]
    model = arm["model"]
    run_name = arm["preflight_run_id"] if preflight else arm["run_id"]
    argv = [
        str(python),
        str(project_root / "scripts/train_rnn.py"),
        "--name",
        run_name,
        "--arch",
        str(model["arch"]),
        "--hidden",
        str(model["hidden"]),
        "--layers",
        str(model["layers"]),
        "--seed",
        str(common["seed"]),
        "--train-source",
        str(common["fit_source"]),
        "--split-manifest",
        str(split_path),
        "--no-eval-during-fit",
        "--checkpoint-policy",
        str(common["checkpoint_policy"]),
        "--checkpoint-epoch",
        str(common["checkpoint_epoch"]),
        "--epochs",
        str(common["epochs"]),
        "--batch-order",
        str(common["batch_order"]),
        "--batch",
        str(common["batch"]),
        "--chunk",
        str(common["chunk"]),
        "--precision",
        str(common["precision"]),
        "--lr",
        str(common["learning_rate"]),
        "--wd",
        str(common["weight_decay"]),
        "--optim",
        str(common["optimizer"]),
        "--beta2",
        str(common["beta2"]),
        "--sched",
        str(common["scheduler"]),
        "--clip",
        str(common["gradient_clip"]),
        "--loss",
        str(common["loss"]),
        "--loss-norm",
        str(common["loss_norm"]),
        "--dropout",
        str(common["dropout"]),
        "--device",
        "cuda",
        "--check-finite-updates",
    ]
    option_map = {
        "attention_heads": "--attention-heads",
        "attention_window": "--attention-window",
        "attention_ffn_mult": "--attention-ffn-mult",
        "mamba_state_size": "--mamba-state-size",
        "mamba_head_dim": "--mamba-head-dim",
        "mamba_expand": "--mamba-expand",
        "mamba_conv_width": "--mamba-conv-width",
        "mamba_chunk_size": "--mamba-chunk-size",
    }
    for key, option in option_map.items():
        if key in model:
            argv.extend((option, str(model[key])))
    if preflight:
        argv.extend(("--max-batches", str(common["preflight_batches"])))
    else:
        argv.append("--save-training-state")
    return argv


def make_worker_spec(
    *,
    run_id: str,
    arm: dict[str, Any] | None,
    steps: list[dict[str, Any]],
    expected_outputs: list[Path],
    project_root: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one explicit, single-run spec for the existing bounded worker."""
    if not RUN_ID_RE.fullmatch(run_id):
        raise PreparationError(f"invalid worker run id: {run_id}")
    spec: dict[str, Any] = {
        "schema": "wnn.warm-queue-run.v1",
        "run_id": run_id,
        "arm": "coordinator" if arm is None else arm["arm"],
        "hypothesis": "architecture-gpu-wave-r1" if arm is None else arm["hypothesis_id"],
        "priority": 1,
        "steps": steps,
        "expected_outputs": [str(path) for path in expected_outputs],
        "fidelity": {
            "fit_count": 1,
            "seed": 3,
            "fold": 0,
            "epochs": 1,
            "labels_only": True,
            "precision": "fp32",
            "batch": 8,
            "chunk": 2000,
            "batch_order": "seeded",
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "beta2": 0.999,
            "scheduler": "OneCycle",
            "gradient_clip": 1.0,
        },
        "exposure": "Repeatedly inspected F0 development panel; not a fresh holdout or public forecast.",
        "limits": [
            "Target-GPU source/data validation is required before the fit stage.",
            "CPU submission compatibility is not established by this research teacher.",
        ],
    }
    if extra:
        spec.update(extra)
    return spec


def write_queue_spec(output_dir: Path, queue_name: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Write immutable pending JSON and its exact one-run SHA256 allowlist."""
    queue = output_dir / "queues" / queue_name
    pending = queue / "pending" / f"{spec['run_id']}.json"
    if (
        pending.exists()
        or (queue / "done" / pending.name).exists()
        or (queue / "failed" / pending.name).exists()
    ):
        raise PreparationError(f"worker spec path already exists: {pending}")
    pending.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(spec, indent=2, sort_keys=True).encode() + b"\n"
    with pending.open("xb") as stream:
        stream.write(payload)
    spec_sha256 = hashlib.sha256(payload).hexdigest()
    allowlist = queue / "allowed_spec_manifest.json"
    atomic_json_write(allowlist, {spec["run_id"]: spec_sha256})
    return {
        "run_id": spec["run_id"],
        "queue": str(queue),
        "spec_path": str(pending),
        "spec_sha256": spec_sha256,
        "allowed_spec_manifest": str(allowlist),
        "allowed_spec_manifest_sha256": sha256_file(allowlist),
    }


def prepare_specs(
    output_dir: Path,
    project_root: Path,
    data_root: Path,
    split_path: Path,
    python: Path,
    recipe: dict[str, Any],
    data_identity: dict[str, Any],
    source_closure: dict[str, Any],
    gpu_ids: list[str],
    cpu_threads: int,
    loader_ram_gb: float,
    portable_manifest: Path | None,
) -> list[dict[str, Any]]:
    """Write preflight, fit/export, and deferred coordinator grade specs."""
    grades = output_dir / "grades"
    queue_records: list[dict[str, Any]] = []
    split_sha = data_identity["split_sha256"]
    manifest_sha = hashlib.sha256(
        canonical_bytes(
            {
                "recipe_sha256": sha256_file(project_root / RECIPE_PATH),
                "source_closure_sha256": source_closure["closure_sha256"],
                "data_identity_sha256": hashlib.sha256(canonical_bytes(data_identity)).hexdigest(),
                "gpu_ids": gpu_ids,
                "cpu_threads_per_worker": cpu_threads,
                "loader_ram_gb_per_worker": loader_ram_gb,
            }
        )
    ).hexdigest()

    for index, arm in enumerate(recipe["arms"]):
        gpu_id = gpu_ids[index % len(gpu_ids)]
        environment = worker_environment(
            data_root, project_root, portable_manifest, cpu_threads, loader_ram_gb, gpu_id
        )
        preflight_id = arm["preflight_run_id"]
        preflight_root = data_root / "artifacts/runs" / preflight_id
        fit_id = arm["run_id"]
        fit_root = data_root / "artifacts/runs" / fit_id
        for run_root in (preflight_root, fit_root):
            if run_root.exists():
                raise PreparationError(f"target run directory already exists: {run_root}")
        for prediction_name in (preflight_id, fit_id):
            prediction = data_root / "cache" / f"pred_{prediction_name}_valid.npy"
            if prediction.exists() or prediction.with_suffix(".json").exists():
                raise PreparationError(f"target prediction output already exists: {prediction}")

        preflight_step = {
            "name": "one_real_batch_preflight",
            "argv": train_argv(python, project_root, split_path, recipe, arm, preflight=True),
            "cwd": str(project_root),
            "env": environment,
            "timeout_seconds": 3600,
        }
        preflight_outputs = [
            preflight_root / "model.pt",
            preflight_root / "model_ep0.pt",
            preflight_root / "history.json",
            preflight_root / "provenance.json",
        ]
        preflight_spec = make_worker_spec(
            run_id=preflight_id,
            arm=arm,
            steps=[preflight_step],
            expected_outputs=preflight_outputs,
            project_root=project_root,
            extra={
                "phase": "gpu_preflight_one_batch",
                "recipe_sha256": sha256_file(project_root / RECIPE_PATH),
                "source_closure_sha256": source_closure["closure_sha256"],
                "split_manifest_sha256": split_sha,
                "assigned_gpu": gpu_id,
            },
        )
        written = write_queue_spec(output_dir, f"preflight/{arm['run_id']}", preflight_spec)
        queue_records.append({"stage": "preflight", "arm": arm["arm"], **written})

        fit_step = {
            "name": "full_fold_fit",
            "argv": train_argv(python, project_root, split_path, recipe, arm, preflight=False),
            "cwd": str(project_root),
            "env": environment,
            "timeout_seconds": 172800,
            "resume_state_path": str(fit_root / "training_state.pt"),
        }
        predict_step = {
            "name": "full_valid_export",
            "argv": [
                str(python),
                str(project_root / "scripts/predict_run.py"),
                fit_id,
                "valid",
                "--epoch",
                "0",
                "--campaign-split-manifest",
                str(split_path),
                "--campaign-full-valid-feature-only",
                "--device",
                "cuda",
                "--inference-precision",
                "fp32",
                "--batch",
                str(recipe["common"]["batch"]),
                "--chunk",
                str(recipe["common"]["chunk"]),
            ],
            "cwd": str(project_root),
            "env": environment,
            "timeout_seconds": 172800,
        }
        fit_outputs = [
            fit_root / "model.pt",
            fit_root / "model_ep0.pt",
            fit_root / "training_state.pt",
            fit_root / "history.json",
            fit_root / "provenance.json",
            data_root / "cache" / f"pred_{fit_id}_valid.npy",
            data_root / "cache" / f"pred_{fit_id}_valid.json",
        ]
        fit_spec = make_worker_spec(
            run_id=fit_id,
            arm=arm,
            steps=[fit_step, predict_step],
            expected_outputs=fit_outputs,
            project_root=project_root,
            extra={
                "phase": "fit_and_full_valid_export",
                "recipe_sha256": sha256_file(project_root / RECIPE_PATH),
                "source_closure_sha256": source_closure["closure_sha256"],
                "split_manifest_sha256": split_sha,
                "assigned_gpu": gpu_id,
            },
        )
        written = write_queue_spec(output_dir, f"fit/{arm['run_id']}", fit_spec)
        queue_records.append({"stage": "fit", "arm": arm["arm"], **written})

    baseline = recipe["comparison"]["reference_run_id"]
    baseline_grade = grades / f"{baseline}_standalone_grade.json"
    baseline_step = {
        "name": "score_gru_reference",
        "argv": [
            str(python),
            str(project_root / "scripts/score_fold_panel.py"),
            f"{baseline}@ep0",
            "--split",
            str(split_path),
            "--expected-fold",
            "0",
            "--root",
            str(data_root),
            "--out",
            str(baseline_grade),
        ],
        "cwd": str(project_root),
        "env": worker_environment(
            data_root, project_root, portable_manifest, cpu_threads, loader_ram_gb, None
        ),
        "timeout_seconds": 3600,
    }
    baseline_spec = make_worker_spec(
        run_id="architecture_wave_grade_gru_f0_s3_r1",
        arm=None,
        steps=[baseline_step],
        expected_outputs=[baseline_grade],
        project_root=project_root,
        extra={
            "phase": "reference_grade_after_all_exports",
            "recipe_sha256": sha256_file(project_root / RECIPE_PATH),
            "source_closure_sha256": source_closure["closure_sha256"],
            "split_manifest_sha256": split_sha,
        },
    )
    written = write_queue_spec(output_dir, "grade/gru_reference", baseline_spec)
    queue_records.append({"stage": "grade_reference", "arm": "A", **written})

    for arm in recipe["arms"]:
        if arm["run_id"] == baseline:
            continue
        candidate = f"{arm['run_id']}@ep0"
        references = list(recipe["comparison"]["paired_references"][arm["arm"]])
        grade_path = grades / f"{arm['run_id']}_paired_grade.json"
        receipt_path = grades / f"{arm['run_id']}_comparison_receipt.json"
        ledger_path = grades / "comparison_ledger.jsonl"
        score_argv = [
            str(python),
            str(project_root / "scripts/score_fold_panel.py"),
            candidate,
            "--epoch",
            "0",
        ]
        for reference in references:
            score_argv.extend(("--reference", reference))
        score_argv.extend(
            (
                "--split",
                str(split_path),
                "--expected-fold",
                "0",
                "--root",
                str(data_root),
                "--out",
                str(grade_path),
            )
        )
        score_step = {
            "name": "score_candidate_against_gru",
            "argv": score_argv,
            "cwd": str(project_root),
            "env": worker_environment(
                data_root, project_root, portable_manifest, cpu_threads, loader_ram_gb, None
            ),
            "timeout_seconds": 3600,
        }
        comparison = {
            "experiment_id": f"E-ARCH-GPU-{arm['arm']}-F0-S3-R1",
            "hypothesis_id": arm["hypothesis_id"],
            "candidate": candidate,
            "references": references,
            "expected_fold": 0,
            "split_path": str(split_path),
            "grade_path": str(grade_path),
            "receipt_path": str(receipt_path),
            "ledger_path": str(ledger_path),
            "input_hashes": {
                "recipe_sha256": sha256_file(project_root / RECIPE_PATH),
                "source_closure_sha256": source_closure["closure_sha256"],
                "split_manifest_sha256": split_sha,
                "preparation_identity_sha256": manifest_sha,
                **(
                    {"portable_data_manifest_sha256": data_identity["portable_manifest_sha256"]}
                    if data_identity["portable_manifest_sha256"]
                    else {}
                ),
            },
            "fidelity": {
                "fit_count": 1,
                "seed": 3,
                "fold": 0,
                "epochs": 1,
                "labels_only": True,
                "precision": "fp32",
                "fit_budget": "one full epoch, batch 8, chunk 2000",
                "optimizer": "AdamW",
                "learning_rate": 0.001,
                "weight_decay": 0.01,
                "beta2": 0.999,
                "scheduler": "OneCycle",
                "gradient_clip": 1.0,
            },
            "ancestry": "Scratch initialization; labels-only targets; no teacher or initializer.",
            "exposure": "Repeatedly inspected fold-0 development panel; not a fresh holdout or public forecast.",
            "limits": [
                "Layer counts differ by declared architecture; parameter counts are not matched.",
                "Receipt validates the declared grade and artifact identities; it does not rerun fitting.",
            ],
        }
        grade_spec = make_worker_spec(
            run_id=f"grade_{arm['run_id']}",
            arm=arm,
            steps=[score_step],
            expected_outputs=[grade_path],
            project_root=project_root,
            extra={
                "phase": "candidate_grade_after_gru_reference_exists",
                "recipe_sha256": sha256_file(project_root / RECIPE_PATH),
                "source_closure_sha256": source_closure["closure_sha256"],
                "split_manifest_sha256": split_sha,
                "comparison": comparison,
            },
        )
        written = write_queue_spec(output_dir, f"grade/{arm['run_id']}", grade_spec)
        queue_records.append({"stage": "grade_candidate", "arm": arm["arm"], **written})

    fixed_mix = recipe["comparison"]["fixed_mix"]
    for arm in recipe["arms"]:
        if arm["arm"] not in fixed_mix["candidate_arms"]:
            continue
        candidate = (
            f"{arm['run_id']}@ep0*{fixed_mix['candidate_weight']},"
            f"{baseline}@ep0*{fixed_mix['reference_weight']}"
        )
        reference = f"{baseline}@ep0"
        run_id = f"grade_mix_{arm['run_id']}"
        grade_path = grades / f"{arm['run_id']}_mix90_gru10_grade.json"
        receipt_path = grades / f"{arm['run_id']}_mix90_gru10_receipt.json"
        score_step = {
            "name": "score_fixed_mix_against_gru",
            "argv": [
                str(python),
                str(project_root / "scripts/score_fold_panel.py"),
                candidate,
                "--reference",
                reference,
                "--split",
                str(split_path),
                "--expected-fold",
                "0",
                "--root",
                str(data_root),
                "--out",
                str(grade_path),
            ],
            "cwd": str(project_root),
            "env": worker_environment(
                data_root, project_root, portable_manifest, cpu_threads, loader_ram_gb, None
            ),
            "timeout_seconds": 3600,
        }
        comparison = {
            "experiment_id": f"E-ARCH-GPU-{arm['arm']}-MIX90-GRU10-F0-S3-R1",
            "hypothesis_id": arm["hypothesis_id"],
            "candidate": candidate,
            "references": [reference],
            "expected_fold": 0,
            "split_path": str(split_path),
            "grade_path": str(grade_path),
            "receipt_path": str(receipt_path),
            "ledger_path": str(ledger_path),
            "input_hashes": {
                "recipe_sha256": sha256_file(project_root / RECIPE_PATH),
                "source_closure_sha256": source_closure["closure_sha256"],
                "split_manifest_sha256": split_sha,
                "preparation_identity_sha256": manifest_sha,
                **(
                    {"portable_data_manifest_sha256": data_identity["portable_manifest_sha256"]}
                    if data_identity["portable_manifest_sha256"]
                    else {}
                ),
            },
            "fidelity": {
                "fit_count": 1,
                "seed": 3,
                "fold": 0,
                "epochs": 1,
                "labels_only": True,
                "precision": "fp32",
                "fixed_blend": {"candidate_weight": 0.1, "gru_reference_weight": 0.9},
                "fit_budget": "same cached checkpoints as standalone arms; no additional fit",
            },
            "ancestry": "Fixed prediction blend of the scratch candidate and scratch GRU; no additional fit.",
            "exposure": "Repeatedly inspected fold-0 development panel; not a fresh holdout or public forecast.",
            "limits": [
                "Blend weights were fixed in the recipe before scoring.",
                "Receipt validates declared grade and artifact identities; it does not rerun fitting.",
            ],
        }
        grade_spec = make_worker_spec(
            run_id=run_id,
            arm=arm,
            steps=[score_step],
            expected_outputs=[grade_path],
            project_root=project_root,
            extra={
                "phase": "fixed_mix_grade_after_gru_reference_exists",
                "recipe_sha256": sha256_file(project_root / RECIPE_PATH),
                "source_closure_sha256": source_closure["closure_sha256"],
                "split_manifest_sha256": split_sha,
                "comparison": comparison,
            },
        )
        written = write_queue_spec(output_dir, f"grade/{run_id}", grade_spec)
        queue_records.append({"stage": "grade_mix", "arm": arm["arm"], **written})

    return queue_records


def parse_args() -> argparse.Namespace:
    """Parse target-host preparation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True, help="source checkout root")
    parser.add_argument(
        "--data-root", type=Path, required=True, help="data and model-artifact root"
    )
    parser.add_argument("--portable-data-manifest", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument(
        "--gpu-ids", required=True, help="one to four physical GPU indices or UUIDs"
    )
    parser.add_argument("--cpu-threads-per-worker", type=int, required=True)
    parser.add_argument("--loader-ram-gb-per-worker", type=float, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="persistent output outside the source checkout",
    )
    return parser.parse_args()


def main() -> int:
    """Validate target host and write frozen worker specs without executing them."""
    args = parse_args()
    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    split_path = (
        args.split_manifest.resolve()
        if args.split_manifest is not None
        else (project_root / SPLIT_DEFAULT).resolve()
    )
    # Keep the .venv/bin/python symlink itself. Resolving it points at the
    # base interpreter and silently drops the target environment's packages.
    python = Path(sys.executable).absolute()
    portable_value = args.portable_data_manifest or (
        Path(os.environ["WNN_PORTABLE_DATA_MANIFEST"])
        if os.environ.get("WNN_PORTABLE_DATA_MANIFEST")
        else None
    )
    portable_manifest = None if portable_value is None else portable_value.resolve()
    try:
        if platform.system() != "Linux":
            raise PreparationError(
                "architecture wave preparation requires the intended Linux GPU host"
            )
        if not project_root.is_dir() or not (project_root / ".git").exists():
            raise PreparationError(f"project root is not a Git checkout: {project_root}")
        if args.cpu_threads_per_worker < 1 or args.loader_ram_gb_per_worker <= 0:
            raise PreparationError("per-worker CPU thread and loader RAM budgets must be positive")
        if output_dir == project_root or project_root in output_dir.parents:
            raise PreparationError(
                "choose a persistent output directory outside the source checkout"
            )
        if output_dir.exists() and any(output_dir.iterdir()):
            raise PreparationError(
                f"output directory is nonempty; refusing to overwrite: {output_dir}"
            )
        recipe_source_path = project_root / RECIPE_PATH
        recipe_source_bytes = recipe_source_path.read_bytes()
        recipe = json.loads(recipe_source_bytes)
        if recipe.get("schema") != "wnn.architecture-gpu-wave-recipe.v1":
            raise PreparationError("unsupported architecture wave recipe schema")
        gpu_ids = parse_gpu_ids(args.gpu_ids)
        gpu_devices = gpu_inventory(python, gpu_ids)
        data_identity = validate_data(project_root, data_root, split_path, portable_manifest)
        if split_path != (project_root / recipe["split"]["path"]).resolve():
            raise PreparationError("split-manifest override differs from the frozen recipe path")
        if data_identity["split_sha256"] != recipe["split"]["sha256"]:
            raise PreparationError("frozen fold-0 split SHA256 differs from the recipe pin")
        source_closure = source_lock(project_root, recipe)
        run_ids = [arm[key] for arm in recipe["arms"] for key in ("run_id", "preflight_run_id")]
        run_ids.extend(
            [
                "architecture_wave_grade_gru_f0_s3_r1",
                *(
                    f"grade_{arm['run_id']}"
                    for arm in recipe["arms"]
                    if arm["run_id"] != recipe["comparison"]["reference_run_id"]
                ),
                *(
                    f"grade_mix_{arm['run_id']}"
                    for arm in recipe["arms"]
                    if arm["arm"] in recipe["comparison"]["fixed_mix"]["candidate_arms"]
                ),
            ]
        )
        if len(set(run_ids)) != len(run_ids) or any(
            not RUN_ID_RE.fullmatch(run_id) for run_id in run_ids
        ):
            raise PreparationError("recipe contains an invalid or duplicate worker run id")

        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "recipe.json").open("xb") as stream:
            stream.write(recipe_source_bytes)
        atomic_json_write(output_dir / "source_closure.json", source_closure)
        atomic_json_write(output_dir / "data_identity.json", data_identity)
        queue_records = prepare_specs(
            output_dir,
            project_root,
            data_root,
            split_path,
            python,
            recipe,
            data_identity,
            source_closure,
            gpu_ids,
            args.cpu_threads_per_worker,
            args.loader_ram_gb_per_worker,
            portable_manifest,
        )
        manifest = {
            "schema": "wnn.architecture-gpu-wave-preparation.v1",
            "status": "prepared_target_gpu_preflight_pending",
            "created_utc": datetime.now(UTC).isoformat(),
            "project_root": str(project_root),
            "data_root": str(data_root),
            "output_dir": str(output_dir),
            "python": str(python),
            "recipe_path": str(output_dir / "recipe.json"),
            "recipe_sha256": sha256_file(output_dir / "recipe.json"),
            "source_closure_path": str(output_dir / "source_closure.json"),
            "source_closure_sha256": source_closure["closure_sha256"],
            "data_identity_path": str(output_dir / "data_identity.json"),
            "data_identity_sha256": hashlib.sha256(canonical_bytes(data_identity)).hexdigest(),
            "split_manifest": str(split_path),
            "split_manifest_sha256": data_identity["split_sha256"],
            "portable_data_manifest": None if portable_manifest is None else str(portable_manifest),
            "portable_data_manifest_sha256": data_identity["portable_manifest_sha256"],
            "gpu_ids": gpu_ids,
            "gpu_devices": gpu_devices,
            "cpu_threads_per_worker": args.cpu_threads_per_worker,
            "aggregate_cpu_threads": args.cpu_threads_per_worker * len(gpu_ids),
            "loader_ram_gb_per_worker": args.loader_ram_gb_per_worker,
            "aggregate_loader_ram_gb": args.loader_ram_gb_per_worker * len(gpu_ids),
            "queues": queue_records,
            "spec_manifest_sha256": hashlib.sha256(canonical_bytes(queue_records)).hexdigest(),
            "execution_started": False,
        }
        atomic_json_write(output_dir / "manifest.json", manifest)
        atomic_json_write(
            output_dir / "manifest.sha256.json",
            {"manifest_sha256": sha256_file(output_dir / "manifest.json")},
        )
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "manifest": str(output_dir / "manifest.json"),
                    "gpu_ids": gpu_ids,
                    "gpu_count": len(gpu_ids),
                    "arm_count": len(recipe["arms"]),
                    "split_sha256": data_identity["split_sha256"],
                    "source_closure_sha256": source_closure["closure_sha256"],
                    "execution_started": False,
                },
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError, PreparationError) as error:
        print(
            json.dumps(
                {
                    "status": "HOLD",
                    "error": str(error),
                    "execution_started": False,
                    "gpu_fit_started": False,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
