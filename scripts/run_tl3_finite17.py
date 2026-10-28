"""Execute the frozen train-only TL3 17-fit DAG on receipt-proven free GPUs."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from scripts.tl3_chronological_recipes import FINAL_WEIGHTS, PREFIX, W18_COEFFICIENTS
from scripts.tl3_finite17_contract import (
    FIT_LIMIT,
    MAX_CELL_SECONDS,
    MAX_GPU_SECONDS,
    MAX_WALL_SECONDS,
    MIN_FREE_BYTES,
    PLAN_SCHEMA,
    STATE_SCHEMA,
    TRAIN_GROUPS,
    VALID_GROUPS,
    DriverError,
    build_plan,
    canonical_sha256,
    expected_prediction_bytes,
    source_identity,
    validate_adoption,
    verify_data_contract,
    write_atomic_json,
)
from scripts.tl3_finite17_process import (
    free_gpu_devices,
    gpu_cpu_affinities,
    nvidia_inventory,
    owned_group_state,
    proc_identity,
    protected_gpu_indices,
    terminate_owned_group,
)
from wnn.chronological_source_index import verify_source_index_receipt
from wnn.distillation_cache import row_groups_sha256
from wnn.portability import sha256_file
from wnn.splits import load_campaign_split

MODEL_ARTIFACT_RESERVE_BYTES = 2 << 30
SPAWN_CLEANUP_SECONDS = 5
POLL_SECONDS = 20
TRAIN_EXPORT_SCOPE = "chronological-full-train-index-feature-only"
NODE_PRIORITY = (
    "fin256_s4",
    "fin256ft_s3",
    "fin256ft_s4",
    "fttall_tlead_s5",
    "fttall_tlead_s6",
    "fbc1k_s5",
    "ffull_s3",
    "ffull_s4",
    "frawW_s3ft",
    "frawW_s4ft",
    "orgbase",
    "fmu128_s3",
    "frawW_s3",
    "fmu128_s4",
    "frawW_s4",
)


def read_json(path: Path) -> dict[str, object]:
    """Read one regular JSON object without following a symlink."""
    if path.is_symlink() or not path.is_file():
        raise DriverError(f"expected a regular receipt file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DriverError(f"expected a JSON object: {path}")
    return value


def _node_map(plan: dict[str, object]) -> dict[str, dict[str, object]]:
    """Index the finite recipe by its immutable node IDs."""
    nodes = plan.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != FIT_LIMIT:
        raise DriverError("frozen finite plan does not contain exactly 17 nodes")
    result: dict[str, dict[str, object]] = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            raise DriverError("frozen plan has a malformed node")
        if node["id"] in result:
            raise DriverError("frozen plan repeats a node ID")
        result[str(node["id"])] = node
    return result


def _state_paths(state_dir: Path) -> tuple[Path, Path, Path]:
    """Return the durable plan, mutable state, and immutable terminal receipt paths."""
    return (
        state_dir / "plan.json",
        state_dir / "status.json",
        state_dir / "terminal.json",
    )


def _new_status(plan: dict[str, object], adoption_identity: dict[str, object]) -> dict[str, object]:
    """Seed the status ledger with the existing fin256_s3 process, never a duplicate."""
    nodes = _node_map(plan)
    node_status = {
        name: {
            "state": "not_started",
            "phase": "not_started",
            "cell_started_unix": None,
            "verified_run": None,
        }
        for name in nodes
    }
    first = node_status["fin256_s3"]
    first.update(
        {
            "state": "running",
            "phase": "fit",
            "cell_started_unix": adoption_identity["started_unix"],
            "source_sha256": adoption_identity["source_sha256"],
            "args_sha256": adoption_identity["args_sha256"],
        }
    )
    now = time.time()
    task_id = "fit:fin256_s3"
    tasks = {
        task_id: {
            "task_id": task_id,
            "kind": "fit",
            "node_id": "fin256_s3",
            "state": "running",
            "adopted": True,
            **adoption_identity,
        }
    }
    return {
        "schema": STATE_SCHEMA,
        "status": "running",
        "plan_sha256": plan["sha256"],
        "started_unix": min(now, float(adoption_identity["started_unix"])),
        "updated_unix": now,
        "fits_used": 1,
        "gpu_seconds_completed": 0.0,
        "nodes": node_status,
        "tasks": tasks,
        "cache": {"state": "not_started"},
        "final_exports": {run: "not_started" for run in plan["final_runs"]},
        "score": {"state": "not_started"},
        "controller": {},
        "last_error": None,
    }


def _validate_plan(plan: dict[str, object]) -> None:
    """Validate plan hash and the exact finite graph before any process launch."""
    if plan.get("schema") != PLAN_SCHEMA:
        raise DriverError("unsupported finite TL3 plan schema")
    supplied = plan.get("sha256")
    without_hash = dict(plan)
    without_hash.pop("sha256", None)
    if supplied != canonical_sha256(without_hash):
        raise DriverError("frozen finite plan bytes do not match their SHA-256")
    nodes = _node_map(plan)
    if sum(int(node["expected_updates"]) for node in nodes.values()) != 257740:
        raise DriverError("frozen plan optimizer-update total changed")
    if tuple(plan.get("final_runs", [])) != tuple(PREFIX + name for name in FINAL_WEIGHTS):
        raise DriverError("fixed final TL3 member identity changed")
    if abs(float(plan.get("coefficient_sum", 0.0)) - 0.8125) > 1e-7:
        raise DriverError("W18 source-scale coefficient sum changed")


def _write_status(state_path: Path, status: dict[str, object]) -> None:
    """Publish a durable status snapshot after every phase transition."""
    status["updated_unix"] = time.time()
    write_atomic_json(state_path, status)


def _controller_identity() -> dict[str, object]:
    """Bind this detached controller to its current Linux PID start ticks."""
    identity = proc_identity(os.getpid())
    if identity is None:
        raise DriverError("cannot record controller /proc identity")
    return {
        "pid": os.getpid(),
        "start_ticks": str(identity["start_ticks"]),
        "process_group": identity["process_group"],
        "session": identity["session"],
    }


def _load_or_create_state(
    state_dir: Path,
    plan: dict[str, object],
    adoption_identity: dict[str, object],
) -> tuple[dict[str, object], bool]:
    """Create state once, or resume only an identical frozen plan."""
    if state_dir.is_symlink():
        raise DriverError("state directory cannot be a symlink")
    state_dir.mkdir(parents=True, exist_ok=True)
    plan_path, status_path, terminal_path = _state_paths(state_dir)
    if terminal_path.exists():
        raise DriverError("finite driver already has a terminal receipt")
    if status_path.exists() or plan_path.exists():
        if not status_path.exists() or not plan_path.exists():
            raise DriverError("state directory contains an incomplete plan/status pair")
        old_plan = read_json(plan_path)
        old_status = read_json(status_path)
        _validate_plan(old_plan)
        if old_plan.get("sha256") != plan.get("sha256"):
            raise DriverError("existing state belongs to another source, recipe, or adoption")
        if old_status.get("schema") != STATE_SCHEMA or old_status.get("plan_sha256") != plan.get(
            "sha256"
        ):
            raise DriverError("existing status does not match the frozen finite plan")
        return old_status, True
    permitted = {"driver.lock"}
    if any(path.name not in permitted for path in state_dir.iterdir()):
        raise DriverError("state directory is not fresh")
    _validate_plan(plan)
    write_atomic_json(plan_path, plan)
    status = _new_status(plan, adoption_identity)
    write_atomic_json(status_path, status)
    return status, False


def _child_environment(root: Path, gpu_uuid: str | None) -> dict[str, str]:
    """Build a small isolated process environment for one owned fit/export/cache."""
    allowed = (
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "LD_LIBRARY_PATH",
        "VIRTUAL_ENV",
        "CUDA_HOME",
        "CUDA_PATH",
        "TORCH_HOME",
    )
    environment = {key: os.environ[key] for key in allowed if key in os.environ}
    environment.update(
        {
            "WNN_PROJECT_ROOT": str(root),
            "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root), str(root / "scripts"))),
            "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": "0",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "ARROW_NUM_THREADS": "1",
            "WNN_THREADS": "4",
            "WNN_LOADER_WORKERS": "2",
            "WNN_LOADER_RAM_GB": "8",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        }
    )
    if gpu_uuid is None:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    else:
        environment["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return environment


def _node_args(node: dict[str, object]) -> list[str]:
    """Return only the frozen trainer argument vector from the plan."""
    args = node.get("args")
    if not isinstance(args, list) or not all(isinstance(value, str) for value in args):
        raise DriverError(f"{node.get('id')}: malformed frozen trainer arguments")
    if node.get("args_sha256") != canonical_sha256(args):
        raise DriverError(f"{node.get('id')}: frozen trainer arguments changed")
    return args


def _fit_command(root: Path, python: Path, node: dict[str, object]) -> list[str]:
    """Build one exact frozen training command without an external timeout wrapper."""
    return [str(python), str(root / "scripts/train_rnn.py"), *_node_args(node)]


def _predict_command(
    root: Path,
    python: Path,
    split_path: Path,
    run: str,
    split: str,
    selector: str,
) -> list[str]:
    """Build feature-only full-index teacher or complete-valid final exports."""
    command = [
        str(python),
        str(root / "scripts/predict_run.py"),
        run,
        split,
        "--campaign-split-manifest",
        str(split_path),
    ]
    if selector == "train-index":
        command.append("--campaign-full-train-index-feature-only")
    elif selector == "valid-evaluation":
        command.append("--campaign-evaluation-only")
    else:
        raise DriverError(f"unsupported prediction selector: {selector}")
    return command


def _cache_command(root: Path, python: Path, plan: dict[str, object]) -> list[str]:
    """Build the six-source unnormalized W18 target cache at the frozen path."""
    split_path = Path(str(plan["split"]["split_manifest"]))
    command = [
        str(python),
        str(root / "scripts/build_distill_cache.py"),
        "--data-root",
        str(root),
        "--teacher-cache-root",
        str(root / "cache"),
        "--checkpoint-root",
        str(root / "artifacts/runs"),
        "--split-manifest",
        str(split_path),
        "--output",
        str(plan["w18_cache"]),
        "--homogeneous-family",
        "strong",
        "--expected-coefficient-sum",
        "0.8125",
        "--output-profile",
        "all",
        "--chunk-sequences",
        "8",
        "--output-budget-gb",
        "30",
        "--min-free-gb",
        "12",
    ]
    for name, record in plan["w18_coefficients"].items():
        run = PREFIX + name
        coefficient = format(float(record["coefficient"]), ".8g")
        command.extend(
            [
                "--teacher",
                f"{record['family']}={run}@ep{int(record['epoch'])}*{coefficient}",
            ]
        )
    return command


def _score_command(root: Path, python: Path, plan: dict[str, object]) -> list[str]:
    """Build the fixed three-member scorer invocation supplied with the packet."""
    return [
        str(python),
        str(root / "scripts/score_tl3_chronological.py"),
        "--data-root",
        str(root),
        "--split-manifest",
        str(plan["split"]["split_manifest"]),
        "--output",
        str(root / "final-score.json"),
    ]


def _active_tasks(status: dict[str, object]) -> dict[str, dict[str, object]]:
    """Return process records that still occupy a GPU or run the bounded cache/scorer."""
    tasks = status.get("tasks")
    if not isinstance(tasks, dict):
        raise DriverError("finite status has no task ledger")
    return {
        str(task_id): task
        for task_id, task in tasks.items()
        if isinstance(task, dict) and task.get("state") == "running"
    }


def _stage_state(value: object) -> str:
    """Normalize durable stage strings and completed receipt objects."""
    if isinstance(value, dict):
        state = value.get("state")
        return str(state) if isinstance(state, str) else "invalid"
    return value if isinstance(value, str) else "invalid"


def _gpu_reserved(status: dict[str, object]) -> set[int]:
    """Reserve cards for every active owned fit or prediction process."""
    reserved = set()
    for task in _active_tasks(status).values():
        index = task.get("gpu_index")
        if type(index) is int:
            reserved.add(index)
    return reserved


def _prediction_groups(root: Path, split_path: Path, split: str) -> list[int]:
    """Read exact full-train index or official-valid membership after admission."""
    data = verify_data_contract(root, split_path)
    membership = data["split"]
    if split == "train":
        groups, _ = verify_source_index_receipt(root, "train")
        return list(map(int, groups))
    if split == "valid":
        return list(map(int, membership.evaluation_groups))
    raise DriverError("unsupported prediction source")


def _prediction_output_paths(root: Path, run: str, split: str) -> tuple[Path, ...]:
    """List every immutable and partial output a prediction command can create."""
    array = root / "cache" / f"pred_{run}_{split}.npy"
    return (array, array.with_suffix(".json"), array.with_name(array.name + ".partial"))


def _normalize_persisted_args(raw: object, expected: dict[str, object]) -> dict[str, object]:
    """Resolve the full-valid campaign's effective vfold defaults in saved args."""
    if not isinstance(raw, dict):
        raise DriverError("checkpoint lacks its serialized trainer arguments")
    actual = dict(raw)
    if actual.get("checkpoint_policy") == "fixed" and actual.get("checkpoint_epoch") is None:
        actual["checkpoint_epoch"] = int(actual["epochs"]) - 1
    if actual.get("sched") == "onecycle" and actual.get("warmup_frac") is None:
        actual["warmup_frac"] = 0.05
    if actual.get("split_manifest") is not None:
        actual["eval_source"] = "vfold"
        actual["valid_fold"] = 0
        actual["n_valid_folds"] = 1
    if "train_eval_seqs" in expected and "train_eval_seqs" not in actual:
        actual["train_eval_seqs"] = 0
    return actual


_FIT_ARGUMENTS = (
    "name",
    "seed",
    "hidden",
    "layers",
    "dropout",
    "cell",
    "enc",
    "raw_only",
    "train_source",
    "train_seqs",
    "train_classes",
    "eval_source",
    "valid_fold",
    "n_valid_folds",
    "full_valid",
    "split_manifest",
    "no_eval_during_fit",
    "checkpoint_policy",
    "checkpoint_epoch",
    "batch",
    "chunk",
    "epochs",
    "lr",
    "wd",
    "beta2",
    "loss",
    "loss_norm",
    "sched",
    "warmup_frac",
    "precision",
    "optim",
    "muon_lr_mult",
    "aux_next",
    "aux_struct",
    "aux_weight",
    "ema_teacher",
    "ema_decay",
    "eval_ema",
    "since_weight",
    "since_floor",
    "init",
    "inherit_init_preprocessing",
    "distill",
    "distill_alpha",
    "distill_boot_h",
    "distill_cache_root",
    "distill_family_weight",
    "propensity",
    "prop_input",
    "prop_cache",
    "mask_weight",
    "include_heldout_valid",
    "valid_train_fraction",
    "class_balance",
    "fit_subset_manifest",
    "fit_groups_manifest",
    "save_training_state",
    "train_eval_seqs",
)


def _expected_effective_args(node: dict[str, object]) -> dict[str, object]:
    """Return frozen fit arguments after canonical membership resolution."""
    from scripts.tl3_finite17_contract import _normalized_args

    expected = _normalized_args(_node_args(node))
    expected["eval_source"] = "vfold"
    expected["valid_fold"] = 0
    expected["n_valid_folds"] = 1
    return expected


def _verify_checkpoint_args(node: dict[str, object], checkpoint_args: object) -> None:
    """Compare every scientific fit field after parser and split defaults."""
    expected = _expected_effective_args(node)
    actual = _normalize_persisted_args(checkpoint_args, expected)
    for key in _FIT_ARGUMENTS:
        # R3's AdamW parser had no Muon multiplier; the option is unused here.
        if (
            key == "muon_lr_mult"
            and key not in actual
            and actual.get("optim") == expected.get("optim") == "adamw"
        ):
            continue
        if key not in actual or key not in expected or actual[key] != expected[key]:
            raise DriverError(f"{node['id']}: saved fit argument changed: {key}")
    if actual["train_source"] != "train" or actual["no_eval_during_fit"]:
        raise DriverError(f"{node['id']}: invalid chronological fit/evaluation mode")


def _verify_membership(provenance: dict[str, object], split: object, node_id: str) -> None:
    """Require exact official train fitting and full-valid report-only membership."""
    fit = provenance.get("fit_row_groups")
    if not isinstance(fit, dict):
        raise DriverError(f"{node_id}: provenance lacks fitting membership")
    train = fit.get("train")
    valid = fit.get("valid")
    evaluation = provenance.get("evaluation_row_groups")
    if (
        not isinstance(train, list)
        or set(train) != set(split.train_groups)
        or len(train) != len(split.train_groups)
        or valid != []
        or not isinstance(evaluation, list)
        or set(evaluation) != set(split.evaluation_groups)
        or len(evaluation) != len(split.evaluation_groups)
    ):
        raise DriverError(f"{node_id}: checkpoint membership is not train-only/full-valid")


def _verify_history(history: object, node: dict[str, object], expected_updates: int) -> None:
    """Require every fixed epoch, complete update count, and full-valid report."""
    epochs = int(node["epochs"])
    if not isinstance(history, list) or len(history) != epochs:
        raise DriverError(f"{node['id']}: history is not the full configured epoch count")
    per_epoch = expected_updates // epochs
    for index, row in enumerate(history):
        if not isinstance(row, dict) or row.get("epoch") != index:
            raise DriverError(f"{node['id']}: history epoch sequence is incomplete")
        if row.get("split") != f"vfold{VALID_GROUPS}":
            raise DriverError(f"{node['id']}: epoch did not report complete official validation")
        score = row.get("weighted_pearson")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise DriverError(f"{node['id']}: validation history contains a nonfinite score")
        fit_summary = row.get("fit_summary")
        fit_counts = fit_summary.get("optimizer_updates") if isinstance(fit_summary, dict) else None
        update_diagnostics = row.get("update_diagnostics")
        epoch_counts = (
            update_diagnostics.get("epoch") if isinstance(update_diagnostics, dict) else None
        )
        cumulative_counts = (
            update_diagnostics.get("cumulative")
            if isinstance(update_diagnostics, dict)
            else None
        )
        if not all(isinstance(counts, dict) for counts in (fit_counts, epoch_counts, cumulative_counts)):
            raise DriverError(f"{node['id']}: history lacks optimizer update evidence")
        expected_counts = (
            (fit_counts, per_epoch),
            (epoch_counts, per_epoch),
            (cumulative_counts, per_epoch * (index + 1)),
        )
        for counters, expected in expected_counts:
            if (
                type(counters.get("optimizer_attempts")) is not int
                or counters["optimizer_attempts"] != expected
                or type(counters.get("optimizer_completed_updates")) is not int
                or counters["optimizer_completed_updates"] != expected
                or type(counters.get("amp_skipped_updates")) is not int
                or counters["amp_skipped_updates"] != 0
            ):
                raise DriverError(f"{node['id']}: fixed schedule update count or skip count changed")


def verify_run_output(
    root: Path,
    node: dict[str, object],
    split: object,
    expected_source_sha256: str,
) -> dict[str, object]:
    """Verify fixed checkpoint, source identity, history, and primary membership."""
    run_dir = Path(str(node["run_directory"]))
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise DriverError(f"{node['id']}: run directory is missing or unsafe")
    model_path = run_dir / "model.pt"
    endpoint = int(node["endpoint"])
    endpoint_path = run_dir / f"model_ep{endpoint}.pt"
    provenance_path = run_dir / "provenance.json"
    history_path = run_dir / "history.json"
    for path in (model_path, endpoint_path, provenance_path, history_path):
        if path.is_symlink() or not path.is_file():
            raise DriverError(f"{node['id']}: fixed-final artifact is missing: {path.name}")
    provenance = read_json(provenance_path)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    endpoint_checkpoint = torch.load(endpoint_path, map_location="cpu", weights_only=False)
    model_sha = sha256_file(model_path)
    endpoint_sha = sha256_file(endpoint_path)
    hashes = provenance.get("checkpoint_sha256")
    if not isinstance(hashes, dict) or hashes.get("model.pt") != model_sha:
        raise DriverError(f"{node['id']}: selected model hash differs from provenance")
    if hashes.get(endpoint_path.name) != endpoint_sha:
        raise DriverError(f"{node['id']}: fixed endpoint hash differs from provenance")
    if checkpoint.get("epoch") != endpoint or endpoint_checkpoint.get("epoch") != endpoint:
        raise DriverError(f"{node['id']}: model.pt is not the frozen zero-based endpoint")
    if provenance.get("selected_epoch") != endpoint:
        raise DriverError(f"{node['id']}: provenance selected a different checkpoint epoch")
    if checkpoint.get("args") != endpoint_checkpoint.get("args"):
        raise DriverError(f"{node['id']}: selected and endpoint checkpoint arguments differ")
    _verify_checkpoint_args(node, checkpoint.get("args"))
    if provenance.get("split_manifest_sha256") != split.sha256:
        raise DriverError(f"{node['id']}: run provenance split identity changed")
    _verify_membership(provenance, split, str(node["id"]))
    checkpoint_provenance = checkpoint.get("provenance")
    if not isinstance(checkpoint_provenance, dict):
        raise DriverError(f"{node['id']}: checkpoint lacks embedded provenance")
    _verify_membership(checkpoint_provenance, split, str(node["id"]))
    training_source = provenance.get("trainer_implementation")
    if not isinstance(training_source, dict):
        raise DriverError(f"{node['id']}: run lacks exact trainer source identity")
    source_sha = training_source.get("combined_sha256", training_source.get("sha256"))
    if source_sha != expected_source_sha256:
        raise DriverError(f"{node['id']}: trainer source SHA differs from the frozen cell")
    _verify_history(history, node, int(node["expected_updates"]))
    cumulative = provenance.get("cumulative_updates")
    if (
        not isinstance(cumulative, dict)
        or cumulative.get("optimizer_attempts") != int(node["expected_updates"])
        or cumulative.get("optimizer_completed_updates") != int(node["expected_updates"])
        or cumulative.get("amp_skipped_updates") != 0
    ):
        raise DriverError(f"{node['id']}: terminal cumulative optimizer counters differ")
    init = checkpoint["args"].get("init")
    if init:
        run_spec = str(init).partition("@ep")
        parent = root / "artifacts/runs" / run_spec[0] / f"model_ep{run_spec[2]}.pt"
        if not parent.is_file() or provenance.get("initial_checkpoint_sha256") != sha256_file(
            parent
        ):
            raise DriverError(f"{node['id']}: initialized checkpoint ancestry changed")
    cache = provenance.get("distillation_cache")
    if checkpoint["args"].get("distill_cache_root") and (
        not isinstance(cache, dict) or abs(float(cache.get("coefficient_sum", 0.0)) - 0.8125) > 1e-7
    ):
        raise DriverError(f"{node['id']}: W18 cache provenance differs from 0.8125")
    return {
        "run": str(node["run"]),
        "epoch": endpoint,
        "model_sha256": model_sha,
        "endpoint_sha256": endpoint_sha,
        "provenance_sha256": sha256_file(provenance_path),
        "history_sha256": sha256_file(history_path),
        "source_sha256": source_sha,
        "expected_updates": int(node["expected_updates"]),
        "split_manifest_sha256": split.sha256,
    }


def verify_prediction(
    root: Path,
    run: str,
    split_name: str,
    groups: list[int],
    split_sha256: str,
    scope: str,
) -> dict[str, object]:
    """Check producer sidecar and NPY header without loading prediction arrays."""
    path = root / "cache" / f"pred_{run}_{split_name}.npy"
    sidecar = path.with_suffix(".json")
    if path.is_symlink() or sidecar.is_symlink() or not path.is_file() or not sidecar.is_file():
        raise DriverError(f"{run}/{split_name}: prediction output or sidecar is missing")
    metadata = read_json(sidecar)
    checkpoint_path = root / "artifacts/runs" / run / "model.pt"
    provenance_path = checkpoint_path.with_name("provenance.json")
    expected_groups_hash = (
        __import__("hashlib").sha256(np.asarray(groups, dtype="<i8").tobytes()).hexdigest()
    )
    if (
        metadata.get("run") != run
        or metadata.get("split") != split_name
        or metadata.get("row_groups") != groups
        or metadata.get("split_manifest_sha256") != split_sha256
        or metadata.get("campaign_split_manifest_sha256") != split_sha256
        or metadata.get("campaign_export_scope") != scope
        or metadata.get("feature_only") is not True
        or metadata.get("checkpoint_file") != "model.pt"
        or metadata.get("checkpoint_sha256") != sha256_file(checkpoint_path)
        or metadata.get("run_provenance_sha256") != sha256_file(provenance_path)
        or metadata.get("evaluation_membership_sha256") != expected_groups_hash
    ):
        raise DriverError(f"{run}/{split_name}: feature-only export identity or membership changed")
    digest = metadata.get("prediction_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise DriverError(f"{run}/{split_name}: prediction producer did not record its SHA-256")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (len(groups), 20000, 2) or array.dtype != np.float32:
        raise DriverError(f"{run}/{split_name}: full-sequence FP32 output shape changed")
    return {
        "path": str(path),
        "sha256": digest,
        "sidecar_sha256": sha256_file(sidecar),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "group_count": len(groups),
        "row_groups_sha256": expected_groups_hash,
        "scope": scope,
    }


def verify_distillation_cache(plan: dict[str, object]) -> dict[str, object]:
    """Bind W18's cache to all six recovered source coefficients and train-only data."""
    path = Path(str(plan["w18_cache"]))
    metadata_path = path / "metadata.json"
    if path.is_symlink() or metadata_path.is_symlink() or not metadata_path.is_file():
        raise DriverError("W18 mixed target cache is incomplete")
    metadata = read_json(metadata_path)
    split = load_campaign_split(Path(str(plan["split"]["split_manifest"])), Path(str(plan["root"])))
    train_index = Path(str(plan["root"])) / "cache/train_targets.npz"
    with np.load(train_index, allow_pickle=False) as archive:
        train_groups = np.asarray(archive["row_group"], dtype=np.int64)
    expected_train_groups = [
        int(group) for group in train_groups if int(group) in set(split.train_groups)
    ]
    expected_sources = [
        {
            "family": value["family"],
            "run": PREFIX + name,
            "epoch": int(value["epoch"]),
            "coefficient": float(value["coefficient"]),
        }
        for name, value in plan["w18_coefficients"].items()
    ]
    if (
        metadata.get("split_manifest_sha256") != plan["split"]["split_manifest_sha256"]
        or metadata.get("source_coefficients") != expected_sources
        or abs(float(metadata.get("coefficient_sum", 0.0)) - 0.8125) > 1e-7
        or metadata.get("output_profile") != "all"
        or metadata.get("fit_groups", {}).get("train") != list(split.train_groups)
        or metadata.get("fit_groups", {}).get("valid") != []
        or metadata.get("evaluation_valid_groups") != list(split.evaluation_groups)
        or metadata.get("stored_targets")
        != [
            "main_source",
            "main_normalized",
            "main_homogeneous",
            "families",
        ]
    ):
        raise DriverError("W18 cache metadata differs from the frozen chronological recipe")
    files = metadata.get("files")
    if not isinstance(files, dict) or len(files) != 8:
        raise DriverError("W18 cache does not contain all four train/empty-valid targets")
    expected_keys = {
        f"{target}_{split_name}"
        for target in ("main_source", "main_normalized", "main_homogeneous", "families")
        for split_name in ("train", "valid")
    }
    if set(files) != expected_keys:
        raise DriverError("W18 cache target file inventory changed")
    for key, record in files.items():
        if not isinstance(record, dict):
            raise DriverError(f"W18 cache file record is invalid: {key}")
        relative = Path(str(record.get("path", "")))
        if relative.is_absolute() or len(relative.parts) != 1:
            raise DriverError(f"W18 cache filename escapes its root: {key}")
        target = path / relative
        if (
            target.is_symlink()
            or not target.is_file()
            or target.stat().st_size != record.get("bytes")
        ):
            raise DriverError(f"W18 cache file is missing or changed: {key}")
        if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
            raise DriverError(f"W18 cache file lacks a producer hash: {key}")
        expected_groups = expected_train_groups if key.endswith("_train") else []
        if record.get("row_groups") != expected_groups:
            raise DriverError(f"W18 cache target membership changed: {key}")
        if record.get("row_groups_sha256") != row_groups_sha256(expected_groups):
            raise DriverError(f"W18 cache target membership digest changed: {key}")
        target_width = 8 if key.startswith("families_") else 2
        expected_shape = [len(expected_groups), 20000, target_width]
        expected_bytes = (
            int(np.prod(expected_shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
        )
        if (
            record.get("shape") != expected_shape
            or record.get("dtype") != "float32"
            or int(record.get("bytes", -1)) < expected_bytes
        ):
            raise DriverError(f"W18 cache array metadata changed: {key}")
        array = np.load(target, mmap_mode="r" if expected_groups else None, allow_pickle=False)
        if array.shape != tuple(expected_shape) or array.dtype != np.float32:
            raise DriverError(f"W18 cache array header changed: {key}")
        del array
        if sha256_file(target) != record["sha256"]:
            raise DriverError(f"W18 cache file SHA-256 changed: {key}")
    return {
        "path": str(path),
        "metadata_sha256": sha256_file(metadata_path),
        "file_count": len(files),
        "coefficient_sum": metadata["coefficient_sum"],
        "split_manifest_sha256": metadata["split_manifest_sha256"],
    }


def _remaining_write_bytes(
    root: Path,
    plan: dict[str, object],
    status: dict[str, object],
    train_group_count: int,
) -> int:
    """Conservatively reserve all still-required prediction/cache/model outputs."""
    nodes = _node_map(plan)
    node_status = status["nodes"]
    remaining_train_exports = sum(
        1
        for name, node in nodes.items()
        if node.get("export_train")
        and node_status[name].get("phase")
        not in {
            "train_export_running",
            "train_export_complete",
            "fit_failed",
        }
        and node_status[name].get("state") not in {"failed", "blocked"}
    )
    remaining_valid_exports = sum(
        1
        for run, phase in status["final_exports"].items()
        if _stage_state(phase) not in {"complete", "running", "failed", "blocked"}
    )
    train_bytes = expected_prediction_bytes(train_group_count)
    valid_bytes = expected_prediction_bytes(VALID_GROUPS)
    cache_path = Path(str(plan["w18_cache"]))
    cache_bytes = 0
    if not cache_path.exists() and status["cache"]["state"] == "not_started":
        from scripts.build_distill_cache import estimate_output_bytes

        cache_bytes = estimate_output_bytes(TRAIN_GROUPS, "all")
    remaining_fits = sum(
        1
        for name in nodes
        if status["nodes"][name].get("state") in {"not_started", "running"} and name != "fin256_s3"
    )
    return (
        remaining_train_exports * train_bytes
        + remaining_valid_exports * valid_bytes
        + cache_bytes
        + MODEL_ARTIFACT_RESERVE_BYTES
        + remaining_fits * (32 << 20)
    )


def _require_disk_headroom(root: Path, remaining: int) -> None:
    """Keep at least 12 GiB free after projected bounded artifacts."""
    free = shutil.disk_usage(root / "cache").free
    if free - remaining < MIN_FREE_BYTES:
        raise DriverError(
            f"disk plan needs {remaining} additional bytes plus the 12 GiB free reserve; "
            f"currently free={free}"
        )


def _log_path(state_dir: Path, task_id: str) -> Path:
    """Map a task ID to a private, unique log path."""
    safe = task_id.replace(":", "__").replace("/", "_")
    return state_dir / "logs" / f"{safe}.log"


def _mark_launch_failed(status: dict[str, object], task: dict[str, object], message: str) -> None:
    """Fail one immutable task intent without leaving a runnable node behind."""
    now = time.time()
    task.update({"state": "failed", "finished_unix": now, "error": message})
    if task.get("popen_started") and task.get("gpu_index") is not None:
        status["gpu_seconds_completed"] = float(status["gpu_seconds_completed"]) + max(
            0.0, now - float(task["started_unix"])
        )
    kind, node_id = task.get("kind"), task.get("node_id")
    if kind in {"fit", "train_export"} and isinstance(node_id, str):
        status["nodes"][node_id].update({"state": "failed", "phase": str(kind), "error": message})
    elif kind == "valid_export" and isinstance(node_id, str):
        status["final_exports"][node_id] = "failed"
    elif kind == "cache":
        status["cache"] = {"state": "failed", "error": message}
    elif kind == "score":
        status["score"] = {"state": "failed", "error": message}
    status["last_error"] = message


def _stop_spawned_child(child: subprocess.Popen[bytes]) -> None:
    """Terminate and reap one exact Popen child, escalating to kill after a bound."""
    if child.poll() is not None:
        child.wait()
        return
    try:
        child.terminate()
    except ProcessLookupError:
        pass
    except OSError:
        # A concurrent exit or a failed TERM still gets a bounded wait and KILL attempt.
        pass
    try:
        child.wait(timeout=SPAWN_CLEANUP_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    except OSError:
        pass
    try:
        child.kill()
    except ProcessLookupError:
        pass
    except OSError as error:
        raise DriverError(f"failed to kill spawned child PID {child.pid}: {error}") from error
    try:
        child.wait(timeout=SPAWN_CLEANUP_SECONDS)
    except subprocess.TimeoutExpired as error:
        raise DriverError(f"spawned child PID {child.pid} survived TERM and KILL") from error
    except OSError as error:
        raise DriverError(f"could not reap spawned child PID {child.pid}: {error}") from error


def _launch_task(
    *,
    root: Path,
    state_dir: Path,
    state_path: Path,
    status: dict[str, object],
    children: dict[str, subprocess.Popen[bytes]],
    task_id: str,
    kind: str,
    command: list[str],
    gpu: tuple[int, str] | None,
    node_id: str | None,
    cpu_affinity: tuple[int, ...] | None = None,
    expected_source_sha256: str | None = None,
) -> None:
    """Persist one intent, then spawn one detached task in its own session."""
    tasks = status["tasks"]
    if task_id in tasks:
        raise DriverError(f"task already has an immutable launch record: {task_id}")
    if (gpu is None) != (cpu_affinity is None):
        raise DriverError("GPU task and receipt-bound CPU affinity must be supplied together")
    launch_command = command
    if gpu is not None and cpu_affinity is not None:
        launch_command = ["taskset", "-c", ",".join(map(str, cpu_affinity)), *command]
    args_sha256 = canonical_sha256(command[command.index("--name") :]) if kind == "fit" else None
    log_path = _log_path(state_dir, task_id)
    log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    record: dict[str, object] = {
        "task_id": task_id,
        "kind": kind,
        "node_id": node_id,
        "state": "launch_intent",
        "adopted": False,
        "command": launch_command,
        "command_sha256": canonical_sha256(launch_command),
        "source_sha256": expected_source_sha256,
        "args_sha256": args_sha256,
        "log": str(log_path),
        "gpu_index": None if gpu is None else gpu[0],
        "gpu_uuid": None if gpu is None else gpu[1],
        "cpu_affinity": None if cpu_affinity is None else list(cpu_affinity),
        "started_unix": time.time(),
    }
    tasks[task_id] = record
    _write_status(state_path, status)
    environment = _child_environment(root, None if gpu is None else gpu[1])
    try:
        descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        log_file = os.fdopen(descriptor, "ab", buffering=0)
    except OSError as error:
        _mark_launch_failed(status, record, f"{type(error).__name__}: {error}")
        _write_status(state_path, status)
        return
    try:
        with log_file:
            child = subprocess.Popen(
                launch_command,
                cwd=root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
            )
    except OSError as error:
        _mark_launch_failed(status, record, f"{type(error).__name__}: {error}")
        _write_status(state_path, status)
        return
    children[task_id] = child
    record["popen_started"] = True
    if kind == "fit":
        status["fits_used"] = int(status["fits_used"]) + 1
    identity_deadline = time.monotonic() + 2.0
    identity = proc_identity(child.pid)
    while identity is None and time.monotonic() < identity_deadline:
        time.sleep(0.01)
        identity = proc_identity(child.pid)
    if (
        identity is None
        or identity["process_group"] != child.pid
        or identity["session"] != child.pid
    ):
        record["pid"] = child.pid
        failure = "spawned child did not establish its receipt-bound process group"
        try:
            _stop_spawned_child(child)
        except DriverError as error:
            record.update(
                {
                    "state": "running",
                    "spawn_cleanup_pending": True,
                    "spawn_failure": failure,
                    "cleanup_error": str(error),
                }
            )
            if kind == "fit":
                status["nodes"][str(node_id)].update(
                    {
                        "state": "running",
                        "phase": "spawn_cleanup_pending",
                        "cell_started_unix": record["started_unix"],
                    }
                )
            _write_status(state_path, status)
            raise DriverError(
                f"{task_id}: child cleanup remains pending; its GPU stays reserved: {error}"
            ) from error
        children.pop(task_id, None)
        _mark_launch_failed(status, record, failure)
        _write_status(state_path, status)
        return
    record.update(
        {
            "state": "running",
            "pid": child.pid,
            "start_ticks": str(identity["start_ticks"]),
            "process_group": child.pid,
            "session": child.pid,
        }
    )
    if kind == "fit":
        node = status["nodes"][str(node_id)]
        node.update(
            {
                "state": "running",
                "phase": "fit",
                "cell_started_unix": record["started_unix"],
                "source_sha256": expected_source_sha256,
                "args_sha256": args_sha256,
            }
        )
    elif kind == "train_export":
        status["nodes"][str(node_id)]["phase"] = "train_export_running"
    elif kind == "valid_export":
        status["final_exports"][str(node_id)] = "running"
    elif kind == "cache":
        status["cache"] = {"state": "running", "task_id": task_id}
    elif kind == "score":
        status["score"] = {"state": "running", "task_id": task_id}
    _write_status(state_path, status)


def _task_deadline(status: dict[str, object], task: dict[str, object]) -> float:
    """Use the fixed six-hour per-cell wall bound for every finite subtask."""
    node_id = task.get("node_id")
    if isinstance(node_id, str) and node_id in status["nodes"]:
        started = status["nodes"][node_id].get("cell_started_unix")
        if isinstance(started, (int, float)):
            return float(started) + MAX_CELL_SECONDS
    return float(task["started_unix"]) + MAX_CELL_SECONDS


def _record_task_failure(
    status: dict[str, object], task: dict[str, object], message: str, now: float
) -> None:
    """Fail one immutable attempt; descendants are blocked and no retry is created."""
    task["state"] = "failed"
    task["finished_unix"] = now
    task["error"] = message
    if task.get("gpu_index") is not None:
        status["gpu_seconds_completed"] = float(status["gpu_seconds_completed"]) + max(
            0.0, now - float(task["started_unix"])
        )
    kind, node_id = task.get("kind"), task.get("node_id")
    if kind in {"fit", "train_export"} and isinstance(node_id, str):
        status["nodes"][node_id].update({"state": "failed", "phase": str(kind), "error": message})
    elif kind == "cache":
        status["cache"] = {"state": "failed", "error": message}
    elif kind == "valid_export" and isinstance(node_id, str):
        status["final_exports"][node_id] = "failed"
    elif kind == "score":
        status["score"] = {"state": "failed", "error": message}


def _complete_task(
    root: Path,
    status_path: Path,
    status: dict[str, object],
    plan: dict[str, object],
    split: object,
    task: dict[str, object],
    now: float,
) -> None:
    """Verify each task's durable outputs before marking its dependents ready."""
    kind = str(task["kind"])
    if not task.get("adopted") and task.get("return_code") != 0:
        _record_task_failure(status, task, f"child exited with {task.get('return_code')}", now)
        return
    if kind == "fit":
        node = _node_map(plan)[str(task["node_id"])]
        source_sha = str(task["source_sha256"])
        verified = verify_run_output(root, node, split, source_sha)
        status["nodes"][str(node["id"])]["verified_run"] = verified
        if node["export_train"]:
            status["nodes"][str(node["id"])].update({"state": "running", "phase": "export_pending"})
        else:
            status["nodes"][str(node["id"])].update(
                {"state": "complete", "phase": "fit_complete", "completed_unix": now}
            )
    elif kind == "train_export":
        node = _node_map(plan)[str(task["node_id"])]
        groups = _prediction_groups(root, Path(str(plan["split"]["split_manifest"])), "train")
        receipt = verify_prediction(
            root,
            str(node["run"]),
            "train",
            groups,
            str(plan["split"]["split_manifest_sha256"]),
            TRAIN_EXPORT_SCOPE,
        )
        status["nodes"][str(node["id"])].update(
            {
                "state": "complete",
                "phase": "train_export_complete",
                "train_export": receipt,
                "completed_unix": now,
            }
        )
    elif kind == "cache":
        cache_receipt = verify_distillation_cache(plan)
        status["cache"] = {"state": "complete", **cache_receipt, "completed_unix": now}
    elif kind == "valid_export":
        run = str(task["node_id"])
        groups = _prediction_groups(root, Path(str(plan["split"]["split_manifest"])), "valid")
        receipt = verify_prediction(
            root,
            run,
            "valid",
            groups,
            str(plan["split"]["split_manifest_sha256"]),
            "evaluation",
        )
        status["final_exports"][run] = {"state": "complete", **receipt}
    elif kind == "score":
        score_path = root / "final-score.json"
        score = read_json(score_path)
        if (
            score.get("schema") != "wnn.tl3-chronological-score.v1"
            or score.get("split_manifest_sha256") != plan["split"]["split_manifest_sha256"]
            or score.get("valid_sequences") != VALID_GROUPS
            or score.get("selected_rows") != 3528273
        ):
            raise DriverError("fixed scorer output does not cover the official-valid identity")
        status["score"] = {
            "state": "complete",
            "path": str(score_path),
            "sha256": sha256_file(score_path),
            "completed_unix": now,
        }
    else:
        raise DriverError(f"unknown finite driver task kind: {kind}")
    task["state"] = "complete"
    task["finished_unix"] = now
    task["return_code"] = 0 if task.get("return_code") is None else task["return_code"]
    if task.get("gpu_index") is not None:
        status["gpu_seconds_completed"] = float(status["gpu_seconds_completed"]) + max(
            0.0, now - float(task["started_unix"])
        )
    _write_status(status_path, status)


def _poll_tasks(
    root: Path,
    state_path: Path,
    status: dict[str, object],
    plan: dict[str, object],
    split: object,
    children: dict[str, subprocess.Popen[bytes]],
    now: float,
) -> None:
    """Monitor child ownership and outputs; only the live controller owns Popen return codes.

    After controller restart, an exited non-adopted child lacks a trusted return code and is
    marked failed without artifact adoption. Spawn-cleanup retries likewise require the local
    Popen handle; a restart refuses to release that GPU or retry the fit.
    """
    for task_id, task in list(_active_tasks(status).items()):
        if task.get("spawn_cleanup_pending"):
            child = children.get(task_id)
            if child is None:
                raise DriverError(
                    f"{task_id}: spawn cleanup lost its Popen handle; refusing to retry or release GPU"
                )
            try:
                _stop_spawned_child(child)
            except DriverError as error:
                task["cleanup_error"] = str(error)
                status["last_error"] = str(error)
                _write_status(state_path, status)
                continue
            children.pop(task_id, None)
            _mark_launch_failed(status, task, str(task["spawn_failure"]))
            _write_status(state_path, status)
            continue
        if task.get("command_sha256") != canonical_sha256(task.get("command")) and not task.get(
            "adopted"
        ):
            _record_task_failure(status, task, "stored child command hash changed", now)
            continue
        if now >= _task_deadline(status, task):
            try:
                terminated = terminate_owned_group(task)
            except DriverError as error:
                _record_task_failure(status, task, f"timeout ownership check failed: {error}", now)
                continue
            _record_task_failure(
                status,
                task,
                "six-hour bounded task timeout"
                if terminated
                else "task survived owned-group timeout",
                now,
            )
            continue
        try:
            group_state = owned_group_state(task)
        except (DriverError, ValueError) as error:
            _record_task_failure(status, task, f"process identity check failed: {error}", now)
            continue
        if group_state == "running":
            continue
        if group_state != "exited":
            _record_task_failure(status, task, f"process group is not owned: {group_state}", now)
            continue
        child = children.get(task_id)
        if child is not None:
            return_code = child.poll()
            if return_code is None:
                try:
                    return_code = child.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    continue
            task["return_code"] = int(return_code)
        elif not task.get("adopted") and task.get("return_code") is None:
            _record_task_failure(status, task, "child receipt vanished before completion", now)
            continue
        if not task.get("adopted") and task.get("return_code") != 0:
            _record_task_failure(status, task, f"child exited with {task.get('return_code')}", now)
            continue
        try:
            _complete_task(root, state_path, status, plan, split, task, now)
        except (DriverError, OSError, ValueError, KeyError, TypeError) as error:
            _record_task_failure(status, task, f"terminal artifact check failed: {error}", now)


def _block_failed_descendants(status: dict[str, object], plan: dict[str, object]) -> None:
    """Mark only descendants of failed nodes as blocked; keep independent fits runnable."""
    nodes = _node_map(plan)
    changed = True
    while changed:
        changed = False
        for node_id, node in nodes.items():
            current = status["nodes"][node_id]
            if current.get("state") != "not_started":
                continue
            parents = [str(parent) for parent in node["parents"]]
            if any(status["nodes"][parent]["state"] in {"failed", "blocked"} for parent in parents):
                current.update({"state": "blocked", "phase": "dependency_failed"})
                changed = True


def _all_w18_sources_complete(status: dict[str, object]) -> bool:
    """Check the exact six target-cache teachers, including their train exports."""
    return all(status["nodes"][name]["state"] == "complete" for name in W18_COEFFICIENTS)


def _dependency_ready(node: dict[str, object], status: dict[str, object]) -> bool:
    """Require each parent export and, for W18, its final six-source cache."""
    if any(status["nodes"][str(parent)]["state"] != "complete" for parent in node["parents"]):
        return False
    if node["id"] == "w18final_distilled_s5":
        return status["cache"]["state"] == "complete"
    return True


def _start_ready_gpu_task(
    *,
    root: Path,
    python: Path,
    state_dir: Path,
    state_path: Path,
    status: dict[str, object],
    plan: dict[str, object],
    gpu: tuple[int, str],
    cpu_affinity: tuple[int, ...],
    children: dict[str, subprocess.Popen[bytes]],
) -> bool:
    """Start one eligible train/export task on a physically empty allowed GPU."""
    nodes = _node_map(plan)
    split_path = Path(str(plan["split"]["split_manifest"]))
    ordered_nodes = sorted(
        nodes.items(),
        key=lambda item: (
            NODE_PRIORITY.index(item[0]) if item[0] in NODE_PRIORITY else len(NODE_PRIORITY)
        ),
    )
    for node_id, node in ordered_nodes:
        node_state = status["nodes"][node_id]
        if node_state.get("phase") == "export_pending":
            run = str(node["run"])
            output_paths = _prediction_output_paths(root, run, "train")
            if any(path.exists() or path.is_symlink() for path in output_paths):
                _record_unstarted_failure(
                    status,
                    node_id,
                    "unowned train prediction or partial output already exists",
                )
                _write_status(state_path, status)
                return True
            command = _predict_command(root, python, split_path, run, "train", "train-index")
            task_id = f"train-export:{node_id}"
            _launch_task(
                root=root,
                state_dir=state_dir,
                state_path=state_path,
                status=status,
                children=children,
                task_id=task_id,
                kind="train_export",
                command=command,
                gpu=gpu,
                node_id=node_id,
                cpu_affinity=cpu_affinity,
                expected_source_sha256=str(node_state["source_sha256"]),
            )
            return True

    for run, record in status["final_exports"].items():
        if record != "not_started":
            continue
        node = next((item for item in nodes.values() if item["run"] == run), None)
        if node is None:
            raise DriverError(f"final run is missing from the frozen DAG: {run}")
        if status["nodes"][str(node["id"])]["state"] != "complete":
            continue
        output_paths = _prediction_output_paths(root, run, "valid")
        if any(path.exists() or path.is_symlink() for path in output_paths):
            status["final_exports"][run] = "failed"
            status["last_error"] = f"unowned final prediction output already exists: {run}"
            _write_status(state_path, status)
            return True
        command = _predict_command(root, python, split_path, run, "valid", "valid-evaluation")
        _launch_task(
            root=root,
            state_dir=state_dir,
            state_path=state_path,
            status=status,
            children=children,
            task_id=f"valid-export:{run}",
            kind="valid_export",
            command=command,
            gpu=gpu,
            node_id=run,
            cpu_affinity=cpu_affinity,
            expected_source_sha256=str(plan["source_identity"]["training_source_sha256"]),
        )
        return True

    for node_id, node in ordered_nodes:
        node_state = status["nodes"][node_id]
        if node_state.get("state") != "not_started" or not _dependency_ready(node, status):
            continue
        if int(status["fits_used"]) >= FIT_LIMIT:
            return False
        current_source = source_identity(root)
        if (
            current_source["training_source_sha256"]
            != plan["source_identity"]["training_source_sha256"]
        ):
            _record_unstarted_failure(
                status, node_id, "frozen training source changed before launch"
            )
            _write_status(state_path, status)
            return True
        run_dir = Path(str(node["run_directory"]))
        if run_dir.exists() or run_dir.is_symlink():
            _record_unstarted_failure(status, node_id, "unowned run directory already exists")
            _write_status(state_path, status)
            return True
        command = _fit_command(root, python, node)
        _launch_task(
            root=root,
            state_dir=state_dir,
            state_path=state_path,
            status=status,
            children=children,
            task_id=f"fit:{node_id}",
            kind="fit",
            command=command,
            gpu=gpu,
            node_id=node_id,
            cpu_affinity=cpu_affinity,
            expected_source_sha256=str(plan["source_identity"]["training_source_sha256"]),
        )
        return True
    return False


def _record_unstarted_failure(status: dict[str, object], node_id: str, message: str) -> None:
    """Close a duplicate/unsafe output path without launching or retrying a fit."""
    status["nodes"][node_id].update({"state": "failed", "phase": "preflight", "error": message})
    status["last_error"] = f"{node_id}: {message}"


def _start_cache_if_ready(
    *,
    root: Path,
    python: Path,
    state_dir: Path,
    state_path: Path,
    status: dict[str, object],
    plan: dict[str, object],
    children: dict[str, subprocess.Popen[bytes]],
) -> bool:
    """Build W18 cache as soon as its six required train exports finish."""
    if status["cache"]["state"] != "not_started" or not _all_w18_sources_complete(status):
        return False
    output = Path(str(plan["w18_cache"]))
    if output.exists() or output.is_symlink():
        _record_cache_failure(status, "unowned W18 cache directory already exists")
        _write_status(state_path, status)
        return True
    command = _cache_command(root, python, plan)
    _launch_task(
        root=root,
        state_dir=state_dir,
        state_path=state_path,
        status=status,
        children=children,
        task_id="cache:w18-mixed-targets",
        kind="cache",
        command=command,
        gpu=None,
        node_id=None,
    )
    return True


def _record_cache_failure(status: dict[str, object], message: str) -> None:
    """Persist a nonretryable cache-stage failure."""
    status["cache"] = {"state": "failed", "error": message}
    status["last_error"] = message


def _start_score_if_ready(
    *,
    root: Path,
    python: Path,
    state_dir: Path,
    state_path: Path,
    status: dict[str, object],
    plan: dict[str, object],
    children: dict[str, subprocess.Popen[bytes]],
) -> bool:
    """Invoke the fixed scorer only after all three official-valid exports finish."""
    if status["score"]["state"] != "not_started" or _active_tasks(status):
        return False
    if any(value == "failed" for value in status["final_exports"].values()):
        status["score"] = {"state": "blocked", "reason": "a final member export failed"}
        return True
    if not all(
        isinstance(value, dict) and value.get("state") == "complete"
        for value in status["final_exports"].values()
    ):
        return False
    command = _score_command(root, python, plan)
    output = root / "final-score.json"
    if output.exists() or output.is_symlink():
        status["score"] = {"state": "failed", "error": "score output already exists"}
        return True
    _launch_task(
        root=root,
        state_dir=state_dir,
        state_path=state_path,
        status=status,
        children=children,
        task_id="score:fixed-three-member-valid",
        kind="score",
        command=command,
        gpu=None,
        node_id=None,
    )
    return True


def _gpu_seconds(status: dict[str, object], now: float) -> float:
    """Include active receipt-bound task time in the hard 256 GPU-hour budget."""
    active = sum(
        max(0.0, now - float(task["started_unix"]))
        for task in _active_tasks(status).values()
        if task.get("gpu_index") is not None
    )
    return float(status["gpu_seconds_completed"]) + active


def _terminate_active(
    status: dict[str, object], children: dict[str, subprocess.Popen[bytes]]
) -> list[str]:
    """Stop receipt-bound groups or exact Popen handles awaiting spawn cleanup."""
    errors = []
    for task_id, task in _active_tasks(status).items():
        if task.get("spawn_cleanup_pending"):
            child = children.get(task_id)
            if child is None:
                errors.append(f"{task_id}: spawn cleanup lost its Popen handle")
                continue
            try:
                _stop_spawned_child(child)
            except DriverError as error:
                errors.append(f"{task_id}: {error}")
                continue
            children.pop(task_id, None)
            _mark_launch_failed(status, task, str(task["spawn_failure"]))
            continue
        try:
            if not terminate_owned_group(task):
                errors.append(f"{task['task_id']}: owned process group did not exit")
        except DriverError as error:
            errors.append(f"{task['task_id']}: {error}")
    return errors


def _terminalize(
    state_dir: Path,
    state_path: Path,
    status: dict[str, object],
    plan: dict[str, object],
    terminal_status: str,
    reason: str | None = None,
) -> None:
    """Write a durable terminal summary exactly once."""
    status["status"] = terminal_status
    status["finished_unix"] = time.time()
    if reason is not None:
        status["last_error"] = reason
    _write_status(state_path, status)
    terminal = {
        "schema": "wnn.tl3-finite17-terminal.v1",
        "status": terminal_status,
        "plan_sha256": plan["sha256"],
        "state_sha256": sha256_file(state_path),
        "fits_used": status["fits_used"],
        "gpu_seconds": _gpu_seconds(status, float(status["finished_unix"])),
        "score": status["score"],
        "reason": reason,
    }
    write_atomic_json(state_dir / "terminal.json", terminal)


def _mark_blocked_stages(status: dict[str, object], plan: dict[str, object]) -> None:
    """Block only impossible cache, W18, final-export, or score stages after failures."""
    _block_failed_descendants(status, plan)
    if status["cache"]["state"] == "not_started" and any(
        status["nodes"][name]["state"] in {"failed", "blocked"} for name in W18_COEFFICIENTS
    ):
        status["cache"] = {"state": "blocked", "reason": "a W18 teacher failed"}
    if status["cache"]["state"] in {"failed", "blocked"}:
        status["nodes"]["w18final_distilled_s5"].update(
            {"state": "blocked", "phase": "cache_unavailable"}
        )
    for run in plan["final_runs"]:
        node = next(item for item in _node_map(plan).values() if item["run"] == run)
        if (
            status["nodes"][str(node["id"])]["state"] in {"failed", "blocked"}
            and status["final_exports"][run] == "not_started"
        ):
            status["final_exports"][run] = "blocked"
    if status["score"]["state"] == "not_started" and any(
        _stage_state(value) in {"failed", "blocked"} for value in status["final_exports"].values()
    ):
        status["score"] = {"state": "blocked", "reason": "a final member is unavailable"}


def _has_possible_work(status: dict[str, object], plan: dict[str, object]) -> bool:
    """Report whether any finite DAG node or terminal stage can still advance."""
    if _active_tasks(status):
        return True
    if status["cache"]["state"] in {"not_started", "running"}:
        return True
    if any(record["state"] in {"not_started", "running"} for record in status["nodes"].values()):
        return True
    if any(
        _stage_state(value) in {"not_started", "running"}
        for value in status["final_exports"].values()
    ):
        return True
    return status["score"]["state"] in {"not_started", "running"}


def _enforce_budgets(
    state_dir: Path,
    state_path: Path,
    status: dict[str, object],
    plan: dict[str, object],
    now: float,
    children: dict[str, subprocess.Popen[bytes]],
) -> bool:
    """Stop only owned groups when wall or GPU-hour authority is exhausted."""
    wall = now - float(status["started_unix"])
    gpu = _gpu_seconds(status, now)
    if wall > MAX_WALL_SECONDS:
        errors = _terminate_active(status, children)
        _terminalize(
            state_dir,
            state_path,
            status,
            plan,
            "budget_exhausted",
            "168-hour wall cap reached; " + "; ".join(errors),
        )
        return True
    if gpu > MAX_GPU_SECONDS:
        errors = _terminate_active(status, children)
        _terminalize(
            state_dir,
            state_path,
            status,
            plan,
            "budget_exhausted",
            "256 GPU-hour cap reached; " + "; ".join(errors),
        )
        return True
    return False


def _file_lock(state_dir: Path):
    """Open the cross-SSH single-controller lock on Linux."""
    try:
        import fcntl
    except ImportError as error:
        raise DriverError("the finite receiver controller requires Linux flock") from error
    handle = (state_dir / "driver.lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise DriverError("another finite TL3 controller already holds this state") from error
    return handle


def run_controller(args: argparse.Namespace) -> int:
    """Run or resume the finite TL3 graph without touching unrelated GPU processes."""
    if not sys.platform.startswith("linux"):
        raise DriverError("finite GPU scheduling is available only on the receiving Linux host")
    root = args.root.resolve(strict=True)
    split_path = args.split_manifest.resolve(strict=True)
    state_dir = args.state.expanduser().absolute()
    if state_dir.exists() and state_dir.is_symlink():
        raise DriverError("state path cannot be a symlink")
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = _file_lock(state_dir)
    try:
        adoption = read_json(args.adoption_receipt)
        plan = build_plan(root, split_path, state_dir, adoption)
        _validate_plan(plan)
        first_node = _node_map(plan)["fin256_s3"]
        adoption_identity = validate_adoption(adoption, first_node, root)
        devices, applications = nvidia_inventory()
        if devices.get(int(adoption_identity["gpu_index"])) != adoption_identity["gpu_uuid"]:
            raise DriverError("adopted fin256_s3 physical GPU UUID differs from current inventory")
        if owned_group_state(adoption_identity) not in {"running", "exited"}:
            raise DriverError("adopted fin256_s3 PID/session no longer matches its receipt")
        status, resumed = _load_or_create_state(state_dir, plan, adoption_identity)
        status["controller"] = _controller_identity()
        state_path = _state_paths(state_dir)[1]
        write_atomic_json(state_path, status)
        split = load_campaign_split(split_path, root)
        train_groups, _ = verify_source_index_receipt(root, "train")
        private_index = plan.get("private_train_index")
        if not isinstance(private_index, dict) or private_index.get("group_count") != len(
            train_groups
        ):
            raise DriverError("private row-group-only train index is not receipt-bound")
        if (root / "final-score.json").exists():
            raise DriverError("fixed score output already exists; use a fresh chain state")
        if not resumed:
            _require_disk_headroom(
                root,
                _remaining_write_bytes(root, plan, status, len(train_groups)),
            )
        children: dict[str, subprocess.Popen[bytes]] = {}
        stop = {"requested": False, "signal": None}

        def request_stop(signum: int, _frame: object) -> None:
            stop["requested"] = True
            stop["signal"] = signum

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        python = args.python.resolve(strict=True)
        if not python.is_file():
            raise DriverError("--python must point to the isolated receiver interpreter")
        poll_seconds = args.poll_seconds
        if not 1 <= poll_seconds <= 300:
            raise DriverError("poll interval must be from 1 through 300 seconds")
        while True:
            now = time.time()
            if stop["requested"]:
                errors = _terminate_active(status, children)
                reason = f"controller received signal {stop['signal']}; " + "; ".join(errors)
                _terminalize(state_dir, state_path, status, plan, "interrupted", reason)
                return 130
            if _enforce_budgets(state_dir, state_path, status, plan, now, children):
                return 2
            _poll_tasks(root, state_path, status, plan, split, children, now)
            _mark_blocked_stages(status, plan)
            if status["status"] != "running":
                return 1
            if status["score"]["state"] == "complete":
                _terminalize(state_dir, state_path, status, plan, "completed")
                return 0
            if status["fits_used"] > FIT_LIMIT:
                _terminalize(state_dir, state_path, status, plan, "failed", "fit cap exceeded")
                return 2
            if not _has_possible_work(status, plan):
                _terminalize(
                    state_dir,
                    state_path,
                    status,
                    plan,
                    "finished_with_failures",
                    "no runnable nodes remain",
                )
                return 1
            if _start_cache_if_ready(
                root=root,
                python=python,
                state_dir=state_dir,
                state_path=state_path,
                status=status,
                plan=plan,
                children=children,
            ):
                time.sleep(poll_seconds)
                continue
            remaining = _remaining_write_bytes(root, plan, status, len(train_groups))
            try:
                _require_disk_headroom(root, remaining)
            except DriverError as error:
                status["last_error"] = str(error)
                _terminalize(
                    state_dir, state_path, status, plan, "disk_capacity_blocked", str(error)
                )
                return 2
            devices, applications = nvidia_inventory()
            protected = protected_gpu_indices(args.protected_galam_receipt, devices)
            cpu_affinities = gpu_cpu_affinities(args.protected_galam_receipt, devices)
            free = free_gpu_devices(
                devices,
                applications,
                protected,
                _gpu_reserved(status),
            )
            for gpu in free:
                if gpu[0] in _gpu_reserved(status):
                    continue
                try:
                    started = _start_ready_gpu_task(
                        root=root,
                        python=python,
                        state_dir=state_dir,
                        state_path=state_path,
                        status=status,
                        plan=plan,
                        gpu=gpu,
                        cpu_affinity=cpu_affinities[gpu[0]],
                        children=children,
                    )
                except (DriverError, OSError, ValueError) as error:
                    status["last_error"] = str(error)
                    _write_status(state_path, status)
                    started = True
                if started:
                    continue
            if _start_score_if_ready(
                root=root,
                python=python,
                state_dir=state_dir,
                state_path=state_path,
                status=status,
                plan=plan,
                children=children,
            ):
                time.sleep(poll_seconds)
                continue
            _write_status(state_path, status)
            time.sleep(poll_seconds)
    finally:
        lock.close()


def main() -> None:
    """Parse receiver paths and run only the frozen finite chain."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--adoption-receipt", required=True, type=Path)
    parser.add_argument("--protected-galam-receipt", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    args = parser.parse_args()
    try:
        exit_code = run_controller(args)
    except (DriverError, OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"finite TL3 driver blocked: {error}\n")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
