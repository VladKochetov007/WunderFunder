"""Run six R3 B300 engineering profiles with UUID pins and bounded ownership."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

PRIMARY_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
AUXILIARY_SHA256 = "26b448bad7f60612cea482f4abaf5f184e5a724228eef227d7e78a8155190075"
REGISTRY_SHA256 = "dc0cb4d1957b354509ae46f88b6bdf3708b44adc8ef6f375d2ad430f4a9a07a1"
SOURCE_CAPSULE_SHA256 = "41929c513e8e585c15424560ef9e4c2ceaee842efb641c73dd365848fd9d311b8"
TRAIN_SHA256 = "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81"
VALID_SHA256 = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
GIB = 1 << 30
PROFILE_SECONDS = 180
WATCHDOG_GRACE_SECONDS = 5
POSTFLIGHT_RESERVE_SECONDS = 5


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from a regular file."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    """Atomically retain launcher evidence and flush it to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    encoded = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def process_identity(pid: int) -> dict[str, int | str] | None:
    """Capture PID, Linux birth counter, and process-group ID from procfs."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    fields = raw.rsplit(")", 1)[-1].split()
    if len(fields) <= 19:
        return None
    return {"pid": pid, "birth": fields[19], "pgid": int(fields[2])}


def signal_owned(owner: dict[str, int | str] | None, signum: int) -> bool:
    """Signal only a still-identical process group rooted at the recorded PID."""
    if not owner or process_identity(int(owner["pid"])) != owner:
        return False
    if owner["pgid"] != owner["pid"]:
        return False
    try:
        os.killpg(int(owner["pgid"]), signum)
    except ProcessLookupError:
        return False
    return True


def _query_gpu(gpu_uuid: str, timeout: float) -> dict[str, object]:
    """Read one exact GPU UUID's identity and memory without changing device state."""
    query = "index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu"
    result = subprocess.run(
        ["nvidia-smi", "--id", gpu_uuid, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    rows = list(csv.reader(result.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 7:
        raise ValueError(f"NVML query returned an unexpected row for {gpu_uuid}")
    row = [item.strip() for item in rows[0]]
    return {
        "index": int(row[0]),
        "uuid": row[1],
        "name": row[2],
        "total_memory_bytes": int(row[3]) * (1 << 20),
        "free_memory_bytes": int(row[4]) * (1 << 20),
        "used_memory_bytes": int(row[5]) * (1 << 20),
        "utilization_percent": None if row[6] in ("[N/A]", "N/A") else int(row[6]),
    }


def _device_inventory(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Require a root-frozen six-device allowlist with exact B300 UUIDs."""
    value = read_json(path)
    devices = value.get("devices")
    if value.get("schema") != "wnn.b300-device-inventory.v1" or value.get("status") != "complete":
        raise ValueError("device inventory is not a completed root-issued B300 inventory")
    if not isinstance(devices, list):
        raise ValueError("device inventory is missing its device list")
    allowed = [item for item in devices if isinstance(item, dict) and item.get("profile_allowed") is True]
    allowed.sort(key=lambda item: item.get("gpu_index", -1))
    if len(allowed) != 6:
        raise ValueError("root inventory must explicitly allow exactly six profile devices")
    seen: set[str] = set()
    for item in allowed:
        uuid = item.get("uuid")
        if (
            type(item.get("gpu_index")) is not int
            or not isinstance(uuid, str)
            or not uuid.startswith("GPU-")
            or uuid in seen
            or "B300" not in str(item.get("name", "")).upper()
            or type(item.get("total_memory_bytes")) is not int
            or item["total_memory_bytes"] <= 64 * GIB
        ):
            raise ValueError("root device inventory has an invalid UUID, model, index, or memory size")
        seen.add(uuid)
    return allowed, sha256_file(path)


def _recipe_configs(source: Path) -> list[dict[str, Any]]:
    """Bind all six launch configs to the frozen R3 node hashes."""
    packet = source / "research/experiment_drafts/multistage-optuna-b300-20261031"
    recipe = read_json(packet / "recipes/base-controls-r3.json")
    if recipe.get("schema") != "wnn.multistage-recipe-nodes.v1" or recipe.get("recipe_revision") != "R3":
        raise ValueError("the frozen R3 recipe document is missing or has the wrong schema")
    nodes = recipe.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 6:
        raise ValueError("the R3 profile ladder must contain exactly six nodes")
    configs: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for node in nodes:
        args = node.get("args") if isinstance(node, dict) else None
        if not isinstance(args, dict):
            raise ValueError("R3 node is missing its exact trainer config reference")
        relative, expected_hash = args.get("config_path"), args.get("config_sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError("R3 config reference is incomplete")
        config_path = (packet / relative).resolve(strict=True)
        if packet.resolve() not in config_path.parents or sha256_file(config_path) != expected_hash:
            raise ValueError(f"R3 config bytes differ from its recipe pin: {relative}")
        config = read_json(config_path)
        model = config.get("model")
        family, batch = node.get("family"), config.get("batch_size")
        if (
            not isinstance(model, dict)
            or model.get("architecture") != family
            or model.get("hidden") != 256
            or model.get("layers") != 2
            or config.get("precision_mode") != "fp32"
            or config.get("required_world_size") != 1
            or config.get("required_device") != "B300"
            or config.get("training_mask") != "all_after_warmup"
            or config.get("kd_weight") != 0
            or config.get("hint_weight") != 0
            or (family, batch) in seen
        ):
            raise ValueError(f"R3 profile config has an unsupported identity: {relative}")
        seen.add((str(family), int(batch)))
        configs.append(
            {
                "anchor_id": node.get("id"),
                "family": family,
                "batch_size": batch,
                "config_path": config_path,
                "config_sha256": expected_hash,
                "config": config,
            }
        )
    if seen != {(family, batch) for family in ("gru", "lstm") for batch in (16, 32, 64)}:
        raise ValueError("R3 batch ladder does not cover the six pinned family/batch anchors")
    configs.sort(key=lambda item: (str(item["family"]), int(item["batch_size"])))
    return configs


def _validate_receiver(
    root: Path, source: Path, primary_split: Path, auxiliary_split: Path, receipt_path: Path
) -> dict[str, object]:
    """Bind complete receiver counts and exact data/source/index SHA receipts."""
    packet = source / "research/experiment_drafts/multistage-optuna-b300-20261031"
    receipt = read_json(receipt_path)
    transport = read_json(packet / "receiving/INPUT_TRANSPORT_R1.json")
    registry = packet / "receiving/PORTABLE_DATA_R1.json"
    original_primary = packet / "receiving/ORIGINAL_PRIMARY_SPLIT.json"
    primary_sha = sha256_file(original_primary)
    if (
        primary_sha != PRIMARY_SHA256
        or sha256_file(primary_split) != PRIMARY_SHA256
        or sha256_file(registry) != REGISTRY_SHA256
        or sha256_file(root / "inputs/portable-data-r1.json") != REGISTRY_SHA256
        or receipt.get("status") not in ("complete", "success")
        or receipt.get("primary_split_sha256") != PRIMARY_SHA256
        or receipt.get("auxiliary_split_sha256") != AUXILIARY_SHA256
        or receipt.get("source_capsule_sha256") != SOURCE_CAPSULE_SHA256
    ):
        raise ValueError("receiver receipt or exact primary/auxiliary/registry identities are incomplete")
    primary = read_json(primary_split)
    auxiliary = read_json(auxiliary_split)
    counts = {
        "primary_fit_train_sequences": len(primary.get("fit", {}).get("train", [])),
        "primary_fit_valid_sequences": len(primary.get("fit", {}).get("valid", [])),
        "primary_eval_valid_sequences": len(primary.get("evaluation", {}).get("valid", [])),
        "auxiliary_fit_train_sequences": len(auxiliary.get("fit", {}).get("train", [])),
        "auxiliary_fit_valid_sequences": len(auxiliary.get("fit", {}).get("valid", [])),
        "auxiliary_control_train_sequences": len(auxiliary.get("evaluation", {}).get("train", [])),
    }
    expected_counts = {
        "primary_fit_train_sequences": 10325,
        "primary_fit_valid_sequences": 0,
        "primary_eval_valid_sequences": 1873,
        "auxiliary_fit_train_sequences": 8260,
        "auxiliary_fit_valid_sequences": 0,
        "auxiliary_control_train_sequences": 2065,
    }
    receiver_counts = receipt.get("counts") if isinstance(receipt.get("counts"), dict) else receipt
    aliases = {
        "primary_fit_train_sequences": ("primary_fit_train_sequences", "primary_fit_sequences"),
        "primary_fit_valid_sequences": ("primary_fit_valid_sequences", "primary_fit_valid_count"),
        "primary_eval_valid_sequences": ("primary_eval_valid_sequences", "primary_valid_eval_sequences"),
        "auxiliary_fit_train_sequences": ("auxiliary_fit_train_sequences", "aux_fit_train_sequences", "aux_fit_sequences"),
        "auxiliary_fit_valid_sequences": ("auxiliary_fit_valid_sequences", "aux_fit_valid_sequences"),
        "auxiliary_control_train_sequences": ("auxiliary_control_train_sequences", "aux_control_train_sequences", "aux_control_sequences"),
    }
    receipt_counts = {
        name: next((receiver_counts[key] for key in names if key in receiver_counts), None)
        for name, names in aliases.items()
    }
    if counts != expected_counts or receipt_counts != expected_counts:
        raise ValueError(f"receiver membership counts are incomplete or wrong: {counts}")
    if (
        auxiliary.get("chronology_basis")
        != "none; deterministic whole-component auxiliary partition, not a time split"
        or auxiliary.get("parent", {}).get("sha256") != PRIMARY_SHA256
        or auxiliary.get("evaluation", {}).get("source") != "train"
    ):
        raise ValueError("auxiliary partition is not the declared train-only component control")
    fit_groups = auxiliary.get("fit", {}).get("train", [])
    control_groups = auxiliary.get("evaluation", {}).get("train", [])
    primary_train_groups = primary.get("fit", {}).get("train", [])
    if set(fit_groups).intersection(control_groups) or set(fit_groups).union(control_groups) != set(primary_train_groups):
        raise ValueError("auxiliary fitting and control memberships are not disjoint/exhaustive")
    component_map = {
        int(item["row_group"]): int(item["component"])
        for item in primary.get("sequences", {}).get("train", [])
    }
    fit_components = {component_map[group] for group in fit_groups}
    control_components = {component_map[group] for group in control_groups}
    if fit_components.intersection(control_components):
        raise ValueError("auxiliary split separates members of a known connected component")
    if sha256_file(auxiliary_split) != AUXILIARY_SHA256:
        raise ValueError("auxiliary manifest bytes differ from the root-pinned R3 manifest")

    source_audit = root / "data/cache/source_audit.json"
    graph = root / "data/cache/chronological_chains.npz"
    source_audit_sha = sha256_file(source_audit)
    graph_sha = sha256_file(graph)
    expected_files = transport.get("files")
    if not isinstance(expected_files, dict):
        raise ValueError("input transport is missing its sealed source-file map")
    for relative, entry in expected_files.items():
        path = root / "data" / relative
        if not isinstance(entry, dict) or path.stat().st_size != entry.get("bytes"):
            raise ValueError(f"receiving artifact size differs from transport: {relative}")
        if sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"receiving artifact SHA differs from transport: {relative}")
    audit = read_json(source_audit)
    source_entries = audit.get("sources")
    if not isinstance(source_entries, dict):
        raise ValueError("receiver source audit lacks official train and valid entries")
    source_hashes = {
        "train": source_entries.get("datasets/train.parquet", {}).get("sha256"),
        "valid": source_entries.get("datasets/valid.parquet", {}).get("sha256"),
    }
    if source_hashes != {"train": TRAIN_SHA256, "valid": VALID_SHA256}:
        raise ValueError("receiver full-file source audit differs from sealed official SHA pins")
    source_counts = {
        "train": source_entries["datasets/train.parquet"].get("row_group_count"),
        "valid": source_entries["datasets/valid.parquet"].get("row_group_count"),
    }
    if source_counts != {"train": 10607, "valid": 1873}:
        raise ValueError("receiver source audit does not bind the full official row-group counts")
    expected_index_receipts = {
        split: sha256_file(root / f"data/cache/{split}_index.identity.json")
        for split in ("train", "valid")
    }
    recorded_sources = receipt.get("source_sha256", receipt.get("raw_source_sha256"))
    if isinstance(recorded_sources, dict):
        recorded_sources = {
            "train": recorded_sources.get("train", recorded_sources.get("datasets/train.parquet")),
            "valid": recorded_sources.get("valid", recorded_sources.get("datasets/valid.parquet")),
        }
    recorded_index_receipts = receipt.get(
        "index_receipt_sha256", receipt.get("source_index_receipt_sha256")
    )
    if isinstance(recorded_index_receipts, dict):
        recorded_index_receipts = {
            "train": recorded_index_receipts.get("train", recorded_index_receipts.get("cache/train_index.identity.json")),
            "valid": recorded_index_receipts.get("valid", recorded_index_receipts.get("cache/valid_index.identity.json")),
        }
    if (
        receipt.get("source_audit_sha256") != source_audit_sha
        or recorded_sources != source_hashes
        or receipt.get("graph_sha256", receipt.get("chronological_graph_sha256")) != graph_sha
        or recorded_index_receipts != expected_index_receipts
    ):
        raise ValueError("receiver receipt is missing source-audit, graph, or index-receipt SHA bindings")
    if receipt.get("primary_split_sha256") != PRIMARY_SHA256:
        raise ValueError("receiver receipt does not bind the exact official primary manifest")
    return {
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": sha256_file(receipt_path),
        "primary_split_sha256": PRIMARY_SHA256,
        "auxiliary_split_sha256": AUXILIARY_SHA256,
        "portable_registry_sha256": REGISTRY_SHA256,
        "source_audit_sha256": source_audit_sha,
        "source_sha256": source_hashes,
        "source_counts": source_counts,
        "index_receipt_sha256": expected_index_receipts,
        "graph_sha256": graph_sha,
        "counts": counts,
    }


def _validate_started_child(proc: subprocess.Popen[bytes], timeout_owner: dict[str, int | str]) -> dict[str, int | str]:
    """Identify the exact Python profile child launched beneath GNU timeout."""
    child_list = Path(f"/proc/{proc.pid}/task/{proc.pid}/children")
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            children = [int(value) for value in child_list.read_text().split()]
        except (FileNotFoundError, PermissionError):
            children = []
        if len(children) == 1:
            identity = process_identity(children[0])
            if identity is not None:
                return identity
        if proc.poll() is not None:
            break
        time.sleep(0.02)
    if process_identity(int(timeout_owner["pid"])) != timeout_owner:
        raise RuntimeError("GNU timeout owner exited before the training child was identified")
    raise RuntimeError("could not capture the exact profile child PID/birth/PGID")


def _run_one(
    *,
    root: Path,
    source: Path,
    python: Path,
    attempt_suffix: str,
    config: dict[str, Any],
    device: dict[str, Any],
    inventory_sha256: str,
    receiver: dict[str, object],
    first_profile_epoch: float,
) -> dict[str, object]:
    """Run one exact-profile attempt; GNU timeout remains as a parent-death watchdog."""
    started = time.time()
    hard_deadline = started + PROFILE_SECONDS
    name = f"{config['family']}-b{config['batch_size']}-{attempt_suffix}"
    packet = source / "research/experiment_drafts/multistage-optuna-b300-20261031"
    output = root / "profiles" / name
    receipt_path = root / "receipts" / f"anchor-{name}.json"
    log_path = root / "logs" / f"anchor-{name}.log"
    if output.exists() or receipt_path.exists() or log_path.exists():
        raise FileExistsError(f"preserve prior profile attempt: {name}")
    state: dict[str, object] = {
        "schema": "wnn.multistage-anchor-profile-attempt.v2",
        "name": name,
        "anchor_id": config["anchor_id"],
        "family": config["family"],
        "batch_size": config["batch_size"],
        "config_path": str(config["config_path"]),
        "config_sha256": config["config_sha256"],
        "source_root": str(source),
        "python": str(python),
        "device": dict(device),
        "device_inventory_sha256": inventory_sha256,
        "receiver_validation": receiver,
        "first_profile_epoch": first_profile_epoch,
        "started_at_epoch": started,
        "hard_deadline_epoch": hard_deadline,
        "status": "preflight",
        "cpu_thread_env": {
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
        },
        "pre_gpu_snapshot": None,
        "post_gpu_snapshot": {"status": "not_observed"},
        "timeout_owner": None,
        "training_child": None,
        "exit_code": None,
    }
    timeout_path = shutil.which("timeout")
    proc: subprocess.Popen[bytes] | None = None
    child: dict[str, int | str] | None = None
    status, error = "failed", None
    try:
        if timeout_path is None:
            raise FileNotFoundError("GNU timeout is required for parent-death wall enforcement")
        snapshot = _query_gpu(str(device["uuid"]), timeout=max(0.1, min(3.0, hard_deadline - time.time())))
        state["pre_gpu_snapshot"] = snapshot
        if (
            snapshot["uuid"] != device["uuid"]
            or snapshot["index"] != device["gpu_index"]
            or snapshot["name"] != device["name"]
            or snapshot["total_memory_bytes"] != device["total_memory_bytes"]
            or snapshot["free_memory_bytes"] < 96 * GIB
        ):
            raise RuntimeError("GPU UUID/model/total-memory identity or 32+64GiB headroom check failed")
        remaining = int(hard_deadline - time.time() - POSTFLIGHT_RESERVE_SECONDS - WATCHDOG_GRACE_SECONDS)
        if remaining < 1:
            raise TimeoutError("profile preflight consumed the bounded attempt window")
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES=str(device["uuid"]),
            OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
            PYTHONPATH=f"{source / 'src'}:{root / 'python-deps'}",
            WNN_PORTABLE_DATA_MANIFEST=str(root / "inputs/portable-data-r1.json"),
        )
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
            env.pop(key, None)
        command = [
            timeout_path,
            "--signal=TERM",
            f"--kill-after={WATCHDOG_GRACE_SECONDS}s",
            f"{remaining}s",
            str(python),
            str(source / "scripts/train_sequence.py"),
            "--config",
            str(config["config_path"]),
            "--data-root",
            str(root / "data"),
            "--split-manifest",
            str(root / "data/cache/multistage/auxiliary-r3.json"),
            "--output",
            str(output),
            "--profile-updates",
            "50",
        ]
        state["command"] = command
        state["command_sha256"] = hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode()
        ).hexdigest()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("xb") as log_stream:
            proc = subprocess.Popen(
                command,
                cwd=source,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            owner = process_identity(proc.pid)
            if owner is None or owner["pgid"] != owner["pid"]:
                proc.kill()
                proc.wait(timeout=max(0.1, hard_deadline - time.time()))
                raise RuntimeError("failed to bind the GNU timeout process group")
            state["timeout_owner"] = owner
            child = _validate_started_child(proc, owner)
            state["training_child"] = child
            state["status"] = "running"
            write_json(receipt_path, state)
            try:
                proc.wait(timeout=max(0.1, hard_deadline - time.time() - POSTFLIGHT_RESERVE_SECONDS))
                state["exit_code"] = proc.returncode
                status = "completed" if proc.returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                status = "timed_out"
                signal_owned(child, signal.SIGTERM)
                signal_owned(owner, signal.SIGTERM)
                remaining_wait = max(0.1, hard_deadline - time.time() - 2.0)
                try:
                    proc.wait(timeout=remaining_wait)
                except subprocess.TimeoutExpired:
                    signal_owned(child, signal.SIGKILL)
                    signal_owned(owner, signal.SIGKILL)
                    try:
                        proc.wait(timeout=max(0.1, hard_deadline - time.time() - 1.0))
                    except subprocess.TimeoutExpired:
                        status = "cleanup_unconfirmed_watchdog_armed"
            state["exit_code"] = proc.returncode
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        if proc is not None and proc.poll() is None:
            signal_owned(child, signal.SIGTERM)
            signal_owned(state.get("timeout_owner"), signal.SIGTERM)  # type: ignore[arg-type]
            try:
                proc.wait(timeout=max(0.1, hard_deadline - time.time() - 2.0))
            except subprocess.TimeoutExpired:
                signal_owned(child, signal.SIGKILL)
                signal_owned(state.get("timeout_owner"), signal.SIGKILL)  # type: ignore[arg-type]
        if status not in ("timed_out", "cleanup_unconfirmed_watchdog_armed"):
            status = "preflight_failed" if proc is None else "launch_or_runtime_failed"
    finally:
        try:
            snapshot = _query_gpu(str(device["uuid"]), timeout=max(0.1, min(2.0, hard_deadline - time.time())))
            state["post_gpu_snapshot"] = {"status": "observed", **snapshot}
        except Exception as exc:
            state["post_gpu_snapshot"] = {
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
                "informational": True,
            }
        finished = time.time()
        state.update(
            {
                "status": status if finished <= hard_deadline else "wall_limit_breached",
                "error": error,
                "finished_at_epoch": finished,
                "total_wall_seconds": finished - started,
                "hard_deadline_met": finished <= hard_deadline,
            }
        )
        if finished > hard_deadline and proc is not None and proc.poll() is None:
            signal_owned(child, signal.SIGKILL)
            signal_owned(state.get("timeout_owner"), signal.SIGKILL)  # type: ignore[arg-type]
        write_json(receipt_path, state)
    return state


def _source_identity(source: Path) -> dict[str, object]:
    """Hash every Python source module plus the public training entrypoint."""
    paths = sorted((source / "src").rglob("*.py"))
    entrypoint = source / "scripts/train_sequence.py"
    if entrypoint not in paths:
        paths.append(entrypoint)
    files = {path.relative_to(source).as_posix(): sha256_file(path) for path in paths}
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {"revision": _git_revision(source), "files": files, "sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def _git_revision(source: Path) -> str:
    """Record the source checkout revision when git metadata is available."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True,
            text=True, timeout=3,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def main() -> None:
    """Validate the receiver and exact R3 pins, then launch only six profiles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--primary-split", type=Path, required=True)
    parser.add_argument("--auxiliary-split", type=Path, required=True)
    parser.add_argument("--receiver-receipt", type=Path)
    parser.add_argument("--device-inventory", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--attempt", choices=("r3", "r4"), default="r3")
    args = parser.parse_args()
    root, source = args.root.resolve(strict=True), args.source.resolve(strict=True)
    packet = source / "research/experiment_drafts/multistage-optuna-b300-20261031"
    receiver_path = args.receiver_receipt or root / "logs/receiver-data-r2.log"
    primary, auxiliary = args.primary_split.resolve(strict=True), args.auxiliary_split.resolve(strict=True)
    receiver = _validate_receiver(root, source, primary, auxiliary, receiver_path.resolve(strict=True))
    devices, inventory_sha = _device_inventory(args.device_inventory.resolve(strict=True))
    configs = _recipe_configs(source)
    # Preserve the venv launcher path: resolving its symlink can escape to
    # system Python, which does not have the source environment's torch build.
    python = args.python.absolute() if args.python else source / ".venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError(f"source-bound Python interpreter is missing: {python}")
    source_identity = _source_identity(source)
    (root / "profiles").mkdir(exist_ok=True)
    (root / "receipts").mkdir(exist_ok=True)
    first_profile = root / "receipts/first-gpu-profile-r2.json"
    if first_profile.exists():
        # A retry keeps the original six-hour campaign epoch and leaves its
        # first-attempt receipt immutable.
        campaign = read_json(first_profile)
        started = float(campaign["first_gpu_profile_epoch"])
    else:
        started = time.time()
        campaign = {
            "schema": "wnn.multistage-anchor-profile-campaign.v2",
            "first_gpu_profile_epoch": started,
            "deadline_epoch": started + 21600,
            "reserve_start_epoch": started + 20400,
            "profile_attempts": 6,
            "scientific_fit_attempts": 0,
            "primary_split_sha256": PRIMARY_SHA256,
            "auxiliary_split_sha256": AUXILIARY_SHA256,
            "device_inventory_sha256": inventory_sha,
            "receiver_validation": receiver,
            "source_identity": source_identity,
            "config_hashes": {str(item["anchor_id"]): item["config_sha256"] for item in configs},
        }
        write_json(first_profile, campaign)
    pairs = list(zip(configs, devices, strict=True))
    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [
            pool.submit(
                _run_one,
                root=root,
                source=source,
                python=python,
                attempt_suffix=args.attempt,
                config=config,
                device=device,
                inventory_sha256=inventory_sha,
                receiver=receiver,
                first_profile_epoch=started,
            )
            for config, device in pairs
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except BaseException as exc:
                results.append({"status": "launcher_error", "error": f"{type(exc).__name__}: {exc}"})
    write_json(
        root / f"receipts/anchor-profiles-terminal-{args.attempt}.json",
        {"schema": "wnn.multistage-anchor-profile-terminal.v2", "attempt_suffix": args.attempt, "campaign": campaign, "profiles": results},
    )
    if len(results) != 6:
        raise RuntimeError("profile launcher did not retain six terminal outcomes")


if __name__ == "__main__":
    main()
