"""Source, checkpoint-tensor and process-envelope checks for feature fits."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import json
import math
import sys
from collections.abc import Iterator, Mapping
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Any


def sha256(path: Path) -> str:
    """Hash a regular file without following a final symlink."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one regular JSON object."""
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"expected a regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def require_owned_file(path: Path, boundary: Path) -> Path:
    """Require a regular non-symlink file whose ancestors remain inside boundary."""
    boundary = boundary.resolve(strict=True)
    if not path.is_absolute():
        path = boundary / path
    current = path
    while current != boundary and boundary in current.parents:
        if current.is_symlink():
            raise ValueError(f"fit evidence path contains a symlink: {current}")
        current = current.parent
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(boundary) or not resolved.is_file():
        raise ValueError(f"fit evidence file escapes its owned tree: {path}")
    return resolved


@contextlib.contextmanager
def r01_helpers(code_dir: Path) -> Iterator[tuple[ModuleType, ModuleType, ModuleType]]:
    """Import the pinned R01 helper pair without inheriting a cached adapter."""
    names = ("adapter_common", "fit_receipt", "fit_recipe", "source_package_verify")
    saved = {name: sys.modules.pop(name, None) for name in names}
    original_path = list(sys.path)
    write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code_dir))
    try:
        yield (
            importlib.import_module("adapter_common"),
            importlib.import_module("fit_receipt"),
            importlib.import_module("fit_recipe"),
        )
    finally:
        for name in names:
            sys.modules.pop(name, None)
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module
        sys.path[:] = original_path
        sys.dont_write_bytecode = write_bytecode


@contextlib.contextmanager
def checkpoint_helpers(path: Path) -> Iterator[ModuleType]:
    """Load the independently pinned PR15 semantic checkpoint identity code."""
    name = "_feature_checkpoint_identity"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load checkpoint identity helper: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def verify_source_pin(root: Path, inputs: dict[str, Any], common: ModuleType) -> dict[str, Any]:
    """Verify the full LF-deployed source subset and the canonical trainer subset."""
    builder = common._json(common._RECIPE)
    source_root = root / "runtime/source-a266-lf-subset-v1"
    if source_root.is_symlink():
        raise ValueError("trainer source root is not the owned a266 runtime path")
    source_manifest = common._json(common._SOURCE_MANIFEST)
    source_manifest_sha = common._sha256(common._SOURCE_MANIFEST)
    if source_manifest_sha != builder.get("source_manifest_sha256") or source_manifest.get(
        "source_revision"
    ) != builder.get("expected_base_source_revision"):
        raise ValueError("R01 command builder and portable trainer manifest disagree")
    trainer = common.verify_source_manifest(source_root, source_manifest)
    tree_manifest = common._json(common._TREE_MANIFEST)
    tree_sha = common._sha256(common._TREE_MANIFEST)
    if tree_sha != builder.get("source_tree_manifest_sha256") or tree_manifest.get(
        "archive_sha256"
    ) != builder.get("source_tree_archive_sha256"):
        raise ValueError("R01 command builder and deployed source-tree manifest disagree")
    deployed = common.verify_deployed_tree_manifest(
        source_root, tree_manifest, manifest_sha256=tree_sha
    )
    if (
        trainer != inputs.get("trainer_source_identity")
        or deployed != inputs.get("deployed_source_tree")
        or trainer.get("source_identity_sha256") != builder.get("expected_source_identity_sha256")
        or deployed.get("deployed_tree_identity_sha256")
        != builder.get("expected_source_tree_identity_sha256")
    ):
        raise ValueError("fit INPUTS.json source identity differs from the frozen a266 package")
    producer = canonical_producer_identity(source_manifest)
    feature_source_sha = (
        tree_manifest.get("files", {})
        .get("src/wnn/research_features.py", {})
        .get("deployed_sha256")
    )
    if not isinstance(feature_source_sha, str) or len(feature_source_sha) != 64:
        raise ValueError("full deployed-source manifest omits research_features.py")
    return {
        "source_revision": builder["expected_base_source_revision"],
        "trainer_manifest_sha256": source_manifest_sha,
        "trainer_source_identity_sha256": trainer["source_identity_sha256"],
        "deployed_tree_manifest_sha256": tree_sha,
        "deployed_tree_identity_sha256": deployed["deployed_tree_identity_sha256"],
        "upstream_archive_sha256": deployed["archive_sha256"],
        "producer_identity": producer,
        "feature_source_sha256": feature_source_sha,
    }


def canonical_producer_identity(source_manifest: dict[str, Any]) -> dict[str, Any]:
    """Derive the trainer's exact 60-file identity from the portable 61-file pin."""
    files = source_manifest.get("files")
    readme = "src/wnn/training/README.md"
    if not isinstance(files, dict) or readme not in files:
        raise ValueError("portable source manifest lacks the producer-excluded training README")
    producer_files = dict(files)
    producer_files.pop(readme)
    if len(producer_files) != 60:
        raise ValueError("canonical trainer producer manifest must contain exactly sixty files")
    identity = hashlib.sha256(
        json.dumps(producer_files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "wnn.training-source.v1",
        "sha256": identity,
        "files": producer_files,
        "missing_dependency_files": [],
        "runtime_dependencies": {"attrs": version("attrs")},
    }


def check_model(checkpoint: dict[str, Any], width: int, torch: Any) -> None:
    """Require the exact FP32 H512x1 model tensors at the declared input width."""
    state = checkpoint.get("model")
    shapes = {
        "mean": (width,),
        "std": (width,),
        "inp.weight": (512, width),
        "inp.bias": (512,),
        "gru.weight_ih_l0": (1536, 512),
        "gru.weight_hh_l0": (1536, 512),
        "gru.bias_ih_l0": (1536,),
        "gru.bias_hh_l0": (1536,),
        "head.0.weight": (512, 1024),
        "head.0.bias": (512,),
        "head.2.weight": (2, 512),
        "head.2.bias": (2,),
    }
    if not isinstance(state, Mapping) or set(state) != set(shapes):
        raise ValueError("feature checkpoint tensor set differs from the original H512x1 GRU")
    for key, shape in shapes.items():
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"feature checkpoint tensor {key} has the wrong shape")
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all().item():
            raise ValueError(f"feature checkpoint tensor {key} is not finite strict FP32")
    if not torch.all(state["std"] > 0).item():
        raise ValueError("feature checkpoint input normalization has nonpositive scales")


def check_phase(
    root: Path, work: Path, phase: dict[str, Any], job: dict[str, Any]
) -> dict[str, str]:
    """Require successful actor, durable log and exact resource-envelope receipts."""
    package = root / "runtime/pr15-host-v2"
    expected_suffix = [
        "-u",
        str(package / "gpu_entrypoint.py"),
        str(package / "train_cell.py"),
        "--request",
        str(work / "worker_request.json"),
    ]
    command = phase.get("command")
    if (
        type(phase.get("exit_code")) is not int
        or phase.get("exit_code") != 0
        or not isinstance(phase.get("command"), list)
        or not phase["command"]
        or len(command) != len(expected_suffix) + 1
        or not isinstance(command[0], str)
        or not Path(command[0]).is_absolute()
        or command[1:] != expected_suffix
    ):
        raise ValueError("fit actor phase is not the canonical PR15 train-cell invocation")
    start, finish, elapsed = (
        phase.get("started_epoch"),
        phase.get("finished_epoch"),
        phase.get("elapsed_seconds"),
    )
    if any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
        for x in (start, finish, elapsed)
    ):
        raise ValueError("fit actor phase has invalid timing evidence")
    if finish < start or elapsed < 0 or abs((finish - start) - elapsed) > max(2.0, elapsed * 0.05):
        raise ValueError("fit actor phase timing fields are inconsistent")
    log_path = Path(str(phase.get("log", "")))
    if not log_path.is_absolute():
        log_path = work / log_path
    require_owned_file(log_path, root)
    log_path = log_path.resolve(strict=True)
    if not log_path.is_relative_to(root) or sha256(log_path) != phase.get("log_sha256"):
        raise ValueError("fit actor log escapes the project or differs from its phase hash")
    resource_path = Path(str(phase.get("resource_receipt", "")))
    if not resource_path.is_absolute():
        resource_path = work / resource_path
    require_owned_file(resource_path, work)
    resource_path = resource_path.resolve(strict=True)
    if not resource_path.is_relative_to(work) or sha256(resource_path) != phase.get(
        "resource_receipt_sha256"
    ):
        raise ValueError("fit process envelope escapes its workspace or differs from phase hash")
    resource = read_json(resource_path)
    if (
        resource.get("schema") != "tabdata-b300-process-envelope.v1"
        or resource.get("status") != "actor_returned"
        or resource.get("script") != "train_cell.py"
        or resource.get("allocator_budget_bytes") != 6 * 1024**3
        or resource.get("torch_threads") != 2
        or "B300" not in str(resource.get("gpu", ""))
        or not isinstance(resource.get("visible_devices"), str)
        or not resource["visible_devices"]
    ):
        raise ValueError("fit process resource envelope differs from the frozen B300 actor")
    if job.get("gpu_uuid") is not None and resource.get("visible_devices") != job["gpu_uuid"]:
        raise ValueError("fit process envelope is bound to a different GPU UUID")
    cpus = job.get("cpus", job.get("cpu_ids_initial"))
    if cpus is not None and resource.get("cpu_affinity") != list(cpus):
        raise ValueError("fit process envelope CPU affinity differs from the scheduled cell")
    for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
        value = resource.get(key)
        if type(value) is not int or not 0 <= value <= 6 * 1024**3:
            raise ValueError(f"fit actor CUDA peak exceeds its allocator envelope: {key}")
    return {
        "phase_sha256": sha256(work / "fit_phase.json"),
        "log_sha256": phase["log_sha256"],
        "resource_receipt_sha256": phase["resource_receipt_sha256"],
        "command_sha256": hashlib.sha256(
            json.dumps(phase["command"], separators=(",", ":")).encode()
        ).hexdigest(),
    }
