"""Validate old R01 fits and create receipt views without modifying their workspace."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

from r01_recovery_common import (
    CHECKPOINT_IDENTITY_SHA256,
    producer_source_identity,
    read_json,
    require_under,
    sha256_file,
    validate_epoch_records,
    write_json_new,
)
from r01_recovery_placement import verify_placement

EXPECTED_PANEL_SHA256 = "471f465cc1d13fc6e5153a4c998390bb14e06823fed4c6721f929ce95fee47fb"
EXPECTED_ACTOR_SHA256 = "e6a155ee13a7e907bb7fca2c1ebed07879e8466422015546e20360af9b404220"
EXPECTED_SOURCE_REVISION = "a266b036b02e8e012c6eb31072291dad618a169d"
EXPECTED_GRADER_SHA256 = "56ffc693da52668726bab1f7008e31753a76aed3ff5d79089167785079d9eb31"
EXPECTED_WIDTH = {"wide": 512, "deep": 384}
EXPECTED_LAYERS = {"wide": 1, "deep": 2}
EXPECTED_PARAMETERS = {"wide": 2_210_306, "deep": 2_151_554}
CACHE_FILES = ("chains.npz", "chain_folds.npz", "train_targets.npz", "valid_targets.npz")


def _load_module(name: str, path: Path) -> Any:
    """Load one exact local module by path for isolated recovery checks."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _same_group_membership(actual: Any, expected: Any) -> bool:
    """Compare canonical membership without imposing a trainer-specific list order."""
    return (
        isinstance(actual, list)
        and isinstance(expected, list)
        and all(type(group) is int and group >= 0 for group in actual)
        and all(type(group) is int and group >= 0 for group in expected)
        and len(actual) == len(set(actual))
        and len(expected) == len(set(expected))
        and set(actual) == set(expected)
    )


def _verify_model_payload(
    checkpoint: dict[str, Any],
    args: dict[str, Any],
    arm: str,
    source_root: Path,
    expected_epoch: int,
) -> None:
    """Compare every checkpoint state tensor with the pinned RNN factory on CPU."""
    import torch

    if (
        checkpoint.get("epoch") != expected_epoch
        or checkpoint.get("n_in") != 211
        or checkpoint.get("args") != args
        or args.get("hidden") != EXPECTED_WIDTH[arm]
        or args.get("layers") != EXPECTED_LAYERS[arm]
    ):
        raise ValueError("checkpoint metadata differs from the fixed R01 arm or epoch")
    source_path = str(source_root / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from wnn.training.models import make_model

    model = make_model(args, n_in=211, n_out=2)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS[arm]:
        raise ValueError("pinned model factory parameter count differs from the R01 contract")
    expected = model.state_dict()
    actual = checkpoint.get("model")
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError("checkpoint tensor names differ from the pinned R01 model")
    for name, reference in expected.items():
        value = actual[name]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != tuple(reference.shape)
            or value.dtype != torch.float32
            or not torch.isfinite(value).all().item()
        ):
            raise ValueError(f"checkpoint tensor is not finite FP32 with canonical shape: {name}")
    if not torch.all(actual["std"] > 0).item():
        raise ValueError("checkpoint normalization has a nonpositive input scale")


def _verify_checkpoint_set(
    run_dir: Path,
    run_name: str,
    arm: str,
    provenance: dict[str, Any],
    source_root: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Check all six snapshots, selected endpoint, raw hashes and loaded payload identity."""
    import torch

    checkpoint_module_path = project_root / "runtime/pr15-host-v2/checkpoint_identity.py"
    if sha256_file(checkpoint_module_path) != CHECKPOINT_IDENTITY_SHA256:
        raise ValueError("checkpoint identity helper differs from its frozen host-v2 hash")
    identity_module = _load_module("r01_recovery_checkpoint_identity", checkpoint_module_path)
    output = run_dir / "artifacts/runs" / run_name
    for path in (
        run_dir / "artifacts",
        run_dir / "artifacts/runs",
        output,
    ):
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"original trainer output directory is not an owned directory: {path}")
    checkpoint_hashes = provenance.get("checkpoint_sha256")
    if not isinstance(checkpoint_hashes, dict):
        raise ValueError("trainer provenance lacks per-checkpoint archive hashes")
    for epoch in range(6):
        path = output / f"model_ep{epoch}.pt"
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"R01 epoch checkpoint is missing or symlinked: {path}")
        if sha256_file(path) != checkpoint_hashes.get(path.name):
            raise ValueError(f"epoch-{epoch} checkpoint bytes differ from provenance")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("epoch") != epoch:
            raise ValueError(f"saved checkpoint is not the declared epoch {epoch}")
        _verify_model_payload(payload, provenance["args"], arm, source_root, epoch)
    selected_path = output / "model.pt"
    if selected_path.is_symlink() or not selected_path.is_file():
        raise ValueError("selected R01 model.pt is missing or symlinked")
    identity = identity_module.verify_checkpoint_pair(
        output / "model_ep5.pt",
        selected_path,
        expected_epoch=5,
        expected_input_width=211,
    )
    if identity.fixed_file_sha256 != checkpoint_hashes.get(
        "model_ep5.pt"
    ) or identity.selected_file_sha256 != checkpoint_hashes.get("model.pt"):
        raise ValueError("selected/fixed checkpoint raw hashes differ from provenance")
    selected = torch.load(selected_path, map_location="cpu", weights_only=False)
    _verify_model_payload(selected, provenance["args"], arm, source_root, 5)
    return identity.receipt()


def _verify_completed_training_phase(run_dir: Path, job: dict[str, Any]) -> dict[str, Any]:
    """Bind complete fit artifacts to the B300 actor's specific post-fit outcome."""
    resource_path = run_dir / "train_cell_resource_receipt.json"
    resource = read_json(resource_path)
    elapsed = resource.get("elapsed_seconds")
    if (
        resource.get("schema") != "tabdata-b300-process-envelope.v1"
        or resource.get("script") != "train_cell.py"
        or resource.get("visible_devices") != job.get("gpu_uuid")
        or "B300" not in str(resource.get("gpu", ""))
        or resource.get("allocator_budget_bytes") != 6 * 1024**3
        or resource.get("torch_threads") != 2
        or resource.get("status") not in {"actor_returned", "actor_failed"}
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed <= 0
        or type(resource.get("cuda_peak_allocated_bytes")) is not int
        or type(resource.get("cuda_peak_reserved_bytes")) is not int
        or not 0 <= resource["cuda_peak_allocated_bytes"] <= 6 * 1024**3
        or not 0 <= resource["cuda_peak_reserved_bytes"] <= 6 * 1024**3
    ):
        raise ValueError("original R01 train actor resource receipt is not the panel-bound fit")
    placement = verify_placement(run_dir.parent.parent, job, resource)
    producer_path = run_dir / "producer_receipt.json"
    producer_status = "missing_after_postfit_receipt_error"
    producer_sha256 = None
    if producer_path.is_file():
        original = read_json(producer_path)
        producer_status = original.get("status")
        producer_sha256 = sha256_file(producer_path)
        if producer_status == "trainer_failed":
            raise ValueError(
                "original producer records a trainer failure, not a receipt-only error"
            )
        if producer_status != "completed_fixed_epoch5":
            raise ValueError("original producer receipt has an unsupported terminal status")
    if resource.get("status") == "actor_failed":
        if resource.get("error_type") != "ValueError" or producer_status != (
            "missing_after_postfit_receipt_error"
        ):
            raise ValueError(
                "failed original actor is not the identified post-fit receipt mismatch"
            )
    elif producer_status != "completed_fixed_epoch5":
        raise ValueError("returned original train actor lacks its successful producer receipt")
    return {
        "status": resource["status"],
        "error_type": resource.get("error_type"),
        "resource_receipt_sha256": sha256_file(resource_path),
        "placement_authorization": placement,
        "original_producer_receipt_status": producer_status,
        "original_producer_receipt_sha256": producer_sha256,
        "fit_completion_verified_from": [
            "panel-bound original B300 actor envelope",
            "exact fit argv and provenance",
            "six full withheld-evaluation epoch records",
            "all six epoch checkpoints and semantic epoch-5 model pair",
        ],
    }


def _verify_source_and_adapter(
    project_root: Path, packet_root: Path, source_root: Path
) -> tuple[dict[str, Any], dict[str, Any], Any, Any, Any]:
    """Verify the frozen R01 adapter, complete 61-file source and deployed tree."""
    code = project_root / "runtime/b300-r01-v1/code"
    if sha256_file(project_root / "runtime/b300-r01-v1/EXECUTION_MANIFEST.json") != (
        "95f82d442eefe425dde6574222b037ba1deba2d3ae36534aeec41e31e450874f"
    ):
        raise ValueError("R01 execution manifest differs from the frozen V4 packet")
    expected_members = read_json(project_root / "runtime/b300-r01-v1/EXECUTION_MANIFEST.json")[
        "files"
    ]
    for relative, expected in expected_members.items():
        if sha256_file(project_root / "runtime/b300-r01-v1" / relative) != expected:
            raise ValueError(f"R01 frozen actor member changed: {relative}")
    sys.path.insert(0, str(code))
    adapter = _load_module("r01_recovery_adapter_common", code / "adapter_common.py")
    fit_recipe = _load_module("r01_recovery_fit_recipe", code / "fit_recipe.py")
    fit_receipt = _load_module("r01_recovery_fit_receipt", code / "fit_receipt.py")
    manifest_path = code / "a266_source_manifest.json"
    source_manifest = read_json(manifest_path)
    source_identity = adapter.verify_source_manifest(source_root, source_manifest)
    producer_identity = producer_source_identity(source_manifest["files"])
    if (
        source_identity.get("file_count") != 61
        or source_identity.get("source_revision_expected") != EXPECTED_SOURCE_REVISION
        or source_identity.get("files") != source_manifest.get("files")
    ):
        raise ValueError("complete portable source does not match the 61-file frozen identity")
    tree_path = code / "source_package_manifest.json"
    tree_manifest = read_json(tree_path)
    deployed = adapter.verify_deployed_tree_manifest(
        source_root, tree_manifest, manifest_sha256=sha256_file(tree_path)
    )
    builder = read_json(code / "command_builder.json")
    recipe_path = packet_root / "recipe/RANK01_RECIPE.json"
    bundle = adapter.adapter_bundle_identity()
    if (
        bundle.get("sha256") != EXPECTED_ACTOR_SHA256
        or sha256_file(recipe_path) != builder.get("source_recipe_sha256")
        or builder.get("source_manifest_sha256") != sha256_file(manifest_path)
        or builder.get("source_tree_manifest_sha256") != sha256_file(tree_path)
        or builder.get("expected_source_tree_identity_sha256")
        != deployed.get("deployed_tree_identity_sha256")
        or deployed.get("source_revision") != EXPECTED_SOURCE_REVISION
        or builder.get("r4_grader_sha256") != EXPECTED_GRADER_SHA256
    ):
        raise ValueError("R01 recipe, actor bundle, source tree or grader pin differs")
    return source_identity, producer_identity, fit_recipe, fit_receipt, (adapter, bundle, deployed)


def _make_fit_view(
    project_root: Path,
    recovery_root: Path,
    run_dir: Path,
    job: dict[str, Any],
    preflight: dict[str, Any],
    command: dict[str, Any],
    fit_receipt: dict[str, Any],
    producer: dict[str, Any],
    identity: dict[str, Any],
    update_summary: dict[str, Any],
    actor_evidence: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Create a fresh export view with read-only links and a separate translated receipt."""
    name = str(job["id"])
    view = recovery_root / "fit_views" / name
    view.parent.mkdir(parents=True, exist_ok=True)
    view.mkdir(exist_ok=False)
    data_root = project_root / "data"
    data_datasets = (data_root / "datasets").resolve(strict=True)
    (view / "datasets").symlink_to(data_datasets, target_is_directory=True)
    cache = view / "cache"
    cache.mkdir()
    cache_receipts = preflight.get("cache_inputs", {})
    for filename in CACHE_FILES:
        source = (run_dir / "cache" / filename).resolve(strict=True)
        record = cache_receipts.get(filename, {})
        expected_source = (data_root / "cache" / filename).resolve(strict=True)
        if (
            source != expected_source
            or not source.is_file()
            or record.get("path") != str(expected_source)
            or record.get("bytes") != source.stat().st_size
            or record.get("mtime_ns") != source.stat().st_mtime_ns
        ):
            raise ValueError(f"R01 cached input differs from its original preflight: {filename}")
        if record.get("sha256") and record["sha256"] != sha256_file(source):
            raise ValueError(f"R01 hashed cache input differs from its preflight: {filename}")
        (cache / filename).symlink_to(source)
    (view / "artifacts").symlink_to(
        (run_dir / "artifacts").resolve(strict=True), target_is_directory=True
    )
    translated = dict(preflight)
    translated["trainer_source_identity"] = {
        "source_identity_sha256": producer["source_identity_sha256"],
        "files": producer["files"],
        "portable_source_identity_sha256": producer["portable_source_identity_sha256"],
    }
    export_compatible_receipt = dict(fit_receipt)
    export_compatible_receipt["schema"] = "tabdata-l4-fit-producer.v1"
    write_json_new(view / "command.json", command)
    write_json_new(view / "adapter_preflight.json", translated)
    write_json_new(view / "producer_receipt.json", export_compatible_receipt)
    receipt = {
        "schema": "tabdata-r01-fit-recovery.v2",
        "status": "existing_fit_validated_no_refit",
        "job": job,
        "original_run_dir": str(run_dir),
        "original_command_sha256": sha256_file(run_dir / "command.json"),
        "original_preflight_sha256": sha256_file(run_dir / "adapter_preflight.json"),
        "recovery_fit_receipt": export_compatible_receipt,
        "isolated_export_receipt_schema_translation": {
            "from": fit_receipt.get("schema"),
            "to": export_compatible_receipt["schema"],
            "reason": "frozen export contract expects the historical L4 receipt schema",
        },
        "recovery_fit_receipt_sha256": sha256_file(view / "producer_receipt.json"),
        "fit_view": str(view),
        "source_identity_bridge": {
            "portable_file_count": 61,
            "producer_file_count": 60,
            "portable_only_file": "src/wnn/training/README.md",
            "portable_identity_sha256": translated["trainer_source_identity"].get(
                "portable_source_identity_sha256"
            ),
            "producer_identity_sha256": translated["trainer_source_identity"][
                "source_identity_sha256"
            ],
        },
        "checkpoint_identity": identity,
        "optimizer_budget": update_summary,
        "original_actor_phase": actor_evidence,
        "old_fit_outputs_modified": False,
    }
    receipt_path = recovery_root / "receipts" / f"{name}.json"
    write_json_new(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path)
    receipt["receipt_sha256"] = sha256_file(receipt_path)
    return view, receipt


def validate_existing_fit(
    project_root: Path,
    packet_root: Path,
    source_root: Path,
    recovery_root: Path,
    job: dict[str, Any],
) -> dict[str, Any]:
    """Validate one completed R01 fit and emit a new isolated receipt/view; never train."""
    root = project_root.resolve(strict=True)
    run_dir = require_under(Path(str(job["run_dir"])), root / "runs", "fit workspace")
    if Path(str(job["run_dir"])).is_symlink():
        raise ValueError("original fit workspace cannot be a symlink")
    if source_root.is_symlink():
        raise ValueError("trainer source root cannot be a symlink")
    source_root = require_under(source_root, root / "runtime", "trainer source")
    if (root / "state").is_symlink():
        raise ValueError("project state directory cannot be a symlink")
    state_root = (root / "state").resolve(strict=True)
    if recovery_root.name != "r01-delivery-recovery-v2":
        raise ValueError("recovery root must use the isolated v2 delivery directory name")
    if recovery_root.parent.resolve(strict=True) != state_root or recovery_root.is_symlink():
        raise ValueError("recovery root must be a nonsymlink child of the project state directory")
    if recovery_root.exists() and not recovery_root.is_dir():
        raise ValueError("recovery root exists but is not a directory")
    recovery_root = state_root / recovery_root.name
    if recovery_root == root or recovery_root.is_relative_to(run_dir):
        raise ValueError("recovery outputs must be outside the original fit workspace")
    arm = str(job.get("arm"))
    name = str(job.get("id"))
    fold, seed = job.get("fold"), job.get("seed")
    if arm not in EXPECTED_WIDTH or type(fold) is not int or type(seed) is not int:
        raise ValueError("panel cell has invalid R01 arm/fold/seed identity")
    if sha256_file(packet_root / "PANEL.json") != EXPECTED_PANEL_SHA256:
        raise ValueError("R01 panel differs from the frozen source panel")
    source_identity, producer, fit_recipe, fit_helper, objects = _verify_source_and_adapter(
        root, packet_root, source_root
    )
    adapter, bundle, deployed = objects
    split_path = root / "splits" / f"fold{fold}.json"
    split = read_json(split_path)
    split_sha = sha256_file(split_path)
    command_path, preflight_path = run_dir / "command.json", run_dir / "adapter_preflight.json"
    for path in (command_path, preflight_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"original R01 command ancestry is missing or symlinked: {path}")
    command, preflight = read_json(command_path), read_json(preflight_path)
    canonical_data = (root / "data").resolve(strict=True)
    expected_datasets = (canonical_data / "datasets").resolve(strict=True)
    if (
        command.get("schema") != "tabdata-b300-command.v1"
        or command.get("mode") != "fit"
        or command.get("run_name") != name
        or command.get("python") != job.get("argv", [None])[0]
        or command.get("arm") != arm
        or command.get("fold") != fold
        or command.get("seed") != seed
        or Path(command.get("run_dir", "")).resolve() != run_dir
        or Path(command.get("source_root", "")).resolve() != source_root
        or command.get("source_manifest_sha256")
        != sha256_file(root / "runtime/b300-r01-v1/code/a266_source_manifest.json")
        or command.get("trainer_source_identity") != source_identity
        or command.get("deployed_source_tree") != deployed
        or command.get("adapter_bundle_identity") != bundle
        or command.get("adapter_sha256") != bundle.get("files", {}).get("train_cell.py")
        or preflight.get("trainer_source_identity") != source_identity
        or preflight.get("deployed_source_tree") != deployed
        or preflight.get("adapter_bundle_identity") != bundle
        or preflight.get("split_manifest_sha256") != split_sha
        or Path(preflight.get("split_manifest", "")).resolve() != split_path.resolve()
        or command.get("data_root") != str(canonical_data)
        or preflight.get("data_root") != str(canonical_data)
        or preflight.get("fit_train_groups") != split.get("fit", {}).get("train")
        or preflight.get("fit_valid_groups") != split.get("fit", {}).get("valid")
        or split.get("evaluation", {}).get("valid") != preflight.get("evaluation_groups")
        or preflight.get("source_identities") != split.get("sources")
        or preflight.get("fold") != fold
        or preflight.get("n_folds") != 4
        or split.get("fold") != fold
        or split.get("n_folds") != 4
        or not (run_dir / "datasets").is_symlink()
        or (run_dir / "datasets").resolve(strict=True) != expected_datasets
        or preflight.get("workspace_links", {}).get("datasets") != str(expected_datasets)
        or preflight.get("workspace_links", {}).get("cache_is_local_overlay") is not True
    ):
        raise ValueError(
            "R01 fit command, source, adapter or split ancestry differs from the panel"
        )
    training_argv = fit_recipe.build_training_argv(
        read_json(root / "runtime/b300-r01-v1/code/command_builder.json"),
        arm=arm,
        fold=fold,
        seed=seed,
        run_name=name,
        split_manifest=split_path,
        mode="fit",
    )
    if command.get("training_argv") != training_argv:
        raise ValueError("saved R01 command is not the exact frozen cold-start fit argv")
    training_argv_hash = hashlib.sha256(
        json.dumps(training_argv, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if command.get("training_argv_sha256") != training_argv_hash:
        raise ValueError("R01 command hash differs from its canonical training argv")
    if command.get("source_git_metadata_required") is not False:
        raise ValueError("R01 source provenance unexpectedly requires mutable Git metadata")
    runtime = command.get("runtime_environment", {})
    if any(
        runtime.get(key) != value
        for key, value in {
            "NVIDIA_TF32_OVERRIDE": "0",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
            "WNN_PROJECT_ROOT": str(run_dir),
            "WNN_LOADER_RAM_GB": "1",
            "WNN_LOADER_WORKERS": "0",
            "WNN_GPU_MEMORY_BYTES": str(6 * 1024**3),
            "PYTHONDONTWRITEBYTECODE": "1",
            "WNN_THREADS": "2",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
        }.items()
    ):
        raise ValueError("R01 fit runtime did not record the frozen strict-FP32/thread environment")
    source_path = str(source_root / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from wnn.training.checkpoints import serialized_args
    from wnn.training.cli import build_parser

    parsed_args = serialized_args(build_parser().parse_args(training_argv))
    # The canonical trainer appends this verified derived field after CLI parsing.
    parsed_args["split_manifest_sha256"] = split_sha
    run_name = name
    output = run_dir / "artifacts/runs" / run_name
    for path in (
        output / "provenance.json",
        output / "history.json",
        output / "training_state.pt",
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"original R01 fit evidence is missing or symlinked: {path}")
    provenance = read_json(output / "provenance.json")
    history = json.loads((output / "history.json").read_text(encoding="utf-8"))
    if not isinstance(history, list) or provenance.get("args") != parsed_args:
        raise ValueError("trainer provenance arguments differ from the frozen R01 command")
    trainer = provenance.get("trainer_implementation")
    if (
        not isinstance(trainer, dict)
        or trainer.get("schema") != producer["schema"]
        or trainer.get("sha256") != producer["source_identity_sha256"]
        or trainer.get("files") != producer["files"]
        or trainer.get("missing_dependency_files") != []
        or not isinstance(trainer.get("runtime_dependencies", {}).get("attrs"), str)
    ):
        raise ValueError(
            "producer provenance is not the exact 60-file subset of the portable source"
        )
    if (
        provenance.get("split_manifest_sha256") != split_sha
        or provenance.get("selected_epoch") != 5
        or provenance.get("seed") != seed
    ):
        raise ValueError("R01 fit provenance differs from the split or fixed endpoint")
    provenance_groups = provenance.get("fit_row_groups", {})
    if (
        not isinstance(provenance_groups, dict)
        or not _same_group_membership(
            provenance_groups.get("train"), preflight.get("fit_train_groups")
        )
        or not _same_group_membership(
            provenance_groups.get("valid"), preflight.get("fit_valid_groups")
        )
    ):
        raise ValueError("R01 fit train/valid group membership differs from the split")
    update_summary = validate_epoch_records(history, provenance)
    actor_evidence = _verify_completed_training_phase(run_dir, job)
    checkpoint_identity = _verify_checkpoint_set(
        run_dir, run_name, arm, provenance, source_root, root
    )
    expected_producer_input = dict(preflight)
    expected_producer_input["trainer_source_identity"] = {
        "source_identity_sha256": producer["source_identity_sha256"],
        "files": producer["files"],
    }
    fit_receipt = fit_helper.extract_fit_receipt(
        run_dir,
        output,
        run_name=run_name,
        arm=arm,
        fold=fold,
        seed=seed,
        input_receipt=expected_producer_input,
        adapter_bundle_identity=bundle,
    )
    fit_receipt["recovery_checkpoint_identity"] = checkpoint_identity
    fit_receipt["recovery_update_budget"] = update_summary
    fit_receipt["source_identity_bridge"] = {
        "portable_identity_sha256": source_identity["source_identity_sha256"],
        "producer_identity_sha256": producer["source_identity_sha256"],
        "portable_only_file": producer["portable_only_file"],
    }
    _, receipt = _make_fit_view(
        root,
        recovery_root,
        run_dir,
        job,
        preflight,
        command,
        fit_receipt,
        producer,
        checkpoint_identity,
        update_summary,
        actor_evidence,
    )
    return receipt
