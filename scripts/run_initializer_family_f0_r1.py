#!/usr/bin/env python3
"""Run a bounded, component-complete F0 frozen-parent initializer experiment.

The standard-library preflight is read-only. The profile and fit-export phases
require an externally admitted single-GPU resource envelope.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

FORMAT = "wnn.initializer-family-f0-recipe"
RESULT_FORMAT = "wnn.initializer-family-f0-result"
RUN_MODES = ("constant", "first_row", "prefix")
ALL_MODES = ("zero", *RUN_MODES)
ROWS = 20_000
CHUNK_ROWS = 2_000
N_FEATURES = 211
EPOCHS = 2
UPDATES_PER_MODE = 208
HOST_LIMIT_BYTES = 5_000_000_000
CUDA_RESERVED_LIMIT_BYTES = 3_500_000_000
OBSERVED_VRAM_LIMIT_BYTES = 5_000_000_000
MAX_CONTRACT_BYTES = 256_000
MAX_BINDING_BYTES = 128_000
MAX_SUBSET_BYTES = 1_000_000
MAX_SPLIT_BYTES = 2_000_000
MAX_JSON_BYTES = 8_000_000
MAX_CACHE_METADATA_BYTES = 512_000


def sha256_file(path: Path) -> str:
    """Hash a file without loading its contents into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_json(path: Path, max_bytes: int, label: str) -> tuple[dict[str, Any], bytes]:
    """Read one JSON object only after checking its byte bound."""
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ValueError(f"{label} is outside its {max_bytes}-byte read bound")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError(f"{label} changed while being read")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, raw


def digest_ints(values: list[int]) -> str:
    """Hash canonical little-endian int64 IDs."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<q", int(value)))
    return digest.hexdigest()


def validate_subset(split: dict[str, Any], subset: dict[str, Any], split_sha: str) -> None:
    """Prove every selected component includes all fit nodes across both sources."""
    if subset.get("split_manifest_sha256") != split_sha:
        raise ValueError("fit subset was prepared from another fold-0 split")
    if subset.get("experiment_id") != "initializer-family-f0-r1":
        raise ValueError("fit subset belongs to another experiment")
    component_rows: dict[int, set[tuple[str, int]]] = {}
    seq_identity: dict[tuple[str, int], tuple[int, int]] = {}
    for source in ("train", "valid"):
        for row in split["sequences"][source]:
            group = int(row["row_group"])
            component = int(row["component"])
            identity = (source, group)
            seq_identity[identity] = (int(row["seq_ix"]), component)
            component_rows.setdefault(component, set()).add(identity)

    fit = {(source, int(group)) for source in ("train", "valid") for group in split["fit"][source]}
    evaluation = {("valid", int(group)) for group in split["evaluation"]["valid"]}
    excluded = {
        (source, int(group))
        for source in ("train", "valid")
        for group in split["excluded"].get(source, [])
    }
    selected_groups = subset.get("selected_groups")
    if not isinstance(selected_groups, dict):
        raise ValueError("fit subset has no source-separated selected_groups")
    selected = {
        (source, int(group))
        for source in ("train", "valid")
        for group in selected_groups.get(source, [])
    }
    selected_components = subset.get("selected_components")
    if (
        not isinstance(selected_components, list)
        or selected_components != sorted(set(selected_components))
        or any(type(component) is not int for component in selected_components)
    ):
        raise ValueError("selected component IDs are not canonical unique integers")
    actual_components = {seq_identity[identity][1] for identity in selected}
    if actual_components != set(selected_components):
        raise ValueError("selected groups and component IDs disagree")
    expected_selected = {
        member for component in selected_components for member in component_rows[component]
    }
    if selected != expected_selected:
        raise ValueError("selected subset omits a train/valid node from a chosen component")
    if selected != selected.intersection(fit):
        raise ValueError("selected component includes a non-fitting node")
    if selected.intersection(evaluation) or selected.intersection(excluded):
        raise ValueError("selected fitting rows overlap evaluation or excluded membership")

    members = subset.get("selected_members")
    if not isinstance(members, list):
        raise ValueError("fit subset lacks source-qualified member identities")
    observed_members = {(str(row.get("source")), int(row.get("row_group", -1))) for row in members}
    if observed_members != selected or len(members) != len(selected):
        raise ValueError("fit subset member table differs from selected_groups")
    for row in members:
        identity = (str(row["source"]), int(row["row_group"]))
        expected_seq_ix, expected_component = seq_identity[identity]
        if int(row.get("seq_ix", -1)) != expected_seq_ix:
            raise ValueError("fit subset seq_ix differs from the frozen source-qualified split")
        if int(row.get("component", -1)) != expected_component:
            raise ValueError("fit subset component differs from the frozen split")

    sorted_members = sorted(selected)
    members_bytes = json.dumps(sorted_members, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    if hashlib.sha256(members_bytes).hexdigest() != subset.get("selected_members_sha256"):
        raise ValueError("selected member digest is invalid")
    if digest_ints(sorted(selected_components)) != subset.get("selected_component_sha256"):
        raise ValueError("selected component digest is invalid")
    evaluation_groups = sorted(int(group) for group in split["evaluation"]["valid"])
    if (
        subset.get("evaluation_groups") != evaluation_groups
        or subset.get("evaluation_group_count") != len(evaluation_groups)
        or digest_ints(evaluation_groups) != subset.get("evaluation_membership_sha256")
    ):
        raise ValueError("fit subset changed the fixed exposed evaluation membership")
    counts = Counter(source for source, _ in selected)
    if subset.get("selected_source_group_counts") != {
        "train": counts["train"],
        "valid": counts["valid"],
    }:
        raise ValueError("fit subset source counts are inconsistent")
    if len(selected) != subset.get("selection", {}).get("selected_fit_sequence_count"):
        raise ValueError("fit subset count differs from its selection record")


def validate_source_binding(
    script: Path, contract_path: Path, contract: dict[str, Any], contract_raw: bytes
) -> dict[str, Any]:
    """Verify this driver and all frozen runtime sources against its binding."""
    binding_path = script.resolve().parents[1] / contract["source_binding_path"]
    binding, binding_raw = bounded_json(binding_path, MAX_BINDING_BYTES, "source binding")
    if hashlib.sha256(contract_raw).hexdigest() != binding.get("contract_sha256"):
        raise ValueError("source binding refers to another recipe contract")
    source_roots = binding.get("source_roots", {})
    for root_key, files_key in (
        ("trainer", "trainer_files"),
        ("initializer", "initializer_files"),
        ("recipe", "recipe_files"),
        ("grader", "grader_files"),
    ):
        root = Path(source_roots[root_key]).resolve(strict=True)
        files = binding.get(files_key)
        if not isinstance(files, dict) or not files:
            raise ValueError(f"source binding has no {root_key} files")
        actual = {relative: sha256_file(root / relative) for relative in files}
        if actual != files:
            raise ValueError(f"frozen {root_key} source tree has changed")
    relative_script = str(script.resolve().relative_to(script.resolve().parents[1]))
    recipe_root = Path(source_roots["recipe"]).resolve()
    actual_script = sha256_file(recipe_root / relative_script)
    if actual_script != binding.get("runner_sha256"):
        raise ValueError("initializer family runner source changed after packet freeze")
    subset_path = script.resolve().parents[1] / contract["subset_manifest_path"]
    if sha256_file(subset_path) != binding.get("subset_manifest_sha256"):
        raise ValueError("complete-component subset manifest changed")
    return {
        "path": str(binding_path),
        "sha256": hashlib.sha256(binding_raw).hexdigest(),
        "runner_sha256": actual_script,
        "source_roots": source_roots,
    }


def static_preflight(
    script: Path, contract_path: Path, receipt_path: Path | None
) -> dict[str, Any]:
    """Validate pinned metadata and component membership without importing ML libraries."""
    contract, contract_raw = bounded_json(contract_path, MAX_CONTRACT_BYTES, "recipe contract")
    if contract.get("format") != FORMAT or contract.get("version") != 1:
        raise ValueError("unsupported initializer family contract")
    source_binding = validate_source_binding(script, contract_path, contract, contract_raw)
    repo = script.resolve().parents[1]
    if contract.get("contract_path") != str(contract_path.resolve().relative_to(repo)):
        raise ValueError("contract path differs from its declared recipe location")
    root = Path(contract["data_root"]).resolve(strict=True)
    split_path = root / contract["split_path"]
    split, split_raw = bounded_json(split_path, MAX_SPLIT_BYTES, "frozen split manifest")
    split_sha = hashlib.sha256(split_raw).hexdigest()
    if split_sha != contract["split_sha256"]:
        raise ValueError("frozen fold-0 split changed")
    subset_path = repo / contract["subset_manifest_path"]
    subset, subset_raw = bounded_json(subset_path, MAX_SUBSET_BYTES, "component subset")
    if hashlib.sha256(subset_raw).hexdigest() != source_binding.get("subset_manifest_sha256"):
        raise ValueError("component subset hash differs from its source binding")
    validate_subset(split, subset, split_sha)
    selected_count = len(subset["selected_members"])
    if selected_count != 104 or selected_count * EPOCHS != UPDATES_PER_MODE:
        raise ValueError(
            "fixed two-epoch initializer budget must contain exactly 208 updates per mode"
        )
    if contract.get("profile_member") not in subset["selected_members"]:
        raise ValueError("resource profile sequence is not a member of the frozen fit subset")

    cache = contract["cache"]
    metadata_path = root / cache["root"] / "metadata.json"
    metadata, metadata_raw = bounded_json(
        metadata_path, MAX_CACHE_METADATA_BYTES, "clean T3 cache metadata"
    )
    metadata_sha256 = hashlib.sha256(metadata_raw).hexdigest()
    if metadata_sha256 != cache["metadata_sha256"]:
        raise ValueError("clean T3 producer metadata changed")
    if (
        metadata.get("fold") != 0
        or metadata.get("split_manifest_sha256") != split_sha
        or metadata.get("fit_groups") != split.get("fit")
        or metadata.get("evaluation_valid_groups") != split.get("evaluation", {}).get("valid")
        or metadata.get("source_coefficients") != cache["source_coefficients"]
        or metadata.get("sources") != cache["teacher_sources"]
    ):
        raise ValueError("clean T3 cache target ancestry or membership differs from the contract")
    target_files = {
        name: value.get("sha256")
        for name, value in metadata.get("files", {}).items()
        if isinstance(value, dict) and name in {"main_homogeneous_train", "main_homogeneous_valid"}
    }
    if target_files != cache["target_files"]:
        raise ValueError("clean T3 target file identities differ from the frozen contract")
    for teacher in cache["teacher_sources"]:
        for path_key, digest_key in (
            ("checkpoint_path", "checkpoint_sha256"),
            ("provenance_path", "provenance_sha256"),
        ):
            if sha256_file(root / teacher[path_key]) != teacher[digest_key]:
                raise ValueError(f"clean T3 source changed: {teacher[path_key]}")
        teacher_provenance, _ = bounded_json(
            root / teacher["provenance_path"], MAX_JSON_BYTES, "clean T3 teacher provenance"
        )
        teacher_fit = teacher_provenance.get("fit_row_groups", {})
        if (
            teacher_provenance.get("split_manifest_sha256") != split_sha
            or teacher_provenance.get("args", {}).get("name") != teacher["run"]
            or teacher_provenance.get("args", {}).get("seed") != teacher["seed"]
            or teacher_provenance.get("checkpoint_sha256", {}).get(
                Path(teacher["checkpoint_path"]).name
            )
            != teacher["checkpoint_sha256"]
            or any(
                set(map(int, teacher_fit.get(source, []))) != set(map(int, split["fit"][source]))
                for source in ("train", "valid")
            )
        ):
            raise ValueError(f"clean T3 teacher provenance is not source-clean: {teacher['run']}")

    parent = contract["parent"]
    for key, path_key in (
        ("checkpoint_sha256", "checkpoint_path"),
        ("provenance_sha256", "provenance_path"),
    ):
        path = root / parent[path_key]
        if sha256_file(path) != parent[key]:
            raise ValueError(f"frozen parent {path_key} hash changed")
    provenance, _ = bounded_json(
        root / parent["provenance_path"], MAX_JSON_BYTES, "parent provenance"
    )
    expected_args = parent["expected_args"]
    args = provenance.get("args", {})
    if any(args.get(key) != value for key, value in expected_args.items()):
        raise ValueError("frozen CP5 parent arguments or teacher recipe changed")
    if provenance.get("checkpoint_sha256", {}).get("model_ep5.pt") != parent["checkpoint_sha256"]:
        raise ValueError("parent provenance does not bind the CP5 checkpoint")
    if provenance.get("split_manifest_sha256") != split_sha:
        raise ValueError("parent provenance belongs to another clean F0 split")
    if provenance.get("fit_row_groups") != parent["full_backbone_fit_row_groups"]:
        raise ValueError("parent fitting membership differs from the pinned backbone ancestry")
    parent_fit_groups = provenance.get("fit_row_groups", {})
    if any(
        set(map(int, parent_fit_groups.get(source, []))) != set(map(int, split["fit"][source]))
        for source in ("train", "valid")
    ):
        raise ValueError("CP5 parent fitting membership is not the complete frozen F0 fit set")

    propensity = contract["propensity"]
    for path_key, digest_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("provenance", "provenance_sha256"),
    ):
        if sha256_file(root / propensity[path_key]) != propensity[digest_key]:
            raise ValueError(f"frozen propensity source changed: {propensity[path_key]}")
    propensity_provenance, _ = bounded_json(
        root / propensity["provenance"], MAX_JSON_BYTES, "propensity model provenance"
    )
    if propensity_provenance.get("args", {}).get("name") != propensity["run"]:
        raise ValueError("propensity checkpoint provenance names a different model")

    sidecar_path = root / contract["parent_prediction_sidecar_path"]
    if sha256_file(sidecar_path) != contract["parent_prediction_sidecar_sha256"]:
        raise ValueError("frozen parent prediction sidecar changed")
    sidecar, _ = bounded_json(sidecar_path, MAX_JSON_BYTES, "parent prediction sidecar")
    expected_row_groups = [int(row["row_group"]) for row in split["sequences"]["valid"]]
    if (
        sidecar.get("run") != parent["run"]
        or sidecar.get("epoch") != 5
        or sidecar.get("split") != "valid"
        or sidecar.get("inference_precision") != "fp32"
        or sidecar.get("checkpoint_sha256") != parent["checkpoint_sha256"]
        or sidecar.get("split_manifest_sha256") != split_sha
        or sidecar.get("row_groups") != expected_row_groups
        or sidecar.get("prediction_sha256") != contract["parent_prediction_sha256"]
    ):
        raise ValueError("existing CP5 parent predictions are not the pinned FP32 full-valid panel")

    report = {
        "format": "wnn.initializer-family-f0-preflight",
        "version": 1,
        "contract_sha256": hashlib.sha256(contract_raw).hexdigest(),
        "source_binding": source_binding,
        "split_sha256": split_sha,
        "subset_manifest_sha256": hashlib.sha256(subset_raw).hexdigest(),
        "selected_components": len(subset["selected_components"]),
        "selected_groups": subset["selected_source_group_counts"],
        "selected_total_sequences": len(subset["selected_members"]),
        "selected_all_component_nodes_across_train_valid": True,
        "evaluation_groups": len(subset["evaluation_groups"]),
        "parent": {
            "run": parent["run"],
            "checkpoint_sha256": parent["checkpoint_sha256"],
            "full_backbone_fit_row_groups": {
                split_name: len(groups)
                for split_name, groups in parent["full_backbone_fit_row_groups"].items()
            },
            "prediction_sha256": contract["parent_prediction_sha256"],
            "prediction_rows": len(expected_row_groups),
        },
        "clean_t3_cache": {
            "metadata_sha256": metadata_sha256,
            "target_files": target_files,
            "teacher_count": len(cache["teacher_sources"]),
            "target_formula": cache["target_formula"],
        },
        "propensity": {
            "run": propensity["run"],
            "checkpoint_sha256": propensity["checkpoint_sha256"],
            "provenance_sha256": propensity["provenance_sha256"],
        },
        "fit_budget": {
            "epochs": EPOCHS,
            "updates_per_mode": UPDATES_PER_MODE,
            "learned_modes": list(RUN_MODES),
            "total_updates": EPOCHS * len(subset["selected_members"]) * len(RUN_MODES),
            "checkpoint": "fixed terminal epoch index 1 (second epoch)",
        },
        "preflight_reads_payload_arrays": False,
        "gpu_launch_performed": False,
        "quality_claim": None,
    }
    if receipt_path is not None:
        write_new_json(receipt_path, report)
    return report


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    """Write one immutable JSON receipt without replacing an earlier attempt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite an existing attempt: {path}")
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def require_absent_outputs(contract: dict[str, Any], phase: str) -> None:
    """Reject reused run, prediction, receipt, or profile paths before device work."""
    root = Path(contract["data_root"]).resolve()
    if phase == "profile":
        paths = [Path(contract["profile_result_path"])]
    elif phase == "fit-export":
        paths = [Path(contract["fit_result_path"])]
        paths.extend(root / relative for relative in contract["result_paths"].values())
        for run in contract["runs"].values():
            run_root = root / "artifacts" / "runs" / run
            prediction = root / "cache" / f"pred_{run}_valid.npy"
            paths.extend(
                (
                    run_root,
                    prediction,
                    prediction.with_suffix(".json"),
                    prediction.with_name(prediction.name + ".partial"),
                    prediction.with_suffix(".json.partial"),
                )
            )
    else:
        raise ValueError(f"unknown execution phase: {phase}")
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to reuse existing output paths: " + ", ".join(existing))


def validate_profile_receipt(contract: dict[str, Any], expected_sha256: str) -> dict[str, Any]:
    """Require the exact completed CP5 resource smoke before any fit/export."""
    path = Path(contract["profile_result_path"]).resolve(strict=True)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError("profile receipt SHA-256 differs from the fit command pin")
    profile, _ = bounded_json(path, MAX_JSON_BYTES, "CP5 initializer profile receipt")
    expected_contract = hashlib.sha256(
        (Path(contract["source_root"]) / contract["contract_path"]).read_bytes()
    ).hexdigest()
    modes = profile.get("modes")
    resources = profile.get("resources_at_start", {})
    host_max = resources.get("host_memory_max_bytes")
    if (
        profile.get("format") != "wnn.initializer-family-f0-resource-profile"
        or profile.get("contract_sha256") != expected_contract
        or profile.get("attempted_updates") != len(RUN_MODES)
        or profile.get("completed_updates") != len(RUN_MODES)
        or profile.get("source_member") != contract["profile_member"]
        or (
            host_max is not None
            and (host_max > HOST_LIMIT_BYTES or HOST_LIMIT_BYTES - host_max >= 4096)
        )
        or resources.get("observed_vram_bytes_before", OBSERVED_VRAM_LIMIT_BYTES)
        >= OBSERVED_VRAM_LIMIT_BYTES
        or not isinstance(modes, list)
        or [row.get("mode") for row in modes] != list(RUN_MODES)
        or any(
            row.get("one_full_sequence_update", {}).get("nonzero_initializer_gradient") is not True
            or not math.isfinite(row.get("one_full_sequence_update", {}).get("loss", float("nan")))
            or row.get("source_member") != contract["profile_member"]
            or row.get("cuda_peak_reserved_bytes", CUDA_RESERVED_LIMIT_BYTES + 1)
            > CUDA_RESERVED_LIMIT_BYTES
            or row.get("observed_vram_bytes_after", OBSERVED_VRAM_LIMIT_BYTES)
            >= OBSERVED_VRAM_LIMIT_BYTES
            for row in modes
        )
    ):
        raise ValueError(
            "profile receipt is incomplete, mismatched, or outside the declared resource caps"
        )
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "completed_updates": profile["completed_updates"],
        "contract_sha256": expected_contract,
    }


def load_runtime_modules(contract: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    """Load the exact frozen trainer and initializer sources for an admitted run.

    ``Any`` is unavoidable here because these source modules are loaded from
    hash-bound paths after the standard-library preflight.
    """
    trainer_root = Path(contract["trainer_source_root"]).resolve(strict=True)
    initializer_root = Path(contract["initializer_source_root"]).resolve(strict=True)
    sys.path.insert(0, str(trainer_root / "src"))
    sys.path.insert(0, str(trainer_root / "scripts"))
    trainer_spec = importlib.util.spec_from_file_location(
        "initializer_family_frozen_train_rnn", trainer_root / "scripts/train_rnn.py"
    )
    if trainer_spec is None or trainer_spec.loader is None:
        raise RuntimeError("could not load the frozen training source")
    import types

    trainer_module = types.ModuleType(trainer_spec.name)
    sys.modules[trainer_spec.name] = trainer_module
    trainer_spec.loader.exec_module(trainer_module)

    helper_spec = importlib.util.spec_from_file_location(
        "initializer_family_profile_helpers",
        initializer_root / "scripts/profile_initializer_frontier_r3.py",
    )
    if helper_spec is None or helper_spec.loader is None:
        raise RuntimeError("could not load the reviewed initializer profile helpers")
    helper_module = types.ModuleType(helper_spec.name)
    sys.modules[helper_spec.name] = helper_module
    helper_spec.loader.exec_module(helper_module)

    import numpy as np
    import torch

    # The frozen trainer owns general WNN imports. Resolve the two initializer
    # modules only after its package is loaded, from their separate pinned tree.
    import wnn

    initializer_package = str((initializer_root / "src" / "wnn").resolve(strict=True))
    if initializer_package in wnn.__path__:
        wnn.__path__.remove(initializer_package)
    wnn.__path__.insert(0, initializer_package)
    initializer_training_path = (
        initializer_root / "src" / "wnn" / "initializer_training.py"
    ).resolve(strict=True)
    state_initialization_path = (
        initializer_root / "src" / "wnn" / "state_initialization.py"
    ).resolve(strict=True)
    for module_name in ("wnn.initializer_training", "wnn.state_initialization"):
        if module_name in sys.modules:
            raise RuntimeError(
                f"initializer module was imported before its pinned source path: {module_name}"
            )
    initializer_training = importlib.import_module("wnn.initializer_training")
    state_initialization = importlib.import_module("wnn.state_initialization")
    if (
        Path(initializer_training.__file__).resolve() != initializer_training_path
        or Path(state_initialization.__file__).resolve() != state_initialization_path
    ):
        raise RuntimeError(
            "initializer modules did not load from their separately pinned source tree"
        )

    trainer_module.ROOT = Path(contract["data_root"]).resolve()
    trainer_module.HEURISTICS = False
    trainer_module.RAW_ONLY = False
    trainer_module.ONLINE_NORM = False
    trainer_module.RESEARCH_FEATURES = "none"
    trainer_module.PRIVILEGED_MODE = "none"
    trainer_module.INPUT_PROPENSITY = None
    return (
        trainer_module,
        helper_module,
        initializer_training.initializer_sequence_loss,
        state_initialization.InitializedGRU,
        (torch, np, state_initialization.export_native_initializer),
    )


def cgroup_memory_value(filename: str) -> int | None:
    """Read one cgroup-v2 memory counter for admission evidence."""
    try:
        entries = [
            line.split(":", 2)[2]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        ]
        if len(entries) != 1:
            return None
        path = (Path("/sys/fs/cgroup") / entries[0].lstrip("/")).resolve()
        root = Path("/sys/fs/cgroup").resolve()
        if not path.is_relative_to(root):
            return None
        raw = (path / filename).read_text().strip()
        return int(raw) if raw.isascii() and raw.isdigit() else None
    except (OSError, ValueError):
        return None


def require_runtime_envelope(torch: Any, helper: Any, device: Any) -> dict[str, int | None]:
    """Fail before GPU work if the exact pre-admitted host/VRAM caps are absent."""
    host_max = cgroup_memory_value("memory.max")
    if host_max is not None and (
        host_max > HOST_LIMIT_BYTES or HOST_LIMIT_BYTES - host_max >= 4096
    ):
        raise RuntimeError("cgroup memory.max differs from the declared 5 GB reservation")
    if not torch.cuda.is_available() or device.type != "cuda" or device.index is None:
        raise RuntimeError("an explicit indexed CUDA device is required")
    observed = helper.nvidia_smi_used_bytes()
    if observed >= OBSERVED_VRAM_LIMIT_BYTES:
        raise RuntimeError("observed VRAM is already at the 5 GB ceiling")
    return {
        "host_memory_max_bytes": host_max,
        "host_memory_current_bytes": cgroup_memory_value("memory.current"),
        "observed_vram_bytes_before": observed,
    }


def canonical_tensor_sha(value: Any, torch: Any) -> str:
    """Hash one tensor's canonical little-endian FP32 CPU representation."""
    array = (
        value.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
    )
    return hashlib.sha256(array.tobytes()).hexdigest()


def build_backbone(trainer: Any, state: dict[str, Any], device: Any) -> Any:
    """Reconstruct the parent H256×2 plain GRU with its exact frozen buffers."""
    model_args = {
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "hidden": 256,
        "layers": 2,
        "dropout": 0.0,
        "readout_activation": "gelu",
        "numeric_map": "none",
        "ple_intervals": 8,
    }
    model = trainer.make_model(model_args, N_FEATURES, 2)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("CP5 checkpoint tensors do not match the frozen linear H256×2 GRU")
    return model.to(device=device)


def load_fit_data(
    trainer: Any, contract: dict[str, Any], subset: dict[str, Any], split: Any
) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    """Strictly validate target/propensity caches and build source-local datasets."""
    root = Path(contract["data_root"]).resolve()
    cache = contract["cache"]
    target_validation = trainer.validate_distillation_cache(
        root / cache["root"], root, split, "homogeneous", require_families=False
    )
    if (
        target_validation["metadata_sha256"] != cache["metadata_sha256"]
        or target_validation["verified_files"] != cache["target_files"]
        or target_validation["metadata"].get("source_coefficients") != cache["source_coefficients"]
        or target_validation["metadata"].get("sources") != cache["teacher_sources"]
    ):
        raise ValueError("strict T3 cache validation differs from the frozen recipe")
    propensity = contract["propensity"]
    for path_key, digest_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("provenance", "provenance_sha256"),
    ):
        if sha256_file(root / propensity[path_key]) != propensity[digest_key]:
            raise ValueError(f"frozen propensity source changed: {propensity[path_key]}")
    prop_cache_paths = {
        source: trainer.prop_cache_path("c20260929_prop_f0", source)
        for source in ("train", "valid")
    }
    for source, pin in contract["propensity"]["caches"].items():
        path = prop_cache_paths[source]
        expected_path = root / pin["path"]
        if path.resolve() != expected_path.resolve():
            raise ValueError(f"propensity cache path differs from the frozen recipe: {path}")
        if path.stat().st_size != pin["bytes"] or sha256_file(path) != pin["sha256"]:
            raise ValueError(f"frozen propensity cache changed: {path}")
        sidecar = path.with_suffix(".json")
        if (
            sidecar.resolve() != (root / pin["sidecar_path"]).resolve()
            or sha256_file(sidecar) != pin["sidecar_sha256"]
        ):
            raise ValueError(f"frozen propensity cache sidecar changed: {sidecar}")
    datasets = {
        source: trainer.Sequences(
            source,
            subset["selected_groups"][source],
            prop_cache=prop_cache_paths[source],
            distill_cache_root=root / cache["root"],
            distill_cache_variant="homogeneous",
            distill_cache_metadata=target_validation,
            include_family_targets=False,
        )
        for source in ("train", "valid")
    }
    positions = {
        source: {int(group): index for index, group in enumerate(datasets[source].indices)}
        for source in ("train", "valid")
    }
    return target_validation, datasets, positions, prop_cache_paths


def batch_for_member(
    member: dict[str, Any], datasets: dict[str, Any], positions: dict[str, dict[int, int]]
) -> tuple[Any, ...]:
    """Read one selected source-local sequence and add the batch dimension."""
    source, row_group = str(member["source"]), int(member["row_group"])
    return tuple(value.unsqueeze(0) for value in datasets[source][positions[source][row_group]])


def initialize_batch(
    trainer: Any,
    helper: Any,
    batch: tuple[Any, ...],
    propensity_model: Any,
    torch: Any,
    device: Any,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Apply the frozen feature, T3 blend and main row-weight semantics."""
    x, inputs, targets, supervision = helper.initialize_inputs(
        trainer, batch, propensity_model, torch, device
    )
    return x, inputs, targets, supervision.pop("weights"), supervision


def one_loss_update(
    model: Any,
    optimizer: Any,
    inputs: Any,
    targets: Any,
    weights: Any,
    sequence_loss: Any,
    torch: Any,
) -> dict[str, float]:
    """Perform one full-sequence initializer-only optimizer update."""
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters or any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise ValueError("only initializer parameters may be trainable")
    if {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} != {
        id(parameter) for parameter in parameters
    }:
        raise ValueError("optimizer parameter groups differ from trainable initializer tensors")
    optimizer.zero_grad(set_to_none=True)
    loss = sequence_loss(
        model,
        inputs,
        targets,
        weights,
        chunk_rows=CHUNK_ROWS,
        checkpoint_chunks=True,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("complete-sequence initializer loss is nonfinite")
    loss.backward()
    if any(parameter.grad is None for parameter in parameters):
        raise FloatingPointError("an initializer parameter did not receive a gradient")
    if any(not torch.isfinite(parameter.grad).all().item() for parameter in parameters):
        raise FloatingPointError("an initializer gradient is nonfinite")
    has_nonzero_gradient = any(
        torch.count_nonzero(parameter.grad).item() for parameter in parameters
    )
    if not has_nonzero_gradient:
        raise FloatingPointError("initializer received no nonzero gradient")
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0, error_if_nonfinite=True)
    if any(parameter.grad is not None for parameter in model.backbone.parameters()):
        raise RuntimeError("frozen backbone accumulated a gradient")
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "gradient_norm_before_clip": float(norm.detach().cpu()),
        "nonzero_initializer_gradient": True,
    }


def make_initializer_model(
    trainer: Any,
    InitializedGRU: Any,
    backbone_state: dict[str, Any],
    mode: str,
    torch: Any,
    device: Any,
    seed: int,
) -> Any:
    """Create a fresh learned initializer over a cloned frozen parent state."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    backbone = trainer.make_model(
        {
            "arch": "gru",
            "cell": "gru",
            "enc": "linear",
            "hidden": 256,
            "layers": 2,
            "dropout": 0.0,
            "readout_activation": "gelu",
            "numeric_map": "none",
            "ple_intervals": 8,
        },
        N_FEATURES,
        2,
    )
    incompatible = backbone.load_state_dict(backbone_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("CP5 parent state changed during initializer construction")
    backbone.to(device=device)
    backbone.requires_grad_(False)
    model = InitializedGRU(backbone, mode, residual_bound=1.0).to(device=device)
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise ValueError("parent backbone parameters must remain frozen")
    if any(parameter.device != device for parameter in model.parameters()):
        raise RuntimeError("initializer module and parent backbone are on different devices")
    if any(buffer.device != device for buffer in model.buffers()):
        raise RuntimeError("initializer buffers are on a different device")
    return model


def make_optimizer(model: Any, torch: Any) -> Any:
    """Build the one frozen AdamW schedule shared by all three learned arms."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("learned initializer has no trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        fused=False,
    )


def run_profile(
    contract: dict[str, Any],
    subset: dict[str, Any],
    trainer: Any,
    helper: Any,
    sequence_loss: Any,
    InitializedGRU: Any,
    modules: tuple[Any, Any, Any],
) -> dict[str, Any]:
    """Measure one CP5 full-sequence update for every learned mode; never fit."""
    torch, np, _ = modules
    from wnn.propensity import load_propensity
    from wnn.resources import configure_torch
    from wnn.splits import load_campaign_split

    root = Path(contract["data_root"]).resolve()
    configure_torch()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", torch.cuda.current_device())
    resources = require_runtime_envelope(torch, helper, device)
    split = load_campaign_split(root / contract["split_path"], root)
    if split.sha256 != contract["split_sha256"]:
        raise ValueError("runtime campaign split differs from the frozen preflight")
    trainer.load_component_fit_subset(
        Path(contract["source_root"]) / contract["subset_manifest_path"],
        split,
        expected_experiment_id=contract["experiment_id"],
    )
    _, datasets, positions, _ = load_fit_data(trainer, contract, subset, split)
    parent = torch.load(
        root / contract["parent"]["checkpoint_path"], map_location="cpu", weights_only=False
    )
    if parent.get("epoch") != 5 or parent.get("args") != json.loads(
        (root / contract["parent"]["provenance_path"]).read_text()
    ).get("args"):
        raise ValueError("loaded parent checkpoint is not the exact frozen CP5 producer")
    base_state = {
        name: value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        for name, value in parent["model"].items()
    }
    if canonical_tensor_sha(base_state["mean"], torch) != contract["normalization"]["mean_sha256"]:
        raise ValueError("loaded CP5 normalization mean differs from the shared 211-feature buffer")
    if canonical_tensor_sha(base_state["std"], torch) != contract["normalization"]["std_sha256"]:
        raise ValueError("loaded CP5 normalization std differs from the shared 211-feature buffer")
    del parent
    propensity_model = load_propensity(root / contract["propensity"]["checkpoint"], device)

    first_member = contract["profile_member"]
    if first_member not in subset["selected_members"]:
        raise ValueError("profile sequence is not included in the frozen fit subset")
    batch = batch_for_member(first_member, datasets, positions)
    x, inputs, targets, weights, supervision = initialize_batch(
        trainer, helper, batch, propensity_model, torch, device
    )
    zero_backbone = build_backbone(trainer, base_state, device)
    zero_backbone.requires_grad_(False)
    zero_model = InitializedGRU(zero_backbone, "zero", residual_bound=1.0).to(device)
    zero_model.eval()
    with torch.no_grad():
        zero_parts = []
        zero_state = None
        for start in range(0, ROWS, CHUNK_ROWS):
            prediction, zero_state = zero_model(inputs[:, start : start + CHUNK_ROWS], zero_state)
            zero_parts.append(prediction)
        zero_prediction = torch.cat(zero_parts, dim=1)[..., :2]

    results = []
    for mode in RUN_MODES:
        model = make_initializer_model(trainer, InitializedGRU, base_state, mode, torch, device, 3)
        model.eval()
        with torch.no_grad():
            initial_parts = []
            state = None
            for start in range(0, ROWS, CHUNK_ROWS):
                prediction, state = model(inputs[:, start : start + CHUNK_ROWS], state)
                initial_parts.append(prediction)
            initial_prediction = torch.cat(initial_parts, dim=1)[..., :2]
        torch.testing.assert_close(initial_prediction, zero_prediction, rtol=1e-5, atol=1e-5)
        optimizer = make_optimizer(model, torch)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        update = one_loss_update(model, optimizer, inputs, targets, weights, sequence_loss, torch)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        results.append(
            {
                "mode": mode,
                "source_member": first_member,
                "one_full_sequence_update": update,
                "initial_prediction_matches_zero": True,
                "elapsed_seconds": elapsed,
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "observed_vram_bytes_after": helper.nvidia_smi_used_bytes(),
            }
        )
        del optimizer, model
        torch.cuda.empty_cache()
    result = {
        "format": "wnn.initializer-family-f0-resource-profile",
        "version": 1,
        "contract_sha256": hashlib.sha256(
            (Path(contract["source_root"]) / contract["contract_path"]).read_bytes()
        ).hexdigest(),
        "execution_device": str(device),
        "precision": "FP32; autocast disabled; TF32 disabled",
        "source_member": first_member,
        "input_sha256": supervision["input_sha256"],
        "target_sha256": supervision["target_sha256"],
        "weight_sha256": supervision["weight_sha256"],
        "global_weight_mass": supervision["total_weight_mass"],
        "cache_target_validation": "strict producer validator passed before profile",
        "parent_backbone": contract["parent"]["checkpoint_sha256"],
        "normalization": contract["normalization"],
        "resources_at_start": resources,
        "modes": results,
        "attempted_updates": len(results),
        "completed_updates": len(results),
        "quality_claim": None,
        "interpretation": "CP5 same-shape one-update resource smoke only; no fitted quality result",
    }
    if any(row["cuda_peak_reserved_bytes"] > CUDA_RESERVED_LIMIT_BYTES for row in results):
        raise RuntimeError("initializer profile exceeded the 3.5 GB CUDA allocator cap")
    if any(row["observed_vram_bytes_after"] >= OBSERVED_VRAM_LIMIT_BYTES for row in results):
        raise RuntimeError("initializer profile reached the 5 GB observed VRAM ceiling")
    return result


def make_provenance(
    contract: dict[str, Any],
    mode: str,
    subset: dict[str, Any],
    orders: list[list[int]],
    updates: list[dict[str, float]],
) -> dict[str, Any]:
    """Build the checkpoint and grader ancestry record for one initialized model."""
    parent = contract["parent"]
    args = {
        "name": contract["runs"][mode],
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "hidden": 256,
        "layers": 2,
        "dropout": 0.0,
        "readout_activation": "gelu",
        "precision": "fp32",
        "seed": 3,
        "epochs": EPOCHS,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 1,
        "no_eval_during_fit": True,
        "split_manifest_sha256": contract["split_sha256"],
        "valid_fold": 0,
        "state_initializer_mode": mode,
        "initialization_parent": f"{parent['run']}@ep5",
        "lr": 1e-3,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "chunk_rows": CHUNK_ROWS,
        "batch_size": 1,
        "auxiliary_losses": "none; main frozen-T3 objective only",
    }
    return {
        "format": "wnn.initializer-family-f0-provenance",
        "version": 1,
        "args": args,
        # This field records all fitting exposure in the inherited checkpoint
        # ancestry. The nested initializer_fit_subset below records the smaller
        # direct update set used by this experiment.
        "fit_row_groups": parent["full_backbone_fit_row_groups"],
        "fit_membership_identity": parent["fit_membership_identity"],
        "initializer_fit_subset": {
            "manifest_path": contract["subset_manifest_path"],
            "manifest_sha256": contract["subset_manifest_sha256"],
            "selected_components": subset["selected_components"],
            "selected_groups": subset["selected_groups"],
            "selected_source_group_counts": subset["selected_source_group_counts"],
            "member_order_indices_by_epoch": orders,
        },
        "split_manifest_sha256": contract["split_sha256"],
        "normalization_identity": contract["normalization"],
        "backbone_ancestry": {
            "run": parent["run"],
            "checkpoint": parent["checkpoint_path"],
            "checkpoint_sha256": parent["checkpoint_sha256"],
            "provenance_sha256": parent["provenance_sha256"],
            "backbone_weights_frozen": True,
        },
        "target_ancestry": {
            "cache_root": contract["cache"]["root"],
            "cache_metadata_sha256": contract["cache"]["metadata_sha256"],
            "verified_target_files": contract["cache"]["target_files"],
            "source_coefficients": contract["cache"]["source_coefficients"],
            "teacher_sources": contract["cache"]["teacher_sources"],
            "distill_alpha": 0.9,
            "target_formula": "0.1*clamp(label,-2,2)+0.9*clamp(0.8125*mean(T3),-2,2)",
        },
        "propensity_ancestry": contract["propensity"],
        "training": {
            "objective": "weighted squared error over rows 99..19999, normalized by total sequence weight mass",
            "sequence_updates": len(subset["selected_members"]),
            "updates_attempted": len(updates),
            "updates_completed": len(updates),
            "finite_losses_and_gradients": True,
            "updates": updates,
            "same_epoch_order_across_modes": True,
            "evaluation_during_fit": False,
            "backbone_updates": 0,
        },
        "checkpoint_sha256": {},
        "exposure": "repeatedly exposed F0 fold-0 evaluation panel; not a fresh holdout",
    }


def tensor_state_cpu(model: Any) -> dict[str, Any]:
    """Copy a model state dictionary to CPU tensors for durable checkpointing."""
    return {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in model.state_dict().items()
    }


def export_valid_predictions(
    contract: dict[str, Any],
    mode: str,
    run_root: Path,
    model: Any,
    trainer: Any,
    torch: Any,
    np: Any,
    device: Any,
) -> dict[str, Any]:
    """Stream all 1,873 F0 valid sequences to the official FP32 prediction format."""
    root = Path(contract["data_root"]).resolve()
    sidecar, _ = bounded_json(
        root / contract["parent_prediction_sidecar_path"],
        MAX_JSON_BYTES,
        "parent full-valid prediction sidecar",
    )
    row_groups = list(map(int, sidecar["row_groups"]))
    dataset = trainer.Sequences("valid", row_groups)
    prediction_path = root / "cache" / f"pred_{contract['runs'][mode]}_valid.npy"
    sidecar_path = prediction_path.with_suffix(".json")
    if prediction_path.exists() or sidecar_path.exists():
        raise FileExistsError("candidate prediction paths already exist; preserve old attempts")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = np.lib.format.open_memmap(
        prediction_path, mode="w+", dtype=np.float32, shape=(len(row_groups), ROWS, 2)
    )
    trainer.HEURISTICS = False
    trainer.RAW_ONLY = False
    trainer.ONLINE_NORM = False
    trainer.RESEARCH_FEATURES = "none"
    trainer.PRIVILEGED_MODE = "none"
    trainer.INPUT_PROPENSITY = None
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        for index in range(len(row_groups)):
            x_cpu = dataset[index][0].unsqueeze(0)
            x = x_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
            inputs = trainer.build_inputs(x)
            state = None
            pieces = []
            for start in range(0, ROWS, CHUNK_ROWS):
                output, state = model(inputs[:, start : start + CHUNK_ROWS], state)
                pieces.append(output)
            prediction = torch.cat(pieces, dim=1)[0, :, :2].to("cpu").numpy()
            if prediction.shape != (ROWS, 2) or not np.isfinite(prediction).all():
                raise FloatingPointError(
                    f"nonfinite or malformed prediction at row group {row_groups[index]}"
                )
            predictions[index] = prediction.astype(np.float32, copy=False)
    predictions.flush()
    del predictions
    torch.cuda.synchronize(device)
    prediction_sha = sha256_file(prediction_path)
    checkpoint_path = run_root / "model_ep1.pt"
    sidecar_value = {
        "run": contract["runs"][mode],
        "split": "valid",
        "epoch": 1,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "prediction_sha256": prediction_sha,
        "split_manifest_sha256": contract["split_sha256"],
        "row_groups": row_groups,
        "inference_precision": "fp32",
        "state_initializer_mode": mode,
        "parent_checkpoint_sha256": contract["parent"]["checkpoint_sha256"],
    }
    write_new_json(sidecar_path, sidecar_value)
    return {
        "path": str(prediction_path),
        "sidecar_path": str(sidecar_path),
        "sha256": prediction_sha,
        "bytes": prediction_path.stat().st_size,
        "shape": [len(row_groups), ROWS, 2],
        "dtype": "float32",
        "row_groups_sha256": digest_ints(row_groups),
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_fit_export(
    contract: dict[str, Any],
    subset: dict[str, Any],
    trainer: Any,
    helper: Any,
    sequence_loss: Any,
    InitializedGRU: Any,
    modules: tuple[Any, Any, Any],
) -> dict[str, Any]:
    """Fit all three learned arms for two complete-sequence epochs and export."""
    torch, np, export_native_initializer = modules
    from wnn.propensity import load_propensity
    from wnn.resources import configure_torch
    from wnn.splits import load_campaign_split

    root = Path(contract["data_root"]).resolve()
    configure_torch()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", torch.cuda.current_device())
    resources = require_runtime_envelope(torch, helper, device)
    split = load_campaign_split(root / contract["split_path"], root)
    if split.sha256 != contract["split_sha256"]:
        raise ValueError("runtime campaign split differs from the frozen preflight")
    trainer.load_component_fit_subset(
        Path(contract["source_root"]) / contract["subset_manifest_path"],
        split,
        expected_experiment_id=contract["experiment_id"],
    )
    _, datasets, positions, _ = load_fit_data(trainer, contract, subset, split)
    parent = torch.load(
        root / contract["parent"]["checkpoint_path"], map_location="cpu", weights_only=False
    )
    if parent.get("epoch") != 5:
        raise ValueError("fixed clean T3 parent checkpoint is not CP5")
    backbone_state = {
        name: value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        for name, value in parent["model"].items()
    }
    if (
        canonical_tensor_sha(backbone_state["mean"], torch)
        != contract["normalization"]["mean_sha256"]
        or canonical_tensor_sha(backbone_state["std"], torch)
        != contract["normalization"]["std_sha256"]
    ):
        raise ValueError("CP5 normalization does not match the shared 211-feature buffers")
    del parent
    propensity_model = load_propensity(root / contract["propensity"]["checkpoint"], device)
    member_rows = subset["selected_members"]
    sampler = trainer.EpochOrderSampler(len(member_rows), seed=3, batch_size=1)
    epoch_orders: list[list[int]] = []
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        order = list(iter(sampler))
        if len(order) != len(member_rows) or set(order) != set(range(len(member_rows))):
            raise RuntimeError("EpochOrderSampler did not cover the full component subset")
        epoch_orders.append(order)

    mode_results = []
    for mode in RUN_MODES:
        run = contract["runs"][mode]
        run_root = root / "artifacts" / "runs" / run
        result_path = root / contract["result_paths"][mode]
        if run_root.exists() or (root / "cache" / f"pred_{run}_valid.npy").exists():
            raise FileExistsError(f"candidate run already exists: {run}")
        run_root.mkdir(parents=True)
        model = make_initializer_model(
            trainer, InitializedGRU, backbone_state, mode, torch, device, 3
        )
        optimizer = make_optimizer(model, torch)
        backbone_before = {
            name: value.detach().clone() for name, value in model.backbone.state_dict().items()
        }
        updates: list[dict[str, float]] = []
        epoch_summaries = []
        started_mode = time.perf_counter()
        for epoch, order in enumerate(epoch_orders):
            losses = []
            for position in order:
                member = member_rows[position]
                batch = batch_for_member(member, datasets, positions)
                _, inputs, targets, weights, supervision = initialize_batch(
                    trainer, helper, batch, propensity_model, torch, device
                )
                update = one_loss_update(
                    model, optimizer, inputs, targets, weights, sequence_loss, torch
                )
                update["epoch"] = epoch
                update["sampler_position"] = len(losses)
                update["member_source"] = member["source"]
                update["member_row_group"] = member["row_group"]
                update["member_seq_ix"] = member["seq_ix"]
                update["weight_mass"] = supervision["total_weight_mass"]
                updates.append(update)
                losses.append(update["loss"])
                del batch, inputs, targets, weights, supervision
            if len(losses) != len(member_rows):
                raise RuntimeError("an initializer epoch did not complete its fixed update budget")
            epoch_summaries.append(
                {
                    "epoch_index_zero_based": epoch,
                    "updates_completed": len(losses),
                    "mean_complete_sequence_loss": sum(losses) / len(losses),
                    "finite_losses": all(math.isfinite(value) for value in losses),
                }
            )
            if not epoch_summaries[-1]["finite_losses"]:
                raise FloatingPointError("nonfinite loss observed in the initializer epoch")
        if len(updates) != UPDATES_PER_MODE:
            raise RuntimeError("initializer update count differs from the frozen 2×104 budget")
        if any(not torch.isfinite(value).all().item() for value in model.state_dict().values()):
            raise FloatingPointError("initializer or frozen backbone contains nonfinite tensors")
        backbone_unchanged = all(
            torch.equal(backbone_before[name], value.detach())
            for name, value in model.backbone.state_dict().items()
        )
        if not backbone_unchanged:
            raise RuntimeError("initializer training changed a frozen CP5 backbone tensor")

        run_args = {
            "name": run,
            "arch": "gru",
            "cell": "gru",
            "enc": "linear",
            "hidden": 256,
            "layers": 2,
            "dropout": 0.0,
            "readout_activation": "gelu",
            "precision": "fp32",
            "seed": 3,
            "epochs": EPOCHS,
            "checkpoint_policy": "fixed",
            "checkpoint_epoch": 1,
            "no_eval_during_fit": True,
            "split_manifest_sha256": contract["split_sha256"],
            "valid_fold": 0,
            "state_initializer_mode": mode,
            "initialization_parent": f"{contract['parent']['run']}@ep5",
            "optimizer": "AdamW",
            "lr": 1e-3,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "sequence_chunk_rows": CHUNK_ROWS,
            "fit_sequences_per_epoch": len(member_rows),
            "auxiliary_losses": "none; main frozen-T3 objective only",
        }
        checkpoint_path = run_root / "model_ep1.pt"
        checkpoint = {
            "format": "wnn.initializer-family-f0-checkpoint",
            "version": 1,
            "run": run,
            "epoch": 1,
            "args": run_args,
            "n_in": N_FEATURES,
            "model": tensor_state_cpu(model.backbone),
            "initializer_mode": mode,
            "initializer_state_dict": {
                name.removeprefix("backbone."): value.detach().cpu().contiguous().clone()
                for name, value in model.state_dict().items()
                if not name.startswith("backbone.")
            },
            "initialized_model_state_dict": tensor_state_cpu(model),
            "state_initializer": export_native_initializer(model),
            "normalization_sha256": contract["normalization"],
            "parent_checkpoint_sha256": contract["parent"]["checkpoint_sha256"],
        }
        provenance = make_provenance(contract, mode, subset, epoch_orders, updates)
        checkpoint["provenance"] = {
            key: value for key, value in provenance.items() if key != "checkpoint_sha256"
        }
        torch.save(checkpoint, checkpoint_path)
        checkpoint_sha = sha256_file(checkpoint_path)
        provenance["checkpoint_sha256"] = {"model_ep1.pt": checkpoint_sha}
        write_new_json(run_root / "provenance.json", provenance)

        reloaded_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        reloaded_backbone = trainer.make_model(
            {
                "arch": "gru",
                "cell": "gru",
                "enc": "linear",
                "hidden": 256,
                "layers": 2,
                "dropout": 0.0,
                "readout_activation": "gelu",
                "numeric_map": "none",
                "ple_intervals": 8,
            },
            N_FEATURES,
            2,
        )
        reloaded_backbone.load_state_dict(reloaded_checkpoint["model"], strict=True)
        reloaded = InitializedGRU(reloaded_backbone, mode, residual_bound=1.0)
        reloaded.load_state_dict(reloaded_checkpoint["initialized_model_state_dict"], strict=True)
        reloaded.to(device).eval()
        model.eval()
        first_member = member_rows[epoch_orders[0][0]]
        first_batch = batch_for_member(first_member, datasets, positions)
        _, first_inputs, _, _, _ = initialize_batch(
            trainer, helper, first_batch, propensity_model, torch, device
        )
        with torch.no_grad():
            actual_state = None
            reloaded_state = None
            actual_parts = []
            reloaded_parts = []
            for start in range(0, ROWS, CHUNK_ROWS):
                actual_part, actual_state = model(
                    first_inputs[:, start : start + CHUNK_ROWS], actual_state
                )
                reloaded_part, reloaded_state = reloaded(
                    first_inputs[:, start : start + CHUNK_ROWS], reloaded_state
                )
                actual_parts.append(actual_part)
                reloaded_parts.append(reloaded_part)
            reload_delta = float(
                (torch.cat(actual_parts) - torch.cat(reloaded_parts)).abs().max().cpu()
            )
        if reload_delta != 0:
            raise RuntimeError("checkpoint reload changed initialized full-sequence output")
        del (
            checkpoint,
            reloaded_checkpoint,
            reloaded_backbone,
            reloaded,
            actual_parts,
            reloaded_parts,
        )

        predictions = export_valid_predictions(
            contract, mode, run_root, model, trainer, torch, np, device
        )
        elapsed_mode = time.perf_counter() - started_mode
        mode_result = {
            "run": run,
            "mode": mode,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "provenance_path": str(run_root / "provenance.json"),
            "provenance_sha256": sha256_file(run_root / "provenance.json"),
            "checkpoint_reload_max_abs_difference": reload_delta,
            "epoch_summaries": epoch_summaries,
            "updates_attempted": len(updates),
            "updates_completed": len(updates),
            "backbone_unchanged": backbone_unchanged,
            "predictions": predictions,
            "mode_elapsed_seconds": elapsed_mode,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "observed_vram_bytes_after": helper.nvidia_smi_used_bytes(),
        }
        if mode_result["cuda_peak_reserved_bytes"] > CUDA_RESERVED_LIMIT_BYTES:
            raise RuntimeError("initializer fit exceeded its 3.5 GB CUDA allocator cap")
        if mode_result["observed_vram_bytes_after"] >= OBSERVED_VRAM_LIMIT_BYTES:
            raise RuntimeError("initializer fit reached the 5 GB observed VRAM ceiling")
        write_new_json(result_path, mode_result)
        mode_results.append(mode_result)
        del optimizer, model
        torch.cuda.empty_cache()

    return {
        "format": RESULT_FORMAT,
        "version": 1,
        "contract_sha256": hashlib.sha256(
            (Path(contract["source_root"]) / contract["contract_path"]).read_bytes()
        ).hexdigest(),
        "split_sha256": contract["split_sha256"],
        "subset_manifest_sha256": contract["subset_manifest_sha256"],
        "parent": contract["parent"],
        "target_cache": contract["cache"],
        "selected_component_count": len(subset["selected_components"]),
        "selected_source_group_counts": subset["selected_source_group_counts"],
        "selected_total_sequences": len(member_rows),
        "fixed_evaluation_groups": len(subset["evaluation_groups"]),
        "fixed_evaluation_components": 141,
        "resources_at_start": resources,
        "modes": mode_results,
        "quality_claim": None,
        "interpretation": "F0 exposed-panel initializer screen; no independent seed/fold or serving claim",
    }


def parse_args() -> argparse.Namespace:
    """Parse the explicit recipe phase and paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "profile", "fit-export"), required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "research/campaign_2026-10-15/initializer_family_f0_r1/recipes/CONTRACT_R1.json",
    )
    parser.add_argument("--receipt", type=Path)
    parser.add_argument(
        "--profile-result-sha256",
        help="required hash pin for the completed resource profile before fit-export",
    )
    return parser.parse_args()


def main() -> None:
    """Run the chosen immutable phase and preserve a unique terminal receipt."""
    args = parse_args()
    script = Path(__file__).resolve()
    contract_path = args.contract.resolve(strict=True)
    contract, _ = bounded_json(contract_path, MAX_CONTRACT_BYTES, "recipe contract")
    preflight_path = args.receipt
    if args.phase == "preflight":
        report = static_preflight(script, contract_path, preflight_path)
        print(json.dumps({"phase": args.phase, "preflight": report}, sort_keys=True))
        return

    static_preflight(script, contract_path, None)
    if args.phase == "profile":
        fixed_profile_output = Path(contract["profile_result_path"]).resolve()
        if args.receipt is not None and args.receipt.resolve() != fixed_profile_output:
            raise ValueError("profile output is fixed by the contract; --receipt differs")
        require_absent_outputs(contract, "profile")
    else:
        fixed_fit_output = Path(contract["fit_result_path"]).resolve()
        if args.receipt is not None and args.receipt.resolve() != fixed_fit_output:
            raise ValueError("fit output is fixed by the contract; --receipt differs")
        if not args.profile_result_sha256:
            raise ValueError(
                "fit-export requires --profile-result-sha256 from the completed profile"
            )
        validate_profile_receipt(contract, args.profile_result_sha256)
        require_absent_outputs(contract, "fit-export")
    source_root = script.resolve().parents[1]
    subset, subset_raw = bounded_json(
        source_root / contract["subset_manifest_path"], MAX_SUBSET_BYTES, "component subset"
    )
    if hashlib.sha256(subset_raw).hexdigest() != contract["subset_manifest_sha256"]:
        raise ValueError("fit subset differs from the frozen contract")
    trainer, helper, sequence_loss, initialized_gru, modules = load_runtime_modules(contract)
    if args.phase == "profile":
        report = run_profile(
            contract, subset, trainer, helper, sequence_loss, initialized_gru, modules
        )
        output = Path(contract["profile_result_path"])
    else:
        report = run_fit_export(
            contract, subset, trainer, helper, sequence_loss, initialized_gru, modules
        )
        output = Path(contract["fit_result_path"])
    write_new_json(output, report)
    print(json.dumps({"phase": args.phase, "receipt": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
