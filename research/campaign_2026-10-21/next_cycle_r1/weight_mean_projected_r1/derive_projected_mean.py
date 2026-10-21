from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PLAN = Path(__file__).with_name("plan.json")
SCRATCH_PARENT = Path("/home/vlad/development/wunder/research_tmp")
OUTPUT_NAME = "overnight_nll_f0_s3_wmse_projected_mean_r1"
EXPECTED_INITIALIZER_SHA256 = "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e"

# Checkpoint payloads intentionally preserve the trainer's heterogeneous schema.
Checkpoint = dict[str, Any]


class Dependencies(NamedTuple):
    """Source-pinned helpers required for the one-time derivation."""

    mean_state: Callable[..., dict[str, torch.Tensor]]
    export_state: Callable[..., dict[str, np.ndarray]]
    make_model: Callable[..., torch.nn.Module]
    input_metadata: Callable[..., dict[str, object]]


def sha256(path: Path) -> str:
    """Hash a file without retaining its contents."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Read one small JSON metadata file and require an object root."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def verify_pins(plan: dict[str, Any], project_root: Path, training_root: Path) -> dict[str, str]:
    """Verify trainer, serving, and checkpoint-sidecar source identities."""
    checked: dict[str, str] = {}
    roots = {"training": training_root, "serving": CODE_ROOT}
    for group, root in roots.items():
        for relative, expected in plan["source_pins"][group].items():
            path = root / relative
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f"source pin mismatch for {path}: {actual}")
            checked[f"{group}:{relative}"] = actual
    for item in plan["checkpoints"]:
        provenance_path = (project_root / item["provenance_path"]).resolve(strict=True)
        if not provenance_path.is_relative_to(project_root):
            raise ValueError("provenance sidecar escapes the project root")
        actual = sha256(provenance_path)
        if actual != item["provenance_sha256"]:
            raise ValueError(f"provenance pin mismatch for {provenance_path}: {actual}")
        checked[f"sidecar:{item['role']}"] = actual
    return checked


def load_dependencies(training_root: Path) -> Dependencies:
    """Import the exact frozen trainer helpers and current native layout validator."""
    sys.path.insert(0, str(training_root))
    sys.path.insert(0, str(training_root / "src"))
    from scripts.average_trajectory_checkpoints import mean_state
    from scripts.export_native_weights import state_dict as export_state
    from wnn.training.models import make_model

    metadata_path = CODE_ROOT / "src/wnn/native_representation.py"
    spec = importlib.util.spec_from_file_location(
        "projected_mean_native_representation", metadata_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load native input metadata helper from {metadata_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    input_metadata = module.native_input_metadata
    return Dependencies(mean_state, export_state, make_model, input_metadata)


def _require_mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} is not a mapping")
    return value


def _projected_export_state(
    checkpoint: Checkpoint,
    export_state: dict[str, np.ndarray],
    item: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Cross-check the canonical native exporter and return its two-output state."""
    source_state = _require_mapping(checkpoint.get("model"), "checkpoint model")
    if any(name.startswith("dynamics_aux_head.") for name in source_state):
        raise ValueError("pinned sources unexpectedly contain a dynamics auxiliary head")
    if "dynamics_target_mean" in source_state or "dynamics_target_std" in source_state:
        raise ValueError("pinned sources unexpectedly contain dynamics normalization buffers")

    manual = {
        str(name): np.ascontiguousarray(value.detach().cpu().float().numpy())
        for name, value in source_state.items()
    }
    if manual["head.2.weight"].shape[0] != item["output_rows_expected"]:
        raise ValueError(f"{item['role']}: source head output rows differ from the frozen receipt")
    if manual["head.2.bias"].shape[0] != item["output_rows_expected"]:
        raise ValueError(f"{item['role']}: source head bias rows differ from the frozen receipt")
    manual["head.2.weight"] = np.ascontiguousarray(manual["head.2.weight"][:2])
    manual["head.2.bias"] = np.ascontiguousarray(manual["head.2.bias"][:2])
    if set(manual) != set(export_state):
        raise ValueError(f"{item['role']}: canonical exporter changed tensor keys unexpectedly")
    for name in manual:
        if manual[name].dtype != export_state[name].dtype or not np.array_equal(
            manual[name], export_state[name]
        ):
            raise ValueError(f"{item['role']}: exporter projection mismatch at {name}")
    if any(not np.isfinite(value).all() for value in export_state.values()):
        raise ValueError(f"{item['role']}: projected state contains nonfinite tensors")
    return {
        name: torch.from_numpy(np.array(value, dtype=np.float32, copy=True))
        for name, value in export_state.items()
    }


def _check_args_and_provenance(
    checkpoint: Checkpoint, sidecar: dict[str, Any], item: dict[str, Any], plan: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind layout, exact fold membership, normalization, and parent ancestry."""
    args = dict(_require_mapping(checkpoint.get("args"), "checkpoint args"))
    provenance = dict(_require_mapping(checkpoint.get("provenance"), "checkpoint provenance"))
    side_args = dict(_require_mapping(sidecar.get("args"), "sidecar args"))
    side_provenance = sidecar
    layout_args = plan["common_fitting_identity"]["layout_args"]
    for field, expected in layout_args.items():
        if field == "research_feature_control":
            if args.get(field) not in expected or side_args.get(field) not in expected:
                raise ValueError(f"{item['role']}: {field} is not a permitted no-op layout value")
        elif args.get(field) != expected or side_args.get(field) != expected:
            raise ValueError(f"{item['role']}: model input/training argument differs: {field}")
    if checkpoint.get("n_in") != plan["common_fitting_identity"]["n_in"]:
        raise ValueError(f"{item['role']}: declared model input width is not 211")
    if checkpoint.get("epoch") != item["epoch_expected"]:
        raise ValueError(f"{item['role']}: checkpoint epoch differs from the source pin")
    if args.get("init") != item["args_init_expected"]:
        raise ValueError(f"{item['role']}: declared initializer differs from the source pin")
    if provenance.get("initial_checkpoint_sha256") != item["initial_checkpoint_sha256"]:
        raise ValueError(f"{item['role']}: checkpoint initializer digest differs")
    if side_provenance.get("initial_checkpoint_sha256") != item["initial_checkpoint_sha256"]:
        raise ValueError(f"{item['role']}: sidecar initializer digest differs")
    if side_provenance.get("checkpoint_sha256", {}).get(Path(item["path"]).name) != item["sha256"]:
        raise ValueError(f"{item['role']}: embedded checkpoint digest differs")
    fields = (
        "split_manifest_sha256",
        "fit_membership_identity",
        "normalization_identity",
        "propensity_checkpoint_sha256",
        "distillation_cache",
    )
    expected_identity = plan["common_fitting_identity"]
    for field in fields:
        value = provenance.get(field)
        if value is None or value != side_provenance.get(field):
            raise ValueError(f"{item['role']}: embedded/sidecar lineage differs for {field}")
        expected = expected_identity[field]
        if field == "distillation_cache":
            expected_cache = expected
            cache = value
            for cache_field, expected_value in expected_cache.items():
                if cache.get(cache_field) != expected_value:
                    raise ValueError(
                        f"{item['role']}: distillation cache identity differs: {cache_field}"
                    )
        elif value != expected:
            raise ValueError(f"{item['role']}: common fitting identity differs for {field}")
    if item["role"] == "parent_cp5":
        if provenance.get("initial_checkpoint_sha256") != EXPECTED_INITIALIZER_SHA256:
            raise ValueError("parent initializer digest differs from the frozen lineage receipt")
    else:
        for field in ("split_manifest_sha256", "fit_membership_identity", "normalization_identity"):
            if provenance.get(field) != expected_identity[field]:
                raise ValueError(f"{item['role']}: child does not share parent {field}")
    return args, provenance


def _load_source(
    project_root: Path,
    item: dict[str, Any],
    plan: dict[str, Any],
    dependencies: Dependencies,
) -> tuple[Checkpoint, dict[str, Any], dict[str, torch.Tensor], dict[str, object]]:
    """Load one hash-pinned checkpoint, prove its serving projection, and retain it."""
    checkpoint_path = (project_root / item["path"]).resolve(strict=True)
    if not checkpoint_path.is_relative_to(project_root):
        raise ValueError(f"{item['role']}: checkpoint path escapes the project")
    if sha256(checkpoint_path) != item["sha256"]:
        raise ValueError(f"{item['role']}: checkpoint digest differs from the frozen pin")
    provenance_path = (project_root / item["provenance_path"]).resolve(strict=True)
    sidecar = load_json(provenance_path)
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    args, _ = _check_args_and_provenance(checkpoint, sidecar, item, plan)
    run_reference = f"{item['run']}@ep{item['epoch']}"
    native_state = dependencies.export_state(run_reference, project_root / "artifacts/runs")
    projected = _projected_export_state(checkpoint, native_state, item)
    layout = dependencies.input_metadata(args, native_state, checkpoint.get("n_in"))
    if layout != {
        "schema": "wnn.native_input_layout.v1",
        "representation_mode": "none",
        "input_width": 211,
        "layout": ["raw112", "event99"],
    }:
        raise ValueError(f"{item['role']}: unsupported native input layout {layout}")
    if (
        sha256(checkpoint_path) != item["sha256"]
        or sha256(provenance_path) != item["provenance_sha256"]
    ):
        raise ValueError(f"{item['role']}: source bytes changed while loading")
    return checkpoint, sidecar, projected, layout


def _projection_parity(
    checkpoint: Checkpoint,
    projected: dict[str, torch.Tensor],
    make_model: Callable[..., torch.nn.Module],
    *,
    seed: int,
    rows: int,
    chunk_rows: int,
    tolerance: float,
) -> dict[str, object]:
    """Compare full-head and served two-row models on fixed normalized CPU inputs."""
    args = dict(_require_mapping(checkpoint.get("args"), "checkpoint args"))
    source_state = _require_mapping(checkpoint.get("model"), "checkpoint model")
    output_rows = int(source_state["head.2.weight"].shape[0])
    full = make_model(args, int(checkpoint["n_in"]), output_rows).cpu().eval()
    full.load_state_dict(source_state, strict=True)
    stripped = make_model(args, int(checkpoint["n_in"]), 2).cpu().eval()
    stripped.load_state_dict(projected, strict=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    normalized = torch.randn((2, rows, int(checkpoint["n_in"])), generator=generator)
    mean = source_state["mean"].reshape(1, 1, -1)
    std = source_state["std"].reshape(1, 1, -1)
    raw = mean + normalized * std
    maximum = 0.0
    compared = 0
    with torch.inference_mode():
        for sequence in raw:
            full_state = None
            projected_state = None
            for start in range(0, rows, chunk_rows):
                part = sequence.unsqueeze(0)[:, start : start + chunk_rows]
                full_out, full_state = full(part, full_state)
                projected_out, projected_state = stripped(part, projected_state)
                delta = (full_out[..., :2] - projected_out).abs()
                if not bool(torch.isfinite(delta).all()):
                    raise ValueError("nonfinite scored outputs during projection parity")
                maximum = max(maximum, float(delta.max()))
                compared += int(delta.numel())
    if maximum > tolerance:
        raise ValueError(f"serving projection parity exceeds {tolerance}: {maximum}")
    return {
        "status": "passed",
        "max_abs_difference": maximum,
        "tolerance": tolerance,
        "scored_values_compared": compared,
        "sequences": 2,
        "rows_per_sequence": rows,
        "chunk_rows": chunk_rows,
        "sequence_state_reset": "state=None for each independent sequence",
        "precision": "CPU float32; autocast disabled",
    }


def derive(plan_path: Path, project_root: Path, scratch_root: Path) -> dict[str, object]:
    """Derive one new scratch checkpoint after all static and parity gates pass."""
    plan = load_json(plan_path)
    project_root = project_root.resolve(strict=True)
    scratch_root = scratch_root.resolve()
    required_parent = SCRATCH_PARENT.resolve(strict=True)
    if project_root != Path(plan["source_roots"]["project_root"]).resolve(strict=True):
        raise ValueError("project root differs from the frozen plan")
    if not scratch_root.is_relative_to(required_parent):
        raise ValueError(f"scratch output must be on the physical disk under {required_parent}")
    if plan["derivation"]["output_run"] != OUTPUT_NAME:
        raise ValueError("output run name differs from the frozen plan")
    training_root = Path(plan["source_roots"]["training_worktree"]).resolve(strict=True)
    code_hashes = verify_pins(plan, project_root, training_root)
    dependencies = load_dependencies(training_root)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    loaded = []
    layouts = []
    for item in plan["checkpoints"]:
        checkpoint, sidecar, projected, layout = _load_source(
            project_root, item, plan, dependencies
        )
        loaded.append((checkpoint, sidecar, projected))
        layouts.append(layout)
    if any(layout != layouts[0] for layout in layouts[1:]):
        raise ValueError("projected sources have different native input layouts")
    cache_identities = [entry[0]["provenance"]["distillation_cache"] for entry in loaded]
    if any(cache != cache_identities[0] for cache in cache_identities[1:]):
        raise ValueError("parent and WMSE child checkpoints do not share the exact teacher cache")
    states = [entry[2] for entry in loaded]
    parities = []
    for item, (checkpoint, _, projected) in zip(plan["checkpoints"], loaded, strict=True):
        parities.append(
            {
                "role": item["role"],
                **_projection_parity(
                    checkpoint,
                    projected,
                    dependencies.make_model,
                    seed=271828,
                    rows=4097,
                    chunk_rows=2000,
                    tolerance=1e-5,
                ),
            }
        )

    averaged_state = dependencies.mean_state(states)
    if averaged_state["mean"].shape != (211,) or averaged_state["std"].shape != (211,):
        raise ValueError("averaged normalization buffers have unsupported geometry")
    if not torch.equal(averaged_state["mean"], states[0]["mean"]):
        raise ValueError("mean normalization buffer changed during averaging")
    if not torch.equal(averaged_state["std"], states[0]["std"]):
        raise ValueError("std normalization buffer changed during averaging")
    if verify_pins(plan, project_root, training_root) != code_hashes:
        raise ValueError("source code or checkpoint provenance changed during derivation")
    for item in plan["checkpoints"]:
        source_checkpoint = (project_root / item["path"]).resolve(strict=True)
        if sha256(source_checkpoint) != item["sha256"]:
            raise ValueError(f"{item['role']}: source checkpoint changed before derivation write")
    output_root = scratch_root / "artifacts/runs" / OUTPUT_NAME
    receipt_path = scratch_root / "receipts/PROJECTED_MEAN_DERIVATION_R1.json"
    if output_root.exists() or receipt_path.exists():
        raise FileExistsError("create-only output checkpoint or receipt already exists")

    output_checkpoint = copy.deepcopy(loaded[-1][0])
    output_checkpoint["model"] = averaged_state
    output_checkpoint["args"] = dict(output_checkpoint["args"])
    output_checkpoint["args"]["name"] = OUTPUT_NAME
    derivation = {
        "method": "equal_parameter_mean_after_canonical_two_output_serving_projection",
        "sources": [
            {"role": item["role"], "path": item["path"], "sha256": item["sha256"]}
            for item in plan["checkpoints"]
        ],
        "source_weights": plan["derivation"]["weights"],
        "projection": "native exporter head.2 first-two-row projection; no new fit",
        "accumulation": "canonical mean_state float64 accumulation, cast back to source dtype",
        "quality_evaluated": False,
    }
    output_checkpoint["weight_derivation"] = derivation
    output_checkpoint["provenance"] = dict(output_checkpoint["provenance"])
    output_checkpoint["provenance"]["args"] = dict(output_checkpoint["provenance"]["args"])
    output_checkpoint["provenance"]["args"]["name"] = OUTPUT_NAME
    output_checkpoint["provenance"]["weight_derivation"] = derivation
    output_checkpoint["provenance"].pop("checkpoint_sha256", None)
    output_root.mkdir(parents=True)
    checkpoint_path = output_root / "model_ep1.pt"
    torch.save(output_checkpoint, checkpoint_path)
    checkpoint_digest = sha256(checkpoint_path)
    output_provenance = dict(loaded[-1][1])
    output_provenance["args"] = dict(output_provenance["args"])
    output_provenance["args"]["name"] = OUTPUT_NAME
    output_provenance["checkpoint_sha256"] = {checkpoint_path.name: checkpoint_digest}
    output_provenance["weight_derivation"] = derivation
    (output_root / "provenance.json").write_text(
        json.dumps(output_provenance, indent=2) + "\n", encoding="utf-8"
    )
    receipt = {
        "schema": "wnn.serving_projected_weight_mean_derivation.v1",
        "status": "derived_cpu_projection_parity_passed_quality_not_evaluated",
        "plan_sha256": sha256(plan_path),
        "derive_script_sha256": sha256(Path(__file__).resolve()),
        "source_code_sha256": code_hashes,
        "source_checkpoints": [
            {"role": item["role"], "sha256": item["sha256"]} for item in plan["checkpoints"]
        ],
        "source_projection_parity": parities,
        "native_input_layout": layouts[0],
        "normalization_bitwise_equal": True,
        "mean_method": derivation["method"],
        "output_checkpoint": str(checkpoint_path),
        "output_checkpoint_sha256": checkpoint_digest,
        "fit_count": 0,
        "official_quality_read": False,
        "limitations": [
            "F0 is an exposed diagnostic panel and does not estimate fit-seed or public-transfer uncertainty.",
            "This amendment preserves the historical full-state mean block; only its serving-projected descendant is derived.",
            "This receipt does not grade the checkpoint or establish a production candidate.",
        ],
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    """Run the bounded, create-only projected mean derivation after admission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(derive(args.plan, args.project_root, args.scratch_root), indent=2))


if __name__ == "__main__":
    main()
