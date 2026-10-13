"""Source validation and exact model loading for team-member FP32 diagnostics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


SEQUENCE_LENGTH = 20_000
WARMUP = 99
N_RAW_FEATURES = 112
N_MODEL_FEATURES = 211
N_OUTPUTS = 2
SUPPORTED_CANDIDATES = frozenset({"pair_ftt2", "xteam_h4", "e6a", "E"})


@dataclass(frozen=True)
class MemberArchitecture:
    """Dimensions inferred from one archived recurrent member."""

    input_features: int
    hidden: int
    layers: int
    head_width: int
    outputs: int


def infer_architecture(tensor_shapes: Mapping[str, tuple[int, ...]]) -> MemberArchitecture:
    """Validate and infer the supported GRU member dimensions from NPZ shapes."""
    required = {
        "mean",
        "std",
        "inp.weight",
        "inp.bias",
        "head.0.weight",
        "head.0.bias",
        "head.2.weight",
        "head.2.bias",
    }
    missing = required - tensor_shapes.keys()
    if missing:
        raise ValueError(f"member is missing tensors: {sorted(missing)}")
    layer_indexes = set()
    for name in tensor_shapes:
        if name.startswith("gru."):
            if "_l" not in name:
                raise ValueError("unsupported GRU tensor name")
            suffix = name.rsplit("_l", 1)[1]
            if not suffix.isdigit():
                raise ValueError("unsupported GRU direction or projection")
            layer_indexes.add(int(suffix))
    layers = sorted(layer_indexes)
    if not layers or layers != list(range(len(layers))):
        raise ValueError("GRU layers must be contiguous from layer zero")
    input_shape = tensor_shapes["inp.weight"]
    if len(input_shape) != 2:
        raise ValueError("inp.weight must be a matrix")
    hidden, input_features = input_shape
    if input_features != N_MODEL_FEATURES or tensor_shapes["inp.bias"] != (hidden,):
        raise ValueError("member must use the 211-feature linear input projection")
    if tensor_shapes["mean"] != (input_features,) or tensor_shapes["std"] != (input_features,):
        raise ValueError("mean/std widths do not match the input projection")
    recurrent_names = {
        f"gru.{kind}_l{layer}"
        for layer in layers
        for kind in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
    }
    actual_recurrent = {name for name in tensor_shapes if name.startswith("gru.")}
    if actual_recurrent != recurrent_names:
        raise ValueError("member has unsupported recurrent tensors or GRU directions")
    for layer in layers:
        expected_shapes = {
            f"gru.weight_ih_l{layer}": (3 * hidden, hidden),
            f"gru.weight_hh_l{layer}": (3 * hidden, hidden),
            f"gru.bias_ih_l{layer}": (3 * hidden,),
            f"gru.bias_hh_l{layer}": (3 * hidden,),
        }
        for name, expected in expected_shapes.items():
            if tensor_shapes[name] != expected:
                raise ValueError(f"{name} has shape {tensor_shapes[name]}, expected {expected}")
    head_shape = tensor_shapes["head.0.weight"]
    output_shape = tensor_shapes["head.2.weight"]
    if len(head_shape) != 2 or len(output_shape) != 2:
        raise ValueError("head weights must be matrices")
    head_width, head_input = head_shape
    outputs, output_input = output_shape
    if head_input != 2 * hidden or tensor_shapes["head.0.bias"] != (head_width,):
        raise ValueError("head.0 must consume the GRU output and projection skip")
    if (
        outputs != N_OUTPUTS
        or output_input != head_width
        or tensor_shapes["head.2.bias"] != (outputs,)
    ):
        raise ValueError("member must have a two-output linear head")
    if head_width not in (hidden, 2 * hidden):
        raise ValueError("only canonical and merged-width team heads are supported")
    return MemberArchitecture(input_features, hidden, len(layers), head_width, outputs)


def select_row_groups(first: int, count: int | None, total: int) -> tuple[int, ...]:
    """Validate a contiguous, bounded group selection for full-sequence profiling."""
    if total < 1 or first < 0 or first >= total:
        raise ValueError("first row group is outside the dataset")
    selected = total - first if count is None else count
    if selected < 1 or first + selected > total:
        raise ValueError("row-group selection is empty or exceeds the dataset")
    return tuple(range(first, first + selected))


def iter_chunks(length: int, chunk_rows: int) -> tuple[tuple[int, int], ...]:
    """Return contiguous sequence slices for stateful recurrent inference."""
    if length < 1 or chunk_rows < 1 or chunk_rows > length:
        raise ValueError("chunk size and sequence length must be positive and bounded")
    return tuple((start, min(start + chunk_rows, length)) for start in range(0, length, chunk_rows))


def digest(path: Path) -> str:
    """Hash a file with bounded memory for exact source and dataset provenance."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def member_source(
    inventory_path: Path, candidate: str, member_name: str
) -> tuple[dict[str, object], Path, dict[str, object], dict[str, object]]:
    """Resolve a declared member and its expected inventory records."""
    inventory = json.loads(inventory_path.read_text())
    if candidate not in SUPPORTED_CANDIDATES:
        raise ValueError(f"unsupported team package for this exporter: {candidate}")
    package = inventory["packages"].get(candidate)
    if package is None:
        raise ValueError(f"unknown candidate in inventory: {candidate}")
    if Path(member_name).name != member_name or member_name.startswith("det_"):
        raise ValueError("member must be one declared predictor NPZ basename")
    file_record = package["files"].get(member_name)
    if file_record is None or not member_name.endswith(".npz"):
        raise ValueError(f"member is not listed in {candidate} inventory")
    member_record = next(
        (entry for entry in package.get("members", []) if entry.get("weights") == member_name),
        None,
    )
    if member_record is None:
        raise ValueError("NPZ is not an active predictor member in the declared package")
    package_path = Path(package["package"])
    members_path = package_path / "members.json"
    members_file_record = package["files"].get("members.json")
    if members_file_record is None or digest(members_path) != members_file_record["sha256"]:
        raise ValueError("package member map differs from the inventory")
    packaged_members = json.loads(members_path.read_text())
    packaged_member = next(
        (entry for entry in packaged_members if entry.get("weights") == member_name), None
    )
    if packaged_member != member_record:
        raise ValueError("package and inventory disagree about member ancestry or coefficient")
    weights_path = package_path / member_name
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    return inventory, weights_path, file_record, member_record


def load_fp32_model(
    weights_path: Path, architecture: MemberArchitecture, device: str
) -> torch.nn.Module:
    """Load one exact FP32 state into the project's GRU implementation."""
    import numpy as np
    import torch

    from wnn.nn import RNNModel

    with np.load(weights_path, allow_pickle=False) as archive:
        state_arrays = {name: archive[name] for name in archive.files}
    if any(array.dtype != np.float32 for array in state_arrays.values()):
        raise ValueError("team NPZ tensors must already be FP32")
    shapes = {name: tuple(array.shape) for name, array in state_arrays.items()}
    if infer_architecture(shapes) != architecture:
        raise ValueError("member tensor shapes changed after inventory validation")
    model = RNNModel(
        n_in=architecture.input_features,
        hidden=architecture.hidden,
        layers=architecture.layers,
        dropout=0.0,
        n_out=architecture.outputs,
        cell="gru",
        enc="linear",
    )
    if architecture.head_width != architecture.hidden:
        model.head[0] = torch.nn.Linear(2 * architecture.hidden, architecture.head_width)
        model.head[2] = torch.nn.Linear(architecture.head_width, architecture.outputs)
    state = {name: torch.from_numpy(array) for name, array in state_arrays.items()}
    model.load_state_dict(state, strict=True)
    del state, state_arrays
    return model.to(device=device, dtype=torch.float32).eval()


def native_difference(candidate: str, package: dict[str, object]) -> str:
    """Describe the original callback precision without claiming numerical parity."""
    native_file = Path(package["package"]) / "solution.py"
    native_record = package["files"].get("solution.py")
    if native_record is None or digest(native_file) != native_record["sha256"]:
        raise ValueError("native callback source differs from the inventory")
    if candidate in ("pair_ftt2", "e6a"):
        runtime = "dynamic_int8 weights, fast_activations=True"
    else:
        precision_path = Path(package["package"]) / "precision.txt"
        precision_record = package["files"].get("precision.txt")
        if precision_record is None or digest(precision_path) != precision_record["sha256"]:
            raise ValueError("native precision setting differs from the inventory")
        precision = precision_path.read_text().strip()
        runtime = f"precision={precision} (8-bit Q8VNNI), fast_activations=True"
    return (
        f"Per-member unweighted PyTorch FP32 output; original callback uses {runtime} with "
        "fixed ensemble coefficients. FP32 weights/activations, the omitted ensemble weighting, "
        "and GPU PyTorch GRU/GELU arithmetic mean outputs are not expected to be numerically "
        "identical to the quantized native callback. No FP32-versus-native numerical delta is "
        "measured by this exporter; this is a diagnostic endpoint, not native callback parity."
    )
