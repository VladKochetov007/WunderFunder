from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from wnn.evaluation import paired_component_moments
from wnn.gmetric import METRIC_ID, global_wp
from wnn.splits import load_campaign_split

# JSON receipts have heterogeneous values at the serialization boundary.


def sha256_file(path: Path) -> str:
    """Hash a file in bounded blocks without loading its payload into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, maximum_bytes: int = 2_000_000) -> dict[str, Any]:
    """Read a small frozen manifest only after checking its size."""
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"manifest size {size} exceeds the {maximum_bytes}-byte bound: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one durable JSON receipt by atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def source_inputs(
    root: Path, recipe: dict[str, Any]
) -> tuple[dict[str, np.ndarray], list[dict[str, int]]]:
    """Verify frozen prediction sources and align them to exposed F0 groups.

    This function does not open target or mask arrays.
    """
    if recipe["metric_id"] != METRIC_ID:
        raise ValueError(f"unexpected metric: {recipe['metric_id']}")
    code = recipe["code"]
    code_paths = {
        "grader_sha256": root / code["grader_path"],
        "source_verifier_sha256": Path(__file__).resolve(),
        "metric_sha256": root / "src/wnn/gmetric.py",
        "bootstrap_sha256": root / "src/wnn/evaluation.py",
        "split_loader_sha256": root / "src/wnn/splits.py",
    }
    for key, path in code_paths.items():
        if sha256_file(path) != code[key]:
            raise ValueError(f"frozen code hash changed: {key} ({path})")

    contract_path = root / recipe["contract"]["path"]
    if sha256_file(contract_path) != recipe["contract"]["sha256"]:
        raise ValueError("campaign contract changed after recipe freeze")
    split_path = root / recipe["split"]["path"]
    split = load_campaign_split(split_path, root)
    if split.sha256 != recipe["split"]["sha256"] or split.fold != 0:
        raise ValueError("fold0 source-qualified split differs from the frozen recipe")

    expected_groups = int(recipe["panel"]["groups"])
    expected_components = int(recipe["panel"]["components"])
    groups = tuple(map(int, split.evaluation_groups))
    if len(groups) != expected_groups or len(set(groups)) != expected_groups:
        raise ValueError(f"unexpected fold0 evaluation groups: {len(groups)}")
    component_by_group = {
        int(group): int(component) for side, group, component in split.components if side == "valid"
    }
    selected_components = {component_by_group[group] for group in groups}
    if len(selected_components) != expected_components:
        raise ValueError(f"unexpected fold0 component count: {len(selected_components)}")

    source_arrays: dict[str, np.ndarray] = {}
    group_records: list[dict[str, int]] = []
    cache_members: dict[str, dict[int, tuple[int, int]]] = {}
    sidecars: dict[str, dict[str, Any]] = {}
    for name, source in recipe["experts"].items():
        prediction_path = root / source["prediction_path"]
        sidecar_path = root / source["sidecar_path"]
        if sha256_file(prediction_path) != source["prediction_sha256"]:
            raise ValueError(f"{name} prediction cache hash mismatch")
        if sha256_file(sidecar_path) != source["sidecar_sha256"]:
            raise ValueError(f"{name} prediction sidecar hash mismatch")
        sidecar = read_json(sidecar_path)
        if sidecar["split_manifest_sha256"] != split.sha256:
            raise ValueError(f"{name} prediction sidecar uses a different split")
        if sidecar.get("prediction_sha256") != source["prediction_sha256"]:
            raise ValueError(f"{name} sidecar does not identify its frozen prediction cache")
        row_groups = list(map(int, sidecar["row_groups"]))
        sequence_ids = list(map(int, sidecar["sequence_ids"]))
        if len(row_groups) != expected_groups or len(sequence_ids) != expected_groups:
            raise ValueError(f"{name} sidecar has an unexpected group count")
        members = {group: (index, sequence_ids[index]) for index, group in enumerate(row_groups)}
        if len(members) != len(row_groups):
            raise ValueError(f"{name} sidecar has duplicate row groups")
        mapped = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (expected_groups, int(recipe["panel"]["steps"]), 2)
        if mapped.shape != expected_shape or mapped.dtype != np.float32:
            raise ValueError(f"{name} cache shape or dtype changed: {mapped.shape}/{mapped.dtype}")
        cache_members[name] = members
        sidecars[name] = sidecar
        source_arrays[name] = mapped

    primary_sidecar = sidecars["tl3"]
    sequence_by_group = {
        int(group): int(seq_ix)
        for group, seq_ix in zip(
            primary_sidecar["row_groups"], primary_sidecar["sequence_ids"], strict=True
        )
    }
    for group in groups:
        if group not in component_by_group or group not in sequence_by_group:
            raise ValueError(f"evaluation group {group} is missing from source metadata")
        seq_ix = sequence_by_group[group]
        record = {"row_group": group, "seq_ix": seq_ix, "component": component_by_group[group]}
        for name in recipe["experts"]:
            cache_index, cached_seq_ix = cache_members[name][group]
            if cached_seq_ix != seq_ix:
                raise ValueError(f"{name} sequence identity mismatch for row group {group}")
            record[f"{name}_index"] = cache_index
        group_records.append(record)
    return source_arrays, group_records


def one_target_wp(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    """Compute official clipped WP for one target using the canonical scorer."""
    y = np.asarray(target, dtype=np.float32)
    p = np.asarray(prediction, dtype=np.float32)
    selected = np.asarray(mask, dtype=bool)
    repeated_y = np.repeat(y[:, None], 2, axis=1)
    repeated_p = np.repeat(p[:, None], 2, axis=1)
    return float(global_wp(repeated_y, repeated_p, selected)["wp"])


def endpoint_moments(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return stable per-sequence moments for a paired component bootstrap."""
    candidate, _ = paired_component_moments(
        target[None, ...], prediction[None, ...], prediction[None, ...], mask[None, ...]
    )
    return candidate[0]
