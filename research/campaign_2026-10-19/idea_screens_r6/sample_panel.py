from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
from attrs import frozen

SAMPLE_SHA256 = "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
MEMBERSHIP_SHA256 = "e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217"
SPLIT_SHA256 = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
R6_DIR = Path(__file__).resolve().parent
R1_SAMPLE_DATA = R6_DIR.parent / "idea_screens_r1" / "sample_data.py"


@frozen
class FitSequence:
    """Hash-bound, read-only arrays for one selected fitting sequence."""

    record: dict[str, object]
    x: np.ndarray
    y: np.ndarray
    scored: np.ndarray


def _r1_helpers() -> ModuleType:
    """Load the pinned R1 streaming digest helper after source verification."""
    name = "nf19_r1_sample_data_for_r6"
    spec = importlib.util.spec_from_file_location(name, R1_SAMPLE_DATA)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the pinned R1 helper at {R1_SAMPLE_DATA}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _same_identity(member: dict[str, object], sample: dict[str, object]) -> None:
    """Check the immutable selection fields shared by membership and sample manifests."""
    fields = (
        "role",
        "source",
        "group",
        "row_group",
        "seq_ix",
        "component",
        "included_in_low32",
        "fit_low32_order",
        "eligible_groups_in_component",
        "selection_probabilities",
    )
    if any(member.get(field) != sample.get(field) for field in fields):
        raise ValueError("sample record differs from fixed membership")


def _load_selected_arrays(record: dict[str, object], paths: dict[str, Path]) -> FitSequence:
    """Map arrays after a caller has verified all selected NPY payloads."""
    arrays = {key: np.load(path, mmap_mode="r", allow_pickle=False) for key, path in paths.items()}
    expected = {
        "x": (np.dtype(np.float32), (20_000, 112)),
        "y": (np.dtype(np.float32), (20_000, 2)),
    }
    for key, (dtype, shape) in expected.items():
        if arrays[key].dtype != dtype or arrays[key].shape != shape:
            raise ValueError(
                f"invalid selected {key} header: {arrays[key].dtype} {arrays[key].shape}"
            )
    if arrays["scored"].dtype != np.dtype(np.bool_) or arrays["scored"].shape != (20_000,):
        raise ValueError("invalid selected scored mask NPY header")
    return FitSequence(record, arrays["x"], arrays["y"], arrays["scored"])


def select_eight_fit_records(
    sample: dict[str, object], membership: dict[str, object]
) -> list[dict[str, object]]:
    """Validate complete R3 role metadata and return its fixed first eight fit records."""
    if (
        sample.get("schema") != "idea_screens_r3.sample.v2"
        or sample.get("split_sha256") != SPLIT_SHA256
    ):
        raise ValueError("unexpected R3 sample schema or split identity")
    if membership.get("schema") != "idea_screens_r3.membership.v2":
        raise ValueError("unexpected R3 membership schema")
    if membership.get("split_sha256") != SPLIT_SHA256:
        raise ValueError("R3 membership split identity differs")
    selection = sample.get("selection")
    records, member_records = sample.get("records"), membership.get("records")
    if (
        not isinstance(selection, dict)
        or not isinstance(records, list)
        or not isinstance(member_records, list)
    ):
        raise ValueError("incomplete R3 panel metadata")
    if membership.get("selection") != selection or membership.get("metadata_counts") != sample.get(
        "metadata_counts"
    ):
        raise ValueError("sample and membership selection metadata disagree")
    if len(records) != 96 or len(member_records) != 96:
        raise ValueError("R3 panel must contain 64 fit and 32 evaluation records")
    for member, record in zip(member_records, records, strict=True):
        _same_identity(member, record)
    fit_order, eval_order = (
        selection.get("fit_full64_component_order"),
        selection.get("eval32_component_order"),
    )
    if (
        not isinstance(fit_order, list)
        or len(fit_order) != 64
        or not isinstance(eval_order, list)
        or len(eval_order) != 32
    ):
        raise ValueError("invalid fixed R3 component orders")
    fit_records, eval_records = records[:64], records[64:]
    fit_components = [int(record["component"]) for record in fit_records]
    eval_components = [int(record["component"]) for record in eval_records]
    if fit_components != [int(value) for value in fit_order] or eval_components != [
        int(value) for value in eval_order
    ]:
        raise ValueError("R3 records do not follow fixed component order")
    if set(fit_components) & set(eval_components):
        raise ValueError("fit and evaluation components overlap")
    if len(set(fit_components)) != 64 or len(set(eval_components)) != 32:
        raise ValueError("R3 component order contains duplicates")
    if any(
        record.get("role") != "fit" or record.get("source") != "train" for record in fit_records
    ):
        raise ValueError("R3 fitting records must be source-qualified train sequences")
    if any(
        record.get("role") != "eval" or record.get("source") != "valid" for record in eval_records
    ):
        raise ValueError("R3 evaluation metadata has unexpected role or source")
    return fit_records[:8]


def load_first_fit_eight(directory: Path) -> tuple[list[FitSequence], dict[str, object]]:
    """Verify fixed R3 metadata and map only the first eight full64 fit sequences.

    The sample and membership manifests bind the complete panel, but this function
    hashes and maps only the 24 NPYs consumed by the fit-only diagnostic.
    """
    helper = _r1_helpers()
    digest = helper.digest
    sample_path, membership_path = directory / "SAMPLE.json", directory / "MEMBERSHIP.json"
    if digest(sample_path) != SAMPLE_SHA256 or digest(membership_path) != MEMBERSHIP_SHA256:
        raise ValueError("accepted R3 sample or membership identity differs")
    sample = json.loads(sample_path.read_bytes())
    membership = json.loads(membership_path.read_bytes())
    selected = select_eight_fit_records(sample, membership)
    selection = sample["selection"]
    assert isinstance(selection, dict)
    verified_paths: list[dict[str, Path]] = []
    for record in selected:
        file_map = record.get("files")
        if not isinstance(file_map, dict) or set(file_map) != {"x", "y", "scored"}:
            raise ValueError("selected sequence must bind exactly x, y and scored arrays")
        paths: dict[str, Path] = {}
        for key in ("x", "y", "scored"):
            binding = file_map[key]
            if not isinstance(binding, dict):
                raise ValueError(f"invalid {key} NPY binding")
            name = binding.get("name")
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("sample NPY binding must be a basename")
            path = directory / name
            if path.stat().st_size != binding.get("bytes") or digest(path) != binding.get("sha256"):
                raise ValueError(f"sample NPY binding changed: {name}")
            paths[key] = path
        verified_paths.append(paths)
    sequences = [
        _load_selected_arrays(record, paths)
        for record, paths in zip(selected, verified_paths, strict=True)
    ]
    if [int(sequence.record["component"]) for sequence in sequences] != [
        int(v) for v in selection["fit_full64_component_order"][:8]
    ]:
        raise ValueError("selected records differ from the first eight fixed fit components")
    summary: dict[str, object] = {
        "sample_sha256": SAMPLE_SHA256,
        "membership_sha256": MEMBERSHIP_SHA256,
        "split_sha256": SPLIT_SHA256,
        "fit_component_order": [int(value) for value in selection["fit_full64_component_order"]],
        "selected_components": [int(record["component"]) for record in selected],
        "selected_sequences": [
            {key: record[key] for key in ("source", "group", "row_group", "seq_ix", "component")}
            for record in selected
        ],
        "eval_npy_files_read": 0,
        "selected_npy_files_hashed_and_mapped": 24,
    }
    return sequences, summary
