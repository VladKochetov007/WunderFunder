"""Validate the fixed R3 panel and read only its fitting inputs and masks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

SAMPLE_SHA256 = "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
MEMBERSHIP_SHA256 = "e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217"
SPLIT_SHA256 = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
IDENTITY_FIELDS = (
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
ARRAY_KEYS = ("x", "y", "scored")


class ValidatedPanel(NamedTuple):
    """Full metadata validation result with ordered fitting records only."""

    fit_component_order: tuple[int, ...]
    eval_component_order: tuple[int, ...]
    fit_records: tuple[dict[str, object], ...]


class InputBinding(NamedTuple):
    """One fit record's verified feature and scored-mask file bindings."""

    record: dict[str, object]
    x_path: Path
    x_sha256: str
    x_bytes: int
    scored_path: Path
    scored_sha256: str
    scored_bytes: int


class FitArrays(NamedTuple):
    """Read-only mapped features and original row mask for one fit sequence."""

    record: dict[str, object]
    x: np.ndarray
    scored: np.ndarray


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_metadata(sample_dir: Path, expected_sample_sha256: str) -> ValidatedPanel:
    """Verify exact accepted R3 manifests and all role/order/component relations."""
    if expected_sample_sha256 != SAMPLE_SHA256:
        raise ValueError("requested sample identity differs from the accepted R3 panel")
    sample_path, membership_path = sample_dir / "SAMPLE.json", sample_dir / "MEMBERSHIP.json"
    if (
        sha256_file(sample_path) != SAMPLE_SHA256
        or sha256_file(membership_path) != MEMBERSHIP_SHA256
    ):
        raise ValueError("accepted R3 sample or membership identity differs")
    sample = _read_json_object(sample_path)
    membership = _read_json_object(membership_path)
    return validate_metadata_objects(sample, membership)


def validate_metadata_objects(
    sample: dict[str, object], membership: dict[str, object]
) -> ValidatedPanel:
    """Validate pinned R3 v2 schema semantics without opening any array file."""
    if (
        sample.get("schema") != "idea_screens_r3.sample.v2"
        or sample.get("split_sha256") != SPLIT_SHA256
    ):
        raise ValueError("unexpected R3 sample schema or split identity")
    if (
        membership.get("schema") != "idea_screens_r3.membership.v2"
        or membership.get("split_sha256") != SPLIT_SHA256
    ):
        raise ValueError("unexpected R3 membership schema or split identity")
    if sample.get("membership_file") != "MEMBERSHIP.json":
        raise ValueError("R3 sample does not bind its expected membership filename")
    selection = _mapping(sample.get("selection"), "sample selection")
    member_selection = _mapping(membership.get("selection"), "membership selection")
    if member_selection != selection or membership.get("metadata_counts") != sample.get(
        "metadata_counts"
    ):
        raise ValueError("sample and membership selection metadata disagree")
    if selection.get("scheme") != "uniform_component_sample_then_uniform_one_group_per_component":
        raise ValueError("R3 selection scheme differs")
    if selection.get("seeds") != {
        "fit_components": 19021,
        "fit_groups": 19022,
        "eval_components": 19023,
        "eval_groups": 19024,
    }:
        raise ValueError("R3 selection seeds differ")
    if selection.get("selected_component_counts") != {
        "fit_full64": 64,
        "fit_nested_low32": 32,
        "eval32": 32,
    }:
        raise ValueError("R3 component selection counts differ")

    records = _record_list(sample.get("records"), "sample records")
    member_records = _record_list(membership.get("records"), "membership records")
    if len(records) != 96 or len(member_records) != 96:
        raise ValueError("R3 metadata must contain 64 fit and 32 evaluation records")
    for index, (record, member) in enumerate(zip(records, member_records, strict=True)):
        if any(record.get(field) != member.get(field) for field in IDENTITY_FIELDS):
            raise ValueError(f"sample and membership record {index} identities disagree")
        _validate_record_files(record, index)

    fit_order = _integer_tuple(selection.get("fit_full64_component_order"), 64, "fit order")
    eval_order = _integer_tuple(selection.get("eval32_component_order"), 32, "eval order")
    low32_order = _integer_tuple(selection.get("fit_low32_components"), 32, "low32 order")
    fit_records, eval_records = records[:64], records[64:]
    fit_components = tuple(_record_int(record, "component") for record in fit_records)
    eval_components = tuple(_record_int(record, "component") for record in eval_records)
    if fit_components != fit_order or eval_components != eval_order:
        raise ValueError("R3 records do not follow the fixed component orders")
    if low32_order != fit_order[:32] or len(set(fit_order)) != 64 or len(set(eval_order)) != 32:
        raise ValueError("R3 fit order or nested low32 order is invalid")
    if set(fit_order) & set(eval_order):
        raise ValueError("R3 fit and evaluation components overlap")
    for index, record in enumerate(fit_records):
        if record.get("role") != "fit" or record.get("source") != "train":
            raise ValueError("R3 fit records must be source-qualified train sequences")
        expected_low32 = index < 32
        if record.get("included_in_low32") is not expected_low32:
            raise ValueError("R3 nested low32 membership differs from fixed component order")
        expected_low32_index = index if expected_low32 else None
        if record.get("fit_low32_order") != expected_low32_index:
            raise ValueError("R3 nested low32 order differs from fixed component order")
    for record in eval_records:
        if record.get("role") != "eval" or record.get("source") != "valid":
            raise ValueError("R3 evaluation records must be source-qualified valid sequences")
    return ValidatedPanel(fit_order, eval_order, tuple(fit_records))


def verify_fit_inputs(sample_dir: Path, panel: ValidatedPanel) -> list[InputBinding]:
    """Hash the 64 fitting x/mask NPYs, without opening target or eval files."""
    verified: list[InputBinding] = []
    root = sample_dir.resolve()
    for record in panel.fit_records:
        files = _mapping(record.get("files"), "fit record files")
        x_path, x_sha, x_bytes = _verify_binding(root, _mapping(files.get("x"), "x binding"))
        mask_path, mask_sha, mask_bytes = _verify_binding(
            root, _mapping(files.get("scored"), "scored binding")
        )
        verified.append(
            InputBinding(record, x_path, x_sha, x_bytes, mask_path, mask_sha, mask_bytes)
        )
    if len(verified) != 64:
        raise ValueError("R10 must bind exactly 64 fitting sequences")
    return verified


def map_fit_arrays(binding: InputBinding) -> FitArrays:
    """Map one already-hashed fitting feature array and scored mask read-only."""
    x = np.load(binding.x_path, mmap_mode="r", allow_pickle=False)
    scored = np.load(binding.scored_path, mmap_mode="r", allow_pickle=False)
    if x.dtype != np.dtype(np.float32) or x.shape != (20_000, 112):
        raise ValueError("R3 fitting feature array must be float32[20000,112]")
    if scored.dtype != np.dtype(np.bool_) or scored.shape != (20_000,):
        raise ValueError("R3 fitting scored mask must be bool[20000]")
    if bool(np.asarray(scored[:99]).any()) or int(scored.sum()) != int(
        binding.record.get("scored_rows", -1)
    ):
        raise ValueError("R3 fitting mask differs from its source-bound warmup/count metadata")
    return FitArrays(binding.record, x, scored)


def input_file_receipts(bindings: list[InputBinding]) -> list[dict[str, object]]:
    """Return compact scalar hashes for only the mapped fitting inputs."""
    receipts: list[dict[str, object]] = []
    for binding in bindings:
        common = {
            "role": "fit",
            "source": binding.record["source"],
            "group": binding.record["group"],
            "component": binding.record["component"],
            "seq_ix": binding.record["seq_ix"],
        }
        receipts.append(
            {
                **common,
                "key": "x",
                "name": binding.x_path.name,
                "sha256": binding.x_sha256,
                "bytes": binding.x_bytes,
            }
        )
        receipts.append(
            {
                **common,
                "key": "scored",
                "name": binding.scored_path.name,
                "sha256": binding.scored_sha256,
                "bytes": binding.scored_bytes,
            }
        )
    return receipts


def _read_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    return _mapping(value, path.name)


def _validate_record_files(record: dict[str, object], index: int) -> None:
    files = _mapping(record.get("files"), f"record {index} files")
    if set(files) != set(ARRAY_KEYS):
        raise ValueError(f"record {index} must bind exactly x, y and scored metadata")
    source = record.get("source")
    group = _record_int(record, "group")
    for key in ARRAY_KEYS:
        binding = _mapping(files.get(key), f"record {index} {key} binding")
        name = binding.get("name")
        expected_name = f"{source}_{group}_{key}.npy"
        if name != expected_name or Path(str(name)).name != name:
            raise ValueError(f"record {index} has an unsafe or unexpected {key} filename")
        if not _is_sha256(binding.get("sha256")) or _positive_int(binding.get("bytes")) is None:
            raise ValueError(f"record {index} has an invalid {key} hash or byte count")


def _verify_binding(root: Path, binding: dict[str, object]) -> tuple[Path, str, int]:
    name = binding.get("name")
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("R3 array names must be safe basenames")
    path = root / name
    if path.is_symlink() or path.resolve().parent != root or not path.is_file():
        raise ValueError(f"R3 fit input is missing or escapes the sample directory: {name}")
    expected_bytes = _positive_int(binding.get("bytes"))
    expected_sha = binding.get("sha256")
    if expected_bytes is None or not _is_sha256(expected_sha):
        raise ValueError(f"invalid byte count or SHA-256 for {name}")
    if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha:
        raise ValueError(f"R3 fit input bytes differ from source manifest: {name}")
    return path, str(expected_sha), expected_bytes


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed object")
    return value


def _record_list(value: object, label: str) -> list[dict[str, object]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{label} must be a list of objects")
    return value


def _integer_tuple(value: object, size: int, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"invalid {label} length")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ValueError(f"{label} values must be integers")
    return tuple(value)


def _record_int(record: dict[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"record {key} must be an integer")
    return value


def _positive_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
