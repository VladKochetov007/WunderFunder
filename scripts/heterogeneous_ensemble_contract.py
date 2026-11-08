"""Bind heterogeneous prediction producers to the frozen chronological evaluation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from attrs import define, field

from wnn.chronological_source_index import OFFICIAL_SOURCE_PINS
from wnn.splits import CampaignSplit, file_sha256, load_campaign_split
from wnn.training.chronological_teacher import validate_chronological_teacher_membership
from wnn.training.sequence_epoch_prediction import validate_epoch_producer

# Producer JSON is heterogeneous; validate it at each deserialization boundary.
JsonObject = dict[str, Any]
SCHEMA = "wnn.heterogeneous-ensemble-inputs.v1"
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
TL3_WEIGHTS = {"w18": 0.34, "tiedlead_s5": 0.33, "tiedlead_s6": 0.33}
RESERVED_ROLES = {
    "w18": ("tl3c_w18final_distilled_s5", 1),
    "tiedlead_s5": ("tl3c_fttall_tlead_s5", 3),
    "tiedlead_s6": ("tl3c_fttall_tlead_s6", 3),
    "fbc": ("tl3c_fbc1k_s5", 3),
    "fin256_s3": ("tl3c_fin256_s3", 3),
    "fin256_s4": ("tl3c_fin256_s4", 3),
    "control_s1301": ("het6_control_s1301", 3),
    "control_s1302": ("het6_control_s1302", 3),
    "hetero_s1301": ("het6_heterogeneous_s1301", 3),
    "hetero_s1302": ("het6_heterogeneous_s1302", 3),
}
DEFAULT_ENSEMBLES = {
    "TL3": TL3_WEIGHTS,
    "PR51_replacement": {"fbc": 0.34, "tiedlead_s5": 0.33, "tiedlead_s6": 0.33},
    "GRU_family": {"fin256_s3": 0.5, "fin256_s4": 0.5},
    "heterogeneous_teacher_pool": {
        "fin256_s3": 0.35,
        "fin256_s4": 0.35,
        "galam": 0.1,
        "lstm": 0.1,
        "tcn": 0.1,
    },
    "TL3_plus_GALAM": {"TL3": 0.9, "galam": 0.1},
    "TL3_plus_LSTM": {"TL3": 0.9, "lstm": 0.1},
    "TL3_plus_TCN": {"TL3": 0.9, "tcn": 0.1},
    "TL3_heterogeneous": {"TL3": 0.7, "galam": 0.1, "lstm": 0.1, "tcn": 0.1},
    "GRU_student_mean": {"control_s1301": 0.5, "control_s1302": 0.5},
    "heterogeneous_student_mean": {"hetero_s1301": 0.5, "hetero_s1302": 0.5},
    "TL3_plus_GRU_students": {"TL3": 0.8, "GRU_student_mean": 0.2},
    "TL3_plus_heterogeneous_students": {"TL3": 0.8, "heterogeneous_student_mean": 0.2},
}


def validate_role(name: str, record: JsonObject) -> None:
    """Keep canonical TL3 roles distinct from replacement or wrong-epoch weights."""
    if name in RESERVED_ROLES and record.get("state") != "pending":
        run, epoch = RESERVED_ROLES[name]
        if (record.get("run"), record.get("epoch"), record.get("kind")) != (
            run,
            epoch,
            "legacy_bank",
        ):
            raise ValueError(f"reserved TL3 role {name} must bind {run}@ep{epoch}")


def read_object(path: Path) -> JsonObject:
    """Read one JSON object, rejecting duplicate keys and non-object roots."""

    def unique(pairs: list[tuple[str, object]]) -> JsonObject:
        result: JsonObject = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(result, dict):
        raise ValueError(f"JSON object required: {path}")
    return result


def pinned_file(record: object, base: Path) -> Path:
    """Require an existing file whose bytes match its explicit SHA256."""
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError("file identity requires path and SHA256")
    expected = record.get("sha256")
    if not isinstance(expected, str) or re.fullmatch("[0-9a-f]{64}", expected) is None:
        raise ValueError("file identity has an invalid SHA256")
    path = Path(record["path"])
    path = path if path.is_absolute() else base / path
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"file is absent or its SHA256 changed: {path}")
    return path.resolve()


@define
class PredictionSource:
    """Provide one explicitly aligned raw FP32 sequence from a bank or native shards."""

    record: JsonObject
    metadata_path: Path
    checkpoint_path: Path
    positions: dict[int, int]
    bank: np.ndarray | None = None
    shards: dict[int, JsonObject] = field(factory=dict)

    def sequence(self, group: int) -> np.ndarray:
        """Read an immutable aligned sequence, verifying each native shard before use."""
        if self.bank is not None:
            values = np.asarray(self.bank[self.positions[group]])
        else:
            entry = self.shards[group]
            path = pinned_file(
                {"path": entry["file"], "sha256": entry["sha256"]}, self.metadata_path.parent
            )
            if not path.is_relative_to(self.metadata_path.parent):
                raise ValueError("prediction shard escapes its producer directory")
            values = np.load(path, allow_pickle=False)
        if values.shape != (20_000, 2) or values.dtype != np.float32:
            raise ValueError("prediction sequence must be FP32 with shape[20000,2]")
        if not np.isfinite(values).all():
            raise ValueError("nonfinite prediction sequence")
        return values


def load_bank(
    record: JsonObject, base: Path, split: CampaignSplit, order: list[int]
) -> PredictionSource:
    """Bind a canonical legacy bank to its real exporter and fitting provenance."""
    metadata_path = pinned_file(record.get("metadata"), base)
    checkpoint = pinned_file(record.get("checkpoint"), base)
    provenance_path = pinned_file(record.get("provenance"), base)
    path = pinned_file(record.get("predictions"), base)
    metadata, provenance = read_object(metadata_path), read_object(provenance_path)
    validate_chronological_teacher_membership(provenance, split, str(record.get("run")))
    expected = {
        "run": record.get("run"),
        "split": "valid",
        "epoch": record.get("epoch"),
        "checkpoint_sha256": record["checkpoint"]["sha256"],
        "run_provenance_sha256": record["provenance"]["sha256"],
        "prediction_sha256": record["predictions"]["sha256"],
        "split_manifest_sha256": split.sha256,
        "inference_precision": "fp32",
        "row_groups": order,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("legacy prediction producer/epoch/order identity mismatch")
    args = provenance.get("args", {})
    if (
        args.get("name") != record["run"]
        or args.get("precision") != "fp32"
        or args.get("checkpoint_policy") != "fixed"
        or args.get("checkpoint_epoch") != record["epoch"]
        or provenance.get("checkpoint_sha256", {}).get(metadata.get("checkpoint_file"))
        != record["checkpoint"]["sha256"]
    ):
        raise ValueError("legacy checkpoint does not match its fixed source recipe")
    bank = np.load(path, mmap_mode="r", allow_pickle=False)
    if bank.shape != (len(order), 20_000, 2) or bank.dtype != np.float32:
        raise ValueError("legacy bank must cover every complete FP32 evaluation sequence")
    return PredictionSource(
        record, metadata_path, checkpoint, dict(zip(order, range(len(order)), strict=True)), bank
    )


def load_shards(record: JsonObject, base: Path, split: CampaignSplit) -> PredictionSource:
    """Validate retained-epoch GALAM exports without converting their source layout."""
    path = pinned_file(record.get("metadata"), base)
    checkpoint = pinned_file(record.get("checkpoint"), base)
    metadata = read_object(path)
    if (
        metadata.get("schema") != "sequence-predictions-v1"
        or metadata.get("complete") is not True
        or metadata.get("membership") != "evaluation"
        or metadata.get("split_sha256") != split.sha256
        or metadata.get("sequence_length") != 20_000
        or metadata.get("hint_dim") != 0
    ):
        raise ValueError("sequence producer is not a complete official-valid prediction export")
    producer = metadata.get("producer", {})
    validate_epoch_producer(
        producer, split.sha256, [("train", g) for g in split.train_groups], primary_report_only=True
    )
    if (
        producer.get("checkpoint_sha256") != record["checkpoint"]["sha256"]
        or producer["epoch_admission"]["entry"]["epoch"] != record.get("epoch")
        or producer.get("config", {}).get("precision_mode", "fp32") != "fp32"
    ):
        raise ValueError("sequence checkpoint/epoch/precision binding differs")
    entries = metadata.get("sequences")
    if not isinstance(entries, list) or any(
        not isinstance(e, dict) or e.get("split") != "valid" or type(e.get("row_group")) is not int
        for e in entries
    ):
        raise ValueError("sequence export has malformed membership")
    shards = {entry["row_group"]: entry for entry in entries}
    if len(shards) != len(entries) or set(shards) != set(split.evaluation_groups):
        raise ValueError("sequence export omits or duplicates official-valid members")
    return PredictionSource(record, path, checkpoint, {}, shards=shards)


def load_contract(path: Path) -> tuple[JsonObject, CampaignSplit, Path, list[int]]:
    """Verify actual sources and freeze complete official train/valid membership."""
    manifest = read_object(path)
    if manifest.get("schema") != SCHEMA or manifest.get("baseline") != "TL3":
        raise ValueError("manifest requires the heterogeneous schema and named TL3 baseline")
    base = path.resolve().parent
    split_path = pinned_file(manifest.get("split"), base)
    if manifest["split"]["sha256"] != SPLIT_SHA256:
        raise ValueError("chronological primary split differs from the admitted recipe")
    root = Path(manifest["data_root"])
    root = root if root.is_absolute() else base / root
    split = load_campaign_split(split_path, root)
    if (
        len(split.train_groups) != 10325
        or split.valid_groups
        or len(split.evaluation_groups) != 1873
    ):
        raise ValueError("the full10325/1873 chronological boundary is mandatory")
    validation = pinned_file(manifest.get("validation"), base)
    if manifest["validation"]["sha256"] != OFFICIAL_SOURCE_PINS["valid"].sha256:
        raise ValueError("validation is not the admitted official source bytes")
    with np.load(root / "cache/valid_index.npz", allow_pickle=False) as index:
        order = index["row_group"].tolist()
    if len(order) != 1873 or len(set(order)) != 1873 or set(order) != set(split.evaluation_groups):
        raise ValueError("actual source index does not cover full official valid")
    if not isinstance(manifest.get("members"), dict):
        raise ValueError("manifest members must be a named object")
    return manifest, split, validation, order
