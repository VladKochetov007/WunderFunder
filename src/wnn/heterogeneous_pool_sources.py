from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from wnn.splits import CampaignSplit, file_sha256
from wnn.training.chronological_teacher import validate_chronological_teacher_membership
from wnn.training.sequence_epoch_prediction import digest, validate_index

# Producer sidecars have heterogeneous schemas; validate their dynamic JSON boundary.
JsonObject = dict[str, Any]
FORMAT = "wnn.heterogeneous-teacher-pool.v1"
INPUT_FORMAT = "wnn.heterogeneous-teacher-pool-inputs.v1"
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
PROFILES = {
    "gru-only": (("tl3c_fin256_s3", "gru", 3, 0.5), ("tl3c_fin256_s4", "gru", 3, 0.5)),
    "heterogeneous": (
        ("tl3c_fin256_s3", "gru", 3, 0.35),
        ("tl3c_fin256_s4", "gru", 3, 0.35),
        ("g2-galam-lstm128", "galam", 2, 0.1),
        ("het6_lstm_s1201", "lstm", 3, 0.1),
        ("het6_tcn_s1202", "tcn", 3, 0.1),
    ),
}


def read_object(path: Path) -> JsonObject:
    """Read bounded producer metadata, rejecting non-object JSON."""
    if not 0 < path.stat().st_size <= 2_000_000:
        raise ValueError("source metadata exceeds its two-megabyte bound")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("source metadata must be a JSON object")
    return value


def pinned_path(record: JsonObject, key: str) -> Path:
    """Verify the actual bytes of one explicitly pinned regular file."""
    value = record.get(key)
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValueError(f"source lacks pinned {key}")
    path = Path(value["path"])
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"source {key} must be an absolute regular file")
    if file_sha256(path) != value.get("sha256"):
        raise ValueError(f"source {key} bytes changed")
    return path


def ordered_groups(value: object) -> list[int]:
    """Reject ambiguous, duplicate or malformed source-qualified row indexes."""
    if (
        not isinstance(value, list)
        or any(type(group) is not int or group < 0 for group in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError("source row_groups must be unique nonnegative integers")
    return value


def groups_hash(groups: list[int]) -> str:
    """Hash row indexes with the canonical little-endian int64 representation."""
    return hashlib.sha256(np.asarray(groups, dtype="<i8").tobytes()).hexdigest()


def validate_profile(profile: str, sources: list[JsonObject]) -> None:
    """Bind source order, fixed endpoint and coefficients to the six-fit proposal."""
    expected = PROFILES.get(profile)
    if expected is None or len(sources) != len(expected):
        raise ValueError("unknown pool profile or changed teacher count")
    for record, (run, family, epoch, coefficient) in zip(sources, expected, strict=True):
        if (
            record.get("run") != run
            or record.get("family") != family
            or record.get("epoch") != epoch
            or not math.isclose(float(record.get("coefficient", 0)), coefficient, abs_tol=1e-12)
        ):
            raise ValueError("teacher order, epoch or coefficient differs from frozen profile")


def _legacy_qualification(record: JsonObject, split: CampaignSplit, side: JsonObject) -> str:
    provenance_path = pinned_path(record, "provenance")
    provenance = read_object(provenance_path)
    args = provenance.get("args", {})
    architecture = {
        "gru": {"arch": "gru", "cell": "gru", "hidden": 256, "layers": 2},
        "lstm": {"arch": "gru", "cell": "lstm", "hidden": 256, "layers": 2},
        "tcn": {"arch": "tcn", "cell": "gru", "hidden": 128, "layers": 12},
    }.get(record["family"])
    if architecture is None or any(args.get(key) != value for key, value in architecture.items()):
        raise ValueError("legacy source architecture differs from its declared teacher family")
    validate_chronological_teacher_membership(provenance, split, record["run"])
    rows = ordered_groups(side.get("row_groups"))
    if (
        side.get("campaign_export_scope") != "chronological-full-train-index-feature-only"
        or side.get("campaign_split_manifest_sha256") != split.sha256
        or side.get("evaluation_membership_sha256") != groups_hash(rows)
    ):
        raise ValueError("legacy prediction rows lack the original full-train export binding")
    if (
        provenance.get("split_manifest_sha256") != split.sha256
        or args.get("train_source") != "train"
        or args.get("precision") != "fp32"
        or args.get("name") != record["run"]
        or args.get("checkpoint_policy") != "fixed"
        or args.get("checkpoint_epoch") != record["epoch"]
        or provenance.get("selected_epoch") != record["epoch"]
        or any(
            args.get(key)
            for key in (
                "init",
                "distill",
                "distill_cache_root",
                "propensity",
                "prop_input",
                "mask_weight",
            )
        )
    ):
        raise ValueError("legacy source is not the declared fresh train-only root teacher")
    checkpoint_name = Path(record["checkpoint"]["path"]).name
    if (
        provenance.get("checkpoint_sha256", {}).get(checkpoint_name)
        != record["checkpoint"]["sha256"]
        or side.get("run_provenance_sha256") != record["provenance"]["sha256"]
    ):
        raise ValueError("legacy checkpoint/export is not bound to the original provenance")
    implementation = provenance.get("trainer_implementation", {})
    if digest(implementation.get("files")) != implementation.get("sha256"):
        raise ValueError("legacy trainer source inventory disagrees with its digest")
    normalization = provenance.get("normalization_identity", {})
    if set(normalization) != {"mean", "std"} or any(
        normalization[name].get("dtype") != "torch.float32" for name in normalization
    ):
        raise ValueError("legacy normalization provenance is absent or unsupported")
    return implementation["sha256"]


def _galam_qualification(record: JsonObject, split: CampaignSplit, side: JsonObject) -> str:
    index = read_object(pinned_path(record, "epoch_index"))
    validate_index(index, record["epoch_index_canonical_sha256"])
    entries = [entry for entry in index["entries"] if entry["epoch"] == record["epoch"]]
    identity = index["identity"]
    data = identity.get("data", {})
    if (
        len(entries) != 1
        or entries[0]["checkpoint_sha256"] != record["checkpoint"]["sha256"]
        or side.get("epoch_index_sha256") != record["epoch_index"]["sha256"]
        or data.get("split_sha256") != split.sha256
        or data.get("fit_members") != [["train", group] for group in split.train_groups]
        or data.get("evaluation_members") != [["valid", group] for group in split.evaluation_groups]
        or data.get("teacher") is not None
        or data.get("hint_members")
        or digest(data) != identity.get("data_sha256")
        or digest(identity.get("source_files")) != identity.get("source_sha256")
    ):
        raise ValueError("GALAM epoch/index/source/data qualification failed")
    normalization = data.get("normalization", {})
    if normalization.get("scope") != "all_fit" or normalization.get("sequence_count") != len(
        split.train_groups
    ):
        raise ValueError("GALAM normalization is not fitting-only")
    return identity["source_sha256"]


def validate_source(record: JsonObject, split: CampaignSplit) -> tuple[np.ndarray, dict[int, int]]:
    """Verify real producer ancestry and bind every prediction row before combining."""
    prediction = pinned_path(record, "prediction")
    side = read_object(pinned_path(record, "sidecar"))
    pinned_path(record, "checkpoint")
    qualification = read_object(pinned_path(record, "qualification"))
    rows = ordered_groups(side.get("row_groups"))
    source_split = side.get("split_manifest_sha256", side.get("campaign_split_manifest_sha256"))
    if (
        side.get("run") != record["run"]
        or side.get("epoch") != record["epoch"]
        or side.get("split") != "train"
        or side.get("feature_only") is not True
        or side.get("inference_precision") != "fp32"
        or source_split != split.sha256
        or side.get("checkpoint_sha256") != record["checkpoint"]["sha256"]
        or side.get("prediction_sha256") != record["prediction"]["sha256"]
        or not set(split.train_groups).issubset(rows)
    ):
        raise ValueError("source export identity/coverage/precision differs")
    producer = record.get("producer")
    if producer == "legacy_rnn_chronological":
        source = _legacy_qualification(record, split, side)
    elif producer == "galam_feature_export":
        source = _galam_qualification(record, split, side)
    else:
        raise ValueError("unsupported actual source producer")
    expected = {
        "schema": "wnn.teacher-source-qualification.v1",
        "run": record["run"],
        "producer": producer,
        "epoch": record["epoch"],
        "checkpoint_sha256": record["checkpoint"]["sha256"],
        "prediction_sha256": record["prediction"]["sha256"],
        "sidecar_sha256": record["sidecar"]["sha256"],
        "split_manifest_sha256": split.sha256,
        "fit_groups": {"train": list(split.train_groups), "valid": []},
        "evaluation_valid_groups": list(split.evaluation_groups),
        "source_sha256": source,
        "normalization_fit_groups_sha256": groups_hash(list(split.train_groups)),
    }
    if any(qualification.get(key) != value for key, value in expected.items()):
        raise ValueError("source qualification disagrees with actual producer evidence")
    values = np.load(prediction, mmap_mode="r", allow_pickle=False)
    if values.dtype != np.float32 or values.shape != (len(rows), 20000, 2):
        raise ValueError("source prediction shape/dtype differs")
    return values, {group: position for position, group in enumerate(rows)}
