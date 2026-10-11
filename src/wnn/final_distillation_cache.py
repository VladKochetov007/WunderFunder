"""H390 all-data cache validation, separate from clean-fold cache validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

FINAL_CACHE_FORMAT = "wnn.all-data-homogeneous-distillation-cache"
FINAL_CACHE_VERSION = 2
SEQUENCE_ROWS = 20_000
TARGET_WIDTH = 2
DISK_SAFETY_RESERVE_BYTES = 16 << 20
TEACHER_SPLIT_MANIFEST_RELATIVE = Path("research/campaign_2026-09-29/artifacts/splits/fold1.json")
EXPECTED_SPLIT_SHA256 = "31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae"
EXPECTED_PROPENSITY_SHA256 = "1fbf412df8823c703fdf79263830f341269a68e769bc4a7258d93cdf8e40dc5c"
R1_ARCHIVE_SHA256 = "35761b8cf6bc5bcf5079b134bfcce81e48b5fd83136630c51b89d48f8759caf1"
R1_BASELINE_MANIFEST_SHA256 = "382beb68d556dd63215a9f7467bda09a23626ec3e1a2195739cffcab6819d9cc"
TEACHERS = {
    "vt_f1_muonall_h384_e3": {
        "seed": 3,
        "checkpoint_sha256": "9c7d5e55fde7112dc691c80670fb17b62f24b9d759d2c90312a9ec89334843c6",
        "provenance_sha256": "ca9f7c32910e5d71acddf78a474c06eee324b54d781ad9c736cd1f9a4a83deae",
    },
    "vt_f1_muonall_h384_e3_s4": {
        "seed": 4,
        "checkpoint_sha256": "752f462036a8e340db00de98dcb141795f3a0ae54cfffce3e95ce7ceed7f5e9e",
        "provenance_sha256": "e11ca9b3bfb9fc302f199514eb075960933e8fd27dfa49d58a31b76d5f5011f6",
    },
    "vt_f1_muonall_h384_e3_s5": {
        "seed": 5,
        "checkpoint_sha256": "ee154f035720069ef26bf0fd328c61887cdbb426c49d51fb25172e5bb52b1f80",
        "provenance_sha256": "45555945babdc6af0d56e1b5ab6a68585383bf32a73335fe8b92d84af7bdf03d",
    },
}
PARENTS = {
    3: {
        "run": "vt_all_fin_mu256p0e4_s3",
        "checkpoint_sha256": "ca50c35c05a680b8009bb444485f1ff73aa99e3e50f96138faf266a92c146e5d",
        "provenance_sha256": "7afe92cbcc9f5f1f615d0b536571bd4fe526365ddbd663e48b8c14619a237a06",
        "reporting_groups_sha256": "8cf7c5cdf70a7dd4c26dc672d6b8f2d704de734eedbea1b545d92dd855d672a3",
    },
    4: {
        "run": "vt_all_fin_mu256p0e4_s4",
        "checkpoint_sha256": "a6c1b1bf2b6b6c68df71af359cc732a7842c119bdbd43a5d4965bc321b47f0fc",
        "provenance_sha256": "8a33e822638a292beda829163e1f2a750e6f70649b3182aaba7890ae44c4605e",
        "reporting_groups_sha256": "8cf7c5cdf70a7dd4c26dc672d6b8f2d704de734eedbea1b545d92dd855d672a3",
    },
}
TARGET_MULTIPLIER = 0.8125


def file_sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    """Hash a file in bounded reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def row_groups_sha256(groups: list[int] | tuple[int, ...]) -> str:
    """Hash ordered row-group identities using little-endian int64 bytes."""
    values = np.asarray(groups, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def source_qualified_groups_sha256(source: str, groups: list[int] | tuple[int, ...]) -> str:
    """Hash a source name and ordered row-group IDs as one distinct identity."""
    digest = hashlib.sha256(source.encode("utf-8") + b"\0")
    digest.update(np.asarray(groups, dtype="<i8").tobytes())
    return digest.hexdigest()


def cache_disk_requirement(row_count: int) -> tuple[int, int, int]:
    """Return output bytes, reserved metadata/headroom bytes, and admitted bytes."""
    if row_count <= 0:
        raise ValueError("H390 cache row count must be positive")
    output_bytes = row_count * SEQUENCE_ROWS * TARGET_WIDTH * np.dtype(np.float32).itemsize
    return output_bytes, DISK_SAFETY_RESERVE_BYTES, output_bytes + DISK_SAFETY_RESERVE_BYTES


def homogeneous_teacher_target(teacher_values: list[np.ndarray]) -> np.ndarray:
    """Return the H390 float32 mean scaled by the declared total coefficient."""
    if len(teacher_values) != len(TEACHERS):
        raise ValueError(f"H390 requires exactly {len(TEACHERS)} teacher chunks")
    shape = teacher_values[0].shape
    if any(values.shape != shape or values.dtype != np.float32 for values in teacher_values):
        raise ValueError("H390 teacher chunks must have matching float32 shapes")
    if any(not np.isfinite(values).all() for values in teacher_values):
        raise ValueError("H390 teacher chunks must contain only finite values")
    mean = np.add.reduce(teacher_values, dtype=np.float32) / np.float32(len(teacher_values))
    return mean * np.float32(TARGET_MULTIPLIER)


def target_row_groups(data_root: Path, split: str) -> list[int]:
    """Read the canonical full-source row-group order for one supplied source."""
    path = data_root / "cache" / f"{split}_targets.npz"
    with np.load(path, allow_pickle=False) as targets:
        if "row_group" not in targets.files:
            raise ValueError(f"target index lacks row_group: {path}")
        groups = np.asarray(targets["row_group"], dtype=np.int64)
    if groups.ndim != 1 or len(set(map(int, groups))) != len(groups):
        raise ValueError(f"target index has invalid or repeated row groups: {path}")
    return list(map(int, groups))


def verify_final_population_sources(
    data_root: Path, split_manifest: dict[str, Any]
) -> dict[str, list[int]]:
    """Check the raw parquet identities and target-index membership for H390."""
    source_identities = split_manifest.get("sources")
    if not isinstance(source_identities, dict):
        raise ValueError("frozen teacher split manifest lacks source identities")
    registry_value = os.environ.get("WNN_PORTABLE_DATA_MANIFEST")
    registry = Path(registry_value) if registry_value else None
    result: dict[str, list[int]] = {}
    for split in ("train", "valid"):
        parquet_name = f"datasets/{split}.parquet"
        target_name = f"cache/{split}_targets.npz"
        try:
            parquet_identity = source_identities[parquet_name]
            target_identity = source_identities[target_name]
        except KeyError as error:
            raise ValueError(f"frozen teacher split manifest omits {error.args[0]}") from error
        parquet_path = data_root / parquet_name
        if registry is not None:
            from wnn.portability import verify_portable_source

            verify_portable_source(parquet_path, parquet_name, parquet_identity, registry)
        else:
            stat = parquet_path.stat()
            if stat.st_size != parquet_identity.get(
                "bytes"
            ) or stat.st_mtime_ns != parquet_identity.get("mtime_ns"):
                raise ValueError(
                    f"supplied source identity differs from the frozen split: {parquet_name}"
                )

        target_path = data_root / target_name
        target_stat = target_path.stat()
        if target_stat.st_size != target_identity.get("bytes"):
            raise ValueError(
                f"target-index source identity differs from the frozen split: {target_name}"
            )
        digest = hashlib.sha256()
        with np.load(target_path, allow_pickle=False) as targets:
            for field in ("row_group", "seq_ix"):
                if field not in targets.files:
                    raise ValueError(f"target index lacks {field}: {target_name}")
                values = np.asarray(targets[field], dtype="<i8")
                digest.update(values.tobytes())
        if digest.hexdigest() != target_identity.get("index_sha256"):
            raise ValueError(f"target-index contents differ from the frozen split: {target_name}")
        result[split] = target_row_groups(data_root, split)
        manifest_groups = [
            int(item["row_group"]) for item in split_manifest.get("sequences", {}).get(split, [])
        ]
        if len(manifest_groups) != len(set(manifest_groups)) or set(manifest_groups) != set(
            result[split]
        ):
            raise ValueError(f"{split} target population differs from the frozen split manifest")
    return result


def verify_h390_r1_source_snapshot(source_root: Path) -> dict[str, Any]:
    """Verify the original frozen R1 packet without applying R2 rules."""
    source_root = source_root.resolve(strict=True)
    parent_manifest = source_root / "SOURCE_MANIFEST.json"
    if file_sha256(parent_manifest) != R1_BASELINE_MANIFEST_SHA256:
        raise ValueError("H390 source parent manifest differs from frozen R1")
    parent_root = source_root.parent / "r1_baseline"
    archive_path = source_root.parent / "h390_reference_r1.tar.gz"
    if file_sha256(archive_path) != R1_ARCHIVE_SHA256:
        raise ValueError("H390 R1 reference archive differs from the frozen source")
    packet_manifest_path = source_root.parent / "PACKET_MANIFEST_R1.json"
    packet_manifest = json.loads(packet_manifest_path.read_text())
    packet_files = packet_manifest.get("files")
    expected_packet_files = {
        "README.md",
        "h390_reference_r1.tar.gz",
        "parent_receipts_r1/vt_all_fin_mu256p0e4_s3.provenance.json",
        "parent_receipts_r1/vt_all_fin_mu256p0e4_s4.provenance.json",
        "child_source_v1/SOURCE_MANIFEST.json",
        "child_source_v1/SOURCE_MANIFEST_H390.json",
    }
    if (
        packet_manifest.get("schema") != "wnn.h390-packet-manifest.v1"
        or not isinstance(packet_files, dict)
        or set(packet_files) != expected_packet_files
    ):
        raise ValueError("H390 packet manifest has an unexpected file set")
    for relative, digest in packet_files.items():
        if file_sha256(source_root.parent / relative) != digest:
            raise ValueError(f"H390 packet file changed after freezing: {relative}")
    parent_manifest_data = json.loads(parent_manifest.read_text())
    parent_sources = parent_manifest_data.get("sources")
    if not isinstance(parent_sources, dict) or not parent_sources:
        raise ValueError("H390 R1 parent manifest has no source list")
    for relative, digest in parent_sources.items():
        if file_sha256(parent_root / relative) != digest:
            raise ValueError(f"H390 copied R1 parent differs from its manifest: {relative}")
    manifest_path = source_root / "SOURCE_MANIFEST_H390.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "wnn.h390-source-snapshot.v1"
        or manifest.get("parent_archive_sha256") != R1_ARCHIVE_SHA256
        or manifest.get("parent_manifest_sha256") != R1_BASELINE_MANIFEST_SHA256
    ):
        raise ValueError("H390 source manifest lacks the reviewed R1 ancestry")
    expected = manifest.get("sources")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("H390 source manifest has no source hashes")
    current_paths = {
        path.relative_to(source_root).as_posix()
        for folder in (source_root / "scripts", source_root / "src/wnn")
        for path in folder.rglob("*.py")
        if path.is_file()
    }
    if set(expected) != current_paths:
        raise ValueError("H390 source file set differs from its frozen manifest")
    for relative, digest in expected.items():
        if file_sha256(source_root / relative) != digest:
            raise ValueError(f"H390 source file changed after freezing: {relative}")
    return {
        "source_manifest_sha256": file_sha256(manifest_path),
        "packet_manifest_sha256": file_sha256(packet_manifest_path),
        "parent_archive_sha256": R1_ARCHIVE_SHA256,
        "parent_manifest_sha256": R1_BASELINE_MANIFEST_SHA256,
        "verified_sources": expected,
    }


def verify_h390_source_snapshot(source_root: Path) -> dict[str, Any]:
    """Verify the R2 child source, packet receipts, and immutable R1 ancestry."""
    source_root = source_root.resolve(strict=True)
    packet_root = source_root.parent
    if file_sha256(source_root / "SOURCE_MANIFEST.json") != R1_BASELINE_MANIFEST_SHA256:
        raise ValueError("H390 R2 parent manifest differs from frozen R1")
    if file_sha256(packet_root / "h390_reference_r1.tar.gz") != R1_ARCHIVE_SHA256:
        raise ValueError("H390 R2 reference archive differs from frozen R1")

    packet_manifest_path = packet_root / "PACKET_MANIFEST_R2.json"
    packet_manifest = json.loads(packet_manifest_path.read_text())
    packet_files = packet_manifest.get("files")
    expected_packet_files = {
        "README.md",
        "h390_reference_r1.tar.gz",
        "parent_receipts_r1/vt_all_fin_mu256p0e4_s3.provenance.json",
        "parent_receipts_r1/vt_all_fin_mu256p0e4_s4.provenance.json",
        "teacher_receipts_r2/vt_f1_muonall_h384_e3.provenance.json",
        "teacher_receipts_r2/vt_f1_muonall_h384_e3_s4.provenance.json",
        "teacher_receipts_r2/vt_f1_muonall_h384_e3_s5.provenance.json",
        "child_source_v1/SOURCE_MANIFEST.json",
        "child_source_v1/SOURCE_MANIFEST_H390.json",
    }
    if (
        packet_manifest.get("schema") != "wnn.h390-packet-manifest.v2"
        or not isinstance(packet_files, dict)
        or set(packet_files) != expected_packet_files
    ):
        raise ValueError("H390 R2 packet manifest has an unexpected file set")
    for relative, digest in packet_files.items():
        if file_sha256(packet_root / relative) != digest:
            raise ValueError(f"H390 R2 packet file changed after freezing: {relative}")
    for run, spec in TEACHERS.items():
        receipt = packet_root / "teacher_receipts_r2" / f"{run}.provenance.json"
        if file_sha256(receipt) != spec["provenance_sha256"]:
            raise ValueError(f"H390 R2 teacher receipt differs from its reviewed producer: {run}")

    parent_root = packet_root / "r1_baseline"
    parent_manifest_data = json.loads((source_root / "SOURCE_MANIFEST.json").read_text())
    parent_sources = parent_manifest_data.get("sources")
    if not isinstance(parent_sources, dict) or not parent_sources:
        raise ValueError("H390 R1 parent manifest has no source list")
    for relative, digest in parent_sources.items():
        if file_sha256(parent_root / relative) != digest:
            raise ValueError(f"H390 R2 copied R1 parent differs from its manifest: {relative}")

    manifest_path = source_root / "SOURCE_MANIFEST_H390.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "wnn.h390-source-snapshot.v2"
        or manifest.get("parent_archive_sha256") != R1_ARCHIVE_SHA256
        or manifest.get("parent_manifest_sha256") != R1_BASELINE_MANIFEST_SHA256
        or manifest.get("teacher_provenance_sha256")
        != {run: spec["provenance_sha256"] for run, spec in TEACHERS.items()}
    ):
        raise ValueError("H390 R2 source manifest lacks the reviewed R1/teacher ancestry")
    expected = manifest.get("sources")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("H390 R2 source manifest has no source hashes")
    current_paths = {
        path.relative_to(source_root).as_posix()
        for folder in (source_root / "scripts", source_root / "src/wnn")
        for path in folder.rglob("*.py")
        if path.is_file()
    }
    if set(expected) != current_paths:
        raise ValueError("H390 R2 source file set differs from its frozen manifest")
    for relative, digest in expected.items():
        if file_sha256(source_root / relative) != digest:
            raise ValueError(f"H390 R2 source file changed after freezing: {relative}")
    return {
        "source_manifest_sha256": file_sha256(manifest_path),
        "packet_manifest_sha256": file_sha256(packet_manifest_path),
        "parent_archive_sha256": R1_ARCHIVE_SHA256,
        "parent_manifest_sha256": R1_BASELINE_MANIFEST_SHA256,
        "teacher_provenance_sha256": manifest["teacher_provenance_sha256"],
        "verified_sources": expected,
    }


def _safe_array_path(cache_root: Path, entry: dict[str, Any], key: str) -> Path:
    relative = Path(str(entry.get("path", "")))
    if relative.is_absolute() or len(relative.parts) != 1:
        raise ValueError(f"cache filename must be one relative path: {key}")
    path = (cache_root / relative).resolve(strict=True)
    try:
        path.relative_to(cache_root)
    except ValueError as error:
        raise ValueError(f"cache path escapes its root: {key}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_final_distillation_cache(cache_root: Path, data_root: Path) -> dict[str, Any]:
    """Validate the explicit H390 all-data target schema and exact raw membership.

    The ordinary fold-bound cache validator remains unchanged and still requires
    a clean campaign split. This validator accepts only the H390 final schema,
    two main-target arrays, and all raw train/valid row groups.
    """
    cache_root = cache_root.resolve(strict=True)
    metadata_path = cache_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("format") != FINAL_CACHE_FORMAT
        or metadata.get("version") != FINAL_CACHE_VERSION
    ):
        raise ValueError("unsupported all-data final distillation cache")
    if metadata.get("mode") != "all_data_final" or metadata.get("heldout_claim") is not False:
        raise ValueError("all-data final cache must explicitly make no heldout claim")
    if metadata.get("source_split_manifest_sha256") != EXPECTED_SPLIT_SHA256:
        raise ValueError("teacher ancestry is not bound to the frozen fold-1 split")
    split_manifest_path = data_root / TEACHER_SPLIT_MANIFEST_RELATIVE
    if file_sha256(split_manifest_path) != EXPECTED_SPLIT_SHA256:
        raise ValueError("mounted teacher split manifest differs from the frozen fold-1 source")
    split_manifest = json.loads(split_manifest_path.read_text())
    fit_pools = split_manifest.get("fit", {})
    evaluation_groups = list(map(int, split_manifest.get("evaluation", {}).get("valid", [])))
    if set(fit_pools) != {"train", "valid"} or any(
        not isinstance(fit_pools[name], list) for name in ("train", "valid")
    ):
        raise ValueError("frozen teacher split manifest has an invalid fitting population")
    if metadata.get("target_contract") != {
        "family": "strong",
        "formula": "float32(0.8125 * float32_mean(three_epoch2_predictions))",
        "coefficient_multiplier": TARGET_MULTIPLIER,
        "teacher_mean": "unweighted arithmetic mean in float32",
        "teacher_epoch": 2,
        "clipping": "none during cache construction",
    }:
        raise ValueError("all-data homogeneous target semantics differ from H390")
    output_bytes, safety_reserve_bytes, admitted_bytes = cache_disk_requirement(10_606 + 1_873)
    if (
        metadata.get("output_bytes") != output_bytes
        or metadata.get("disk_safety_reserve_bytes") != safety_reserve_bytes
        or metadata.get("estimated_bytes") != admitted_bytes
    ):
        raise ValueError("all-data cache disk estimate omits the H390 safety reserve")
    if (
        not isinstance(metadata.get("builder_source_manifest_sha256"), str)
        or len(metadata["builder_source_manifest_sha256"]) != 64
    ):
        raise ValueError("all-data cache is not bound to the frozen H390 builder source")
    if (
        not isinstance(metadata.get("packet_manifest_sha256"), str)
        or len(metadata["packet_manifest_sha256"]) != 64
    ):
        raise ValueError("all-data cache is not bound to the frozen H390 packet")

    groups = {name: target_row_groups(data_root, name) for name in ("train", "valid")}
    if len(groups["train"]) != 10_606 or len(groups["valid"]) != 1_873:
        raise ValueError("canonical supplied target populations differ from H390")
    membership = metadata.get("membership")
    if not isinstance(membership, dict):
        raise ValueError("all-data final cache lacks membership metadata")
    for split, expected in groups.items():
        record = membership.get(split)
        if not isinstance(record, dict):
            raise ValueError(f"all-data cache lacks {split} membership")
        if (
            record.get("source") != f"datasets/{split}.parquet"
            or record.get("count") != len(expected)
            or record.get("row_groups") != expected
            or record.get("row_groups_sha256") != row_groups_sha256(expected)
            or record.get("source_qualified_sha256")
            != source_qualified_groups_sha256(f"datasets/{split}.parquet", expected)
        ):
            raise ValueError(f"all-data cache {split} membership differs from target index")
    if membership.get("evaluation_row_groups") != []:
        raise ValueError("final all-data cache cannot declare evaluation row groups")
    if membership.get("source_qualified_groups_sha256") != {
        split: source_qualified_groups_sha256(f"datasets/{split}.parquet", rows)
        for split, rows in groups.items()
    }:
        raise ValueError("all-data cache source-qualified population identities differ")

    source_records = metadata.get("sources")
    if (
        not isinstance(source_records, list)
        or any(not isinstance(item, dict) for item in source_records)
        or [item.get("run") for item in source_records] != list(TEACHERS)
    ):
        raise ValueError("all-data cache teacher set or order differs from H390")
    for record in source_records:
        expected = TEACHERS.get(str(record.get("run")))
        if (
            expected is None
            or record.get("seed") != expected["seed"]
            or record.get("epoch") != 2
            or record.get("checkpoint_sha256") != expected["checkpoint_sha256"]
            or record.get("provenance_sha256") != expected["provenance_sha256"]
            or record.get("propensity_checkpoint_sha256") != EXPECTED_PROPENSITY_SHA256
            or record.get("split_manifest_sha256") != EXPECTED_SPLIT_SHA256
        ):
            raise ValueError(f"all-data cache has invalid teacher lineage: {record.get('run')}")
        expected_fit = {
            split: {
                "count": len(fit_pools[split]),
                "row_groups_sha256": row_groups_sha256(sorted(map(int, fit_pools[split]))),
            }
            for split in ("train", "valid")
        }
        expected_predictions = {
            split: {
                "rows": len(groups[split]),
                "row_groups_sha256": row_groups_sha256(groups[split]),
            }
            for split in ("train", "valid")
        }
        prediction_records = record.get("predictions")
        if (
            record.get("fit_membership") != expected_fit
            or record.get("evaluation_annotation")
            != {
                "count": len(evaluation_groups),
                "classification": "teacher clean-fold evaluation pool excluded from teacher fit",
            }
            or not isinstance(prediction_records, dict)
            or set(prediction_records) != {"train", "valid"}
        ):
            raise ValueError(f"all-data cache teacher fit population differs: {record.get('run')}")
        for split, expected_prediction in expected_predictions.items():
            prediction = prediction_records[split]
            if (
                not isinstance(prediction, dict)
                or any(prediction.get(key) != value for key, value in expected_prediction.items())
                or not isinstance(prediction.get("prediction_sha256"), str)
                or len(prediction["prediction_sha256"]) != 64
                or not isinstance(prediction.get("prediction_sidecar_sha256"), str)
                or len(prediction["prediction_sidecar_sha256"]) != 64
            ):
                raise ValueError(
                    f"all-data cache teacher export lineage differs: {record.get('run')} {split}"
                )

    parent_records = metadata.get("parents")
    if not isinstance(parent_records, list) or len(parent_records) != len(PARENTS):
        raise ValueError("all-data cache lacks both declared student parents")
    expected_parents = {item["run"] for item in PARENTS.values()}
    if {item.get("run") for item in parent_records if isinstance(item, dict)} != expected_parents:
        raise ValueError("all-data cache parent lineage differs from H390")
    for record in parent_records:
        spec = next(item for item in PARENTS.values() if item["run"] == record.get("run"))
        if (
            record.get("checkpoint_epoch") != 3
            or record.get("checkpoint_sha256") != spec["checkpoint_sha256"]
            or record.get("provenance_sha256") != spec["provenance_sha256"]
            or record.get("fit_membership")
            != {
                name: {
                    "count": len(groups[name]),
                    "row_groups_sha256": row_groups_sha256(groups[name]),
                }
                for name in ("train", "valid")
            }
            or record.get("evaluation_annotation", {}).get("classification")
            != "reporting_population_not_heldout"
            or record.get("evaluation_annotation", {}).get("count") != 600
            or record.get("evaluation_annotation", {}).get("row_groups_sha256")
            != spec["reporting_groups_sha256"]
            or record.get("evaluation_annotation", {}).get("no_eval_during_fit") is not True
        ):
            raise ValueError(f"all-data cache parent lineage differs: {record.get('run')}")

    files = metadata.get("files")
    expected_keys = {"main_homogeneous_train", "main_homogeneous_valid"}
    if not isinstance(files, dict) or set(files) != expected_keys:
        raise ValueError("H390 final cache must contain exactly two homogeneous main arrays")
    if verify_final_population_sources(data_root, split_manifest) != groups:
        raise ValueError(
            "all-data row groups differ between target indices and frozen source identity"
        )
    verified: dict[str, str] = {}
    for split in ("train", "valid"):
        key = f"main_homogeneous_{split}"
        entry = files[key]
        if not isinstance(entry, dict):
            raise ValueError(f"invalid cache entry: {key}")
        expected = groups[split]
        if entry.get("row_groups") != expected or entry.get(
            "row_groups_sha256"
        ) != row_groups_sha256(expected):
            raise ValueError(f"cache row order differs from target index: {key}")
        path = _safe_array_path(cache_root, entry, key)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        shape = (len(expected), SEQUENCE_ROWS, TARGET_WIDTH)
        if array.shape != shape or array.dtype != np.float32:
            raise ValueError(f"cache shape/dtype differs for {key}: {array.shape}, {array.dtype}")
        if entry.get("shape") != list(shape) or entry.get("dtype") != "float32":
            raise ValueError(f"cache manifest shape/dtype differs for {key}")
        digest = file_sha256(path)
        if digest != entry.get("sha256") or path.stat().st_size != entry.get("bytes"):
            raise ValueError(f"cache bytes disagree with metadata for {key}")
        for start in range(0, len(array), 64):
            if not np.isfinite(array[start : start + 64]).all():
                raise ValueError(f"cache contains nonfinite targets: {key}")
        verified[key] = digest

    return {
        "metadata": metadata,
        "metadata_sha256": file_sha256(metadata_path),
        "verified_files": verified,
    }
