"""Read the final frozen artifacts without importing the production grader."""

from __future__ import annotations

import hashlib
import json
import mmap
import re
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from attrs import define

from scripts.independent_heterogeneous_numeric import MIXTURES

# Heterogeneous report/source metadata is checked at this deserialization boundary.
JsonObject = dict[str, Any]
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
VALID_SHA256 = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
PROBE_SCRIPT_SHA256 = "514d3aae37d828f115396cedea37f4afd8506519da998225e7d74638b9c79e1d"
GALAM_CHECKPOINT_SHA256 = "c825ac5221490109844b090a0699b30225ac2a211287b7f8abf409b2a69f05d9"
ROLES = {
    "w18": ("tl3c_w18final_distilled_s5", 1),
    "tiedlead_s5": ("tl3c_fttall_tlead_s5", 3),
    "tiedlead_s6": ("tl3c_fttall_tlead_s6", 3),
    "fbc": ("tl3c_fbc1k_s5", 3),
    "fin256_s3": ("tl3c_fin256_s3", 3),
    "fin256_s4": ("tl3c_fin256_s4", 3),
    "lstm": ("het6_lstm_s1201", 3),
    "tcn": ("het6_tcn_s1202", 3),
    "control_s1301": ("het6_control_s1301", 3),
    "control_s1302": ("het6_control_s1302", 3),
    "hetero_s1301": ("het6_heterogeneous_s1301", 3),
    "hetero_s1302": ("het6_heterogeneous_s1302", 3),
}


def read_json(path: Path, limit: int = 2 << 20) -> JsonObject:
    """Read a bounded JSON object; never deserialize model objects."""
    if not 0 < path.stat().st_size <= limit:
        raise ValueError(f"JSON bound exceeded: {path}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def sha256(path: Path) -> str:
    """Hash a physical evidence file with bounded read buffers."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def pin(value: JsonObject, base: Path, evidence: dict[str, str]) -> Path:
    """Verify the manifest's actual file bytes and retain them for final rechecks."""
    path = Path(value["path"])
    path = path if path.is_absolute() else base / path
    expected = value.get("sha256")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("invalid evidence SHA256")
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"evidence file differs: {path}")
    evidence[str(path.resolve())] = expected
    return path.resolve()


def integer_members(value: object) -> list[int]:
    """Require physical integer group identifiers with no duplicates."""
    if not isinstance(value, list) or any(type(v) is not int or v < 0 for v in value):
        raise ValueError("physical row groups must be nonnegative integers")
    if len(value) != len(set(value)):
        raise ValueError("physical row groups must not repeat")
    return value


def selected_layout(
    parquet: pq.ParquetFile, groups: list[int]
) -> tuple[dict[int, np.ndarray], dict[int, tuple[int, int]], str]:
    """Reconstruct actual scoring positions and the original scored-probe ordering."""
    positions, offsets, count = {}, {}, 0
    digest = hashlib.sha256()
    for group in groups:
        table = parquet.read_row_group(group, columns=["is_scored"], use_threads=False)
        if table.num_rows != 20000:
            raise ValueError("official source group is not a complete sequence")
        mask = table["is_scored"].to_numpy().astype(bool, copy=True)
        mask[:99] = False
        chosen = np.flatnonzero(mask).astype("<i4")
        digest.update(np.asarray(("valid", group), dtype="U16").tobytes())
        digest.update(chosen.tobytes())
        positions[group], offsets[group] = chosen, (count, count + len(chosen))
        count += len(chosen)
    if count != 3528273:
        raise ValueError("official scoring mask does not select 3528273 rows")
    return positions, offsets, digest.hexdigest()


@define
class Sources:
    """Keep verified memory-mapped banks and source-qualified row mappings."""

    parquet: pq.ParquetFile
    groups: list[int]
    positions: dict[int, np.ndarray]
    offsets: dict[int, tuple[int, int]]
    banks: dict[str, np.ndarray]
    bank_positions: dict[str, dict[int, int]]
    evidence: dict[str, str]

    def selected(self, group: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Read only scored outputs; never invent observations at unscored positions."""
        rows = self.positions[group]
        table = self.parquet.read_row_group(group, columns=["t0", "t1"], use_threads=False)
        y = np.column_stack([table[k].to_numpy() for k in ("t0", "t1")]).astype(np.float32)
        outputs = {
            name: np.asarray(bank[self.bank_positions[name][group]])[rows]
            for name, bank in self.banks.items()
            if name != "galam"
        }
        begin, end = self.offsets[group]
        outputs["galam"] = np.asarray(self.banks["galam"][begin:end])
        return y[rows], outputs

    def release_mapped_pages(self) -> None:
        """Drop only this reader's clean mapped pages, without clearing shared file caches."""
        advice = getattr(mmap, "MADV_DONTNEED", None)
        if advice is not None:
            for bank in self.banks.values():
                backing = getattr(bank, "_mmap", None)
                if backing is not None and hasattr(backing, "madvise"):
                    backing.madvise(advice)


def legacy_source(
    name: str,
    record: JsonObject,
    base: Path,
    train: list[int],
    valid: list[int],
    evidence: dict[str, str],
) -> tuple[np.ndarray, dict[int, int]]:
    """Join a full bank by its original sidecar group identifiers and immutable pins."""
    run, epoch = ROLES[name]
    if (record.get("kind"), record.get("run"), record.get("epoch")) != ("legacy_bank", run, epoch):
        raise ValueError(f"wrong final source role: {name}")
    path = pin(record["predictions"], base, evidence)
    side = read_json(pin(record["metadata"], base, evidence))
    provenance = read_json(pin(record["provenance"], base, evidence))
    pin(record["checkpoint"], base, evidence)
    rows = integer_members(side.get("row_groups"))
    args, fitting = provenance["args"], provenance["fit_row_groups"]
    if (
        set(rows) != set(valid)
        or set(integer_members(fitting["train"])) != set(train)
        or integer_members(fitting["valid"])
        or set(integer_members(provenance["evaluation_row_groups"])) != set(valid)
        or provenance.get("split_manifest_sha256") != SPLIT_SHA256
        or args.get("name") != run
        or args.get("checkpoint_policy") != "fixed"
        or args.get("checkpoint_epoch") != epoch
        or provenance.get("selected_epoch") != epoch
    ):
        raise ValueError(f"wrong frozen final source membership/selection: {name}")
    expected = {
        "run": run,
        "split": "valid",
        "epoch": epoch,
        "inference_precision": "fp32",
        "split_manifest_sha256": SPLIT_SHA256,
        "prediction_sha256": record["predictions"]["sha256"],
        "checkpoint_sha256": record["checkpoint"]["sha256"],
        "run_provenance_sha256": record["provenance"]["sha256"],
    }
    if any(side.get(k) != v for k, v in expected.items()):
        raise ValueError(f"wrong actual bank sidecar: {name}")
    bank = np.load(path, mmap_mode="r", allow_pickle=False)
    if bank.dtype != np.float32 or bank.shape != (1873, 20000, 2):
        raise ValueError("complete FP32 source bank required")
    return bank, {group: i for i, group in enumerate(rows)}


def galam_source(
    record: JsonObject, base: Path, row_hash: str, evidence: dict[str, str]
) -> np.ndarray:
    """Read the original raw scored-only bank without a production adapter."""
    if (record.get("kind"), record.get("run"), record.get("epoch")) != (
        "galam_scored_probe",
        "g2-galam-lstm128",
        2,
    ):
        raise ValueError("the original human-epoch3 GALAM scored probe is required")
    objects = {
        key: read_json(pin(record[key], base, evidence))
        for key in ("metadata", "plan", "result", "epoch_index")
    }
    checkpoint = pin(record["checkpoint"], base, evidence)
    receipt, plan, result, index = (
        objects[k] for k in ("metadata", "plan", "result", "epoch_index")
    )
    endpoint = receipt["endpoint"]
    if (
        record["checkpoint"]["sha256"] != GALAM_CHECKPOINT_SHA256
        or endpoint.get("checkpoint_sha256") != GALAM_CHECKPOINT_SHA256
        or endpoint.get("epoch") != 3
        or endpoint.get("cell") != "g2-galam-lstm128"
        or endpoint not in plan.get("endpoints", [])
        or result.get("plan_sha256") != record["plan"]["sha256"]
        or result.get("script_sha256") != PROBE_SCRIPT_SHA256
        or receipt.get("row_order_sha256") != row_hash
        or result.get("row_order_sha256") != row_hash
        or receipt.get("scored_rows") != 3528273
        or result.get("scored_rows") != 3528273
        or plan.get("split_sha256") != SPLIT_SHA256
    ):
        raise ValueError("original GALAM artifact/mask provenance differs")
    index_body = {k: v for k, v in index.items() if k != "index_sha256"}
    canonical = hashlib.sha256(
        json.dumps(
            index_body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()
    if canonical != record.get("epoch_index_sha256") or index.get("index_sha256") != canonical:
        raise ValueError("original native epoch-index identity differs")
    matching = [e for e in index["entries"] if e["epoch"] == 2]
    if len(matching) != 1 or matching[0]["checkpoint_sha256"] != evidence[str(checkpoint)]:
        raise ValueError("GALAM checkpoint is not in the pinned native epoch index")
    path = pin(record["predictions"], base, evidence)
    if receipt.get("prediction_sha256") != record["predictions"]["sha256"]:
        raise ValueError("GALAM prediction bytes differ from original receipt")
    bank = np.load(path, mmap_mode="r", allow_pickle=False)
    if bank.dtype != np.float32 or bank.shape != (3528273, 2):
        raise ValueError("GALAM bank must contain exactly the observed raw scoring rows")
    return bank


def load_sources(manifest_path: Path, report_path: Path) -> tuple[Sources, JsonObject]:
    """Freeze all thirteen singletons and twelve mixtures before two-pass rescoring."""
    manifest, report = read_json(manifest_path), read_json(report_path)
    if (
        report.get("status") != "complete"
        or report.get("baseline") != "TL3"
        or report.get("input_manifest") != manifest
        or report.get("input_manifest_sha256") != sha256(manifest_path)
        or report.get("valid_sequences") != 1873
        or report.get("selected_rows") != 3528273
        or report.get("split_sha256") != SPLIT_SHA256
    ):
        raise ValueError(
            "a complete, unchanged final report and its exact input manifest are required"
        )
    expected = {name: dict(terms) for name, terms in MIXTURES.items()}
    rows = report.get("rows", [])
    if (
        manifest.get("ensembles") != expected
        or [r.get("name") for r in rows] != list(expected)
        or any(
            r.get("status") != "complete" or r.get("weights") != expected[r["name"]] for r in rows
        )
        or set(manifest.get("members", {})) != set(ROLES) | {"galam"}
        or set(report.get("member_scores", {})) != set(ROLES) | {"galam"}
    ):
        raise ValueError("the thirteen final models and twelve frozen mixture recipes must match")
    base, evidence = manifest_path.resolve().parent, {}
    split_path = pin(manifest["split"], base, evidence)
    if manifest["split"]["sha256"] != SPLIT_SHA256:
        raise ValueError("official split identity differs")
    split = read_json(split_path, 64 << 20)
    train, valid = (
        integer_members(split["fit"]["train"]),
        integer_members(split["evaluation"]["valid"]),
    )
    if len(train) != 10325 or split["fit"]["valid"] or len(valid) != 1873:
        raise ValueError("chronological primary membership differs")
    validation = pin(manifest["validation"], base, evidence)
    if manifest["validation"]["sha256"] != VALID_SHA256:
        raise ValueError("official validation source identity differs")
    parquet = pq.ParquetFile(validation)
    if parquet.num_row_groups != 1873 or set(valid) != set(range(1873)):
        raise ValueError("official-valid whole-sequence coverage differs")
    positions, offsets, row_hash = selected_layout(parquet, valid)
    banks, bank_positions = {}, {}
    for name in ROLES:
        banks[name], bank_positions[name] = legacy_source(
            name, manifest["members"][name], base, train, valid, evidence
        )
    banks["galam"] = galam_source(manifest["members"]["galam"], base, row_hash, evidence)
    evidence[str(manifest_path.resolve())], evidence[str(report_path.resolve())] = (
        sha256(manifest_path),
        sha256(report_path),
    )
    return Sources(parquet, valid, positions, offsets, banks, bank_positions, evidence), report
