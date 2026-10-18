"""Frozen source, data membership, and inherited-normalization contract."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from kalman_features import RAW_COLUMNS, KalmanConfig
from kalman_normalization import prior_from_parent_normalization

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
SCRATCH = Path("/home/vlad/development/wunder/research_tmp/kalman-online-20261018")
SOURCE_ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT / "research/campaign_2026-10-18/online_features_r1"
SPLIT = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
DATA = ROOT / "datasets/valid.parquet"
CHECKPOINT = ROOT / "artifacts/runs/warm_mechanism_f0_s3_c_r3/model_ep1.pt"
NORMALIZATION = CAMPAIGN / "kalman_parent_normalization.json"
PREFLIGHT = SCRATCH / "kalman_gru_probe_preflight_r2.json"
OUTPUT = SCRATCH / "kalman_gru_probe_r2"
CHECKPOINT_SHA256 = "78357eca2aec188e227544d46887935265720c35e99a1437269b952d01eb88c2"
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
GROUPS = (4, 5, 7, 9, 12, 18, 19, 22, 26, 35, 39, 41, 42, 50, 51, 53)
SEQUENCE_IDS = (
    1990332,
    2308749,
    6746551,
    9147005,
    11423352,
    24654613,
    25927382,
    28399078,
    29519713,
    38268719,
    45444605,
    47001707,
    48100848,
    51305808,
    51608922,
    52382536,
)
ARMS = ("raw", "random_walk_posterior", "local_trend_posterior")
SOURCE_FILES = (
    "src/wnn/data.py",
    "src/wnn/features.py",
    "src/wnn/training/environment.py",
    "src/wnn/training/evaluation.py",
    "src/wnn/training/forward.py",
    "src/wnn/training/inputs.py",
    "src/wnn/training/models.py",
    "src/wnn/nn.py",
)


def sha256_file(path: Path) -> str:
    """Hash a file incrementally without holding its contents in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cgroup_receipt() -> dict[str, object]:
    """Capture live cgroup memory, CPU, and OOM counters before scope teardown."""
    memberships = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    unified = next((line.split("::", 1)[1] for line in memberships if line.startswith("0::")), None)
    if unified is None:
        return {"status": "unavailable", "reason": "no unified cgroup membership"}
    directory = Path("/sys/fs/cgroup") / unified.lstrip("/")
    names = (
        "memory.current",
        "memory.peak",
        "memory.max",
        "memory.events",
        "memory.swap.current",
        "cpu.max",
    )
    values: dict[str, object] = {}
    for name in names:
        try:
            content = (directory / name).read_text(encoding="utf-8").strip()
        except OSError as error:
            values[name] = {"unavailable": str(error)}
        else:
            values[name] = (
                dict(line.split() for line in content.splitlines())
                if name == "memory.events"
                else content
            )
    return {"status": "read", "path": str(directory), "values": values}


def source_hashes() -> dict[str, str]:
    """Return exact hashes for probe code and reused input/model components."""
    local_files = (
        "frozen_gru_probe.py",
        "probe_contract.py",
        "kalman_features.py",
        "kalman_normalization.py",
    )
    paths = {name: SOURCE_ROOT / name for name in local_files}
    paths.update({name: ROOT / name for name in SOURCE_FILES})
    return {name: sha256_file(path) for name, path in paths.items()}


def frozen_membership() -> dict[str, object]:
    """Check source-qualified membership, parquet identity, and input columns."""
    import pyarrow.parquet as pq

    split_bytes = SPLIT.read_bytes()
    split = json.loads(split_bytes)
    if hashlib.sha256(split_bytes).hexdigest() != SPLIT_SHA256:
        raise ValueError("fold-0 split manifest changed")
    groups = tuple(map(int, split["evaluation"]["valid"][: len(GROUPS)]))
    if groups != GROUPS:
        raise ValueError("first-16 evaluation group membership changed")
    group_sha = hashlib.sha256(json.dumps(list(groups), separators=(",", ":")).encode()).hexdigest()
    expected_sha = "ef3581b2aac87706237f40562b1350c5f429c275caf454c6e29bb591e97923bf"
    if group_sha != expected_sha:
        raise ValueError("frozen group-list digest changed")
    sequence_map = {
        int(item["row_group"]): int(item["seq_ix"]) for item in split["sequences"]["valid"]
    }
    sequence_ids = tuple(sequence_map[group] for group in groups)
    if sequence_ids != SEQUENCE_IDS:
        raise ValueError("source-qualified sequence IDs changed")
    source = split["sources"]["datasets/valid.parquet"]
    stat = DATA.stat()
    if stat.st_size != int(source["bytes"]) or stat.st_mtime_ns != int(source["mtime_ns"]):
        raise ValueError("valid parquet identity differs from the frozen split")
    parquet = pq.ParquetFile(DATA)
    schema_names = parquet.schema_arrow.names
    raw_names = schema_names[4:116]
    selected_names = [raw_names[index] for index in RAW_COLUMNS]
    if selected_names != ["i0_p0", "i0_p3", "i1_p0", "i1_p3"]:
        raise ValueError("selected raw column names or order changed")
    if parquet.num_row_groups <= max(GROUPS):
        raise ValueError("valid parquet does not contain the frozen row groups")
    return {
        "split_sha256": SPLIT_SHA256,
        "group_list_sha256": group_sha,
        "groups": list(groups),
        "sequence_ids": list(sequence_ids),
        "row_groups": len(GROUPS),
        "rows_per_group": 20_000,
        "selected_raw_columns": list(RAW_COLUMNS),
        "selected_raw_names": selected_names,
        "runtime_payload_fields": ["seq_ix", *raw_names],
        "payload_access": "SequenceFile.read(targets=False, mask=False)",
        "valid_parquet_bytes": stat.st_size,
        "valid_parquet_mtime_ns": stat.st_mtime_ns,
        "valid_parquet_row_groups": parquet.num_row_groups,
        "targets_and_scoring_masks_requested": False,
    }


def read_recipe_inputs() -> tuple[dict[str, object], str]:
    """Validate inherited normalization ancestry and return its pinned digest."""
    payload = json.loads(NORMALIZATION.read_text(encoding="utf-8"))
    if payload.get("source_checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("normalization prior names a different parent checkpoint")
    if payload.get("source_args_split_manifest_sha256") != SPLIT_SHA256:
        raise ValueError("normalization prior names a different split manifest")
    prior_from_parent_normalization(payload)
    encoded = NORMALIZATION.read_bytes()
    return payload, hashlib.sha256(encoded).hexdigest()


def write_preflight() -> dict[str, object]:
    """Freeze source, checkpoint, normalization, and membership identities."""
    membership = frozen_membership()
    normalization, normalization_sha = read_recipe_inputs()
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if "torch" in sys.modules:
        raise AssertionError("CPU preflight unexpectedly imported Torch")
    receipt: dict[str, object] = {
        "schema": "kalman-frozen-gru-probe-preflight.v1",
        "status": "source and membership frozen; no model or payload loaded",
        "source_hashes": source_hashes(),
        "parent_checkpoint": {
            "path": str(CHECKPOINT),
            "sha256": CHECKPOINT_SHA256,
            "size_bytes": CHECKPOINT.stat().st_size,
            "loaded": False,
        },
        "normalization": {
            "path": str(NORMALIZATION),
            "sha256": normalization_sha,
            "mode": "inherited-checkpoint-buffer; no evaluation fitting",
            "selected_mean": normalization["mean"],
            "selected_std": normalization["std"],
            "source_full_mean_sha256": normalization["source_full_mean_sha256"],
            "source_full_std_sha256": normalization["source_full_std_sha256"],
        },
        "membership": membership,
        "arms": list(ARMS),
        "chunk_rows": 4_000,
        "batch_size": 1,
        "dtype": "float32",
        "kalman_config": KalmanConfig().__dict__,
        "torch_imported": False,
        "gpu_used": False,
        "cgroup": cgroup_receipt(),
    }
    PREFLIGHT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def verify_preflight() -> dict[str, object]:
    """Reject execution if any frozen source or membership differs."""
    receipt = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if receipt.get("schema") != "kalman-frozen-gru-probe-preflight.v1":
        raise ValueError("missing or unsupported CPU preflight receipt")
    if receipt["source_hashes"] != source_hashes():
        raise ValueError("probe or reused training source changed after preflight")
    if receipt["membership"] != frozen_membership():
        raise ValueError("source-qualified inputs changed after preflight")
    _, normalization_sha = read_recipe_inputs()
    if receipt["normalization"]["sha256"] != normalization_sha:
        raise ValueError("parent normalization changed after preflight")
    return receipt
