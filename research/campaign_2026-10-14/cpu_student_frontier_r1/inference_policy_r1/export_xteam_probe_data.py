"""Export bounded, hash-bound probes from the immutable original xteam_h4 assets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import struct
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pyarrow.parquet as pq

MAIN_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
ARCHIVE_SHA256 = "f1635d14c0e1b47848b5caeb2ec4ad9362a85800f1b2e2df80702127d3867b27"
ORIGINAL_PACKAGE = MAIN_ROOT / "artifacts/team_models_20261013/xteam_h4"
ORIGINAL_ARCHIVE = MAIN_ROOT / "artifacts/team_models_20261013/xteam_h4.zip"
VALID_PATH = MAIN_ROOT / "datasets/valid.parquet"
SCORE_PATH = MAIN_ROOT / "research/campaign_2026-10-13/team_models_r1/score_archive.py"


def digest(path: Path) -> str:
    """Hash a file without loading the whole payload into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_score_adapter() -> ModuleType:
    """Load the campaign's official feature and sequence validator from MAIN."""
    sys.path.insert(0, str(MAIN_ROOT))
    spec = importlib.util.spec_from_file_location("cpu_policy_score_archive", SCORE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load score adapter: {SCORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_member(weights_path: Path, output_path: Path) -> dict[str, object]:
    """Write the exact original float32 tensors in the native bench_int8 WNNWTS01 format."""
    with np.load(weights_path, allow_pickle=False) as archive:
        state = {key: np.ascontiguousarray(archive[key], dtype="<f4") for key in archive.files}
    layers = 0
    while f"gru.weight_ih_l{layers}" in state:
        layers += 1
    if layers == 0:
        raise ValueError(f"no GRU layers in {weights_path}")
    dims = (
        int(state["mean"].shape[0]),
        int(state["inp.bias"].shape[0]),
        layers,
        int(state["head.0.bias"].shape[0]),
        int(state["head.2.bias"].shape[0]),
    )
    if dims[4] != 2:
        raise ValueError(f"xteam_h4 must have two outputs, got {dims[4]}")
    tensor_names = ["mean", "std", "inp.weight", "inp.bias"]
    for layer in range(layers):
        tensor_names.extend(
            f"gru.{kind}_l{layer}" for kind in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
        )
    tensor_names.extend(("head.0.weight", "head.0.bias", "head.2.weight", "head.2.bias"))
    with output_path.open("wb") as stream:
        stream.write(b"WNNWTS01" + struct.pack("<5I", *dims))
        for name in tensor_names:
            stream.write(state[name].tobytes())
    return {
        "source_file": weights_path.name,
        "source_sha256": digest(weights_path),
        "output_file": output_path.name,
        "output_sha256": digest(output_path),
        "output_bytes": output_path.stat().st_size,
        "dimensions": list(dims),
        "tensor_names": tensor_names,
    }


def export_rows(score: ModuleType, group: int, output_path: Path) -> dict[str, object]:
    """Export one complete raw-feature sequence for native cell and callback probes."""
    table = pq.ParquetFile(VALID_PATH).read_row_group(group, use_threads=False)
    sequence, need = score.validate_sequence(table)
    features = np.column_stack([table[name].to_numpy() for name in score.FEATURE_COLUMNS]).astype(
        np.float32, copy=False
    )
    features = np.ascontiguousarray(features)
    if features.shape != (20_000, 112) or not np.isfinite(features).all():
        raise ValueError(f"row group {group} has invalid raw features: {features.shape}")
    if need.shape != (20_000,) or np.any(need[:99]) or not np.all(need[99:]):
        raise ValueError(f"row group {group} has an invalid causal prediction mask")
    features.astype("<f4", copy=False).tofile(output_path)
    return {
        "row_group": group,
        "sequence_id": int(sequence),
        "rows": int(features.shape[0]),
        "features_sha256": hashlib.sha256(memoryview(features).cast("B")).hexdigest(),
        "need_mask_sha256": hashlib.sha256(
            memoryview(np.ascontiguousarray(need, dtype=np.bool_)).cast("B")
        ).hexdigest(),
        "output_file": output_path.name,
        "output_sha256": digest(output_path),
        "output_bytes": output_path.stat().st_size,
        "scope": "complete supplied-validation sequence used only for fixed runtime/parity probes",
    }


def parse_args() -> argparse.Namespace:
    """Parse the fixed source paths and requested whole-sequence groups."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--groups", type=int, nargs="+", default=[0, 1])
    return parser.parse_args()


def main() -> None:
    """Validate immutable inputs and write the bounded probe data and provenance receipt."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if digest(ORIGINAL_ARCHIVE) != ARCHIVE_SHA256:
        raise ValueError("original xteam_h4 archive hash differs from the fixed receipt")
    args.out.mkdir(parents=True)
    (args.out / "weights").mkdir()
    (args.out / "rows").mkdir()
    score = load_score_adapter()
    members = json.loads((ORIGINAL_PACKAGE / "members.json").read_text())
    weight_records = []
    for index, member in enumerate(members):
        source = ORIGINAL_PACKAGE / member["weights"]
        target = args.out / "weights" / f"member_{index}.wbin"
        weight_records.append(export_member(source, target))
    row_records = []
    for group in args.groups:
        target = args.out / "rows" / f"group_{group}.f32"
        row_records.append(export_rows(score, group, target))
    receipt = {
        "source_archive": str(ORIGINAL_ARCHIVE),
        "source_archive_sha256": ARCHIVE_SHA256,
        "source_callback_sha256": digest(ORIGINAL_PACKAGE / "solution.py"),
        "source_precision": (ORIGINAL_PACKAGE / "precision.txt").read_text().strip(),
        "validation_file": str(VALID_PATH),
        "validation_file_sha256": digest(VALID_PATH),
        "weights": weight_records,
        "rows": row_records,
        "claim_scope": "bounded native and callback timing/parity only; no quality grade",
    }
    (args.out / "EXPORT_MANIFEST_R1.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    logging.info("wrote %s", args.out / "EXPORT_MANIFEST_R1.json")


if __name__ == "__main__":
    main()
