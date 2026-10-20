"""Fail-closed preflight and post-export audit for the frozen F0 score plan.

Preflight pins sources, inputs, reused caches, and absent outputs. Post-export
binds checkpoints, provenance, sidecars, and finite float32 predictions into a
create-only receipt. This CLI imports NumPy and the standard library only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SHAPE = (1873, 20_000, 2)
NBYTES = 299_680_128
GROUPS = list(range(1873))


def _json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pin(path: Path, digest: str, size: int | None = None) -> dict[str, Any]:
    """Check and return a pinned file identity."""
    actual_size, actual_sha = path.stat().st_size, _sha(path)
    if actual_sha != digest or (size is not None and actual_size != size):
        raise ValueError(f"file identity mismatch: {path}")
    return {"path": str(path), "bytes": actual_size, "sha256": actual_sha}


def _prediction(path: Path, scan: bool = False) -> str | None:
    """Check NPY geometry and optionally hash finite values row by row."""
    if path.stat().st_size != NBYTES:
        raise ValueError(f"prediction size mismatch: {path}")
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        reader = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
        }.get(version)
        if reader is None:
            raise ValueError(f"unsupported NPY version {version}: {path}")
        shape, fortran, dtype = reader(stream)
        offset = stream.tell()
        if tuple(shape) != SHAPE or fortran or dtype != np.dtype(np.float32):
            raise ValueError(f"prediction shape/order/dtype mismatch: {path}")
        if not scan:
            return None
        stream.seek(0)
        header = stream.read(offset)
        if len(header) != offset:
            raise ValueError(f"truncated NPY header: {path}")
        digest = hashlib.sha256(header)
        row_bytes = SHAPE[1] * SHAPE[2] * 4
        for row in range(SHAPE[0]):
            block = stream.read(row_bytes)
            values = np.frombuffer(block, dtype=dtype)
            if len(block) != row_bytes or values.size != SHAPE[1] * SHAPE[2]:
                raise ValueError(f"truncated prediction row {row}: {path}")
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite predictions at row {row}: {path}")
            digest.update(block)
        if stream.read(1):
            raise ValueError(f"trailing prediction bytes: {path}")
    return digest.hexdigest()


def _footer(path: Path) -> tuple[int, str]:
    """Hash a Parquet footer and trailer without reading data pages."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        if size < 8:
            raise ValueError(f"Parquet file is too short: {path}")
        stream.seek(-8, 2)
        trailer = stream.read(8)
        length = int.from_bytes(trailer[:4], "little")
        if trailer[4:] != b"PAR1" or length > size - 8:
            raise ValueError(f"invalid Parquet footer: {path}")
        stream.seek(-8 - length, 2)
        digest, left = hashlib.sha256(), length + 8
        while left:
            block = stream.read(min(left, 1 << 20))
            if not block:
                raise ValueError(f"truncated Parquet footer: {path}")
            digest.update(block)
            left -= len(block)
    return length, digest.hexdigest()


def _sources(plan: dict[str, Any], eval_path: Path) -> list[dict[str, Any]]:
    """Check pinned predictor, scorer, imports, and frozen manifests."""
    pins, checked = plan["source_pins"], []
    root, predictor, scorer = (
        Path(plan["source_pins"]["project_root"]),
        pins["predictor"],
        pins["screen_scorer"],
    )
    worktree = Path(predictor["path"]).parents[1]
    files = [(Path(predictor["path"]), predictor["sha256"])]
    files += [(worktree / name, digest) for name, digest in predictor["imports_sha256"].items()]
    files += [
        (Path(scorer["path"]), scorer["sha256"]),
        (Path(scorer["frozen_component"]["path"]), scorer["frozen_component"]["sha256"]),
    ]
    files += [(root / name, digest) for name, digest in scorer["imports_sha256"].items()]
    checked.extend(_pin(path, digest) for path, digest in files)
    for key, path in (
        ("split_manifest", Path(pins["split_manifest"]["path"])),
        ("frozen_evaluation_manifest", eval_path),
        ("screen_definition", Path(pins["screen_definition"]["path"])),
    ):
        pin = pins[key]
        if path.resolve() != Path(pin["path"]).resolve():
            raise ValueError(f"frozen {key} path mismatch")
        checked.append(_pin(path, pin["sha256"]))
    return checked


def _fold(
    path: Path, digest: str, evaluation: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[int, int]]]:
    """Check F0 source-qualified evaluation and component-safe fit groups."""
    if _sha(path) != digest:
        raise ValueError("fold0 manifest hash mismatch")
    split, rows = _json(path), _json(path)["sequences"]
    if split.get("fold") != 0:
        raise ValueError("score plan requires fold 0")
    components = {
        s: {int(r["row_group"]): int(r["component"]) for r in rows[s]} for s in ("train", "valid")
    }
    scored = evaluation["evaluation_groups"]
    by_row = {
        int(r["row_group"]): int(r["component"]) for r in scored if r.get("source") == "valid"
    }
    expected = set(map(int, split["evaluation"]["valid"]))
    if len(scored) != 468 or len(by_row) != len(scored) or set(by_row) != expected:
        raise ValueError("evaluation manifest must match the 468 valid-source fold rows")
    if any(components["valid"].get(g) != c for g, c in by_row.items()):
        raise ValueError("evaluation component IDs differ from fold0.json")
    eval_components = set(by_row.values())
    for source in ("train", "valid"):
        fitted = set(map(int, split["fit"][source]))
        if (source == "valid" and fitted.intersection(by_row)) or eval_components.intersection(
            components[source][g] for g in fitted
        ):
            raise ValueError("fold fitting membership overlaps scored components")
    return split, components


def _targets(root: Path, evaluation: dict[str, Any]) -> dict[str, Any]:
    """Check target-cache bytes and row index without loading targets or masks."""
    pin = evaluation["data_pins"]["valid_targets"]
    path = root / pin["path"]
    identity = _pin(path, pin["file_sha256_from_prior_frozen_grade"], pin["bytes"])
    with np.load(path, allow_pickle=False) as archive:
        groups, seqs = (
            np.asarray(archive["row_group"], dtype=np.int64),
            np.asarray(archive["seq_ix"], dtype=np.int64),
        )
    if groups.tolist() != GROUPS or groups.shape != seqs.shape:
        raise ValueError("valid_targets row order changed")
    digest = hashlib.sha256()
    for values in (groups, seqs):
        digest.update(np.asarray(values, dtype="<i8").tobytes())
    if digest.hexdigest() != pin["index_sha256"]:
        raise ValueError("valid_targets index digest changed")
    return {**identity, "index_sha256": digest.hexdigest(), "row_groups": [0, 1872, len(groups)]}


def _cache(item: dict[str, Any], split_sha: str, checkpoint_sha: str) -> dict[str, Any]:
    """Check a reused prediction cache, its sidecar, and its checkpoint pin."""
    path, side_path = (
        Path(item["prediction_path"]),
        Path(item["prediction_path"]).with_suffix(".json"),
    )
    prediction = _pin(path, item["prediction_sha256"], item["prediction_bytes"])
    side_identity = _pin(side_path, item["sidecar_sha256"])
    side = _json(side_path)
    if (
        side.get("run") != item["run"]
        or side.get("split") != "valid"
        or side.get("epoch") != item["epoch"]
        or side.get("prediction_sha256") != item["prediction_sha256"]
        or side.get("checkpoint_sha256") != checkpoint_sha
        or side.get("split_manifest_sha256") != split_sha
        or side.get("inference_precision") != item["inference_precision"]
        or side.get("row_groups") != GROUPS
    ):
        raise ValueError(f"reused cache sidecar mismatch: {side_path}")
    if item.get("id") == "parent_cp5" and side.get("campaign_split_manifest_sha256") != split_sha:
        raise ValueError("parent cache campaign split hash mismatch")
    _prediction(path)
    return {"prediction": prediction, "sidecar": side_identity}


def _base(
    plan_path: Path, eval_path: Path, verification: dict[str, Any]
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, dict[int, int]], dict[str, Any]
]:
    """Verify all pinned code/data inputs and reused cache identities."""
    plan, evaluation = _json(plan_path), _json(eval_path)
    sources = _sources(plan, eval_path)
    pins, root = plan["source_pins"], Path(plan["source_pins"]["project_root"])
    split_pin = pins["split_manifest"]
    split_path = Path(split_pin["path"])
    split, components = _fold(split_path, split_pin["sha256"], evaluation)
    valid_input = next(
        row
        for row in verification["inputs"]
        if Path(row["path"]) == root / "datasets/valid.parquet"
    )
    parquet = evaluation["data_pins"]["valid_parquet"]
    parquet_path = root / parquet["path"]
    stat = parquet_path.stat()
    if (
        stat.st_size != parquet["bytes"]
        or stat.st_size != valid_input["bytes"]
        or stat.st_mtime_ns != valid_input["mtime_ns"]
    ):
        raise ValueError("valid.parquet size or frozen mtime changed")
    footer_bytes, footer_sha = _footer(parquet_path)
    if footer_sha != parquet["footer_sha256"] or footer_sha != valid_input["footer_sha256"]:
        raise ValueError("valid.parquet footer hash changed")
    parent = next(x for x in plan["endpoints"] if x["id"] == "parent_cp5")
    old = next(x for x in plan["endpoints"] if x["id"] == "oldz4_cp1_fp32")
    ref = dict(plan["historical_reference_artifacts"][0])
    ref["checkpoint_sha256"] = old["checkpoint_sha256"]
    caches = {
        "parent_cp5": _cache(parent, split_pin["sha256"], parent["checkpoint_sha256"]),
        "oldz4_historical_fp16": _cache(ref, split_pin["sha256"], old["checkpoint_sha256"]),
    }
    root_ids = []
    for item in (parent, old):
        run_root = root / "artifacts" / "runs" / item["run"]
        checkpoint = run_root / f"model_ep{item['epoch']}.pt"
        root_ids.append(_pin(checkpoint, item["checkpoint_sha256"]))
    base = {
        "source_identities": sources,
        "split_manifest": {"path": str(split_path), "sha256": split_pin["sha256"], "fold": 0},
        "inputs": {
            "valid_parquet": {
                "path": str(parquet_path),
                "bytes": parquet["bytes"],
                "mtime_ns": stat.st_mtime_ns,
                "footer_bytes": footer_bytes,
                "footer_sha256": footer_sha,
            },
            "valid_targets": _targets(root, evaluation),
        },
        "prediction_contract": {
            "shape": list(SHAPE),
            "dtype": "float32",
            "bytes": NBYTES,
            "row_groups": [0, 1872, len(GROUPS)],
        },
        "reused_caches": caches,
        "checkpoint_identities": root_ids,
    }
    return plan, evaluation, split, components, base


def _absent(plan: dict[str, Any], receipt: Path, post_receipt: Path) -> list[str]:
    """Require new prediction, temporary, sidecar, seqsum, and receipt paths absent."""
    root, paths = Path(plan["source_pins"]["project_root"]), [receipt, post_receipt]
    paths.append(Path(plan["screen"]["report_path"]))
    for item in plan["endpoints"]:
        if item.get("argv"):
            final, stem = Path(item["prediction_path"]), item["output_name"]
            paths += [
                final,
                final.with_name(final.name + ".partial"),
                final.with_suffix(".json"),
                root / "cache" / f"seqsums_{stem}.npz",
            ]
    present = [str(path) for path in paths if path.exists()]
    if present:
        raise FileExistsError("new output paths must be absent: " + ", ".join(present))
    return [str(path) for path in paths]
