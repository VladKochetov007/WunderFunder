"""Fit one fixed OOF shrinkage rule and score one source-qualified outer panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np

from wnn.blend_optimization import fit_shared_clip2_weights
from wnn.gmetric import global_wp

MAX_MANIFEST_BYTES = 4_000_000
MAX_SELECTED_ROWS = 5_000_000
MAX_MODELS = 8
NODE_KINDS = {"model", "checkpoint", "normalizer", "teacher", "transform", "selector"}


def file_sha256(path: Path) -> str:
    """Hash bytes incrementally without loading the file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qualified_file(root: Path, record: dict) -> Path:
    """Resolve a relative artifact path within the manifest root and verify SHA-256."""
    relative = Path(record["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact path must be relative and contained")
    result = (root / relative).resolve(strict=True)
    if not result.is_relative_to(root) or not result.is_file():
        raise ValueError("artifact escapes root or is not a file")
    expected = record["sha256"]
    if not isinstance(expected, str) or len(expected) != 64 or file_sha256(result) != expected:
        raise ValueError(f"artifact hash differs: {relative}")
    return result


def components(value: object) -> set[str]:
    """Require nonempty explicit unique string component IDs."""
    if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x for x in value):
        raise ValueError("components must be a nonempty list of strings")
    result = set(value)
    if len(result) != len(value):
        raise ValueError("duplicate component IDs")
    return result


def validate_lineage(manifest: dict, root: Path, outer: set[str]) -> dict[str, dict]:
    """Validate declared receipt-bound complete DAG and reject cycles and missing nodes."""
    att = manifest.get("operator_attestation", {})
    for key in ("complete_ancestry", "component_membership_verified", "row_alignment_and_masks_verified", "receipts_reviewed"):
        if att.get(key) is not True:
            raise ValueError(f"operator attestation missing: {key}")
    if not isinstance(att.get("operator"), str) or not att["operator"].strip():
        raise ValueError("operator name is required for attestation")
    nodes: dict[str, dict] = {}
    for node in manifest["lineage_nodes"]:
        name = node["id"]
        if not isinstance(name, str) or not name or name in nodes:
            raise ValueError("node IDs must be unique nonempty strings")
        if node["kind"] not in NODE_KINDS or not isinstance(node["parents"], list):
            raise ValueError(f"unknown node kind or parents: {name}")
        fit = components(node["fit_components"])
        if fit & outer:
            raise ValueError(f"ancestor {name} used outer components")
        qualified_file(root, node["receipt"])
        nodes[name] = node
    active: set[str] = set()
    done: set[str] = set()

    def visit(name: str) -> None:
        if name in active:
            raise ValueError("lineage cycle")
        if name in done:
            return
        if name not in nodes:
            raise ValueError(f"unknown ancestor: {name}")
        active.add(name)
        for parent in nodes[name]["parents"]:
            visit(parent)
        active.remove(name)
        done.add(name)

    for name in nodes:
        visit(name)
    return nodes


def check_producers(ids: list[str], nodes: dict[str, dict], excluded: set[str], members: list[str]) -> None:
    """Ensure each member's entire declared dependency graph excludes its scored units."""
    if len(ids) != len(members) or len(set(ids)) != len(members):
        raise ValueError("each ensemble member needs a distinct source producer")
    for name, member in zip(ids, members, strict=True):
        if name not in nodes or nodes[name]["kind"] not in {"model", "checkpoint"} or nodes[name].get("member") != member:
            raise ValueError("producer identity and declared member order differ")
    pending = list(ids)
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        if name not in nodes:
            raise ValueError(f"unknown producer: {name}")
        seen.add(name)
        node = nodes[name]
        if components(node["fit_components"]) & excluded:
            raise ValueError(f"producer or ancestor {name} saw scored components")
        pending.extend(node["parents"])


def load_part(root: Path, part: dict, expected: set[str], count: int) -> tuple[np.ndarray, ...]:
    """Load a hash-bound NPZ with aligned per-row arrays and verify group coverage."""
    path = qualified_file(root, part["data"])
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) != 4 or sum(info.file_size for info in entries) > 1_500_000_000:
            raise ValueError("NPZ uncompressed size or entry count exceeds limit")
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != {"component", "target", "prediction", "mask"}:
            raise ValueError("NPZ must contain exactly component,target,prediction,mask")
        groups = loaded["component"]
        target, prediction, mask = loaded["target"], loaded["prediction"], loaded["mask"]
    if groups.ndim != 1 or groups.dtype.kind not in {"U", "S"}:
        raise ValueError("component must be a 1D string array")
    if groups.dtype.kind == "S":
        groups = groups.astype("U")
    n = len(groups)
    if n > MAX_SELECTED_ROWS or groups.dtype.itemsize > 512:
        raise ValueError("row count or component string width exceeds limit")
    if target.shape != (n, 2) or prediction.shape != (n, count, 2) or mask.shape != (n,):
        raise ValueError("target/prediction/mask shapes or member count differ")
    if mask.dtype != np.bool_ or set(groups.tolist()) != expected:
        raise ValueError("component coverage or bool mask differs from manifest")
    if int(mask.sum()) < 1 or int(mask.sum()) > MAX_SELECTED_ROWS:
        raise ValueError("selected rows exceed configured limit or are empty")
    if not np.isfinite(target[mask]).all() or not np.isfinite(prediction[mask]).all():
        raise ValueError("selected rows contain nonfinite values")
    return target[mask], prediction[mask]


def execute(path: Path, output: Path, expected_manifest_sha256: str) -> None:
    """Validate manifest and provenance, fit inner-only weights, and grade held-out rows."""
    if output.exists():
        raise ValueError("output already exists")
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("manifest exceeds limit")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError("manifest hash does not match frozen expected hash")
    manifest = json.loads(raw)
    if manifest["schema"] != "wnn.nested-oof-shrinkage.v1":
        raise ValueError("unsupported manifest schema")
    root = path.parent.resolve(strict=True)
    members = manifest["members"]
    if not isinstance(members, list) or not 2 <= len(members) <= MAX_MODELS or len(set(members)) != len(members):
        raise ValueError("members must be 2–8 distinct ordered IDs")
    count = len(members)
    datasets = manifest["source_datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("source dataset identities are required")
    for dataset in datasets:
        qualified_file(root, dataset)
    original = np.asarray(manifest["baseline_weights"], dtype=np.float64)
    if original.shape != (count,) or not np.isfinite(original).all() or np.any(original < 0) or not np.isclose(original.sum(), 1, atol=1e-8, rtol=0):
        raise ValueError("baseline weights must be a simplex vector")
    shrink, ridge = float(manifest["shrink_to_baseline"]), float(manifest["ridge"])
    if not math.isfinite(shrink) or not 0 <= shrink <= 1 or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("shrink must be in [0,1] and ridge must be positive")
    outer_part = manifest["outer"]
    outer = components(outer_part["components"])
    nodes = validate_lineage(manifest, root, outer)
    check_producers(outer_part["producers"], nodes, outer, members)
    inner_parts = manifest["inner"]
    if not isinstance(inner_parts, list) or len(inner_parts) < 2:
        raise ValueError("at least two disjoint inner OOF parts are required")
    seen: set[str] = set()
    targets, predictions = [], []
    for part in inner_parts:
        groups = components(part["components"])
        if groups & (outer | seen):
            raise ValueError("inner groups overlap outer or another OOF part")
        seen.update(groups)
        check_producers(part["producers"], nodes, groups | outer, members)
        y, pred = load_part(root, part, groups, count)
        targets.append(y)
        predictions.append(pred)
    if sum(len(y) for y in targets) > MAX_SELECTED_ROWS:
        raise ValueError("inner total exceeds selected-row limit")
    y_inner, p_inner = np.concatenate(targets), np.concatenate(predictions)
    fitted = fit_shared_clip2_weights(y_inner, p_inner, np.ones(len(y_inner), dtype=bool), original, ridge=ridge)
    weights = (1.0 - shrink) * fitted.weights + shrink * original
    y_outer, p_outer = load_part(root, outer_part, outer, count)
    outer_mask = np.ones(len(y_outer), dtype=bool)
    baseline = np.einsum("nmk,m->nk", p_outer, original, dtype=np.float64).astype(np.float32)
    candidate = np.einsum("nmk,m->nk", p_outer, weights, dtype=np.float64).astype(np.float32)
    result = {
        "schema": "wnn.nested-oof-shrinkage-result.v1",
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "members": members,
        "inner_rows": len(y_inner),
        "outer_rows": len(y_outer),
        "baseline_weights": original.tolist(),
        "unshrunk_weights": fitted.weights.tolist(),
        "candidate_weights": weights.tolist(),
        "inner_selected_candidate": fitted.selected_candidate,
        "inner_unshrunk_wp": fitted.official_score,
        "outer_baseline": global_wp(y_outer, baseline, outer_mask),
        "outer_candidate": global_wp(y_outer, candidate, outer_mask),
        "limitations": "DAG completeness and row/component identity and official masks rely on operator attestation; no independent holdout or serving parity is established.",
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    """Read a frozen manifest and create one immutable analysis receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    execute(args.manifest.resolve(strict=True), args.output, args.manifest_sha256)


if __name__ == "__main__":
    main()
