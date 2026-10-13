"""Grade aligned full-valid predictions on previously frozen domain mixtures."""

from __future__ import annotations

import argparse
import io
import json
import struct
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from alias_grade_prediction import _npy_header

from wnn.domain_stress import _paired_mixture_scores
from wnn.evaluation import _score_component_moments, paired_component_moments
from wnn.gmetric import METRIC_ID, WARMUP
from wnn.splits import file_sha256

SOURCE_ROOT = Path(__file__).resolve().parents[1]


def bounded_target_header(
    archive: zipfile.ZipFile, name: str, shape: tuple[int, ...], dtype: np.dtype
) -> None:
    """Check the real NPZ member before NumPy can allocate its payload."""
    info = archive.getinfo(name + ".npy")
    if info.file_size > int(np.prod(shape)) * dtype.itemsize + 10_012:
        raise ValueError(f"oversized target member: {name}")
    with archive.open(info) as stream:
        magic = stream.read(8)
        if magic[:6] != b"\x93NUMPY" or magic[6:] not in (b"\x01\x00", b"\x02\x00"):
            raise ValueError("unsupported target NPY format")
        width = 2 if magic[6] == 1 else 4
        length_bytes = stream.read(width)
        if len(length_bytes) != width:
            raise ValueError("truncated target header length")
        length = struct.unpack("<H" if width == 2 else "<I", length_bytes)[0]
        if not 0 < length <= 10_000:
            raise ValueError("target NPY header exceeds bound")
        header = stream.read(length)
        if len(header) != length:
            raise ValueError("truncated target header")
    buffer = io.BytesIO(magic + length_bytes + header)
    version = np.lib.format.read_magic(buffer)
    reader = (
        np.lib.format.read_array_header_1_0
        if version == (1, 0)
        else np.lib.format.read_array_header_2_0
    )
    actual_shape, fortran, actual_dtype = reader(buffer, max_header_size=10_000)
    if actual_shape != shape or actual_dtype != dtype or fortran:
        raise ValueError(f"target schema differs for {name}")
    if info.file_size != buffer.tell() + int(np.prod(shape)) * dtype.itemsize:
        raise ValueError(f"target payload length differs for {name}")


def prediction(path: Path, groups: list[int], split_hash: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Bind one full-valid prediction to its producer and exact source order."""
    sidecar_path = path.with_suffix(".json")
    if sidecar_path.stat().st_size > 1_000_000:
        raise ValueError("prediction sidecar exceeds bound")
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("split") != "valid" or sidecar.get("row_groups") != groups:
        raise ValueError("prediction source or row order differs from target cache")
    if sidecar.get("split_manifest_sha256") != split_hash:
        raise ValueError("prediction belongs to a different fitting split")
    expected = (len(groups), 20_000, 2)
    shape, dtype, fortran, offset = _npy_header(path)
    if shape != expected or dtype != np.dtype("float32") or fortran:
        raise ValueError("prediction schema differs from full-valid FP32 export")
    if path.stat().st_size != offset + int(np.prod(shape)) * dtype.itemsize:
        raise ValueError("prediction payload length differs")
    digest = file_sha256(path)
    if digest != sidecar.get("prediction_sha256"):
        raise ValueError("prediction hash differs from producer sidecar")
    record = {
        "path": str(path),
        "sha256": digest,
        "sidecar_sha256": file_sha256(sidecar_path),
        "run": sidecar["run"],
        "epoch": sidecar["epoch"],
        "checkpoint_sha256": sidecar["checkpoint_sha256"],
    }
    return np.load(path, mmap_mode="r", allow_pickle=False), record


def grade(args: argparse.Namespace) -> dict[str, Any]:
    """Keep label loading bounded and pool original metric moments by sequence."""
    if args.output.exists():
        raise FileExistsError(f"refusing to replace a grade: {args.output}")
    panel_record = json.loads(args.panels.read_text())
    fold = next(x for x in panel_record["folds"] if x["fold"] == args.fold)
    split_path = args.root / f"research/campaign_2026-09-29/artifacts/splits/fold{args.fold}.json"
    split_hash = file_sha256(split_path)
    if fold["split_sha256"] != split_hash:
        raise ValueError("frozen panel and split hashes differ")
    split = json.loads(split_path.read_text())
    evaluation = split["evaluation"]["valid"]
    if (
        len(set(evaluation)) != len(evaluation)
        or sorted(evaluation) != fold["evaluation_row_groups"]
    ):
        raise ValueError("frozen panel evaluation membership differs")
    membership = {int(x["row_group"]): int(x["component"]) for x in split["sequences"]["valid"]}
    group_count = len(membership)
    target_path = args.root / "cache/valid_targets.npz"
    with zipfile.ZipFile(target_path) as archive:
        for name, shape, dtype in (
            ("row_group", (group_count,), np.dtype("int64")),
            ("y", (group_count, 20_000, 2), np.dtype("float32")),
            ("mask", (group_count, 20_000), np.dtype("bool")),
        ):
            bounded_target_header(archive, name, shape, dtype)
    with np.load(target_path, allow_pickle=False) as cache:
        groups = cache["row_group"].tolist()
        if len(set(groups)) != group_count or set(groups) != set(membership):
            raise ValueError("target source membership differs from split")
        candidate, candidate_record = prediction(args.candidate, groups, split_hash)
        reference, reference_record = prediction(args.reference, groups, split_hash)
        targets, scored = cache["y"], cache["mask"]
    position = {group: i for i, group in enumerate(groups)}
    components = np.array([membership[group] for group in evaluation])
    a = np.empty((len(evaluation), 2, 6), dtype=np.float64)
    b = np.empty_like(a)
    selected_rows = 0
    for j, group in enumerate(evaluation):
        i = position[group]
        mask = scored[i : i + 1].copy()
        mask[:, :WARMUP] = False
        selected_rows += int(mask.sum())
        if not mask.any():
            a[j] = b[j] = 0
            continue
        ma, mb = paired_component_moments(
            targets[i : i + 1], candidate[i : i + 1], reference[i : i + 1], mask
        )
        a[j], b[j] = ma[0], mb[0]
    if selected_rows == 0:
        raise ValueError("no official selected rows")
    panels = {}
    for name, panel in fold["panels"].items():
        if panel["status"] != "available":
            panels[name] = {"status": panel["status"]}
            continue
        assignment = dict(
            zip(
                panel["evaluation_component_ids"],
                panel["evaluation_component_domains"],
                strict=True,
            )
        )
        if set(assignment) != set(components.tolist()):
            raise ValueError("domain assignment and evaluation components differ")
        domains = np.array([assignment[int(component)] for component in components])
        panels[name] = _paired_mixture_scores(
            a, b, components, domains, panel["minimum_components"]
        )
    weights = np.ones(len(a))
    candidate_wp = _score_component_moments(a, weights)
    reference_wp = _score_component_moments(b, weights)
    return {
        "schema": "wnn.strategy.domain-grade.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "comparison_label": args.comparison_label,
        "metric_id": METRIC_ID,
        "mask": "supplied valid scored mask AND row>=WARMUP",
        "fold": args.fold,
        "evaluation_groups": len(evaluation),
        "evaluation_components": len(set(components.tolist())),
        "selected_rows": selected_rows,
        "candidate": candidate_record,
        "reference": reference_record,
        "ordinary": {
            "candidate_wp": candidate_wp,
            "reference_wp": reference_wp,
            "paired_delta": candidate_wp - reference_wp,
        },
        "panels": panels,
        "panel_sha256": file_sha256(args.panels),
        "split_sha256": split_hash,
        "target_cache_sha256": file_sha256(target_path),
        "source_sha256": {
            p: file_sha256(SOURCE_ROOT / p)
            for p in (
                "scripts/grade_strategy_domains.py",
                "scripts/alias_grade_prediction.py",
                "src/wnn/domain_stress.py",
                "src/wnn/evaluation.py",
                "src/wnn/gmetric.py",
            )
        },
        "limits": [
            "Previously exposed evaluation groups; fixed fitted predictions, not training-seed uncertainty.",
            "Constructed component-frequency sensitivities, not a forecast of private data.",
            "Producer bindings do not independently establish all checkpoint ancestry.",
            "A matched control does not substitute for the complete C bridge in endpoint selection.",
        ],
    }


def main() -> None:
    """Write one immutable paired ordinary/stress score receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--comparison-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = grade(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["ordinary"]))


if __name__ == "__main__":
    main()
