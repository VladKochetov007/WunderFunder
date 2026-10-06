"""Chunked, read-only relational-KD diagnostics; this script never trains or scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import torch

import wnn.rkd_component as rkd_component
from wnn.rkd_component import sampled_relation_loss

SEQUENCE_ROWS = 20_000
SOURCE_COEFFICIENT_TOTAL = 0.8125
CACHE_FORMAT = "wnn.heterogeneous-distillation-cache"
CACHE_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, data: str) -> None:
    """Create a new output file and refuse to replace an existing artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        with suppress(FileNotFoundError):
            path.unlink()
        raise


def _load_memmap(path_arg: Path, expected_dtype: np.dtype[Any]) -> np.memmap:
    path = path_arg.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"expected an NPY file: {path_arg}")
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read NPY array: {path_arg}") from exc
    if not isinstance(array, np.memmap):
        raise ValueError(f"expected an uncompressed NPY memory map: {path_arg}")
    if array.dtype != expected_dtype:
        raise ValueError(f"{path_arg} must have dtype {expected_dtype}")
    return array


def _load_source_cache_manifest(teacher_path: Path) -> tuple[dict[str, Any], Path, str]:
    metadata_path = teacher_path.parent / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("source cache requires a readable metadata.json sidecar") from exc
    if not isinstance(metadata, dict):
        raise ValueError("source cache metadata must be a JSON object")
    if metadata.get("format") != CACHE_FORMAT or metadata.get("version") != CACHE_VERSION:
        raise ValueError("unsupported source cache metadata format")
    try:
        coefficients = [float(item["coefficient"]) for item in metadata["source_coefficients"]]
        coefficient_sum = math.fsum(coefficients)
        declared_sum = float(metadata["coefficient_sum"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("source cache metadata lacks valid coefficient provenance") from exc
    if (
        not coefficients
        or any(not math.isfinite(value) or value <= 0 for value in coefficients)
        or abs(coefficient_sum - declared_sum) > 1e-7
        or abs(coefficient_sum - SOURCE_COEFFICIENT_TOTAL) > 1e-7
    ):
        raise ValueError("source cache coefficients must sum to 0.8125")
    files = metadata.get("files")
    if not isinstance(files, dict):
        raise ValueError("source cache metadata has no file manifest")
    keys = [
        key
        for key in ("main_source_train", "main_source_valid")
        if isinstance(files.get(key), dict) and files[key].get("path") == teacher_path.name
    ]
    if len(keys) != 1:
        raise ValueError("teacher file is not registered as one source-scale cache array")
    entry = files[keys[0]]
    if Path(str(entry.get("path", ""))).name != entry.get("path"):
        raise ValueError("source cache metadata path must be a single filename")
    return entry, metadata_path, sha256_file(metadata_path)


def _diagnostic_records(
    teacher: np.ndarray,
    student: np.ndarray,
    valid: np.ndarray | None,
    *,
    batch_size: int,
    time_chunk: int,
    seed: int,
    max_pairs: int,
) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []
    ordinal = 0
    for row_start in range(0, teacher.shape[1], time_chunk):
        row_stop = min(row_start + time_chunk, teacher.shape[1])
        for sequence_start in range(0, teacher.shape[0], batch_size):
            sequence_stop = min(sequence_start + batch_size, teacher.shape[0])
            selection = np.s_[sequence_start:sequence_stop, row_start:row_stop]
            teacher_chunk = torch.from_numpy(np.array(teacher[selection], copy=True))
            student_chunk = torch.from_numpy(np.array(student[selection], copy=True))
            valid_chunk = None
            if valid is not None:
                valid_chunk = torch.from_numpy(np.array(valid[selection[:2]], copy=True))
            value = sampled_relation_loss(
                teacher_chunk,
                student_chunk,
                valid_chunk,
                pair_seed=seed + ordinal,
                max_pairs=max_pairs,
            )
            numeric = float(value.detach().item())
            if not math.isfinite(numeric):
                raise ValueError("relation diagnostic produced a non-finite value")
            records.append(
                {
                    "sequence_start": sequence_start,
                    "sequence_stop": sequence_stop,
                    "row_start": row_start,
                    "row_stop": row_stop,
                    "relation_loss": numeric,
                }
            )
            ordinal += 1
    return records


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Relational KD cache diagnostic",
        "",
        "This read-only report contains relation diagnostics only. It does not fit a model, run a grader, or estimate a submission score.",
        "",
        f"- Source coefficient total: `{report['source_coefficient_total']}`",
        f"- Teacher cache SHA-256: `{report['inputs']['teacher_cache']['sha256']}`",
        f"- Student predictions SHA-256: `{report['inputs']['student_predictions']['sha256']}`",
        f"- Diagnostic chunks: `{len(report['chunk_losses'])}`",
        "",
        "| Sequence range | Row range | Relation loss |",
        "|---:|---:|---:|",
    ]
    for record in report["chunk_losses"]:
        lines.append(
            f"| {record['sequence_start']}–{record['sequence_stop']} | "
            f"{record['row_start']}–{record['row_stop']} | {record['relation_loss']:.8g} |"
        )
    return "\n".join(lines) + "\n"


def run_diagnostic(
    *,
    teacher_path: Path,
    student_path: Path,
    valid_path: Path | None,
    output_json: Path,
    output_markdown: Path | None,
    batch_size: int,
    time_chunk: int,
    seed: int,
    max_pairs: int,
    coefficient_total: float,
) -> dict[str, Any]:
    if coefficient_total != SOURCE_COEFFICIENT_TOTAL:
        raise ValueError(f"source cache coefficient total must equal {SOURCE_COEFFICIENT_TOTAL}")
    if batch_size < 1 or time_chunk < 1 or max_pairs < 2:
        raise ValueError("batch size/time chunk must be positive and max pairs at least 2")
    paths = [teacher_path.resolve(strict=True), student_path.resolve(strict=True)]
    if valid_path is not None:
        paths.append(valid_path.resolve(strict=True))
    cache_metadata_path = paths[0].parent / "metadata.json"
    protected_paths = {*paths, cache_metadata_path.resolve(strict=False)}
    outputs = [output_json.absolute()]
    if output_markdown is not None:
        outputs.append(output_markdown.absolute())
    if any(output.resolve(strict=False) in protected_paths for output in outputs):
        raise ValueError("diagnostic output path must not overwrite an input artifact")
    if len({str(output) for output in outputs}) != len(outputs):
        raise ValueError("JSON and Markdown outputs must use different paths")
    if any(output.exists() or output.is_symlink() for output in outputs):
        raise FileExistsError("diagnostic output already exists; refusing to overwrite")

    teacher_resolved = paths[0]
    source_entry, metadata_path, metadata_sha256 = _load_source_cache_manifest(teacher_resolved)
    teacher = _load_memmap(teacher_resolved, np.dtype(np.float32))
    student = _load_memmap(student_path, np.dtype(np.float32))
    if teacher.ndim != 3 or teacher.shape[1:] != (SEQUENCE_ROWS, 2):
        raise ValueError("teacher source cache must have shape (N, 20000, 2)")
    if student.shape != teacher.shape:
        raise ValueError("student predictions must match the teacher cache shape")
    if teacher.shape[0] == 0:
        raise ValueError("teacher source cache must contain at least one sequence")
    teacher_sha256 = sha256_file(teacher_resolved)
    if (
        source_entry.get("shape") != list(teacher.shape)
        or source_entry.get("dtype") != "float32"
        or source_entry.get("bytes") != teacher_resolved.stat().st_size
        or source_entry.get("sha256") != teacher_sha256
    ):
        raise ValueError("source teacher array disagrees with its metadata entry")
    valid = None
    if valid_path is not None:
        valid = _load_memmap(valid_path, np.dtype(np.bool_))
        if valid.shape != teacher.shape[:2]:
            raise ValueError("valid mask must have shape (N, 20000)")

    records = _diagnostic_records(
        teacher,
        student,
        valid,
        batch_size=batch_size,
        time_chunk=time_chunk,
        seed=seed,
        max_pairs=max_pairs,
    )
    report = {
        "schema_version": 1,
        "mode": "read_only_relation_diagnostic",
        "fit_performed": False,
        "grader_or_score_used": False,
        "source_coefficient_total": SOURCE_COEFFICIENT_TOTAL,
        "configuration": {
            "batch_size": batch_size,
            "time_chunk": time_chunk,
            "seed": seed,
            "max_pairs_per_sequence": max_pairs,
            "valid_mask_supplied": valid is not None,
            "read_mode": "numpy mmap_mode='r'; one chunk copied to CPU tensors at a time",
        },
        "inputs": {
            "teacher_cache": {
                "path": str(teacher_path),
                "sha256": teacher_sha256,
                "metadata_path": str(metadata_path),
                "metadata_sha256": metadata_sha256,
                "shape": list(teacher.shape),
                "dtype": str(teacher.dtype),
            },
            "student_predictions": {
                "path": str(student_path),
                "sha256": sha256_file(paths[1]),
                "shape": list(student.shape),
                "dtype": str(student.dtype),
            },
            "valid_mask": None
            if valid_path is None
            else {
                "path": str(valid_path),
                "sha256": sha256_file(paths[2]),
                "shape": list(valid.shape),
                "dtype": str(valid.dtype),
            },
        },
        "source_code_sha256": {
            "runner": sha256_file(Path(__file__).resolve()),
            "rkd_component": sha256_file(Path(rkd_component.__file__).resolve()),
        },
        "chunk_losses": records,
    }
    write_new(output_json, json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if output_markdown is not None:
        write_new(output_markdown, _markdown_report(report))
    return report


def run_smoke_test() -> dict[str, Any]:
    teacher = torch.tensor(
        [
            [0.0, 0.0],
            [0.3, 1.0],
            [2.0, -0.4],
            [1.0, 3.0],
            [-1.8, 1.4],
            [3.3, 2.1],
            [2.8, -2.2],
            [-1.1, -2.0],
        ],
        dtype=torch.float32,
    ).repeat(2, 1, 1)
    matched = sampled_relation_loss(teacher, teacher.clone(), pair_seed=123, max_pairs=4)
    affine = sampled_relation_loss(
        teacher, 37.0 * teacher + torch.tensor([9.0, -4.0]), pair_seed=123, max_pairs=4
    )
    shuffled = sampled_relation_loss(
        teacher,
        teacher[:, torch.tensor([2, 5, 0, 7, 1, 4, 6, 3])],
        pair_seed=123,
        max_pairs=4,
    )
    if matched.item() > 1e-7 or affine.item() > 1e-6 or shuffled.item() <= matched.item() + 1e-3:
        raise RuntimeError("synthetic relation smoke test failed")
    return {
        "mode": "synthetic_smoke_test",
        "fit_performed": False,
        "grader_or_score_used": False,
        "matched_relation_loss": float(matched.item()),
        "scale_translation_relation_loss": float(affine.item()),
        "time_permutation_relation_loss": float(shuffled.item()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-cache", type=Path)
    parser.add_argument("--student-predictions", type=Path)
    parser.add_argument("--valid-mask", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--time-chunk", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=256)
    parser.add_argument("--source-coefficient-total", type=float, default=SOURCE_COEFFICIENT_TOTAL)
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.smoke_test:
        print(json.dumps(run_smoke_test(), sort_keys=True))
        return 0
    if args.teacher_cache is None or args.student_predictions is None or args.output_json is None:
        parser.error(
            "data diagnostic requires --teacher-cache, --student-predictions, and --output-json"
        )
    try:
        report = run_diagnostic(
            teacher_path=args.teacher_cache,
            student_path=args.student_predictions,
            valid_path=args.valid_mask,
            output_json=args.output_json,
            output_markdown=args.output_markdown,
            batch_size=args.batch_size,
            time_chunk=args.time_chunk,
            seed=args.seed,
            max_pairs=args.max_pairs,
            coefficient_total=args.source_coefficient_total,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"refusing to run relation diagnostic: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"chunk_count": len(report["chunk_losses"]), "output": str(args.output_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
