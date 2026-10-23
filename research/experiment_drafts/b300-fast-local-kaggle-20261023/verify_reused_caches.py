"""Verify reused disk caches without decoding labels or fitting a model."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from collect_online_panel_terminal import read_npy_header, sha256_file  # noqa: E402


def file_identity(path: Path) -> dict[str, int]:
    """Capture file identity to detect concurrent changes during verification."""
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def verify(recipe_path: Path, expected_sha: str) -> dict[str, object]:
    """Hash each original cache once and check its exact producer-bound encoding."""
    started = time.monotonic()
    if sha256_file(recipe_path) != expected_sha:
        raise ValueError("age-screen recipe hash differs")
    recipe = json.loads(recipe_path.read_text())
    source_recipe = recipe["source_cache_recipe"]
    source_path = ROOT / source_recipe["path"]
    if sha256_file(source_path) != source_recipe["sha256"]:
        raise ValueError("original cache recipe hash differs")
    files: list[dict[str, object]] = []
    checked_sources: dict[str, str] = {}
    for fold in recipe["folds"]:
        directory = Path(fold["parent_cache"])
        for stem in ("fit_sequences", "evaluation_sequences", "evaluation_predictions"):
            path = directory / f"{stem}.npy"
            sidecar = directory / f"{stem}.json"
            sidecar_sha = sha256_file(sidecar)
            metadata = json.loads(sidecar.read_text())
            before = file_identity(path)
            header = read_npy_header(path)
            expected_hash = metadata.get("data_sha256", metadata.get("prediction_sha256"))
            if (
                header["descr"] != "<f4"
                or header["fortran_order"] is not False
                or header["shape"] != metadata["shape"]
                or header["payload_bytes"] != math.prod(header["shape"]) * 4
            ):
                raise ValueError(f"cache header or byte coverage differs: {path}")
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash or file_identity(path) != before:
                raise ValueError(f"cache identity differs or changed during reading: {path}")
            if sha256_file(sidecar) != sidecar_sha:
                raise ValueError(f"cache sidecar changed during reading: {sidecar}")
            if metadata["recipe_sha256"] != source_recipe["sha256"]:
                raise ValueError(f"cache belongs to another producer recipe: {path}")
            for relative, digest in metadata.get("source_sha256", {}).items():
                if relative not in checked_sources:
                    if sha256_file(ROOT / relative) != digest:
                        raise ValueError(f"cache producer source differs: {relative}")
                    checked_sources[relative] = digest
                elif checked_sources[relative] != digest:
                    raise ValueError(f"caches disagree on producer source: {relative}")
            files.append(
                {
                    "fold": fold["fold"],
                    "kind": stem,
                    "source_path": str(path),
                    "sha256": actual_hash,
                    "identity": before,
                    "header": header,
                    "sidecar_path": str(sidecar),
                    "sidecar_sha256": sidecar_sha,
                }
            )
            print(
                json.dumps({"verified": str(path), "elapsed": time.monotonic() - started}),
                flush=True,
            )
    return {
        "schema": "wnn.age-readout-local-cache-verification.v1",
        "status": "PASS_PAYLOAD_HASH_AND_HEADER",
        "created_utc": datetime.now(UTC).isoformat(),
        "recipe_sha256": expected_sha,
        "source_cache_recipe": source_recipe,
        "files": files,
        "producer_source_sha256": checked_sources,
        "elapsed_seconds": time.monotonic() - started,
        "limits": "Raw cache bytes hashed; labels not decoded or scored. Semantic membership and ancestry checks remain in the canonical loader and independent grader.",
    }


def main() -> None:
    """Publish one create-only receipt after every source cache passes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = verify(args.recipe, args.recipe_sha256)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
