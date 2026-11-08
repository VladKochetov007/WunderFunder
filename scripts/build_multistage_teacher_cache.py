"""Merge a frozen pool of heterogeneous fitting-only sequence teacher exports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from wnn.splits import file_sha256
from wnn.training.sequence_cache_weighted_pool import merge_weighted_teacher_caches


def build_from_request(
    request_path: Path,
    data_root: Path,
    split_manifest: Path,
    output: Path,
    minimum_free_bytes: int,
) -> Path:
    """Verify a frozen parent/weight request and build its canonical fit cache."""
    if request_path.is_symlink() or not request_path.is_file():
        raise ValueError("weighted teacher pool request must be a regular file")
    request_size = request_path.stat().st_size
    if request_size > 1_000_000:
        raise ValueError("weighted teacher pool request exceeds the 1 MB metadata bound")
    raw_request = request_path.read_bytes()
    if len(raw_request) != request_size:
        raise ValueError("weighted teacher pool request changed while being read")
    request = json.loads(raw_request)
    if not isinstance(request, dict) or request.get("schema") != (
        "wnn.multistage-teacher-pool-request.v1"
    ):
        raise ValueError("unsupported weighted teacher pool request")
    pool_id, parents = request.get("pool_id"), request.get("parents")
    if not isinstance(pool_id, str) or not pool_id or not isinstance(parents, list):
        raise ValueError("pool request needs a stable id and ordered parent list")
    if any(
        not isinstance(parent, dict) or not isinstance(parent.get("cache_root"), str)
        for parent in parents
    ):
        raise ValueError("pool request parent cache roots are malformed")
    input_roots = [Path(parent["cache_root"]) for parent in parents]
    expected_parents = [
        {key: value for key, value in parent.items() if key != "cache_root"} for parent in parents
    ]
    return merge_weighted_teacher_caches(
        input_roots,
        expected_parents,
        data_root,
        split_manifest,
        output,
        pool_id=pool_id,
        request_sha256=hashlib.sha256(raw_request).hexdigest(),
        minimum_free_bytes=minimum_free_bytes,
    )


def main() -> None:
    """Build one immutable heterogeneous cache using a frozen request JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=80.0)
    args = parser.parse_args()
    if args.minimum_free_gib < 0:
        parser.error("--minimum-free-gib must be nonnegative")
    result = build_from_request(
        args.request,
        args.data_root,
        args.split_manifest,
        args.output,
        int(args.minimum_free_gib * (1 << 30)),
    )
    print(json.dumps({"cache": str(result), "request_sha256": file_sha256(args.request)}))


if __name__ == "__main__":
    main()
