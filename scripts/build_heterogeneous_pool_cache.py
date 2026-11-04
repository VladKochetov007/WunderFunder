from __future__ import annotations

import argparse
import json
from pathlib import Path

from wnn.heterogeneous_pool_cache import build_pool_cache


def main() -> None:
    """Build one declared teacher pool, never fit a model or inspect valid targets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-sequences", type=int, default=8)
    parser.add_argument("--min-free-gib", type=float, default=12)
    args = parser.parse_args()
    output = build_pool_cache(
        args.inputs,
        args.data_root,
        args.split_manifest,
        args.output,
        chunk_sequences=args.chunk_sequences,
        min_free_bytes=int(args.min_free_gib * 1024**3),
    )
    print(json.dumps({"cache": str(output), "status": "verified", "new_fits": 0}))


if __name__ == "__main__":
    main()
