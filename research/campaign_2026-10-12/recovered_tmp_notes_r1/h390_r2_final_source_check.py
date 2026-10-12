from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve(strict=True)
    data_root = args.data_root.resolve(strict=True)
    sys.path.insert(0, str(source_root / "src"))
    from wnn.final_distillation_cache import (
        EXPECTED_SPLIT_SHA256,
        TEACHER_SPLIT_MANIFEST_RELATIVE,
        file_sha256,
        verify_h390_source_snapshot,
    )

    snapshot = verify_h390_source_snapshot(source_root)
    split_path = data_root / TEACHER_SPLIT_MANIFEST_RELATIVE
    split_manifest = json.loads(split_path.read_text())
    if file_sha256(split_path) != EXPECTED_SPLIT_SHA256:
        raise ValueError("frozen split manifest changed after preflight")
    footer_receipts = {}
    for split in ("train", "valid"):
        name = f"datasets/{split}.parquet"
        identity = split_manifest["sources"][name]
        source = data_root / name
        footer_size = int(identity["footer_bytes"])
        with source.open("rb") as stream:
            stream.seek(-8, 2)
            trailer = stream.read(8)
            observed_size = int.from_bytes(trailer[:4], "little")
            if trailer[4:] != b"PAR1" or observed_size != footer_size:
                raise ValueError(f"Parquet footer trailer differs: {name}")
            stream.seek(-8 - footer_size, 2)
            footer_and_trailer = stream.read(footer_size + 8)
        observed_sha = hashlib.sha256(footer_and_trailer).hexdigest()
        if observed_sha != identity["footer_sha256"]:
            raise ValueError(f"Parquet footer fingerprint differs: {name}")
        footer_receipts[split] = {
            "path": name,
            "footer_bytes": footer_size,
            "expected_sha256": identity["footer_sha256"],
            "observed_sha256": observed_sha,
        }
    print(
        json.dumps(
            {
                "status": "final_source_check_pass",
                "source_snapshot": snapshot,
                "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
                "parquet_footers": footer_receipts,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
