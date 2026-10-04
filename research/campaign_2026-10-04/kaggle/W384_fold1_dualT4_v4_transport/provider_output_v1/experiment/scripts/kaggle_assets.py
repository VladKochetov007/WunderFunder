from __future__ import annotations

import argparse
import errno
import os
import re
import shutil
from pathlib import Path

from kaggle_common import relative_path, sha256_file, write_json


def prepare(root: Path, inputs: list[str], destination: Path, dataset: str) -> None:
    """Stage explicit private, content-addressed assets for reuse across code waves."""
    if not re.fullmatch(r"vladkochetov07/[a-z0-9][a-z0-9-]*", dataset):
        raise ValueError("Asset owner must match the authorized account")
    if len(inputs) != len(set(inputs)):
        raise ValueError("Duplicate input")
    paths = [relative_path(name) for name in inputs]
    if any(
        Path(name).name in ("train.parquet", "valid.parquet", "valid_mask.parquet")
        for name in paths
    ):
        raise ValueError("Raw files already have a private dataset; do not re-upload them")
    total = sum((root / name).stat().st_size for name in paths)
    if shutil.disk_usage(destination.parent).free < total + 20 * 1024**3:
        raise ValueError("Insufficient projected disk for fallback copies and reserve")
    destination.mkdir(exist_ok=False)
    files = []
    for name in paths:
        source = root / name
        digest = sha256_file(source)
        filename = f"{digest}.wnn"
        target = destination / filename
        if not target.exists():
            try:
                os.link(source, target)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copyfile(source, target)
        files.append(
            {"path": name, "size": source.stat().st_size, "sha256": digest, "filename": filename}
        )
    write_json(
        destination / "assets-manifest.json",
        {"version": 1, "dataset": dataset, "private": True, "files": files},
    )
    write_json(
        destination / "dataset-metadata.json",
        {
            "id": dataset,
            "title": dataset.split("/")[1],
            "isPrivate": True,
            "licenses": [{"name": "other"}],
        },
    )


def main() -> None:
    """Prepare an upload directory; provider creation remains controller-directed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    prepare(args.root, args.input, args.destination, args.dataset)


if __name__ == "__main__":
    main()
