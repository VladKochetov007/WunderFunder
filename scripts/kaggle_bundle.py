from __future__ import annotations

import argparse
import gzip
import io
import json
import shutil
import tarfile
from pathlib import Path

from kaggle_common import read_spec, relative_path, sha256_file, verify_files, write_json


def build_bundle(
    root: Path, specs: list[Path], destination: Path, dataset: str, external: list[str]
) -> None:
    """Build only explicit dependency closures; never copy credentials or raw datasets."""
    if not dataset.startswith("vladkochetov07/") or len(dataset.split("/")) != 2:
        raise ValueError("Private bundle owner must match the authorized account")
    external_map = {}
    for entry in external:
        logical, filename = entry.split("=", 1)
        if filename not in {"train.parquet", "valid.parquet", "valid_mask.parquet"}:
            raise ValueError("Only the authorized raw dataset can be externally mounted")
        external_map[relative_path(logical)] = filename
    files = {}
    payload_specs = []
    for source in specs:
        spec = read_spec(source)
        verify_files(root, spec["code"] + spec["inputs"])
        for item in spec["code"] + spec["inputs"]:
            previous = files.setdefault(item["path"], item)
            if previous != item:
                raise ValueError(f"Conflicting dependency: {item['path']}")
        payload_specs.append(spec)
    if len({spec["name"] for spec in payload_specs}) != len(payload_specs):
        raise ValueError("Experiment names must be unique")
    if set(external_map) - set(files):
        raise ValueError("External inputs must be declared and hashed in a spec")
    for name in (
        "scripts/kaggle_gpu_runner.py",
        "scripts/kaggle_common.py",
        "scripts/kaggle_kernel.py",
    ):
        if name not in files:
            raise ValueError(f"Worker source must be sealed in spec code: {name}")
    total = sum(item["size"] for name, item in files.items() if name not in external_map)
    if shutil.disk_usage(destination.parent).free < total + 20 * 1024**3:
        raise ValueError("Bundle would violate the 20 GiB free-disk reserve")
    destination.mkdir(exist_ok=False)
    archive = destination / "bundle.tar.gz"
    with (
        archive.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as tar,
    ):
        for name, item in sorted(files.items()):
            if name in external_map:
                continue
            info = tarfile.TarInfo(relative_path(name))
            info.size, info.mode = item["size"], 0o644
            with (root / name).open("rb") as source:
                tar.addfile(info, source)
        for spec in payload_specs:
            data = (json.dumps(spec, sort_keys=True) + "\n").encode()
            info = tarfile.TarInfo(f"kaggle_specs/{spec['name']}.json")
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
    manifest = {
        "version": 1,
        "dataset": dataset,
        "private": True,
        "archive_sha256": sha256_file(archive),
        "archive_bytes": archive.stat().st_size,
        "expanded_bytes": total,
        "files": list(files.values()),
        "external_mounts": external_map,
        "spec_paths": [f"kaggle_specs/{s['name']}.json" for s in payload_specs],
    }
    write_json(destination / "bundle-manifest.json", manifest)
    write_json(
        destination / "dataset-metadata.json",
        {
            "id": dataset,
            "title": dataset.split("/")[1],
            "licenses": [{"name": "other"}],
            "isPrivate": True,
        },
    )
    kernel = destination / "kernel"
    kernel.mkdir()
    slug = dataset.split("/")[1]
    bootstrap = (root / "scripts/kaggle_kernel.py").read_text()
    bootstrap = bootstrap.replace('BUNDLE_SLUG = ""', f"BUNDLE_SLUG = {slug!r}")
    bootstrap = bootstrap.replace(
        'EXPECTED_ARCHIVE_SHA256 = ""', f"EXPECTED_ARCHIVE_SHA256 = {manifest['archive_sha256']!r}"
    )
    (kernel / "kernel.py").write_text(bootstrap)
    write_json(
        kernel / "kernel-metadata.json",
        {
            "id": f"vladkochetov07/{slug}-runner",
            "title": f"{slug} runner",
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": False,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [dataset, "vladkochetov07/wunder-funder"],
            "competition_sources": [],
            "kernel_sources": [],
        },
    )


def main() -> None:
    """Construct a private upload directory without creating a provider resource."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, action="append", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--external", action="append", default=[])
    args = parser.parse_args()
    build_bundle(args.root, args.spec, args.destination, args.dataset, args.external)


if __name__ == "__main__":
    main()
