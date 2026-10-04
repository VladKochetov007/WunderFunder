from __future__ import annotations

import ast
import hashlib
import json
import zipfile
from pathlib import Path

PACKET = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    upload = PACKET / "upload_dataset"
    kernel_dir = PACKET / "kernel_push"
    manifest_path = upload / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    archive = upload / "source_bundle.zip"
    kernel = kernel_dir / "kernel.py"
    kernel_metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text())
    dataset_metadata = json.loads((upload / "dataset-metadata.json").read_text())
    assert sha256(archive) == manifest["archive_sha256"]
    assert manifest["private"] is True
    assert all(manifest[key] is False for key in ("enable_gpu", "enable_tpu", "enable_internet"))
    assert kernel_metadata["dataset_sources"] == [manifest["dataset_id"]]
    assert kernel_metadata["enable_gpu"] is False and kernel_metadata["enable_tpu"] is False
    assert kernel_metadata["enable_internet"] is False and kernel_metadata["is_private"] is True
    assert dataset_metadata["id"] == manifest["dataset_id"]
    assert dataset_metadata["isPrivate"] is True
    source = PACKET / "build_inputs/source_bundle"
    with zipfile.ZipFile(archive) as zipped:
        assert set(zipped.namelist()) == set(manifest["files"])
        for name in zipped.namelist():
            raw = zipped.read(name)
            descriptor = manifest["files"][name]
            assert len(raw) == descriptor["size_bytes"]
            assert hashlib.sha256(raw).hexdigest() == descriptor["sha256"]
            assert raw == (source / name).read_bytes()
            if name.endswith(".py"):
                ast.parse(raw.decode("utf-8"), filename=name)
    kernel_text = kernel.read_text()
    assert manifest["archive_sha256"] in kernel_text
    assert sha256(manifest_path) in kernel_text
    assert "__ARCHIVE_SHA256__" not in kernel_text and "__MANIFEST_SHA256__" not in kernel_text
    print(
        json.dumps(
            {
                "status": "static_packet_valid",
                "member_count": len(manifest["files"]),
                "archive_sha256": sha256(archive),
                "manifest_sha256": sha256(manifest_path),
                "kernel_sha256": sha256(kernel),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
