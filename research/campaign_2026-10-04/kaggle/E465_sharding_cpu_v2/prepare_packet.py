from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

PACKET = Path(__file__).resolve().parent
UPLOAD = PACKET / "upload_dataset"
BUNDLE_ROOT = PACKET / "build_inputs" / "source_bundle"
ARCHIVE = UPLOAD / "source_bundle.zip"
DATASET_ID = "vladkochetov07/wnn-jax-sharding-e465-cpu-20261004-v2"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def main() -> None:
    members = sorted(path for path in BUNDLE_ROOT.rglob("*") if path.is_file())
    descriptors = {}
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in members:
            relative = path.relative_to(BUNDLE_ROOT).as_posix()
            data = path.read_bytes()
            info = zipfile.ZipInfo(relative, date_time=(2026, 10, 4, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
            descriptors[relative] = {"size_bytes": len(data), "sha256": sha256_bytes(data)}
    manifest = {
        "dataset_id": DATASET_ID,
        "private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "archive_file": "source_bundle.zip",
        "archive_sha256": sha256_path(ARCHIVE),
        "files": descriptors,
        "classification": "E465 CPU-only two-device objective/sharding/state correctness tests; no fit or score",
    }
    manifest_path = UPLOAD / "source-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (UPLOAD / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": DATASET_ID,
                "title": "E465 JAX sharding CPU correctness tests v1",
                "isPrivate": True,
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n"
    )
    manifest_sha = sha256_path(manifest_path)
    kernel_path = PACKET / "kernel_push" / "kernel.py"
    kernel = kernel_path.read_text()
    kernel, archive_replacements = re.subn(
        r'ARCHIVE_SHA256 = "[0-9a-f]+"',
        f'ARCHIVE_SHA256 = "{manifest["archive_sha256"]}"',
        kernel,
    )
    kernel, manifest_replacements = re.subn(
        r'MANIFEST_SHA256 = "[0-9a-f]+"',
        f'MANIFEST_SHA256 = "{manifest_sha}"',
        kernel,
    )
    if archive_replacements != 1 or manifest_replacements != 1:
        raise RuntimeError("kernel hash pins were not fully rendered")
    kernel_path.write_text(kernel)
    (PACKET / "kernel_push" / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "vladkochetov07/wnn-jax-sharding-e465-cpu-20261004-v2",
                "title": "wnn-jax-sharding-e465-cpu-20261004-v2",
                "code_file": "kernel.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_tpu": False,
                "enable_internet": False,
                "dataset_sources": [DATASET_ID],
                "competition_sources": [],
                "kernel_sources": [],
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"archive_sha256": manifest["archive_sha256"], "manifest_sha256": manifest_sha,
                      "kernel_sha256": sha256_path(kernel_path), "member_count": len(descriptors)}, indent=2))


if __name__ == "__main__":
    main()
