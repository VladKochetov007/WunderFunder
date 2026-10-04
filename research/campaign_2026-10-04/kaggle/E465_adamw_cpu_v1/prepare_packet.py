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
DATASET_ID = "vladkochetov07/wnn-jax-adamw-e465-cpu-20261004-v1"
KERNEL_ID = "vladkochetov07/wnn-jax-adamw-e465-cpu-20261004-v1-runner"
PINNED_SOURCE_HASHES = {
    "scripts/check_e465_adamw_fp32_parity.py": "612435a3ffe899e8c69ea3cb69aec009bcbbe03faa16c0344e1450c427982861",
    "src/wnn_jax/onecycle.py": "f857fe857a361249c2742e3486cd32b846e0f47413a5c4db2ae1fb6c65509ded",
    "src/wnn_jax/rnn.py": "294eff11d906fe434dba2c4d0ed4741ad2191f756c4a35204474cb229ff939dc",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def main() -> None:
    if ARCHIVE.exists():
        raise FileExistsError(f"refusing to overwrite immutable packet archive: {ARCHIVE}")
    members = sorted(
        path
        for path in BUNDLE_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    files = {path.relative_to(BUNDLE_ROOT).as_posix(): path for path in members}
    expected = set(PINNED_SOURCE_HASHES) | {"worker.py"}
    if set(files) != expected:
        raise ValueError(f"unexpected source closure: expected={sorted(expected)} actual={sorted(files)}")
    descriptors = {}
    for relative, path in files.items():
        data = path.read_bytes()
        digest = sha256_bytes(data)
        if relative in PINNED_SOURCE_HASHES and digest != PINNED_SOURCE_HASHES[relative]:
            raise ValueError(f"pinned source changed: {relative} sha256={digest}")
        descriptors[relative] = {"size_bytes": len(data), "sha256": digest}

    with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative, path in files.items():
            info = zipfile.ZipInfo(relative, date_time=(2026, 10, 4, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

    manifest = {
        "dataset_id": DATASET_ID,
        "private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "archive_file": "source_bundle.zip",
        "archive_sha256": sha256_path(ARCHIVE),
        "files": descriptors,
        "classification": "E465 CPU-only FP32 dynamic-beta AdamW parity diagnostic; no model/data/fit/score",
        "parent_environment": {
            "kernel": "vladkochetov07/wnn-jax-sharding-e465-cpu-20261004-v3-runner",
            "runtime_reference": "research/campaign_2026-10-04/kaggle/E465_sharding_cpu_v3/provider_output_complete/e465_sharding_cpu_v3/runtime.json",
            "runtime_sha256": "5c640ab255e596b12af62a7bd71f6223fca721c74f72831d2d33667f36ff5412",
            "parent_source_archive_sha256": "5b0bac21c15ae2b5452568c759532345cbedd1623e5182f720239d987c0b7901",
            "parent_source_manifest_sha256": "f79e41204f296d3a24067c3eeac8574b66b9601374d323cd3c90504a30801d0b",
            "python": "3.13.15",
            "jax": "0.11.1",
            "jaxlib": "0.11.1",
            "numpy": "2.1.3",
            "torch": {
                "version": "2.11.0+cpu",
                "runtime_reference": "research/campaign_2026-10-04/kaggle/jax_parity_v1/live_output_v3/e458_cpu_parity/runtime.json",
                "runtime_sha256": "2fb0c6016c780b4386566675f3d83895fea633ff9d40053d160524343f33e2e9",
            },
        },
        "source_only_closure": sorted(files),
        "raw_data_attached": False,
    }
    manifest_path = UPLOAD / "source-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (UPLOAD / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": DATASET_ID,
                "title": "E465 JAX/Torch FP32 AdamW CPU parity check",
                "isPrivate": True,
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n"
    )

    manifest_hash = sha256_path(manifest_path)
    kernel_path = PACKET / "kernel_push_runner" / "kernel.py"
    kernel = kernel_path.read_text()
    kernel, archive_count = re.subn(
        r'ARCHIVE_SHA256 = "(?:__ARCHIVE_SHA256__|[0-9a-f]{64})"',
        f'ARCHIVE_SHA256 = "{manifest["archive_sha256"]}"',
        kernel,
    )
    kernel, manifest_count = re.subn(
        r'MANIFEST_SHA256 = "(?:__MANIFEST_SHA256__|[0-9a-f]{64})"',
        f'MANIFEST_SHA256 = "{manifest_hash}"',
        kernel,
    )
    if archive_count != 1 or manifest_count != 1:
        raise RuntimeError("kernel hash pins were not fully rendered")
    kernel_path.write_text(kernel)
    (PACKET / "kernel_push_runner" / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": KERNEL_ID,
                "title": "wnn-jax-adamw-e465-cpu-20261004-v1-runner",
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
    print(
        json.dumps(
            {
                "archive_sha256": manifest["archive_sha256"],
                "manifest_sha256": manifest_hash,
                "kernel_sha256": sha256_path(kernel_path),
                "metadata_sha256": sha256_path(PACKET / "kernel_push_runner" / "kernel-metadata.json"),
                "member_count": len(descriptors),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
