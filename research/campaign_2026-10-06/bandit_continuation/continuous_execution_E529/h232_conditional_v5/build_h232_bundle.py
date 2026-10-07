from __future__ import annotations

import base64
import hashlib
import io
import json
import textwrap
import zipfile
from pathlib import Path, PurePosixPath

PAYLOADS = (
    "H232_PROFILE_PLAN.json",
    "H232_AMP4096_SUCCESSOR_RECIPE.json",
    "external_review.json",
    "gpu_memory_domains.py",
    "packet_meta.json",
    "profile_kernel.py",
    "profile_worker.py",
    "prior_input_identity_v2.json",
    "source_archive_manifest.json",
    "source_archive.tar",
    "verified_digest_handoff.py",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(packet: Path) -> dict[str, object]:
    payload_map = {name: (packet / name).read_bytes() for name in PAYLOADS}
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(
        archive_buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, data in sorted(payload_map.items()):
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe packet path: {name}")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compresslevel=9)
    archive_bytes = archive_buffer.getvalue()
    hashes = {name: sha256(data) for name, data in sorted(payload_map.items())}
    encoded = base64.b64encode(archive_bytes).decode("ascii")
    lines = "\n".join(f"    {line!r}" for line in textwrap.wrap(encoded, 100))
    wrapper = f'''"""Self-contained, hash-checked H232 conditional profile-to-fit transport."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WNN_LOADER_WORKERS"] = "2"
os.environ["WNN_LOADER_RAM_GB"] = "0.30"
os.environ["WNN_GPU_FRACTION"] = "0.75"
os.environ["WNN_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

PAYLOAD_ZIP_SHA256 = {sha256(archive_bytes)!r}
PAYLOAD_FILE_SHA256 = {hashes!r}
PAYLOAD_B64 = (
{lines}
)


def main() -> None:
    """Verify and materialize only the exact embedded packet, then run its conditional profile/fit gate."""
    archive_bytes = base64.b64decode(PAYLOAD_B64, validate=True)
    if hashlib.sha256(archive_bytes).hexdigest() != PAYLOAD_ZIP_SHA256:
        raise RuntimeError("H232 V5 embedded packet archive digest mismatch")
    packet_root = Path("/kaggle/working/h232_cloud_packet_v5")
    packet_root.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
        for item in archive.infolist():
            relative = PurePosixPath(item.filename)
            mode = item.external_attr >> 16
            if relative.is_absolute() or ".." in relative.parts or stat.S_ISLNK(mode):
                raise RuntimeError(f"unsafe embedded path: {{item.filename}}")
            destination = packet_root.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(item))
    for name, expected in PAYLOAD_FILE_SHA256.items():
        observed = hashlib.sha256((packet_root / name).read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f"H232 V5 embedded member digest mismatch: {{name}}")
    sys.path.insert(0, str(packet_root))
    from profile_kernel import main as run_profile

    run_profile()


if __name__ == "__main__":
    main()
'''
    wrapper_path = packet / "h232_profile_kernel_bundle.py"
    wrapper_path.write_text(wrapper, encoding="utf-8")
    return {
        "bundle_sha256": sha256(archive_bytes),
        "bundle_bytes": len(archive_bytes),
        "code_file_sha256": sha256(wrapper.encode()),
        "code_file_bytes": len(wrapper.encode()),
        "payload_hashes": hashes,
    }


if __name__ == "__main__":
    import sys

    print(json.dumps(build(Path(sys.argv[1])), indent=2, sort_keys=True))
