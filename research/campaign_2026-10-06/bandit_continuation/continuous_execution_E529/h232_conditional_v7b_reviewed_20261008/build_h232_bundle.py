from __future__ import annotations

import base64
import hashlib
import io
import json
import textwrap
import zipfile
from difflib import unified_diff
from pathlib import Path, PurePosixPath

PAYLOADS = (
    "H232_PROFILE_PLAN.json",
    "H232_AMP4096_SUCCESSOR_RECIPE.json",
    "cache_pressure_recovery.py",
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


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes())


def build_execution_diff(root: Path) -> bytes:
    """Write the reviewable text diff for V7 execution changes against frozen V6."""
    v6_root = root.parent / "h232_conditional_v6"
    pairs = (
        ("prepare_h232_v6_packet.py", "prepare_h232_v7_packet.py"),
        ("build_h232_source_archive.py", "build_h232_source_archive.py"),
        ("build_h232_bundle.py", "build_h232_bundle.py"),
        ("packet/cache_pressure_recovery.py", "packet/cache_pressure_recovery.py"),
        ("packet/profile_worker.py", "packet/profile_worker.py"),
        ("packet/H232_PROFILE_PLAN.json", "packet/H232_PROFILE_PLAN.json"),
        (
            "packet/H232_AMP4096_SUCCESSOR_RECIPE.json",
            "packet/H232_AMP4096_SUCCESSOR_RECIPE.json",
        ),
        ("packet/external_review.json", "packet/external_review.json"),
        ("packet/kernel-metadata.json", "packet/kernel-metadata.json"),
        ("packet/packet_meta.json", "packet/packet_meta.json"),
        ("tests/test_cache_pressure_recovery.py", "tests/test_cache_pressure_recovery.py"),
        ("tests/test_conditional_fit_gate.py", "tests/test_conditional_fit_gate.py"),
    )
    chunks = [
        "# Exact text diff of V7 execution, packet, and test files against frozen V6.\n",
        "# The generated transport wrapper is omitted; rebuild it with build_h232_bundle.py.\n",
        "# V7 source archive and manifest are verified byte-identical to V6 by the build receipt.\n",
    ]
    for old_name, new_name in pairs:
        old_path = v6_root / old_name
        new_path = root / new_name
        old_lines = old_path.read_text(encoding="utf-8").splitlines(keepends=True)
        new_lines = new_path.read_text(encoding="utf-8").splitlines(keepends=True)
        chunks.extend(
            unified_diff(
                old_lines,
                new_lines,
                fromfile=f"V6/{old_name}",
                tofile=f"V7/{new_name}",
                n=0,
            )
        )
    diff_path = root / "V7_V6_EXECUTION_DIFF.patch"
    diff_bytes = "".join(chunks).encode("utf-8")
    diff_path.write_bytes(diff_bytes)
    return diff_bytes


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
        raise RuntimeError("H232 V7 embedded packet archive digest mismatch")
    packet_root = Path("/kaggle/working/h232_cloud_packet_v7_proactive_20261008")
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
            raise RuntimeError(f"H232 V7 embedded member digest mismatch: {{name}}")
    sys.path.insert(0, str(packet_root))
    from profile_kernel import main as run_profile

    run_profile()


if __name__ == "__main__":
    main()
'''
    wrapper_path = packet / "h232_profile_kernel_bundle.py"
    wrapper_path.write_text(wrapper, encoding="utf-8")
    result = {
        "bundle_sha256": sha256(archive_bytes),
        "bundle_bytes": len(archive_bytes),
        "code_file_sha256": sha256(wrapper.encode()),
        "code_file_bytes": len(wrapper.encode()),
        "payload_hashes": hashes,
    }
    root = Path(__file__).resolve().parent
    basis = root / "basis"
    execution_diff = build_execution_diff(root)
    packet_meta = json.loads((packet / "packet_meta.json").read_text(encoding="utf-8"))
    packet_hashes = {
        path.relative_to(packet).as_posix(): sha256_file(path)
        for path in sorted(packet.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    builder_hashes = {
        name: sha256_file(root / name)
        for name in (
            "prepare_h232_v7_packet.py",
            "build_h232_source_archive.py",
            "build_h232_bundle.py",
        )
    }
    test_hashes = {path.name: sha256_file(path) for path in sorted((root / "tests").glob("*.py"))}
    build_receipt = {
        "schema": "h232-v7-proactive-build-receipt-v1",
        "experiment_id": packet_meta["packet_id"],
        "status": "built_pending_independent_review_and_root_admission",
        "provider_push_authorized": False,
        "parent_v6": {
            "terminal_receipt_sha256": sha256_file(basis / "V6_TERMINAL_POLL_RECEIPT.json"),
            "source_archive_sha256": sha256_file(basis / "V6_SOURCE_ARCHIVE.tar"),
            "source_archive_manifest_sha256": sha256_file(
                basis / "V6_SOURCE_ARCHIVE_MANIFEST.json"
            ),
        },
        "source_archive": {
            "sha256": sha256_file(packet / "source_archive.tar"),
            "manifest_sha256": sha256_file(packet / "source_archive_manifest.json"),
            "byte_identical_to_v6": (
                (packet / "source_archive.tar").read_bytes()
                == (basis / "V6_SOURCE_ARCHIVE.tar").read_bytes()
            ),
            "member_count": len(
                json.loads((packet / "source_archive_manifest.json").read_text(encoding="utf-8"))[
                    "members"
                ]
            ),
        },
        "builders": builder_hashes,
        "tests": test_hashes,
        "execution_diff": {
            "path": "V7_V6_EXECUTION_DIFF.patch",
            "sha256": sha256(execution_diff),
            "bytes": len(execution_diff),
        },
        "diagnosis_report_sha256": sha256_file(root / "H232_V7_PROACTIVE_RECOVERY_REPORT_V1.md"),
        "packet_files": packet_hashes,
        "kernel_bundle": result,
        "provider_status": "not_queried",
    }
    (root / "V7_BUILD_RECEIPT.json").write_text(
        json.dumps(build_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    import sys

    print(json.dumps(build(Path(sys.argv[1])), indent=2, sort_keys=True))
