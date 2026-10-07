"""Build an E520 profile packet only after independent external review."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import textwrap
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

PACKET_ID = "E520_PROFILE_VERIFIED_DIGEST_HANDOFF_V7"
KAGGLE_ID = "vladkochetov07/e520-profile-v7-digest-handoff"
KAGGLE_TITLE = "E520 Profile V7 Digest Handoff"
PROFILE_WALL_CAP_S = 1_700
PINNED_COMMIT = "69b2331825e193a070015ca4182872724595209a"
PINNED_ARCHIVE_SHA256 = "78f9e5c3d001b8a0254ed535846f07b3b9a906adba2a270f7ddc1ab6606081fc"
PINNED_TRAINER_SHA256 = "8a29459e998f7f5d6a03b7a4b87c6f4b569002e65b8c87127c37138360909cbf"
BUNDLE_PAYLOAD_FILES = (
    "E520_PACKET_PLAN.json",
    "export_guards.py",
    "external_review.json",
    "kernel-metadata.json",
    "packet_meta.json",
    "prepare_profile_packet.py",
    "profile_kernel.py",
    "profile_worker.py",
    "source_archive.tar",
    "tests/conftest.py",
    "tests/test_portable_manifest_binding.py",
    "tests/test_profile_bundle.py",
    "tests/test_loader_hash_advice.py",
    "verified_digest_handoff.py",
    "tests/test_digest_handoff_integration.py",
    "tests/test_profile_guards.py",
)
REQUIRED_REVIEWED = (
    "profile_worker.py",
    "profile_kernel.py",
    "prepare_profile_packet.py",
    "export_guards.py",
    "tests/conftest.py",
    "tests/test_profile_guards.py",
    "tests/test_portable_manifest_binding.py",
    "tests/test_profile_bundle.py",
    "tests/test_loader_hash_advice.py",
    "verified_digest_handoff.py",
    "tests/test_digest_handoff_integration.py",
    "E520_PACKET_PLAN.json",
    "source_archive.tar",
)


def sha256_path(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def render_profile_kernel_bundle(payloads: dict[str, bytes]) -> str:
    """Render a deterministic, self-contained Kaggle wrapper for reviewed inputs.

    Args:
        payloads: Relative packet paths mapped to their exact bytes.

    Returns:
        Python source embedding a deterministic ZIP and per-file SHA-256 map.

    Raises:
        ValueError: If a payload path is unsafe or no payload was supplied.
    """
    if not payloads:
        raise ValueError("profile bundle needs at least one payload file")
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(
        archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, content in sorted(payloads.items()):
            relative = PurePosixPath(name)
            if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                raise ValueError(f"unsafe profile bundle path: {name}")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content, compresslevel=9)
    archive_bytes = archive_buffer.getvalue()
    file_hashes = {
        name: hashlib.sha256(content).hexdigest() for name, content in sorted(payloads.items())
    }
    archive_hash = hashlib.sha256(archive_bytes).hexdigest()
    encoded = base64.b64encode(archive_bytes).decode("ascii")
    encoded_lines = "\n".join(f"    {line!r}" for line in textwrap.wrap(encoded, 100))
    return f'''"""Self-contained transport wrapper for the reviewed E520 v7 profile packet."""

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
os.environ["WNN_LOADER_WORKERS"] = "0"
os.environ["WNN_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

PAYLOAD_ZIP_SHA256 = {archive_hash!r}
PAYLOAD_FILE_SHA256 = {file_hashes!r}
PAYLOAD_B64 = (
{encoded_lines}
)


def main() -> None:
    """Verify and extract only the reviewed packet, then run its profile worker."""
    archive_bytes = base64.b64decode(PAYLOAD_B64, validate=True)
    if hashlib.sha256(archive_bytes).hexdigest() != PAYLOAD_ZIP_SHA256:
        raise RuntimeError("embedded E520 packet archive digest mismatch")
    packet_root = Path("/kaggle/working/e520_profile_packet_v7")
    packet_root.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
        for item in archive.infolist():
            relative = PurePosixPath(item.filename)
            mode = item.external_attr >> 16
            if relative.is_absolute() or ".." in relative.parts or stat.S_ISLNK(mode):
                raise RuntimeError(f"unsafe embedded packet path: {{item.filename}}")
            destination = packet_root.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(item))
    for name, expected in PAYLOAD_FILE_SHA256.items():
        observed = hashlib.sha256((packet_root / name).read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f"embedded E520 v7 file hash mismatch: {{name}}")
    sys.path.insert(0, str(packet_root))
    from profile_worker import run_profile

    result = run_profile(packet_root)
    print(
        f"E520 profile status={{result['status']}} "
        f"wall_seconds={{result['wall_seconds']:.1f}}",
        flush=True,
    )


if __name__ == "__main__":
    main()
'''


def write_profile_kernel_bundle(packet_root: Path) -> str:
    """Write the deterministic self-contained wrapper from reviewed packet files.

    Args:
        packet_root: Staged packet containing every declared bundle member.

    Returns:
        SHA-256 of the emitted wrapper source.

    Raises:
        FileNotFoundError: If a required packet member is absent.
    """
    payloads = {name: (packet_root / name).read_bytes() for name in BUNDLE_PAYLOAD_FILES}
    bundle_path = packet_root / "profile_kernel_bundle.py"
    bundle_path.write_text(render_profile_kernel_bundle(payloads), encoding="utf-8")
    return sha256_path(bundle_path)


def archive_pinned_source(repo_root: Path, target: Path) -> str:
    """Create the exact previously used v5 source archive, refusing drift."""
    command = [
        "git",
        "-C",
        str(repo_root),
        "archive",
        "--format=tar",
        PINNED_COMMIT,
        "scripts/train_rnn.py",
        "src/wnn",
    ]
    with target.open("wb") as output:
        subprocess.run(command, check=True, stdout=output, stderr=subprocess.PIPE, timeout=30)
    digest = sha256_path(target)
    if digest != PINNED_ARCHIVE_SHA256:
        raise ValueError(f"pinned v5 source archive hash mismatch: {digest}")
    with tempfile.TemporaryDirectory(prefix="e520-source-check-") as temporary:
        subprocess.run(
            ["tar", "-xf", str(target), "-C", temporary, "scripts/train_rnn.py"],
            check=True,
            timeout=30,
        )
        if sha256_path(Path(temporary) / "scripts/train_rnn.py") != PINNED_TRAINER_SHA256:
            raise ValueError("pinned source archive contains another train_rnn.py")
    return digest


def review_targets(repo_root: Path, plan_path: Path, archive_path: Path) -> dict[str, str]:
    """Compute review hashes without creating a packet or calling Kaggle."""
    archive_pinned_source(repo_root, archive_path)
    return {
        "profile_worker.py": sha256_path(plan_path.parent / "profile_worker.py"),
        "profile_kernel.py": sha256_path(plan_path.parent / "profile_kernel.py"),
        "prepare_profile_packet.py": sha256_path(Path(__file__).resolve()),
        "export_guards.py": sha256_path(plan_path.parent / "export_guards.py"),
        "tests/conftest.py": sha256_path(plan_path.parent / "tests/conftest.py"),
        "tests/test_profile_guards.py": sha256_path(
            plan_path.parent / "tests/test_profile_guards.py"
        ),
        "tests/test_portable_manifest_binding.py": sha256_path(
            plan_path.parent / "tests/test_portable_manifest_binding.py"
        ),
        "tests/test_profile_bundle.py": sha256_path(
            plan_path.parent / "tests/test_profile_bundle.py"
        ),
        "tests/test_loader_hash_advice.py": sha256_path(
            plan_path.parent / "tests/test_loader_hash_advice.py"
        ),
        "verified_digest_handoff.py": sha256_path(plan_path.parent / "verified_digest_handoff.py"),
        "tests/test_digest_handoff_integration.py": sha256_path(
            plan_path.parent / "tests/test_digest_handoff_integration.py"
        ),
        "E520_PACKET_PLAN.json": sha256_path(plan_path),
        "source_archive.tar": sha256_path(archive_path),
    }


def validate_review(
    receipt_bytes: bytes, expected_sha256: str, target_hashes: dict[str, str]
) -> dict[str, Any]:
    """Require the controller's exact receipt hash and reviewed code map."""
    if len(expected_sha256) != 64 or hashlib.sha256(receipt_bytes).hexdigest() != expected_sha256:
        raise ValueError("an exact independent-controller review receipt hash is required")
    receipt = json.loads(receipt_bytes)
    reviewer = receipt.get("reviewer", {})
    if (
        receipt.get("experiment_id") != PACKET_ID
        or receipt.get("scope") != "profile-only"
        or receipt.get("status") != "approved"
        or reviewer.get("role") != "independent-controller"
        or not reviewer.get("id")
        or reviewer.get("id") == "weak_diversity_kaggle_actor"
    ):
        raise ValueError("review receipt is not an independent profile-only approval")
    if receipt.get("reviewed_hashes") != target_hashes:
        raise ValueError("independent review did not approve these exact source/code hashes")
    return receipt


def build_packet(
    repo_root: Path,
    plan_path: Path,
    output_dir: Path,
    review_path: Path,
    review_sha256: str,
    archive_path: Path,
) -> None:
    """Stage a private profile-only Kaggle script packet after external approval."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite an existing E520 packet directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    target_hashes = review_targets(repo_root, plan_path, archive_path)
    review = validate_review(review_path.read_bytes(), review_sha256, target_hashes)
    for name in REQUIRED_REVIEWED[:-1]:
        destination = output_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plan_path.parent / name, destination)
    shutil.copy2(archive_path, output_dir / "source_archive.tar")
    shutil.copy2(review_path, output_dir / "external_review.json")
    metadata = {
        "packet_id": PACKET_ID,
        "profile_only": True,
        "fit_enabled": False,
        "prediction_export_enabled": False,
        "profile_worker_wall_cap_seconds": PROFILE_WALL_CAP_S,
        "source_commit": PINNED_COMMIT,
        "source_archive_sha256": target_hashes["source_archive.tar"],
        "trainer_sha256": PINNED_TRAINER_SHA256,
        "external_review_sha256": review_sha256,
        "reviewed_hashes": target_hashes,
        "reviewer_id": review["reviewer"]["id"],
    }
    (output_dir / "packet_meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    kaggle_meta = {
        "id": KAGGLE_ID,
        "title": KAGGLE_TITLE,
        "code_file": "profile_kernel_bundle.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "dataset_sources": [
            "vladkochetov07/wunder-funder",
            "vladkochetov07/wnn-c20261004-fold0-assets",
        ],
        "competition_sources": [],
    }
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(kaggle_meta, indent=2, sort_keys=True) + "\n"
    )
    write_profile_kernel_bundle(output_dir)


def main() -> int:
    """Expose separate review-target and gated build modes; never submit a kernel."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--review-targets-only", action="store_true")
    mode.add_argument("--build", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--plan", type=Path, default=Path(__file__).with_name("E520_PACKET_PLAN.json")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--review-receipt", type=Path)
    parser.add_argument("--review-sha256")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="e520-profile-plan-") as temporary:
        archive = Path(temporary) / "source_archive.tar"
        hashes = review_targets(args.repo_root, args.plan, archive)
        if args.review_targets_only:
            print(
                json.dumps(
                    {
                        "packet_id": PACKET_ID,
                        "status": "review-targets-only-no-packet-created",
                        "reviewed_hashes": hashes,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if not args.output_dir or not args.review_receipt or not args.review_sha256:
            parser.error(
                "--build requires --output-dir, --review-receipt and controller-published --review-sha256"
            )
        # build_packet rechecks archive and review before staging.
        build_packet(
            args.repo_root,
            args.plan,
            args.output_dir,
            args.review_receipt,
            args.review_sha256,
            archive,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI is a staging tool
    raise SystemExit(main())
