"""Build and stage a create-only, source-bound R2 preflight project on B."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from bindings import (
    B_EXPORT_PREP,
    B_PROJECT_ROOT,
    B_PYTHON,
    EXECUTION_DIR,
    EXECUTION_REL,
    MEMBERSHIP,
    MEMBERSHIP_REL,
    OLD_SPLIT_REL,
    OVERLAY_DIR,
    OVERLAY_MANIFEST,
    OVERLAY_REL,
    R1_REVIEW,
    R1_REVIEW_REL,
    R2_REVIEW,
    REGISTRY,
    REGISTRY_REL,
    SNAPSHOT_DIR,
    SPLIT,
    build_stage_binding,
    sha256_file,
    verify_local_prerequisites,
)
from local_common import (
    create_record_dir,
    parse_remote_json,
    run_ssh,
    utc_now,
    verify_remote_file_sha256,
    write_json_exclusive,
)


def _add_tree(archive: tarfile.TarFile, source: Path, archive_name: str) -> None:
    """Add a pinned tree without following or preserving symlinks."""
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"refusing to package symlink: {path}")
    archive.add(source, arcname=archive_name, recursive=True)


def _build_local_bundle(bundle_path: Path, binding: dict[str, Any]) -> int:
    """Create a small source-only tarball containing exact reviewed files."""
    manifest = json.loads(OVERLAY_MANIFEST.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="preflight-r2-stage-") as temp_name:
        stage_tree = Path(temp_name) / "bundle"
        stage_tree.mkdir()
        with tarfile.open(bundle_path, "w:gz") as archive:
            _add_tree(archive, SNAPSHOT_DIR, "frozen_source_snapshot")
            overlay_copy = stage_tree / OVERLAY_REL
            overlay_copy.mkdir(parents=True)
            shutil.copy2(OVERLAY_MANIFEST, overlay_copy / OVERLAY_MANIFEST.name)
            for entry in manifest["source_files"] + manifest["support_files"]:
                source = OVERLAY_DIR / entry["path"]
                destination = overlay_copy / entry["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            archive.add(overlay_copy, arcname=OVERLAY_REL.as_posix(), recursive=True)
            files = (
                (SPLIT, OLD_SPLIT_REL),
                (REGISTRY, REGISTRY_REL),
                (MEMBERSHIP, MEMBERSHIP_REL),
                (R1_REVIEW, R1_REVIEW_REL),
                (R2_REVIEW, EXECUTION_REL / "INDEPENDENT_REVIEW_R2.json"),
                (
                    EXECUTION_DIR / "remote_stage.py",
                    EXECUTION_REL / "remote_stage.py",
                ),
            )
            for source, relative in files:
                archive.add(source, arcname=relative.as_posix(), recursive=False)
            binding_path = stage_tree / (EXECUTION_REL / "STAGE_BINDING_R2.json")
            binding_path.parent.mkdir(parents=True, exist_ok=True)
            binding_path.write_text(
                json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            archive.add(binding_path, arcname=(EXECUTION_REL / "STAGE_BINDING_R2.json").as_posix())
    return bundle_path.stat().st_size


def stage(attempt_id: str, alias: str) -> Path:
    """Stage code and metadata, then create and verify exact B-side links."""
    record_dir = create_record_dir(attempt_id)
    phase = "local_prerequisite_verification"
    try:
        local_pins = verify_local_prerequisites()
        binding = build_stage_binding(attempt_id)
        binding_path = record_dir / "STAGE_BINDING_R2.json"
        write_json_exclusive(binding_path, binding)
        with tempfile.TemporaryDirectory(prefix="preflight-r2-archive-") as temp_name:
            archive_path = Path(temp_name) / "stage-bundle.tar.gz"
            phase = "local_bundle_construction"
            bytes_sent = _build_local_bundle(archive_path, binding)
            phase = "remote_create_only_root"
            create_root = (
                "set -eu; umask 077; "
                f"mkdir -p -- {shlex.quote(str(B_EXPORT_PREP))}; "
                f"test ! -L {shlex.quote(str(B_EXPORT_PREP))}; "
                f"test ! -e {shlex.quote(str(B_PROJECT_ROOT))}; "
                f"test ! -L {shlex.quote(str(B_PROJECT_ROOT))}; "
                f"mkdir -m 700 -- {shlex.quote(str(B_PROJECT_ROOT))}"
            )
            created = run_ssh(alias, ["sh", "-c", create_root], timeout=30)
            if created.returncode != 0:
                raise RuntimeError(
                    "remote create-only root failed: "
                    + created.stderr.decode("utf-8", errors="replace")[-2000:]
                )
            phase = "small_bundle_transfer"
            tar_command = [
                "tar",
                "--extract",
                "--no-same-owner",
                "--keep-old-files",
                "--file=-",
                f"--directory={B_PROJECT_ROOT}",
            ]
            transferred = run_ssh(
                alias, tar_command, timeout=120, input_bytes=archive_path.read_bytes()
            )
            if transferred.returncode != 0:
                raise RuntimeError(
                    "source-only bundle transfer failed: "
                    + transferred.stderr.decode("utf-8", errors="replace")[-2000:]
                )
            phase = "remote_link_stage_and_verification"
            helper = B_PROJECT_ROOT / EXECUTION_REL / "remote_stage.py"
            verify_remote_file_sha256(
                alias,
                helper,
                binding["remote"]["stage_helper"]["sha256"],
            )
            verified = run_ssh(
                alias,
                [
                    str(B_PYTHON),
                    "-B",
                    str(helper),
                    "--mode",
                    "stage",
                    "--attempt-id",
                    attempt_id,
                    "--root",
                    str(B_PROJECT_ROOT),
                ],
                timeout=180,
            )
            if verified.returncode != 0:
                raise RuntimeError(
                    "remote stage verification failed: "
                    + verified.stderr.decode("utf-8", errors="replace")[-4000:]
                )
            remote_receipt = parse_remote_json(verified.stdout)
            if remote_receipt.get("status") != "STAGED":
                raise ValueError("remote helper did not report create-only staging")
            remote_raw = verified.stdout.rstrip(b"\r\n")
            with (record_dir / "STAGE_REMOTE_RECEIPT.json").open("xb") as stream:
                stream.write(remote_raw)
                stream.write(b"\n")
            write_json_exclusive(
                record_dir / "STAGE.json",
                {
                    "schema": "quality_gap.preflight_stage_execution.r2",
                    "status": "STAGED",
                    "attempt_id": attempt_id,
                    "ssh_alias": alias,
                    "created_utc": utc_now(),
                    "remote_project_root": str(B_PROJECT_ROOT),
                    "local_verified_pins": local_pins,
                    "bundle_bytes": bytes_sent,
                    "remote_stage_receipt_sha256": sha256_file(
                        record_dir / "STAGE_REMOTE_RECEIPT.json"
                    ),
                },
            )
        return record_dir
    except Exception as exc:
        failure = {
            "schema": "quality_gap.preflight_stage_execution.r2",
            "status": "STAGE_FAILED",
            "attempt_id": attempt_id,
            "ssh_alias": alias,
            "phase": phase,
            "created_utc": utc_now(),
            "error": str(exc),
        }
        write_json_exclusive(record_dir / "STAGE_FAILURE.json", failure)
        raise


def main() -> int:
    """Create one local record and run the explicit B-side staging sequence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-alias", default="beleriand")
    parser.add_argument(
        "--attempt-id", required=True, help="Use a fresh preflight-r2-... identifier"
    )
    args = parser.parse_args()
    result = stage(args.attempt_id, args.ssh_alias)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
