from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# JSON objects here have heterogeneous, contract-controlled nested values.

ROOT = Path(__file__).resolve().parents[3]
PACKET = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r2"


def sha256_file(path: Path) -> str:
    """Hash a bundle artifact in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a small, locally generated JSON manifest."""
    return json.loads(path.read_text())


def main() -> None:
    """Build and verify the bound private source bundle without uploading it."""
    binding = read_json(PACKET / "SOURCE_BINDING_R2.json")
    command = binding["offline_bundle_command"]
    if not isinstance(command, list) or command[:2] != [
        "/usr/bin/python3",
        "scripts/kaggle_bundle.py",
    ]:
        raise ValueError("source binding has an unexpected offline bundle command")
    destination = Path(command[command.index("--destination") + 1])
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite bundle destination {destination}")
    started = datetime.now(UTC)
    subprocess.run(command, cwd=ROOT, check=True)
    manifest_path = destination / "bundle-manifest.json"
    archive_path = destination / "bundle.wnn"
    manifest = read_json(manifest_path)
    if (
        manifest["dataset"] != binding["kernel_dataset"]
        or manifest["private"] is not True
    ):
        raise ValueError(
            "built source dataset identity/privacy differs from sealed binding"
        )
    expected_mounts = {
        row["logical_path"]: {"dataset": row["dataset"], "filename": row["filename"]}
        for row in read_json(PACKET / "INPUT_BINDING_R2.json")["external_mounts"]
        if row["dataset"] != "vladkochetov07/wunder-funder"
    }
    if manifest["external_mounts"] != expected_mounts | {
        row["logical_path"]: row["filename"]
        for row in read_json(PACKET / "INPUT_BINDING_R2.json")["external_mounts"]
        if row["dataset"] == "vladkochetov07/wunder-funder"
    }:
        raise ValueError("built bundle mounts differ from the exact input binding")
    receipt = {
        "schema": "wnn.student-shape-pair-offline-bundle-build.r2",
        "status": "built_offline_private_profile_source_no_provider_action",
        "provider_action_performed": False,
        "start_utc": started.isoformat(),
        "end_utc": datetime.now(UTC).isoformat(),
        "command": command,
        "source_binding_sha256": sha256_file(PACKET / "SOURCE_BINDING_R2.json"),
        "input_binding_sha256": sha256_file(PACKET / "INPUT_BINDING_R2.json"),
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "archive_path": str(archive_path.relative_to(ROOT)),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "expanded_bytes": manifest["expanded_bytes"],
        "mount_count": len(manifest["external_mounts"]),
        "teacher_mounts": [
            {
                "logical_path": row["logical_path"],
                "dataset": row["dataset"],
                "filename": row["filename"],
                "sha256": row["sha256"],
                "size": row["size"],
            }
            for row in read_json(PACKET / "INPUT_BINDING_R2.json")["external_mounts"]
            if row["dataset"] == binding["teacher_cp2_asset_dataset"]
        ],
    }
    (PACKET / "OFFLINE_BUNDLE_BUILD_R2.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "archive_sha256": receipt["archive_sha256"],
                "archive_bytes": receipt["archive_bytes"],
                "mount_count": receipt["mount_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
