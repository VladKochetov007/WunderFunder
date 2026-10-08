from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

PACKET = Path(__file__).resolve().parent / "h217_packet_v3"
IDENTITY = "E529-H217-TCN-RAW-SCRATCH-V3-AMP4096"
V2_PACKET_SHA256 = "401e3a7e5315ad851f3b7aae3135fcf81e9dda71420d1fd26a6b595f197745b6"
ARTIFACT_A3D_SHA256 = "5ba1f584b73b12546e34b6c511166ad1194bc8e69e59473e3dce587eecc11dbb"
EXCLUDED_DIRS = {".git", "__pycache__", "cache", "datasets", "runs"}
EXCLUDED_SUFFIXES = {".pyc", ".parquet", ".npy", ".npz", ".pt", ".tar"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def included_files() -> list[Path]:
    paths = [
        PACKET / "recipe.json",
        PACKET / "runtime_resources.json",
        PACKET / "worker.py",
        PACKET / "run.sh",
    ]
    project = PACKET / "project"
    for path in sorted(project.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"runtime source must not contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(project)
        if (
            any(part in EXCLUDED_DIRS for part in relative.parts)
            or path.suffix in EXCLUDED_SUFFIXES
        ):
            continue
        paths.append(path)
    if len(paths) < 10:
        raise ValueError("project source set is unexpectedly small")
    return sorted(paths, key=lambda path: path.relative_to(PACKET).as_posix())


def build_archive(paths: list[Path], destination: Path) -> None:
    with tarfile.open(destination, "w", format=tarfile.PAX_FORMAT) as archive:
        for path in paths:
            relative = path.relative_to(PACKET).as_posix()
            metadata = path.stat()
            info = tarfile.TarInfo(relative)
            info.size = metadata.st_size
            info.mode = metadata.st_mode & 0o777
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = 0
            with path.open("rb") as stream:
                archive.addfile(info, stream)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PACKET.parent / "h217_packet_v3.tar")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not (PACKET / "recipe.json").is_file():
        raise FileNotFoundError(PACKET / "recipe.json")
    recipe = json.loads((PACKET / "recipe.json").read_text())
    runtime = json.loads((PACKET / "runtime_resources.json").read_text())
    files = included_files()
    build_archive(files, args.output)
    repo = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "identity": IDENTITY,
        "base_commit": base_commit,
        "scientific_factor": recipe["numerical_successor"],
        "basis": {
            "v2_packet_sha256": V2_PACKET_SHA256,
            "a3d_terminal_artifact_sha256": ARTIFACT_A3D_SHA256,
        },
        "packet_members": {
            path.relative_to(PACKET).as_posix(): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        },
        "runtime_contract": runtime,
        "packet_sha256": sha256(args.output),
    }
    manifest_path = PACKET.parent / "H217_V3_SOURCE_MANIFEST.json"
    write_json(manifest_path, manifest)
    manifest_sha256 = sha256(manifest_path)
    receipt = {
        "schema_version": 1,
        "status": "prepared_for_independent_root_review",
        "identity": IDENTITY,
        "build_command": f"python3 {Path(__file__).resolve().relative_to(repo)} --output {args.output.relative_to(repo)}",
        "builder_sha256": sha256(Path(__file__).resolve()),
        "source_manifest": str(manifest_path.relative_to(repo)),
        "source_manifest_sha256": manifest_sha256,
        "packet": str(args.output.relative_to(repo)),
        "packet_sha256": sha256(args.output),
        "member_count": len(files),
        "member_hashes": manifest["packet_members"],
        "scientific_source_diff_contract": "The only trainer change is explicit FP16 GradScaler initial scale 4096; all model, split, target, optimizer, schedule, batch, precision, independent initialization and fixed epoch-3 settings remain frozen from V2. Strict finite-update profile checks remain enabled.",
        "execution_contract": "No fit, profile, rental or provider operation was performed by this build. Fresh independent review and then fresh provider/account/quota admission are required before any create.",
    }
    receipt_path = PACKET.parent / "H217_V3_BUILD_RECEIPT.json"
    write_json(receipt_path, receipt)
    admission = {
        "schema_version": 1,
        "status": "not_admitted",
        "attempt_id": "e529-vast-h217-v3-20261008",
        "identity": IDENTITY,
        "packet_sha256": sha256(args.output),
        "source_manifest_sha256": manifest_sha256,
        "requested_worker_max_seconds": int(runtime["cohort_max_runtime_seconds"]),
        "recipe": str((PACKET / "recipe.json").relative_to(repo)),
        "runtime_contract": str((PACKET / "runtime_resources.json").relative_to(repo)),
        "resource_scope": "One campaign-owned Vast worker only after separate root review and fresh account/market/quota/headroom checks. No retry is implied by this prepared request.",
        "arms": [
            "TCN128x12 raw-only seed7 fold0 B8 scratch",
            "GRU256x2 raw-only seed7 fold0 B8 scratch",
        ],
        "per_arm_budget_seconds": {
            "profile": int(runtime["profile_timeout_seconds"]),
            "fit": int(runtime["fit_timeout_seconds"]),
            "valid_export": int(runtime["valid_export_timeout_seconds"]),
        },
        "maximum_worker_seconds": int(runtime["cohort_max_runtime_seconds"]),
        "scientific_change": "Explicit FP16 GradScaler initial scale4096 in profile and fit, an intentional numerical successor to V2's default65536; no finite-check relaxation.",
        "required_independent_checks": [
            "Verify frozen packet/source hashes and root-reviewed fit source.",
            "Confirm environment and data/fold membership, actual market/account headroom, and sufficient worker plus provider quota.",
            "Run each arm's B8 profile independently; retain failure evidence and allow a passing arm to proceed within fixed cohort headroom.",
            "Export only fixed epoch3 valid predictions in FP32. No train export, fit-time grade, public forecast, or teammate weights.",
        ],
    }
    write_json(PACKET.parent / "H217_V3_ADMISSION_REQUEST.json", admission)
    print(
        json.dumps(
            {
                "packet_sha256": receipt["packet_sha256"],
                "source_manifest_sha256": manifest_sha256,
                "member_count": len(files),
                "receipt": str(receipt_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
