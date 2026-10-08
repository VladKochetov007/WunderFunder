from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
PACKET = BASE / "h217_packet_v4"
V3_PACKET = BASE / "h217_packet_v3.tar"
V3_MANIFEST = BASE / "H217_V3_SOURCE_MANIFEST.json"
V3_REVIEW = BASE / "H217_V3_ROOT_STATIC_REVIEW_V1.json"
IDENTITY = "E529-H217-TCN-RAW-SCRATCH-V4-AMP4096-BOUNDED-DEADLINE"
V3_PACKET_SHA256 = "d58a175d25ebd959fa3523eb3d9d7305a1dc84e36f1f2663355d7eebf4dbda3a"
V3_MANIFEST_SHA256 = "9548066a449cb21f021524279f406daeea4638b9acb9d2cbd7f78d0f00565c6f"
EXPECTED_RUNTIME_ONLY_CHANGES = {"worker.py", "run.sh", "runtime_resources.json"}
EXCLUDED_DIRS = {".git", "__pycache__", "cache", "datasets", "runs"}
EXCLUDED_SUFFIXES = {".pyc", ".parquet", ".npy", ".npz", ".pt", ".tar"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def included_files() -> list[Path]:
    paths = [
        PACKET / name for name in ("recipe.json", "runtime_resources.json", "worker.py", "run.sh")
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


def archive_members(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, "r") as archive:
        members: dict[str, bytes] = {}
        for member in archive.getmembers():
            if not member.isfile():
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read predecessor member: {member.name}")
            members[member.name] = source.read()
    return members


def verify_successor(files: list[Path]) -> list[str]:
    previous = archive_members(V3_PACKET)
    current = {path.relative_to(PACKET).as_posix(): path.read_bytes() for path in files}
    if set(current) != set(previous):
        raise ValueError("V4 packet member set differs from the reviewed V3 packet")
    changed = sorted(name for name in current if current[name] != previous[name])
    if set(changed) != EXPECTED_RUNTIME_ONLY_CHANGES:
        raise ValueError(f"unexpected V3-to-V4 source diff: {changed}")
    return changed


def build_archive(files: list[Path], destination: Path) -> None:
    with tarfile.open(destination, "w", format=tarfile.PAX_FORMAT) as archive:
        for path in files:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=BASE / "h217_packet_v4.tar")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if sha256(V3_PACKET) != V3_PACKET_SHA256 or sha256(V3_MANIFEST) != V3_MANIFEST_SHA256:
        raise ValueError("frozen V3 packet or source-manifest identity differs from root review")
    review = json.loads(V3_REVIEW.read_text())
    if review.get("status") != "static_recipe_review_passed_not_provider_admitted":
        raise ValueError("V3 root static review is not a passing predecessor")
    recipe = json.loads((PACKET / "recipe.json").read_text())
    runtime = json.loads((PACKET / "runtime_resources.json").read_text())
    if recipe["numerical_successor"]["v3_explicit_value"] != 4096:
        raise ValueError("V4 must preserve the reviewed V3 AMP4096 science")
    files = included_files()
    changed = verify_successor(files)
    build_archive(files, args.output)
    repo = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "identity": IDENTITY,
        "base_commit": base_commit,
        "predecessor": {
            "v3_packet_sha256": V3_PACKET_SHA256,
            "v3_source_manifest_sha256": V3_MANIFEST_SHA256,
            "v3_root_review_sha256": sha256(V3_REVIEW),
        },
        "science": {
            "numerical_successor": recipe["numerical_successor"],
            "recipe_sha256": sha256(PACKET / "recipe.json"),
            "trainer_and_amp_code_unchanged_from_v3": True,
            "fit_sources_and_argv_unchanged_from_v3": True,
            "runtime_only_member_changes": changed,
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
    manifest_path = BASE / "H217_V4_SOURCE_MANIFEST.json"
    write_json(manifest_path, manifest)
    manifest_sha = sha256(manifest_path)
    receipt = {
        "schema_version": 1,
        "status": "prepared_for_root_review_not_admitted",
        "identity": IDENTITY,
        "build_command": f"python3 {Path(__file__).resolve().relative_to(repo)} --output {args.output.relative_to(repo)}",
        "builder_sha256": sha256(Path(__file__).resolve()),
        "source_manifest": str(manifest_path.relative_to(repo)),
        "source_manifest_sha256": manifest_sha,
        "packet": str(args.output.relative_to(repo)),
        "packet_sha256": sha256(args.output),
        "member_count": len(files),
        "changed_runtime_members": changed,
        "unchanged_scientific_inputs": {
            "recipe_sha256": sha256(PACKET / "recipe.json"),
            "training_source_path": "h217_packet_v4/project/scripts/train_rnn.py",
            "amp_helper_path": "h217_packet_v4/project/src/wnn/amp.py",
            "amp_initial_scale": 4096,
        },
        "execution_scope": "No provider query/create/rental, profile, fit, export, or grade was performed. Root review and fresh provider/account/quota admission remain required.",
    }
    write_json(BASE / "H217_V4_BUILD_RECEIPT.json", receipt)
    admission = {
        "schema_version": 1,
        "status": "not_admitted",
        "attempt_id": "e529-vast-h217-v4-20261008",
        "identity": IDENTITY,
        "packet_sha256": sha256(args.output),
        "source_manifest_sha256": manifest_sha,
        "requested_worker_max_seconds": int(runtime["cohort_max_runtime_seconds"]),
        "incoming_runtime_policy": "Supervisor may pass any positive integer no greater than 21600 seconds; worker computes its phase deadline from that actual incoming value after setup and terminal-packaging reserves.",
        "recipe": "h217_packet_v4/recipe.json",
        "runtime_contract": "h217_packet_v4/runtime_resources.json",
        "resource_scope": "Prepared packet only. No rental, provider action, quota change, or launch authorization.",
        "scientific_change": "None in V4. V3 AMP initial scale 4096, both raw-only scratch arms, sources, data, targets, optimizer, schedule, batch, precision and fixed epoch3 remain unchanged.",
        "fit_eligibility": "Each arm requires a passing strict finite-update profile. Use that arm's profile-projected fit estimate plus its valid-export phase cap against actual time remaining; later arms are rechecked after each terminal phase. Per-profile, fit and export caps stay 900, 7200 and 1800 seconds.",
        "required_independent_checks": [
            "Root verifies the V3-to-V4 diff contains only worker/run-shell/runtime-accounting changes and confirms exact AMP4096 training sources and recipe.",
            "Fresh market/account/owned-instance/quota/headroom and staging-duration admission before any provider create.",
            "Pass the actual post-staging remaining runtime unchanged to WNN_MAX_RUNTIME_SECONDS; it must be positive and no greater than 21600.",
            "Preserve V3 strict profile/failure isolation, fixed-epoch3 valid-only FP32 prediction exports and root-owned grading.",
        ],
    }
    write_json(BASE / "H217_V4_ADMISSION_REQUEST.json", admission)
    print(
        json.dumps(
            {
                "packet_sha256": receipt["packet_sha256"],
                "source_manifest_sha256": manifest_sha,
                "member_count": len(files),
                "changed_members": changed,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
