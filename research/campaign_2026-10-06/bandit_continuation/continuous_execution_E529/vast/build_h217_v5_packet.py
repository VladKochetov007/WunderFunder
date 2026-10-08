from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import tarfile
import types
from pathlib import Path

BASE = Path(__file__).resolve().parent
PACKET = BASE / "h217_packet_v5"
V4_PACKET_DIR = BASE / "h217_packet_v4"
V4_ARCHIVE = BASE / "h217_packet_v4.tar"
V4_MANIFEST = BASE / "H217_V4_SOURCE_MANIFEST.json"
IDENTITY = "E529-H217-TCN-FULLFEATURE-CAPACITY-V5-AMP4096"
V4_PACKET_SHA256 = "38bb1893b946c85850268a95afcf88c40593f06f0f5a956f71eafd599e681656"
V4_MANIFEST_SHA256 = "7ea2bf3cfb6fee065148dfb1c4834d15b5cc92058bb4119153b814b6a940a8dc"
ROOT_MEMBERS = (
    "README.md",
    "recipe.json",
    "run.sh",
    "runtime_resources.json",
    "worker.py",
)
EXCLUDED_DIRS = {".git", "__pycache__", "cache", "datasets", "runs"}
EXCLUDED_SUFFIXES = {".pyc", ".parquet", ".npy", ".npz", ".pt", ".tar"}


def sha256(path: Path) -> str:
    """Hash one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write sorted JSON with a trailing newline."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def pending_root_review() -> dict[str, object]:
    """Return the explicit not-authorized review block shared by V5 receipts."""
    return {
        "fit_authorized": False,
        "grade_authorized": False,
        "profile_authorized": False,
        "provider_create_authorized": False,
        "push_authorized": False,
        "required_before_any_create_or_fit": [
            "Verify the V5 project tree is byte-identical to h217_packet_v4/project.",
            "Verify exactly two scratch full-feature TCNs and no GRU or --raw-only command.",
            "Verify startup source hashes, sealed input hashes, and fold0 membership.",
            "Verify the strict 230-update gate and phase caps 900/7200/1800 within 21600 seconds.",
            "Admit each fit only from its measured profile. Raw-only 823.97 seconds does not admit V5.",
            "Fresh market, account, quota, headroom, and cost admission are still required.",
        ],
        "status": "pending_root_review",
    }


def _project_files(root: Path) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"project source must not contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if (
            any(part in EXCLUDED_DIRS for part in relative.parts)
            or path.suffix in EXCLUDED_SUFFIXES
        ):
            continue
        found[relative.as_posix()] = path.read_bytes()
    if len(found) < 10:
        raise ValueError("project source set is unexpectedly small")
    return found


def seal_project_copy() -> dict[str, str]:
    """Copy the sealed V4 project into V5 and return relative path hashes.

    Raises:
        ValueError: V5 contains a drifted or extra project file, or V4 is incomplete.
    """
    source_root = V4_PACKET_DIR / "project"
    destination_root = PACKET / "project"
    source = _project_files(source_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    for relative, payload in source.items():
        target = destination_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != payload:
            raise ValueError(f"V5 project drifted from sealed V4 source: {relative}")
        if not target.exists():
            target.write_bytes(payload)
    copied = _project_files(destination_root)
    if set(copied) != set(source) or any(copied[name] != source[name] for name in source):
        raise ValueError("V5 project is not a byte copy of h217_packet_v4/project")
    return {name: hashlib.sha256(payload).hexdigest() for name, payload in copied.items()}


def _v4_project_members() -> dict[str, bytes]:
    with tarfile.open(V4_ARCHIVE, "r") as archive:
        members: dict[str, bytes] = {}
        for member in archive.getmembers():
            if not member.isfile() or not member.name.startswith("project/"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot read V4 archive member: {member.name}")
            members[member.name.removeprefix("project/")] = extracted.read()
    return members


def assert_project_matches_v4_archive(project_hashes: dict[str, str]) -> None:
    """Require every V5 project byte to match the sealed V4 archive member."""
    archived = _v4_project_members()
    current = _project_files(PACKET / "project")
    if set(current) != set(archived) or any(current[name] != archived[name] for name in current):
        raise ValueError("V5 project bytes differ from the sealed V4 packet archive")
    if set(project_hashes) != set(archived):
        raise ValueError("V5 project hash set differs from the sealed V4 archive")


def included_files() -> list[Path]:
    """Return packet root files plus the sealed project, in archive order."""
    paths = [PACKET / name for name in ROOT_MEMBERS]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"V5 packet is missing root files: {missing}")
    for relative in _project_files(PACKET / "project"):
        paths.append(PACKET / "project" / relative)
    return sorted(paths, key=lambda path: path.relative_to(PACKET).as_posix())


def build_archive(files: list[Path], destination: Path) -> None:
    """Write a deterministic tar whose metadata does not depend on the host."""
    with tarfile.open(destination, "w", format=tarfile.PAX_FORMAT) as archive:
        for path in files:
            info = tarfile.TarInfo(path.relative_to(PACKET).as_posix())
            info.size = path.stat().st_size
            info.mode = path.stat().st_mode & 0o777
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = 0
            with path.open("rb") as stream:
                archive.addfile(info, stream)


def _load_worker() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("h217_v5_packet_worker", PACKET / "worker.py")
    if spec is None or spec.loader is None:
        raise ValueError("V5 worker cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_packet_recipe(
    worker: types.ModuleType,
) -> tuple[dict[str, object], dict[str, object]]:
    recipe = json.loads((PACKET / "recipe.json").read_text())
    runtime = json.loads((PACKET / "runtime_resources.json").read_text())
    worker.validate_recipe(recipe, runtime)
    worker.verify_source_provenance(PACKET / "project", recipe)
    worker.verify_split_membership(PACKET / "project", recipe)
    return recipe, runtime


def main() -> int:
    """Build the V5 packet and write source, archive, receipt, and admission files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=BASE / "h217_packet_v5.tar")
    args = parser.parse_args()
    output = args.output.resolve()
    if sha256(V4_ARCHIVE) != V4_PACKET_SHA256 or sha256(V4_MANIFEST) != V4_MANIFEST_SHA256:
        raise ValueError("sealed V4 packet or source manifest identity drifted")
    review = pending_root_review()
    if any(
        review[field]
        for field in (
            "fit_authorized",
            "provider_create_authorized",
            "push_authorized",
            "grade_authorized",
            "profile_authorized",
        )
    ):
        raise ValueError("builder must not authorize create, push, profile, fit, or grade")
    project_hashes = seal_project_copy()
    assert_project_matches_v4_archive(project_hashes)
    worker = _load_worker()
    recipe, runtime = _validate_packet_recipe(worker)
    files = included_files()
    build_archive(files, output)
    archived = _archive_members(output)
    if set(archived) != {path.relative_to(PACKET).as_posix() for path in files}:
        raise ValueError("archive member set differs from the V5 packet files")
    repo = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest = {
        "base_commit": base_commit,
        "identity": IDENTITY,
        "packet_members": {
            relative: {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
            for relative, payload in sorted(archived.items())
        },
        "packet_sha256": sha256(output),
        "pending_root_review": review,
        "predecessor": {
            "project_byte_identical_to_v4": True,
            "v4_packet_sha256": V4_PACKET_SHA256,
            "v4_source_manifest_sha256": V4_MANIFEST_SHA256,
        },
        "runtime_contract": runtime,
        "schema_version": 1,
        "science": {
            "arms": ["tcn_full128", "tcn_full256"],
            "feature_view": recipe["feature_view"],
            "fit_authorized": False,
            "historical_raw_only_is_paired_factor": False,
            "membership": recipe["membership"],
            "recipe_sha256": sha256(PACKET / "recipe.json"),
            "trainer_and_model_code_unchanged_from_v4": True,
        },
    }
    manifest_path = BASE / "H217_V5_SOURCE_MANIFEST.json"
    write_json(manifest_path, manifest)
    archive_manifest = {
        "identity": IDENTITY,
        "members": manifest["packet_members"],
        "member_count": len(files),
        "packet": str(output.relative_to(repo)),
        "packet_sha256": manifest["packet_sha256"],
        "pending_root_review": review,
        "schema_version": 1,
        "source_manifest_sha256": sha256(manifest_path),
    }
    archive_path = BASE / "H217_V5_ARCHIVE_MANIFEST.json"
    write_json(archive_path, archive_manifest)
    manifest_sha = sha256(manifest_path)
    receipt = {
        "archive_manifest": str(archive_path.relative_to(repo)),
        "archive_manifest_sha256": sha256(archive_path),
        "build_command": (
            f"python3 {Path(__file__).resolve().relative_to(repo)} --output {output.relative_to(repo)}"
        ),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "execution_scope": (
            "No provider query, create, rental, push, profile, fit, export, or grade was performed. "
            "Root review and fresh provider, account, quota, and cost admission remain required."
        ),
        "identity": IDENTITY,
        "member_count": len(files),
        "packet": str(output.relative_to(repo)),
        "packet_sha256": sha256(output),
        "pending_root_review": review,
        "project_byte_identical_to_v4": True,
        "schema_version": 1,
        "source_manifest": str(manifest_path.relative_to(repo)),
        "source_manifest_sha256": manifest_sha,
        "status": "prepared_for_root_review_not_admitted",
    }
    write_json(BASE / "H217_V5_BUILD_RECEIPT.json", receipt)
    admission = {
        "attempt_id": "e529-vast-h217-fullfeature-v5-20261008",
        "fit_authorized": False,
        "grade_authorized": False,
        "identity": IDENTITY,
        "packet_sha256": receipt["packet_sha256"],
        "pending_root_review": review,
        "profile_authorized": False,
        "provider_create_authorized": False,
        "push_authorized": False,
        "recipe": "h217_packet_v5/recipe.json",
        "requested_worker_max_seconds": int(runtime["cohort_max_runtime_seconds"]),
        "resource_scope": "Prepared packet only. No rental, provider action, quota change, or launch.",
        "runtime_contract": "h217_packet_v5/runtime_resources.json",
        "schema_version": 1,
        "source_manifest_sha256": manifest_sha,
        "status": "not_admitted",
    }
    write_json(BASE / "H217_V5_ADMISSION_REQUEST.json", admission)
    print(
        json.dumps(
            {
                "member_count": len(files),
                "packet_sha256": receipt["packet_sha256"],
                "source_manifest_sha256": manifest_sha,
                "status": receipt["status"],
            },
            sort_keys=True,
        )
    )
    return 0


def _archive_members(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, "r") as archive:
        members: dict[str, bytes] = {}
        for member in archive.getmembers():
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot read archive member: {member.name}")
            members[member.name] = extracted.read()
    return members


if __name__ == "__main__":
    raise SystemExit(main())
