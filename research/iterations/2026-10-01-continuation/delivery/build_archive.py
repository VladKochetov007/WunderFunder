"""Preserve completed continuation evidence and every new training checkpoint.

Run only after all owned studies have stopped and source has been committed.
Large remote datasets/teacher/OOF caches are deliberately outside this scope.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
from datetime import datetime, timezone


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    destination = args.destination.resolve()
    continuation = Path("research/iterations/2026-10-01-continuation")
    runs = [
        "cont_sh256_s3_f0", "cont_sh256_s3h2_f0",
        "cont_sh256_s3_f1", "cont_sh256_s3h2_f1",
        "cont_sh256_final_s3", "cont_sh256_final_s3h2",
        "cont_sh256_s4_f0", "cont_sh256_s4h2_f0",
        "cont_sh256_s4_f1", "cont_sh256_s4h2_f1",
    ]
    directories = [root / continuation] + [root / "artifacts/runs" / r for r in runs]
    files = {root / "research/CURRENT_STATE.md"}
    empty_directories = set()
    for directory in directories:
        if not directory.is_dir() or directory.is_symlink():
            raise RuntimeError(f"Missing or linked archive directory: {directory}")
        for p in directory.rglob("*"):
            if "__pycache__" in p.parts or p.name == ".DS_Store":
                continue
            if p.is_symlink():
                raise RuntimeError(f"Archive will not follow a symlink: {p}")
            if p.is_file():
                files.add(p)
            elif p.is_dir() and not any(p.iterdir()):
                empty_directories.add(str(p.relative_to(root)))
    paths = sorted(files, key=lambda p: str(p.relative_to(root)))
    rows = {str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in paths}
    checkpoint_paths = [p for p in paths if p.suffix == ".pt" and "artifacts/runs" in str(p)]
    if len(checkpoint_paths) != 40:
        raise RuntimeError(f"Expected all 40 checkpoints from ten fits; got {len(checkpoint_paths)}")
    if sum(r["bytes"] for r in rows.values()) > 1024 ** 3:
        raise RuntimeError("Archive scope unexpectedly exceeds 1 GiB")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=all", "--", str(continuation), "research/CURRENT_STATE.md"], cwd=root, text=True)
    if dirty:
        raise RuntimeError("Commit continuation sources before creating final archive")
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "research-continuation.tar.gz"
    manifest_path = destination / "research-archive.manifest.json"
    if archive.exists() or manifest_path.exists():
        raise RuntimeError("Final archive paths already exist; refusing to overwrite")
    with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
        for p in paths:
            tar.add(p, arcname=str(p.relative_to(root)), recursive=False)
        for name in sorted(empty_directories):
            tar.add(root / name, arcname=name, recursive=False)
    verified = {}
    verified_directories = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.isdir():
                if member.name not in empty_directories or member.name in verified_directories:
                    raise RuntimeError(f"Unexpected archive directory: {member.name}")
                verified_directories.add(member.name)
                continue
            if not member.isfile() or member.name not in rows or member.name in verified:
                raise RuntimeError(f"Unexpected archive member: {member.name}")
            h = hashlib.sha256()
            f = tar.extractfile(member)
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
            verified[member.name] = {"bytes": member.size, "sha256": h.hexdigest()}
    if verified != rows or verified_directories != empty_directories:
        raise RuntimeError("Archive content differs from manifest")
    changed = [str(p) for p in paths if digest(p) != rows[str(p.relative_to(root))]["sha256"]]
    if changed:
        raise RuntimeError(f"Source changed during archive creation: {changed}")
    manifest = {
        "status": "PASS_ALL_ARCHIVE_MEMBERS_READ_BACK_AND_SOURCE_STABLE",
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": head, "new_training_runs": runs,
        "checkpoint_count": len(checkpoint_paths), "file_count": len(rows),
        "empty_directories": sorted(empty_directories),
        "uncompressed_bytes": sum(r["bytes"] for r in rows.values()),
        "archive": {"path": archive.name, "bytes": archive.stat().st_size, "sha256": digest(archive)},
        "files": rows,
        "exclusions": ["Organizer dataset bodies", "Large remote teacher and OOF prediction arrays", "Python bytecode", "Historical closed campaign artifacts (previous delivery)"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
