from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import re
import subprocess
import tarfile
import types
from pathlib import Path

BASE = Path(__file__).resolve().parent
PACKET = BASE / "h217_packet_v5r2"
V5_DIR = BASE / "h217_packet_v5"
V5_ARCHIVE = BASE / "h217_packet_v5.tar"
IDENTITY = "E529-H217-TCN-FULLFEATURE-CAPACITY-V5R2-EXACT-RECIPE-GUARD"
V5_PACKET_SHA256 = "f369fee4ba081f430e67a44a1f1d2eadf3ba860126c0f79ef49831630d5b58f2"
ROOT_MEMBERS = (
    "README.md",
    "recipe.json",
    "run.sh",
    "runtime_resources.json",
    "worker.py",
)
UNCHANGED_ROOT = ("recipe.json", "run.sh", "runtime_resources.json")
EXECUTION_GUARD_MEMBERS = ("worker.py",)
DOCUMENTATION_MEMBERS = ("README.md",)
EXCLUDED_DIRS = {".git", "__pycache__", "cache", "datasets", "runs"}
EXCLUDED_SUFFIXES = {".pyc", ".parquet", ".npy", ".npz", ".pt", ".tar"}
PIN_PATTERN = re.compile(r'EXPECTED_RECIPE_SHA256 = "([0-9a-f]{64})"')


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
    """Return the not-authorized review block for the V5R2 guard packet."""
    return {
        "fit_authorized": False,
        "grade_authorized": False,
        "profile_authorized": False,
        "provider_create_authorized": False,
        "push_authorized": False,
        "required_before_any_create_or_fit": [
            "Verify the changed-member receipt: only the execution guard and README differ from V5.",
            "Verify recipe.json, run.sh, runtime_resources.json, and every project member match V5 bytes.",
            "Verify worker.py pins the reviewed recipe SHA256 and rejects mismatches before parse.",
            "Fresh market, account, quota, headroom, and cost admission remain required.",
        ],
        "status": "pending_root_review",
    }


def pinned_recipe_sha256(worker_text: str) -> str:
    """Read the recipe pin from worker source. Do not hash the recipe here."""
    found = PIN_PATTERN.findall(worker_text)
    if len(found) != 1:
        raise ValueError("worker must pin exactly one expected recipe SHA256 literal")
    return found[0]


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
        raise ValueError("V5 project source set is unexpectedly small")
    return found


def _tar_members(path: Path) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    with tarfile.open(path, "r") as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"archive contains a link member: {member.name}")
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot read archive member: {member.name}")
            members[member.name] = extracted.read()
    return members


def changed_member_receipt(before: dict[str, bytes], after: dict[str, bytes]) -> dict[str, object]:
    """Prove the R2 archive changes only the execution guard and its README."""
    names = set(before) | set(after)
    changed = sorted(name for name in names if before.get(name) != after.get(name))
    expected = sorted([*EXECUTION_GUARD_MEMBERS, *DOCUMENTATION_MEMBERS])
    if changed != expected:
        raise ValueError(f"V5 to V5R2 member diff is not execution-guard-only: {changed}")
    before_project = sorted(name for name in before if name.startswith("project/"))
    after_project = sorted(name for name in after if name.startswith("project/"))
    if len(before_project) < 10 or before_project != after_project:
        raise ValueError("project member set is incomplete")
    if any(before[name] != after[name] for name in before_project):
        raise ValueError("project bytes changed")
    if any(before[name] != after[name] for name in UNCHANGED_ROOT):
        raise ValueError("recipe, run wrapper, or runtime bytes changed")
    if before["recipe.json"] != after["recipe.json"]:
        raise ValueError("recipe argv bytes changed")
    return {
        "after_packet_identity": IDENTITY,
        "argv_unchanged_because_recipe_bytes_unchanged": True,
        "before_packet": "h217_packet_v5.tar",
        "changed_members": changed,
        "documentation_members": list(DOCUMENTATION_MEMBERS),
        "execution_guard_members": list(EXECUTION_GUARD_MEMBERS),
        "project_bytes_unchanged": True,
        "project_member_count": len(before_project),
        "recipe_sha256": hashlib.sha256(after["recipe.json"]).hexdigest(),
        "schema_version": 1,
        "unchanged_root_members": list(UNCHANGED_ROOT),
        "worker_sha256": hashlib.sha256(after["worker.py"]).hexdigest(),
    }


def _load_worker() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("h217_v5r2_packet_worker", PACKET / "worker.py")
    if spec is None or spec.loader is None:
        raise ValueError("V5R2 worker cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_archive(payloads: dict[str, bytes], destination: Path) -> None:
    """Write a deterministic tar of regular file members."""
    with tarfile.open(destination, "w", format=tarfile.PAX_FORMAT) as archive:
        for name in sorted(payloads):
            payload = payloads[name]
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644 if not name.endswith(".sh") else 0o755
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = 0
            info.type = tarfile.REGTYPE
            archive.addfile(info, io.BytesIO(payload))


def main() -> int:
    """Build the V5R2 guard packet without copying the project tree into the packet dir."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=BASE / "h217_packet_v5r2.tar")
    args = parser.parse_args()
    output = args.output.resolve()
    if sha256(V5_ARCHIVE) != V5_PACKET_SHA256:
        raise ValueError("sealed V5 packet identity drifted")
    review = pending_root_review()
    if any(
        review[field]
        for field in (
            "fit_authorized",
            "grade_authorized",
            "profile_authorized",
            "provider_create_authorized",
            "push_authorized",
        )
    ):
        raise ValueError("builder must not authorize create, push, profile, fit, or grade")
    if (PACKET / "project").exists():
        raise ValueError("V5R2 packet directory must not contain a second project tree")
    worker_text = (PACKET / "worker.py").read_text()
    pin = pinned_recipe_sha256(worker_text)
    recipe_bytes = (PACKET / "recipe.json").read_bytes()
    if hashlib.sha256(recipe_bytes).hexdigest() != pin:
        raise ValueError("worker pin does not match the V5R2 recipe bytes")
    if recipe_bytes != (V5_DIR / "recipe.json").read_bytes():
        raise ValueError("V5R2 recipe.json is not the original V5 recipe bytes")
    for name in UNCHANGED_ROOT:
        if (PACKET / name).read_bytes() != (V5_DIR / name).read_bytes():
            raise ValueError(f"{name} drifted from V5")
    worker = _load_worker()
    if pin != worker.EXPECTED_RECIPE_SHA256:
        raise ValueError("loaded worker pin drifted from its source literal")
    bound = worker.load_bound_recipe(PACKET / "recipe.json")
    runtime = json.loads((PACKET / "runtime_resources.json").read_text())
    worker.validate_recipe(bound, runtime)
    project = _project_files(V5_DIR / "project")
    archived_v5 = _tar_members(V5_ARCHIVE)
    v5_project = {
        name.removeprefix("project/"): payload
        for name, payload in archived_v5.items()
        if name.startswith("project/")
    }
    if project != v5_project:
        raise ValueError("on-disk V5 project differs from the sealed V5 archive")
    payloads = {name: (PACKET / name).read_bytes() for name in ROOT_MEMBERS}
    payloads.update({f"project/{name}": payload for name, payload in project.items()})
    build_archive(payloads, output)
    archived_r2 = _tar_members(output)
    receipt_body = changed_member_receipt(archived_v5, archived_r2)
    if (
        hashlib.sha256(archived_r2["worker.py"]).hexdigest()
        != hashlib.sha256(payloads["worker.py"]).hexdigest()
    ):
        raise ValueError("archive worker bytes differ from the guard source")
    if hashlib.sha256(archived_r2["recipe.json"]).hexdigest() != pin:
        raise ValueError("archive recipe bytes differ from the pinned hash")
    repo = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest = {
        "base_commit": base_commit,
        "identity": IDENTITY,
        "packet_members": {
            name: {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
            for name, payload in sorted(archived_r2.items())
        },
        "packet_sha256": sha256(output),
        "pending_root_review": review,
        "pinned_recipe_sha256": pin,
        "predecessor": {"v5_packet_sha256": V5_PACKET_SHA256},
        "schema_version": 1,
        "science": {
            "argv_unchanged": True,
            "fit_authorized": False,
            "project_byte_identical_to_v5": True,
            "recipe_sha256": pin,
            "worker_sha256": hashlib.sha256(archived_r2["worker.py"]).hexdigest(),
        },
    }
    manifest_path = BASE / "H217_V5R2_SOURCE_MANIFEST.json"
    write_json(manifest_path, manifest)
    archive_manifest = {
        "identity": IDENTITY,
        "member_count": len(archived_r2),
        "members": manifest["packet_members"],
        "packet": str(output.relative_to(repo)),
        "packet_sha256": manifest["packet_sha256"],
        "pending_root_review": review,
        "pinned_recipe_sha256": pin,
        "schema_version": 1,
        "source_manifest_sha256": sha256(manifest_path),
        "worker_sha256": manifest["science"]["worker_sha256"],
    }
    archive_path = BASE / "H217_V5R2_ARCHIVE_MANIFEST.json"
    write_json(archive_path, archive_manifest)
    receipt_body.update(
        {
            "archive_recipe_sha256": pin,
            "archive_worker_sha256": manifest["science"]["worker_sha256"],
            "pending_root_review": review,
            "v5_packet_sha256": V5_PACKET_SHA256,
            "v5r2_packet_sha256": manifest["packet_sha256"],
        }
    )
    receipt_path = BASE / "H217_V5R2_CHANGED_MEMBER_RECEIPT.json"
    write_json(receipt_path, receipt_body)
    build_receipt = {
        "archive_manifest": str(archive_path.relative_to(repo)),
        "archive_manifest_sha256": sha256(archive_path),
        "build_command": (
            f"python3 {Path(__file__).resolve().relative_to(repo)} --output {output.relative_to(repo)}"
        ),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "changed_member_receipt": str(receipt_path.relative_to(repo)),
        "changed_member_receipt_sha256": sha256(receipt_path),
        "execution_scope": (
            "No provider query, create, rental, push, profile, fit, export, or grade was performed."
        ),
        "identity": IDENTITY,
        "member_count": len(archived_r2),
        "packet": str(output.relative_to(repo)),
        "packet_sha256": manifest["packet_sha256"],
        "pending_root_review": review,
        "pinned_recipe_sha256": pin,
        "schema_version": 1,
        "source_manifest": str(manifest_path.relative_to(repo)),
        "source_manifest_sha256": sha256(manifest_path),
        "status": "prepared_for_root_review_not_admitted",
    }
    write_json(BASE / "H217_V5R2_BUILD_RECEIPT.json", build_receipt)
    admission = {
        "attempt_id": "e529-vast-h217-fullfeature-v5r2-20261008",
        "fit_authorized": False,
        "grade_authorized": False,
        "identity": IDENTITY,
        "packet_sha256": manifest["packet_sha256"],
        "pending_root_review": review,
        "pinned_recipe_sha256": pin,
        "profile_authorized": False,
        "provider_create_authorized": False,
        "push_authorized": False,
        "schema_version": 1,
        "source_manifest_sha256": sha256(manifest_path),
        "status": "not_admitted",
    }
    write_json(BASE / "H217_V5R2_ADMISSION_REQUEST.json", admission)
    print(
        json.dumps(
            {
                "changed_members": receipt_body["changed_members"],
                "member_count": len(archived_r2),
                "packet_sha256": manifest["packet_sha256"],
                "pinned_recipe_sha256": pin,
                "status": build_receipt["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
