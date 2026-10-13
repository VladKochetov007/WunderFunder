"""Build a deterministic, inventory-bound cross-team submission package.

The bundle reuses the xteam_h4 callback, precision setting, and native libraries
byte-for-byte. It does not run the scorer or establish numerical parity with the
source packages' callbacks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import zipfile
from pathlib import Path
from typing import cast

MAX_ZIP_BYTES = 20_000_000
BASE_CANDIDATE = "xteam_h4"
BASE_FILES = (
    "solution.py",
    "precision.txt",
    "v2/wnn_native.abi3.so",
    "avx2/wnn_native.abi3.so",
    "v3/wnn_native.abi3.so",
)
NATIVE_PRECISION = "q8vnni\n"


def sha256_file(path: Path) -> str:
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def parse_recipe(path: Path) -> tuple[str, list[dict[str, object]]]:
    """Load and validate a normalized recipe of inventory member references."""
    recipe = _object(json.loads(path.read_text(encoding="utf-8")), "recipe")
    name = recipe.get("name")
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None:
        raise ValueError("recipe name must be a safe single filename component")
    raw_members = recipe.get("members")
    if not isinstance(raw_members, list) or not raw_members:
        raise ValueError("recipe members must be a non-empty list")

    members: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    total = 0.0
    for index, raw_member in enumerate(raw_members):
        member = _object(raw_member, f"recipe member {index}")
        candidate = member.get("candidate")
        weights = member.get("weights")
        coefficient = member.get("coef")
        if not isinstance(candidate, str) or not candidate:
            raise ValueError(f"recipe member {index} needs a candidate name")
        if (
            not isinstance(weights, str)
            or Path(weights).name != weights
            or not weights.endswith(".npz")
            or weights.startswith("det_")
        ):
            raise ValueError(f"recipe member {index} needs a predictor NPZ basename")
        if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)):
            raise ValueError(f"recipe member {index} coefficient must be numeric")
        coefficient = float(coefficient)
        if not math.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError(f"recipe member {index} coefficient must be finite and non-negative")
        identity = (candidate, weights)
        if identity in seen:
            raise ValueError(f"duplicate recipe weight: {candidate}/{weights}")
        seen.add(identity)
        total += coefficient
        members.append({"candidate": candidate, "weights": weights, "coef": coefficient})
    if not any(member["coef"] > 0.0 for member in members):
        raise ValueError("at least one recipe coefficient must be positive")
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"recipe coefficients sum to {total}, expected 1")
    return name, members


def _inventory_record(inventory: dict[str, object], candidate: str) -> dict[str, object]:
    packages = _object(inventory.get("packages"), "inventory packages")
    package = packages.get(candidate)
    if package is None:
        raise ValueError(f"candidate is absent from inventory: {candidate}")
    return _object(package, f"inventory package {candidate}")


def _package_root(package: dict[str, object], candidate: str) -> Path:
    root = package.get("package")
    if not isinstance(root, str):
        raise ValueError(f"inventory package {candidate} lacks a package path")
    return Path(root).resolve(strict=True)


def _verify_archive(package: dict[str, object], candidate: str) -> dict[str, object]:
    archive_name = package.get("archive")
    expected_hash = package.get("sha256")
    if not isinstance(archive_name, str) or not isinstance(expected_hash, str):
        raise ValueError(f"inventory package {candidate} lacks an archive hash")
    archive = Path(archive_name).resolve(strict=True)
    actual_hash = sha256_file(archive)
    if actual_hash != expected_hash:
        raise ValueError(f"source archive differs from inventory: {candidate}")
    return {"path": str(archive), "sha256": actual_hash, "bytes": archive.stat().st_size}


def _verify_package_file(
    package: dict[str, object], candidate: str, relative_name: str
) -> tuple[Path, dict[str, object]]:
    files = _object(package.get("files"), f"inventory files for {candidate}")
    record = _object(files.get(relative_name), f"inventory file {candidate}/{relative_name}")
    expected_hash = record.get("sha256")
    if not isinstance(expected_hash, str):
        raise ValueError(f"inventory file lacks a SHA-256: {candidate}/{relative_name}")
    root = _package_root(package, candidate)
    source = (root / relative_name).resolve(strict=True)
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError(f"invalid package source path: {candidate}/{relative_name}")
    if sha256_file(source) != expected_hash:
        raise ValueError(f"package file differs from inventory: {candidate}/{relative_name}")
    expected_bytes = record.get("bytes")
    if isinstance(expected_bytes, int) and source.stat().st_size != expected_bytes:
        raise ValueError(f"package file size differs from inventory: {candidate}/{relative_name}")
    return source, record


def _resolve_member(
    inventory: dict[str, object], item: dict[str, object]
) -> tuple[Path, dict[str, object], dict[str, object]]:
    candidate = cast(str, item["candidate"])
    weights = cast(str, item["weights"])
    package = _inventory_record(inventory, candidate)
    _verify_package_file(package, candidate, "members.json")
    root = _package_root(package, candidate)
    member_map = json.loads((root / "members.json").read_text(encoding="utf-8"))
    active_members = package.get("members")
    if not isinstance(active_members, list) or not isinstance(member_map, list):
        raise ValueError(f"inventory package {candidate} has no member list")
    inventory_member = next(
        (
            _object(record, f"inventory member in {candidate}")
            for record in active_members
            if _object(record, f"inventory member in {candidate}").get("weights") == weights
        ),
        None,
    )
    packaged_member = next(
        (
            _object(record, f"packaged member in {candidate}")
            for record in member_map
            if _object(record, f"packaged member in {candidate}").get("weights") == weights
        ),
        None,
    )
    if inventory_member is None or packaged_member != inventory_member:
        raise ValueError(
            f"member is not an active inventory-bound predictor: {candidate}/{weights}"
        )
    source, file_record = _verify_package_file(package, candidate, weights)
    return source, file_record, inventory_member


def _verify_base_package(inventory: dict[str, object]) -> tuple[dict[str, str], dict[str, object]]:
    package = _inventory_record(inventory, BASE_CANDIDATE)
    archive_record = _verify_archive(package, BASE_CANDIDATE)
    source_files: dict[str, str] = {}
    output_records: dict[str, object] = {}
    for relative_name in BASE_FILES:
        source, record = _verify_package_file(package, BASE_CANDIDATE, relative_name)
        source_files[relative_name] = str(source)
        output_records[relative_name] = record
    precision = Path(source_files["precision.txt"]).read_text(encoding="ascii")
    if precision != NATIVE_PRECISION:
        raise ValueError(f"expected inherited {NATIVE_PRECISION.strip()} precision file")
    return source_files, {
        "archive": archive_record,
        "files": output_records,
    }


def _physical_output_parent(out_dir: Path) -> Path:
    requested = out_dir.expanduser()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"refusing to overwrite output directory: {requested}")
    parent = requested.parent.resolve(strict=True)
    resolved = parent / requested.name
    if resolved.is_relative_to(Path.home() / "Downloads"):
        raise ValueError("output must not be written under Downloads")
    if resolved == Path("/tmp") or resolved.is_relative_to(Path("/tmp")):
        raise ValueError("output must use a durable physical directory, not /tmp")
    if resolved == Path("/dev/shm") or resolved.is_relative_to(Path("/dev/shm")):
        raise ValueError("output must use a durable physical directory, not /dev/shm")

    mount_type = None
    mount_depth = -1
    try:
        mount_lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError("cannot verify the output filesystem type") from error
    for line in mount_lines:
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        fields = before.split()
        filesystem = after.split()
        if len(fields) < 5 or not filesystem:
            continue
        mountpoint = Path(
            fields[4].replace("\\040", " ").replace("\\011", "\t").replace("\\134", "\\")
        )
        if resolved == mountpoint or mountpoint in resolved.parents:
            depth = len(mountpoint.parts)
            if depth > mount_depth:
                mount_depth = depth
                mount_type = filesystem[0]
    if mount_type in {"tmpfs", "ramfs", "devtmpfs"}:
        raise ValueError(f"output filesystem is memory-backed: {mount_type}")
    return resolved


def _write_deterministic_zip(
    package_dir: Path, destination: Path, max_bytes: int
) -> dict[str, object]:
    files = sorted(path for path in package_dir.rglob("*") if path.is_file())
    if not files or not (package_dir / "solution.py").is_file():
        raise ValueError("package must contain a root-level solution.py")
    names = [path.relative_to(package_dir).as_posix() for path in files]
    if len(names) != len(set(names)) or any(
        Path(name).is_absolute() or ".." in Path(name).parts for name in names
    ):
        raise ValueError("package contains duplicate or unsafe ZIP paths")
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path, name in zip(files, names, strict=True):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            info.flag_bits = 0x800
            with path.open("rb") as source, archive.open(info, "w", force_zip64=False) as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    byte_count = destination.stat().st_size
    if byte_count > max_bytes:
        raise ValueError(f"submission ZIP is {byte_count} bytes; limit is {max_bytes}")
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("submission ZIP failed its CRC check")
        if archive.namelist() != names:
            raise ValueError("submission ZIP members do not match the sorted package files")
        member_hashes = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in names}
    return {
        "path": str(destination.resolve()),
        "bytes": byte_count,
        "sha256": sha256_file(destination),
        "members": member_hashes,
    }


def build_package(
    inventory_path: Path, recipe_path: Path, out_dir: Path, max_zip_bytes: int
) -> dict[str, object]:
    """Build a new Q8 package and provenance manifest in a durable directory."""
    if max_zip_bytes < 1 or max_zip_bytes > MAX_ZIP_BYTES:
        raise ValueError(f"max ZIP size must be between 1 and {MAX_ZIP_BYTES} bytes")
    inventory_path = inventory_path.resolve(strict=True)
    recipe_path = recipe_path.resolve(strict=True)
    inventory_hash = sha256_file(inventory_path)
    inventory = _object(json.loads(inventory_path.read_text(encoding="utf-8")), "inventory")
    name, members = parse_recipe(recipe_path)

    base_sources, base_records = _verify_base_package(inventory)
    source_archives: dict[str, dict[str, object]] = {BASE_CANDIDATE: base_records["archive"]}
    resolved_members: list[
        tuple[Path, dict[str, object], dict[str, object], dict[str, object]]
    ] = []
    seen_hashes: set[str] = set()
    for item in members:
        candidate = cast(str, item["candidate"])
        package = _inventory_record(inventory, candidate)
        if candidate not in source_archives:
            source_archives[candidate] = _verify_archive(package, candidate)
        source, file_record, ancestry = _resolve_member(inventory, item)
        digest = cast(str, file_record["sha256"])
        if digest in seen_hashes:
            raise ValueError("recipe contains duplicate weight contents under different names")
        seen_hashes.add(digest)
        resolved_members.append((source, file_record, ancestry, item))

    output_root = _physical_output_parent(out_dir)
    output_root.mkdir()
    package_root = output_root / "pkg"
    package_root.mkdir()
    file_records: dict[str, dict[str, object]] = {}
    for relative_name, source_name in base_sources.items():
        source = Path(source_name)
        destination = package_root / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256_file(destination) != sha256_file(source):
            raise ValueError(f"copied base file changed: {relative_name}")
        file_records[relative_name] = {
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
            "source_candidate": BASE_CANDIDATE,
        }

    package_members: list[dict[str, object]] = []
    source_member_records: list[dict[str, object]] = []
    for index, (source, source_record, ancestry, item) in enumerate(resolved_members):
        filename = f"weights_{index}.npz"
        destination = package_root / filename
        shutil.copyfile(source, destination)
        copied_hash = sha256_file(destination)
        if copied_hash != source_record["sha256"]:
            raise ValueError(f"copied team weight differs from inventory: {source}")
        candidate = cast(str, item["candidate"])
        source_weights = cast(str, item["weights"])
        coefficient = cast(float, item["coef"])
        package_members.append(
            {
                "weights": filename,
                "coef": coefficient,
                "source": f"{candidate}/{source_weights}",
            }
        )
        source_member_records.append(
            {
                "candidate": candidate,
                "weights": source_weights,
                "coef": coefficient,
                "source_sha256": copied_hash,
                "source_bytes": destination.stat().st_size,
                "inventory_member_record": ancestry,
            }
        )
        file_records[filename] = {
            "sha256": copied_hash,
            "bytes": destination.stat().st_size,
            "source_candidate": candidate,
            "source_member": source_weights,
        }

    members_path = package_root / "members.json"
    members_path.write_text(json.dumps(package_members, indent=1) + "\n", encoding="utf-8")
    file_records["members.json"] = {
        "sha256": sha256_file(members_path),
        "bytes": members_path.stat().st_size,
        "generated_from_recipe": str(recipe_path),
    }
    for relative_name, source_name in base_sources.items():
        source = Path(source_name)
        if sha256_file(source) != base_records["files"][relative_name]["sha256"]:
            raise ValueError(f"inherited xteam_h4 source changed during build: {relative_name}")

    archive_path = output_root / f"{name}.zip"
    zip_record = _write_deterministic_zip(package_root, archive_path, max_zip_bytes)
    if sha256_file(inventory_path) != inventory_hash:
        raise ValueError("inventory changed during package construction")
    manifest: dict[str, object] = {
        "status": "built",
        "name": name,
        "intent": "exploratory cross-team submission package; no performance claim",
        "inventory_path": str(inventory_path),
        "inventory_sha256": inventory_hash,
        "source_archives": source_archives,
        "base_package": BASE_CANDIDATE,
        "base_package_files": base_records["files"],
        "precision": NATIVE_PRECISION.strip(),
        "precision_note": "Inherited q8vnni runtime; source-to-target output parity was not measured.",
        "numerical_comparison": {
            "measured": False,
            "note": (
                "The package keeps the original xteam_h4 Q8 VNNI callback and native libraries. "
                "The selected source packages may use different callback arithmetic; this build "
                "does not assert numerical equivalence to their scores or predictions."
            ),
        },
        "candidate": name,
        "fixed_members": package_members,
        "members": [
            {
                **record,
                "run_reference": f"{record['candidate']}/{record['weights']}",
                "sources": [
                    {
                        "candidate": record["candidate"],
                        "weights": record["weights"],
                        "sha256": record["source_sha256"],
                    }
                ],
            }
            for record in source_member_records
        ],
        "files": file_records,
        "artifact": zip_record,
        "source_code": {
            "path_in_zip": "solution.py",
            "sha256": file_records["solution.py"]["sha256"],
        },
        "native_runtime": {
            "callback_source": BASE_CANDIDATE,
            "precision": NATIVE_PRECISION.strip(),
            "library_sha256": {
                relative_name: file_records[relative_name]["sha256"]
                for relative_name in BASE_FILES
                if relative_name.endswith(".so")
            },
        },
        "gate_status": "package_crc_passed; offline_import_and_streaming_gate_pending",
        "max_zip_bytes": max_zip_bytes,
        "builder_path": str(Path(__file__).resolve()),
        "builder_sha256": sha256_file(Path(__file__)),
    }
    manifest_path = output_root / f"{name}.manifest.json"
    with manifest_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    """Parse CLI arguments and build the requested package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="new durable physical directory"
    )
    parser.add_argument("--max-zip-bytes", type=int, default=MAX_ZIP_BYTES)
    args = parser.parse_args()
    manifest = build_package(args.inventory, args.recipe, args.out_dir, args.max_zip_bytes)
    print(
        json.dumps(
            {
                "zip": manifest["artifact"],
                "manifest": str(args.out_dir / f"{manifest['name']}.manifest.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
