#!/usr/bin/env python3
"""Prepare and finalize a no-launch Kaggle CPU cache-profile packet."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

STORAGE_ROOT = Path("/home/vlad/development/wunder/research_disk/b300-fast-20261023")
DEFAULT_SOURCE_ROOT = Path.cwd()
DEFAULT_DRIVER_TEMPLATE = STORAGE_ROOT / "kaggle_ops/profile_kernel_driver.py"
MACHINE_RECIPE = Path("research/experiment_drafts/b300-fast-local-kaggle-20261023/recipe.json")
SOURCE_MANIFEST_R1 = Path(
    "research/experiment_drafts/b300-fast-local-kaggle-20261023/ACTOR_SOURCE_SHA256_R1.json"
)
SOURCE_MANIFEST_R2 = Path(
    "research/experiment_drafts/b300-fast-local-kaggle-20261023/ACTOR_SOURCE_SHA256_R2.json"
)
EXPORT_RECEIPT = "EXPORT_RECEIPT_R1.json"
ARCHIVE_NAME = "age_actor_bundle.wnn"
MANIFEST_NAME = "bundle_manifest.json"
ADMISSION_NAME = "PROFILE_F0_CPU_R2_ADMISSION.json"
DATASET_METADATA_NAME = "dataset-metadata.json"
KERNEL_METADATA_NAME = "kernel-metadata.json"
DRIVER_TEMPLATE_MARKER = "__PROFILE_ADMISSION_SHA256__"
MACHINE_RECIPE_SHA256 = "0a67fe12c520e56bcf14a3a1efecb5030f3aa7ce932101814c9dd1c675c930fd"
SOURCE_CACHE_RECIPE_SHA256 = "edc873c813de77f55859d1a13180d907004e5f18fcc537e64eda7af1308e2508"
PARENT_MILESTONE_SHA256 = "0dd9f9cec9bf54a4c2abdaeb835ad13f099e201a4b578714838014e39eb4d11e"
LOCAL_CACHE_RECEIPT_SHA256 = "6659cdb989beceb75778e978699a46fede50d6f6661887354279883cbab4e9c1"
ACTUAL_PARENT_CHECK_SHA256 = "ed540e31b5d79da39e13d624b9c4faadf277d9efe953d7f8c24eecad3e4467c6"
EXPECTED_TRANSPORT_SHA256 = "1ff63aaf02cf94176677e19b1bfebdce03404fe9ce8c800669953451dfcb8c6a"
EXPECTED_TRANSPORT_BYTES = 1_140_018_035
EXPECTED_SOURCE_MANIFEST_R2_SHA256 = (
    "939e8a00a003d9299a397cac80753df67a56b65b34ed3bd0e8dce6726b109523"
)
MAX_BUNDLE_BYTES = 1_600_000_000
MAX_BUILD_SECONDS = 300
CHUNK_BYTES = 8 * 1024**2
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")
FORBIDDEN_RE = re.compile(
    r"(^|[/_.-])(token|secret|credentials?|password|kaggle\.json|\.env)([/_.-]|$)",
    re.IGNORECASE,
)


class PacketError(RuntimeError):
    """A bundle input, source pin, or local packet stage is invalid."""


class Inventory:
    """Collect a finite source-to-archive map without reading payload contents."""

    def __init__(self) -> None:
        self.files: dict[str, dict[str, Any]] = {}
        self.sources: set[str] = set()

    def add(self, source: Path, bundle_path: str, digest: str, size: int) -> None:
        """Register one pinned, non-aliased regular file."""
        if source.is_symlink():
            raise PacketError(f"bundle source is a symlink: {source}")
        source = source.resolve(strict=True)
        name = safe_bundle_path(bundle_path)
        if source.is_symlink() or not source.is_file():
            raise PacketError(f"bundle source is not a regular file: {source}")
        if name in self.files or str(source) in self.sources:
            raise PacketError(f"bundle path or source has an alias: {name}")
        if SHA256_RE.fullmatch(digest) is None or size < 0:
            raise PacketError(f"bundle file pin is invalid: {name}")
        if source.stat().st_size != size:
            raise PacketError(f"bundle file size differs from its pin: {source}")
        self.files[name] = {
            "source_path": str(source),
            "bundle_path": name,
            "sha256": digest,
            "bytes": size,
        }
        self.sources.add(str(source))

    def entries(self) -> list[dict[str, Any]]:
        """Return deterministic archive manifest entries."""
        return [self.files[name] for name in sorted(self.files)]

    def expanded_bytes(self) -> int:
        """Return the exact expanded archive payload size."""
        return sum(int(item["bytes"]) for item in self.files.values())


def safe_bundle_path(value: str) -> str:
    """Require one canonical relative path with no secret-like names."""
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or str(path) != value
        or FORBIDDEN_RE.search(value)
        or value.startswith(".git/")
    ):
        raise PacketError(f"unsafe archive member path: {value}")
    return str(path)


def read_object(path: Path, maximum_bytes: int = 32_000_000) -> dict[str, Any]:
    """Read one bounded JSON contract without decoding array payloads."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum_bytes:
        raise PacketError(f"metadata source is absent, nonregular, or oversized: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PacketError(f"expected a JSON object: {path}")
    return value


def sha256_file(path: Path, deadline: float, maximum_bytes: int | None = None) -> str:
    """Hash bounded memory chunks and enforce the local package deadline."""
    if path.is_symlink() or not path.is_file():
        raise PacketError(f"hash source is absent or nonregular: {path}")
    if maximum_bytes is not None and path.stat().st_size > maximum_bytes:
        raise PacketError(f"metadata source exceeds its size limit: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(CHUNK_BYTES), b""):
            if time.monotonic() >= deadline:
                raise TimeoutError("portable bundle preparation reached its 300-second limit")
            digest.update(block)
    return digest.hexdigest()


def stat_identity(path: Path) -> dict[str, int]:
    """Capture file identity before streaming a large already-hashed export."""
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
    }


def add_source_map(
    inventory: Inventory,
    source_root: Path,
    repo_root_path: str,
    map_path: Path,
    destination: str,
    deadline: float,
) -> tuple[str, dict[str, Any]]:
    """Add the actor's source map, R1 ancestry map, and all source-pinned code."""
    manifest = read_object(map_path)
    if (
        manifest.get("schema") != "wnn.age-readout-source-manifest.v1"
        or manifest.get("recipe_sha256") != MACHINE_RECIPE_SHA256
    ):
        raise PacketError("actor source manifest does not bind the frozen machine recipe")
    code_maps = [
        manifest.get("source_sha256", {}),
        manifest.get("fit_source_sha256", {}),
        manifest.get("authored_file_sha256", {}),
    ]
    if any(not isinstance(item, dict) for item in code_maps):
        raise PacketError("actor source manifest maps are malformed")
    code_pins: dict[str, str] = {}
    for map_index, source_map in enumerate(code_maps):
        for relative, digest in source_map.items():
            path = safe_bundle_path(relative)
            if map_index == 2 and path in {"README.md", "IMPORT_MAP.md"}:
                continue
            previous = code_pins.get(path)
            if previous is not None and previous != digest:
                raise PacketError(f"actor source maps disagree on code hash: {path}")
            code_pins[path] = digest
    for relative, digest in sorted(code_pins.items()):
        source = source_root / relative
        size = source.stat().st_size
        if sha256_file(source, deadline) != digest:
            raise PacketError(f"actor code differs from its source manifest: {relative}")
        inventory.add(source, f"{repo_root_path}/{relative}", digest, size)
    manifest_sha = sha256_file(map_path, deadline, 4_000_000)
    inventory.add(map_path, destination, manifest_sha, map_path.stat().st_size)
    return manifest_sha, manifest


def build_inventory(
    repo_root: Path,
    export_root: Path,
    export_admission_path: Path,
    actor_source_root: Path,
    deadline: float,
) -> tuple[Inventory, dict[str, Any]]:
    """Bind the exact suffix export, actor code, ancestry and recipe metadata."""
    export_admission = read_object(export_admission_path, 2_000_000)
    export_admission_sha = sha256_file(export_admission_path, deadline, 2_000_000)
    expected_root = Path(export_admission["output_root"]).resolve(strict=True)
    if export_admission.get("schema") != "wnn.age-readout-suffix-export-admission.v1":
        raise PacketError("bundle preparation requires the root suffix-export admission")
    if export_admission.get("status") != "admitted" or expected_root != export_root.resolve(
        strict=True
    ):
        raise PacketError("export root differs from the root-issued suffix-export admission")

    receipt_path = export_root / EXPORT_RECEIPT
    export_receipt = read_object(receipt_path, 8_000_000)
    export_receipt_sha = sha256_file(receipt_path, deadline, 8_000_000)
    if (
        export_receipt.get("schema") != "wnn.age-readout-suffix-export-receipt.v1"
        or export_receipt.get("status") != "complete"
        or export_receipt.get("admission_sha256") != export_admission_sha
        or export_receipt.get("machine_recipe_sha256") != MACHINE_RECIPE_SHA256
        or export_receipt.get("cache_verification_sha256") != LOCAL_CACHE_RECEIPT_SHA256
        or export_receipt.get("parent_cache_milestone_sha256") != PARENT_MILESTONE_SHA256
        or export_receipt.get("actual_parent_metadata_check_sha256") != ACTUAL_PARENT_CHECK_SHA256
        or export_receipt.get("new_scientific_fits") != 0
    ):
        raise PacketError("completed suffix-export receipt does not bind the frozen inputs")
    if (
        export_receipt.get(
            "total_payload_and_metadata_bytes",
            export_receipt.get("payload_and_metadata_bytes_excluding_this_receipt"),
        )
        is None
    ):
        raise PacketError("suffix-export receipt omits its output-size accounting")
    transport_path = export_root / "transport.json"
    transport_sha = sha256_file(transport_path, deadline, 4_000_000)
    if transport_sha != EXPECTED_TRANSPORT_SHA256 or transport_path.stat().st_size > 4_000_000:
        raise PacketError("portable transport manifest differs from the root-pinned source")
    transport = read_object(transport_path, 4_000_000)
    if transport.get("schema") != "wnn.age-readout-portable-transport.v1":
        raise PacketError("portable transport schema differs from the actor's exact parser")
    expected_shapes = {
        "0": {
            "fit_suffix5": ([1236, 20000, 5], 494_400_128),
            "evaluation_predictions": ([468, 20000, 2], 74_880_128),
        },
        "1": {
            "fit_suffix5": ([1243, 20000, 5], 497_200_128),
            "evaluation_predictions": ([457, 20000, 2], 73_120_128),
        },
    }
    for fold, kinds in expected_shapes.items():
        for kind, (shape, size) in kinds.items():
            item = transport["folds"][fold][kind]
            if (
                item.get("shape") != shape
                or item.get("dtype") != "<f4"
                or item.get("bytes") != size
            ):
                raise PacketError(f"portable transport header differs from frozen F{fold} {kind}")
    if (
        sum(path.stat().st_size for path in export_root.rglob("*") if path.is_file())
        != EXPECTED_TRANSPORT_BYTES
    ):
        raise PacketError("completed suffix transport tree byte count differs from root receipt")

    inventory = Inventory()
    expected_exports = export_receipt.get("outputs", {})
    if not isinstance(expected_exports, dict) or not expected_exports:
        raise PacketError("export receipt contains no completed output digest map")
    actual_export_paths = {
        path.relative_to(export_root).as_posix()
        for path in export_root.rglob("*")
        if path.is_file()
    }
    if actual_export_paths != set(expected_exports) | {EXPORT_RECEIPT}:
        raise PacketError("suffix export directory has extra or missing files")
    if expected_exports.get("transport.json", {}).get("sha256") != transport_sha:
        raise PacketError("export receipt transport digest differs from the root pin")
    for relative, record in sorted(expected_exports.items()):
        source = export_root / safe_bundle_path(relative)
        inventory.add(source, safe_bundle_path(relative), record["sha256"], int(record["bytes"]))
    inventory.add(
        receipt_path,
        EXPORT_RECEIPT,
        export_receipt_sha,
        receipt_path.stat().st_size,
    )

    machine_recipe_path = repo_root / MACHINE_RECIPE
    if sha256_file(machine_recipe_path, deadline, 4_000_000) != MACHINE_RECIPE_SHA256:
        raise PacketError("machine recipe differs from the frozen root pin")
    recipe = read_object(machine_recipe_path, 4_000_000)
    inventory.add(
        machine_recipe_path,
        f"repo/{MACHINE_RECIPE.as_posix()}",
        MACHINE_RECIPE_SHA256,
        machine_recipe_path.stat().st_size,
    )
    source_recipe_path = repo_root / recipe["source_cache_recipe"]["path"]
    if sha256_file(source_recipe_path, deadline, 8_000_000) != SOURCE_CACHE_RECIPE_SHA256:
        raise PacketError("source-cache recipe differs from its frozen cache pin")
    inventory.add(
        source_recipe_path,
        f"repo/{recipe['source_cache_recipe']['path']}",
        SOURCE_CACHE_RECIPE_SHA256,
        source_recipe_path.stat().st_size,
    )
    for fold in recipe["folds"]:
        for key in ("split_manifest",):
            relative = fold[key]
            source = repo_root / relative
            expected_sha = fold["split_manifest_sha256"]
            observed = sha256_file(source, deadline, 8_000_000)
            if observed != expected_sha:
                raise PacketError(f"frozen fold metadata differs from recipe: {relative}")
            inventory.add(source, f"repo/{relative}", expected_sha, source.stat().st_size)
        subset = fold["subset"]
        source = repo_root / subset["path"]
        observed = sha256_file(source, deadline, 8_000_000)
        if observed != subset["sha256"]:
            raise PacketError(f"fit subset differs from frozen recipe: {subset['path']}")
        inventory.add(source, f"repo/{subset['path']}", subset["sha256"], source.stat().st_size)
        provenance = fold["parent"]["provenance"]
        source = repo_root / provenance
        expected_sha = fold["parent"]["provenance_sha256"]
        observed = sha256_file(source, deadline, 2_000_000)
        if observed != expected_sha:
            raise PacketError(f"parent provenance differs from frozen recipe: {provenance}")
        inventory.add(source, f"repo/{provenance}", expected_sha, source.stat().st_size)

    r2_path = actor_source_root / SOURCE_MANIFEST_R2
    source_manifest_sha, source_manifest = add_source_map(
        inventory,
        actor_source_root,
        "repo",
        r2_path,
        "repo/source/SOURCE_SHA256_R2.json",
        deadline,
    )
    if source_manifest_sha != EXPECTED_SOURCE_MANIFEST_R2_SHA256:
        raise PacketError("actor R2 source manifest differs from the root source-freeze pin")
    r1_path = actor_source_root / SOURCE_MANIFEST_R1
    r1_sha = sha256_file(r1_path, deadline, 4_000_000)
    inventory.add(r1_path, "repo/source/SOURCE_SHA256_R1.json", r1_sha, r1_path.stat().st_size)
    for source_name, bundle_name in (
        ("ACTOR_PORTABLE_README_R2.md", "repo/ACTOR_PORTABLE_README_R2.md"),
        ("ACTOR_IMPORT_MAP_R2.md", "repo/ACTOR_IMPORT_MAP_R2.md"),
    ):
        source = (
            actor_source_root
            / "research/experiment_drafts/b300-fast-local-kaggle-20261023"
            / source_name
        )
        digest = sha256_file(source, deadline, 4_000_000)
        inventory.add(source, bundle_name, digest, source.stat().st_size)

    if inventory.expanded_bytes() > MAX_BUNDLE_BYTES:
        raise PacketError(
            f"portable bundle exceeds its 1.6 GB expanded gate: {inventory.expanded_bytes()}"
        )
    metadata = {
        "export_admission_sha256": export_admission_sha,
        "export_receipt_sha256": export_receipt_sha,
        "transport_sha256": transport_sha,
        "source_manifest_r1_sha256": r1_sha,
        "source_manifest_r2_sha256": source_manifest_sha,
        "source_manifest_r2": source_manifest,
        "machine_recipe_sha256": MACHINE_RECIPE_SHA256,
        "source_cache_recipe_sha256": SOURCE_CACHE_RECIPE_SHA256,
        "parent_cache_milestone_sha256": PARENT_MILESTONE_SHA256,
        "local_cache_receipt_sha256": LOCAL_CACHE_RECEIPT_SHA256,
        "actual_parent_metadata_check_sha256": ACTUAL_PARENT_CHECK_SHA256,
        "transport_bytes": sum(
            int(item["bytes"])
            for item in inventory.files.values()
            if item["bundle_path"].startswith("fold")
        ),
    }
    return inventory, metadata


def checked_copy_stream(source: BinaryIO, destination: BinaryIO, size: int, deadline: float) -> str:
    """Stream one source into the archive while checking its pinned content digest."""
    digest, copied = hashlib.sha256(), 0
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("portable bundle archive reached its 300-second deadline")
        block = source.read(CHUNK_BYTES)
        if not block:
            break
        copied += len(block)
        if copied > size:
            raise PacketError("archive source grew beyond its pinned size")
        digest.update(block)
        destination.write(block)
    if copied != size:
        raise PacketError("archive source ended before its pinned size")
    return digest.hexdigest()


def create_archive(
    inventory: Inventory,
    output_root: Path,
    source_root: str,
    deadline: float,
) -> tuple[Path, str, str, int]:
    """Create an uncompressed manifest-first tar and hash every member as copied."""
    entries = inventory.entries()
    manifest_payload = (
        json.dumps(
            {
                "schema": "wnn.age-readout-portable-profile-bundle.v1",
                "source_root": source_root,
                "files": entries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    archive_partial = output_root / f"{ARCHIVE_NAME}.partial"
    archive_path = output_root / ARCHIVE_NAME
    if archive_partial.exists() or archive_path.exists():
        raise PacketError("refusing to overwrite a prior portable archive")
    before = {item["bundle_path"]: stat_identity(Path(item["source_path"])) for item in entries}
    with archive_partial.open("xb", buffering=CHUNK_BYTES) as destination:
        with tarfile.open(fileobj=destination, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            manifest_info = tarfile.TarInfo(MANIFEST_NAME)
            manifest_info.size = len(manifest_payload)
            manifest_info.mtime = 0
            manifest_info.mode = 0o444
            manifest_info.uid = manifest_info.gid = 0
            manifest_info.uname = manifest_info.gname = ""
            archive.addfile(manifest_info, io.BytesIO(manifest_payload))
            for item in entries:
                if time.monotonic() >= deadline:
                    raise TimeoutError("portable archive reached its 300-second deadline")
                source_path = Path(item["source_path"])
                info = tarfile.TarInfo(item["bundle_path"])
                info.size = int(item["bytes"])
                info.mtime = 0
                info.mode = 0o444
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with source_path.open("rb") as source:
                    digest = hashlib.sha256()
                    copied = 0

                    class HashingSource:
                        """Update one streaming digest as tar copies the source."""

                        def read(
                            self,
                            size: int = -1,
                            source_arg: BinaryIO = source,
                            expected_bytes: int = int(item["bytes"]),
                            digest_arg: Any = digest,
                        ) -> bytes:
                            nonlocal copied
                            if time.monotonic() >= deadline:
                                raise TimeoutError(
                                    "portable archive reached its 300-second deadline"
                                )
                            block = source_arg.read(size)
                            copied += len(block)
                            if copied > expected_bytes:
                                raise PacketError("archive source grew beyond its pinned size")
                            digest_arg.update(block)
                            return block

                    archive.addfile(info, HashingSource())
                if copied != int(item["bytes"]) or digest.hexdigest() != item["sha256"]:
                    raise PacketError(
                        f"source content changed after its cache/source pin: {source_path}"
                    )
                if stat_identity(source_path) != before[item["bundle_path"]]:
                    raise PacketError(f"source file identity changed while bundling: {source_path}")
        destination.flush()
        os.fsync(destination.fileno())
    archive_sha = sha256_file(archive_partial, deadline, MAX_BUNDLE_BYTES + 16_000_000)
    archive_bytes = archive_partial.stat().st_size
    if archive_bytes > MAX_BUNDLE_BYTES + 16_000_000:
        raise PacketError("portable archive exceeds the 1.6 GB upload gate")
    os.replace(archive_partial, archive_path)
    return archive_path, archive_sha, manifest_sha, archive_bytes


def prepare(args: argparse.Namespace) -> int:
    """Build the private profile archive from the admitted export and frozen source map."""
    started = time.monotonic()
    deadline = started + MAX_BUILD_SECONDS
    repo_root = args.repo_root.resolve(strict=True)
    export_root = args.export_root.resolve(strict=True)
    export_admission = args.export_admission.resolve(strict=True)
    actor_source_root = args.actor_source_root.resolve(strict=True)
    output_root = args.output_root.resolve(strict=False)
    if not output_root.is_relative_to(STORAGE_ROOT.resolve()) or output_root.exists():
        raise PacketError("prepared output must be a fresh directory under research_disk")
    inventory, pins = build_inventory(
        repo_root, export_root, export_admission, actor_source_root, deadline
    )
    output_root.mkdir(parents=True, exist_ok=False)
    archive_path, archive_sha, manifest_sha, archive_bytes = create_archive(
        inventory,
        output_root,
        str(repo_root),
        deadline,
    )
    manifest = {
        "schema": "wnn.age-readout-portable-profile-bundle.v1",
        "source_root": str(repo_root),
        "files": inventory.entries(),
    }
    manifest_path = output_root / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if sha256_file(manifest_path, deadline, 32_000_000) != manifest_sha:
        raise PacketError("standalone manifest bytes differ from the manifest-first archive")
    receipt = {
        "schema": "wnn.age-readout-portable-profile-prepare-receipt.v1",
        "status": "prepared_no_provider_mutations",
        "bundle_archive": {
            "path": str(archive_path),
            "sha256": archive_sha,
            "bytes": archive_bytes,
        },
        "bundle_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha,
            "bytes": manifest_path.stat().st_size,
        },
        **pins,
        "expanded_bytes": inventory.expanded_bytes(),
        "file_count": len(inventory.files),
        "array_paths": sorted(path for path in inventory.files if path.endswith(".npy")),
        "target_cache_included": False,
        "raw_parquet_included": False,
        "actual_full_cache_arrays_included": False,
        "new_scientific_fits": 0,
        "elapsed_seconds": time.monotonic() - started,
    }
    receipt_path = output_root / "PREPARE_RECEIPT_R1.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "archive": str(archive_path),
                "sha256": archive_sha,
                "bytes": archive_bytes,
            },
            sort_keys=True,
        )
    )
    return 0


def admission_sha256(admission_path: Path) -> str:
    """Hash the root-issued profile admission before local finalization."""
    return sha256_file(admission_path, time.monotonic() + 2_000_000_000, 2_000_000)


def render_driver(template_path: Path, admission_sha: str, expected_template_sha: str) -> bytes:
    """Patch only the one admitted constant and verify equality after AST normalization."""
    source = template_path.read_text(encoding="utf-8")
    if hashlib.sha256(source.encode()).hexdigest() != expected_template_sha:
        raise PacketError("profile driver template SHA differs from the root admission")
    original_tree = ast.parse(source, filename=str(template_path))
    original_assignment = find_admission_assignment(original_tree)
    if (
        not isinstance(original_assignment.value, ast.Constant)
        or original_assignment.value.value != DRIVER_TEMPLATE_MARKER
    ):
        raise PacketError("driver template AST marker is malformed")
    value = original_assignment.value
    if value.end_lineno is None or value.end_col_offset is None:
        raise PacketError("profile driver admission marker has no complete source span")
    source_bytes = source.encode("utf-8")
    source_lines = source_bytes.splitlines(keepends=True)
    start = sum(len(line) for line in source_lines[: value.lineno - 1]) + value.col_offset
    end = sum(len(line) for line in source_lines[: value.end_lineno - 1]) + value.end_col_offset
    replacement = json.dumps(admission_sha).encode("ascii")
    rendered = (source_bytes[:start] + replacement + source_bytes[end:]).decode("utf-8")
    rendered_tree = ast.parse(rendered, filename="kernel.py")
    rendered_assignment_node = find_admission_assignment(rendered_tree)
    if (
        not isinstance(rendered_assignment_node.value, ast.Constant)
        or rendered_assignment_node.value.value != admission_sha
    ):
        raise PacketError("rendered driver AST does not contain the pinned admission SHA")
    rendered_assignment_node.value = ast.Constant(value=DRIVER_TEMPLATE_MARKER)
    if ast.dump(original_tree, include_attributes=False) != ast.dump(
        rendered_tree, include_attributes=False
    ):
        raise PacketError("rendered kernel differs from the template beyond its admission constant")
    return rendered.encode()


def find_admission_assignment(tree: ast.Module) -> ast.Assign:
    """Find the unique top-level assignment to the embedded admission SHA."""
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "PROFILE_ADMISSION_SHA256"
            for target in node.targets
        )
    ]
    if len(matches) != 1:
        raise PacketError("profile kernel must define exactly one admission-SHA assignment")
    return matches[0]


def finalize(args: argparse.Namespace) -> int:
    """Stage private upload folders and a root-pinned no-launch receipt."""
    prepared_root = args.prepared_root.resolve(strict=True)
    admission_path = args.admission.resolve(strict=True)
    template_path = args.driver_template.resolve(strict=True)
    destination = args.destination.resolve(strict=False)
    if not destination.is_relative_to(STORAGE_ROOT.resolve()) or destination.exists():
        raise PacketError("final packet destination must be a fresh physical-disk directory")
    if not args.dataset_id.startswith("vladkochetov07/") or not args.kernel_id.startswith(
        "vladkochetov07/"
    ):
        raise PacketError("private Kaggle IDs must use the authorized account owner")
    dataset_slug, kernel_slug = args.dataset_id.split("/", 1)[1], args.kernel_id.split("/", 1)[1]
    if SAFE_NAME_RE.fullmatch(dataset_slug) is None or SAFE_NAME_RE.fullmatch(kernel_slug) is None:
        raise PacketError("Kaggle dataset or kernel slug is invalid")
    prepared = read_object(prepared_root / "PREPARE_RECEIPT_R1.json")
    admission = read_object(admission_path, 2_000_000)
    admission_sha = admission_sha256(admission_path)
    driver_sha = sha256_file(template_path, time.monotonic() + 300, 2_000_000)
    if (
        prepared.get("status") != "prepared_no_provider_mutations"
        or admission.get("schema") != "wnn.age-readout-profile-admission.v1"
        or admission.get("status") != "admitted"
        or admission.get("mode") != "profile"
        or admission.get("device") != "cpu"
        or admission.get("recipe_sha256") != MACHINE_RECIPE_SHA256
        or admission.get("fold") != 0
        or admission.get("fit_sequences") != 16
        or admission.get("evaluation_sequences") != 8
        or admission.get("new_scientific_fits") != 0
        or admission.get("quality_scoring") is not False
    ):
        raise PacketError("profile admission or prepared bundle differs from the F0 CPU profile")
    archive_sha = prepared["bundle_archive"]["sha256"]
    archive_bytes = int(prepared["bundle_archive"]["bytes"])
    manifest_sha = prepared["bundle_manifest"]["sha256"]
    if (
        admission.get("bundle_archive_sha256") != archive_sha
        or admission.get("bundle_archive_bytes") != archive_bytes
        or admission.get("bundle_manifest_sha256") != manifest_sha
        or admission.get("transport_sha256") != prepared["transport_sha256"]
        or admission.get("export_receipt_sha256") != prepared["export_receipt_sha256"]
        or admission.get("source_manifest_sha256") != prepared["source_manifest_r2_sha256"]
        or admission.get("driver_template_sha256") != driver_sha
    ):
        raise PacketError("profile admission does not pin the prepared source, archive, and driver")
    rendered_driver = render_driver(template_path, admission_sha, driver_sha)
    destination.mkdir(parents=True, exist_ok=False)
    dataset_upload = destination / "dataset_upload"
    kernel_upload = destination / "kernel_upload"
    dataset_upload.mkdir()
    kernel_upload.mkdir()
    archive_source = Path(prepared["bundle_archive"]["path"])
    archive_target = dataset_upload / ARCHIVE_NAME
    os.link(archive_source, archive_target)
    shutil.copyfile(admission_path, dataset_upload / ADMISSION_NAME)
    dataset_metadata = {
        "id": args.dataset_id,
        "title": f"WNN age readout portable CPU profile {args.release}",
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    }
    if len(dataset_metadata["title"]) < 6 or len(dataset_metadata["title"]) > 50:
        raise PacketError("dataset title exceeds Kaggle's title length limit")
    dataset_metadata_path = dataset_upload / DATASET_METADATA_NAME
    dataset_metadata_path.write_text(
        json.dumps(dataset_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    kernel_path = kernel_upload / "kernel.py"
    kernel_path.write_bytes(rendered_driver)
    kernel_metadata = {
        "code_file": "kernel.py",
        "competition_sources": [],
        "dataset_sources": [args.dataset_id, args.raw_dataset_id],
        "enable_gpu": False,
        "enable_internet": False,
        "enable_tpu": False,
        "id": args.kernel_id,
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "",
        "title": f"WNN age readout F0 CPU profile {args.release}",
    }
    kernel_metadata_path = kernel_upload / KERNEL_METADATA_NAME
    kernel_metadata_path.write_text(
        json.dumps(kernel_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    launch_receipt = {
        "schema": "wnn.age-readout-profile-launch-packet.v1",
        "status": "prepared_no_provider_mutations",
        "profile_admission_sha256": admission_sha,
        "bundle_archive_sha256": archive_sha,
        "bundle_archive_bytes": archive_bytes,
        "bundle_manifest_sha256": manifest_sha,
        "transport_sha256": prepared["transport_sha256"],
        "export_receipt_sha256": prepared["export_receipt_sha256"],
        "source_manifest_r2_sha256": prepared["source_manifest_r2_sha256"],
        "driver_template_sha256": driver_sha,
        "rendered_kernel_sha256": hashlib.sha256(rendered_driver).hexdigest(),
        "dataset_metadata_sha256": sha256_file(dataset_metadata_path, time.monotonic() + 30),
        "kernel_metadata_sha256": sha256_file(kernel_metadata_path, time.monotonic() + 30),
        "dataset_id": args.dataset_id,
        "kernel_id": args.kernel_id,
        "dataset_upload_dir": str(dataset_upload),
        "kernel_upload_dir": str(kernel_upload),
        "dataset_command": [
            "kaggle",
            "datasets",
            "create",
            "--path",
            str(dataset_upload),
            "--dir-mode",
            "skip",
        ],
        "kernel_command": ["kaggle", "kernels", "push", "-p", str(kernel_upload)],
        "kernel_push_starts_execution": True,
        "provider_mutations": 0,
    }
    receipt_path = destination / "LAUNCH_PACKET_R2.json"
    receipt_path.write_text(
        json.dumps(launch_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": launch_receipt["status"],
                "packet": str(receipt_path),
                "kernel_sha256": launch_receipt["rendered_kernel_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    """Define local-only prepare/finalize operations for the profile bundle."""
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    prepare_parser.add_argument("--export-root", type=Path, required=True)
    prepare_parser.add_argument("--export-admission", type=Path, required=True)
    prepare_parser.add_argument("--actor-source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.set_defaults(run=prepare)
    finalize_parser = commands.add_parser("finalize")
    finalize_parser.add_argument("--prepared-root", type=Path, required=True)
    finalize_parser.add_argument("--admission", type=Path, required=True)
    finalize_parser.add_argument("--driver-template", type=Path, default=DEFAULT_DRIVER_TEMPLATE)
    finalize_parser.add_argument("--dataset-id", required=True)
    finalize_parser.add_argument("--kernel-id", required=True)
    finalize_parser.add_argument("--raw-dataset-id", default="vladkochetov07/wunder-funder")
    finalize_parser.add_argument("--release", default="20261023-r2")
    finalize_parser.add_argument("--destination", type=Path, required=True)
    finalize_parser.set_defaults(run=finalize)
    return result


def main() -> int:
    """Run one bounded local packet-preparation step without contacting Kaggle."""
    args = parser().parse_args()
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
