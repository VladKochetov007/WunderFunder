#!/usr/bin/env python3
"""Prepare and finalize the finite, root-admitted Kaggle age-grade bundle.

`prepare` requires all six four-file outputs and emits the manifest/archive.
Root then freezes admission against that manifest. `finalize` checks the receipt
and stages one private dataset folder plus one CPU kernel folder. Neither command
contacts Kaggle or decodes prediction/target arrays.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import time
from pathlib import Path, PurePosixPath

# Parsed JSON contracts are heterogenous and are checked at each external boundary.
from typing import Any, cast

STORAGE_ROOT = Path("/home/vlad/development/wunder/research_disk/b300-fast-20261023")
DEFAULT_RECIPE = Path("research/experiment_drafts/b300-fast-local-kaggle-20261023/recipe.json")
DEFAULT_CACHE_RECEIPT = Path(
    "research/experiment_drafts/b300-fast-local-kaggle-20261023/LOCAL_CACHE_VERIFICATION_R1.json"
)
DEFAULT_CACHE_ADMISSION = Path(
    "research/experiment_drafts/b300-fast-local-kaggle-20261023/CACHE_VERIFY_ADMISSION_R1.json"
)
DEFAULT_CACHE_SPEC = Path(
    "research/experiment_drafts/b300-fast-local-kaggle-20261023/CACHE_VERIFY_SPEC_R1.json"
)
DEFAULT_CACHE_GUARD = Path(
    "research/experiment_drafts/b300-fast-local-kaggle-20261023/CACHE_VERIFY_GUARD_R1.json"
)
DEFAULT_CACHE_LOG = Path(
    "research/experiment_drafts/b300-fast-local-kaggle-20261023/cache_verify_r1.log"
)
DEFAULT_PARENT_MILESTONE = Path(
    "research/experiment_drafts/online-relationships-20261022/PARENT_CACHE_MILESTONE_R1.json"
)
DEFAULT_SOURCE_MAP = STORAGE_ROOT / "b300_scout/source/SOURCE_SHA256_R1.json"
DEFAULT_GRADER_SOURCE = STORAGE_ROOT / "grader/age_readout_grade.py"
DEFAULT_DRIVER_SOURCE = STORAGE_ROOT / "kaggle_ops/kernel_driver.py"
DEFAULT_DATASET_ID = "vladkochetov07/wnn-age-readout-grade-20261023-r1"
DEFAULT_KERNEL_ID = "vladkochetov07/wnn-age-readout-grade-20261023-r1-runner"
RECIPE_SHA256 = "0a67fe12c520e56bcf14a3a1efecb5030f3aa7ce932101814c9dd1c675c930fd"
TARGET_SHA256 = "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955"
PARENT_MILESTONE_SHA256 = "0dd9f9cec9bf54a4c2abdaeb835ad13f099e201a4b578714838014e39eb4d11e"
CACHE_ADMISSION_SHA256 = "052424d4d631bce24741ea0b2f75218618e78ca04b9be812ed3a616a42ccfff6"
CACHE_SPEC_SHA256 = "ebf1f1fa163f46bbae4e690bdf4507b4f4caca4351d75af42c57a18138fee8a6"
CACHE_GUARD_SHA256 = "f5fec8727cdbfc449995dc3e50fc04af70432941870257ec8f7646d5a0e13e12"
ADMISSION_FILENAME = "GRADE_ADMISSION_R1.json"
ARCHIVE_FILENAME = "grade_bundle.wnn"
MANIFEST_FILENAME = "bundle_manifest.json"
MAX_BUNDLE_BYTES = 1_000_000_000
CHUNK_BYTES = 8 * 1024**2
SHA256_RE = re.compile(r"[0-9a-f]{64}")
FORBIDDEN_RE = re.compile(
    r"(^|[/_.-])(token|secret|credentials?|password|kaggle\.json|\.env)([/_.-]|$)",
    re.IGNORECASE,
)


def read_object(path: Path) -> dict[str, Any]:
    """Read a bounded-size JSON object used as a source contract."""
    if path.stat().st_size > 32_000_000:
        raise ValueError(f"metadata unexpectedly large: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def safe_bundle_path(value: str) -> str:
    """Require a canonical, credential-free relative POSIX bundle path."""
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
        raise ValueError(f"unsafe bundle path: {value}")
    return str(path)


def source_path(root: Path, value: str | Path) -> Path:
    """Keep recipe-relative identities root-qualified and external paths absolute."""
    path = Path(value)
    return Path(os.path.normpath(str(path if path.is_absolute() else root / path)))


def sha256_file(path: Path, deadline: float) -> str:
    """Hash a source file in bounded memory, checking the staging deadline."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            if time.monotonic() >= deadline:
                raise TimeoutError("bundle preparation exceeded its time budget")
            digest.update(block)
    return digest.hexdigest()


def grader_core_dependencies(path: Path) -> dict[str, str]:
    """Read the adapter's literal core dependency map without importing its code."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CORE_DEPENDENCIES"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, dict) and all(
                isinstance(key, str) and isinstance(digest, str) for key, digest in value.items()
            ):
                return value
    raise ValueError("grader source lacks a literal CORE_DEPENDENCIES map")


class Inventory:
    """Collect exact original paths before hashing or reading large payloads."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.entries: dict[str, dict[str, Any]] = {}
        self.bundle_paths: set[str] = set()

    def add(
        self,
        source: Path,
        bundle_path: str,
        expected_sha256: str | None = None,
        expected_bytes: int | None = None,
    ) -> None:
        """Register a source without opening its contents."""
        source = source_path(self.root, source)
        name = safe_bundle_path(bundle_path)
        original = source.as_posix()
        prior = self.entries.get(original)
        if prior is not None:
            if prior["bundle_path"] != name or prior["expected_sha256"] != expected_sha256:
                raise ValueError(f"source path has conflicting bundle aliases: {original}")
            return
        if name in self.bundle_paths:
            raise ValueError(f"bundle path aliases another source: {name}")
        self.bundle_paths.add(name)
        self.entries[original] = {
            "source_path": original,
            "bundle_path": name,
            "expected_sha256": expected_sha256,
            "expected_bytes": expected_bytes,
        }

    def preflight(self) -> int:
        """Check all required source file metadata and the size ceiling first."""
        missing: list[str] = []
        total = 0
        for item in self.entries.values():
            path = Path(item["source_path"])
            if path.is_symlink() or not path.is_file():
                missing.append(str(path))
                continue
            size = path.stat().st_size
            if item["expected_bytes"] is not None and size != item["expected_bytes"]:
                raise ValueError(f"source byte count differs: {path}")
            item["bytes"] = size
            total += size
        if missing:
            preview = "\n".join(missing[:12])
            raise FileNotFoundError(f"required bundle inputs are missing/nonregular:\n{preview}")
        if total > MAX_BUNDLE_BYTES:
            raise ValueError(f"uncompressed bundle exceeds one-gigabyte gate: {total}")
        return total

    def hash_all(self, deadline: float) -> int:
        """Hash every staged source and compare any earlier provenance pin."""
        for item in self.entries.values():
            path = Path(item["source_path"])
            stat_before = path.stat()
            digest = sha256_file(path, deadline)
            stat_after = path.stat()
            if (
                stat_before.st_size != stat_after.st_size
                or stat_before.st_mtime_ns != stat_after.st_mtime_ns
            ):
                raise RuntimeError(f"source changed while hashing: {path}")
            expected = item["expected_sha256"]
            if expected is not None and digest != expected:
                raise ValueError(f"source SHA256 differs from its frozen pin: {path}")
            item["sha256"] = digest
            item["bytes"] = stat_after.st_size
        return sum(int(item["bytes"]) for item in self.entries.values())

    def manifest_entries(self) -> list[dict[str, Any]]:
        """Return the adapter's exact content map, sorted deterministically."""
        return [
            {
                "source_path": item["source_path"],
                "bundle_path": item["bundle_path"],
                "sha256": item["sha256"],
                "bytes": item["bytes"],
            }
            for item in sorted(self.entries.values(), key=lambda entry: entry["bundle_path"])
        ]


def add_source_hash_map(inventory: Inventory, source_root: Path, hashes: dict[str, Any]) -> None:
    """Add repo code paths from a pinned source-hash mapping."""
    for relative, digest in hashes.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError("source hash map is malformed")
        path = source_path(source_root, relative)
        inventory.add(path, relative, digest)


def build_inventory(
    root: Path,
    source_map_path: Path,
    grader_path: Path,
) -> tuple[Inventory, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Derive the complete finite input closure from recipe and frozen receipts."""
    recipe_path = root / DEFAULT_RECIPE
    recipe = read_object(recipe_path)
    if sha256_file(recipe_path, float("inf")) != RECIPE_SHA256:
        raise ValueError("frozen recipe SHA256 changed")
    cache_path = root / DEFAULT_CACHE_RECEIPT
    cache = read_object(cache_path)
    if (
        cache.get("status") != "PASS_PAYLOAD_HASH_AND_HEADER"
        or cache.get("recipe_sha256") != RECIPE_SHA256
    ):
        raise ValueError("root local cache verification is not complete for this recipe")
    cache_admission_path = root / DEFAULT_CACHE_ADMISSION
    cache_spec_path = root / DEFAULT_CACHE_SPEC
    cache_guard_path = root / DEFAULT_CACHE_GUARD
    cache_admission = read_object(cache_admission_path)
    source_map = read_object(source_map_path)
    if source_map.get("schema") != "wnn.age-readout-source-manifest.v1":
        raise ValueError("actor source map schema differs")
    if source_map.get("recipe_sha256") != RECIPE_SHA256:
        raise ValueError("actor source map recipe pin differs")
    milestone_path = root / DEFAULT_PARENT_MILESTONE
    milestone = read_object(milestone_path)
    grader_deps = grader_core_dependencies(grader_path)
    inventory = Inventory(root)
    inventory.add(recipe_path, DEFAULT_RECIPE.as_posix(), RECIPE_SHA256)
    inventory.add(
        source_path(root, recipe["source_cache_recipe"]["path"]),
        recipe["source_cache_recipe"]["path"],
        recipe["source_cache_recipe"]["sha256"],
    )
    inventory.add(
        root / recipe["target_cache"]["path"],
        recipe["target_cache"]["path"],
        TARGET_SHA256,
    )
    inventory.add(cache_path, DEFAULT_CACHE_RECEIPT.as_posix())
    inventory.add(cache_admission_path, DEFAULT_CACHE_ADMISSION.as_posix(), CACHE_ADMISSION_SHA256)
    inventory.add(cache_spec_path, DEFAULT_CACHE_SPEC.as_posix(), CACHE_SPEC_SHA256)
    inventory.add(cache_guard_path, DEFAULT_CACHE_GUARD.as_posix(), CACHE_GUARD_SHA256)
    inventory.add(root / DEFAULT_CACHE_LOG, DEFAULT_CACHE_LOG.as_posix())
    inventory.add(milestone_path, DEFAULT_PARENT_MILESTONE.as_posix(), PARENT_MILESTONE_SHA256)

    for fold in recipe["folds"]:
        inventory.add(
            root / fold["split_manifest"], fold["split_manifest"], fold["split_manifest_sha256"]
        )
        inventory.add(
            root / fold["subset"]["path"], fold["subset"]["path"], fold["subset"]["sha256"]
        )
        inventory.add(
            root / fold["parent"]["provenance"],
            fold["parent"]["provenance"],
            fold["parent"]["provenance_sha256"],
        )
    for name, digest in cache_admission.get("source_sha256", {}).items():
        inventory.add(root / name, name, digest)
    for name, digest in cache.get("producer_source_sha256", {}).items():
        inventory.add(root / name, name, digest)
    for field in ("source_sha256", "fit_source_sha256", "authored_file_sha256"):
        mapping = source_map.get(field, {})
        if not isinstance(mapping, dict):
            raise ValueError(f"actor source map field is malformed: {field}")
        add_source_hash_map(inventory, root, mapping)
    inventory.add(source_map_path, "source/SOURCE_SHA256_R1.json")
    inventory.add(grader_path, "grader/age_readout_grade.py")
    for name, digest in grader_deps.items():
        inventory.add(root / name, name, digest)

    for record in cache["files"]:
        fold_id, kind = int(record["fold"]), str(record["kind"])
        sidecar = Path(record["sidecar_path"])
        inventory.add(sidecar, f"parent_cache/f{fold_id}/{sidecar.name}", record["sidecar_sha256"])
        if kind == "evaluation_predictions":
            prediction = Path(record["source_path"])
            size = int(record["identity"]["bytes"])
            inventory.add(
                prediction,
                f"parent_cache/f{fold_id}/evaluation_predictions.npy",
                record["sha256"],
                size,
            )
        elif kind not in {"fit_sequences", "evaluation_sequences"}:
            raise ValueError(f"unexpected cache payload type: {kind}")

    candidate_sidecars: dict[str, tuple[Path, Path, Path, Path]] = {}
    expected_runs = [
        "age_readout_f0_s1001_a_r1",
        "age_readout_f0_s1001_g_r1",
        "age_readout_f0_s1001_f_r1",
        "age_readout_f1_s1002_a_r1",
        "age_readout_f1_s1002_g_r1",
        "age_readout_f1_s1002_f_r1",
    ]
    if [str(cell["run"]) for cell in recipe["cells"]] != expected_runs:
        raise ValueError("recipe cell order differs from the frozen six-cell panel")
    for cell in recipe["cells"]:
        output = Path(cell["output"])
        run = str(cell["run"])
        paths = tuple(
            output / name
            for name in ("model_ep2.pt", "prediction.npy", "prediction.json", "provenance.json")
        )
        candidate_sidecars[run] = cast(tuple[Path, Path, Path, Path], paths)
        for path in paths:
            inventory.add(path, f"candidate/{run}/{path.name}")
    return (
        inventory,
        recipe,
        source_map,
        {
            "cache": cache,
            "cache_admission": cache_admission,
            "milestone": milestone,
            "grader_deps": grader_deps,
            "candidate_paths": candidate_sidecars,
            "source_map_path": source_map_path,
            "grader_path": grader_path,
        },
    )


class HashingReader:
    """Stream a source into tar while comparing it with the precomputed digest."""

    def __init__(self, path: Path, deadline: float) -> None:
        self.stream = path.open("rb")
        self.deadline = deadline
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        """Read one chunk and enforce the finite archive deadline."""
        if time.monotonic() >= self.deadline:
            raise TimeoutError("archive packing exceeded its time budget")
        block = self.stream.read(size)
        self.digest.update(block)
        self.bytes_read += len(block)
        return block

    def close(self) -> None:
        """Close the underlying source file."""
        self.stream.close()


def write_manifest_and_archive(
    inventory: Inventory, source_root: Path, prepare_dir: Path, deadline: float
) -> dict[str, Any]:
    """Write deterministic manifest-first tar.gz without materializing source copies."""
    manifest_path = prepare_dir / MANIFEST_FILENAME
    source_path_value = str(manifest_path.absolute())
    manifest = {
        "schema": "wnn.age-readout-grade-bundle.v1",
        "source_root": str(source_root),
        "source_path": source_path_value,
        "files": inventory.manifest_entries(),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(manifest_bytes)
    archive_path = prepare_dir / ARCHIVE_FILENAME
    with (
        archive_path.open("xb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=1) as gz,
        tarfile.open(fileobj=gz, mode="w|") as tar,
    ):
        info = tarfile.TarInfo(MANIFEST_FILENAME)
        info.size, info.mode = len(manifest_bytes), 0o644
        tar.addfile(info, io.BytesIO(manifest_bytes))
        for item in manifest["files"]:
            if time.monotonic() >= deadline:
                raise TimeoutError("archive packing exceeded its time budget")
            source = Path(item["source_path"])
            info = tarfile.TarInfo(item["bundle_path"])
            info.size, info.mode = int(item["bytes"]), 0o444
            reader = HashingReader(source, deadline)
            try:
                tar.addfile(info, reader)
            finally:
                reader.close()
            if (
                reader.bytes_read != int(item["bytes"])
                or reader.digest.hexdigest() != item["sha256"]
            ):
                raise RuntimeError(f"source changed while packaging: {source}")
    archive_sha = sha256_file(archive_path, deadline)
    return {
        "manifest_path": source_path_value,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha,
        "archive_bytes": archive_path.stat().st_size,
        "mapped_bytes": sum(int(item["bytes"]) for item in manifest["files"]),
        "file_count": len(manifest["files"]),
    }


def verify_candidate_metadata(
    context: dict[str, Any],
    inventory: Inventory,
    source_map: dict[str, Any],
) -> None:
    """Bind candidate sidecars/checkpoints to the source map without decoding NPYs."""
    expected_sources = source_map.get("fit_source_sha256")
    if not isinstance(expected_sources, dict) or not expected_sources:
        raise ValueError("actor source map has no fit source pins")
    candidates = {entry["source_path"]: entry for entry in inventory.entries.values()}
    device_set: set[str] = set()
    for run, paths in context["candidate_paths"].items():
        checkpoint_path, prediction_path, sidecar_path, provenance_path = paths
        sidecar = read_object(sidecar_path)
        provenance = read_object(provenance_path)
        checkpoint_sha = candidates[str(checkpoint_path)]["sha256"]
        prediction_sha = candidates[str(prediction_path)]["sha256"]
        if (
            sidecar.get("run") != run
            or sidecar.get("checkpoint_sha256") != checkpoint_sha
            or sidecar.get("prediction_sha256") != prediction_sha
            or provenance.get("run") != run
            or provenance.get("source_sha256") != expected_sources
            or provenance.get("fit_source_sha256") != expected_sources
            or provenance.get("checkpoint", {}).get("sha256") != checkpoint_sha
            or provenance.get("prediction", {}).get("sha256") != prediction_sha
        ):
            raise ValueError(f"candidate metadata does not match pinned outputs: {run}")
        device = str(provenance.get("runtime", {}).get("device", ""))
        if not (device == "cpu" or re.fullmatch(r"cuda:[0-9]+", device)):
            raise ValueError(f"candidate fit device is not pinned: {run}")
        device_set.add(device)
    if len(device_set) != 1:
        raise ValueError("six candidate fits do not share one device")
    context["candidate_device"] = next(iter(device_set))


def prepare(args: argparse.Namespace) -> int:
    """Build the content manifest and archive after all six outputs exist."""
    started = time.monotonic()
    deadline = started + args.max_seconds
    root = args.repo_root.resolve(strict=True)
    prepare_dir = args.prepare_dir.resolve(strict=False)
    if not prepare_dir.is_relative_to(STORAGE_ROOT.resolve()) or prepare_dir.exists():
        raise ValueError("prepare output must be a fresh directory under the assigned disk root")
    inventory, recipe, source_map, context = build_inventory(
        root, args.source_map.absolute(), args.grader_source.absolute()
    )
    uncompressed_bytes = inventory.preflight()
    prepare_dir.mkdir(parents=True, exist_ok=False)
    # Hash only after the complete source closure, including all six four-file
    # candidate outputs, has passed the metadata-only existence/size gate.
    hash_started = time.monotonic()
    inventory.hash_all(deadline)
    hash_seconds = time.monotonic() - hash_started
    verify_candidate_metadata(context, inventory, source_map)
    archive_started = time.monotonic()
    archive = write_manifest_and_archive(inventory, root, prepare_dir, deadline)
    archive_seconds = time.monotonic() - archive_started
    if archive["mapped_bytes"] != uncompressed_bytes or archive["mapped_bytes"] > MAX_BUNDLE_BYTES:
        raise RuntimeError("bundle byte accounting changed during preparation")
    receipt = {
        "schema": "wnn.age-readout-grade-package-prepare.v1",
        "status": "prepared_waiting_root_admission",
        "recipe_sha256": RECIPE_SHA256,
        "source_map_path": str(args.source_map.absolute()),
        "source_map_sha256": sha256_file(args.source_map, deadline),
        "candidate_device": context["candidate_device"],
        "uncompressed_bytes": uncompressed_bytes,
        "hash_seconds": hash_seconds,
        "archive_seconds": archive_seconds,
        "elapsed_seconds": time.monotonic() - started,
        **archive,
    }
    (prepare_dir / "PACKAGE_PREPARE_R1.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in (
                    "status",
                    "file_count",
                    "mapped_bytes",
                    "archive_bytes",
                    "manifest_sha256",
                    "archive_sha256",
                    "elapsed_seconds",
                )
            },
            sort_keys=True,
        )
    )
    return 0


def check_admission(
    admission: dict[str, Any],
    admission_path: Path,
    prepared: dict[str, Any],
    manifest: dict[str, Any],
    root: Path,
    source_map: dict[str, Any],
    grader_deps: dict[str, str],
) -> str:
    """Require the final root receipt to bind this exact prepared closure."""
    if (
        admission.get("schema") != "wnn.age-readout-grade-admission.v1"
        or admission.get("status") != "admitted"
        or admission.get("allow_partial") is not False
        or admission.get("experiment_id") != "E-AGE-READOUT-20261023-R1"
        or admission.get("recipe_sha256") != RECIPE_SHA256
        or admission.get("metric") != "global_wp_clip2_v1"
    ):
        raise ValueError("root grade admission is not final/admitted")
    artifact = admission.get("artifact_manifest", {})
    if artifact.get("path") != manifest.get("source_path") or artifact.get(
        "sha256"
    ) != prepared.get("manifest_sha256"):
        raise ValueError("root admission does not bind the prepared manifest")
    cache = admission.get("cache_verification", {})
    if cache.get("path") != DEFAULT_CACHE_RECEIPT.as_posix() or cache.get("sha256") != next(
        item["sha256"]
        for item in manifest["files"]
        if item["bundle_path"] == DEFAULT_CACHE_RECEIPT.as_posix()
    ):
        raise ValueError("root admission does not bind the local cache verification receipt")
    expected_cells = [
        "age_readout_f0_s1001_a_r1",
        "age_readout_f0_s1001_g_r1",
        "age_readout_f0_s1001_f_r1",
        "age_readout_f1_s1002_a_r1",
        "age_readout_f1_s1002_g_r1",
        "age_readout_f1_s1002_f_r1",
    ]
    if admission.get("required_cells") != expected_cells:
        raise ValueError("root admission cell order differs")
    limits = admission.get("limits", {})
    if (
        not 1 <= int(limits.get("cpu_cores", 0)) <= 2
        or not 1 <= int(limits.get("ram_bytes", 0)) <= 4_000_000_000
        or not 1 <= int(limits.get("wall_seconds", 0)) <= 1_800
    ):
        raise ValueError("root grade limits exceed the Kaggle job envelope")
    if admission.get("fit_source_sha256") != source_map.get("fit_source_sha256"):
        raise ValueError("root fit source map differs from the actor source manifest")
    adapter_sha = next(
        item["sha256"]
        for item in manifest["files"]
        if item["bundle_path"] == "grader/age_readout_grade.py"
    )
    grader_hashes = admission.get("grader_source_sha256", {})
    if grader_hashes.get("grader/age_readout_grade.py") != adapter_sha:
        raise ValueError("root admission grader adapter hash differs")
    for name, digest in grader_deps.items():
        if grader_hashes.get(name) != digest:
            raise ValueError(f"root admission scorer dependency differs: {name}")
    if admission.get("shared_readout_device") != prepared.get("candidate_device"):
        raise ValueError("root shared device differs from six candidate provenance files")
    return hashlib.sha256(admission_path.read_bytes()).hexdigest()


def replace_driver_constants(
    source: Path,
    destination: Path,
    dataset_slug: str,
    archive_sha256: str,
    admission_sha256: str,
) -> str:
    """Patch only the four blank delivery constants in a fresh driver copy."""
    text = source.read_text(encoding="utf-8")
    values = {
        "BUNDLE_DATASET_SLUG": dataset_slug,
        "BUNDLE_ARCHIVE_SHA256": archive_sha256,
        "ADMISSION_RELATIVE_PATH": ADMISSION_FILENAME,
        "ADMISSION_SHA256": admission_sha256,
    }
    for name, value in values.items():
        pattern = re.compile(rf"(?m)^{name} = \"\"$")
        text, count = pattern.subn(f"{name} = {value!r}", text)
        if count != 1:
            raise ValueError(f"expected one blank driver constant: {name}")
    ast.parse(text, filename=str(destination))
    destination.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def finalize(args: argparse.Namespace) -> int:
    """Stage private dataset/kernel folders after root admission is complete."""
    started = time.monotonic()
    prepared_dir = args.prepared_dir.resolve(strict=True)
    root = args.repo_root.resolve(strict=True)
    output_dir = args.output_dir.resolve(strict=False)
    if not output_dir.is_relative_to(STORAGE_ROOT.resolve()) or output_dir.exists():
        raise ValueError("final output must be a fresh directory under the assigned disk root")
    prepared = read_object(prepared_dir / "PACKAGE_PREPARE_R1.json")
    manifest_path = prepared_dir / MANIFEST_FILENAME
    archive_path = prepared_dir / ARCHIVE_FILENAME
    manifest = read_object(manifest_path)
    source_map_path = Path(prepared["source_map_path"])
    source_map = read_object(source_map_path)
    if sha256_file(source_map_path, time.monotonic() + 600) != prepared["source_map_sha256"]:
        raise ValueError("actor source map changed after package preparation")
    grader_path = args.grader_source.absolute()
    grader_deps = grader_core_dependencies(grader_path)
    if sha256_file(manifest_path, time.monotonic() + 600) != prepared["manifest_sha256"]:
        raise ValueError("prepared bundle manifest changed")
    if manifest.get("source_path") != prepared.get("manifest_path"):
        raise ValueError("prepared manifest path receipt differs")
    mapped_hashes = {item["bundle_path"]: item["sha256"] for item in manifest["files"]}
    if sha256_file(grader_path, time.monotonic() + 600) != mapped_hashes.get(
        "grader/age_readout_grade.py"
    ):
        raise ValueError("grader source changed after package preparation")
    for name, digest in grader_deps.items():
        if mapped_hashes.get(name) != digest:
            raise ValueError(f"grader dependency changed after package preparation: {name}")
    admission = read_object(args.admission.resolve(strict=True))
    admission_sha = check_admission(
        admission, args.admission, prepared, manifest, root, source_map, grader_deps
    )
    verify_started = time.monotonic()
    archive_sha = sha256_file(archive_path, verify_started + 600)
    if archive_sha != prepared["archive_sha256"]:
        raise ValueError("prepared bundle archive changed")
    archive_size = archive_path.stat().st_size
    metadata = {
        "id": args.dataset_id,
        "title": args.dataset_id.split("/", 1)[1],
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    }
    upload_bytes = (
        archive_size
        + args.admission.stat().st_size
        + len((json.dumps(metadata, indent=2) + "\n").encode())
    )
    if upload_bytes > MAX_BUNDLE_BYTES:
        raise ValueError(f"final private dataset exceeds one-gigabyte gate: {upload_bytes}")
    dataset_slug = args.dataset_id.split("/", 1)[1]
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,99}", dataset_slug):
        raise ValueError("invalid private dataset slug")
    kernel_slug = args.kernel_id.split("/", 1)[1]
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,99}", kernel_slug):
        raise ValueError("invalid CPU grader kernel slug")
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_dir = output_dir / "dataset_upload"
    kernel_dir = output_dir / "kernel_upload"
    dataset_dir.mkdir()
    kernel_dir.mkdir()
    os.link(archive_path, dataset_dir / ARCHIVE_FILENAME)
    os.chmod(dataset_dir / ARCHIVE_FILENAME, 0o444)
    (dataset_dir / ADMISSION_FILENAME).write_bytes(args.admission.read_bytes())
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    driver_sha = replace_driver_constants(
        args.driver_source.absolute(),
        kernel_dir / "kernel_driver.py",
        dataset_slug,
        archive_sha,
        admission_sha,
    )
    kernel_metadata = {
        "code_file": "kernel_driver.py",
        "competition_sources": [],
        "dataset_sources": [args.dataset_id],
        "enable_gpu": False,
        "enable_internet": False,
        "enable_tpu": False,
        "id": args.kernel_id,
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "",
        "title": f"{dataset_slug} CPU grader",
    }
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(kernel_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    dataset_files = sorted(p.name for p in dataset_dir.iterdir())
    kernel_files = sorted(p.name for p in kernel_dir.iterdir())
    if dataset_files != sorted([ARCHIVE_FILENAME, ADMISSION_FILENAME, "dataset-metadata.json"]):
        raise RuntimeError("dataset upload folder has unexpected files")
    if kernel_files != ["kernel-metadata.json", "kernel_driver.py"]:
        raise RuntimeError("kernel upload folder has unexpected files")
    receipt = {
        "schema": "wnn.age-readout-grade-package-finalize.v1",
        "status": "private_cpu_packet_ready_not_uploaded",
        "dataset_id": args.dataset_id,
        "kernel_id": args.kernel_id,
        "manifest_path": prepared["manifest_path"],
        "manifest_sha256": prepared["manifest_sha256"],
        "archive_sha256": archive_sha,
        "admission_sha256": admission_sha,
        "kernel_driver_sha256": driver_sha,
        "dataset_upload_files": dataset_files,
        "kernel_upload_files": kernel_files,
        "dataset_upload_bytes": upload_bytes,
        "dataset_upload_dir": str(dataset_dir),
        "kernel_upload_dir": str(kernel_dir),
        "dataset_create_command": f"kaggle datasets create --path {dataset_dir} --dir-mode skip",
        "kernel_push_command": f"kaggle kernels push -p {kernel_dir}",
        "status_command": f"kaggle kernels status {args.kernel_id}",
        "collect_command": (
            "scripts/kaggle_collect_kernel_output.py --kernel "
            f"{args.kernel_id} --dest <fresh-disk-destination>"
        ),
        "provider_mutations": 0,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "PACKAGE_FINALIZATION_R1.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in (
                    "status",
                    "dataset_upload_bytes",
                    "manifest_sha256",
                    "archive_sha256",
                    "admission_sha256",
                )
            },
            sort_keys=True,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    """Build the two-phase no-launch operator CLI."""
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--repo-root", type=Path, default=Path.cwd())
    prep.add_argument("--source-map", type=Path, default=DEFAULT_SOURCE_MAP)
    prep.add_argument("--grader-source", type=Path, default=DEFAULT_GRADER_SOURCE)
    prep.add_argument(
        "--prepare-dir",
        type=Path,
        default=STORAGE_ROOT / "kaggle_ops/prepared_bundle_r1",
    )
    prep.add_argument("--max-seconds", type=int, default=600)
    final = commands.add_parser("finalize")
    final.add_argument("--repo-root", type=Path, default=Path.cwd())
    final.add_argument("--prepared-dir", type=Path, required=True)
    final.add_argument("--admission", type=Path, required=True)
    final.add_argument("--grader-source", type=Path, default=DEFAULT_GRADER_SOURCE)
    final.add_argument("--driver-source", type=Path, default=DEFAULT_DRIVER_SOURCE)
    final.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    final.add_argument("--kernel-id", default=DEFAULT_KERNEL_ID)
    final.add_argument(
        "--output-dir",
        type=Path,
        default=STORAGE_ROOT / "kaggle_ops/final_packet_r1",
    )
    return root


def main() -> int:
    """Run a source-only prepare/finalize phase; never call Kaggle APIs."""
    args = parser().parse_args()
    if args.command == "prepare":
        if args.max_seconds <= 0 or args.max_seconds > 600:
            raise ValueError("prepare wall cap must be 1–600 seconds")
        return prepare(args)
    if args.command == "finalize":
        if not args.dataset_id.startswith("vladkochetov07/") or not args.kernel_id.startswith(
            "vladkochetov07/"
        ):
            raise ValueError("private dataset and kernel IDs must use the authorized account")
        return finalize(args)
    raise ValueError("unknown builder command")


if __name__ == "__main__":
    raise SystemExit(main())
