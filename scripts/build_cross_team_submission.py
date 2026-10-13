"""Build deterministic inventory-bound cross-team packages.

The legacy path reuses the xteam_h4 callback and libraries. An optional QAT path
replaces the fixed fourth member with exact versioned codes/scales and a
hash-bound portable native runtime. Neither path runs the scorer or establishes
numerical equivalence to source callbacks.
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
QAT_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
QAT_ARCHIVE_MEMBER = "qat_archive.py"
QAT_CODE_CONTRACT = "exact_q8_codes_scales_v1"
NATIVE_VARIANTS = (
    "v2/wnn_native.abi3.so",
    "avx2/wnn_native.abi3.so",
    "v3/wnn_native.abi3.so",
)


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


def _sha256_value(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _hash_bound_file(record: object, label: str) -> tuple[Path, dict[str, object]]:
    item = _object(record, label)
    source_name = item.get("path")
    expected = _sha256_value(item.get("sha256"), f"{label} sha256")
    if not isinstance(source_name, str):
        raise ValueError(f"{label} needs a file path")
    source = Path(source_name).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"{label} is not a regular file")
    actual = sha256_file(source)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    expected_bytes = item.get("bytes")
    if expected_bytes is not None and expected_bytes != source.stat().st_size:
        raise ValueError(f"{label} byte count mismatch")
    return source, {
        **item,
        "path": str(source),
        "sha256": actual,
        "bytes": source.stat().st_size,
    }


def _read_qat_release_spec(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    spec = _object(json.loads(path.read_text(encoding="utf-8")), "QAT release spec")
    if spec.get("version") != 1:
        raise ValueError("unsupported QAT release spec version")
    replacement = _object(spec.get("replacement"), "QAT replacement")
    index = replacement.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("QAT replacement index must be a non-negative integer")
    if not isinstance(replacement.get("candidate"), str) or not isinstance(
        replacement.get("weights"), str
    ):
        raise ValueError("QAT replacement must bind an original candidate and member")
    _sha256_value(replacement.get("source_member_sha256"), "replaced source member sha256")
    coefficient = replacement.get("coef")
    if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)):
        raise ValueError("QAT replacement coefficient must be numeric")
    if not math.isfinite(float(coefficient)) or float(coefficient) < 0:
        raise ValueError("QAT replacement coefficient must be finite and non-negative")

    archive_path, archive = _hash_bound_file(spec.get("archive"), "QAT archive")
    archive_item = _object(spec["archive"], "QAT archive")
    preset = archive_item.get("preset")
    if preset not in ("recurrent_w4", "input_w4_recurrent_w8"):
        raise ValueError("QAT archive preset must be a versioned recurrent W4 policy")
    if __package__:
        from .templates.qat_archive import read_qat_archive
    else:
        # Direct ``python scripts/build_cross_team_submission.py`` places the
        # scripts directory, rather than the repository root, on sys.path.
        from templates.qat_archive import read_qat_archive

    archive_weights, _, archive_metadata = read_qat_archive(
        archive_path, expected_preset=cast(str, preset)
    )
    weight_schema = {
        name: {"shape": list(value.shape), "dtype": value.dtype.str}
        for name, value in archive_weights.items()
    }
    del archive_weights

    checkpoint_path, checkpoint = _hash_bound_file(
        spec.get("checkpoint"), "QAT training checkpoint"
    )
    checkpoint_item = _object(spec["checkpoint"], "QAT training checkpoint")
    for field in ("run_reference", "training_rule", "data_membership"):
        if not isinstance(checkpoint_item.get(field), str) or not checkpoint_item[field]:
            raise ValueError(f"QAT checkpoint provenance needs {field}")
    seed = checkpoint_item.get("seed")
    updates = checkpoint_item.get("updates")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("QAT checkpoint seed must be a non-negative integer")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
        raise ValueError("QAT checkpoint update count must be positive")
    _sha256_value(checkpoint_item.get("parent_checkpoint_sha256"), "QAT parent checkpoint sha256")
    scale_semantics = checkpoint_item.get("scale_learning")
    if scale_semantics is not None and not isinstance(scale_semantics, dict):
        raise ValueError("QAT scale-learning provenance must be an object or null")
    if (
        any(
            token in cast(str, checkpoint_item["training_rule"]).lower()
            for token in ("learned", "lsq")
        )
        and not isinstance(scale_semantics, dict)
    ):
        raise ValueError("learned-scale QAT must include its exact scale-learning semantics")
    return spec, {
        "spec": {"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)},
        "archive": {
            **archive,
            "preset": preset,
            "metadata": archive_metadata,
            "weight_schema": weight_schema,
        },
        "checkpoint": checkpoint,
    }


def _read_native_manifest(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    resolved = path.resolve(strict=True)
    manifest_hash = sha256_file(resolved)
    manifest = _object(json.loads(resolved.read_text(encoding="utf-8")), "native manifest")
    if (
        manifest.get("native_api") != QAT_CODE_CONTRACT
        or manifest.get("abi") != "abi3-py311"
        or manifest.get("target_cpu") != "x86-64-v2"
        or manifest.get("runtime_dispatch") is not True
    ):
        raise ValueError("native manifest does not bind the required portable exact-Q8 ABI")
    raw_files = _object(manifest.get("files"), "native manifest files")
    if set(raw_files) != set(NATIVE_VARIANTS):
        raise ValueError("native manifest must provide v2, avx2, and v3 library paths")
    files: dict[str, dict[str, object]] = {}
    for relative_name in NATIVE_VARIANTS:
        source, record = _hash_bound_file(raw_files[relative_name], f"native {relative_name}")
        if record["bytes"] < 1:
            raise ValueError(f"native library is empty: {relative_name}")
        files[relative_name] = {**record, "source_path": str(source)}
    return manifest, {
        "path": str(resolved),
        "sha256": manifest_hash,
        "native_api": manifest["native_api"],
        "abi": manifest["abi"],
        "target_cpu": manifest["target_cpu"],
        "runtime_dispatch": True,
        "files": files,
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
    inventory_path: Path,
    recipe_path: Path,
    out_dir: Path,
    max_zip_bytes: int,
    qat_release_spec_path: Path | None = None,
    native_manifest_path: Path | None = None,
) -> dict[str, object]:
    """Build a deterministic cross-team package, optionally with exact QAT import."""
    if max_zip_bytes < 1 or max_zip_bytes > MAX_ZIP_BYTES:
        raise ValueError(f"max ZIP size must be between 1 and {MAX_ZIP_BYTES} bytes")
    if (qat_release_spec_path is None) != (native_manifest_path is None):
        raise ValueError("QAT release spec and exact-import native manifest must be supplied together")
    inventory_path = inventory_path.resolve(strict=True)
    recipe_path = recipe_path.resolve(strict=True)
    inventory_hash = sha256_file(inventory_path)
    recipe_hash = sha256_file(recipe_path)
    inventory = _object(json.loads(inventory_path.read_text(encoding="utf-8")), "inventory")
    name, members = parse_recipe(recipe_path)

    qat_spec: dict[str, object] | None = None
    qat_binding: dict[str, object] | None = None
    native_binding: dict[str, object] | None = None
    if qat_release_spec_path is not None and native_manifest_path is not None:
        qat_spec, qat_binding = _read_qat_release_spec(qat_release_spec_path.resolve(strict=True))
        _, native_binding = _read_native_manifest(native_manifest_path.resolve(strict=True))
        replacement = _object(qat_spec["replacement"], "QAT replacement")
        replace_index = cast(int, replacement["index"])
        if replace_index >= len(members):
            raise ValueError("QAT replacement index is outside the recipe")
        if (
            len(members) != 4
            or replace_index != 3
            or replacement["candidate"] != "E"
            or replacement["weights"] != "weights_0.npz"
            or not math.isclose(float(replacement["coef"]), 0.25, rel_tol=0.0, abs_tol=1e-8)
        ):
            raise ValueError("QAT replacement must be the fixed fourth E/weights_0 blend member")
        original_item = members[replace_index]
        if (
            original_item["candidate"] != replacement["candidate"]
            or original_item["weights"] != replacement["weights"]
            or not math.isclose(
                cast(float, original_item["coef"]),
                float(replacement["coef"]),
                rel_tol=0.0,
                abs_tol=1e-8,
            )
        ):
            raise ValueError("QAT release spec does not match the selected recipe member")

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

    if qat_spec is not None and qat_binding is not None:
        replacement = _object(qat_spec["replacement"], "QAT replacement")
        replace_index = cast(int, replacement["index"])
        replaced_hash = cast(str, resolved_members[replace_index][1]["sha256"])
        if replaced_hash != replacement["source_member_sha256"]:
            raise ValueError("QAT replacement source hash differs from inventory-bound member")
        import numpy as np

        original_source = resolved_members[replace_index][0]
        with np.load(original_source, allow_pickle=False) as original:
            original_schema = {
                name: {"shape": list(original[name].shape), "dtype": original[name].dtype.str}
                for name in original.files
            }
        archive_binding = _object(qat_binding["archive"], "QAT archive binding")
        if original_schema != archive_binding["weight_schema"]:
            raise ValueError("QAT archive tensor names, shapes, or dtypes differ from replaced member")

    output_root = _physical_output_parent(out_dir)
    output_root.mkdir()
    package_root = output_root / "pkg"
    package_root.mkdir()
    file_records: dict[str, dict[str, object]] = {}
    if qat_spec is None:
        package_base_sources = base_sources
    else:
        # The QAT callback requires the exact-import native API and a dedicated
        # versioned archive reader; legacy base libraries are not reused.
        package_base_sources = {"precision.txt": base_sources["precision.txt"]}
    for relative_name, source_name in package_base_sources.items():
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

    if qat_spec is not None and qat_binding is not None and native_binding is not None:
        template_solution = QAT_TEMPLATE_DIR / "qat_solution.py"
        template_reader = QAT_TEMPLATE_DIR / "qat_archive.py"
        for source, relative_name in (
            (template_solution, "solution.py"),
            (template_reader, QAT_ARCHIVE_MEMBER),
        ):
            destination = package_root / relative_name
            shutil.copyfile(source, destination)
            file_records[relative_name] = {
                "sha256": sha256_file(destination),
                "bytes": destination.stat().st_size,
                "source_template": str(source.resolve(strict=True)),
            }
        for relative_name in NATIVE_VARIANTS:
            native_record = _object(
                _object(native_binding["files"], "native binding files")[relative_name],
                f"native binding {relative_name}",
            )
            source = Path(cast(str, native_record["source_path"]))
            destination = package_root / relative_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if sha256_file(destination) != native_record["sha256"]:
                raise ValueError(f"copied native library changed: {relative_name}")
            file_records[relative_name] = {
                "sha256": native_record["sha256"],
                "bytes": native_record["bytes"],
                "source_native_manifest": native_binding["path"],
            }
        release_replacement = _object(qat_spec["replacement"], "QAT replacement")
        archive_record = _object(qat_binding["archive"], "QAT archive binding")
        checkpoint_record = _object(qat_binding["checkpoint"], "QAT checkpoint binding")
        provenance_path = package_root / "qat_provenance.json"
        provenance_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "replacement": release_replacement,
                    "archive": {
                        "sha256": archive_record["sha256"],
                        "preset": archive_record["preset"],
                        "format": archive_record["metadata"]["format"],
                        "version": archive_record["metadata"]["version"],
                    },
                    "checkpoint": {
                        key: value for key, value in checkpoint_record.items() if key != "path"
                    },
                    "native_api": native_binding["native_api"],
                    "native_manifest_sha256": native_binding["sha256"],
                    "note": (
                        "Training membership and rule are provenance labels from the bound release "
                        "spec; this package build makes no quality or serving-speed claim."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        file_records["qat_provenance.json"] = {
            "sha256": sha256_file(provenance_path),
            "bytes": provenance_path.stat().st_size,
            "generated_from_release_spec": qat_binding["spec"],
        }

    package_members: list[dict[str, object]] = []
    source_member_records: list[dict[str, object]] = []
    for index, (source, source_record, ancestry, item) in enumerate(resolved_members):
        candidate = cast(str, item["candidate"])
        source_weights = cast(str, item["weights"])
        coefficient = cast(float, item["coef"])
        if qat_spec is not None and qat_binding is not None and index == cast(
            int, _object(qat_spec["replacement"], "QAT replacement")["index"]
        ):
            archive_binding = _object(qat_binding["archive"], "QAT archive binding")
            archive_source = Path(cast(str, archive_binding["path"]))
            archive_name = f"qat_weights_{index}.npz"
            destination = package_root / archive_name
            shutil.copyfile(archive_source, destination)
            copied_hash = sha256_file(destination)
            if copied_hash != archive_binding["sha256"]:
                raise ValueError("copied QAT archive differs from its release spec")
            checkpoint = _object(qat_binding["checkpoint"], "QAT checkpoint binding")
            package_member = {
                "qat_archive": archive_name,
                "qat_preset": archive_binding["preset"],
                "coef": coefficient,
                "source": f"{candidate}/{source_weights} (replaced by exact QAT transport)",
                "provenance": "qat_provenance.json",
                "training_checkpoint_sha256": checkpoint["sha256"],
                "training_rule": _object(qat_spec["checkpoint"], "QAT checkpoint")[
                    "training_rule"
                ],
            }
            package_members.append(package_member)
            source_member_records.append(
                {
                    "candidate": candidate,
                    "weights": source_weights,
                    "coef": coefficient,
                    "source_sha256": source_record["sha256"],
                    "source_bytes": source.stat().st_size,
                    "inventory_member_record": ancestry,
                    "qat_archive_sha256": copied_hash,
                    "qat_archive_bytes": destination.stat().st_size,
                    "training_checkpoint_sha256": checkpoint["sha256"],
                    "training_rule": package_member["training_rule"],
                }
            )
            file_records[archive_name] = {
                "sha256": copied_hash,
                "bytes": destination.stat().st_size,
                "source_path": str(archive_source),
                "source_preset": archive_binding["preset"],
            }
        else:
            filename = f"weights_{index}.npz"
            destination = package_root / filename
            shutil.copyfile(source, destination)
            copied_hash = sha256_file(destination)
            if copied_hash != source_record["sha256"]:
                raise ValueError(f"copied team weight differs from inventory: {source}")
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
    for relative_name, source_name in package_base_sources.items():
        source = Path(source_name)
        if sha256_file(source) != base_records["files"][relative_name]["sha256"]:
            raise ValueError(f"inherited xteam_h4 source changed during build: {relative_name}")

    archive_path = output_root / f"{name}.zip"
    zip_record = _write_deterministic_zip(package_root, archive_path, max_zip_bytes)
    if sha256_file(inventory_path) != inventory_hash:
        raise ValueError("inventory changed during package construction")
    if sha256_file(recipe_path) != recipe_hash:
        raise ValueError("recipe changed during package construction")
    if qat_binding is not None:
        qat_spec_binding = _object(qat_binding["spec"], "QAT release spec binding")
        if sha256_file(Path(cast(str, qat_spec_binding["path"]))) != qat_spec_binding["sha256"]:
            raise ValueError("QAT release spec changed during package construction")
        for key in ("archive", "checkpoint"):
            file_binding = _object(qat_binding[key], f"QAT {key} binding")
            if sha256_file(Path(cast(str, file_binding["path"]))) != file_binding["sha256"]:
                raise ValueError(f"QAT {key} changed during package construction")
    if native_binding is not None and sha256_file(
        Path(cast(str, native_binding["path"]))
    ) != native_binding["sha256"]:
        raise ValueError("native manifest changed during package construction")
    manifest: dict[str, object] = {
        "status": "built",
        "name": name,
        "intent": "exploratory cross-team submission package; no performance claim",
        "inventory_path": str(inventory_path),
        "inventory_sha256": inventory_hash,
        "recipe_path": str(recipe_path),
        "recipe_sha256": recipe_hash,
        "source_archives": source_archives,
        "base_package": BASE_CANDIDATE,
        "base_package_files": {
            relative_name: base_records["files"][relative_name]
            for relative_name in package_base_sources
        },
        "precision": NATIVE_PRECISION.strip(),
        "precision_note": (
            "Exact QAT codes/scales import through the portable q8vnni runtime; the callback-level "
            "prediction path is deterministic with budget_s=None. No package-level parity or "
            "runtime measurement is claimed."
            if qat_spec is not None
            else "Inherited q8vnni runtime; source-to-target output parity was not measured."
        ),
        "numerical_comparison": {
            "measured": False,
            "note": (
                "The package retains the cross-team members and coefficients but replaces one "
                "member with exact QAT row codes/scales. No score or prediction parity is claimed."
                if qat_spec is not None
                else "The package keeps the original xteam_h4 Q8 VNNI callback and native libraries. "
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
        "native_runtime": (
            {
                "callback_source": "QAT fixed-four callback template",
                "precision": NATIVE_PRECISION.strip(),
                "native_manifest": native_binding,
                "library_sha256": {
                    relative_name: file_records[relative_name]["sha256"]
                    for relative_name in NATIVE_VARIANTS
                },
            }
            if native_binding is not None
            else {
                "callback_source": BASE_CANDIDATE,
                "precision": NATIVE_PRECISION.strip(),
                "library_sha256": {
                    relative_name: file_records[relative_name]["sha256"]
                    for relative_name in BASE_FILES
                    if relative_name.endswith(".so")
                },
            }
        ),
        "qat_release": (
            {
                "release_spec": qat_binding["spec"],
                "archive": qat_binding["archive"],
                "checkpoint": qat_binding["checkpoint"],
                "replacement": qat_spec["replacement"],
            }
            if qat_spec is not None and qat_binding is not None
            else None
        ),
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
    parser.add_argument(
        "--qat-release-spec",
        type=Path,
        help="hash-bound archive/checkpoint semantics for one exact-code QAT replacement",
    )
    parser.add_argument(
        "--native-manifest",
        type=Path,
        help="hash-bound portable abi3 native libraries implementing exact Q8 code import",
    )
    args = parser.parse_args()
    manifest = build_package(
        args.inventory,
        args.recipe,
        args.out_dir,
        args.max_zip_bytes,
        qat_release_spec_path=args.qat_release_spec,
        native_manifest_path=args.native_manifest,
    )
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
