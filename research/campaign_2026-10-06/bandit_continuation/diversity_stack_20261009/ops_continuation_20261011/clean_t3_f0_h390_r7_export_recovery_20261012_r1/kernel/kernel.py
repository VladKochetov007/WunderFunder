from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

BINDINGS_FILENAME = "R7_EXPORT_BINDINGS_R1.json"
ROOT_RELEASE_FILENAME = "ROOT_EXPORT_RELEASE_R7.json"
EXPECTED_BINDINGS_SHA256 = "39127244077e0937934fce1fedc131f7fa440ff18f7bc7fc3508f6f69e318d0f"
BUNDLE_SLUG = "wnn-clean-t3-f0-teachers-20261011-r5"
R6_KERNEL_SLUG = "wnn-clean-t3-f0-teachers-20261011-r5-runner"
KERNEL_PROCESS_WALL_SECONDS_MAX = 16_800
FINALIZATION_RESERVE_SECONDS = 300
FREE_DISK_RESERVE_BYTES = 4_294_967_296
MAX_SINGLE_NPY_COPY_BYTES = 2_000_000_000
NON_ARRAY_OUTPUT_RESERVE_BYTES = 268_435_456


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_bindings(kernel_dir: Path) -> dict[str, Any]:
    path = kernel_dir / BINDINGS_FILENAME
    if sha256_file(path) != EXPECTED_BINDINGS_SHA256:
        raise PermissionError("R7 export binding bytes differ from the reviewed packet")
    bindings = read_json(path)
    if bindings.get("schema") != "wnn.kaggle.clean_t3_f0_export_only_r7_bindings.r1":
        raise ValueError("unexpected R7 export binding schema")
    return bindings


def require_root_release(kernel_dir: Path, bindings: dict[str, Any]) -> dict[str, Any]:
    release = read_json(kernel_dir / ROOT_RELEASE_FILENAME)
    if (
        release.get("schema") != "wnn.root.clean_t3_f0_export_only_release.r7.r1"
        or release.get("status") != "admitted"
        or release.get("authorized") is not True
    ):
        raise PermissionError("R7 root export release is pending; no workers may start")
    if release.get("binding") != bindings.get("root_release_binding"):
        raise PermissionError("R7 root release does not bind this exact export packet")
    return release


def sha_member(path: Path, expected_size: int, expected_sha: str) -> None:
    if path.stat().st_size != expected_size or sha256_file(path) != expected_sha:
        raise ValueError(f"source bytes differ from the R7 seal: {path.name}")


def checked_extract_bundle(
    archive: Path,
    manifest: dict[str, Any],
    manifest_sha256: str,
    expected_archive_sha256: str,
    destination: Path,
    bindings: dict[str, Any],
    deadline: float,
) -> None:
    if sha256_file(archive) != expected_archive_sha256:
        raise ValueError("R5 package archive hash differs from the sealed binding")
    if manifest.get("archive_sha256") != expected_archive_sha256:
        raise ValueError("R5 manifest archive digest differs from the sealed binding")
    if (
        manifest.get("archive_bytes") != archive.stat().st_size
        or manifest.get("private") is not True
    ):
        raise ValueError("R5 archive size/privacy metadata is invalid")
    if manifest_sha256 != bindings["r5_bundle_manifest_sha256"]:
        raise ValueError("R5 manifest hash differs from the sealed R7 binding")
    file_rows = manifest.get("files", [])
    expected_files = {row["path"]: (row["size"], row["sha256"]) for row in file_rows}
    if len(expected_files) != len(file_rows):
        raise ValueError("R5 bundle manifest repeats a file path")
    spec_by_path = {row["spec_path"]: row for row in bindings["r5_export_spec_catalog"].values()}
    declared_spec_paths = set(manifest.get("spec_paths", []))
    if not set(spec_by_path).issubset(declared_spec_paths):
        raise ValueError("R5 spec path inventory omits a declared R5 export spec")
    for path, spec in spec_by_path.items():
        if expected_files.get(path) != (spec["size_bytes"], spec["spec_sha256"]):
            raise ValueError(f"R5 spec metadata differs from its binding: {path}")
    allowed = set(expected_files)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("R7 refuses to extract over an existing project tree")
    destination.mkdir(parents=True, exist_ok=False)
    seen: set[str] = set()
    with tarfile.open(archive, mode="r:gz") as tar:
        for member in tar:
            if time.monotonic() >= deadline:
                raise TimeoutError("R7 bootstrap deadline elapsed during extraction")
            relative = PurePosixPath(member.name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in member.name
                or not member.isfile()
                or member.name not in allowed
                or member.name in seen
            ):
                raise ValueError("R5 archive contains an unsafe, duplicate, or undeclared member")
            seen.add(member.name)
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError("R5 archive member has no content")
            digest = hashlib.sha256()
            byte_count = 0
            with source, target.open("xb") as output:
                while block := source.read(1 << 20):
                    output.write(block)
                    digest.update(block)
                    byte_count += len(block)
            expected_size, expected_sha = expected_files[member.name]
            if byte_count != expected_size or digest.hexdigest() != expected_sha:
                raise ValueError(f"R5 archive member hash/size mismatch: {member.name}")
    if seen != allowed:
        raise ValueError(f"R5 archive member closure failed; missing={sorted(allowed - seen)[:5]}")


def find_source_mount(input_root: Path) -> Path:
    matches = [
        path
        for path in input_root.glob(f"**/{R6_KERNEL_SLUG}")
        if path.is_dir() and path.name == R6_KERNEL_SLUG
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one R6 source mount; found {len(matches)}")
    return matches[0]


def canonical_source_artifacts(source_root: Path, run_name: str) -> tuple[Path, Path]:
    suffix = ("experiment", "artifacts", "runs", run_name, "model_ep2.pt")
    matches = [path for path in source_root.rglob("model_ep2.pt") if path.parts[-5:] == suffix]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one canonical R6 checkpoint for {run_name}; found {len(matches)}"
        )
    checkpoint = matches[0]
    provenance = checkpoint.with_name("provenance.json")
    if not provenance.is_file() or provenance.is_symlink():
        raise FileNotFoundError(f"R6 provenance is absent or indirect for {run_name}")
    return checkpoint, provenance


def validate_provenance(path: Path, run_name: str, expected: dict[str, Any]) -> dict[str, Any]:
    provenance = read_json(path)
    args = provenance.get("args", {})
    if (
        provenance.get("selected_epoch") != 2
        or provenance.get("seed") != expected["seed"]
        or args.get("name") != run_name
        or args.get("seed") != expected["seed"]
        or args.get("hidden") != 384
        or args.get("layers") != 2
        or args.get("epochs") != 3
        or provenance.get("run_identity_source_sha256") != expected["run_identity_source_sha256"]
        or provenance.get("split_manifest_sha256") != expected["split_manifest_sha256"]
        or provenance.get("checkpoint_sha256", {}).get("model_ep2.pt")
        != expected["model_ep2_sha256"]
    ):
        raise ValueError(f"R6 provenance identity fields differ from R7 binding: {run_name}")
    return provenance


def mount_verified_checkpoints(
    root: Path, source_root: Path, checkpoint_bindings: dict[str, Any]
) -> list[Path]:
    mounted: list[Path] = []
    for run_name, expected in checkpoint_bindings.items():
        source_checkpoint, source_provenance = canonical_source_artifacts(source_root, run_name)
        sha_member(source_checkpoint, expected["model_ep2_bytes"], expected["model_ep2_sha256"])
        sha_member(source_provenance, expected["provenance_bytes"], expected["provenance_sha256"])
        validate_provenance(source_provenance, run_name, expected)
        target_dir = root / "artifacts" / "runs" / run_name
        if target_dir.exists() or target_dir.is_symlink():
            raise FileExistsError(f"R7 refuses to overwrite checkpoint run directory: {run_name}")
        target_dir.mkdir(parents=True, exist_ok=False)
        cp_link, prov_link = target_dir / "model_ep2.pt", target_dir / "provenance.json"
        cp_link.symlink_to(source_checkpoint)
        prov_link.symlink_to(source_provenance)
        mounted.extend((cp_link, prov_link))
    return mounted


def mount_verified_inputs(
    input_root: Path, root: Path, external_mounts: dict[str, Any], declared_paths: set[str]
) -> list[Path]:
    mounted: list[Path] = []
    for logical_name, mapping in external_mounts.items():
        logical = PurePosixPath(logical_name)
        if logical.is_absolute() or ".." in logical.parts or logical_name not in declared_paths:
            raise ValueError(f"unsafe R5 external input mapping: {logical_name}")
        dataset = (
            mapping["dataset"] if isinstance(mapping, dict) else "vladkochetov07/wunder-funder"
        )
        filename = mapping["filename"] if isinstance(mapping, dict) else mapping
        slug = dataset.split("/", maxsplit=1)[-1]
        matches = [p for p in input_root.glob(f"**/{slug}/{filename}") if p.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one authorized R5 input {dataset}/{filename}; found {len(matches)}"
            )
        target = root.joinpath(*logical.parts)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"R7 refuses to overwrite mounted input {logical_name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(matches[0])
        mounted.append(target)
    return mounted


def import_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import sealed R5 helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_export_specs(
    root: Path, bindings: dict[str, Any]
) -> tuple[Any, Any, list[dict[str, Any]]]:
    driver = import_source_module("r7_r5_export_driver", root / bindings["r5_driver_relative_path"])
    helpers = driver.load_bootstrap_helpers(root)
    names = list(bindings["export_names"])
    if set(names) != set(bindings["export_specs"]) or len(names) != len(bindings["export_specs"]):
        raise ValueError("R7 export order and spec inventory differ")
    if bindings["r5_declared_export_names"] != driver.EXPORT_NAMES:
        raise ValueError(
            "unchanged R5 driver inventory differs from the original nine-spec manifest"
        )
    if driver.FINALIZATION_RESERVE_SECONDS != FINALIZATION_RESERVE_SECONDS:
        raise ValueError("R5 finalization reserve differs from the R7 runner policy")
    specs = []
    for name in names:
        entry = bindings["export_specs"][name]
        path = root / entry["spec_path"]
        sha_member(path, entry["size_bytes"], entry["spec_sha256"])
        spec = json.loads(path.read_text(encoding="utf-8"))
        argv = spec["argv"]
        if (
            spec.get("name") != name
            or argv[3] != entry["run_name"]
            or argv[4] != entry["split"]
            or argv[argv.index("--epoch") + 1] != "2"
            or argv[argv.index("--device") + 1] != "cuda"
            or argv[argv.index("--inference-precision") + 1] != "fp32"
            or spec.get("limits")
            != {
                "cpus": 2,
                "gpu_memory_bytes": 5_000_000_000,
                "gpu_stop_bytes": 6_000_000_000,
                "ram_bytes": 8_000_000_000,
                "timeout_seconds": 1_800,
            }
            or spec.get("artifacts") != entry["artifacts"]
        ):
            raise ValueError(f"R5 export command/limits changed: {name}")
        full_valid = "--campaign-full-valid-feature-only" in argv
        if full_valid != (entry["campaign_fit_scope"] == "full-valid-feature-only"):
            raise ValueError(f"R5 campaign fit scope changed: {name}")
        specs.append(spec)
    return driver, helpers, specs


def remaining_npy_bytes(output_root: Path, expected_total: int) -> int:
    written = sum(p.stat().st_size for p in output_root.rglob("pred_*.npy") if p.is_file())
    if written > expected_total:
        raise ValueError("R7 prediction bytes exceed the sealed six-array total")
    return expected_total - written


def regular_file_bytes(root: Path) -> int:
    return sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink()
    )


def check_disk_reserve(root: Path, output_root: Path, bindings: dict[str, Any]) -> dict[str, int]:
    usage = shutil.disk_usage(root.parent)
    remaining = remaining_npy_bytes(output_root, bindings["expected_total_npy_bytes"])
    required = (
        remaining
        + MAX_SINGLE_NPY_COPY_BYTES
        + FREE_DISK_RESERVE_BYTES
        + NON_ARRAY_OUTPUT_RESERVE_BYTES
    )
    if usage.free < required:
        raise OSError(f"R7 disk gate refused: free={usage.free}, required={required}")
    return {
        "free_bytes": usage.free,
        "capacity_bytes": usage.total,
        "remaining_prediction_bytes": remaining,
        "max_single_npy_copy_bytes": MAX_SINGLE_NPY_COPY_BYTES,
        "free_reserve_bytes": FREE_DISK_RESERVE_BYTES,
        "non_array_output_reserve_bytes": NON_ARRAY_OUTPUT_RESERVE_BYTES,
        "required_free_bytes": required,
    }


def read_npy_header(path: Path) -> tuple[tuple[int, ...], str]:
    import numpy as np

    with path.open("rb") as source:
        version = np.lib.format.read_magic(source)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(source)
        elif version in ((2, 0), (3, 0)):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(source)
        else:
            raise ValueError(f"unsupported prediction NPY version {version}")
    if fortran_order or dtype != np.dtype("<f4"):
        raise ValueError("R5 prediction NPY must be C-order float32")
    return tuple(int(item) for item in shape), dtype.str


def finite_npy_sha256(path: Path, expected_shape: tuple[int, ...]) -> str:
    """Stream a sealed prediction copy once for both finiteness and digest validation."""
    import numpy as np

    with path.open("rb") as source:
        version = np.lib.format.read_magic(source)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(source)
        elif version in ((2, 0), (3, 0)):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(source)
        else:
            raise ValueError(f"unsupported prediction NPY version {version}")
        if tuple(shape) != expected_shape or fortran_order or dtype != np.dtype("<f4"):
            raise ValueError("prediction copy has an unexpected NPY shape/order/dtype")
        data_offset = source.tell()
        source.seek(0)
        digest = hashlib.sha256()
        offset = 0
        data_bytes = 0
        while block := source.read(4 << 20):
            digest.update(block)
            start = max(0, data_offset - offset)
            payload = block[start:]
            if len(payload) % dtype.itemsize:
                raise ValueError("prediction NPY payload is not float32 aligned")
            if payload:
                values = np.frombuffer(payload, dtype=dtype)
                if not np.isfinite(values).all():
                    raise ValueError("prediction NPY contains non-finite values")
                data_bytes += len(payload)
            offset += len(block)
    expected_data_bytes = int(np.prod(expected_shape, dtype=np.int64)) * dtype.itemsize
    if data_bytes != expected_data_bytes:
        raise ValueError("prediction NPY payload byte count differs from its declared shape")
    return digest.hexdigest()


def validate_one_export(
    root: Path, stage_output: Path, spec: dict[str, Any], entry: dict[str, Any]
) -> dict[str, Any]:
    worker_dir = stage_output / spec["name"]
    result = read_json(worker_dir / "result.json")
    artifacts = result.get("artifacts", {})
    if (
        result.get("status") != "complete"
        or result.get("returncode") != 0
        or artifacts.get("missing")
    ):
        raise RuntimeError(f"R5 export worker incomplete: {spec['name']}")
    rows = {row.get("path"): row for row in artifacts.get("files", [])}
    if set(rows) != set(entry["artifacts"]):
        raise RuntimeError(f"worker artifact set differs from R5 spec: {spec['name']}")
    npy_name = next(path for path in entry["artifacts"] if path.endswith(".npy"))
    sidecar_name = next(path for path in entry["artifacts"] if path.endswith(".json"))
    original_npy, original_json = root / npy_name, root / sidecar_name
    if (
        not original_npy.is_file()
        or original_npy.is_symlink()
        or not original_json.is_file()
        or original_json.is_symlink()
        or original_npy.with_name(original_npy.name + ".partial").exists()
    ):
        raise RuntimeError(f"R5 output NPY/sidecar is incomplete: {spec['name']}")
    copied_npy, copied_json = (
        worker_dir / "artifacts" / npy_name,
        worker_dir / "artifacts" / sidecar_name,
    )
    npy_row, json_row = rows[npy_name], rows[sidecar_name]
    if (
        not copied_npy.is_file()
        or not copied_json.is_file()
        or copied_npy.stat().st_size != original_npy.stat().st_size
        or copied_json.stat().st_size != original_json.stat().st_size
        or npy_row.get("size") != original_npy.stat().st_size
        or json_row.get("size") != original_json.stat().st_size
        or sha256_file(original_json) != json_row.get("sha256")
        or sha256_file(copied_json) != json_row.get("sha256")
    ):
        raise RuntimeError(f"runner artifact copy differs from original: {spec['name']}")
    metadata = read_json(original_json)
    if (
        metadata.get("run") != entry["run_name"]
        or metadata.get("split") != entry["split"]
        or metadata.get("epoch") != 2
        or metadata.get("checkpoint_sha256") != entry["checkpoint_sha256"]
        or metadata.get("campaign_split_manifest_sha256") != entry["campaign_split_manifest_sha256"]
        or metadata.get("campaign_fit_scope") != entry["campaign_fit_scope"]
        or metadata.get("inference_precision") != "fp32"
        or not isinstance(metadata.get("row_groups"), list)
    ):
        raise ValueError(f"R5 sidecar checkpoint/split identity mismatch: {spec['name']}")
    groups = metadata["row_groups"]
    if (
        not groups
        or any(type(group) is not int or group < 0 for group in groups)
        or groups != sorted(set(groups))
    ):
        raise ValueError(
            f"R5 sidecar row groups are empty, duplicated, or unordered: {spec['name']}"
        )
    shape, dtype = read_npy_header(copied_npy)
    if (
        shape != (entry["expected_row_groups_count"], 20_000, 2)
        or len(groups) != entry["expected_row_groups_count"]
    ):
        raise ValueError(f"R5 prediction shape differs from sidecar: {spec['name']}")
    import numpy as np

    group_hash = hashlib.sha256(np.asarray(groups, dtype="<i8").tobytes()).hexdigest()
    if group_hash != entry["expected_ordered_row_groups_sha256"]:
        raise ValueError(
            f"R5 sidecar row-group order differs from source-qualified fit membership: {spec['name']}"
        )
    prediction_sha = metadata.get("prediction_sha256")
    if (
        not isinstance(prediction_sha, str)
        or len(prediction_sha) != 64
        or npy_row.get("sha256") != prediction_sha
    ):
        raise ValueError(f"runner copy SHA differs from predictor sidecar: {spec['name']}")
    copied_sha = finite_npy_sha256(copied_npy, shape)
    if copied_sha != prediction_sha:
        raise ValueError(
            f"delivered prediction copy SHA differs from sidecar/runner receipts: {spec['name']}"
        )
    if npy_row.get("size") > MAX_SINGLE_NPY_COPY_BYTES:
        raise OSError(f"single NPY exceeds transient-copy bound: {spec['name']}")
    return {
        "name": spec["name"],
        "run_name": entry["run_name"],
        "split": entry["split"],
        "campaign_fit_scope": entry["campaign_fit_scope"],
        "npy_path": (worker_dir / "artifacts" / npy_name).as_posix(),
        "sidecar_path": (worker_dir / "artifacts" / sidecar_name).as_posix(),
        "npy_bytes": int(npy_row["size"]),
        "npy_sha256": prediction_sha,
        "sidecar_sha256": json_row["sha256"],
        "shape": list(shape),
        "dtype": dtype,
        "row_groups": len(groups),
    }


def remove_verified_transient_originals(root: Path, artifacts: list[str]) -> None:
    for relative in artifacts:
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.resolve().parent != (root / "cache").resolve()
        ):
            raise RuntimeError(f"refusing to remove unsafe transient output: {relative}")
        path.unlink()


def main() -> None:
    kernel_dir = Path(__file__).resolve().parent
    bindings = load_bindings(kernel_dir)
    require_root_release(kernel_dir, bindings)
    started = time.monotonic()
    deadline = started + KERNEL_PROCESS_WALL_SECONDS_MAX
    input_root = Path("/kaggle/input")
    bundle_manifests = list(input_root.glob(f"**/{BUNDLE_SLUG}/bundle-manifest.json"))
    if len(bundle_manifests) != 1:
        raise FileNotFoundError(f"expected one R5 bundle mount; found {len(bundle_manifests)}")
    manifest_path = bundle_manifests[0]
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha != bindings["r5_bundle_manifest_sha256"]:
        raise ValueError("mounted R5 manifest differs from the reviewed source")
    manifest = json.loads(manifest_bytes)
    archive = manifest_path.parent / "bundle.wnn"
    root = Path("/kaggle/working/experiment")
    output_root = Path("/kaggle/working/results/r7_export_recovery")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("R7 refuses to reuse an existing output directory")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=False, exist_ok=False)
    pipeline_path = output_root / "r7_export_pipeline.json"
    pipeline: dict[str, Any] = {
        "schema": "wnn.kaggle.clean_t3_f0_export_only_pipeline.r7.r1",
        "status": "running",
        "fit_workers_requested": 0,
        "export_workers_requested": len(bindings["export_specs"]),
        "maximum_concurrent_export_workers": 1,
        "r5_archive_sha256": bindings["r5_bundle_archive_sha256"],
        "r6_kernel_id": bindings["r6_kernel_id"],
        "r6_kernel_version": bindings["r6_kernel_version"],
        "local_full_valid_counterparts": bindings["local_full_valid_counterparts"],
        "exports": [],
    }
    write_json(pipeline_path, pipeline)
    mounts: list[Path] = []
    try:
        initial_disk = shutil.disk_usage(root.parent)
        initial_required = (
            int(manifest["expanded_bytes"])
            + int(bindings["expected_total_npy_bytes"])
            + MAX_SINGLE_NPY_COPY_BYTES
            + FREE_DISK_RESERVE_BYTES
            + NON_ARRAY_OUTPUT_RESERVE_BYTES
        )
        if initial_disk.free < initial_required:
            raise OSError(
                f"R7 initial disk gate refused: free={initial_disk.free}, required={initial_required}"
            )
        checked_extract_bundle(
            archive,
            manifest,
            manifest_sha,
            bindings["r5_bundle_archive_sha256"],
            root,
            bindings,
            deadline,
        )
        source_root = find_source_mount(input_root)
        mounts.extend(mount_verified_checkpoints(root, source_root, bindings["checkpoints"]))
        declared = {row["path"] for row in manifest["files"]} | set(manifest["spec_paths"])
        mounts.extend(
            mount_verified_inputs(input_root, root, manifest["external_mounts"], declared)
        )
        for name in bindings["export_names"]:
            entry = bindings["export_specs"][name]
            for artifact in entry["artifacts"]:
                target = root / artifact
                if (
                    target.exists()
                    or target.is_symlink()
                    or target.with_name(target.name + ".partial").exists()
                ):
                    raise FileExistsError(f"R7 prediction destination is not empty: {artifact}")
        driver, helpers, specs = verify_export_specs(root, bindings)
        pipeline["initial_disk_gate"] = {
            "available_bytes": initial_disk.free,
            "required_bytes": initial_required,
            "free_reserve_bytes": FREE_DISK_RESERVE_BYTES,
            "expected_npy_bytes": bindings["expected_total_npy_bytes"],
            "transient_copy_bound_bytes": MAX_SINGLE_NPY_COPY_BYTES,
        }
        pipeline["checkpoint_mounts_verified"] = sorted(bindings["checkpoints"])
        write_json(pipeline_path, pipeline)
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        os.environ["WNN_KERNEL_DEADLINE_MONOTONIC"] = repr(deadline)

        for index, name in enumerate(bindings["export_names"], start=1):
            spec = specs[index - 1]
            entry = bindings["export_specs"][name]
            if time.monotonic() >= deadline - FINALIZATION_RESERVE_SECONDS:
                raise TimeoutError("R7 deadline leaves no bounded export/finalization time")
            disk_gate = check_disk_reserve(root, output_root, bindings)
            stage = f"export_{index:02d}_{name}"
            record: dict[str, Any] = {
                "name": name,
                "stage": stage,
                "status": "running",
                "disk_gate_before_worker": disk_gate,
                "fresh_host_gate": "unchanged R5 runner performs host/device inventory and admit_specs for this one-spec invocation",
            }
            pipeline["exports"].append(record)
            write_json(pipeline_path, pipeline)
            stage_output = output_root / stage
            try:
                stage_result = driver.run_stage(root, output_root, stage, [name], deadline, helpers)
                worker_results = driver.validate_stage(root, stage_output, [name])
                worker_result = worker_results[name]
                inventory = read_json(stage_output / "inventory.json")
                host = inventory.get("host", {})
                capacity, available = host.get("capacity_bytes"), host.get("available_bytes")
                if type(capacity) is not int or type(available) is not int:
                    raise RuntimeError(f"runner inventory lacks host memory values: {name}")
                budget = min(int(capacity * 0.8), available - int(capacity * 0.2))
                if budget < 8_000_000_000:
                    raise RuntimeError(f"worker ran without the unchanged 8GB reservation: {name}")
                summary = read_json(stage_output / "summary.json")
                finished = {row.get("name") for row in summary.get("finished", [])}
                if (
                    summary.get("incident") is not None
                    or summary.get("not_started")
                    or finished != {name}
                ):
                    raise RuntimeError(f"single-export runner stage is incomplete: {name}")
                artifact_record = validate_one_export(root, stage_output, spec, entry)
                remove_verified_transient_originals(root, entry["artifacts"])
                record.update(
                    {
                        "status": "complete",
                        "runner_returncode": stage_result["returncode"],
                        "runner_elapsed_seconds": stage_result["elapsed_seconds"],
                        "host_inventory": host,
                        "host_operational_budget_bytes": budget,
                        "device_inventory": inventory.get("devices"),
                        "worker_result": worker_result,
                        "validated_artifact": artifact_record,
                    }
                )
                write_json(pipeline_path, pipeline)
            except BaseException as error:
                record.update(
                    {
                        "status": "failed_or_blocked",
                        "error_type": type(error).__name__,
                        "error": str(error)[:1000],
                    }
                )
                if stage_output.exists() and (stage_output / "inventory.json").is_file():
                    record["host_inventory"] = read_json(stage_output / "inventory.json")
                write_json(pipeline_path, pipeline)
                raise

        copied_npy_bytes = sum(
            int(row["validated_artifact"]["npy_bytes"])
            for row in pipeline["exports"]
            if row.get("status") == "complete"
        )
        if copied_npy_bytes != bindings["expected_total_npy_bytes"] or len(
            pipeline["exports"]
        ) != len(bindings["export_names"]):
            raise ValueError(
                f"R7 export closure mismatch: n={len(pipeline['exports'])}, bytes={copied_npy_bytes}"
            )
        if any(row.get("status") != "complete" for row in pipeline["exports"]):
            raise RuntimeError("R7 did not complete all six exports")
        if {row.get("name") for row in pipeline["exports"]} != set(bindings["export_names"]):
            raise RuntimeError("R7 completed export names differ from the six frozen specs")
        if remaining_npy_bytes(output_root, bindings["expected_total_npy_bytes"]) != 0:
            raise RuntimeError("R7 output tree does not contain the declared NPY payload")
        working_bytes = regular_file_bytes(Path("/kaggle/working"))
        if working_bytes > bindings["disk_and_collection"]["collector_cap_bytes"]:
            raise OSError("R7 working output exceeds the frozen collector cap")
        pipeline["status"] = "complete_export_only"
        pipeline["fit_workers_started"] = 0
        pipeline["completed_elapsed_seconds"] = time.monotonic() - started
        pipeline["working_regular_file_bytes"] = working_bytes
        pipeline["final_free_disk_bytes"] = shutil.disk_usage(root.parent).free
        write_json(pipeline_path, pipeline)
    except BaseException as error:
        pipeline["status"] = "failed_or_blocked"
        pipeline["error_type"] = type(error).__name__
        pipeline["error"] = str(error)[:1000]
        write_json(pipeline_path, pipeline)
        raise
    finally:
        # The R5 cache/artifact originals are generated outputs; source and raw inputs are read-only.
        for mount in reversed(mounts):
            if mount.is_symlink():
                mount.unlink()


if __name__ == "__main__":
    main()
