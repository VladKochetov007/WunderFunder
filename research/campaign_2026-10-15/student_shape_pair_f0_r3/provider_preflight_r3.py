from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[3]
MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
PACKET = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r3"
PACKET_R2 = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r2"
BUNDLE_R2 = ROOT / "private_assets/upload_wnn-f0-student-shape-t4-profile-20261015-r2"
OWNER = "vladkochetov07"
SOURCE_DATASET = f"{OWNER}/wnn-f0-student-shape-t4-profile-20261015-r1"
NORM_DATASET = f"{OWNER}/wnn-f0-h256l2-cp5-norm-20261015-r1"
TEACHER_DATASET = f"{OWNER}/wnn-clean-t3-f0-h384-cp2-20261012-r9"
KERNEL_ID = f"{SOURCE_DATASET}-runner"
R9_MANIFEST = MAIN / (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/clean_t3_f0_h390_r9_export_from_owned_checkpoints_20261012_r2/"
    "R9_CHECKPOINT_INPUT_MANIFEST_R2.json"
)
NORM_READBACK = PACKET_R2 / "NORM_ASSET_REMOTE_READBACK_R1.json"
NORM_READBACK_FILES = ROOT / "private_assets/remote_verify_norm_r1"
EXACT_SEARCH_PAGE_SIZE = 100
RECENT_KERNEL_PAGE_SIZE = 20
MAX_FILE_PAGES = 10
FILE_PAGE_SIZE = 100
MIN_QUOTA_SECONDS = 3_600
EXIT_STATUSES = {
    "CANCELED",
    "CANCELLED",
    "COMPLETE",
    "ERROR",
    "NO_SESSION",
    "NOSESSION",
}

# JSON documents contain heterogeneous values defined by the frozen packet schemas.


def sha256_file(path: Path) -> str:
    """Hash a local asset in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, *, max_bytes: int = 4_000_000) -> dict[str, Any]:
    """Read one bounded packet JSON file."""
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"refusing to read oversized JSON metadata: {path} ({size} bytes)"
        )
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def canonical_sha256(value: object) -> str:
    """Hash a stable JSON encoding of a bounded owner inventory."""
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(data).hexdigest()


def typed_seconds(value: object) -> float | None:
    """Convert the typed SDK duration fields to seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    total_seconds = getattr(value, "total_seconds", None)
    if callable(total_seconds):
        return float(total_seconds())
    return None


def quota_lane(quota: object) -> dict[str, float | None]:
    """Summarize a provider quota lane without using rounded CLI text."""
    total = typed_seconds(getattr(quota, "total_time_allowed", None))
    used = typed_seconds(getattr(quota, "time_used", None))
    reserved = typed_seconds(getattr(quota, "time_reserved", None))
    remaining = (
        total - used - reserved if total is not None and used is not None else None
    )
    return {
        "total_time_allowed_seconds": total,
        "time_used_seconds": used,
        "time_reserved_seconds": reserved,
        "remaining_seconds": remaining,
    }


def exact_dataset(api: KaggleApi, ref: str) -> object | None:
    """Find one exact owner dataset ref through a filtered provider query."""
    owner, slug = ref.split("/", maxsplit=1)
    response = api.dataset_list_with_response(
        user=owner,
        search=slug,
        sort_by="published",
        page=1,
        page_size=EXACT_SEARCH_PAGE_SIZE,
    )
    matches = [
        model
        for model in (getattr(response, "datasets", None) or [])
        if dataset_ref(model) == ref
    ]
    if len(matches) > 1:
        raise ValueError(f"exact dataset query returned duplicate rows for {ref}")
    return matches[0] if matches else None


def exact_kernel(api: KaggleApi, ref: str) -> object | None:
    """Find one exact owner kernel ref through a filtered provider query."""
    owner, slug = ref.split("/", maxsplit=1)
    response = api.kernels_list_with_response(
        user=owner,
        search=slug,
        sort_by="dateRun",
        page=1,
        page_size=EXACT_SEARCH_PAGE_SIZE,
    )
    matches = [
        model
        for model in (getattr(response, "kernels", None) or [])
        if kernel_ref(model) == ref
    ]
    if len(matches) > 1:
        raise ValueError(f"exact kernel query returned duplicate rows for {ref}")
    return matches[0] if matches else None


def recent_kernel_rows(api: KaggleApi) -> list[object]:
    """Read a single latest-run page for a bounded overlap check."""
    response = api.kernels_list_with_response(
        user=OWNER,
        sort_by="dateRun",
        page=1,
        page_size=RECENT_KERNEL_PAGE_SIZE,
    )
    return list(getattr(response, "kernels", None) or [])


def dataset_ref(model: object) -> str:
    """Return a dataset's canonical owner/slug identifier."""
    value = getattr(model, "ref", None)
    if isinstance(value, str) and "/" in value:
        return value
    owner = getattr(model, "owner", None) or getattr(model, "author", None) or OWNER
    slug = getattr(model, "slug", None)
    if isinstance(slug, str) and slug:
        return f"{owner}/{slug}"
    raise ValueError("owner dataset row has no canonical ref")


def kernel_ref(model: object) -> str | None:
    """Return a canonical kernel ref, or None for opaque private placeholders."""
    value = getattr(model, "ref", None) or getattr(model, "_ref", None)
    if isinstance(value, str) and "/" in value and value.rsplit("/", maxsplit=1)[-1]:
        return value
    owner = getattr(model, "author", None) or getattr(model, "_author", None)
    slug = getattr(model, "slug", None) or getattr(model, "_slug", None)
    if isinstance(slug, str) and slug and isinstance(owner, str) and "/" in owner:
        return f"{owner}/{slug}"
    return None


def dataset_inventory_row(model: object) -> dict[str, object]:
    """Extract the small identity/privacy/version fields needed for admission."""
    return {
        "ref": dataset_ref(model),
        "private": bool(getattr(model, "is_private", False)),
        "current_version": getattr(model, "current_version_number", None),
        "last_updated": str(getattr(model, "last_updated", "")),
    }


def kernel_inventory_row(model: object) -> dict[str, object]:
    """Extract only fields exposed by the provider's kernel-list response."""
    return {
        "ref": kernel_ref(model),
        "title": getattr(model, "title", None) or getattr(model, "_title", None),
        "last_run_time": str(
            getattr(model, "last_run_time", None)
            or getattr(model, "_last_run_time", None)
            or ""
        ),
        "private": getattr(model, "is_private", None)
        if getattr(model, "is_private", None) is not None
        else getattr(model, "_is_private", None),
        "enable_gpu": getattr(model, "enable_gpu", None)
        if getattr(model, "enable_gpu", None) is not None
        else getattr(model, "_enable_gpu", None),
    }


def kernel_status(api: KaggleApi, model: object) -> dict[str, str]:
    """Read one identifiable kernel session and normalize its status enum."""
    ref = kernel_ref(model)
    if ref is None:
        raise ValueError("cannot query an opaque owner kernel row")
    response = api.kernels_status(ref)
    value = getattr(response, "status", None)
    if value is None:
        value = getattr(response, "_status", None)
    if value is None:
        raise ValueError(f"kernel status response omitted a state: {ref}")
    status = str(value)
    return {"ref": ref, "status": status, "state": status.rsplit(".", 1)[-1].upper()}


def opaque_kernel_is_recent(model: object, *, now: datetime) -> bool:
    """Fail closed when an unaddressable placeholder could be a live session."""
    value = getattr(model, "last_run_time", None) or getattr(
        model, "_last_run_time", None
    )
    if value is None:
        return True
    if not isinstance(value, datetime):
        value = datetime.fromisoformat(str(value))
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return now - value.astimezone(UTC) <= timedelta(hours=24)


def remote_file_rows(api: KaggleApi, dataset: str) -> dict[str, int]:
    """Read bounded file metadata for one exact dataset version."""
    result: dict[str, int] = {}
    page_token: str | None = None
    for _ in range(MAX_FILE_PAGES):
        response = api.dataset_list_files(
            dataset, page_token=page_token, page_size=FILE_PAGE_SIZE
        )
        files = list(getattr(response, "files", None) or [])
        for item in files:
            name = getattr(item, "name", None) or getattr(item, "file_name", None)
            size = getattr(item, "total_bytes", None)
            if size is None:
                size = getattr(item, "totalBytes", None)
            if not isinstance(name, str) or not isinstance(size, (int, float)):
                raise TypeError(f"dataset file row is missing name/size for {dataset}")
            if name in result:
                raise ValueError(f"duplicate remote file row {dataset}/{name}")
            result[name] = int(size)
        page_token = getattr(response, "next_page_token", None) or None
        if not page_token:
            return result
    raise RuntimeError(f"file inventory for {dataset} exceeded {MAX_FILE_PAGES} pages")


def require_files(
    api: KaggleApi,
    dataset: str,
    version: int,
    expected: dict[str, int],
    *,
    exact: bool = False,
) -> dict[str, int]:
    """Require remote filenames and byte sizes to match pinned local metadata."""
    actual = remote_file_rows(api, f"{dataset}/{version}")
    missing_or_changed = {
        name: {"expected_bytes": size, "actual_bytes": actual.get(name)}
        for name, size in expected.items()
        if actual.get(name) != size
    }
    extra = sorted(set(actual) - set(expected)) if exact else []
    if missing_or_changed or extra:
        raise ValueError(
            f"remote file inventory mismatch for {dataset}/{version}: "
            f"changed={missing_or_changed}, extra={extra}"
        )
    return actual


def required_sources(input_binding: dict[str, Any]) -> dict[str, int]:
    """Return all previously published data/asset dataset version pins."""
    required: dict[str, int] = {}
    for item in input_binding["external_mounts"]:
        dataset = item["dataset"]
        version = int(item["version_expected"])
        previous = required.setdefault(dataset, version)
        if previous != version:
            raise ValueError(f"input binding mixes versions for dataset {dataset}")
    required.pop(NORM_DATASET, None)
    required.pop(SOURCE_DATASET, None)
    return required


def normalize_stage(stage: Path, dataset: str) -> dict[str, Any]:
    """Verify every content-addressed normalization payload before upload."""
    manifest_path = stage / "assets-manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("dataset") != dataset or manifest.get("private") is not True:
        raise ValueError(
            "normalization stage has the wrong owner/slug or is not private"
        )
    local_files: dict[str, int] = {}
    for row in manifest["files"]:
        path = stage / row["filename"]
        if path.stat().st_size != row["size"] or sha256_file(path) != row["sha256"]:
            raise ValueError(f"normalization payload mismatch: {row['filename']}")
        local_files[row["filename"]] = int(row["size"])
    metadata = read_json(stage / "dataset-metadata.json")
    if metadata.get("id") != dataset or metadata.get("isPrivate") is not True:
        raise ValueError(
            "normalization dataset metadata is not private or correctly named"
        )
    return {
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_metadata_sha256": sha256_file(stage / "dataset-metadata.json"),
        "files": local_files,
        "payload_bytes": sum(local_files.values()),
    }


def source_stage() -> dict[str, Any]:
    """Verify the sealed profile bundle, generated kernel and private metadata."""
    binding = read_json(PACKET / "SOURCE_BINDING_R3.json")
    seal = read_json(PACKET / "GENERATED_KERNEL_SEAL_R3_SEALED.json")
    preflight = read_json(PACKET / "OFFLINE_SPEC_PREFLIGHT_R3_SEALED.json")
    upload = ROOT / "private_assets/upload_wnn-f0-student-shape-t4-profile-20261015-r3"
    archive = upload / "bundle.wnn"
    manifest_path = upload / "bundle-manifest.json"
    kernel_path = upload / "kernel/kernel.py"
    kernel_metadata_path = upload / "kernel/kernel-metadata.json"
    expected = seal["bundle"]
    if sha256_file(archive) != expected["archive_sha256"]:
        raise ValueError("source archive changed since the R3 kernel seal")
    if sha256_file(manifest_path) != expected["manifest_sha256"]:
        raise ValueError("source manifest changed since the R3 kernel seal")
    if sha256_file(kernel_path) != seal["kernel"]["patched_sha256"]:
        raise ValueError("generated kernel changed since the R3 seal")
    if sha256_file(kernel_metadata_path) != seal["kernel"]["kernel_metadata_sha256"]:
        raise ValueError("kernel metadata changed since the R3 seal")
    if preflight["status"] != "passed_offline_no_provider_action":
        raise ValueError("R3 offline spec preflight did not pass")
    if (
        preflight["source_binding_sha256"]
        != sha256_file(PACKET / "SOURCE_BINDING_R3.json")
        or seal["source_binding_sha256"] != preflight["source_binding_sha256"]
    ):
        raise ValueError("R3 source binding changed after the sealed offline preflight")
    if (
        preflight["input_binding_sha256"]
        != sha256_file(PACKET / "INPUT_BINDING_R3.json")
        or seal["input_binding_sha256"] != preflight["input_binding_sha256"]
    ):
        raise ValueError("R3 input binding changed after the sealed offline preflight")
    if preflight["kernel_seal_sha256"] != sha256_file(
        PACKET / "GENERATED_KERNEL_SEAL_R3_SEALED.json"
    ):
        raise ValueError("R3 kernel seal changed after the offline preflight")
    build_receipt = read_json(PACKET / "OFFLINE_BUNDLE_BUILD_R3.json")
    mount_receipt = read_json(PACKET / "KERNEL_MOUNT_SUCCESSOR_R3.json")
    if binding["bundle_successor_receipt_sha256"] != sha256_file(
        PACKET / "OFFLINE_BUNDLE_BUILD_R3.json"
    ) or binding["kernel_mount_successor_receipt_sha256"] != sha256_file(
        PACKET / "KERNEL_MOUNT_SUCCESSOR_R3.json"
    ):
        raise ValueError("R3 transport successor receipts changed after review")
    if (
        build_receipt["archive_byte_identical_to_r2"] is not True
        or build_receipt["r3_source_archive_sha256"]
        != sha256_file(BUNDLE_R2 / "bundle.wnn")
        or mount_receipt["r3_kernel_sha256"] != sha256_file(kernel_path)
    ):
        raise ValueError("R3 outer bundle successor no longer matches its receipts")
    if binding["kernel_dataset"] != SOURCE_DATASET or binding["kernel_id"] != KERNEL_ID:
        raise ValueError("source binding has the wrong title-derived provider target")
    return {
        "source_binding_sha256": sha256_file(PACKET / "SOURCE_BINDING_R3.json"),
        "input_binding_sha256": sha256_file(PACKET / "INPUT_BINDING_R3.json"),
        "kernel_seal_sha256": sha256_file(
            PACKET / "GENERATED_KERNEL_SEAL_R3_SEALED.json"
        ),
        "offline_preflight_sha256": sha256_file(
            PACKET / "OFFLINE_SPEC_PREFLIGHT_R3_SEALED.json"
        ),
        "archive_sha256": sha256_file(archive),
        "archive_bytes": archive.stat().st_size,
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "kernel_sha256": sha256_file(kernel_path),
        "kernel_metadata_sha256": sha256_file(kernel_metadata_path),
        "dataset_files": {
            "bundle.wnn": archive.stat().st_size,
            "bundle-manifest.json": manifest_path.stat().st_size,
        },
    }


def normalization_remote_files(
    input_binding: dict[str, Any], source_binding: dict[str, Any]
) -> dict[str, int]:
    """Bind the already-created asset's actual remote filenames to verified bytes."""
    readback = read_json(NORM_READBACK)
    correction = source_binding["normalization_mount_correction"]
    if sha256_file(NORM_READBACK) != correction["remote_readback_sha256"]:
        raise ValueError("normalizer remote readback changed after the R3 mount seal")
    if (
        readback["dataset"]["ref"] != NORM_DATASET
        or readback["dataset"]["version"] != 1
    ):
        raise ValueError("normalizer readback identifies a different owner/version")
    if (
        readback["dataset"]["private"] is not True
        or readback["dataset"]["status"] != "ready"
    ):
        raise ValueError("normalizer readback is not private READY version 1")

    rows = [
        row
        for row in input_binding["external_mounts"]
        if row["dataset"] == NORM_DATASET
    ]
    payloads = {row["sha256"]: row for row in readback["approved_payloads"]}
    if len(rows) != 2 or len(payloads) != 2:
        raise ValueError("normalizer binding/readback must contain two payloads")
    expected_files = dict(readback["remote_file_inventory"])
    for row in rows:
        verified = payloads.get(row["sha256"])
        if verified is None or (row["filename"], row["size"]) != (
            verified["remote_filename"],
            verified["size"],
        ):
            raise ValueError("R3 normalizer mount differs from verified remote payload")
    if set(expected_files) != {
        "assets-manifest.json",
        *(row["filename"] for row in rows),
    }:
        raise ValueError("normalizer remote file inventory differs from R3 mounts")
    manifest_path = NORM_READBACK_FILES / "assets-manifest.json"
    remote_manifest = read_json(manifest_path)
    manifest_rows = {
        row["filename"]: (row["sha256"], int(row["size"]))
        for row in remote_manifest["files"]
    }
    if (
        remote_manifest.get("dataset") != NORM_DATASET
        or remote_manifest.get("private") is not True
    ):
        raise ValueError("normalizer asset manifest has the wrong identity or privacy")
    expected_manifest_rows = {
        row["remote_filename"]: (row["sha256"], int(row["size"]))
        for row in readback["approved_payloads"]
    }
    if manifest_rows != expected_manifest_rows:
        raise ValueError(
            "normalizer asset manifest payload hashes differ from R3 binding"
        )
    for filename, size in expected_files.items():
        path = NORM_READBACK_FILES / filename
        if path.stat().st_size != size:
            raise ValueError(f"normalizer readback size mismatch for {filename}")
        digest = sha256_file(path)
        expected_hash = (
            readback["remote_manifest"]["sha256"]
            if filename == "assets-manifest.json"
            else next(
                row["sha256"]
                for row in readback["approved_payloads"]
                if row["remote_filename"] == filename
            )
        )
        if digest != expected_hash:
            raise ValueError(f"normalizer readback hash mismatch for {filename}")
    return {str(name): int(size) for name, size in expected_files.items()}


def main() -> None:
    """Run a read-only, phase-specific gate before one authorized mutation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("norm-create", "source-create", "kernel-push"),
        required=True,
    )
    args = parser.parse_args()
    phase = args.phase
    started = datetime.now(UTC)
    api = KaggleApi()
    api.authenticate()

    input_binding = read_json(PACKET / "INPUT_BINDING_R3.json")
    source_binding = read_json(PACKET / "SOURCE_BINDING_R3.json")
    norm_stage_path = ROOT / "private_assets/upload_wnn-f0-h256l2-cp5-norm-20261015-r1"
    norm_local = normalize_stage(norm_stage_path, NORM_DATASET)
    source_local = source_stage()
    norm_remote = normalization_remote_files(input_binding, source_binding)
    required = required_sources(input_binding)
    checked_dataset_refs = sorted({*required, NORM_DATASET, SOURCE_DATASET})
    dataset_map = {ref: exact_dataset(api, ref) for ref in checked_dataset_refs}
    dataset_rows = [
        dataset_inventory_row(model)
        for model in dataset_map.values()
        if model is not None
    ]
    source_statuses: dict[str, str] = {}
    source_rows: list[dict[str, object]] = []
    for ref, version in required.items():
        model = dataset_map.get(ref)
        if model is None:
            raise ValueError(f"required mounted source dataset is absent: {ref}")
        row = dataset_inventory_row(model)
        if row["private"] is not True or row["current_version"] != version:
            raise ValueError(f"mounted source dataset privacy/version mismatch: {ref}")
        status = str(api.dataset_status(ref))
        source_statuses[ref] = status
        if status.upper().rsplit(".", maxsplit=1)[-1] != "READY":
            raise ValueError(f"mounted source dataset is not READY: {ref} ({status})")
        source_rows.append(row)

    manifest = read_json(R9_MANIFEST)
    if manifest.get("dataset_id") != TEACHER_DATASET:
        raise ValueError("local R9 manifest names a different dataset")
    r9_expected = {row["path"]: int(row["bytes"]) for row in manifest["files"]}
    r9_expected[R9_MANIFEST.name] = R9_MANIFEST.stat().st_size
    r9_live = require_files(api, TEACHER_DATASET, 1, r9_expected, exact=True)
    r9_local_hashes = {row["path"]: row["sha256"] for row in manifest["files"]}
    r9_local_hashes[R9_MANIFEST.name] = sha256_file(R9_MANIFEST)
    r9_mounts = [
        {
            "dataset": TEACHER_DATASET,
            "version": 1,
            "filename": row["filename"],
            "logical_path": row["logical_path"],
            "sha256": row["sha256"],
            "size": row["size"],
            "live_size": r9_live[row["filename"]],
        }
        for row in source_binding["r9_mount_correction"]["mounts"]
    ]
    if len(r9_mounts) != 6 or any(
        item["filename"] not in r9_local_hashes
        or item["sha256"] != r9_local_hashes[item["filename"]]
        or item["size"] != r9_live.get(item["filename"])
        for item in r9_mounts
    ):
        raise ValueError(
            "live R9 dataset files differ from the six bound CP2/provenance mounts"
        )

    # Check every other externally mounted filename and byte count without opening payloads.
    mounted_files: dict[str, dict[str, int]] = {}
    for item in input_binding["external_mounts"]:
        dataset = item["dataset"]
        if dataset in {NORM_DATASET, SOURCE_DATASET}:
            continue
        mounted_files.setdefault(dataset, {})[item["filename"]] = int(item["size"])
    for ref, expected_files in mounted_files.items():
        require_files(api, ref, required[ref], expected_files)

    target_states: dict[str, str] = {}
    for ref in (NORM_DATASET, SOURCE_DATASET):
        model = dataset_map.get(ref)
        target_states[ref] = "absent" if model is None else "present"

    if phase == "norm-create":
        if target_states != {NORM_DATASET: "absent", SOURCE_DATASET: "absent"}:
            raise ValueError(
                f"private create target slugs are not both absent: {target_states}"
            )
        next_action = "create only the private CP5 normalization dataset"
    elif phase == "source-create":
        norm_model = dataset_map.get(NORM_DATASET)
        source_model = dataset_map.get(SOURCE_DATASET)
        if norm_model is None or source_model is not None:
            raise ValueError(
                "source-create requires READY norm v1 and absent profile source slug"
            )
        norm_row = dataset_inventory_row(norm_model)
        if norm_row["private"] is not True or norm_row["current_version"] != 1:
            raise ValueError("normalization asset is not private version 1")
        norm_status = str(api.dataset_status(NORM_DATASET))
        if norm_status.upper().rsplit(".", maxsplit=1)[-1] != "READY":
            raise ValueError(f"normalization asset is not READY: {norm_status}")
        norm_live = require_files(api, NORM_DATASET, 1, norm_remote, exact=True)
        target_states[NORM_DATASET] = norm_status
        if norm_live != norm_remote:
            raise ValueError("live normalization file inventory differs from readback")
        next_action = "create only the private R3 profile source bundle dataset"
    elif phase == "kernel-push":
        for ref, expected_files in (
            (NORM_DATASET, norm_remote),
            (SOURCE_DATASET, source_local["dataset_files"]),
        ):
            model = dataset_map.get(ref)
            if model is None:
                raise ValueError(
                    f"kernel-push requires source asset READY version 1: {ref}"
                )
            row = dataset_inventory_row(model)
            if row["private"] is not True or row["current_version"] != 1:
                raise ValueError(f"required private asset is not version 1: {ref}")
            status = str(api.dataset_status(ref))
            target_states[ref] = status
            if status.upper().rsplit(".", maxsplit=1)[-1] != "READY":
                raise ValueError(
                    f"required private asset is not READY: {ref} ({status})"
                )
            require_files(api, ref, 1, expected_files, exact=ref == NORM_DATASET)
        next_action = "push exactly the frozen private T4 profile kernel and start the two serial profiles"

    target_kernel = exact_kernel(api, KERNEL_ID)
    if target_kernel is not None:
        raise ValueError(f"profile kernel slug already exists: {KERNEL_ID}")
    recent_kernels = recent_kernel_rows(api)
    recent_kernel_rows_out = [kernel_inventory_row(model) for model in recent_kernels]
    opaque_recent = [
        row
        for model, row in zip(recent_kernels, recent_kernel_rows_out, strict=True)
        if row["ref"] is None and opaque_kernel_is_recent(model, now=datetime.now(UTC))
    ]
    if opaque_recent:
        raise ValueError(
            "recent owner kernel row has no queryable ref or trustworthy terminal status"
        )
    recent_statuses = [
        kernel_status(api, model)
        for model, row in zip(recent_kernels, recent_kernel_rows_out, strict=True)
        if row["ref"] is not None
    ]
    active = [row for row in recent_statuses if row["state"] not in EXIT_STATUSES]
    if active:
        raise ValueError(
            f"recent owner kernels include a nonterminal session: {active}"
        )

    # Keep the quota read last so it is fresh at the one mutation covered here.
    quota_response = api.quota_view()
    quota_time = getattr(quota_response, "quota_refresh_time", None)
    gpu_quota = quota_lane(getattr(quota_response, "gpu_quota", None))
    tpu_quota = quota_lane(getattr(quota_response, "tpu_quota", None))
    remaining = gpu_quota["remaining_seconds"]
    if remaining is None or remaining < MIN_QUOTA_SECONDS:
        raise ValueError(
            f"typed GPU quota {remaining} seconds is below the {MIN_QUOTA_SECONDS}s admission floor"
        )
    if int(source_binding["bootstrap_outer_deadline_seconds"]) != 1_800:
        raise ValueError(
            "source binding is not pinned to the 1,800-second outer deadline"
        )

    ending = datetime.now(UTC)
    datasets_sorted = sorted(dataset_rows, key=lambda row: str(row["ref"]))
    kernels_sorted = sorted(
        recent_kernel_rows_out,
        key=lambda row: (str(row["ref"]), str(row["last_run_time"])),
    )
    receipt = {
        "schema": "wnn.student-shape-pair-provider-preflight.r3",
        "phase": phase,
        "status": f"pass_for_{phase.replace('-', '_')}_only",
        "provider_action_performed": False,
        "start_utc": started.isoformat(),
        "end_utc": ending.isoformat(),
        "sdk": {
            "package_version": importlib.metadata.version("kaggle"),
            "dataset_query": "exact owner+slug search, one filtered page per bound ref",
            "recent_kernel_query": "one dateRun-sorted owner page; status queried per canonical ref",
            "quota_method": "typed KaggleApi.quota_view; duration fields converted with total_seconds",
        },
        "typed_quota": {
            "refresh_time": quota_time.isoformat()
            if hasattr(quota_time, "isoformat")
            else str(quota_time),
            "gpu": gpu_quota,
            "tpu": tpu_quota,
            "minimum_required_gpu_seconds": MIN_QUOTA_SECONDS,
            "quota_passed": True,
        },
        "owner_inventory": {
            "dataset_check_count": len(checked_dataset_refs),
            "dataset_refs_checked": checked_dataset_refs,
            "exact_dataset_rows_sha256": canonical_sha256(datasets_sorted),
            "exact_dataset_rows_found": datasets_sorted,
            "recent_kernel_page_size": RECENT_KERNEL_PAGE_SIZE,
            "recent_kernel_rows_sha256": canonical_sha256(kernels_sorted),
            "recent_kernel_rows": kernels_sorted,
            "recent_kernel_statuses": recent_statuses,
            "nonterminal_recent_kernels": active,
            "gpu_accelerator_flag_available_in_list_response": any(
                row["enable_gpu"] is not None for row in kernels_sorted
            ),
        },
        "required_sources": {
            "rows": source_rows,
            "statuses": source_statuses,
            "remote_mount_files_checked_by_name_and_size": mounted_files,
        },
        "r9_live_readiness": {
            "dataset": TEACHER_DATASET,
            "version": 1,
            "status": source_statuses[TEACHER_DATASET],
            "manifest_path": str(R9_MANIFEST),
            "manifest_sha256": sha256_file(R9_MANIFEST),
            "live_files": r9_live,
            "bound_mounts": r9_mounts,
        },
        "private_targets": {
            "normalization_dataset": {
                "slug": NORM_DATASET,
                "state": target_states[NORM_DATASET],
            },
            "profile_source_dataset": {
                "slug": SOURCE_DATASET,
                "state": target_states[SOURCE_DATASET],
            },
            "kernel_id": KERNEL_ID,
            "kernel_absent": True,
        },
        "local_artifacts": {
            "norm_stage": norm_local,
            "source_bundle": source_local,
            "source_binding_sha256": source_local["source_binding_sha256"],
            "input_binding_sha256": source_local["input_binding_sha256"],
            "kernel_seal_sha256": source_local["kernel_seal_sha256"],
        },
        "scope": {
            "next_action": next_action,
            "only_that_single_provider_mutation_is_authorized_by_this_receipt": True,
            "profile_only": True,
            "quality_fit_or_scoring": False,
            "kernel_deadline_seconds": 1_800,
            "quota_reserve_after_deadline_seconds": 1_800,
            "two_serial_profile_shapes": [[256, 2], [512, 1]],
            "no_fit_or_quality_claim": True,
        },
    }
    output = PACKET / f"PROVIDER_PREFLIGHT_{phase.upper().replace('-', '_')}_R3.json"
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite provider preflight receipt {output}"
        )
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "phase": phase,
                "quota_remaining_seconds": remaining,
                "quota_refresh_time": receipt["typed_quota"]["refresh_time"],
                "r9_file_count": len(r9_live),
                "recent_kernel_status_count": len(recent_statuses),
                "receipt_path": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
