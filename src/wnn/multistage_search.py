from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path


def canonical_json_sha256(value: object) -> str:
    """Hash JSON-compatible data using the sequence runtime's canonical encoding."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    """Load one JSON object and reject non-object payloads."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def atomic_write_json(path: Path, value: Mapping[str, object], *, replace: bool = True) -> None:
    """Durably publish JSON through a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not replace):
        raise FileExistsError(f"refusing to replace existing JSON artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def file_lock(path: Path, *, blocking: bool = True) -> Iterator[None]:
    """Lock a small coordination file across independent controller processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
            msvcrt.locking(stream.fileno(), mode, 1)
        else:
            import fcntl

            flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            fcntl.flock(stream.fileno(), flags)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def append_event(path: Path, event: Mapping[str, object]) -> None:
    """Append and fsync one event while holding the shared journal lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    with file_lock(path.with_suffix(path.suffix + ".lock")):
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            offset = 0
            while offset < len(encoded):
                offset += os.write(descriptor, encoded[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def verify_file_ref(ref: object, label: str) -> tuple[Path, str]:
    """Resolve repo-relative `{path, sha256}` references and verify their bytes."""
    if not isinstance(ref, Mapping):
        raise ValueError(f"{label} must include path and sha256")
    raw_path, expected = ref.get("path"), ref.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} path or SHA256 is malformed")
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    path = path.resolve(strict=True)
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch")
    return path, actual


def validate_contract(contract: Mapping[str, object]) -> dict[str, int]:
    """Extract and bound the immutable fit, time, reserve, and profile limits."""
    allocation = contract.get("allocation")
    if not isinstance(allocation, Mapping):
        raise ValueError("contract is missing its allocation")
    fits = allocation.get("max_scientific_fit_attempts_including_auxiliary_TL3_and_final_refits")
    wall = allocation.get("max_node_wall_seconds_from_first_gpu_profile")
    fit_wall = allocation.get("max_individual_fit_wall_seconds")
    reserve = allocation.get("final_collection_reserve_seconds")
    profile_wall = allocation.get("profile_wall_seconds_per_attempt")
    initial_profiles = allocation.get("max_initial_profile_attempts")
    values = (fits, wall, fit_wall, reserve, profile_wall, initial_profiles)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("contract allocation limits must be positive integers")
    if reserve >= wall or fit_wall > wall or profile_wall > 180:
        raise ValueError("contract wall bounds are inconsistent")
    total_profiles = allocation.get("max_total_profile_attempts", initial_profiles + fits)
    if (
        type(total_profiles) is not int
        or not initial_profiles <= total_profiles <= initial_profiles + fits
    ):
        raise ValueError("contract total profile cap is inconsistent with fit/profile limits")
    return {
        "max_fit_attempts": fits,
        "campaign_wall_seconds": wall,
        "max_fit_wall_seconds": fit_wall,
        "collection_reserve_seconds": reserve,
        "max_profile_wall_seconds": profile_wall,
        "max_initial_profile_attempts": initial_profiles,
        "max_profile_attempts": total_profiles,
    }


def validate_admission(
    contract: Mapping[str, object], admission: Mapping[str, object], contract_sha256: str
) -> dict[str, object]:
    """Require reviewed recipes, exact data inputs, profile policy, and train-only scope."""
    if admission.get("schema") != "wnn.multistage-search-admission.v1":
        raise ValueError("unsupported multistage admission schema")
    if admission.get("status") != "approved":
        raise ValueError("recipe/profile/data admission is not approved")
    if admission.get("experiment_id") != contract.get("experiment_id"):
        raise ValueError("admission experiment id differs from the contract")
    if admission.get("contract_sha256") != contract_sha256:
        raise ValueError("admission does not bind the supplied contract bytes")
    review = admission.get("independent_recipe_review")
    if not isinstance(review, Mapping) or review.get("status") != "approved":
        raise ValueError("an independent recipe review is required")
    data = admission.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("admission is missing the train-only inner data identity")
    if data.get("primary_valid_used") is not False:
        raise ValueError("primary valid must be excluded from search and fitting")
    if data.get("objective_scope") != "inner_train_proxy":
        raise ValueError("stage objectives must be explicitly labelled inner_train_proxy")
    contract_data = contract.get("data_policy")
    contract_primary_sha = (
        contract_data.get("primary_split_sha256") if isinstance(contract_data, Mapping) else None
    )
    primary_sha = data.get("primary_split_sha256")
    if not isinstance(contract_primary_sha, str) or primary_sha != contract_primary_sha:
        raise ValueError("admission primary split differs from the binding contract")
    verify_file_ref(data.get("auxiliary_manifest"), "auxiliary manifest")
    profile = admission.get("profile_policy")
    if not isinstance(profile, Mapping):
        raise ValueError("admission is missing reviewed exact-profile policy")
    review_policy_path, review_policy_sha = verify_file_ref(
        profile.get("independent_checker_policy"), "profile checker policy"
    )
    if profile.get("profile_updates") is None or profile.get("world_size") != 1:
        raise ValueError("profile policy must pin real updates and one-GPU trial workers")
    if not isinstance(profile.get("devices"), Mapping) or not profile["devices"]:
        raise ValueError("profile policy must bind the admitted GPU device set")
    for gpu_key, device in profile["devices"].items():
        if not isinstance(device, Mapping):
            raise ValueError(f"profile policy device {gpu_key} is malformed")
        if any(
            type(device.get(field)) is not int or device[field] <= 0
            for field in ("total_memory_bytes", "torch_total_memory_bytes")
        ):
            raise ValueError(
                f"profile policy device {gpu_key} must preserve separate positive NVML and Torch totals"
            )
    stage_caps = admission.get("stage_fit_caps")
    limits = validate_contract(contract)
    if not isinstance(stage_caps, Mapping) or set(stage_caps) != {"1", "2", "3", "4"}:
        raise ValueError("admission must reserve attempt caps for all four stages")
    if any(type(stage_caps[key]) is not int or stage_caps[key] < 0 for key in stage_caps):
        raise ValueError("stage fit caps must be nonnegative integers")
    if sum(int(stage_caps[key]) for key in stage_caps) > limits["max_fit_attempts"]:
        raise ValueError("stage fit caps exceed the global attempt limit")
    stage1 = admission.get("stage1")
    if not isinstance(stage1, Mapping):
        raise ValueError("admission is missing stage1 conditional search")
    families = stage1.get("families")
    family_extension_path: Path | None = None
    family_extension_sha: str | None = None
    extension_ref = stage1.get("family_extension")
    if extension_ref is not None:
        family_extension_path, family_extension_sha = verify_file_ref(
            extension_ref, "stage1 family extension"
        )
        extension = read_json(family_extension_path)
        extension_stage1 = extension.get("stage1")
        if (
            extension.get("schema") != "wnn.multistage-recipe-family-extension.v1"
            or not isinstance(extension_stage1, Mapping)
            or not isinstance(extension_stage1.get("families"), Mapping)
        ):
            raise ValueError("stage1 family extension schema is unsupported")
        packet_root = family_extension_path.parent.parent
        families = {}
        for name, raw_spec in extension_stage1["families"].items():
            if not isinstance(raw_spec, Mapping):
                raise ValueError("stage1 family extension entries must be objects")
            spec = dict(raw_spec)
            ref = spec.get("base_config")
            if isinstance(ref, Mapping) and isinstance(ref.get("path"), str):
                spec["base_config"] = dict(ref) | {
                    "path": str((packet_root / str(ref["path"])).resolve(strict=True))
                }
            families[name] = spec
    if not isinstance(families, Mapping) or not families:
        raise ValueError("stage1 must admit at least one reviewed family")
    shared_base_ref = stage1.get("base_config")
    shared_base_path: Path | None = None
    shared_base_sha: str | None = None
    if shared_base_ref is not None:
        shared_base_path, shared_base_sha = verify_file_ref(shared_base_ref, "stage1 base config")
        if read_json(shared_base_path).get("precision_mode") != "fp32":
            raise ValueError("initial stage1 configs must use strict FP32")
    family_configs: dict[str, dict[str, object]] = {}
    family_config_hashes: dict[str, str] = {}
    for family_name, raw_spec in families.items():
        if not isinstance(family_name, str) or not isinstance(raw_spec, Mapping):
            raise ValueError("stage1 family definitions must be named objects")
        family_ref = raw_spec.get("base_config", shared_base_ref)
        if family_ref is None:
            raise ValueError(f"stage1 family {family_name} needs an exact base_config reference")
        family_path, family_sha = verify_file_ref(family_ref, f"{family_name} base config")
        family_config = read_json(family_path)
        if family_config.get("precision_mode") != "fp32":
            raise ValueError(f"stage1 family {family_name} must use strict FP32")
        family_configs[family_name] = family_config
        family_config_hashes[family_name] = family_sha
    anchors = stage1.get("anchors", [])
    if not isinstance(anchors, list):
        raise ValueError("stage1 anchors must be a list")
    return {
        "limits": limits,
        "data": dict(data),
        "profile_policy": dict(profile),
        "profile_checker_policy_path": str(review_policy_path),
        "profile_checker_policy_sha256": review_policy_sha,
        "stage_fit_caps": dict(stage_caps),
        "base_config_path": None if shared_base_path is None else str(shared_base_path),
        "base_config_sha256": shared_base_sha,
        "base_config": None,
        "family_configs": family_configs,
        "family_config_hashes": family_config_hashes,
        "families": dict(families),
        "family_extension_path": None
        if family_extension_path is None
        else str(family_extension_path),
        "family_extension_sha256": family_extension_sha,
        "anchors": anchors,
    }
