from __future__ import annotations

import hashlib
import json
from pathlib import Path

SPLIT_SHA = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
FIRST_PROFILE_EPOCH = 1794168764.8507051


def digest(path: Path) -> str:
    """Hash immutable configuration, source, checkpoint or evidence bytes incrementally."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read(path: Path) -> dict:
    """Read ROOT-owned object receipts without following symbolic links."""
    if path.is_symlink():
        raise ValueError("receipt must be a regular native artifact")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def approved(folder: Path, cell: dict, source_sha: str) -> bool:
    """Require ROOT's exact production-profile approval, including both actual reload checks."""
    path = folder / "approvals" / f"{cell['cell']}.json"
    if not path.exists():
        return False
    value = read(path)
    native = read(
        Path(
            cell.get(
                "profile_receipt_path",
                str(folder / "profiles" / cell["cell"] / "profile_receipt.json"),
            )
        )
    )
    if (
        value.get("approved") is not True
        or value.get("reviewer") != "ROOT"
        or any(value.get(k) != item for k, item in native.items() if k != "approved")
        or native.get("status") != "profile_complete"
        or native.get("updates") != 50
        or native.get("errors") != []
        or native.get("scientific_checkpoint_written") is not False
        or native["identity"]["world_size"] != 2
        or native["identity"].get("source_sha256") != source_sha
        or native["identity"].get("config_sha256") != cell["normalized_config_sha256"]
        or native["identity"].get("data", {}).get("split_sha256") != SPLIT_SHA
        or native.get("engineering_reload", {}).get("status") != "passed"
        or len(native["engineering_reload"].get("by_rank", [])) != 2
    ):
        raise ValueError(
            "ROOT approval differs from exact50-update actual precision/reload evidence"
        )
    checks = native["engineering_reload"]["by_rank"]
    if [item.get("rank") for item in checks] != [0, 1] or any(
        item.get("status") != "passed"
        or item.get("model_optimizer_scheduler_rng_equal") is not True
        for item in checks
    ):
        raise ValueError("profile reload proof is incomplete on a production rank")
    return True


def authority(value: dict) -> None:
    """Keep ROOT's earliest eight-hour clock:seven-hour fit plus one-hour collection."""
    if (
        value.get("approved_by") != "ROOT"
        or value.get("first_profile_epoch") != FIRST_PROFILE_EPOCH
        or value.get("fit_cap") != 6
        or value.get("profile_cap") != 6
        or value["fit_deadline_epoch"] - value["first_profile_epoch"] != 25200
        or value["deadline_epoch"] - value["first_profile_epoch"] != 28800
        or value.get("collection_reserve_seconds") != 3600
        or value.get("source_commit") != "5d73bc2b10ae20e607e0d4b4533da85e76e27bca"
    ):
        raise ValueError("ROOT source-bound eight-hour authority or exact six-fit/profile caps differ")


def _sha256_text(value: object) -> bool:
    """Accept only a lowercase SHA256 digest in a ROOT receipt."""
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def software(runtime: Path, plan: dict) -> bool:
    """Require ROOT's source/runtime/CPU qualification and all six exact profile proofs."""
    path = runtime / "software/COMPLETE.json"
    if not path.exists():
        return False
    value = read(path)
    runtime_identity = value.get("runtime", {})
    cpu_tests = value.get("cpu_tests", {})
    if (
        value.get("schema") != "wnn.b300-panel-software-qualification.v1"
        or value.get("approved_by") != "ROOT"
        or value.get("status") != "passed"
        or value.get("source_commit") != plan["source_commit"]
        or value.get("source_manifest_sha256") != plan["source_manifest_sha256"]
        or value.get("split_sha256") != SPLIT_SHA
        or runtime_identity.get("python_version") != "3.12.14"
        or runtime_identity.get("torch_version") != "2.14.0+cu130"
        or runtime_identity.get("cuda_version") != "13.0"
        or runtime_identity.get("nccl_status") != "observed"
        or runtime_identity.get("world_size") != 2
        or runtime_identity.get("device_model") != "NVIDIA_B300"
        or cpu_tests.get("status") != "passed"
        or not _sha256_text(cpu_tests.get("receipt_sha256"))
    ):
        raise ValueError("ROOT source/runtime/CPU qualification is missing or mismatched")
    profiles = value.get("native_profiles")
    expected_cells = {cell["cell"] for cell in plan["cells"]}
    if not isinstance(profiles, dict) or set(profiles) != expected_cells or len(expected_cells) != 6:
        raise ValueError("ROOT qualification must bind exactly the six frozen profile cells")
    for cell in plan["cells"]:
        name = cell["cell"]
        proof = profiles[name]
        profile_path = Path(cell["profile_receipt_path"])
        approval_path = runtime / "approvals" / f"{name}.json"
        if (
            not isinstance(proof, dict)
            or proof.get("profile_receipt_sha256") != digest(profile_path)
            or proof.get("approval_sha256") != digest(approval_path)
        ):
            raise ValueError(f"ROOT qualification profile hashes differ: {name}")
        if not approved(runtime, cell, plan["expected_training_source_sha256"]):
            raise ValueError(f"ROOT profile approval is missing: {name}")
    return True
