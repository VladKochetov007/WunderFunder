from __future__ import annotations

import hashlib
import json
from pathlib import Path

SPLIT_SHA = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"


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
    """Keep the earliest actual GPU qualification clock:3h fit plus1h collection, never reset."""
    if (
        value.get("approved_by") != "ROOT"
        or value.get("fit_cap") != 4
        or value.get("profile_cap") != 8
        or value["fit_deadline_epoch"] - value["first_profile_epoch"] != 10800
        or value["deadline_epoch"] - value["first_profile_epoch"] != 14400
        or value.get("collection_reserve_seconds") != 3600
    ):
        raise ValueError("ROOT four-hour authority or exact fit/profile caps differ")


def software(runtime: Path, plan: dict) -> bool:
    """Bind actual NCCL qualification to the exact loss bytes in the final source capsule."""
    path = runtime / "software/COMPLETE.json"
    if not path.exists():
        return False
    value = read(path)
    if (
        value.get("approved_by") != "ROOT"
        or value.get("status") != "passed"
        or value.get("source_manifest_sha256") != plan["source_manifest_sha256"]
    ):
        raise ValueError("ROOT source/software acceptance is missing or uses another source")
    oracle_spec = value["native_nccl_oracle"]
    oracle_path = Path(oracle_spec["path"])
    oracle = read(oracle_path)
    manifest = read(Path(plan["source_manifest"]))
    if (
        digest(oracle_path) != oracle_spec["sha256"]
        or oracle.get("status") != "passed"
        or oracle.get("backend") != "nccl"
        or oracle.get("world_size") != 2
        or oracle.get("correlation_prediction_std_floor") != 0.01
        or len(oracle.get("cases", [])) != 8
        or any(
            manifest["files"].get(name) != sha for name, sha in oracle["loss_source_sha256"].items()
        )
    ):
        raise ValueError("actual NCCL gradient oracle differs from final source/math contract")
    return True


def retirement(runtime: Path, same: object) -> bool:
    """Require ROOT's safe whole-epoch partial-retirement proof; never send an old-process signal."""
    path = runtime / "control/old-four-retired.json"
    if not path.exists():
        return False
    value = read(path)
    if (
        value.get("approved_by") != "ROOT"
        or value.get("status") != "retired_at_safe_epoch"
        or len(value.get("cells", [])) != 4
    ):
        raise ValueError("old fixed30E attempts lack explicit partial safe-retirement authority")
    for item in value["cells"]:
        epoch = item["next_epoch"]
        if (
            type(epoch) is not int
            or not 1 <= epoch < 30
            or item.get("global_updates") != epoch * 820
            or item.get("fixed30E_complete") is not False
            or digest(Path(item["checkpoint"]["path"])) != item["checkpoint"]["sha256"]
            or any(same(owner) for owner in item["stopped_trainers"])
        ):
            raise ValueError("old fit has not safely retired with its unchanged partial checkpoint")
    return True
