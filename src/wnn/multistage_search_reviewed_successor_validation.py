from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_search import file_sha256, read_json
from wnn.multistage_search_runtime import is_same_process
from wnn.multistage_search_successor_approval import approve_successor_profile
from wnn.multistage_search_successor_parent import parent_identity_sha, qualified_parent
from wnn.training.sequence_config import load_config
from wnn.training.sequence_provenance import validate_profile_receipt

STAGE3_BASE_SHA = "31424878cc230cb2b5ca3b46f25b305495a83b6975f8814927cebcff7b0e75f2"


def ensure_gpu_slot_released(run_dir: Path, gpu_key: str) -> None:
    """Reject launch while an exact owned worker/profile process still uses this GPU."""
    paths = set((run_dir / "workers").glob("*.json"))
    paths.update((run_dir / "workers").glob("*.launch_receipt.json"))
    paths.update(run_dir.glob("stage*/trials/*/profile/worker_receipt.json"))
    paths.update((run_dir.parent / "receipts").glob("*.json"))
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, Mapping) or str(record.get("gpu_key")) != gpu_key:
            continue
        candidates = (record, record.get("owner"), record.get("child"))
        if any(isinstance(owner, Mapping) and _is_live_process(owner) for owner in candidates):
            raise RuntimeError(f"owned process still occupies GPU key {gpu_key}: {path}")


def _is_live_process(owner: Mapping[str, object]) -> bool:
    """Confirm exact procfs ownership and ignore a same-identity zombie process."""
    if not is_same_process(owner):
        return False
    pid = owner.get("pid")
    if type(pid) is not int:
        return False
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
    close = raw.rfind(")")
    fields = raw[close + 2 :].split() if close >= 0 else []
    return bool(fields and fields[0] != "Z" and is_same_process(owner))


def validate_profile_members(
    data: object,
    split: object,
    fit_members: list[tuple[str, int]],
    control_members: list[tuple[str, int]],
) -> None:
    """Bind the native profile to the exact train-only fit/control partition."""
    if (
        not isinstance(data, Mapping)
        or data.get("fit_members") != [list(item) for item in fit_members]
        or data.get("evaluation_members") != [list(item) for item in control_members]
        or data.get("evaluation_role") != "control"
        or data.get("evaluation_source") != "train"
        or data.get("evaluation_mask") != "all_after_warmup"
        or data.get("training_mask") != "all_after_warmup"
        or data.get("split_sha256") != split.sha256
        or data.get("primary_split_sha256") != split.primary_split_sha256
    ):
        raise ValueError("native profile does not bind the exact auxiliary fit/control membership")


def validate_initializer(
    request: Mapping[str, object],
    config: Mapping[str, object],
    identity: object,
    child_data: object,
    fit_members: list[tuple[str, int]],
    control_members: list[tuple[str, int]],
) -> None:
    """Require the exact clean fixed-final Stage1 parent and matching Stage2 ancestry."""
    if not isinstance(identity, Mapping) or not isinstance(child_data, Mapping):
        raise ValueError("Stage2 profile lacks native initializer ancestry")
    parent_path = Path(str(request["parent_result_path"])).resolve(strict=True)
    parent_sha = file_sha256(parent_path)
    parent, checkpoint, parent_config = qualified_parent(parent_path, allowed={1})
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    parent_data = payload.get("data") if isinstance(payload, Mapping) else None
    initializer = child_data.get("initializer")
    fields = (
        "split_sha256",
        "primary_split_sha256",
        "fit_members",
        "evaluation_members",
        "evaluation_role",
        "evaluation_source",
        "evaluation_mask",
        "training_mask",
        "mask",
    )
    if (
        not isinstance(parent_data, Mapping)
        or any(parent_data.get(field) != child_data.get(field) for field in fields)
        or parent_data.get("fit_members") != [list(item) for item in fit_members]
        or parent_data.get("evaluation_members") != [list(item) for item in control_members]
        or parent_sha != request.get("initializer_parent_trial_result_sha256")
        or parent_sha != request.get("parent_result_sha256")
        or str(checkpoint) != str(Path(str(request["initializer_path"])).resolve(strict=True))
        or file_sha256(checkpoint) != request.get("initializer_sha256")
        or parent_identity_sha(parent_path) != request.get("parent_identity_sha256")
        or not isinstance(initializer, Mapping)
        or initializer.get("checkpoint_sha256") != request.get("initializer_sha256")
        or initializer.get("parent_identity_sha256") != request.get("parent_identity_sha256")
        or config.get("model") != parent_config.get("model")
        or config.get("epochs") != 2
        or config.get("precision_mode") != "fp32"
        or config.get("kd_weight") != 0
        or config.get("hint_weight") != 0
        or not isinstance(parent.get("objective"), Mapping)
    ):
        raise ValueError("Stage2 initializer is not the exact clean fixed-final Stage1 endpoint")


def validate_stage3_teacher(
    request: Mapping[str, object],
    config: Mapping[str, object],
    identity: object,
    child_data: object,
    split: object,
    fit_members: list[tuple[str, int]],
) -> None:
    """Require the frozen GRU256x2 pair config and exact weighted fit-only teacher pool."""
    if not isinstance(identity, Mapping) or not isinstance(child_data, Mapping):
        raise ValueError("Stage3 profile lacks native teacher ancestry")
    definition = read_json(Path(str(request["config_path"])).parent / "trial_definition.json")
    base_path = Path(str(definition["base_config_path"])).resolve(strict=True)
    params = request.get("params")
    base = load_config(base_path)
    if not isinstance(params, Mapping):
        raise ValueError("Stage3 request has no paired-treatment parameters")
    expected = dict(base) | {
        "epochs": 2,
        "seed": params.get("seed"),
        "kd_weight": params.get("kd_weight"),
    }
    model = config.get("model")
    if (
        file_sha256(base_path) != STAGE3_BASE_SHA
        or definition.get("base_config_sha256") != STAGE3_BASE_SHA
        or definition.get("family") != "gru"
        or config != expected
        or not isinstance(model, Mapping)
        or (model.get("architecture"), model.get("hidden"), model.get("layers")) != ("gru", 256, 2)
    ):
        raise ValueError("Stage3 student differs from the fixed GRU256x2 pair recipe")
    cache_path, producer_ref = request.get("teacher_cache_path"), child_data.get("teacher")
    if config.get("kd_weight") == 0:
        if (
            cache_path is not None
            or producer_ref is not None
            or request.get("pair_role") != "label_only_control"
        ):
            raise ValueError("Stage3 label-only control cannot receive teacher targets")
        return
    if not isinstance(cache_path, str) or request.get("pair_role") != "kd_student":
        raise ValueError("Stage3 KD student is missing its reviewed teacher cache")
    from wnn.splits import CampaignSplit
    from wnn.training.sequence_cache import TeacherCache

    if not isinstance(split, CampaignSplit):
        raise ValueError("Stage3 auxiliary split has an unsupported type")
    cache_root = Path(cache_path).resolve(strict=True)
    manifest_path = cache_root / "manifest.json"
    if file_sha256(manifest_path) != request.get("teacher_cache_manifest_sha256"):
        raise ValueError("Stage3 teacher cache manifest differs from the reviewed request")
    cache = TeacherCache(cache_root, split, fit_members)
    producer = cache.metadata.get("producer")
    if (
        not isinstance(producer, Mapping)
        or producer.get("kind") != "weighted_teacher_pool_v1"
        or producer.get("pool_id") != request.get("teacher_pool_id")
        or producer_ref != cache.identity
    ):
        raise ValueError("Stage3 teacher pool ancestry is not the exact fit-only cache")


def approve_profile(
    run_dir: Path, stage: int, number: int, review_path: Path, identity: object
) -> Path:
    """Create the separate native-compatible approved copy from the pinned review."""
    trial_dir = run_dir / f"stage{stage}" / "trials" / f"{number:06d}"
    profile_dir = trial_dir / "profile"
    approval_path = profile_dir / "approval_record.json"
    approved_path = profile_dir / "approved_profile_receipt.json"
    if approval_path.exists() or approved_path.exists():
        raise FileExistsError("successor profile already has approval artifacts")
    approval = approve_successor_profile(run_dir, stage, number, review_path)
    if not isinstance(identity, Mapping):
        raise ValueError("successor native profile identity is malformed")
    approved = Path(str(approval["approved_profile_path"])).resolve(strict=True)
    if file_sha256(approved) != approval.get("approved_profile_sha256"):
        raise ValueError("approved native-compatible profile copy changed")
    validate_profile_receipt(approved, dict(identity))
    return approved
