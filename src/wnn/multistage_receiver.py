from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wnn.multistage_gpu_identity import normalize_gpu_uuid
from wnn.multistage_search import (
    atomic_write_json,
    canonical_json_sha256,
    file_sha256,
    read_json,
)

PACKET = Path("research/experiment_drafts/multistage-optuna-b300-20261031")
PRIMARY_SHA = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
AUXILIARY_SHA = "26b448bad7f60612cea482f4abaf5f184e5a724228eef227d7e78a8155190075"
SOURCE3_SHA = "873a3c07fd88d3eec9ed52b2472b8d7f37ef5f488855a3c753db438e4fdee5aa"
SOURCE4_SHA = "c520c67ca8d1fb49fc0c452a4606f5ad130ed51e7fc689fdaf9daea2b429dd81"
SOURCE5B_SHA = "2661a89285bd528e1bca690a340e71a21816a7b54e3ba1635d14925f3a9c1e4e"
CAPS = {"1": 16, "2": 8, "3": 12, "4": 12}


def process_is_owned_and_live(identity: Mapping[str, object]) -> bool:
    """Return whether the recorded Linux PID and birth still identify its worker."""
    pid, birth, pgid = identity.get("pid"), identity.get("birth"), identity.get("pgid")
    if type(pid) is not int or not isinstance(birth, str) or type(pgid) is not int:
        raise ValueError("external reservation has an incomplete PID/birth/PGID tuple")
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        raw = stat_path.read_text(encoding="ascii")
    except FileNotFoundError:
        return False
    except PermissionError as error:
        raise RuntimeError(f"cannot verify owned process identity for PID {pid}") from error
    close = raw.rfind(")")
    if close < 0:
        raise ValueError(f"malformed proc stat for reserved PID {pid}")
    fields = raw[close + 2 :].split()
    if len(fields) <= 19:
        raise ValueError(f"short proc stat for reserved PID {pid}")
    # Keep the slot if the original PID/birth survives, even if its process group changed.
    return fields[19] == birth


def dynamic_reserved_gpu_keys(admission: Mapping[str, object]) -> list[str]:
    """Recompute reserved GPU slots from exact external owner receipts on each poll."""
    rows = admission.get("external_gpu_reservations", [])
    if not isinstance(rows, list):
        raise ValueError("external GPU reservation list is malformed")
    reserved: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("external GPU reservation entry is malformed")
        key, path_value, expected_sha = (
            row.get("gpu_key"),
            row.get("owner_receipt_path"),
            row.get("owner_receipt_sha256"),
        )
        if (
            not isinstance(key, str)
            or not isinstance(path_value, str)
            or not isinstance(expected_sha, str)
        ):
            raise ValueError("external GPU reservation is missing its identity binding")
        path = Path(path_value)
        if path.is_symlink() or file_sha256(path) != expected_sha:
            raise ValueError(f"external GPU owner receipt changed: {path}")
        owner = read_json(path)
        expected = row.get("process")
        if not isinstance(expected, Mapping) or any(
            owner.get(name) != expected.get(name) for name in ("pid", "birth", "pgid")
        ):
            raise ValueError(f"external owner PID/birth/PGID differs from admission: {path}")
        if process_is_owned_and_live(expected):
            reserved.add(key)
    return sorted(reserved, key=int)


def build_r2_admission(
    runtime_root: Path,
    operator_root: Path,
    training_source: Path,
    evaluation_source: Path,
    recipe_review_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Assemble a receipt-bound R2 admission from frozen inputs and actual node artifacts."""
    runtime_root, operator_root, training_source, evaluation_source = (
        path.resolve(strict=True)
        for path in (runtime_root, operator_root, training_source, evaluation_source)
    )
    packet = operator_root / PACKET
    contract_path = packet / "CONTRACT.json"
    template_path = packet / "ADMISSION_TEMPLATE_R1.json"
    template = read_json(template_path)
    source4 = (runtime_root / "operator-native4").resolve(strict=True)
    from wnn.training.sequence_provenance import source_identity, validate_profile_receipt

    source3 = (runtime_root / "source-r3").resolve(strict=True)
    source3_identity = source_identity(source3, source3 / "scripts/train_sequence.py")
    source4_identity = source_identity(source4, source4 / "scripts/train_sequence.py")
    source5_identity = source_identity(
        training_source, training_source / "scripts/train_sequence.py"
    )
    evaluation_identity = source_identity(
        evaluation_source, evaluation_source / "scripts/predict_sequence.py"
    )
    if (
        source3_identity.get("sha256") != SOURCE3_SHA
        or source4_identity.get("sha256") != SOURCE4_SHA
        or source5_identity.get("sha256") != SOURCE5B_SHA
    ):
        raise ValueError("one or more immutable native training-source identities changed")
    data = dict(template["data"])
    data_root = runtime_root / "data"
    primary_manifest = data_root / "chronological_primary.json"
    auxiliary_manifest = data_root / "cache/multistage/auxiliary-r3.json"
    if file_sha256(auxiliary_manifest) != AUXILIARY_SHA:
        raise ValueError("the frozen auxiliary train-only manifest changed")
    from wnn.splits import load_campaign_split
    from wnn.training.sequence_cache_weighted_pool_identity import auxiliary_members

    split = load_campaign_split(auxiliary_manifest, data_root)
    fit_members, control_members = auxiliary_members(split)
    if (
        len(fit_members) != 8260
        or len(control_members) != 2065
        or any(member[0] != "train" for member in (*fit_members, *control_members))
        or getattr(split, "primary_split_sha256", None) != PRIMARY_SHA
    ):
        raise ValueError(
            "actual auxiliary split membership differs from the frozen train-only scope"
        )
    data.update(
        {
            "primary_split_manifest": str(primary_manifest),
            "auxiliary_manifest": {"path": str(auxiliary_manifest), "sha256": AUXILIARY_SHA},
            "data_root": str(data_root),
            "primary_valid_used": False,
            "objective_scope": "inner_train_proxy",
        }
    )
    training_revisions = {
        "1": f"operator-native4:{SOURCE4_SHA}",
        "2": f"source5b-core:{SOURCE5B_SHA}",
        "3": f"source5b-core:{SOURCE5B_SHA}",
    }
    source_identities = {"1": source4_identity, "2": source5_identity, "3": source5_identity}
    stage_source_hashes = {
        stage: {
            "sha256": str(identity["sha256"]),
            "files_sha256": canonical_json_sha256(identity["files"]),
        }
        for stage, identity in source_identities.items()
    }
    template_policy = dict(template["profile_policy"])
    policy_path = packet / "PROFILE_REVIEW_POLICY_AUTOMATED_R2.json"
    reviewer_path = packet / "review_multistage_profile_r2.py"
    policy = read_json(policy_path)
    if (
        policy.get("schema") != "wnn.multistage-profile-review-policy.v3"
        or policy.get("status") != "frozen"
    ):
        raise ValueError("automated profile policy is not root-frozen")
    if not reviewer_path.is_file() or reviewer_path.is_symlink():
        raise FileNotFoundError("successor profile reviewer is missing or unsafe")
    if policy.get("reviewer_script_sha256") != file_sha256(reviewer_path):
        raise ValueError("frozen profile policy does not bind the current reviewer bytes")
    reviewer_path_hash = file_sha256(reviewer_path)
    allocation_path = runtime_root / "receipts/b300-fit-allocation-r3.json"
    allocation = read_json(allocation_path)
    devices, inventory_sha = _device_policy(allocation, template_policy, runtime_root)
    expected_policy_sources = policy.get("identity", {}).get("source_sha256_by_stage")
    if not isinstance(expected_policy_sources, Mapping) or {
        str(key): str(value) for key, value in expected_policy_sources.items()
    } != {key: str(value["sha256"]) for key, value in stage_source_hashes.items()}:
        raise ValueError(
            "profile policy source map does not match the stage-specific trainer sources"
        )
    template_policy.update(
        {
            "independent_checker_policy": {
                "path": str(policy_path),
                "sha256": file_sha256(policy_path),
            },
            "reviewer_script": {
                "path": str(reviewer_path.resolve()),
                "sha256": reviewer_path_hash,
            },
            "profile_updates": 50,
            "profile_wall_seconds": 180,
            "world_size": 1,
            "max_concurrent_workers": 8,
            "devices": devices,
            "device_inventory_sha256": inventory_sha,
        }
    )
    attempts, seen_profiles, reservations = _collect_actual_attempts(
        runtime_root, packet, template, devices, validate_profile_receipt
    )
    seen_profiles.update(_small_profile_identities(runtime_root))
    if len(attempts) != 10 or len(seen_profiles) != 16:
        raise ValueError(
            f"expected 10 preexisting fits and 16 successful profile identities; found "
            f"{len(attempts)} and {len(seen_profiles)}"
        )
    if len(template.get("preexisting_fit_attempts", [])) != 2:
        raise ValueError("R1 baseline must retain exactly the two source3 full-fit registrations")
    r6_review_path = packet / "INDEPENDENT_R6_PROFILE_REVIEW.json"
    r5_review_path = packet / "INDEPENDENT_ACTUAL_PROFILE_REVIEW_R3.json"
    s2_review_path = packet / "INDEPENDENT_STAGE2_PROFILE_REVIEW_R1.json"
    continuation_path = packet / "recipes/R2_CONTINUATION.json"
    continuation = read_json(continuation_path)
    if (
        continuation.get("schema") != "wnn.multistage-r2-continuation-recipe.v1"
        or continuation.get("limits", {}).get("stage_fit_caps") != CAPS
        or continuation.get("limits", {}).get("preexisting_fit_attempts") != 10
        or continuation.get("limits", {}).get("profile_attempts_used") != 22
    ):
        raise ValueError("frozen R2 continuation recipe does not match the authorized cap ledger")
    continuation_sha = file_sha256(continuation_path)
    stage1_space = _stage1_space(packet)
    stage2_admission = dict(template["stage2"]) | {
        "max_trials": 6,
        "lr_multiplier_choices": [0.1, 0.25, 0.5],
        "epochs": 2,
        "preserve_weight_decay": True,
        "search_space": {
            "lr_multiplier": {"kind": "categorical", "choices": [0.1, 0.25, 0.5]},
            "epochs": {"kind": "constant", "value": 2},
        },
    }
    stage3_recipe = continuation.get("stage3")
    student_config_ref = (
        stage3_recipe.get("student_base_config") if isinstance(stage3_recipe, Mapping) else None
    )
    if not isinstance(student_config_ref, Mapping):
        raise ValueError("R2 continuation must pin the fixed Stage3 student config")
    student_config_path = (packet / str(student_config_ref.get("path"))).resolve(strict=True)
    student_config_sha = str(student_config_ref.get("sha256"))
    if file_sha256(student_config_path) != student_config_sha:
        raise ValueError("fixed Stage3 GRU256x2 config differs from the continuation recipe")
    student_config = read_json(student_config_path)
    if (
        student_config.get("model", {}).get("architecture") != "gru"
        or student_config.get("model", {}).get("hidden") != 256
        or student_config.get("model", {}).get("layers") != 2
        or student_config.get("batch_size") != 256
        or student_config.get("input_prefetch_batches") != 1
        or student_config.get("monitoring", {}).get("eval_batch_size") != 64
    ):
        raise ValueError("Stage3 fixed student config is not the admitted GRU256x2 recipe")
    stage3_admission = dict(template["stage3"]) | {
        "max_trials": 12,
        "maximum_pairs": 6,
        "paired_fits": True,
        "epochs": 2,
        "student_family": "gru",
        "student_base_config": {
            "path": str(student_config_path),
            "sha256": student_config_sha,
        },
        "search_space": {
            "kd_weight": {"kind": "categorical", "choices": [0.05, 0.1, 0.25, 0.5, 1.0]},
        },
    }
    stage4_admission = dict(template["stage4"]) | {
        "max_optuna_trials": 24,
        "max_candidate_count": 32,
        "final_refit_cap": 12,
    }
    _validate_continuation_projection(
        continuation, stage1_space, stage2_admission, stage3_admission, stage4_admission
    )
    recipe = {
        "schema": "wnn.multistage-r2-recipe.v1",
        "continuation_recipe": {"path": str(continuation_path), "sha256": continuation_sha},
        "contract_sha256": file_sha256(contract_path),
        "data": {
            "primary_split_sha256": PRIMARY_SHA,
            "auxiliary_split_sha256": AUXILIARY_SHA,
            "fit_members": len(fit_members),
            "control_members": len(control_members),
            "primary_valid_used": False,
        },
        "stage_sources": stage_source_hashes,
        "evaluation_source": {
            "root": str(evaluation_source),
            "predict_source_sha256": evaluation_identity["sha256"],
        },
        "stage_fit_caps": CAPS,
        "stage1": stage1_space,
        "stage2": stage2_admission,
        "stage3": stage3_admission,
        "stage4": stage4_admission,
        "reviews": {
            "r5": file_sha256(r5_review_path),
            "r6": file_sha256(r6_review_path),
            "stage2": file_sha256(s2_review_path),
            "profile_policy": file_sha256(policy_path),
            "profile_reviewer": reviewer_path_hash,
        },
        "limits": {"max_fits": 48, "max_profiles": 56, "wall_seconds": 21600, "reserve": 1200},
    }
    recipe_sha = canonical_json_sha256(recipe)
    recipe_draft_path = runtime_root / "receipts/multistage-r2-recipe.json"
    _write_identical_or_new(recipe_draft_path, recipe)
    review_path = recipe_review_path.resolve(strict=True)
    review = read_json(review_path)
    if (
        review.get("status") != "approved"
        or review.get("continuation_sha256", review.get("recipe_sha256")) != continuation_sha
        or review.get("reviewer") in (None, "root")
    ):
        raise ValueError(
            "independent R2 review does not approve the exact frozen continuation bytes"
        )
    admission = dict(template)
    admission.update(
        {
            "status": "approved",
            "revision": "R2",
            "contract_sha256": file_sha256(contract_path),
            "independent_recipe_review": {
                "status": "approved",
                "review_id": review.get("review_id", "R2"),
                "reviewer": review["reviewer"],
                "path": str(review_path),
                "sha256": file_sha256(review_path),
                "recipe_sha256": continuation_sha,
            },
            "training_source_root": str(training_source),
            "training_source_revision": training_revisions["2"],
            "training_source_identity_sha256": SOURCE5B_SHA,
            "stage_training_source_identities": stage_source_hashes,
            "stage_training_source_revisions": training_revisions,
            "evaluation_source_root": str(evaluation_source),
            "evaluation_source_revision": f"source6-eval:{evaluation_identity['sha256']}",
            "data": data,
            "profile_policy": template_policy,
            "stage_fit_caps": CAPS,
            "preexisting_fit_attempts": attempts,
            "preexisting_profile_attempts": 22,
            "preprofiled_config_identity_sha256s": sorted(seen_profiles),
            "external_gpu_reservations": reservations,
            "reserved_gpu_keys": dynamic_reserved_gpu_keys(
                {"external_gpu_reservations": reservations}
            ),
            "first_gpu_profile_epoch": _first_profile_epoch(runtime_root, template),
            "stage1": {
                "families": stage1_space["families"],
                "anchors": [],
                "max_trials": 8,
                "objective": template["stage1"]["objective"],
            },
            "stage2": stage2_admission,
            "stage3": stage3_admission,
            "stage4": stage4_admission,
        }
    )
    admission["provenance"] = {
        "recipe_path": str(recipe_draft_path),
        "recipe_sha256": recipe_sha,
        "source3_identity_sha256": SOURCE3_SHA,
        "source4_identity_sha256": SOURCE4_SHA,
        "source5b_identity_sha256": SOURCE5B_SHA,
        "evaluation_predict_source_sha256": evaluation_identity["sha256"],
        "allocation_receipt_path": str(allocation_path),
        "allocation_receipt_sha256": file_sha256(allocation_path),
        "profile_history": _profile_history_record(packet, template, seen_profiles),
    }
    admission["independent_recipe_review"]["required_before_prepare"] = False
    return admission, recipe


def _stage1_space(packet: Path) -> dict[str, object]:
    """Bind the exact R6 GRU/LSTM bases and the root-authorized bounded continuation space."""
    review = read_json(packet / "INDEPENDENT_R6_PROFILE_REVIEW.json")
    entries = {item["id"]: item for item in review["profiles"] if isinstance(item, Mapping)}
    families: dict[str, object] = {}
    for family, anchor in (("gru", "gru384x2"), ("lstm", "lstm384x2")):
        entry = entries[anchor]
        relative = Path("recipes/r6") / f"{family}384x2.json"
        config_path = packet / relative
        if file_sha256(config_path) != entry["config_raw_sha256"]:
            raise ValueError(f"R6 {family} base config changed")
        families[family] = {
            "base_config": {"path": str(config_path), "sha256": entry["config_raw_sha256"]},
            "search_space": {
                "hidden": {
                    "kind": "categorical",
                    "choices": [192, 256, 384],
                    "config_path": "model.hidden",
                },
                "layers": {
                    "kind": "categorical",
                    "choices": [1, 2, 3],
                    "config_path": "model.layers",
                },
                "lr": {
                    "kind": "float",
                    "low": 0.0003,
                    "high": 0.003,
                    "log": True,
                    "config_path": "lr",
                },
                "dropout": {
                    "kind": "categorical",
                    "choices": [0.0, 0.1],
                    "config_path": "model.dropout",
                    "conditional_on": {"dimension": "layers", "choices": [2, 3], "otherwise": 0.0},
                },
                "epochs": {"kind": "categorical", "choices": [4, 8], "config_path": "epochs"},
                "seed": {"kind": "categorical", "choices": [1302, 1303], "config_path": "seed"},
            },
            "fixed_overrides": {},
        }
    return {"families": families, "max_trials": 8}


def _validate_continuation_projection(
    continuation: Mapping[str, object],
    stage1: Mapping[str, object],
    stage2: Mapping[str, object],
    stage3: Mapping[str, object],
    stage4: Mapping[str, object],
) -> None:
    """Require each executable stage search to match the independently reviewed recipe."""
    recipe_stage1 = continuation.get("stage1")
    recipe_families = recipe_stage1.get("families") if isinstance(recipe_stage1, Mapping) else None
    actual_families = stage1.get("families")
    if not isinstance(recipe_families, Mapping) or not isinstance(actual_families, Mapping):
        raise ValueError("reviewed R2 stage1 family definitions are malformed")
    if recipe_stage1.get("additional_fit_cap") != stage1.get("max_trials"):
        raise ValueError("stage1 admission cap differs from the reviewed continuation")
    for family in ("gru", "lstm"):
        reviewed = recipe_families.get(family)
        admitted = actual_families.get(family)
        if not isinstance(reviewed, Mapping) or not isinstance(admitted, Mapping):
            raise ValueError(f"reviewed/admitted stage1 family {family} is missing")
        base = reviewed.get("base_config")
        admitted_base = admitted.get("base_config")
        if not isinstance(base, Mapping) or not isinstance(admitted_base, Mapping):
            raise ValueError(f"stage1 {family} base config references are malformed")
        if base.get("sha256") != admitted_base.get("sha256"):
            raise ValueError(f"stage1 {family} base config differs from the reviewed continuation")
        reviewed_search = reviewed.get("search")
        actual_search = admitted.get("search_space")
        if not isinstance(reviewed_search, Mapping) or not isinstance(actual_search, Mapping):
            raise ValueError(f"stage1 {family} search space is malformed")
        expected = {
            "hidden": reviewed_search.get("model.hidden"),
            "layers": reviewed_search.get("model.layers"),
            "dropout": reviewed_search.get("model.dropout"),
            "epochs": reviewed_search.get("epochs"),
            "seed": reviewed_search.get("seed"),
        }
        for name, choices in expected.items():
            spec = actual_search.get(name)
            if not isinstance(spec, Mapping) or spec.get("choices") != choices:
                raise ValueError(f"stage1 {family} {name} dimension differs from reviewed search")
        lr, actual_lr = reviewed_search.get("lr"), actual_search.get("lr")
        if (
            not isinstance(lr, Mapping)
            or not isinstance(actual_lr, Mapping)
            or actual_lr.get("low") != lr.get("low")
            or actual_lr.get("high") != lr.get("high")
            or actual_lr.get("log") != lr.get("log")
        ):
            raise ValueError(f"stage1 {family} LR dimension differs from the reviewed search")
        conditional = actual_search.get("dropout", {}).get("conditional_on")
        if conditional != {
            "dimension": "layers",
            "choices": [2, 3],
            "otherwise": 0.0,
        }:
            raise ValueError("stage1 dropout conditional differs from the reviewed rule")

    reviewed_stage2 = continuation.get("stage2")
    reviewed_stage3 = continuation.get("stage3")
    reviewed_stage4 = continuation.get("stage4")
    if not all(
        isinstance(item, Mapping) for item in (reviewed_stage2, reviewed_stage3, reviewed_stage4)
    ):
        raise ValueError("reviewed R2 successor stage definitions are malformed")
    if (
        stage2.get("max_trials") != reviewed_stage2.get("additional_fit_cap")
        or stage2.get("lr_multiplier_choices") != reviewed_stage2.get("lr_multiplier_choices")
        or stage2.get("epochs") != reviewed_stage2.get("epochs")
        or stage2.get("preserve_weight_decay") is not reviewed_stage2.get("preserve_weight_decay")
        or reviewed_stage2.get("preserve_weight_decay") is not True
        or stage2.get("search_space", {}).get("lr_multiplier", {}).get("choices")
        != reviewed_stage2.get("lr_multiplier_choices")
        or stage2.get("search_space", {}).get("epochs", {}).get("value")
        != reviewed_stage2.get("epochs")
    ):
        raise ValueError("stage2 admission search differs from the reviewed continuation")
    reviewed_student = reviewed_stage3.get("student_base_config")
    admitted_student = stage3.get("student_base_config")
    if (
        stage3.get("max_trials") != reviewed_stage3.get("additional_fit_cap")
        or stage3.get("maximum_pairs") != reviewed_stage3.get("maximum_pairs")
        or stage3.get("paired_fits") is not reviewed_stage3.get("paired_fits")
        or stage3.get("epochs") != reviewed_stage3.get("epochs")
        or stage3.get("student_family") != reviewed_stage3.get("student_family")
        or not isinstance(reviewed_student, Mapping)
        or not isinstance(admitted_student, Mapping)
        or stage3.get("student_family") != "gru"
        or admitted_student.get("sha256") != reviewed_student.get("sha256")
        or stage3.get("search_space", {}).get("kd_weight", {}).get("choices")
        != reviewed_stage3.get("kd_weight_choices")
        or "student_family" in stage3.get("search_space", {})
    ):
        raise ValueError("stage3 admission differs from the reviewed fixed-GRU KD recipe")
    if (
        stage4.get("max_optuna_trials") != reviewed_stage4.get("maximum_optuna_trials")
        or stage4.get("max_candidate_count")
        != reviewed_stage4.get("maximum_distinct_blend_candidates")
        or stage4.get("final_refit_cap") != reviewed_stage4.get("final_refit_cap")
    ):
        raise ValueError("stage4 admission differs from the reviewed continuation")


def _device_policy(
    allocation: Mapping[str, object], template_policy: Mapping[str, object], runtime_root: Path
) -> tuple[dict[str, object], str]:
    """Combine exact NVML inventory with per-UUID Torch totals measured in actual profiles."""
    raw_devices = allocation.get("devices")
    if isinstance(raw_devices, list):
        entries = {
            str(item.get("gpu_index")): item for item in raw_devices if isinstance(item, Mapping)
        }
    elif isinstance(raw_devices, Mapping):
        entries = {str(key): item for key, item in raw_devices.items() if isinstance(item, Mapping)}
    else:
        raise ValueError("owned B300 allocation has no measured devices")
    if set(entries) != {str(index) for index in range(8)}:
        raise ValueError("R2 receiver requires the exact eight-device owner allocation")
    torch_totals = _torch_totals_by_gpu(runtime_root)
    devices: dict[str, object] = {}
    for key, record in entries.items():
        nvml_total = record.get("nvml_total_memory_bytes", record.get("total_memory_bytes"))
        uuid = record.get("uuid", record.get("gpu_uuid"))
        torch_total, torch_uuid = torch_totals.get(key, (None, None))
        if (
            not isinstance(uuid, str)
            or type(nvml_total) is not int
            or type(torch_total) is not int
            or normalize_gpu_uuid(torch_uuid) != normalize_gpu_uuid(uuid)
        ):
            raise ValueError(f"device {key} lacks exact NVML/Torch memory and UUID evidence")
        selector = str(record.get("cuda_visible_devices", uuid))
        devices[key] = {
            "gpu_index": int(key),
            "uuid": uuid,
            "cuda_visible_devices": selector,
            "total_memory_bytes": nvml_total,
            "torch_total_memory_bytes": torch_total,
        }
    inventory_sha = allocation.get("device_inventory_sha256")
    if not isinstance(inventory_sha, str):
        inventory_sha = template_policy.get("device_inventory_sha256")
    if not isinstance(inventory_sha, str) or len(inventory_sha) != 64:
        raise ValueError("allocation does not bind the B300 inventory digest")
    return devices, inventory_sha


def _torch_totals_by_gpu(runtime_root: Path) -> dict[str, tuple[int, str | None]]:
    """Read Torch memory totals and UUIDs from the exact eight successful profiles."""
    profile_paths = [
        runtime_root / f"profiles/{family}-b256-r6-{name}/profile_receipt.json"
        for family, name in (
            ("gru", "gru384x2"),
            ("gru", "gru256x3"),
            ("lstm", "lstm384x2"),
            ("lstm", "lstm256x3"),
            ("galam", "galam64"),
            ("transformer", "transformer384x4"),
        )
    ]
    profile_paths += [
        runtime_root / f"profiles/{family}-b256-s2-pilot-r2/profile_receipt.json"
        for family in ("gru", "lstm")
    ]
    result: dict[str, tuple[int, str | None]] = {}
    for path in profile_paths:
        profile = read_json(path)
        identity, runtime = profile.get("identity"), profile.get("identity", {}).get("runtime")
        devices = runtime.get("devices") if isinstance(runtime, Mapping) else None
        if not isinstance(identity, Mapping) or not isinstance(devices, list) or len(devices) != 1:
            raise ValueError(f"profile has no one-device Torch runtime identity: {path}")
        actual = devices[0]
        gpu = profile.get("gpu_index", profile.get("device_index"))
        # Per-profile native identities carry the actual UUID; the receipt gives the inventory key.
        receipt_path = _profile_attempt_for_profile(path)
        receipt = read_json(receipt_path)
        gpu = receipt.get("device", {}).get("gpu_index", receipt.get("device", {}).get("index"))
        total = actual.get("total_memory_bytes", actual.get("torch_total_memory_bytes"))
        uuid = actual.get("uuid", actual.get("device_uuid"))
        if type(gpu) is not int or type(total) is not int or not isinstance(uuid, str):
            raise ValueError(f"Torch runtime device identity is incomplete: {path}")
        key = str(gpu)
        previous = result.get(key)
        if previous is not None and (
            previous[0] != total or normalize_gpu_uuid(previous[1]) != normalize_gpu_uuid(uuid)
        ):
            raise ValueError(f"profiles disagree about Torch identity for GPU {key}")
        if previous is None:
            result[key] = (total, uuid)
    if set(result) != {str(index) for index in range(8)}:
        raise ValueError("actual profiles do not cover all eight B300 Torch identities")
    return result


def _profile_attempt_for_profile(profile_path: Path) -> Path:
    """Resolve the native attempt receipt paired with an exact profile directory."""
    runtime_root = profile_path.parents[2]
    name = profile_path.parent.name
    if "-r6-" in name:
        suffix = name.split("-r6-", 1)[1]
        family = name.split("-b256", 1)[0]
        return runtime_root / "receipts" / f"anchor-{family}-b256-r6-{suffix}.json"
    if "-s2-pilot-r2" in name:
        family = name.split("-b256", 1)[0]
        return runtime_root / "receipts" / f"anchor-{family}-b256-s2-pilot-r2.json"
    raise ValueError(f"unrecognized profile location: {profile_path}")


def _collect_actual_attempts(
    runtime_root: Path,
    packet: Path,
    template: Mapping[str, object],
    devices: Mapping[str, object],
    validate_profile_receipt: Any,
) -> tuple[list[dict[str, object]], set[str], list[dict[str, object]]]:
    """Bind the ten prior fits, their exact profile/config reviews, and live owner slots."""
    entries: list[tuple[int, str, str, Path, Path, Path, Path, Path, str, str]] = []
    r5_review = packet / "INDEPENDENT_ACTUAL_PROFILE_REVIEW_R3.json"
    r6_review = packet / "INDEPENDENT_R6_PROFILE_REVIEW.json"
    s2_review = packet / "INDEPENDENT_STAGE2_PROFILE_REVIEW_R1.json"
    entries.extend(
        (
            1,
            f"preexisting-stage1-{family}-b256-r5",
            family,
            runtime_root / f"requests/{family}-b256-prefetch1-r5.json",
            runtime_root / f"profiles/{family}-b256-r5/profile_receipt.json",
            runtime_root / f"profiles/{family}-b256-r5/approved_profile_receipt_r3.json",
            runtime_root / f"receipts/fit-{family}-b256-r5.owner.json",
            runtime_root / f"runs/anchor-{family}-b256-r5",
            r5_review,
            "native-source-r3-portable-20261031",
        )
        for family in ("gru", "lstm")
    )
    r6 = read_json(r6_review)
    for profile in r6["profiles"]:
        key = str(profile["id"])
        family = str(profile["family"])
        entries.append(
            (
                1,
                f"preexisting-stage1-r6-{key}",
                family,
                Path(str(profile["remote_config_path"])),
                Path(str(profile["native_profile_path"])),
                Path(str(profile["native_profile_path"])).with_name(
                    "approved_profile_receipt_r4.json"
                ),
                runtime_root / f"receipts/fit-r6-{key}.owner.json",
                runtime_root / f"runs/r6-{key}",
                r6_review,
                f"operator-native4:{SOURCE4_SHA}",
            )
        )
    s2 = read_json(s2_review)
    for profile in s2["profiles"]:
        family = str(profile["family"])
        profile_path = Path(str(profile["native_profile_path"]))
        entries.append(
            (
                2,
                f"preexisting-stage2-{family}-polish-r1",
                family,
                Path(str(profile["config_remote_path"])),
                profile_path,
                profile_path.with_name("approved_profile_receipt_stage2_r1.json"),
                runtime_root / f"receipts/fit-stage2-{family}-polish-r1.owner.json",
                runtime_root / f"runs/stage2-{family}-polish-r1",
                s2_review,
                f"source5b-core:{SOURCE5B_SHA}",
            )
        )
    attempts: list[dict[str, object]] = []
    profile_identities: set[str] = set()
    reservations: list[dict[str, object]] = []
    for _, (
        stage,
        attempt_id,
        family,
        config_path,
        native_path,
        approved_path,
        owner_path,
        output,
        review_path,
        revision,
    ) in enumerate(entries):
        config_path, native_path, approved_path, owner_path = (
            path.resolve(strict=True)
            for path in (config_path, native_path, approved_path, owner_path)
        )
        if any(path.is_symlink() for path in (config_path, native_path, approved_path, owner_path)):
            raise ValueError(f"fit attempt {attempt_id} has a symlink input")
        config, native, approved, owner = map(
            read_json, (config_path, native_path, approved_path, owner_path)
        )
        identity = native.get("identity")
        if not isinstance(identity, Mapping) or native.get("approved") is not False:
            raise ValueError(
                f"fit attempt {attempt_id} lacks an original unapproved native profile"
            )
        config_sha = file_sha256(config_path)
        native_sha = file_sha256(native_path)
        review = read_json(review_path)
        if config_sha != _expected_config_sha(stage, attempt_id, review_path, packet):
            raise ValueError(f"fit attempt {attempt_id} config bytes differ from review")
        _validate_review_entry(stage, attempt_id, review, config_sha, native_sha)
        from wnn.training.sequence_config import load_config

        if identity.get("config_sha256") != canonical_json_sha256(load_config(config_path)):
            raise ValueError(f"fit attempt {attempt_id} native config identity differs from bytes")
        config_identity = str(identity.get("config_sha256"))
        profile_identities.add(config_identity)
        if identity.get("source_sha256") != _expected_source_sha(stage, attempt_id):
            raise ValueError(
                f"fit attempt {attempt_id} source identity differs from its source stage"
            )
        profile_data = identity.get("data")
        if (
            not isinstance(profile_data, Mapping)
            or profile_data.get("split_sha256") != AUXILIARY_SHA
            or profile_data.get("primary_split_sha256") != PRIMARY_SHA
            or profile_data.get("evaluation_role") != "control"
            or profile_data.get("evaluation_source") != "train"
            or profile_data.get("evaluation_mask") != "all_after_warmup"
            or profile_data.get("training_mask") != "all_after_warmup"
            or profile_data.get("teacher") is not None
            or (stage == 1 and profile_data.get("initializer") is not None)
        ):
            raise ValueError(
                f"fit attempt {attempt_id} profile data identity is outside the inner scope"
            )
        review_sha = file_sha256(review_path)
        external = approved.get("external_review")
        if (
            approved.get("approved") is not True
            or approved.get("identity") != identity
            or not isinstance(external, Mapping)
            or external.get("native_profile_sha256") != native_sha
            or external.get("sha256") != review_sha
        ):
            raise ValueError(
                f"fit attempt {attempt_id} approved copy is not bound to its reviewed native bytes"
            )
        validate_profile_receipt(approved_path, dict(identity))
        command = owner.get("command")
        command_sha = hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if (
            not isinstance(command, list)
            or owner.get("command_sha256") != command_sha
            or owner.get("stage") != stage
            or owner.get("output_dir") != str(output.resolve(strict=False))
            or (stage == 1 and owner.get("attempt_id") != attempt_id)
            or type(owner.get("pid")) is not int
            or not isinstance(owner.get("birth"), str)
            or type(owner.get("pgid")) is not int
        ):
            raise ValueError(f"fit attempt {attempt_id} owner receipt does not bind command/output")
        gpu_key = str(owner.get("gpu_key"))
        device = devices.get(gpu_key)
        if not isinstance(device, Mapping) or normalize_gpu_uuid(device.get("uuid")) != (
            normalize_gpu_uuid(_owner_gpu_uuid(owner, stage, attempt_id, review_path, packet))
        ):
            raise ValueError(f"fit attempt {attempt_id} owner uses an unadmitted GPU")
        output = output.resolve(strict=False)
        status = _fit_artifact_status(output, owner, attempt_id, stage, runtime_root)
        attempt: dict[str, object] = {
            "attempt_id": attempt_id,
            "stage": stage,
            "trial_number": owner.get("trial_number"),
            "status": status,
            "family": family,
            "anchor_id": owner.get("anchor_id", _anchor_id(stage, attempt_id)),
            "config_path": str(config_path),
            "config_sha256": config_sha,
            "config_identity_sha256": config_identity,
            "training_source_revision": revision,
            "training_source_sha256": identity["source_sha256"],
            "profile_receipt": {"path": str(native_path), "sha256": native_sha},
            "approved_profile_receipt": {
                "path": str(approved_path),
                "sha256": file_sha256(approved_path),
            },
            "profile_review_path": str(review_path),
            "profile_review_sha256": review_sha,
            "owner_receipt_path": str(owner_path),
            "owner_receipt_sha256": file_sha256(owner_path),
            "output_dir": str(output),
            "gpu_key": gpu_key,
        }
        attempt.update(_write_fit_observation(runtime_root, attempt, owner_path, output))
        attempts.append(attempt)
        owner_identity = {name: owner.get(name) for name in ("pid", "birth", "pgid")}
        if process_is_owned_and_live(owner_identity):
            reservations.append(
                {
                    "attempt_id": attempt_id,
                    "gpu_key": gpu_key,
                    "owner_receipt_path": str(owner_path),
                    "owner_receipt_sha256": file_sha256(owner_path),
                    "process": owner_identity,
                }
            )
    return attempts, profile_identities, reservations


def _validate_review_entry(
    stage: int,
    attempt_id: str,
    review: Mapping[str, object],
    config_sha: str,
    native_sha: str,
) -> None:
    """Require a successful independent review for these exact profile and config bytes."""
    if stage == 1 and "-r6-" in attempt_id:
        identifier = attempt_id.removeprefix("preexisting-stage1-r6-")
        records = review.get("profiles")
        record = (
            next(
                (
                    item
                    for item in records
                    if isinstance(item, Mapping) and item.get("id") == identifier
                ),
                None,
            )
            if isinstance(records, list)
            else None
        )
        passed = (
            review.get("review_status")
            == "approved_for_exact_six_stage1_full_fit_profile_identities"
        )
        expected_config, expected_native = "config_raw_sha256", "native_profile_sha256"
    elif stage == 2:
        family = attempt_id.removeprefix("preexisting-stage2-").removesuffix("-polish-r1")
        records = review.get("profiles")
        record = (
            next(
                (
                    item
                    for item in records
                    if isinstance(item, Mapping) and item.get("family") == family
                ),
                None,
            )
            if isinstance(records, list)
            else None
        )
        passed = (
            review.get("review_status")
            == "approved_for_exact_two_stage2_full_fit_profile_identities"
        )
        expected_config, expected_native = "config_raw_sha256", "native_profile_sha256"
    else:
        family = attempt_id.removeprefix("preexisting-stage1-").removesuffix("-b256-r5")
        records = review.get("batch256_r5_profiles")
        record = (
            next(
                (
                    item
                    for item in records
                    if isinstance(item, Mapping) and item.get("family") == family
                ),
                None,
            )
            if isinstance(records, list)
            else None
        )
        checks = record.get("checks") if isinstance(record, Mapping) else None
        passed = (
            isinstance(record, Mapping)
            and record.get("status", "").startswith("pass_under_R3_host_cap")
            and record.get("checks_passed") == record.get("checks_total")
            and isinstance(checks, Mapping)
            and all(value is True for value in checks.values())
        )
        expected_config, expected_native = "config_sha256_raw", "native_profile_sha256"
    if (
        not passed
        or not isinstance(record, Mapping)
        or record.get(expected_config) != config_sha
        or record.get(expected_native) != native_sha
    ):
        raise ValueError(f"independent review does not approve exact profile bytes: {attempt_id}")


def _expected_config_sha(stage: int, attempt_id: str, review_path: Path, packet: Path) -> str:
    """Get the exact raw config digest from the independent profile review."""
    review = read_json(review_path)
    if stage == 1 and "-r6-" in attempt_id:
        key = attempt_id.removeprefix("preexisting-stage1-r6-")
        entry = next(item for item in review["profiles"] if item.get("id") == key)
        return str(entry["config_raw_sha256"])
    if stage == 2:
        family = attempt_id.removeprefix("preexisting-stage2-").removesuffix("-polish-r1")
        entry = next(item for item in review["profiles"] if item.get("family") == family)
        return str(entry["config_raw_sha256"])
    family = attempt_id.removeprefix("preexisting-stage1-").removesuffix("-b256-r5")
    prior = next(item for item in review["batch256_r5_profiles"] if item["family"] == family)
    return str(prior["config_sha256_raw"])


def _expected_source_sha(stage: int, attempt_id: str) -> str:
    if stage == 2:
        return SOURCE5B_SHA
    if "-r6-" in attempt_id:
        return SOURCE4_SHA
    return SOURCE3_SHA


def _owner_gpu_uuid(
    owner: Mapping[str, object], stage: int, attempt_id: str, review_path: Path, packet: Path
) -> str:
    """Resolve the exact owner GPU UUID from the reviewed identity receipt."""
    if stage == 1 and "-r6-" in attempt_id:
        key = attempt_id.removeprefix("preexisting-stage1-r6-")
        review = read_json(review_path)
        return str(next(item["gpu_uuid"] for item in review["profiles"] if item.get("id") == key))
    if stage == 2:
        family = attempt_id.removeprefix("preexisting-stage2-").removesuffix("-polish-r1")
        review = read_json(review_path)
        return str(
            next(item["gpu_uuid"] for item in review["profiles"] if item.get("family") == family)
        )
    return (
        str(owner.get("gpu_uuid", ""))
        if owner.get("gpu_uuid")
        else _template_gpu_uuid(attempt_id, packet)
    )


def _template_gpu_uuid(attempt_id: str, packet: Path) -> str:
    template = read_json(packet / "ADMISSION_TEMPLATE_R1.json")
    prior = next(
        item for item in template["preexisting_fit_attempts"] if item["attempt_id"] == attempt_id
    )
    profile = read_json(Path(str(prior["profile_receipt"]["path"])))
    runtime = profile["identity"]["runtime"]
    return str(runtime["devices"][0]["uuid"])


def _anchor_id(stage: int, attempt_id: str) -> str | None:
    if stage == 2:
        return attempt_id
    return attempt_id if "r6-" in attempt_id else None


def _fit_artifact_status(
    output: Path, owner: Mapping[str, object], attempt_id: str, stage: int, root: Path
) -> str:
    """Distinguish a live original owner from native terminal artifact completion."""
    if process_is_owned_and_live({key: owner.get(key) for key in ("pid", "birth", "pgid")}):
        return "running"
    status_path, model_path, index_path = (
        output / "terminal_status.json",
        output / "model.pt",
        output / "epoch_index.json",
    )
    if not all(
        path.is_file() and not path.is_symlink() for path in (status_path, model_path, index_path)
    ):
        return "terminal_unverified"
    status, index = read_json(status_path), read_json(index_path)
    index_sha = index.get("index_sha256")
    if not isinstance(index_sha, str):
        return "terminal_unverified"
    from wnn.training.sequence_epoch_prediction import validate_index

    try:
        validate_index(index, index_sha)
    except ValueError:
        return "terminal_unverified"
    if (
        status.get("status") != "completed"
        or status.get("completion_kind") != "max_epochs"
        or status.get("model_sha256") != file_sha256(model_path)
        or status.get("epoch_index_sha256") != index_sha
        or index.get("identity") != status.get("identity")
    ):
        return "terminal_unverified"
    return "artifacts_complete_exit_unknown"


def _write_fit_observation(
    root: Path, attempt: Mapping[str, object], owner_path: Path, output: Path
) -> dict[str, object]:
    """Write an immutable observation/terminal sidecar from current owner and native artifacts."""
    owner, status = (
        read_json(owner_path),
        _fit_artifact_status(
            output, read_json(owner_path), str(attempt["attempt_id"]), int(attempt["stage"]), root
        ),
    )
    receipt_dir = root / "receipts/multistage-r2-fit-observations"
    sidecar = receipt_dir / f"{attempt['attempt_id']}.{status}.json"
    value: dict[str, object] = {
        "schema": "wnn.multistage-external-fit-observation.v1",
        "attempt_id": attempt["attempt_id"],
        "stage": attempt["stage"],
        "status": status,
        "owner_receipt_path": str(owner_path),
        "owner_receipt_sha256": file_sha256(owner_path),
        "owner_process": {key: owner.get(key) for key in ("pid", "birth", "pgid")},
        "profile_receipt_sha256": attempt["profile_receipt"]["sha256"],
        "approved_profile_sha256": attempt["approved_profile_receipt"]["sha256"],
        "config_sha256": attempt["config_sha256"],
        "output_dir": str(output),
        "observed_at_utc": datetime.now(UTC).isoformat(),
    }
    if status == "artifacts_complete_exit_unknown":
        terminal_status = output / "terminal_status.json"
        model, epoch_index = output / "model.pt", output / "epoch_index.json"
        native = read_json(terminal_status)
        failures = [
            path
            for path in output.rglob("*")
            if path.is_file()
            and any(
                marker in path.name.lower() for marker in ("failure", "failures", "error", "errors")
            )
            and path.stat().st_size > 0
        ]
        value.update(
            {
                "terminal": {
                    "schema": "wnn.multistage-fit-terminal.v1",
                    "status": "artifacts_complete_exit_unknown",
                    "attempt_id": attempt["attempt_id"],
                    "owner_receipt_sha256": file_sha256(owner_path),
                    "exit_code": None,
                    "exit_observation": "outer_wrapper_returncode_not_persisted",
                    "failure_logs_empty": not failures,
                    "finished_at_epoch": terminal_status.stat().st_mtime,
                    "output_dir": str(output),
                    "global_updates": native.get("global_updates"),
                    "model_sha256": file_sha256(model),
                    "epoch_index_sha256": read_json(epoch_index).get("index_sha256"),
                    "epoch_index_file_sha256": file_sha256(epoch_index),
                    "terminal_status_sha256": file_sha256(terminal_status),
                    "native_terminal_status_mtime_ns": terminal_status.stat().st_mtime_ns,
                    "failure_artifact_paths": [str(path) for path in failures],
                }
            }
        )
    if sidecar.is_file():
        previous = read_json(sidecar)
        for field, current in value.items():
            if field != "observed_at_utc" and previous.get(field) != current:
                raise ValueError(f"fit observation changed after its first write: {sidecar}")
    else:
        _write_identical_or_new(sidecar, value)
    result = {"observation_path": str(sidecar), "observation_sha256": file_sha256(sidecar)}
    terminal = value.get("terminal")
    if isinstance(terminal, Mapping):
        terminal_path = receipt_dir / f"{attempt['attempt_id']}.terminal.json"
        _write_identical_or_new(terminal_path, terminal)
        result.update(
            {
                "terminal_receipt_path": str(terminal_path),
                "terminal_receipt_sha256": file_sha256(terminal_path),
            }
        )
    return result


def _profile_history_record(
    packet: Path, template: Mapping[str, object], identities: Sequence[str]
) -> dict[str, object]:
    """Bind the 22 consumed profile slots to old admission plus new exact native profiles."""
    return {
        "prior_profile_attempts": template.get("preexisting_profile_attempts"),
        "prior_template_sha256": file_sha256(packet / "ADMISSION_TEMPLATE_R1.json"),
        "new_r6_profile_attempts": 6,
        "new_stage2_profile_attempts": 2,
        "total_profile_attempts": 22,
        "successful_config_identity_sha256s": sorted(identities),
    }


def _small_profile_identities(runtime_root: Path) -> set[str]:
    """Verify and collect the six earlier source3 batch profiles already charged to the ledger."""
    identities: set[str] = set()
    for family in ("gru", "lstm"):
        for batch in (16, 32, 64):
            attempt = read_json(runtime_root / f"receipts/anchor-{family}-b{batch}-r4.json")
            profile_path = runtime_root / f"profiles/{family}-b{batch}-r4/profile_receipt.json"
            profile = read_json(profile_path)
            identity = profile.get("identity")
            if (
                attempt.get("status") != "completed"
                or attempt.get("exit_code") != 0
                or not isinstance(identity, Mapping)
                or profile.get("approved") is not False
                or identity.get("source_sha256") != SOURCE3_SHA
                or file_sha256(Path(str(attempt["config_path"]))) != attempt.get("config_sha256")
            ):
                raise ValueError(f"small source3 profile identity is not exact: {family} b{batch}")
            identities.add(str(identity["config_sha256"]))
    if len(identities) != 6:
        raise ValueError("the six historical source3 profile configs are not unique")
    return identities


def _first_profile_epoch(root: Path, template: Mapping[str, object]) -> float:
    """Bind the immutable first actual profile time from the original campaign receipt."""
    path = root / "receipts/first-gpu-profile-r2.json"
    value = read_json(path)
    epoch = value.get("first_gpu_profile_epoch")
    if not isinstance(epoch, (int, float)) or epoch != template.get("first_gpu_profile_epoch"):
        raise ValueError("first GPU profile epoch differs from the original campaign ledger")
    return float(epoch)


def _write_identical_or_new(path: Path, value: Mapping[str, object]) -> None:
    """Create a durable immutable JSON file or require an existing byte-decoded value match."""
    if path.is_symlink():
        raise ValueError(f"refusing symlink output: {path}")
    if path.exists():
        if read_json(path) != dict(value):
            raise FileExistsError(f"existing immutable artifact differs: {path}")
        return
    atomic_write_json(path, value, replace=False)
