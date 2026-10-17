#!/usr/bin/env python3
"""Prepare and preflight the portable four-arm F0 GPU recipe queue.

This script never launches a trainer, worker, GPU job, or exporter. ``--check``
is read-only. ``--prepare`` writes a pending queue only after every frozen input
and every planned output path has passed its exact identity gate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


CAMPAIGN = Path("research/campaign_2026-10-17/fast_gpu_queue_r1")
BLUEPRINT_PATH = CAMPAIGN / "recipe_blueprint.json"
QUEUE_DIR = CAMPAIGN / "runtime_queue"
PENDING_DIR = QUEUE_DIR / "pending"
ALLOWED_SPEC_PATH = QUEUE_DIR / "allowed_spec_manifest.json"
ADMISSION_PATH = QUEUE_DIR / "PENDING_QUEUE_ADMISSION_R1.json"
PORTABLE_MANIFEST_PATH = CAMPAIGN / "portable_data_manifest.json"
SEED4_CONTRACT_PATH = CAMPAIGN / "contracts/CPU_STUDENT_FRONTIER_F0_SEED4_B8_R1.json"
FOLD0_SPLIT_PATH = Path("research/campaign_2026-09-29/artifacts/splits/fold0.json")
R4_INPUT_BINDING_PATH = Path(
    "research/campaign_2026-10-15/student_shape_pair_f0_r4/INPUT_BINDING_R4.json"
)
R4_INPUT_BINDING_SHA256 = "5877a724351b1ec7b81e67a5f0b2d30316adb04585b7defd92b0ba6710b8beb7"

SOURCE_DOCUMENTS = {
    "warm_commands": (
        Path("research/campaign_2026-10-15/gru_parent_growth_r1/WARM_MECHANISM_COMMANDS_R3.json"),
        "16b8ae1a843b36b779219fb44ec57fc16b5a978dbc52e35703da9af2e2874a6a",
    ),
    "fit_authority": (
        Path("research/campaign_2026-10-15/gru_parent_growth_r1/ROOT_CGFW_FIT_DELIVERY_AUTHORITY_R1.json"),
        "8c3908ee88082dd1d4e619dbae1af2d8e6a28c426b5afbb5550fcb7965479dbe",
    ),
    "r4_inputs": (
        Path("research/campaign_2026-10-15/gru_parent_growth_r1/WARM_MECHANISM_PROFILE_CONTROLLER_PREP_R4.json"),
        "b5b2691568cbddf3b4a6e4399d5a6cf6b0a33ff39dc7f71b305269a38365754d",
    ),
    "r3_manifest": (
        Path("research/campaign_2026-10-15/gru_parent_growth_r1/TRAINER_SOURCE_MANIFEST_R3.json"),
        "5949c1cbd9fbef6383efd793aadaf29dc9fc53fd0c7df70058422da65b33b9a7",
    ),
    "c_export": (
        Path("research/campaign_2026-10-15/gru_parent_growth_r1/C_CP1_EXPORT_RESULT_R1.json"),
        "1df27786078fdb92e8adaba7f96df90dba3b40d5355964b612dafb74b535f9b5",
    ),
    "c_export_prelaunch": (
        Path("research/campaign_2026-10-15/gru_parent_growth_r1/C_CP1_EXPORT_PRELAUNCH_IDENTITY_R1.json"),
        "f1390227f56eb0cec41f6112ceeca7304542ffa8e5ae85fbfc83e3526499df14",
    ),
    "g_export": (
        Path("research/campaign_2026-10-15/gru_parent_growth_r1/G_EXPORT_TERMINAL_VALIDATION_R1.json"),
        "00ac15ed0f3e9e608c3e4a989e0cb559f83fbff4139aebe85478168a2efb076d",
    ),
    "c_gate": (
        Path("research/campaign_2026-10-15/gru_parent_growth_r1/C_FIT_TERMINAL_STATE_GATE_R1.json"),
        "836e05fa65290f2f0c1936de8ecddc3dcad0b2f473c0a47902cf1e9841f16878",
    ),
    "g_gate": (
        Path("research/campaign_2026-10-15/gru_parent_growth_r1/G_FIT_TERMINAL_STATE_GATE_R1.json"),
        "c83b1d47b92f9877f15ad83c2f556d6a97d898c8c93c6bace82ca70fd2ccf2b8",
    ),
    "parent_recipe": (
        Path("research/campaign_2026-10-15/student_quality_tradeoff_r1/frozen_parent_anchor_r1/S_PARENT_ANCHOR_RECIPE_R1.json"),
        "c772f15cfee219ff48dbfc7b99ecf956a51efc7442c671900ab1465e17d7d63d",
    ),
    "s_profile": (
        Path("research/campaign_2026-10-15/student_quality_tradeoff_r1/frozen_parent_anchor_r1/S_PROFILE_TERMINAL_RESULT_R1.json"),
        "f8b38be4c27f206b8530f7337cc66716592b0b2a39ab405ca90f070cc7b60a69",
    ),
    "frontier_contract": (
        Path("research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_SEED3_B8_R2_CONSUMER_CONTRACT.json"),
        "9476373fc3c32aea2ae48e9a2c48d211f0f152d13d086fa68efd1e8b1a02e768",
    ),
    "h512_sequence": (
        Path("research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_S3_H512L1_R6_SEQUENCE.json"),
        "2b44cb8dd21d47e4eca6d8913359ca84976e1789237eb0737f4f17629cebd4fc",
    ),
    "h256_sequence": (
        Path("research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_S3_H256L2_R3_SEQUENCE.json"),
        "609dbcd7b346cd1f65e0237c65d3bd6f6efa228950d49a67fdec17cd2ac22188",
    ),
    "target_diagnostic": (
        Path("research/campaign_2026-10-15/student_quality_tradeoff_r1/ALPHA0_FIT_EVAL_DIAGNOSTIC_R1.json"),
        "e9efb0aa3f7fc38ddf30a890e14ad1397c37fadada77a3f245f8178671454b05",
    ),
    "split_asset_binding": (
        Path("research/campaign_2026-10-15/student_shape_pair_f0_r1/INPUT_BINDING_R1.json"),
        "6d69c3e5c24c08d2c0684a10e81e320729b9f8612414f082b5ad2b51ae3f244c",
    ),
}

# Reconcile frozen manifests against the exact reviewed current source bytes.
WARM_DORMANT_DRIFT_PATH = "src/wnn/student_frontier.py"
H512_DRIFT_PATHS = {
    "scripts/train_rnn.py",
    "src/wnn/nn.py",
    "src/wnn/optim.py",
    "src/wnn/student_frontier.py",
}
REVIEWED_WARM_DORMANT_SHA256 = "6d74438a1f75facaf740efba18d20a3257ab32862a633073e36574eb54693249"
REVIEWED_H512_DRIFT_SHA256 = {
    "scripts/train_rnn.py": "c4bd341ebc0e1dcd5b6eec717332fa609c7947439b7bd38d77588a97106b1de8",
    "src/wnn/nn.py": "7fca0e21fc0da2a40e2a39bed069a871b404adf6a0e41a49dcf24ddcab878d48",
    "src/wnn/optim.py": "549831f55be352c46d740a46e73386d41db8a8c2247b4a6d3f646177bd3cdd97",
    "src/wnn/student_frontier.py": REVIEWED_WARM_DORMANT_SHA256,
}

EXPERIMENTS = [
    {
        "experiment_id": "E-PORTABLE4-PARENT-ANCHOR-F0-S3-R1",
        "hypothesis_id": "H-FROZEN-PARENT-ANCHOR-20261015",
        "name": "S: frozen-parent prediction anchor",
        "run": "portable4_f0_s3_s_r1",
        "epoch": 1,
        "source": "research/campaign_2026-10-15/student_quality_tradeoff_r1/frozen_parent_anchor_r1/S_PARENT_ANCHOR_RECIPE_R1.json",
        "source_sha256": "c772f15cfee219ff48dbfc7b99ecf956a51efc7442c671900ab1465e17d7d63d",
        "contrast": "Same H256x2 seed-3 two-epoch fit recipe as C; replace only clean-T3 cached teacher with the exact frozen parent prediction at scale 1.0. Primary control C@ep1; parent@ep3 is an ancestry reference, not a matched training control.",
        "references": ["warm_mechanism_f0_s3_c_r3@ep1", "c20261003_combo_s3_f0@ep3"],
        "reference_roles": ["primary matched clean-T3 control", "secondary source checkpoint; teacher and scale confounded"],
        "profile_evidence": "S_PROFILE_TERMINAL_RESULT_R1.json: PASS_RESOURCE_PROFILE_ONLY, 5 batches / 50 updates, no fit/checkpoint/export/grade; a new GPU resource profile is still required for this new run name.",
    },
    {
        "experiment_id": "E-PORTABLE4-GRU-GROWTH-F0-S3-R1",
        "hypothesis_id": "H-STUDENT-PARENT-GROWTH-20261015",
        "name": "W: H384x2 grown-parent child",
        "run": "portable4_f0_s3_w_r1",
        "epoch": 1,
        "source": "research/campaign_2026-10-15/gru_parent_growth_r1/ROOT_CGFW_FIT_DELIVERY_AUTHORITY_R1.json#fits.W",
        "source_sha256": "8c3908ee88082dd1d4e619dbae1af2d8e6a28c426b5afbb5550fcb7965479dbe",
        "contrast": "Use the reviewed function-preserving H256x2-to-H384x2 initializer, then the same two-epoch F0 seed-3 clean-T3 recipe as C. Compare only to existing C@ep1; growth initialization and width are the intended treatment.",
        "references": ["warm_mechanism_f0_s3_c_r3@ep1"],
        "reference_roles": ["primary matched H256x2 clean-T3 control"],
        "profile_evidence": "The 2026-10-15 C/G/F/W resource-profile packet records complete 50-update probes; this new exact run name still requires a new GPU profile.",
    },
    {
        "experiment_id": "E-PORTABLE4-FIELD-CLOCK-F0-S3-R1",
        "hypothesis_id": "H-138",
        "name": "F: per-field causal event-age clock",
        "run": "portable4_f0_s3_f_r1",
        "epoch": 1,
        "source": "research/campaign_2026-10-15/gru_parent_growth_r1/ROOT_CGFW_FIT_DELIVERY_AUTHORITY_R1.json#fits.F",
        "source_sha256": "8c3908ee88082dd1d4e619dbae1af2d8e6a28c426b5afbb5550fcb7965479dbe",
        "contrast": "Only research feature is field_age; primary comparison G@ep1 changes the clock representation from seven group clocks to per-field clocks, with same width/init/target/training budget. C@ep1 is secondary no-clock reference.",
        "references": ["warm_mechanism_f0_s3_g_r3@ep1", "warm_mechanism_f0_s3_c_r3@ep1"],
        "reference_roles": ["primary matched group_age control", "secondary no-added-clock control"],
        "profile_evidence": "Per-field input screen found 19-48 separate reset patterns versus seven group clocks over eight inputs; this is deterministic input representation evidence, not a predictive-quality result.",
    },
    {
        "experiment_id": "E-PORTABLE4-H512-SEED4-CONFIRM-F0-R1",
        "hypothesis_id": "H-STUDENT-LADDER-20261014",
        "name": "H512x1 seed-4 confirmation vs H256x2 seed-4",
        "run": "portable4_f0_s4_h512_l1_r1",
        "epoch": 0,
        "paired_run": "portable4_f0_s4_h256_l2_r1",
        "paired_epoch": 0,
        "source": "research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_S3_H512L1_R6_SEQUENCE.json + H256L2_R3_SEQUENCE.json",
        "source_sha256": "2b44cb8dd21d47e4eca6d8913359ca84976e1789237eb0737f4f17629cebd4fc,609dbcd7b346cd1f65e0237c65d3bd6f6efa228950d49a67fdec17cd2ac22188",
        "contrast": "Two cold-start fits, same seed 4, same T3 cache, common H256x2 seed-3 CP5 normalization buffers only, B8/C2000 FP32, one epoch, CP0. No warm parent weights. Seed-4 sampler/order fields are regenerated from production EpochOrderSampler before materialization.",
        "references": ["portable4_f0_s4_h256_l2_r1@ep0"],
        "reference_roles": ["primary matched seed-4 H256x2 cold-start control"],
        "profile_evidence": "The seed-3 H512x1 result was a first-screen candidate; this experiment confirms shape contrast on a new seed, not a fresh fold or public result.",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def read_json(root: Path, path: Path) -> dict[str, Any]:
    value = json.loads((root / path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_tracked_json(root: Path, path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    """Read a tracked source through Git when the research artifact is ignored locally."""
    completed = subprocess.run(
        ["git", "show", f"HEAD:{path.as_posix()}"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    actual_sha256 = hashlib.sha256(completed.stdout).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"tracked input-binding source drift: {path}: expected {expected_sha256}, got {actual_sha256}"
        )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise ValueError(f"expected a tracked JSON object: {path}")
    return value, actual_sha256


def set_option(argv: list[str], key: str, value: str) -> None:
    try:
        argv[argv.index(key) + 1] = value
    except (ValueError, IndexError) as error:
        raise ValueError(f"reviewed command lacks {key}") from error


def trainer_args_from_worker(worker_argv: list[str], run: str) -> list[str]:
    marker = "scripts/train_rnn.py"
    try:
        script_index = next(i for i, token in enumerate(worker_argv) if token.endswith(marker))
    except StopIteration as error:
        raise ValueError("reviewed worker command has no train_rnn.py") from error
    argv = [marker, *worker_argv[script_index + 1 :]]
    set_option(argv, "--name", run)
    return argv


def trainer_args_from_sequence(command: list[str], run: str, *, contract: str, stage: str) -> list[str]:
    marker = "scripts/train_rnn.py"
    try:
        script_index = next(i for i, token in enumerate(command) if token.endswith(marker))
    except StopIteration as error:
        raise ValueError("reviewed sequence command has no train_rnn.py") from error
    argv = [marker, *command[script_index + 1 :]]
    set_option(argv, "--name", run)
    set_option(argv, "--seed", "4")
    set_option(argv, "--split-manifest", "research/campaign_2026-09-29/artifacts/splits/fold0.json")
    set_option(argv, "--student-frontier-contract", contract)
    set_option(argv, "--student-frontier-stage", stage)
    return argv


def source_args(root: Path) -> dict[str, Any]:
    docs = {name: read_json(root, path) for name, (path, _) in SOURCE_DOCUMENTS.items()}
    input_binding, input_binding_sha256 = read_tracked_json(
        root, R4_INPUT_BINDING_PATH, R4_INPUT_BINDING_SHA256
    )
    docs["r4_input_binding"] = input_binding
    warm = docs["warm_commands"]
    authority = docs["fit_authority"]
    parent_recipe = docs["parent_recipe"]
    h512 = docs["h512_sequence"]
    h256 = docs["h256_sequence"]
    commands = {
        "E-PORTABLE4-PARENT-ANCHOR-F0-S3-R1": [
            *parent_recipe["trainer_argv"]
        ],
        "E-PORTABLE4-GRU-GROWTH-F0-S3-R1": trainer_args_from_worker(
            authority["fits"]["W"]["worker_argv"], "portable4_f0_s3_w_r1"
        ),
        "E-PORTABLE4-FIELD-CLOCK-F0-S3-R1": trainer_args_from_worker(
            authority["fits"]["F"]["worker_argv"], "portable4_f0_s3_f_r1"
        ),
        "E-PORTABLE4-H512-SEED4-CONFIRM-F0-R1": trainer_args_from_sequence(
            h512["steps"][0]["expected_command"],
            "portable4_f0_s4_h512_l1_r1",
            contract=SEED4_CONTRACT_PATH.as_posix(),
            stage="f0_seed4_fit",
        ),
    }
    set_option(commands["E-PORTABLE4-PARENT-ANCHOR-F0-S3-R1"], "--name", "portable4_f0_s3_s_r1")
    h256_args = trainer_args_from_sequence(
        h256["steps"][0]["expected_command"],
        "portable4_f0_s4_h256_l2_r1",
        contract=SEED4_CONTRACT_PATH.as_posix(),
        stage="f0_seed4_fit",
    )
    # The confirmation is a paired two-fit experiment; preserve both exact argv vectors.
    return {
        "commands": commands,
        "paired_h256_command": h256_args,
        "docs": docs,
        "tracked_source_hashes": {R4_INPUT_BINDING_PATH.as_posix(): input_binding_sha256},
    }


def add_pin(
    pins: dict[str, dict[str, Any]],
    path: str,
    digest: str | None,
    *,
    size: int | None = None,
    identity: dict[str, Any] | None = None,
    semantic_contract: dict[str, Any] | None = None,
    group: str,
) -> None:
    normalized = path.replace("\\", "/")
    if not normalized or Path(normalized).is_absolute() or ".." in Path(normalized).parts:
        raise ValueError(f"non-portable or unbound input path/hash: {path}")
    if digest is not None and (
        len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"invalid SHA256 for {path}")
    if digest is None and semantic_contract is None:
        raise ValueError(f"missing content identity for {path}")
    item: dict[str, Any] = {"path": normalized, "group": group}
    if digest is not None:
        item["sha256"] = digest
    if size is not None:
        item["size_bytes"] = size
    if identity is not None:
        item["trusted_receipt_identity"] = identity
    if semantic_contract is not None:
        item["semantic_contract"] = semantic_contract
    prior = pins.get(normalized)
    if prior is not None:
        prior_hash, item_hash = prior.get("sha256"), item.get("sha256")
        if prior_hash is not None and item_hash is not None and prior_hash != item_hash:
            raise ValueError(f"conflicting exact identities for input: {normalized}")
        if prior_hash is None and item_hash is not None:
            prior["sha256"] = item_hash
        for key in ("size_bytes", "trusted_receipt_identity", "semantic_contract"):
            if key in prior and key in item and prior[key] != item[key]:
                raise ValueError(f"conflicting {key} bindings for input: {normalized}")
            if key not in prior and key in item:
                prior[key] = item[key]
        prior["groups"] = sorted(set(prior.get("groups", [prior.pop("group", group)]) + [group]))
        return
    pins[normalized] = item


def collect_pins(root: Path, docs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pins: dict[str, dict[str, Any]] = {}
    r4 = docs["r4_inputs"]["fit_and_input_pins"]
    add_pin(pins, r4["split_manifest"]["path"], r4["split_manifest"]["sha256"], group="split")
    for source, spec in r4["dataset_body_hash_and_identity_receipt"]["train_and_valid"].items():
        add_pin(pins, spec["path"], spec["sha256"], size=spec["bytes"], identity=spec["file_identity"], group="dataset")
    # These two caches were explicitly excluded by the earlier shape-pair R1
    # packet. The tracked R4 input binding supplies their exact body hashes.
    for item in docs["r4_input_binding"]["external_mounts"]:
        if item["logical_path"] in {"cache/train_targets.npz", "cache/valid_targets.npz"}:
            add_pin(
                pins,
                item["logical_path"],
                item["sha256"],
                size=item["size"],
                group="scoring_and_target_index_cache",
            )
    parent = r4["parent_checkpoint"]
    add_pin(pins, parent["path"], parent["sha256"], group="initializer_and_ancestry")
    add_pin(pins, parent["provenance_path"], parent["provenance_sha256"], group="initializer_and_ancestry")
    grown = r4["grown_initializer"]
    add_pin(pins, grown["path"], grown["sha256"], group="grown_initializer")
    add_pin(pins, grown["provenance_path"], grown["provenance_sha256"], group="grown_initializer")
    prop = r4["propensity"]
    add_pin(pins, prop["model_path"], prop["model_sha256"], group="propensity")
    prop_contract = docs["frontier_contract"]["propensity"]
    add_pin(
        pins, f'artifacts/runs/{prop_contract["run"]}/provenance.json',
        prop_contract["provenance_sha256"], group="propensity",
    )
    for split_name, sidecar_key in (("train", "train_sidecar"), ("valid", "valid_sidecar")):
        sidecar = prop[sidecar_key]
        add_pin(pins, sidecar["path"], sidecar["sha256"], group="propensity")
        arr_path = prop["memory_mapped_arrays"][0 if split_name == "train" else 1]
        arr_hash = prop["memory_mapped_arrays_sha256"][split_name] if "memory_mapped_arrays_sha256" in prop else None
        # The frozen R4 pin names hashes for these arrays in the seed-3 consumer contract.
        contract = docs["frontier_contract"]
        cache = contract["propensity"]["caches"][split_name]
        if arr_hash is None:
            arr_hash = cache["sha256"]
        add_pin(pins, arr_path, arr_hash, size=cache["bytes"], group="propensity")
    cache = r4["teacher_cache"]
    add_pin(pins, f'{cache["root"]}/metadata.json', cache["metadata_sha256"], group="teacher_cache")
    for key, spec in cache["files"].items():
        add_pin(pins, f'{cache["root"]}/{spec["path"]}', spec["sha256"], size=spec["bytes"], group="teacher_cache")
    for teacher in docs["frontier_contract"]["cache"]["teachers"]:
        add_pin(pins, teacher["checkpoint"], teacher["checkpoint_sha256"], group="teacher_ancestry")
        add_pin(pins, teacher["provenance"], teacher["provenance_sha256"], group="teacher_ancestry")
    # S uses direct parent predictions; they are separate from the clean-T3 cache.
    for prediction in docs["parent_recipe"]["prediction_artifacts"]:
        for key, digest_key in (("prediction_path", "prediction_sha256"), ("sidecar_path", "sidecar_sha256")):
            raw_path = prediction[key]
            relative = raw_path[raw_path.find("cache/") :]
            add_pin(pins, relative, prediction[digest_key], size=prediction["prediction_bytes"] if key == "prediction_path" else prediction["sidecar_bytes"], group="parent_prediction")
    # Existing CP1 matched controls C and G; never fall back to archived CP5.
    c_export, g_export = docs["c_export"], docs["g_export"]
    g_gate = docs["g_gate"]
    c_run = c_export["run"]
    c_prelaunch = docs["c_export_prelaunch"]
    add_pin(
        pins,
        f"artifacts/runs/{c_run}/model_ep1.pt",
        c_export["checkpoint_sha256"],
        group="C_CP1_control",
    )
    add_pin(
        pins,
        f"artifacts/runs/{c_run}/provenance.json",
        None,
        semantic_contract={
            "run": c_run,
            "checkpoint_path": c_prelaunch["checkpoint_path"],
            "checkpoint_sha256": c_prelaunch["checkpoint_sha256"],
            "split_manifest_sha256": c_prelaunch["split_manifest_sha256"],
        },
        group="C_CP1_control_provenance_semantic_receipt",
    )
    add_pin(pins, c_export["prediction_path"], c_export["prediction_sha256"], size=c_export["prediction_bytes"], group="C_CP1_control")
    add_pin(pins, c_export["sidecar_path"], c_export["sidecar_sha256"], size=c_export["sidecar_bytes"], group="C_CP1_control")
    add_pin(pins, g_export["checkpoint_path"], g_export["checkpoint_sha256"], group="G_CP1_control")
    add_pin(pins, g_gate["provenance_path"], g_gate["provenance_sha256"], group="G_CP1_control")
    add_pin(pins, g_export["prediction_path"], g_export["prediction_sha256"], size=g_export["prediction_size_bytes"], group="G_CP1_control")
    add_pin(pins, g_export["metadata_path"], g_export["metadata_sha256"], group="G_CP1_control")
    # The H512/H256 pair must use only the reviewed common normalization buffers.
    norm = docs["frontier_contract"]["normalization_source"]
    norm_run = norm["run"]
    add_pin(pins, f'artifacts/runs/{norm_run}/{norm["checkpoint_file"]}', norm["checkpoint_sha256"], group="common_normalization_buffers")
    add_pin(pins, f'artifacts/runs/{norm_run}/provenance.json', norm["provenance_sha256"], group="common_normalization_buffers")
    target_info = docs["target_diagnostic"]["targets"]
    add_pin(pins, target_info["path"], target_info["sha256"], group="grading_target")
    split_asset = docs["split_asset_binding"]
    for item in split_asset["external_mounts"]:
        if item["logical_path"] in {"cache/chains.npz", "cache/chain_folds.npz"}:
            add_pin(pins, item["logical_path"], item["sha256"], size=item["size"], group="split_graph")
    for item in docs["r4_input_binding"]["external_mounts"]:
        if item["logical_path"] == FOLD0_SPLIT_PATH.as_posix():
            add_pin(
                pins,
                item["logical_path"],
                item["sha256"],
                size=item["size"],
                group="split_manifest",
            )
    return dict(sorted(pins.items()))


def seed4_contract_template(reviewed: dict[str, Any]) -> dict[str, Any]:
    """Port the reviewed seed-3 consumer recipe to the paired seed-4 recipe."""
    contract = copy.deepcopy(reviewed)
    contract["id"] = "cpu-student-frontier-f0-seed4-b8-r1"
    contract["created_utc"] = datetime.now(UTC).isoformat()
    contract["owner"] = "root"
    contract["objective"] = "Matched F0 seed-4 H512x1 versus H256x2 cold-start, one-epoch confirmation."
    contract["source_root"] = "."
    contract["recipe"]["train_seqs_argument"] = 10607
    contract["recipe"]["optimizer"]["gradient_stats"] = True
    fit_stage = copy.deepcopy(reviewed["stages"]["f0_seed3_fit"])
    fit_stage.update(
        {
            "kind": "fit",
            "epochs": 1,
            "checkpoint_epoch": 0,
            "expected_updates": 14920,
            "max_batches": 0,
            "profile_steps": 0,
            "profile_warmup": 5,
        }
    )
    fit_stage["seed"] = 4
    fit_stage["run_name_prefix"] = "portable4_f0_s4_"
    contract["stages"] = {"f0_seed4_fit": fit_stage}
    contract.pop("sample_order", None)
    return contract


def make_blueprint(root: Path) -> dict[str, Any]:
    loaded = source_args(root)
    source_hashes: dict[str, str] = dict(loaded["tracked_source_hashes"])
    for _, (path, expected) in SOURCE_DOCUMENTS.items():
        actual = sha256_file(root / path)
        if expected and actual != expected:
            raise ValueError(f"source document drift: {path}: expected {expected}, got {actual}")
        source_hashes[path.as_posix()] = actual

    pins = collect_pins(root, loaded["docs"])
    r3_path, _ = SOURCE_DOCUMENTS["r3_manifest"]
    warm_manifest = read_json(root, r3_path)["files"]
    warm_current = {rel: sha256_file(root / rel) for rel in warm_manifest}
    warm_mismatches = {
        rel: {"frozen_sha256": warm_manifest[rel], "current_sha256": warm_current[rel]}
        for rel in warm_manifest
        if warm_current[rel] != warm_manifest[rel]
    }
    expected_warm_difference = {
        WARM_DORMANT_DRIFT_PATH: {
            "frozen_sha256": warm_manifest[WARM_DORMANT_DRIFT_PATH],
            "current_sha256": REVIEWED_WARM_DORMANT_SHA256,
        }
    }
    if warm_mismatches != expected_warm_difference:
        raise ValueError(f"warm source closure changed outside its reviewed dormant module: {warm_mismatches}")

    hcontract = loaded["docs"]["frontier_contract"]
    h512_current = {
        rel: sha256_file(root / rel) for rel in hcontract["source_sha256"]
    }
    h512_mismatches = {
        rel: {"frozen_sha256": hcontract["source_sha256"][rel], "current_sha256": h512_current[rel]}
        for rel in hcontract["source_sha256"]
        if h512_current[rel] != hcontract["source_sha256"][rel]
    }
    expected_h512_differences = {
        rel: {
            "frozen_sha256": hcontract["source_sha256"][rel],
            "current_sha256": REVIEWED_H512_DRIFT_SHA256[rel],
        }
        for rel in H512_DRIFT_PATHS
    }
    if h512_mismatches != expected_h512_differences:
        raise ValueError(f"H512 source drift differs from the reviewed four-file set: {h512_mismatches}")

    experiments = []
    for base in EXPERIMENTS:
        experiment = copy.deepcopy(base)
        experiment["trainer_argv"] = loaded["commands"][experiment["experiment_id"]]
        if experiment["experiment_id"] == "E-PORTABLE4-H512-SEED4-CONFIRM-F0-R1":
            experiment["paired_command"] = loaded["paired_h256_command"]
        experiments.append(experiment)
    return {
        "schema": "wnn.portable-four-gpu-queue-blueprint.r1",
        "status": "RECIPES_PREPARED_HOLD_NO_FIT_NO_LAUNCH",
        "campaign": CAMPAIGN.as_posix(),
        "portable_bindings": {"root": "--root", "python": "--python"},
        "source_documents_sha256": source_hashes,
        "source_closure": {
            "warm_manifest_path": r3_path.as_posix(),
            "warm_manifest_sha256": source_hashes[r3_path.as_posix()],
            "warm_source_manifest_count": len(warm_manifest),
            "warm_exact_match_count": len(warm_manifest) - len(warm_mismatches),
            "warm_reconciled_difference": warm_mismatches,
            "warm_call_path_scope": (
                "The reviewed S/W/F argv omit --student-frontier-contract. The only R3 closure difference "
                "is src/wnn/student_frontier.py; trainer call sites are guarded by that contract flag. "
                "All other manifest files, including trainer, research features, sampler, data/split readers, "
                "cache validation and optimizer, match their frozen R3 hashes."
            ),
            "frozen_source_revision": loaded["docs"]["fit_authority"]["source_commit"],
            "h512_consumer_contract": SOURCE_DOCUMENTS["frontier_contract"][0].as_posix(),
            "h512_source_differences": h512_mismatches,
            "h512_current_source_sha256": h512_current,
        },
        "seed4_contract_template": seed4_contract_template(hcontract),
        "split_sha256": hcontract["cache"]["split_manifest_sha256"],
        "inputs": list(pins.values()),
        "experiments": experiments,
        "prohibited_actions": [
            "no worker launch",
            "no trainer invocation",
            "no GPU fit",
            "no upload",
            "no scoring during preparation",
        ],
    }


def source_preflight(root: Path, blueprint: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    manifest_path, _ = SOURCE_DOCUMENTS["r3_manifest"]
    manifest = read_json(root, manifest_path)
    source_map = manifest.get("files")
    if not isinstance(source_map, dict):
        return [f"cannot interpret frozen source manifest: {manifest_path}"], {}
    closure = blueprint["source_closure"]
    for rel, expected in blueprint["source_documents_sha256"].items():
        if rel == R4_INPUT_BINDING_PATH.as_posix():
            _, observed = read_tracked_json(root, R4_INPUT_BINDING_PATH, R4_INPUT_BINDING_SHA256)
        else:
            source_path = root / rel
            if not source_path.is_file():
                errors.append(f"missing frozen source document: {rel}")
                continue
            observed = sha256_file(source_path)
        if observed != expected:
            errors.append(f"frozen source document changed after blueprint generation: {rel}")
    checked = 0
    mismatches: dict[str, dict[str, str]] = {}
    current: dict[str, str] = {}
    for rel, frozen in source_map.items():
        file_path = root / rel
        if not file_path.is_file():
            errors.append(f"missing source-closure file: {rel}")
            continue
        digest = sha256_file(file_path)
        checked += 1
        current[rel] = digest
        if digest != frozen:
            mismatches[rel] = {"frozen_sha256": frozen, "current_sha256": digest}
    if mismatches != closure["warm_reconciled_difference"]:
        errors.append(f"warm source closure differs from the reviewed reconciliation: {mismatches}")
    expected_warm_difference = {
        WARM_DORMANT_DRIFT_PATH: {
            "frozen_sha256": source_map[WARM_DORMANT_DRIFT_PATH],
            "current_sha256": REVIEWED_WARM_DORMANT_SHA256,
        }
    }
    if mismatches != expected_warm_difference:
        errors.append("warm source drift no longer matches its pinned reviewed content")
    if len(current) != len(source_map):
        errors.append("warm source closure is incomplete")

    hcontract = read_json(root, SOURCE_DOCUMENTS["frontier_contract"][0])
    h512_current: dict[str, str] = {}
    h512_mismatches: dict[str, dict[str, str]] = {}
    for rel, frozen in hcontract["source_sha256"].items():
        file_path = root / rel
        if not file_path.is_file():
            errors.append(f"missing H512 source file: {rel}")
            continue
        digest = sha256_file(file_path)
        h512_current[rel] = digest
        if digest != frozen:
            h512_mismatches[rel] = {"frozen_sha256": frozen, "current_sha256": digest}
    if h512_current != closure["h512_current_source_sha256"]:
        errors.append("H512 source changed after blueprint generation")
    if h512_mismatches != closure["h512_source_differences"]:
        errors.append("H512 consumer-contract source drift changed after blueprint generation")
    expected_h512_differences = {
        rel: {
            "frozen_sha256": hcontract["source_sha256"][rel],
            "current_sha256": REVIEWED_H512_DRIFT_SHA256[rel],
        }
        for rel in H512_DRIFT_PATHS
    }
    if h512_mismatches != expected_h512_differences:
        errors.append("H512 source drift no longer matches its pinned reviewed content")
    report = {
        "warm_source_manifest_count": len(source_map),
        "warm_source_checked_count": checked,
        "warm_source_exact_match_count": checked - len(mismatches),
        "warm_source_reconciled_difference": mismatches,
        "h512_contract_source_files": len(hcontract["source_sha256"]),
        "h512_current_source_sha256": h512_current,
        "h512_source_differences": h512_mismatches,
    }
    return errors, report


def verify_provenance(root: Path, item: dict[str, Any], split: dict[str, Any]) -> str:
    """Bind C's otherwise-unhashed historical provenance to its receipt and fit membership."""
    path = root / item["path"]
    provenance = read_json(root, Path(item["path"]))
    contract = item["semantic_contract"]
    if provenance.get("args", {}).get("name") != contract["run"]:
        raise ValueError("C provenance run name differs from the frozen export receipt")
    if provenance.get("split_manifest_sha256") != contract["split_manifest_sha256"]:
        raise ValueError("C provenance split identity differs from the frozen export receipt")
    if provenance.get("checkpoint_sha256", {}).get("model_ep1.pt") != contract["checkpoint_sha256"]:
        raise ValueError("C provenance does not bind the frozen CP1 checkpoint bytes")
    if provenance.get("args", {}).get("split_manifest_sha256") != contract["split_manifest_sha256"]:
        raise ValueError("C provenance args do not bind the frozen split SHA")
    fit = provenance.get("fit_row_groups")
    if not isinstance(fit, dict):
        raise ValueError("C provenance lacks direct fit membership")
    train_groups = fit.get("train")
    valid_groups = fit.get("valid")
    frozen_train = split.get("fit", {}).get("train")
    frozen_valid = split.get("fit", {}).get("valid")
    if not all(isinstance(values, list) for values in (train_groups, valid_groups, frozen_train, frozen_valid)):
        raise ValueError("C or split provenance has malformed fit membership")
    if len(train_groups) != len(set(train_groups)) or set(train_groups) != set(frozen_train):
        raise ValueError("C provenance train fit groups differ from the frozen split membership")
    if valid_groups != frozen_valid:
        raise ValueError("C provenance valid fit group order differs from the frozen split membership")
    args = provenance["args"]
    expected_args = {"seed": 3, "train_source": "train+valid", "hidden": 256, "layers": 2, "checkpoint_epoch": 1}
    for key, expected in expected_args.items():
        if args.get(key) != expected:
            raise ValueError(f"C provenance {key} differs from the reviewed C CP1 recipe")
    return sha256_file(path)


def preflight(root: Path, python: Path, blueprint: dict[str, Any]) -> dict[str, Any]:
    errors, source_report = source_preflight(root, blueprint)
    missing: list[str] = []
    mismatched: list[dict[str, str]] = []
    verified: list[dict[str, Any]] = []
    split_path = root / FOLD0_SPLIT_PATH
    split_doc = read_json(root, FOLD0_SPLIT_PATH) if split_path.is_file() else None
    resolved_hashes: dict[str, str] = {}
    for item in blueprint["inputs"]:
        path = root / item["path"]
        if not path.is_file():
            missing.append(item["path"])
            continue
        stat = path.stat()
        if item.get("size_bytes") is not None and stat.st_size != item["size_bytes"]:
            mismatched.append({"path": item["path"], "expected_size": str(item["size_bytes"]), "actual_size": str(stat.st_size)})
            continue
        identity = item.get("trusted_receipt_identity")
        actual_identity = {
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }
        digest = item.get("sha256")
        identity_reused = bool(identity and actual_identity == identity)
        if item.get("semantic_contract") is not None:
            if split_doc is None:
                errors.append(f"cannot semantically bind historical provenance without frozen split: {item['path']}")
                continue
            try:
                digest = verify_provenance(root, item, split_doc)
            except (OSError, ValueError, KeyError, TypeError) as error:
                mismatched.append({"path": item["path"], "semantic_error": str(error)})
                continue
        elif not identity_reused:
            actual_hash = sha256_file(path)
            if digest is not None and actual_hash != digest:
                mismatched.append({"path": item["path"], "expected_sha256": digest, "actual_sha256": actual_hash})
                continue
            digest = actual_hash
        if digest is None:
            errors.append(f"input lacks an exact or semantically derived identity: {item['path']}")
            continue
        resolved_hashes[item["path"]] = digest
        verified.append({
            "path": item["path"],
            "sha256": digest,
            "verification": (
                "historical byte hash unavailable; run/checkpoint/split/frozen fit membership validated"
                if item.get("semantic_contract") is not None
                else "trusted immutable filesystem identity" if identity_reused
                else "full content SHA256"
            ),
        })
    if not python.is_file() or not os.access(python, os.X_OK):
        errors.append(f"bound Python interpreter is missing or not executable: {python}")

    output_collisions: list[str] = []
    run_names: set[str] = set()
    for experiment in blueprint["experiments"]:
        names = [experiment["run"]]
        if "paired_run" in experiment:
            names.append(experiment["paired_run"])
        for name in names:
            if name in run_names:
                errors.append(f"duplicate planned fit run name: {name}")
            run_names.add(name)
            output_collisions.extend(
                relative.as_posix()
                for relative in (
                    Path("artifacts/runs") / name,
                    Path("cache") / f"pred_{name}_valid.npy",
                    Path("cache") / f"pred_{name}_valid.json",
                )
                if (root / relative).exists()
            )
        for suffix in ("grades", "receipts"):
            relative = CAMPAIGN / suffix / f"{experiment['experiment_id']}.json"
            if (root / relative).exists():
                output_collisions.append(relative.as_posix())
    for relative in (
        BLUEPRINT_PATH,
        PORTABLE_MANIFEST_PATH,
        SEED4_CONTRACT_PATH,
        ALLOWED_SPEC_PATH,
        ADMISSION_PATH,
    ):
        if (root / relative).exists():
            output_collisions.append(relative.as_posix())
    for experiment in blueprint["experiments"]:
        queue_name = experiment["run"] if "paired_run" not in experiment else "portable4_h512_seed4_pair_f0_r1"
        relative = PENDING_DIR / f"{queue_name}.json"
        if (root / relative).exists():
            output_collisions.append(relative.as_posix())
    if output_collisions:
        errors.append("planned output exists; refusing overwrite: " + ", ".join(sorted(set(output_collisions))))
    if missing:
        errors.append(f"{len(missing)} required exact input files are missing")
    if mismatched:
        errors.append(f"{len(mismatched)} required input identities do not match")
    return {
        "ok": not errors,
        "status": "READY_FOR_PENDING_MATERIALIZATION_ONLY" if not errors else "HOLD_MISSING_OR_MISMATCHED_BINDINGS",
        "errors": errors,
        "missing_files": missing,
        "mismatched_files": mismatched,
        "verified_inputs": verified,
        "resolved_input_sha256": resolved_hashes,
        "historical_provenance_limitations": [
            "C CP1 export receipt pins the checkpoint and split but provides no provenance.json SHA256; preparation validates its run, checkpoint, split and complete frozen fit membership, then records the observed provenance SHA."
        ],
        "source_reconciliation": source_report,
        "new_run_output_collisions": sorted(set(output_collisions)),
        "trainer_or_worker_started": False,
        "gpu_touched": False,
    }


def make_portable_data_manifest(
    root: Path, blueprint: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    split = read_json(root, FOLD0_SPLIT_PATH)
    sources = split.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("frozen fold-0 manifest lacks source identities")
    resolved = report["resolved_input_sha256"]
    required = {
        "datasets/train.parquet",
        "datasets/valid.parquet",
        "cache/chains.npz",
        "cache/chain_folds.npz",
        "cache/train_targets.npz",
        "cache/valid_targets.npz",
    }
    if set(sources) != required:
        raise ValueError(f"frozen split source list differs from the six required portable inputs: {sorted(sources)}")
    registry = {"version": 1, "sources": {}}
    for name, identity in sorted(sources.items()):
        if name not in resolved:
            raise ValueError(f"portable source lacks an input identity: {name}")
        registry["sources"][name] = {
            "original_identity": identity,
            "sha256": resolved[name],
        }
    return registry


def seed4_sample_order(
    root: Path,
    python: Path,
    portable_manifest: dict[str, Any],
    seed4_contract: dict[str, Any],
) -> dict[str, Any]:
    """Regenerate the exact seed-4 fit order using production sampler validators on CPU."""
    with tempfile.TemporaryDirectory(prefix=".portable4-preflight-", dir=root) as temp_dir:
        manifest_path = Path(temp_dir) / "portable_data_manifest.json"
        manifest_path.write_bytes(canonical_bytes(portable_manifest))
        code = r'''import json, os
from argparse import Namespace
from pathlib import Path
import numpy as np
from wnn.splits import load_campaign_split
from wnn.student_frontier import validate_sampling
from wnn.student_frontier_order import (
    seeded_epoch_positions, source_qualified_members, positions_sha256, members_sha256,
)
from wnn.run_identity import group_identity
from wnn.training_sampling import EpochOrderSampler

root = Path(os.environ["WNN_PROJECT_ROOT"])
contract = json.loads(os.environ["PORTABLE4_CONTRACT_JSON"])
split = load_campaign_split(root / contract["split"]["path"], root)
seed = 4
train = np.random.default_rng(seed).permutation(np.asarray(split.train_groups, dtype=np.int64))[:contract["recipe"]["train_seqs_argument"]]
valid = np.asarray(split.valid_groups, dtype=np.int64)
batch = contract["recipe"]["batch_size"]
epoch = 0
positions = seeded_epoch_positions(len(train) + len(valid), seed, batch, epoch)
members = source_qualified_members(train, valid, positions)
exposed = (len(positions) // batch) * batch
fit_members = members[:exposed]
dropped_members = members[exposed:]
def batch_digest(rows):
    encoded = "\n".join(f"{i // batch}:{source}:{group}" for i,(source,group) in enumerate(rows)).encode("ascii")
    import hashlib
    return hashlib.sha256(encoded).hexdigest()
sample = {
    "batch_size": batch,
    "seed": seed,
    "epoch": epoch,
    "fit_groups_before_drop_last": len(positions),
    "exposed_sequence_count": exposed,
    "dropped_sequence_count": len(dropped_members),
    "batches_per_epoch": len(positions) // batch,
    "chunks_per_sequence": contract["recipe"]["sequence_rows"] // contract["recipe"]["chunk_rows"],
    "optimizer_updates_per_epoch": (len(positions) // batch) * contract["recipe"]["sequence_rows"] // contract["recipe"]["chunk_rows"],
    "train_group_order_sha256": group_identity(train)["ordered_int64_le_sha256"],
    "valid_group_order_sha256": group_identity(valid)["ordered_int64_le_sha256"],
    "sampler_positions_sha256": positions_sha256(positions),
    "sampler_source_rows_sha256": members_sha256(members),
    "retained_source_rows_sha256": members_sha256(fit_members),
    "batch_membership_sha256": batch_digest(fit_members),
    "dropped_tail_positions": positions[exposed:],
    "dropped_tail_members": [list(row) for row in dropped_members],
    "same_order_and_omissions_across_shapes": True,
}
contract["sample_order"] = sample
stage = contract["stages"]["f0_seed4_fit"]
recipe = contract["recipe"]
class Loader:
    batch_size = recipe["batch_size"]
    drop_last = True
    num_workers = recipe["loader"]["workers"]
    pin_memory = recipe["loader"]["pin_memory"]
    prefetch_factor = recipe["loader"]["prefetch_factor"]
    persistent_workers = False
    dataset = range(len(train) + len(valid))
    sampler = EpochOrderSampler(len(dataset), seed, batch_size=batch_size, drop_last=True)
observed = validate_sampling(
    Namespace(batch_order="seeded", valid_repeat=1), contract, stage,
    split, train, valid, Loader(),
)
print(json.dumps(observed, sort_keys=True))
'''
        env = dict(os.environ)
        env.update(
            {
                "PYTHONPATH": str(root / "src"),
                "WNN_PROJECT_ROOT": str(root),
                "WNN_PORTABLE_DATA_MANIFEST": str(manifest_path),
                "PORTABLE4_CONTRACT_JSON": canonical_bytes(seed4_contract).decode("utf-8"),
            }
        )
        completed = subprocess.run(
            [str(python), "-c", code],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
    observed = json.loads(completed.stdout)
    if observed["seed"] != 4 or observed["epoch"] != 0 or observed["optimizer_updates_per_epoch"] != 14920:
        raise ValueError("production sampler does not produce the reviewed seed-4 fixed update budget")
    return observed


def make_seed4_contract(blueprint: dict[str, Any]) -> dict[str, Any]:
    source = copy.deepcopy(blueprint["seed4_contract_template"])
    source["source_root"] = "."
    source["split"]["path"] = FOLD0_SPLIT_PATH.as_posix()
    source["split"]["sha256"] = blueprint["split_sha256"]
    source["cache"]["root"] = "cache/clean_t3_f0_h390_t3r5_a9_r2"
    source["normalization_source"]["run"] = "clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1"
    source["source_sha256"] = copy.deepcopy(blueprint["source_closure"]["h512_current_source_sha256"])
    return source


def make_source_path_bindings(root: Path, contract: dict[str, Any]) -> dict[str, str]:
    """Relocate old absolute metadata ancestry without rewriting the pinned cache."""
    metadata_path = Path(contract["cache"]["root"]) / "metadata.json"
    if sha256_file(root / metadata_path) != contract["cache"]["metadata_sha256"]:
        raise ValueError("teacher metadata changed after preflight")
    metadata = read_json(root, metadata_path)
    records = metadata.get("sources")
    expected = contract["cache"]["teachers"]
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError("verified teacher metadata does not match the reviewed ancestry count")
    bindings: dict[str, str] = {}
    for record, teacher in zip(records, expected, strict=True):
        if record.get("run") != teacher["run"]:
            raise ValueError("verified teacher metadata run order differs from the consumer contract")
        for producer_key, contract_key in (
            ("checkpoint_path", "checkpoint"),
            ("provenance_path", "provenance"),
        ):
            original = record.get(producer_key)
            if not isinstance(original, str) or not Path(original).is_absolute():
                raise ValueError(f"teacher metadata lacks absolute original {producer_key}")
            target = teacher[contract_key]
            prior = bindings.get(original)
            if prior is not None and prior != target:
                raise ValueError("teacher metadata path maps to conflicting pinned payloads")
            bindings[original] = target
    return bindings


def _timeout_seconds(worker_argv: list[str]) -> int:
    try:
        timeout_index = next(
            index for index, token in enumerate(worker_argv) if Path(token).name == "timeout"
        )
        value = worker_argv[timeout_index + 3]
    except (StopIteration, IndexError) as error:
        raise ValueError("reviewed worker argv lacks a timeout command") from error
    multipliers = {"s": 1, "m": 60, "h": 3600}
    suffix = value[-1]
    if suffix not in multipliers or not value[:-1].isdigit():
        raise ValueError(f"unsupported reviewed timeout: {value}")
    return int(value[:-1]) * multipliers[suffix]


def _sequence_timeout(sequence: dict[str, Any], phase: str) -> int:
    try:
        step = next(item for item in sequence["steps"] if item.get("phase") == phase)
        command = step["launch_command"]
        index = command.index("--kill-after=10s")
        value = command[index + 1]
    except (KeyError, StopIteration, ValueError, IndexError, TypeError) as error:
        raise ValueError(f"reviewed sequence lacks timeout for phase {phase}") from error
    multipliers = {"s": 1, "m": 60, "h": 3600}
    suffix = value[-1]
    if suffix not in multipliers or not value[:-1].isdigit():
        raise ValueError(f"unsupported reviewed timeout for {phase}: {value}")
    return int(value[:-1]) * multipliers[suffix]


def _step(name: str, argv: list[str], *, timeout: int, env: dict[str, str], root: Path) -> dict[str, Any]:
    return {
        "name": name,
        "argv": argv,
        "timeout_seconds": timeout,
        "env": env,
        "cwd": str(root),
    }


def build_worker_specs(
    root: Path, python: Path, blueprint: dict[str, Any], portable_manifest_sha256: str
) -> dict[str, dict[str, Any]]:
    """Build the worker's actual pending run specs with absolute machine bindings."""
    root_s, python_s = str(root), str(python)
    portable_manifest = str(root / PORTABLE_MANIFEST_PATH)
    base_env = {
        "WNN_PROJECT_ROOT": root_s,
        "PYTHONPATH": str(root / "src"),
        "WNN_PORTABLE_DATA_MANIFEST": portable_manifest,
    }
    warm_env = base_env | {
        "WNN_LOADER_RAM_GB": "4",
        "WNN_GPU_MEMORY_BYTES": "4300000000",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    frontier_env = base_env | {
        "WNN_LOADER_WORKERS": "0",
        "WNN_THREADS": "1",
        "WNN_GPU_MEMORY_BYTES": "3500000000",
    }
    docs = source_args(root)["docs"]
    authority = docs["fit_authority"]["fits"]
    sequence_docs = {
        "h512": docs["h512_sequence"],
        "h256": docs["h256_sequence"],
    }
    experiment_by_id = {item["experiment_id"]: item for item in blueprint["experiments"]}
    specs: dict[str, dict[str, Any]] = {}
    for priority, experiment in enumerate(blueprint["experiments"][:3], start=1):
        exp_id = experiment["experiment_id"]
        arm = "W" if exp_id.endswith("GRU-GROWTH-F0-S3-R1") else "F" if exp_id.endswith("FIELD-CLOCK-F0-S3-R1") else None
        timeout = _timeout_seconds(authority[arm]["worker_argv"]) if arm else _timeout_seconds(authority["C"]["worker_argv"])
        run, epoch = experiment["run"], experiment["epoch"]
        fit_args = experiment["trainer_argv"]
        fit_step = _step(
            "fit",
            [python_s, str(root / fit_args[0]), *fit_args[1:]],
            timeout=timeout,
            env=warm_env,
            root=root,
        )
        predict_step = _step(
            "predict",
            [
                python_s,
                str(root / "scripts/predict_run.py"),
                run,
                "valid",
                "--epoch",
                str(epoch),
                "--campaign-split-manifest",
                str(root / FOLD0_SPLIT_PATH),
                "--campaign-full-valid-feature-only",
                "--device",
                "cuda",
                "--inference-precision",
                "fp32",
            ],
            timeout=_sequence_timeout(sequence_docs["h256"], "full-valid-prediction"),
            env=warm_env,
            root=root,
        )
        comparison = {
            "experiment_id": exp_id,
            "hypothesis_id": experiment["hypothesis_id"],
            "candidate": f"{run}@ep{epoch}",
            "references": experiment["references"],
            "grade_path": str(root / CAMPAIGN / "grades" / f"{exp_id}.json"),
            "receipt_path": str(root / CAMPAIGN / "receipts" / f"{exp_id}.json"),
            "ledger_path": str(root / CAMPAIGN / "comparison_ledger.jsonl"),
            "split_path": str(root / FOLD0_SPLIT_PATH),
            "expected_fold": 0,
            "fidelity": {
                "kind": "fixed checkpoint on previously exposed fold 0",
                "metric": "global_wp_clip2_v1",
                "fit_seed": 3,
                "checkpoint_epoch": epoch,
                "exposure": "development screening only; not fresh holdout or public forecast",
            },
        }
        expected = [
            str(root / "artifacts/runs" / run / f"model_ep{epoch}.pt"),
            str(root / "artifacts/runs" / run / "provenance.json"),
            str(root / "cache" / f"pred_{run}_valid.npy"),
            str(root / "cache" / f"pred_{run}_valid.json"),
            comparison["grade_path"],
        ]
        specs[run] = {
            "run_id": run,
            "priority": priority,
            "arm": experiment["name"],
            "hypothesis": experiment["hypothesis_id"],
            "notes": experiment["contrast"],
            "fidelity": comparison["fidelity"],
            "ancestry": experiment["reference_roles"],
            "exposure": comparison["fidelity"]["exposure"],
            "input_bindings": {
                "portable_data_manifest": PORTABLE_MANIFEST_PATH.as_posix(),
                "portable_data_manifest_sha256": portable_manifest_sha256,
                "source_document_sha256": blueprint["source_documents_sha256"],
                "source_closure": blueprint["source_closure"],
            },
            "steps": [fit_step, predict_step, _step(
                "score",
                [
                    python_s,
                    str(root / "scripts/score_fold_panel.py"),
                    comparison["candidate"],
                    *sum((["--reference", ref] for ref in comparison["references"]), []),
                    "--split",
                    comparison["split_path"],
                    "--expected-fold",
                    "0",
                    "--out",
                    comparison["grade_path"],
                ],
                timeout=_sequence_timeout(sequence_docs["h256"], "official-fold0-grade"),
                env=base_env | {"WNN_THREADS": "1"},
                root=root,
            )],
            "expected_outputs": expected,
            "comparison": comparison,
        }

    exp = experiment_by_id["E-PORTABLE4-H512-SEED4-CONFIRM-F0-R1"]
    pair_id = "portable4_h512_seed4_pair_f0_r1"
    h512_run, h256_run = exp["run"], exp["paired_run"]
    h512_doc, h256_doc = sequence_docs["h512"], sequence_docs["h256"]
    h512_timeout = _sequence_timeout(h512_doc, "full-fit")
    h256_timeout = _sequence_timeout(h256_doc, "full-fit")
    prediction_timeout = _sequence_timeout(h512_doc, "full-valid-prediction")
    score_timeout = _sequence_timeout(h512_doc, "official-fold0-grade")
    steps: list[dict[str, Any]] = []
    for run, argv, timeout in (
        (h256_run, exp["paired_command"], h256_timeout),
        (h512_run, exp["trainer_argv"], h512_timeout),
    ):
        steps.append(_step(
            f"fit_{run}",
            [python_s, str(root / argv[0]), *argv[1:]],
            timeout=timeout,
            env=frontier_env,
            root=root,
        ))
        predictor = str(root / "scripts/predict_student_frontier_f0_frozen.py")
        steps.append(_step(
            f"predict_{run}",
            [
                python_s,
                predictor,
                run,
                "valid",
                "--epoch",
                "0",
                "--campaign-split-manifest",
                str(root / FOLD0_SPLIT_PATH),
                "--campaign-full-valid-feature-only",
                "--device",
                "cuda",
                "--inference-precision",
                "fp32",
            ],
            timeout=prediction_timeout,
            env=frontier_env,
            root=root,
        ))
    comparison = {
        "experiment_id": exp["experiment_id"],
        "hypothesis_id": exp["hypothesis_id"],
        "candidate": f"{h512_run}@ep0",
        "references": [f"{h256_run}@ep0"],
        "grade_path": str(root / CAMPAIGN / "grades" / f"{exp['experiment_id']}.json"),
        "receipt_path": str(root / CAMPAIGN / "receipts" / f"{exp['experiment_id']}.json"),
        "ledger_path": str(root / CAMPAIGN / "comparison_ledger.jsonl"),
        "split_path": str(root / FOLD0_SPLIT_PATH),
        "expected_fold": 0,
        "fidelity": {
            "kind": "matched-seed two-shape confirmation; two new cold-start fits",
            "metric": "global_wp_clip2_v1",
            "fit_seed": 4,
            "checkpoint_epoch": 0,
            "batch": 8,
            "chunk": 2000,
            "precision": "fp32",
            "exposure": "development screening only; not fresh holdout or public forecast",
        },
    }
    steps.append(_step(
        "score",
        [
            python_s,
            str(root / "scripts/score_fold_panel.py"),
            comparison["candidate"],
            "--reference",
            comparison["references"][0],
            "--split",
            comparison["split_path"],
            "--expected-fold",
            "0",
            "--out",
            comparison["grade_path"],
        ],
        timeout=score_timeout,
        env=base_env | {"WNN_THREADS": "1"},
        root=root,
    ))
    expected = []
    for run in (h256_run, h512_run):
        expected.extend(
            [
                str(root / "artifacts/runs" / run / "model_ep0.pt"),
                str(root / "artifacts/runs" / run / "provenance.json"),
                str(root / "cache" / f"pred_{run}_valid.npy"),
                str(root / "cache" / f"pred_{run}_valid.json"),
            ]
        )
    expected.append(comparison["grade_path"])
    specs[pair_id] = {
        "run_id": pair_id,
        "priority": 4,
        "arm": exp["name"],
        "hypothesis": exp["hypothesis_id"],
        "notes": exp["contrast"],
        "fidelity": comparison["fidelity"],
        "ancestry": {
            "both student weight initializations": "cold start",
            "shared normalization": "common H256x2 seed-3 CP5 mean/std buffers only",
            "teacher": "same clean T3 cache",
        },
        "exposure": comparison["fidelity"]["exposure"],
        "input_bindings": {
            "portable_data_manifest": PORTABLE_MANIFEST_PATH.as_posix(),
            "portable_data_manifest_sha256": portable_manifest_sha256,
            "source_document_sha256": blueprint["source_documents_sha256"],
            "source_closure": blueprint["source_closure"],
            "seed4_contract": SEED4_CONTRACT_PATH.as_posix(),
        },
        "steps": steps,
        "expected_outputs": expected,
        "comparison": comparison,
    }
    return specs


def materialize(root: Path, python: Path, blueprint: dict[str, Any], report: dict[str, Any]) -> list[str]:
    if not report["ok"]:
        raise ValueError("preflight failed; pending specs and admission hashes were not written")
    portable_manifest = make_portable_data_manifest(root, blueprint, report)
    seed4_contract = make_seed4_contract(blueprint)
    seed4_contract["source_path_bindings"] = make_source_path_bindings(root, seed4_contract)
    seed4_contract["sample_order"] = seed4_sample_order(root, python, portable_manifest, seed4_contract)
    manifest_bytes = canonical_bytes(portable_manifest)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    contract_bytes = canonical_bytes(seed4_contract)
    contract_hash = hashlib.sha256(contract_bytes).hexdigest()
    specs = build_worker_specs(root, python, blueprint, manifest_hash)

    queue_blueprint = copy.deepcopy(blueprint)
    queue_blueprint["status"] = "PENDING_INPUTS_VERIFIED_NO_LAUNCH"
    queue_blueprint["preflight"] = report
    queue_blueprint["root"] = str(root)
    queue_blueprint["python"] = str(python)
    queue_blueprint["portable_data_manifest"] = {
        "path": PORTABLE_MANIFEST_PATH.as_posix(),
        "sha256": manifest_hash,
        "sources": list(portable_manifest["sources"]),
    }
    queue_blueprint["seed4_contract"] = {
        "path": SEED4_CONTRACT_PATH.as_posix(),
        "sha256": contract_hash,
        "sample_order": seed4_contract["sample_order"],
        "source_drift_declared": blueprint["source_closure"]["h512_source_differences"],
    }
    queue_blueprint["worker_spec_sha256"] = {}
    spec_bytes: dict[str, bytes] = {}
    for run_id, spec in sorted(specs.items()):
        raw = (json.dumps(spec, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        spec_bytes[run_id] = raw
        queue_blueprint["worker_spec_sha256"][run_id] = hashlib.sha256(raw).hexdigest()
    blueprint_bytes = (json.dumps(queue_blueprint, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    allowed_bytes = canonical_bytes(queue_blueprint["worker_spec_sha256"])
    admission = {
        "schema": "wnn.portable-four-gpu-queue-admission.r1",
        "status": "PENDING_QUEUE_HASH_NOT_RESOURCE_ADMISSION",
        "worker_queue_dir": QUEUE_DIR.as_posix(),
        "allowed_spec_manifest_path": ALLOWED_SPEC_PATH.as_posix(),
        "allowed_spec_manifest_sha256": hashlib.sha256(allowed_bytes).hexdigest(),
        "pending_specs": [
            {"run_id": run_id, "path": (PENDING_DIR / f"{run_id}.json").as_posix(), "sha256": queue_blueprint["worker_spec_sha256"][run_id]}
            for run_id in sorted(spec_bytes)
        ],
        "portable_data_manifest_path": PORTABLE_MANIFEST_PATH.as_posix(),
        "portable_data_manifest_sha256": manifest_hash,
        "seed4_contract_path": SEED4_CONTRACT_PATH.as_posix(),
        "seed4_contract_sha256": contract_hash,
        "trainer_or_worker_started": False,
        "gpu_touched": False,
    }
    outputs: dict[Path, bytes] = {
        root / BLUEPRINT_PATH: blueprint_bytes,
        root / PORTABLE_MANIFEST_PATH: manifest_bytes,
        root / SEED4_CONTRACT_PATH: contract_bytes,
        root / ALLOWED_SPEC_PATH: allowed_bytes,
        root / ADMISSION_PATH: canonical_bytes(admission),
    }
    outputs.update({root / PENDING_DIR / f"{run_id}.json": raw for run_id, raw in spec_bytes.items()})
    if any(path.exists() for path in outputs):
        raise FileExistsError("one or more pending queue artifacts already exist; refusing overwrite")
    for path, raw in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    return [path.relative_to(root).as_posix() for path in outputs]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="read-only exact-input/source/output preflight")
    mode.add_argument("--prepare", action="store_true", help="materialize worker pending specs only after successful exact-input preflight; never launches")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="portable repository root")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="absolute Python interpreter bound into future worker commands")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    python = args.python.expanduser().resolve()
    try:
        blueprint = make_blueprint(root)
        report = preflight(root, python, blueprint)
        if args.prepare and report["ok"]:
            report["materialized"] = materialize(root, python, blueprint, report)
            report["status"] = "PENDING_QUEUE_MATERIALIZED_NO_LAUNCH"
        elif args.prepare:
            report["status"] = "HOLD_MISSING_OR_MISMATCHED_BINDINGS_NO_QUEUE_WRITTEN"
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 2
    except (OSError, ValueError, KeyError, IndexError, TypeError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "status": "HOLD", "error": str(error), "trainer_or_worker_started": False, "gpu_touched": False}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
