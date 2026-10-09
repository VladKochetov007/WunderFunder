#!/usr/bin/env python3
"""Replay the sealed H230 source-v3 parent-to-student head extension on CPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[4]
LINEAGE = ROOT / "research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230"
SOURCE = ROOT / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h230_source_v3_all_data_distill"
V3_CONTRACT = LINEAGE / "H230_ALLDATA_STUDENT_PAIR_RECIPE_CONTRACT_V3_20261008.json"
SOURCE_MANIFEST = SOURCE / "H230_ALLDATA_DISTILL_SOURCE_MANIFEST_V1_20261008.json"
PARENT_CHECKPOINT = ROOT / "artifacts/runs/bd26_h230_all_data_parent_b16_ep3_retry3_v1/model_ep3.pt"
PARENT_TERMINAL = ROOT / "research/campaign_2026-10-06/bandit_continuation/runtime/attempts/h230-all-data-parent-fit-b16-ep3-retry3-v1-20261008.json"
CACHE_TERMINAL = ROOT / "research/campaign_2026-10-06/bandit_continuation/runtime/attempts/h230-all-data-teacher-cache-v2-20261008.json"
CACHE_METADATA = ROOT / "cache/bd26_h230_fold0_teacher_family_all_rows_v1/metadata.json"
SOURCE_COMPATIBILITY = LINEAGE / "H230_ALLDATA_PARENT_STUDENT_SOURCE_COMPATIBILITY_V1_20261008.json"
OUTPUT = LINEAGE / "H230_ALLDATA_STUDENT_PARENT_LOAD_CHECK_V1_20261008.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_trainer() -> Any:
    sys.path.insert(0, str(SOURCE / "src"))
    trainer_path = SOURCE / "scripts/train_rnn.py"
    spec = importlib.util.spec_from_file_location("h230_source_v3_train_rnn", trainer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the frozen source-v3 trainer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_seed(trainer: Any, parent_state: dict[str, torch.Tensor], seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    args = {
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "hidden": 256,
        "layers": 2,
        "dropout": 0.0,
        "readout_activation": "gelu",
        "sentinel_indicators": "none",
    }
    ordinary_outputs = 2 + trainer.N_SP_AUX + 3 + 4 + 2
    if ordinary_outputs != 23:
        raise AssertionError(f"source-v3 ordinary output formula changed: {ordinary_outputs}")
    initialized = trainer.make_model(args, 211, ordinary_outputs)
    trainer.append_family_output_head(initialized)
    initialized_state = initialized.state_dict()
    initial_family_weight = initialized_state["head.2.weight"][ordinary_outputs:].clone()
    initial_family_bias = initialized_state["head.2.bias"][ordinary_outputs:].clone()
    merged = trainer.extend_family_output_head(parent_state, initialized_state)
    incompatible = initialized.load_state_dict(merged, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError("strict student checkpoint load reported missing or unexpected keys")

    parent_weight = parent_state["head.2.weight"]
    parent_bias = parent_state["head.2.bias"]
    student_weight = initialized.state_dict()["head.2.weight"].detach().cpu()
    student_bias = initialized.state_dict()["head.2.bias"].detach().cpu()
    if tuple(parent_weight.shape) != (23, 256) or tuple(parent_bias.shape) != (23,):
        raise AssertionError("actual parent output rows differ from the sealed checkpoint expectation")
    if tuple(student_weight.shape) != (31, 256) or tuple(student_bias.shape) != (31,):
        raise AssertionError("student head is not 23 ordinary plus eight appended family rows")
    if not torch.equal(student_weight[:23], parent_weight) or not torch.equal(student_bias[:23], parent_bias):
        raise AssertionError("source-v3 loader did not preserve all 23 parent head rows")
    if not torch.equal(student_weight[23:], initial_family_weight) or not torch.equal(student_bias[23:], initial_family_bias):
        raise AssertionError("source-v3 loader changed the freshly initialized family rows")

    shape_changes: list[dict[str, Any]] = []
    for name, parent_value in parent_state.items():
        student_value = initialized.state_dict()[name].detach().cpu()
        if tuple(parent_value.shape) != tuple(student_value.shape):
            shape_changes.append(
                {"name": name, "parent_shape": list(parent_value.shape), "student_shape": list(student_value.shape)}
            )
        if name in ("head.2.weight", "head.2.bias"):
            if not torch.equal(student_value[: parent_value.shape[0]], parent_value):
                raise AssertionError(f"parent rows differ after loading: {name}")
        elif tuple(parent_value.shape) != tuple(student_value.shape) or not torch.equal(student_value, parent_value):
            raise AssertionError(f"non-head parent tensor was not copied exactly: {name}")
        if not torch.isfinite(student_value).all():
            raise AssertionError(f"non-finite tensor after student initialization: {name}")

    expected_shape_changes = [
        {"name": "head.2.bias", "parent_shape": [23], "student_shape": [31]},
        {"name": "head.2.weight", "parent_shape": [23, 256], "student_shape": [31, 256]},
    ]
    if sorted(shape_changes, key=lambda item: item["name"]) != sorted(expected_shape_changes, key=lambda item: item["name"]):
        raise AssertionError(f"unexpected parent/student shape changes: {shape_changes}")
    return {
        "seed": seed,
        "ordinary_output_rows": ordinary_outputs,
        "family_output_rows": 8,
        "student_output_rows": 31,
        "fresh_family_weight_sha256": tensor_sha256(initial_family_weight),
        "fresh_family_bias_sha256": tensor_sha256(initial_family_bias),
        "all_tensors_finite": True,
        "all_non_head_parent_tensors_copied_exactly": True,
        "all_23_parent_head_rows_copied_exactly": True,
        "fresh_8_family_rows_retained_exactly": True,
        "strict_state_dict_load": True,
        "shape_changes": shape_changes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--created-utc", required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    for path in (V3_CONTRACT, SOURCE_MANIFEST, PARENT_CHECKPOINT, PARENT_TERMINAL, CACHE_TERMINAL, CACHE_METADATA, SOURCE_COMPATIBILITY):
        if not path.is_file():
            raise FileNotFoundError(path)

    v3 = json.loads(V3_CONTRACT.read_text())
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    parent_terminal = json.loads(PARENT_TERMINAL.read_text())
    cache_terminal = json.loads(CACHE_TERMINAL.read_text())
    cache_metadata = json.loads(CACHE_METADATA.read_text())
    compatibility = json.loads(SOURCE_COMPATIBILITY.read_text())

    for seed in (3, 4):
        run = next(item for item in v3["student_runs"] if item["seed"] == seed)
        argv = run["worker_argv"]
        expected_pairs = {
            "--seed": str(seed),
            "--arch": "gru",
            "--cell": "gru",
            "--hidden": "256",
            "--layers": "2",
            "--aux-struct": "0.3",
            "--aux-next": "0.3",
            "--init": "bd26_h230_all_data_parent_b16_ep3_retry3_v1@ep3",
        }
        for flag, expected in expected_pairs.items():
            if argv[argv.index(flag) + 1] != expected:
                raise AssertionError(f"frozen student argv drift for seed {seed}: {flag}")
        required_flags = ("--distill-family-head", "--distill-cache-all-data", "--save-training-state")
        if not all(flag in argv for flag in required_flags):
            raise AssertionError(f"frozen student argv is missing a required flag for seed {seed}")

    manifest_hashes = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    trainer_rel = "scripts/train_rnn.py"
    distillation_rel = "src/wnn/distillation.py"
    trainer_path = SOURCE / trainer_rel
    distillation_path = SOURCE / distillation_rel
    for relative, path in ((trainer_rel, trainer_path), (distillation_rel, distillation_path)):
        if sha256_file(path) != manifest_hashes[relative]:
            raise AssertionError(f"sealed source-v3 file drift: {relative}")

    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_args = parent["args"]
    expected_parent_args = {
        "arch": "gru",
        "cell": "gru",
        "hidden": 256,
        "layers": 2,
        "dropout": 0.0,
        "aux": "",
        "aux_struct": 0.3,
        "aux_next": 0.3,
        "aux_next_k": "",
    }
    for key, expected in expected_parent_args.items():
        if parent_args.get(key) != expected:
            raise AssertionError(f"parent checkpoint argument drift: {key}={parent_args.get(key)!r}")
    if parent_args.get("distill_family_head", False):
        raise AssertionError("parent checkpoint unexpectedly contains the student family head")
    if parent_terminal.get("contract_sha256") != v3["parent_dependency"]["contract_sha256"]:
        raise AssertionError("parent terminal contract does not match the student contract dependency")
    if parent_terminal.get("exit_status") != 0:
        raise AssertionError("parent terminal receipt is not successful")
    if sha256_file(PARENT_CHECKPOINT) != "478773bd3d0754d261f8f3c9d46a135d56c3596c269d2a856b760caef05b93c7":
        raise AssertionError("fixed parent ep3 checkpoint hash differs from the root-verified hash")

    trainer = load_trainer()
    parent_state = parent["model"]
    input_weight = parent_state["inp.weight"]
    if input_weight.ndim != 2 or input_weight.shape[1] != 211:
        raise AssertionError(f"parent input width is not 211: {tuple(input_weight.shape)}")
    parent_output_rows = int(parent_state["head.2.weight"].shape[0])
    if parent_output_rows != 23:
        raise AssertionError(f"parent output width is not 23: {parent_output_rows}")

    children = [verify_seed(trainer, parent_state, seed) for seed in (3, 4)]
    if children[0]["fresh_family_weight_sha256"] == children[1]["fresh_family_weight_sha256"]:
        raise AssertionError("seed-specific family output initializations are unexpectedly identical")
    if children[0]["fresh_family_bias_sha256"] == children[1]["fresh_family_bias_sha256"]:
        raise AssertionError("seed-specific family output biases are unexpectedly identical")

    receipt = {
        "schema": "h230-parent-student-load-check-v1",
        "receipt_id": "H230_ALLDATA_STUDENT_PARENT_LOAD_CHECK_V1_20261008",
        "created_utc": args.created_utc,
        "status": "passed_actual_source_v3_checkpoint_load_and_seed_specific_head_extension",
        "v3_contract_path": str(V3_CONTRACT.relative_to(ROOT)),
        "v3_contract_sha256": sha256_file(V3_CONTRACT),
        "v3_dimension_metadata_correction": {
            "preserved_v3": True,
            "stale_claim": "parent 15 ordinary rows; student 23 total rows including family",
            "verified_mapping": "actual parent 23 ordinary main/aux rows -> student 23 copied ordinary rows + 8 fresh family rows = 31 total rows",
            "scientific_argv_changed": False,
        },
        "parent": {
            "run": "bd26_h230_all_data_parent_b16_ep3_retry3_v1",
            "terminal_receipt_path": str(PARENT_TERMINAL.relative_to(ROOT)),
            "terminal_receipt_sha256": sha256_file(PARENT_TERMINAL),
            "checkpoint_path": str(PARENT_CHECKPOINT.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(PARENT_CHECKPOINT),
            "parent_args_verified": expected_parent_args,
            "input_width": 211,
            "output_rows": parent_output_rows,
            "family_head": False,
        },
        "student_source_v3": {
            "load_check_script_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "load_check_script_sha256": sha256_file(Path(__file__).resolve()),
            "manifest_path": str(SOURCE_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": sha256_file(SOURCE_MANIFEST),
            "trainer_path": str(trainer_path.relative_to(ROOT)),
            "trainer_sha256": sha256_file(trainer_path),
            "distillation_helper_path": str(distillation_path.relative_to(ROOT)),
            "distillation_helper_sha256": sha256_file(distillation_path),
            "source_compatibility_receipt_path": str(SOURCE_COMPATIBILITY.relative_to(ROOT)),
            "source_compatibility_receipt_sha256": sha256_file(SOURCE_COMPATIBILITY),
            "source_compatibility_receipt_status": compatibility.get("status"),
            "actual_methods_replayed": [
                "train_rnn.make_model",
                "distillation.append_family_output_head",
                "distillation.extend_family_output_head",
                "torch.nn.Module.load_state_dict(strict=True)",
            ],
        },
        "cache": {
            "terminal_receipt_path": str(CACHE_TERMINAL.relative_to(ROOT)),
            "terminal_receipt_sha256": sha256_file(CACHE_TERMINAL),
            "terminal_contract_sha256": cache_terminal.get("contract_sha256"),
            "metadata_path": str(CACHE_METADATA.relative_to(ROOT)),
            "metadata_sha256": sha256_file(CACHE_METADATA),
            "membership_mode": cache_metadata.get("membership_mode", cache_metadata.get("membership")),
            "validated_all_eight_arrays": True,
            "validation_variants": ["source", "normalized", "homogeneous"],
            "validation_note": "Previously completed source-v3 validator checked all eight arrays, row-group order, ancestry, finite chunk values, and all-data membership; no large prediction files were rehashed in this small model-load check.",
        },
        "students": children,
        "pair_relationship": "Both students share the exact parent checkpoint and teacher cache; only seed-specific fresh family head initialization and training randomness differ. They are not independent whole-model draws.",
        "fit_or_evaluation_performed": False,
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(OUTPUT.relative_to(ROOT)), "sha256": sha256_file(OUTPUT), "status": receipt["status"], "student_output_rows": 31}, sort_keys=True))


if __name__ == "__main__":
    main()
