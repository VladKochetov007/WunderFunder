"""Independently grade one finalized L4 stage-two teacher target cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import resource
except ImportError:  # pragma: no cover - the production grader runs on Linux.
    resource = None

SOURCE_ROOT = Path(
    os.environ.get("WNN_GRADER_SOURCE_ROOT", Path(__file__).resolve().parents[2])
).resolve()
sys.path[:0] = [str(SOURCE_ROOT), str(SOURCE_ROOT / "src")]

from scripts.build_distill_cache import _audit_ancestor_closure, _target_order  # noqa: E402
from scripts.l4_teacher_capacity_20261104.stage2_stream_common import (  # noqa: E402
    EXPECTED_RUNS,
    GRU_SHAPES,
    NATIVE_BANK_MAX_BYTES,
    PR60_CACHE_METADATA_SHA256,
    PR60_FAMILIES_TRAIN_SHA256,
    PR60_MAIN_TRAIN_SHA256,
    SEQUENCE_BANK_MAX_BYTES,
    SOURCE_TRAIN_INDEX_SEQUENCES,
    SPLIT_SHA256,
    TARGET_BYTES,
    TRAIN_SEQUENCES,
    positions_sha256,
    sequence_hash,
)
from scripts.l4_teacher_capacity_20261104.stage2_stream_teachers import (  # noqa: E402
    verify_sequence_teacher,
)
from wnn.distillation_cache import (  # noqa: E402
    SEQUENCE_ROWS,
    file_sha256,
    row_groups_sha256,
    validate_distillation_cache,
)
from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA, load_campaign_split  # noqa: E402
from wnn.training.chronological_teacher import (  # noqa: E402
    validate_chronological_teacher_membership,
    validate_chronological_train_index_order,
)
from wnn.training.sequence_provenance import canonical_sha256  # noqa: E402

OUTPUT_CAP_BYTES = 6_500_000_000
MIN_FREE_BYTES = 3_000_000_000
STREAM_METADATA_OVERHEAD_BYTES = 64 << 20
INITIAL_FREE_BYTES = 10_172_567_552
INITIAL_FREE_BASIS_RELATIVE = Path(
    "source-r1/research/experiment_drafts/l4-teacher-capacity-20261104/CONTRACT_R1.json"
)
FIT_MEMBERS_SHA256 = "075c50995165f7a35f31c1e4f022b27fc6247270cc15ddda28075e6eeb624e19"
FBC_INIT_SHA256 = "840966894144bca5aafbf0a8e21ad0daab07a8503ba80e30d96668ef1bc3a496"
R2_SOURCE_HASHES = {
    "scripts/l4_teacher_capacity_20261104/stage2_stream_common.py": "e52b0e812292caefb0df3b101cec63098efeb1345e7bb37b225169c7ac94405",
    "scripts/l4_teacher_capacity_20261104/stage2_stream_finalize.py": "a84b1e053ff23034af8853b60488c7fa8c2921359aeca71e609e58beca79333b",
    "scripts/l4_teacher_capacity_20261104/stage2_stream_native.py": "15b6bbb87e07359d5379e9b70e45c76dddcd8a0e6e28071270e4a179ab02fa79",
    "scripts/l4_teacher_capacity_20261104/stage2_stream_teachers.py": "316b7fc10734f326cd99b8b3fd0b02ceed70d5c09b2976b308b15e92cb03790",
    "scripts/l4_teacher_capacity_20261104/stream_stage2_targets.py": "9ae855907edbbce2fc0c15d28b05fe82f1cabe9b222a5a4b18274ff1c52f3f2e",
}
R2_HANDOFF_SHA256 = "5316ae2bc4d835efd83ba5d65552024a99540bf53a850ce9883db39f0116868b"
TARGET_FORMULA = "float32(0.8*T_old + 0.2*0.8125*mean(float32_clip(each_new_teacher,-2,2)))"
STUDENT_CONTRACT = {
    "init": "tl3c_fbc1k_s5@ep3",
    "inherit_init_preprocessing": True,
    "seed_set": [1001, 1002],
    "epochs": 2,
    "checkpoint_policy": "fixed",
    "checkpoint_epoch": 1,
    "precision": "fp32",
    "optimizer": "adamw",
    "learning_rate": 0.0001,
    "distill_alpha": 0.7,
    "distill_family_weight": 0.1,
    "hidden": 256,
    "layers": 2,
    "batch": 16,
    "chunk": 2000,
}

# JSON receipts deliberately have evolving, versioned fields; `Any` keeps the
# reader faithful to their serialized schemas while checks remain explicit.


def _read_json(path: Path, limit: int = 32 << 20) -> dict[str, Any]:
    """Read a bounded regular JSON object without following a symlink."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"expected a regular JSON file of at most {limit} bytes: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _option_value(argv: list[str], flag: str, run: str) -> str:
    positions = [index for index, token in enumerate(argv[:-1]) if token == flag]
    _require(len(positions) == 1, f"{run}: expected exactly one {flag} in fit argv")
    return argv[positions[0] + 1]


def _check_initial_free_basis(shared_state: dict[str, Any], runtime_root: Path) -> dict[str, Any]:
    """Tie the original disk observation to its frozen receiving contract."""
    free_bytes = shared_state.get("initial_data2_free_bytes")
    basis_path_value = shared_state.get("initial_data2_free_basis_path")
    basis_sha_expected = shared_state.get("initial_data2_free_basis_sha256")
    _require(
        type(free_bytes) is int and free_bytes == INITIAL_FREE_BYTES,
        "shared wave state initial free-space observation differs from the authorized receipt",
    )
    _require(
        isinstance(basis_path_value, str) and isinstance(basis_sha_expected, str),
        "shared wave state does not bind its initial free-space basis",
    )
    basis_path = Path(basis_path_value)
    if not basis_path.is_absolute():
        parts = basis_path.parts
        if parts and parts[0] == "R":
            basis_path = Path(*parts[1:])
        basis_path = runtime_root / basis_path
    basis_path = basis_path.resolve(strict=True)
    _require(
        basis_path.as_posix().endswith(INITIAL_FREE_BASIS_RELATIVE.as_posix()),
        "initial free-space basis is not the frozen source-r1 contract",
    )
    basis_sha = file_sha256(basis_path)
    _require(
        basis_sha == basis_sha_expected,
        "initial free-space basis hash differs from shared wave state",
    )
    contract = _read_json(basis_path)
    _require(
        contract.get("id") == "E-L4-TEACHER-CAPACITY-20261104"
        and contract.get("budget", {}).get("observed_free_data2_bytes") == free_bytes,
        "frozen contract does not attest the original data2 free-space observation",
    )
    return {
        "observation_kind": "original_receiving_contract_observation",
        "free_bytes": free_bytes,
        "basis_path": str(basis_path),
        "basis_sha256": basis_sha,
    }


def _hash_source_snapshot(source_root: Path) -> dict[str, str]:
    """Require the exact independently reviewed R2 streaming implementation."""
    observed = {}
    for relative, expected in R2_SOURCE_HASHES.items():
        path = source_root / relative
        observed[relative] = file_sha256(path)
        _require(observed[relative] == expected, f"stage-two source hash differs: {relative}")
    return observed


def _check_panel(path: Path, family: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Bind family, continuation panel and teacher admission before reading outputs."""
    panel = _read_json(path)
    _require(
        panel.get("schema") == "wnn.l4-teacher-capacity-continuation.v1"
        and panel.get("status") == "frozen",
        "continuation panel is not the frozen packet-specific panel",
    )
    panel_hash = file_sha256(path)
    paths, identity = panel.get("paths"), panel.get("identity")
    bindings = panel.get("families")
    _require(isinstance(paths, dict) and isinstance(identity, dict), "panel paths/identity missing")
    _require(
        isinstance(bindings, dict) and isinstance(bindings.get(family), dict),
        "family binding missing",
    )
    _require(identity.get("split_sha256") == SPLIT_SHA256, "continuation panel split differs")
    for key in ("source_binding_sha256", "data_binding_sha256", "fit_members_sha256"):
        value = identity.get(key)
        _require(isinstance(value, str) and len(value) == 64, f"panel identity {key} is malformed")
    admission_path = Path(panel.get("admission_receipt", ""))
    admission = _read_json(admission_path)
    _require(
        file_sha256(admission_path) == panel.get("admission_receipt_sha256")
        and str(admission.get("status", "")).startswith("approved")
        and admission.get("continuation_panel_sha256") in (None, panel_hash)
        and all(
            admission.get(key) == identity.get(key)
            for key in (
                "source_binding_sha256",
                "data_binding_sha256",
                "fit_members_sha256",
                "baseline_ancestry_sha256",
                "pr60_panel_sha256",
                "split_sha256",
            )
        ),
        "independent continuation admission receipt does not bind the frozen panel",
    )
    pr60_panel_path = Path(paths["pr60_panel"])
    _require(
        file_sha256(pr60_panel_path) == identity["pr60_panel_sha256"],
        "current PR60 panel hash differs",
    )
    binding = bindings[family]
    teacher_panel_path = Path(binding["teacher_panel"])
    teacher_panel_hash = file_sha256(teacher_panel_path)
    _require(teacher_panel_hash == binding.get("teacher_panel_sha256"), "teacher panel SHA differs")
    teacher_panel = _read_json(teacher_panel_path)
    teacher_identity = teacher_panel.get("identity")
    _require(isinstance(teacher_identity, dict), "teacher panel identity missing")
    teacher_admission_path = Path(binding["teacher_admission_receipt"])
    teacher_admission = _read_json(teacher_admission_path)
    _require(
        file_sha256(teacher_admission_path) == binding.get("teacher_admission_receipt_sha256"),
        "teacher admission receipt SHA differs",
    )
    _require(
        str(teacher_admission.get("status", "")).startswith("approved")
        and teacher_admission.get("family") == family
        and teacher_admission.get("teacher_panel_sha256") == teacher_panel_hash
        and teacher_admission.get("source_binding_sha256")
        == teacher_identity.get("source_binding_sha256")
        and teacher_admission.get("data_binding_sha256")
        == teacher_identity.get("data_binding_sha256")
        and teacher_admission.get("split_sha256") == SPLIT_SHA256,
        "teacher admission does not bind its actual source/data/panel/split",
    )
    return (
        panel,
        panel_hash,
        {
            "binding": binding,
            "teacher_panel": teacher_panel,
            "teacher_panel_hash": teacher_panel_hash,
            "continuation_panel": panel,
            "continuation_panel_hash": panel_hash,
        },
    )


def _check_student_contract(panel: dict[str, Any], family: str, cache: Path) -> str:
    """Bind the frozen four-cell student panel to the intended family cache."""
    cells = panel.get("students", {}).get("cells")
    _require(isinstance(cells, list) and len(cells) == 4, "student panel must contain four cells")
    wanted = {(name, seed) for name in EXPECTED_RUNS for seed in (1001, 1002)}
    seen = set()
    for cell in cells:
        _require(isinstance(cell, dict), "student cell is malformed")
        ref = (cell.get("family"), cell.get("seed"))
        _require(ref in wanted and ref not in seen, "student family/seed set differs")
        seen.add(ref)
        argv = cell.get("train_args")
        _require(
            isinstance(argv, list) and all(isinstance(value, str) for value in argv),
            "student argv missing",
        )
        if ref[0] != family:
            continue
        _require(argv.count("--distill-cache-root") == 1, "student cache argument is ambiguous")
        cache_index = argv.index("--distill-cache-root")
        _require(
            Path(argv[cache_index + 1]).resolve() == cache.resolve(),
            "student points at a different target cache",
        )
        required = {
            "--seed": str(ref[1]),
            "--train-source": "train",
            "--train-seqs": "10325",
            "--eval-source": "dev",
            "--precision": "fp32",
            "--epochs": "2",
            "--checkpoint-policy": "fixed",
            "--batch": "16",
            "--chunk": "2000",
            "--init": "tl3c_fbc1k_s5@ep3",
            "--distill-alpha": ".7",
            "--distill-family-weight": ".1",
        }
        for flag, expected in required.items():
            _require(
                argv.count(flag) == 1 and argv[argv.index(flag) + 1] == expected,
                f"student {flag} differs",
            )
        checkpoint_epoch_positions = [
            index for index, token in enumerate(argv[:-1]) if token == "--checkpoint-epoch"
        ]
        _require(
            len(checkpoint_epoch_positions) <= 1
            and (not checkpoint_epoch_positions or argv[checkpoint_epoch_positions[0] + 1] == "1"),
            "student fixed endpoint differs from epoch 1",
        )
        _require(
            "--full-valid" in argv and "--inherit-init-preprocessing" in argv,
            "student full-valid/init contract differs",
        )
    _require(seen == wanted, "student panel omits an authorized family/seed cell")
    return canonical_sha256(cells)


def _check_base_cache(panel: dict[str, Any], split: Any) -> tuple[Path, dict[str, Any]]:
    """Validate exact current PR60 cache bytes and fit order, not historical cache ancestry."""
    old_root = Path(panel["paths"]["old_cache"]).resolve(strict=True)
    result = validate_distillation_cache(
        old_root, Path(panel["paths"]["project_root"]), split, "source", require_families=True
    )
    metadata, files = result["metadata"], result["metadata"]["files"]
    _require(
        result["metadata_sha256"] == PR60_CACHE_METADATA_SHA256,
        "qualified PR60 metadata hash differs",
    )
    _require(
        files["main_source_train"]["sha256"] == PR60_MAIN_TRAIN_SHA256,
        "qualified PR60 main target hash differs",
    )
    _require(
        files["families_train"]["sha256"] == PR60_FAMILIES_TRAIN_SHA256,
        "qualified PR60 auxiliary hash differs",
    )
    _require(
        files["main_source_train"]["row_groups"] == list(split.train_groups),
        "PR60 fit order differs",
    )
    _require(
        files["families_train"]["row_groups"] == list(split.train_groups),
        "PR60 family fit order differs",
    )
    return old_root, metadata


def _check_stage2_receipts(
    panel: dict[str, Any], panel_hash: str, family: str, cache: Path
) -> list[dict[str, Any]]:
    """Verify every init/add/finalize command receipt for this exact continuation."""
    paths = panel["paths"]
    binding = panel["families"][family]
    cache_arg = str(Path(binding["target_cache"]))
    runs = list(EXPECTED_RUNS[family])
    receipts_root = Path(paths["runtime_root"]) / "receipts" / "stage2"
    project = Path(paths["project_root"])
    stage2_script = (
        Path(paths["source_root"]) / "scripts/l4_teacher_capacity_20261104/stream_stage2_targets.py"
    )
    common = ["--family", family, "--output-cache", cache_arg]
    preinit_path = binding.get("stage2_init_receipt")
    receipts = []
    preinit_audit = None
    expected_steps: list[tuple[str, int, list[str]]] = []
    init_command_receipt = receipts_root / f"{family}_stage2_01.json"
    if preinit_path is None:
        expected_steps.append(
            (
                "init",
                1,
                [
                    "init",
                    *common,
                    "--project-root",
                    paths["project_root"],
                    "--split-manifest",
                    paths["split_manifest"],
                    "--old-cache",
                    paths["old_cache"],
                ],
            )
        )
    else:
        _require(
            not init_command_receipt.exists(),
            "both root-owned and continuation-owned initialization receipts exist",
        )
        preinit = _read_json(Path(preinit_path))
        state_sha = preinit.get("state_sha256")
        accumulator_sha = preinit.get("accumulator_sha256")
        init_process_receipt_sha = preinit.get("init_process_receipt_sha256")
        _require(
            file_sha256(Path(preinit_path)) == binding.get("stage2_init_receipt_sha256")
            and preinit.get("status") == "approved"
            and preinit.get("family") == family
            and preinit.get("exit_code") == 0
            and preinit.get("exit_code_scope") == "receiving_artifact_audit"
            and preinit.get("original_init_exit_code") is None
            and preinit.get("original_init_exit_basis")
            == "successsentinel+completeartifacts;processalreadyreaped"
            and preinit.get("zero_filled") is True
            and preinit.get("source_train_index_count") == SOURCE_TRAIN_INDEX_SEQUENCES
            and preinit.get("fit_count") == TRAIN_SEQUENCES
            and all(
                isinstance(value, str)
                and len(value) == 64
                and all(char in "0123456789abcdef" for char in value)
                for value in (state_sha, accumulator_sha, init_process_receipt_sha)
            )
            and Path(str(preinit.get("output_cache", ""))).resolve() == cache.resolve(),
            "root-owned R2 initialization receipt differs",
        )
        preinit_audit = {
            "status": preinit["status"],
            "receiver_exit_code": preinit["exit_code"],
            "receiver_exit_code_scope": preinit["exit_code_scope"],
            "original_init_exit_code": None,
            "original_init_exit_basis": preinit["original_init_exit_basis"],
            "init_process_receipt_sha256": init_process_receipt_sha,
            "state_sha256": state_sha,
            "accumulator_sha256": accumulator_sha,
            "zero_filled": True,
            "source_train_index_count": preinit["source_train_index_count"],
            "fit_count": preinit["fit_count"],
            "init_source_git": preinit.get("init_source_git"),
        }
        receipts.append(
            {
                "action": "init",
                "sha256": file_sha256(Path(preinit_path)),
                "receiving_artifact_audit": preinit_audit,
            }
        )
    for index, run in enumerate(runs, start=2):
        args = ["add", *common, "--run", run]
        if family == "causal_transformer":
            args.extend(
                [
                    "--sequence-export",
                    str(project / "cache" / "l4_teacher_sequence_exports" / run),
                    "--teacher-panel",
                    str(binding["teacher_panel"]),
                ]
            )
        expected_steps.append(("add", index, args))
    expected_steps.append(("finalize", len(runs) + 2, ["finalize", *common]))
    for action, index, args in expected_steps:
        cell_id = f"{family}_stage2_{index:02d}"
        receipt_path = receipts_root / f"{cell_id}.json"
        receipt = _read_json(receipt_path)
        owner = receipt.get("owner", {})
        command = receipt.get("command")
        _require(
            receipt.get("status") == "completed"
            and receipt.get("exit_code") == 0
            and receipt.get("timed_out") is False,
            f"stage2 command did not complete cleanly: {cell_id}",
        )
        _require(
            owner.get("continuation_panel_sha256") == panel_hash
            and owner.get("cell_id") == cell_id
            and owner.get("source_binding_sha256") == panel["identity"]["source_binding_sha256"]
            and owner.get("data_binding_sha256") == panel["identity"]["data_binding_sha256"]
            and owner.get("split_sha256") == SPLIT_SHA256,
            f"stage2 owner receipt binding differs: {cell_id}",
        )
        _require(
            isinstance(command, list) and command[-len(args) :] == args,
            f"stage2 CLI args differ: {cell_id}",
        )
        _require(
            Path(command[-len(args) - 1]).resolve() == stage2_script.resolve(),
            f"stage2 script path differs: {cell_id}",
        )
        _require(
            owner.get("command_sha256")
            == hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest(),
            f"stage2 command hash differs: {cell_id}",
        )
        _require(
            Path(str(receipt.get("output_dir", ""))).resolve() == cache.resolve(),
            f"stage2 output directory differs: {cell_id}",
        )
        receipts.append(
            {
                "action": action,
                "cell_id": cell_id,
                "sha256": file_sha256(receipt_path),
                "started_at_epoch": receipt.get("started_at_epoch"),
                "finished_at_epoch": receipt.get("finished_at_epoch"),
            }
        )
    return receipts


def _assert_fp32_state(checkpoint: dict[str, Any], state_key: str, run: str) -> int:
    """Require every floating model-state tensor to be ordinary FP32."""
    state = checkpoint.get(state_key)
    _require(isinstance(state, dict) and state, f"{run}: model state is missing")
    tensors = list(state.values())
    _require(
        bool(tensors) and all(isinstance(value, torch.Tensor) for value in tensors),
        f"{run}: model state contains a non-tensor parameter",
    )
    _require(
        all(not tensor.is_floating_point() or tensor.dtype == torch.float32 for tensor in tensors),
        f"{run}: model state contains non-FP32 floating tensors",
    )
    return sum(tensor.numel() for tensor in tensors if tensor.is_floating_point())


def _check_gru_record(
    run: str,
    saved: dict[str, Any],
    teacher: dict[str, Any],
    split: Any,
    project: Path,
    source_order: list[int],
    fit_order: list[int],
) -> dict[str, Any]:
    """Recheck a GRU checkpoint and surviving export sidecar after bank deletion."""
    checkpoint_path = project / "artifacts" / "runs" / run / "model_ep3.pt"
    provenance_path = checkpoint_path.parent / "provenance.json"
    checkpoint_sha = file_sha256(checkpoint_path)
    provenance_sha = file_sha256(provenance_path)
    _require(
        saved.get("checkpoint_sha256") == checkpoint_sha
        and saved.get("provenance_sha256") == provenance_sha,
        f"{run}: checkpoint/provenance hash differs",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    provenance = _read_json(provenance_path)
    args = provenance.get("args", {})
    expected_hidden, expected_layers, expected_seed = GRU_SHAPES[run]
    _require(
        checkpoint.get("epoch") == 3
        and checkpoint.get("args", {}).get("checkpoint_policy") == "fixed",
        f"{run}: endpoint is not fixed epoch 3",
    )
    _require(
        (args.get("hidden"), args.get("layers"), args.get("seed"), args.get("cell"))
        == (expected_hidden, expected_layers, expected_seed, "gru"),
        f"{run}: GRU recipe differs",
    )
    _require(
        args.get("precision") == "fp32" and args.get("split_manifest_sha256") == SPLIT_SHA256,
        f"{run}: precision/split differs",
    )
    checkpoint_args = checkpoint.get("args", {})
    _require(
        all(
            checkpoint_args.get(key) == args.get(key)
            for key in (
                "seed",
                "hidden",
                "layers",
                "cell",
                "precision",
                "train_source",
                "train_seqs",
                "checkpoint_epoch",
                "checkpoint_policy",
                "split_manifest_sha256",
            )
        ),
        f"{run}: checkpoint arguments disagree with run provenance",
    )
    state = checkpoint.get("model", {})
    input_weight = state.get("gru.weight_ih_l0")
    output_weight = state.get("head.2.weight")
    recurrent_layers = sum(1 for key in state if key.startswith("gru.weight_ih_l"))
    _require(
        isinstance(input_weight, torch.Tensor)
        and input_weight.shape[0] == 3 * expected_hidden
        and isinstance(output_weight, torch.Tensor)
        and output_weight.shape[1] == expected_hidden
        and recurrent_layers == expected_layers,
        f"{run}: serialized GRU dimensions differ from the frozen configuration",
    )
    _require(
        args.get("init") in (None, "")
        and args.get("prop_input") in (None, "")
        and args.get("propensity") in (None, ""),
        f"{run}: teacher has an unexpected learned parent",
    )
    validate_chronological_teacher_membership(provenance, split, run)
    _require(
        provenance.get("checkpoint_sha256", {}).get("model_ep3.pt") == checkpoint_sha,
        f"{run}: run provenance does not bind epoch-3 checkpoint",
    )
    closure = _audit_ancestor_closure(
        run,
        split,
        project / "artifacts" / "runs",
        "model_ep3.pt",
        checkpoint_sha,
        data_root=project,
    )
    _require(
        len(closure) == 1 and saved.get("ancestor_closure") == closure,
        f"{run}: recursive scratch ancestry differs",
    )
    model_count = _assert_fp32_state(checkpoint, "model", run)
    panel_hash = teacher["teacher_panel_hash"]
    identity = teacher["teacher_panel"].get("identity", {})
    fit_identity = dict(identity)
    fit_identity["project_root"] = str(project)
    cell = next(cell for cell in teacher["teacher_panel"]["cells"] if cell.get("run_name") == run)
    fit_receipt_path = Path(teacher["binding"]["fit_receipts"][run])
    fit_receipt = _read_json(fit_receipt_path)
    _check_fit_receipt(
        fit_receipt,
        fit_receipt_path,
        cell,
        run,
        panel_hash,
        fit_identity,
        family="wide_gru",
        teacher_panel=teacher["teacher_panel"],
    )
    _check_gru_argv(cell.get("fit_argv", []), args, run, split.path)
    sidecar_path = project / "cache" / f"pred_{run}_train.json"
    sidecar = _read_json(sidecar_path)
    export_index = list(EXPECTED_RUNS["wide_gru"]).index(run) + 2
    export_cell_id = f"wide_gru_native_export_{export_index:02d}_{run}"
    export_receipt_path = (
        Path(teacher["continuation_panel"]["paths"]["runtime_root"])
        / "receipts"
        / "exports"
        / f"{export_cell_id}.json"
    )
    export_receipt = _read_json(export_receipt_path)
    export_owner = export_receipt.get("owner", {})
    export_command = export_receipt.get("command")
    teacher_source = Path(teacher["teacher_panel"]["paths"]["source_root"])
    export_argv = [
        teacher["teacher_panel"]["paths"]["python"],
        str(teacher_source / "scripts/predict_run.py"),
        run,
        "train",
        "--epoch",
        "3",
        "--campaign-split-manifest",
        teacher["continuation_panel"]["paths"]["split_manifest"],
        "--campaign-full-train-index-feature-only",
        "--device",
        "cuda",
        "--inference-precision",
        "fp32",
    ]
    export_family = teacher["continuation_panel"]["families"]["wide_gru"]
    export_affinity = export_family.get("export_cpu_affinity", list(range(16, 24)))
    _require(
        isinstance(export_command, list)
        and len(export_command) >= len(export_argv) + 3
        and Path(export_command[0]).name == "taskset"
        and export_command[1:3] == ["-c", ",".join(map(str, export_affinity))]
        and export_command[-len(export_argv) :] == export_argv
        and export_receipt.get("status") == "completed"
        and export_receipt.get("exit_code") == 0
        and export_receipt.get("timed_out") is False
        and export_receipt.get("gpu_key") == export_family.get("export_gpu_uuid")
        and Path(str(export_receipt.get("output_dir", ""))).resolve()
        == (project / "cache").resolve()
        and export_owner.get("continuation_panel_sha256") == teacher["continuation_panel_hash"]
        and export_owner.get("panel_sha256") == teacher["teacher_panel_hash"]
        and export_owner.get("cell_id") == export_cell_id
        and export_owner.get("run_name") == run
        and export_owner.get("source_binding_sha256") == identity.get("source_binding_sha256")
        and export_owner.get("data_binding_sha256") == identity.get("data_binding_sha256")
        and export_owner.get("command_sha256")
        == hashlib.sha256(json.dumps(export_command, separators=(",", ":")).encode()).hexdigest(),
        f"{run}: per-teacher native export receipt/command differs",
    )
    source_positions = {group: index for index, group in enumerate(source_order)}
    positions = [source_positions[group] for group in fit_order]
    prediction_record = saved.get("predictions", {}).get("train", {})
    prediction_sha = saved.get("prediction_sha256")
    source_hash = sequence_hash(np.asarray(source_order, dtype="<i8"))
    fit_hash = sequence_hash(np.asarray(fit_order, dtype="<i8"))
    _require(
        saved.get("epoch") == 3
        and saved.get("hidden") == expected_hidden
        and saved.get("layers") == expected_layers
        and saved.get("seed") == expected_seed
        and saved.get("cell") == "gru"
        and saved.get("checkpoint_path") == str(checkpoint_path)
        and saved.get("prediction_path")
        == str(prediction_path := project / "cache" / f"pred_{run}_train.npy")
        and saved.get("prediction_sidecar_sha256") == file_sha256(sidecar_path)
        and saved.get("fit_source_positions_sha256")
        == positions_sha256(np.asarray(positions, dtype="<i8"))
        and saved.get("source_train_index_sha256") == source_hash
        and saved.get("fit_row_group_hashes", {}).get("train") == fit_hash,
        f"{run}: saved source-to-fit mapping or prediction lineage differs",
    )
    _require(
        prediction_record.get("prediction_sha256") == prediction_sha
        and prediction_record.get("sidecar_sha256") == file_sha256(sidecar_path)
        and prediction_record.get("checkpoint_sha256") == checkpoint_sha
        and prediction_record.get("row_groups_sha256") == source_hash,
        f"{run}: nested native prediction record differs",
    )
    _require(
        sidecar.get("run") == run
        and sidecar.get("split") == "train"
        and sidecar.get("epoch") == 3
        and sidecar.get("checkpoint_sha256") == checkpoint_sha
        and sidecar.get("run_provenance_sha256") == provenance_sha
        and sidecar.get("split_manifest_sha256") == SPLIT_SHA256
        and sidecar.get("campaign_split_manifest_sha256") == SPLIT_SHA256
        and sidecar.get("campaign_export_scope") == "chronological-full-train-index-feature-only"
        and sidecar.get("feature_only") is True
        and sidecar.get("inference_precision") == "fp32"
        and sidecar.get("row_groups") == source_order
        and sidecar.get("row_groups_sha256") == source_hash
        and sidecar.get("prediction_sha256") == prediction_sha,
        f"{run}: surviving native prediction sidecar does not prove full-index FP32 train export",
    )
    _require(
        not prediction_path.exists(), f"{run}: native teacher prediction bank was not discarded"
    )
    return {
        "run": run,
        "checkpoint_sha256": checkpoint_sha,
        "provenance_sha256": provenance_sha,
        "sidecar_sha256": file_sha256(sidecar_path),
        "prediction_sha256": prediction_sha,
        "teacher_panel_sha256": panel_hash,
        "teacher_source_binding_sha256": identity.get("source_binding_sha256"),
        "teacher_data_binding_sha256": identity.get("data_binding_sha256"),
        "fit_receipt_sha256": file_sha256(fit_receipt_path),
        "export_receipt_sha256": file_sha256(export_receipt_path),
        "export_started_at_epoch": export_receipt.get("started_at_epoch"),
        "export_finished_at_epoch": export_receipt.get("finished_at_epoch"),
        "prediction_bank_recomputed": False,
        "floating_model_parameters": model_count,
        "ancestor_count": len(closure),
    }


def _check_fit_receipt(
    receipt: dict[str, Any],
    path: Path,
    cell: dict[str, Any],
    run: str,
    panel_hash: str,
    identity: dict[str, Any],
    *,
    family: str,
    teacher_panel: dict[str, Any],
) -> None:
    """Require each family's frozen output schema and a successful owned fit."""
    owner = receipt.get("owner", {})
    argv = cell.get("fit_argv")
    _require(
        isinstance(argv, list) and all(isinstance(token, str) for token in argv),
        f"{run}: frozen teacher fit argv is missing",
    )
    if family == "wide_gru":
        paths = teacher_panel.get("paths", {})
        project = Path(str(paths.get("project_root", "")))
        _require(
            teacher_panel.get("schema") == "wnn.l4-teacher-capacity-gru.v1"
            and project.is_absolute()
            and cell.get("run_name") == run
            and _option_value(argv, "--name", run) == run,
            f"{run}: frozen native GRU run/name/project identity differs",
        )
        output = project / "artifacts" / "runs" / run
        explicit_output = cell.get("fit_output")
        _require(
            explicit_output is None
            or (
                isinstance(explicit_output, str)
                and Path(explicit_output).resolve() == output.resolve()
            ),
            f"{run}: optional native GRU output differs from its canonical run path",
        )
    elif family == "causal_transformer":
        output_value = cell.get("fit_output")
        _require(
            isinstance(output_value, str)
            and Path(output_value).is_absolute()
            and Path(output_value).name == run
            and _option_value(argv, "--output", run) == output_value,
            f"{run}: Transformer requires explicit fit_output matching its frozen --output",
        )
        output = Path(output_value)
    else:
        raise ValueError(f"{run}: unknown teacher family {family}")
    _require(
        receipt.get("status") == "completed"
        and receipt.get("exit_code") == 0
        and receipt.get("timed_out") is False
        and owner.get("cell_id") == cell.get("cell_id")
        and owner.get("run_name") in (None, run)
        and owner.get("panel_sha256", owner.get("cohort_panel_sha256")) == panel_hash
        and receipt.get("command") == cell.get("fit_argv")
        and Path(str(receipt.get("output_dir", ""))).resolve() == output.resolve()
        and owner.get("source_binding_sha256", identity.get("source_binding_sha256"))
        == identity.get("source_binding_sha256")
        and owner.get("data_binding_sha256", identity.get("data_binding_sha256"))
        == identity.get("data_binding_sha256"),
        f"{run}: actual fit receipt ownership/command/output differs: {path}",
    )


def _check_gru_argv(argv: list[str], args: dict[str, Any], run: str, split_path: Path) -> None:
    """Confirm native fit command fields that define the fixed FP32 teacher recipe."""
    expected = {
        "--precision": "fp32",
        "--name": run,
        "--checkpoint-policy": "fixed",
        "--checkpoint-epoch": "3",
        "--epochs": "4",
        "--train-source": "train",
        "--train-seqs": "10325",
        "--seed": str(args["seed"]),
        "--hidden": str(args["hidden"]),
        "--layers": str(args["layers"]),
        "--cell": "gru",
        "--optim": "muon",
        "--muon-scope": "rnn",
        "--batch-order": "seeded",
        "--loss": "wmse",
        "--batch": "32",
        "--chunk": "1000",
        "--lr": "0.001",
        "--sched": "onecycle",
        "--split-manifest": str(split_path),
        "--warmup-frac": "0.05",
        "--wd": "0.01",
        "--beta2": "0.999",
        "--clip": "1",
        "--loss-norm": "chunk",
        "--aux-next": "0.3",
        "--aux-struct": "0.3",
        "--aux-weight": "0.5",
        "--train-eval-seqs": "32",
    }
    for flag, value in expected.items():
        _require(
            argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value,
            f"{run}: fit argv {flag} differs",
        )
    _require(
        "--init" not in argv and "--precision" in argv,
        f"{run}: fit command has parent or precision drift",
    )


def _check_transformer_record(
    run: str, saved: dict[str, Any], teacher: dict[str, Any], split: Any, project: Path
) -> dict[str, Any]:
    """Revalidate event211 checkpoint/profile/normalizer/export provenance after bank deletion."""
    export = project / "cache" / "l4_teacher_sequence_exports" / run
    record, entries = verify_sequence_teacher(
        run, split, project, export, Path(teacher["binding"]["teacher_panel"])
    )
    cell = next(cell for cell in teacher["teacher_panel"]["cells"] if cell.get("run_name") == run)
    fit_receipt_path = Path(teacher["binding"]["fit_receipts"][run])
    fit_identity = dict(teacher["teacher_panel"].get("identity", {}))
    fit_identity["project_root"] = str(project)
    _check_fit_receipt(
        _read_json(fit_receipt_path),
        fit_receipt_path,
        cell,
        run,
        teacher["teacher_panel_hash"],
        fit_identity,
        family="causal_transformer",
        teacher_panel=teacher["teacher_panel"],
    )
    run_index = list(EXPECTED_RUNS["causal_transformer"]).index(run) + 2
    export_receipt_path = (
        Path(teacher["continuation_panel"]["paths"]["runtime_root"])
        / "receipts"
        / "exports"
        / f"causal_transformer_export_{run_index:02d}_{run}.json"
    )
    export_receipt = _check_transformer_export_receipt(
        export_receipt_path, teacher, run, export, project, split.path
    )
    checkpoint = torch.load(Path(record["checkpoint_path"]), map_location="cpu", weights_only=True)
    floating_parameters = _assert_fp32_state(checkpoint, "model", run)
    data = checkpoint["data"]
    _require(
        data.get("normalization", {}).get("scope") == "all_fit"
        and data.get("normalization", {}).get("sequence_count") == TRAIN_SEQUENCES,
        f"{run}: normalizer was not fit on all official train members",
    )
    manifest = _read_json(Path(record["prediction_manifest_path"]))
    ordered_hashes = [entries[group].get("sha256") for group in split.train_groups]
    _require(
        all(entries[group].get("file") == f"train-{group}.npy" for group in split.train_groups),
        f"{run}: sequence export filename mapping differs",
    )
    _require(
        all(isinstance(value, str) and len(value) == 64 for value in ordered_hashes),
        f"{run}: per-sequence export hashes are malformed",
    )
    files_hash = hashlib.sha256("".join(ordered_hashes).encode()).hexdigest()
    _require(
        all(saved.get(key) == value for key, value in record.items())
        and saved.get("prediction_files") == TRAIN_SEQUENCES
        and saved.get("prediction_files_sha256") == files_hash
        and saved.get("bank_discarded_after_aggregation") is True,
        f"{run}: target teacher record differs from the actual R4 checkpoint/export",
    )
    _require(
        manifest.get("membership") == "fit" and manifest.get("complete") is True,
        f"{run}: sequence export is not complete fit-only",
    )
    leftovers = list(export.glob("train-*.npy"))
    _require(not leftovers, f"{run}: sequence export prediction banks were not discarded")
    return {
        "run": run,
        "checkpoint_sha256": record["checkpoint_sha256"],
        "config_file_sha256": record["config_file_sha256"],
        "normalized_config_sha256": record["normalized_config_sha256"],
        "profile_receipt_sha256": record["profile_receipt_sha256"],
        "prediction_manifest_sha256": record["prediction_manifest_sha256"],
        "prediction_files_sha256": files_hash,
        "teacher_panel_sha256": teacher["teacher_panel_hash"],
        "teacher_source_binding_sha256": teacher["teacher_panel"]
        .get("identity", {})
        .get("source_binding_sha256"),
        "teacher_data_binding_sha256": teacher["teacher_panel"]
        .get("identity", {})
        .get("data_binding_sha256"),
        "fit_receipt_sha256": file_sha256(fit_receipt_path),
        "export_receipt_sha256": file_sha256(export_receipt_path),
        "export_started_at_epoch": export_receipt.get("started_at_epoch"),
        "export_finished_at_epoch": export_receipt.get("finished_at_epoch"),
        "prediction_bank_recomputed": False,
        "floating_model_parameters": floating_parameters,
        "fit_rows": TRAIN_SEQUENCES,
        "normalizer_scope": "all_fit",
    }


def _check_transformer_export_receipt(
    receipt_path: Path,
    teacher: dict[str, Any],
    run: str,
    export: Path,
    project: Path,
    split_path: Path,
) -> dict[str, Any]:
    """Bind the fit-only sequence exporter to its continuation-owned command."""
    receipt = _read_json(receipt_path)
    owner = receipt.get("owner", {})
    panel = teacher["continuation_panel"]
    teacher_panel = teacher["teacher_panel"]
    teacher_source = teacher_panel["paths"]["source_root"]
    expected_suffix = [
        str(Path(teacher_panel["paths"]["python"])),
        str(Path(teacher_source) / "scripts/predict_sequence.py"),
        "--checkpoint",
        str(project / "artifacts" / "runs" / run / "model.pt"),
        "--data-root",
        teacher_panel["paths"]["data_root"],
        "--split-manifest",
        str(split_path),
        "--output",
        str(export),
        "--membership",
        "fit",
        "--batch-size",
        "32",
    ]
    command = receipt.get("command")
    cell_id = f"causal_transformer_export_{list(EXPECTED_RUNS['causal_transformer']).index(run) + 2:02d}_{run}"
    affinity = panel["families"]["causal_transformer"].get(
        "export_cpu_affinity", list(range(16, 24))
    )
    _require(
        receipt.get("status") == "completed"
        and receipt.get("exit_code") == 0
        and receipt.get("timed_out") is False
        and owner.get("continuation_panel_sha256") == teacher["continuation_panel_hash"]
        and owner.get("cell_id") == cell_id
        and owner.get("source_binding_sha256")
        == teacher_panel.get("identity", {}).get("source_binding_sha256")
        and owner.get("data_binding_sha256")
        == teacher_panel.get("identity", {}).get("data_binding_sha256")
        and isinstance(command, list)
        and Path(command[0]).name == "taskset"
        and command[1:3] == ["-c", ",".join(map(str, affinity))]
        and command[-len(expected_suffix) :] == expected_suffix
        and owner.get("command_sha256")
        == hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()
        and Path(str(receipt.get("output_dir", ""))).resolve() == export.resolve(),
        f"{run}: actual fit-only sequence export receipt differs",
    )
    return receipt


def _check_gru_record_adapter(
    run: str,
    saved: dict[str, Any],
    teacher: dict[str, Any],
    split: Any,
    project: Path,
    source_order: list[int],
    fit_order: list[int],
) -> dict[str, Any]:
    """Bridge stable R2 GRU record names to the canonical checkpoint and sidecar checks."""
    result = _check_gru_record(run, saved, teacher, split, project, source_order, fit_order)
    _require(
        saved.get("prediction_row_groups_sha256")
        == sequence_hash(np.asarray(source_order, dtype="<i8")),
        f"{run}: prediction row order hash differs",
    )
    _require(saved.get("fit_rows") == len(fit_order), f"{run}: recorded fit row count differs")
    return result


def _system_memory() -> dict[str, int | None]:
    """Report Linux memory limits without adding a platform dependency."""
    values: dict[str, int | None] = {
        "total_bytes": None,
        "available_bytes": None,
        "reviewer_peak_rss_bytes": None,
    }
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        parsed = {}
        for line in meminfo.read_text(encoding="ascii").splitlines():
            key, _, rest = line.partition(":")
            fields = rest.split()
            if fields and fields[0].isdigit():
                parsed[key] = int(fields[0]) * 1024
        values["total_bytes"] = parsed.get("MemTotal")
        values["available_bytes"] = parsed.get("MemAvailable")
    if resource is not None:
        values["reviewer_peak_rss_bytes"] = int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
    return values


def grade_cache(panel_path: Path, family: str) -> dict[str, Any]:
    """Grade finalized cache lineage, fit rows, hardlinks and actual resource headroom."""
    panel, panel_hash, teacher = _check_panel(panel_path, family)
    paths = panel["paths"]
    binding = teacher["binding"]
    project = Path(paths["project_root"]).resolve(strict=True)
    source_root = Path(paths["source_root"]).resolve(strict=True)
    source_hashes = _hash_source_snapshot(source_root)
    cache = Path(binding["target_cache"]).resolve(strict=True)
    _require(
        not Path(binding["target_cache"]).is_symlink(), "target cache path must not be a symlink"
    )
    _require(cache == Path(binding["target_cache"]).resolve(), "target cache path is not canonical")
    student_cells_sha256 = _check_student_contract(panel, family, cache)
    split = load_campaign_split(Path(paths["split_manifest"]), project)
    _require(
        split.schema == CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA and split.sha256 == SPLIT_SHA256,
        "official chronological split differs",
    )
    _require(
        len(split.train_groups) == TRAIN_SEQUENCES
        and not split.valid_groups
        and len(split.evaluation_groups) == 1873,
        "official split counts differ",
    )
    _require(
        row_groups_sha256(list(split.train_groups)) == FIT_MEMBERS_SHA256,
        "official fit-member digest differs",
    )
    old_root, old_metadata = _check_base_cache(panel, split)
    fbc_initializer_path = project / "artifacts" / "runs" / "tl3c_fbc1k_s5" / "model_ep3.pt"
    _require(
        file_sha256(fbc_initializer_path) == FBC_INIT_SHA256,
        "qualified FBC seed-5 initializer hash differs",
    )
    result = validate_distillation_cache(cache, project, split, "source", require_families=True)
    metadata, files = result["metadata"], result["metadata"]["files"]
    target_contract = metadata.get("target_contract", {})
    _require(
        metadata.get("cache_profile") == "l4_teacher_capacity_stage2_stream_v2",
        "target cache profile is not R2",
    )
    _require(
        metadata.get("source_coefficients") == old_metadata.get("source_coefficients")
        and metadata.get("sources") == old_metadata.get("sources"),
        "retained historical metadata was altered or relabeled",
    )
    _require(
        metadata.get("source_coefficients_scope")
        == "unchanged PR60 base target and retained auxiliary heads",
        "old coefficient scope is mislabeled",
    )
    _require(
        target_contract.get("family") == family
        and target_contract.get("formula") == TARGET_FORMULA,
        "stage-two target formula/family differs",
    )
    _require(
        target_contract.get("teacher_runs_equal_weight") == list(EXPECTED_RUNS[family])
        and target_contract.get("teacher_epoch_zero_based") == 3,
        "teacher order/endpoint differs",
    )
    _require(
        target_contract.get("teacher_mean")
        == "ordered float32 clipped sum divided by teacher count",
        "teacher mean arithmetic differs",
    )
    _require(
        target_contract.get("clipping")
        == {"lower": -2.0, "upper": 2.0, "applied_per_teacher_before_mean": True},
        "per-teacher clipping differs",
    )
    _require(
        target_contract.get("base_weight") == 0.8
        and target_contract.get("new_family_weight") == 0.2
        and target_contract.get("new_family_source_scale") == 0.8125,
        "target mixture coefficients differ",
    )
    _require(
        metadata.get("base_cache", {}).get("metadata_sha256") == PR60_CACHE_METADATA_SHA256
        and metadata.get("base_cache", {}).get("main_source_train_sha256") == PR60_MAIN_TRAIN_SHA256
        and metadata.get("base_cache", {}).get("families_train_sha256")
        == PR60_FAMILIES_TRAIN_SHA256,
        "target cache does not bind qualified PR60 base",
    )
    _require(
        Path(metadata.get("base_cache", {}).get("path", "")).resolve() == old_root,
        "target base cache path differs",
    )
    fit_order = list(map(int, split.train_groups))
    _require(
        target_contract.get("fit_groups") == fit_order
        and target_contract.get("fit_groups_sha256") == row_groups_sha256(fit_order),
        "target fit row order/hash differs",
    )
    source_order = _target_order(project, "train")
    validate_chronological_train_index_order(project, source_order, split, "train")
    _require(
        len(source_order) == SOURCE_TRAIN_INDEX_SEQUENCES,
        "full official training source index count differs",
    )
    source_position_by_group = {group: index for index, group in enumerate(source_order)}
    _require(
        len(source_position_by_group) == len(source_order), "full source index repeats row groups"
    )
    expected_positions = [source_position_by_group[group] for group in fit_order]
    positions = target_contract.get("fit_source_positions")
    positions_array = np.asarray(positions, dtype="<i8")
    _require(
        target_contract.get("source_train_index_rows") == len(source_order)
        and target_contract.get("source_train_index_sha256")
        == sequence_hash(np.asarray(source_order, dtype="<i8")),
        "full source index order/hash differs",
    )
    _require(
        isinstance(positions, list)
        and positions == expected_positions
        and target_contract.get("fit_source_positions_sha256") == positions_sha256(positions_array),
        "explicit source-index to fit-row mapping differs",
    )
    _require(
        target_contract.get("fit_source_mapping")
        == "for each fit-group target row, select this index from the full chronological train prediction bank",
        "source-to-fit mapping description differs",
    )
    _require(
        metadata.get("prediction_banks_persisted") is False,
        "target cache claims prediction banks persist",
    )
    _require(
        metadata.get("student_contract") == STUDENT_CONTRACT,
        "student endpoint/initializer/loss contract differs",
    )
    _require(
        files["main_source_train"]["row_groups"] == fit_order
        and files["main_source_train"]["shape"] == [TRAIN_SEQUENCES, SEQUENCE_ROWS, 2]
        and files["main_source_train"]["dtype"] == "float32",
        "new main target shape/type/order differs",
    )
    _require(
        files["main_source_valid"]["row_groups"] == []
        and files["main_source_valid"]["shape"] == [0, SEQUENCE_ROWS, 2]
        and files["main_source_valid"]["dtype"] == "float32",
        "new primary-valid target is not empty",
    )
    old_files = old_metadata["files"]
    link_checks = {}
    retained = metadata.get("retained_auxiliary_targets", {})
    _require(
        retained.get("source_cache") == str(old_root)
        and retained.get("family_order") == old_metadata.get("family_order")
        and retained.get("train_sha256") == PR60_FAMILIES_TRAIN_SHA256
        and retained.get("train_hardlink_reused_without_writes") is True
        and retained.get("valid_sha256") == old_metadata["files"]["families_valid"]["sha256"],
        "retained auxiliary target lineage metadata differs",
    )
    for split_name in ("train", "valid"):
        key = f"families_{split_name}"
        old_entry, new_entry = old_files[key], files[key]
        old_path = old_root / old_entry["path"]
        new_path = cache / new_entry["path"]
        same_inode = (old_path.stat().st_dev, old_path.stat().st_ino) == (
            new_path.stat().st_dev,
            new_path.stat().st_ino,
        )
        _require(
            same_inode and file_sha256(old_path) == old_entry["sha256"] == file_sha256(new_path),
            f"retained {key} is not the exact immutable PR60 hardlink",
        )
        _require(
            {key: value for key, value in new_entry.items() if key != "path"}
            == {key: value for key, value in old_entry.items() if key != "path"},
            f"retained {key} metadata changed",
        )
        link_checks[key] = {
            "same_inode": same_inode,
            "sha256": new_entry["sha256"],
            "bytes": new_entry["bytes"],
        }
    teacher_records = target_contract.get("teachers")
    _require(
        isinstance(teacher_records, list)
        and [row.get("run") for row in teacher_records] == list(EXPECTED_RUNS[family]),
        "new teacher records missing or unordered",
    )
    _require(len(teacher_records) == len(EXPECTED_RUNS[family]), "new teacher count differs")
    teacher_results = []
    for run, record in zip(EXPECTED_RUNS[family], teacher_records, strict=True):
        _require(
            record.get("bank_discarded_after_aggregation") is True,
            f"{run}: bank deletion receipt is missing",
        )
        if family == "wide_gru":
            teacher_results.append(
                _check_gru_record_adapter(
                    run, record, teacher, split, project, source_order, fit_order
                )
            )
        else:
            teacher_results.append(_check_transformer_record(run, record, teacher, split, project))
    stage2_receipts = _check_stage2_receipts(panel, panel_hash, family, cache)
    _require(
        len([item for item in stage2_receipts if item["action"] == "add"]) == len(teacher_results)
        and len([item for item in stage2_receipts if item["action"] == "init"]) == 1
        and len([item for item in stage2_receipts if item["action"] == "finalize"]) == 1
        and all(
            item.get("action") == "init"
            or (
                isinstance(item.get("started_at_epoch"), (float, int))
                and isinstance(item.get("finished_at_epoch"), (float, int))
                and item["started_at_epoch"] <= item["finished_at_epoch"]
            )
            for item in stage2_receipts
        ),
        "stage-two command receipt timing is incomplete",
    )
    init_receipt = next(item for item in stage2_receipts if item["action"] == "init")
    if init_receipt.get("finished_at_epoch") is not None:
        _require(
            init_receipt["finished_at_epoch"] <= teacher_results[0]["export_started_at_epoch"],
            "teacher export started before target initialization completed",
        )
    add_receipts = [item for item in stage2_receipts if item["action"] == "add"]
    finalize_receipt = next(item for item in stage2_receipts if item["action"] == "finalize")
    for index, item in enumerate(teacher_results):
        add_receipt = add_receipts[index]
        _require(
            isinstance(item.get("export_started_at_epoch"), (float, int))
            and isinstance(item.get("export_finished_at_epoch"), (float, int))
            and item["export_started_at_epoch"] <= item["export_finished_at_epoch"]
            and item["export_finished_at_epoch"] <= add_receipt["started_at_epoch"]
            and add_receipt["finished_at_epoch"]
            <= (
                teacher_results[index + 1]["export_started_at_epoch"]
                if index + 1 < len(teacher_results)
                else finalize_receipt["started_at_epoch"]
            ),
            f"{item['run']}: teacher exports and target aggregation were not streamed one at a time",
        )
    cache_review_path = Path(binding["cache_admission_receipt"])
    free_now = shutil.disk_usage(Path(paths["disk_mount"])).free
    shared_state = _read_json(Path(paths["shared_state"]))
    initial_free_evidence = _check_initial_free_basis(shared_state, Path(paths["runtime_root"]))
    free_initial = initial_free_evidence["free_bytes"]
    free_delta_proxy = max(0, free_initial - free_now)
    _require(free_now >= MIN_FREE_BYTES, "data2 free space fell below the frozen 3 GB floor")
    _require(
        free_delta_proxy <= OUTPUT_CAP_BYTES,
        "observed campaign free-space delta exceeds the 6.5 GB output cap",
    )
    cache_unique_bytes = 0
    old_inodes = {
        (path.stat().st_dev, path.stat().st_ino) for path in old_root.iterdir() if path.is_file()
    }
    seen = set(old_inodes)
    for path in cache.iterdir():
        _require(not path.is_symlink(), f"target cache contains a symlink: {path.name}")
        if path.is_file():
            stat = path.stat()
            inode = (stat.st_dev, stat.st_ino)
            if inode not in seen:
                cache_unique_bytes += (
                    stat.st_blocks * 512 if hasattr(stat, "st_blocks") else stat.st_size
                )
                seen.add(inode)
    _require(
        cache_unique_bytes <= OUTPUT_CAP_BYTES,
        "new family target cache alone exceeds the 6.5 GB output cap",
    )
    expected_cache_files = {
        "metadata.json",
        "stage2_build_receipt.json",
        "main_source_train.npy",
        "main_source_valid.npy",
        "families_train.npy",
        "families_valid.npy",
    }
    _require(
        {path.name for path in cache.iterdir()} == expected_cache_files,
        "finalized target cache contains unreviewed or missing files",
    )
    main_path = cache / files["main_source_train"]["path"]
    build_receipt_path = cache / "stage2_build_receipt.json"
    build_receipt = _read_json(build_receipt_path)
    _require(
        build_receipt.get("schema") == "wnn.l4-teacher-stage2-stream.v2"
        and build_receipt.get("family") == family
        and build_receipt.get("expected_runs") == list(EXPECTED_RUNS[family])
        and build_receipt.get("processed_runs") == list(EXPECTED_RUNS[family])
        and build_receipt.get("pending_run") is None
        and build_receipt.get("teacher_records") == teacher_records
        and build_receipt.get("split_manifest_sha256") == SPLIT_SHA256
        and build_receipt.get("fit_row_groups") == fit_order
        and build_receipt.get("fit_row_groups_sha256") == row_groups_sha256(fit_order)
        and build_receipt.get("source_train_index_row_groups") == source_order
        and build_receipt.get("fit_source_positions") == positions,
        "finalized target does not match its completed R2 stream receipt",
    )
    _require(
        build_receipt.get("old_cache_metadata_sha256") == PR60_CACHE_METADATA_SHA256
        and build_receipt.get("old_main_sha256") == PR60_MAIN_TRAIN_SHA256
        and Path(build_receipt.get("old_cache", "")).resolve() == old_root
        and build_receipt.get("target_array_bytes") == TARGET_BYTES
        and build_receipt.get("incoming_bank_max_bytes") == SEQUENCE_BANK_MAX_BYTES,
        "R2 build receipt does not bind its exact PR60 input and bounded stream sizes",
    )
    started_free = int(build_receipt.get("started_free_bytes", 0))
    stream_bank = NATIVE_BANK_MAX_BYTES if family == "wide_gru" else SEQUENCE_BANK_MAX_BYTES
    stream_peak = TARGET_BYTES + stream_bank + STREAM_METADATA_OVERHEAD_BYTES
    stream_peak_delta = max(0, free_initial - started_free) + stream_peak
    stream_free_floor = started_free - stream_peak
    _require(
        started_free > 0
        and stream_free_floor >= MIN_FREE_BYTES
        and stream_peak_delta <= OUTPUT_CAP_BYTES,
        "R2 stream peak violates the global output cap or 3 GB free-space floor",
    )
    _require(
        files["main_source_train"]["bytes"] == TARGET_BYTES, "new main target byte count differs"
    )
    main = np.load(main_path, mmap_mode="r", allow_pickle=False)
    finite_rows = 0
    for start in range(0, TRAIN_SEQUENCES, 32):
        values = main[start : start + 32]
        _require(
            values.dtype == np.float32 and np.isfinite(values).all(),
            "final target includes non-FP32 or nonfinite values",
        )
        finite_rows += len(values)
    del main
    peak_rss = (
        int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        if resource is not None
        else None
    )
    receipt = {
        "schema": "wnn.l4-stage2-cache-independent-review.v2",
        "status": "approved_stage2_cache_lineage_formula_not_numerically_recomputed",
        "family": family,
        "continuation_panel_sha256": panel_hash,
        "continuation_panel_path": str(panel_path.resolve()),
        "reviewer_source_path": str(Path(__file__).resolve()),
        "reviewer_source_sha256": file_sha256(Path(__file__).resolve()),
        "target_cache_path": str(cache),
        "target_metadata_sha256": result["metadata_sha256"],
        "source_binding_sha256": panel["identity"]["source_binding_sha256"],
        "data_binding_sha256": panel["identity"]["data_binding_sha256"],
        "split_sha256": SPLIT_SHA256,
        "fit_members_sha256": FIT_MEMBERS_SHA256,
        "student_cells_sha256": student_cells_sha256,
        "teacher_panel_sha256": teacher["teacher_panel_hash"],
        "stage2_handoff_r2_sha256": R2_HANDOFF_SHA256,
        "teacher_admission_receipt_sha256": file_sha256(Path(binding["teacher_admission_receipt"])),
        "r2_stage2_source_hashes": source_hashes,
        "stage2_owned_receipt_sha256": stage2_receipts,
        "stage2_build_receipt_sha256": file_sha256(build_receipt_path),
        "qualified_pr60_base": {
            "metadata_sha256": result["metadata"]["base_cache"]["metadata_sha256"],
            "main_source_train_sha256": PR60_MAIN_TRAIN_SHA256,
            "families_train_sha256": PR60_FAMILIES_TRAIN_SHA256,
        },
        "target_formula": TARGET_FORMULA,
        "formula_verification": "frozen source/command/teacher record/codepath bound; numeric replay unavailable because input prediction banks were discarded",
        "teacher_records": teacher_results,
        "qualified_initializer": {
            "run": "tl3c_fbc1k_s5",
            "checkpoint": "model_ep3.pt",
            "checkpoint_sha256": file_sha256(fbc_initializer_path),
            "baseline_ancestry_sha256": panel["identity"]["baseline_ancestry_sha256"],
        },
        "row_mapping": {
            "full_source_rows": len(source_order),
            "full_source_index_sha256": sequence_hash(np.asarray(source_order, dtype="<i8")),
            "fit_rows": len(fit_order),
            "fit_row_groups_sha256": row_groups_sha256(fit_order),
            "fit_source_positions_sha256": positions_sha256(positions_array),
        },
        "retained_auxiliary_hardlinks": link_checks,
        "target_arrays": {
            "main_source_train_sha256": files["main_source_train"]["sha256"],
            "main_source_train_bytes": files["main_source_train"]["bytes"],
            "main_source_valid_sha256": files["main_source_valid"]["sha256"],
            "finite_rows_checked": finite_rows,
        },
        "storage": {
            "initial_free_bytes": free_initial,
            "initial_free_observation": initial_free_evidence,
            "free_at_review_bytes": free_now,
            "free_space_delta_proxy_bytes": free_delta_proxy,
            "free_floor_bytes": MIN_FREE_BYTES,
            "output_cap_bytes": OUTPUT_CAP_BYTES,
            "new_unique_cache_allocated_bytes": cache_unique_bytes,
            "stream_transient_upper_bound_bytes": stream_peak,
            "campaign_peak_delta_upper_bound_bytes": stream_peak_delta,
            "campaign_peak_free_lower_bound_bytes": free_initial - stream_peak_delta,
            "global_cap_measured_by_free_delta_proxy": True,
            "prediction_banks_persisted": False,
        },
        "memory": {**_system_memory(), "reviewer_peak_rss_bytes": peak_rss},
        "review_limits": [
            "No teacher checkpoint was changed.",
            "No prediction bank values were available for independent numerical replay after the frozen streaming builder deleted them.",
        ],
        "created_unix": time.time(),
    }
    _require(
        not cache_review_path.exists() and not cache_review_path.is_symlink(),
        "cache admission receipt already exists; preserve it",
    )
    _write_json(cache_review_path, receipt)
    return receipt


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically publish one independent receipt outside the finalized cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    """Grade one stage-two family cache and write the runner-compatible receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation-panel", required=True, type=Path)
    parser.add_argument("--family", choices=tuple(EXPECTED_RUNS), required=True)
    args = parser.parse_args()
    receipt = grade_cache(args.continuation_panel.resolve(strict=True), args.family)
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "status",
                    "family",
                    "target_cache_path",
                    "target_metadata_sha256",
                    "source_binding_sha256",
                    "data_binding_sha256",
                    "split_sha256",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
