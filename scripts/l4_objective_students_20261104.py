"""Select a teacher arm and preserve a bounded, no-launch student handoff."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import cast

SOURCE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SOURCE), str(SOURCE / "src")]

from scripts import l4_objective_20261104 as objective_adapter  # noqa: E402
from scripts.l4_teacher_capacity_20261104 import (  # noqa: E402
    continue_students,  # noqa: E402
    runner,  # noqa: E402
)
from wnn.multistage_search import atomic_write_json  # noqa: E402
from wnn.splits import CampaignSplit, file_sha256, load_campaign_split  # noqa: E402
from wnn.training.chronological_teacher import validate_chronological_teacher_membership  # noqa: E402

SPLIT = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
SEEDS = {1001, 1002}
PROFILE_WARMUP = 65
PROFILE_TIMED = 3
THRESHOLD = 0.001
TIE_TOLERANCE = 1e-6
COMMON_FIT_IDENTITY = ("data_binding_sha256", "fit_members_sha256", "split_sha256")


def read_json(path: Path) -> dict[str, object]:
    """Read a bounded regular JSON object."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 << 20:
        raise ValueError(f"expected a regular JSON file no larger than 16 MiB: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, object], value)


def admitted(
    panel_path: Path, authority_path: Path, admission_path: Path
) -> tuple[dict[str, object], str]:
    """Bind actual panel bytes, root authority and source admission."""
    panel, _, digest, _, _, _ = objective_adapter.validate_phase(
        panel_path, authority_path, admission_path, "profile", None
    )
    identity = cast(dict[str, object], panel["identity"])
    if identity.get("split_sha256") != SPLIT:
        raise ValueError("source admission does not bind the frozen chronological split")
    return panel, digest


def _valid_endpoint(history: object, valid_sequences: int, scored_rows: int) -> dict[str, object]:
    """Select fixed epoch 3 only when its official-valid metrics cover the manifest."""
    rows = (
        [row for row in history if isinstance(row, dict) and row.get("epoch") == 3]
        if isinstance(history, list)
        else []
    )
    if len(rows) != 1:
        raise ValueError("history must contain exactly one fixed epoch-3 metric row")
    row = cast(dict[str, object], rows[0])
    if (
        row.get("metric_id") != "global_wp_clip2_v1"
        or row.get("blocks") != valid_sequences
        or row.get("selected_rows") != scored_rows
    ):
        raise ValueError("epoch-3 metric does not cover the exact official-valid manifest")
    return row


def _check_shared_data_identity(
    objective_panel: dict[str, object], control_panel: dict[str, object]
) -> None:
    """Require same data, fit membership and split while preserving source identity per panel."""
    objective_identity = cast(dict[str, object], objective_panel.get("identity", {}))
    control_identity = cast(dict[str, object], control_panel.get("identity", {}))
    if any(
        objective_identity.get(key) is None
        or objective_identity.get(key) != control_identity.get(key)
        for key in COMMON_FIT_IDENTITY
    ):
        raise ValueError("PR61 controls and objective teachers differ in data/split/fit membership")


def teacher_result(
    panel: dict[str, object], panel_sha: str, cell: dict[str, object], split: CampaignSplit
) -> dict[str, object]:
    """Bind one fixed-epoch score to its owned fit, history, provenance and checkpoint."""
    run, cell_id = str(cell["run_name"]), str(cell["cell_id"])
    receipt_path = runner.panel_path(panel, cell, "fit")["receipt"]
    receipt = read_json(receipt_path)
    owner = cast(dict[str, object], receipt.get("owner", {}))
    if (
        receipt.get("status") != "completed"
        or receipt.get("exit_code") != 0
        or receipt.get("timed_out") is not False
        or receipt.get("command") != cell["fit_argv"]
        or owner.get("panel_sha256") != panel_sha
        or owner.get("cell_id") != cell_id
        or owner.get("run_name") != run
    ):
        raise ValueError(f"{cell_id}: owned fit receipt does not bind the panel/cell")
    root = Path(str(panel["paths"]["project_root"])) / "artifacts" / "runs" / run
    history_path, provenance_path, checkpoint = (
        root / "history.json",
        root / "provenance.json",
        root / "model_ep3.pt",
    )
    if any(p.is_symlink() or not p.is_file() for p in (history_path, provenance_path, checkpoint)):
        raise FileNotFoundError(f"{run}: fixed history/provenance/checkpoint is missing")
    if history_path.stat().st_size > 64 << 20:
        raise ValueError(f"{run}: history exceeds reader bound")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    provenance = read_json(provenance_path)
    validate_chronological_teacher_membership(provenance, split, run)
    identity = cast(dict[str, object], panel.get("identity", {}))
    source_binding = identity.get("source_binding_sha256")
    data_binding = identity.get("data_binding_sha256")
    fit_members = identity.get("fit_members_sha256")
    if any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in (source_binding, data_binding, fit_members)
    ):
        raise ValueError(f"{run}: panel identity lacks source/data/fit SHA256 bindings")
    args = cast(dict[str, object], provenance.get("args", {}))
    fixed = {
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 3,
        "epochs": 4,
        "seed": cell["seed"],
        "train_source": "train",
        "eval_source": "dev",
        "full_valid": True,
    }
    if (
        any(args.get(k) != v for k, v in fixed.items())
        or args.get("no_eval_during_fit") is not False
    ):
        raise ValueError(f"{run}: provenance is not the fixed full-valid epoch-3 recipe")
    checkpoint_sha = file_sha256(checkpoint)
    hashes = provenance.get("checkpoint_sha256")
    if not isinstance(hashes, dict) or hashes.get("model_ep3.pt") != checkpoint_sha:
        raise ValueError(f"{run}: fixed checkpoint hash differs from provenance")
    panel_identity = cast(dict[str, object], panel.get("identity", {}))
    valid_sequences = panel_identity.get("primary_valid_sequences")
    scored_rows = panel_identity.get("primary_scored_rows")
    if type(valid_sequences) is not int or type(scored_rows) is not int:
        raise ValueError(f"{run}: panel lacks exact official-valid metric counts")
    row = _valid_endpoint(history, valid_sequences, scored_rows)
    score = row.get("weighted_pearson")
    if type(score) not in (int, float) or not math.isfinite(float(score)):
        raise ValueError(f"{run}: epoch-3 full-valid WP is not finite")
    expected = {"clip5": (5.0, "none"), "aux_late": (1.0, "late_linear_to_zero")}.get(
        str(cell.get("configuration_id")), (1.0, "none")
    )
    if args.get("clip") != expected[0] or args.get("aux_anneal", "none") != expected[1]:
        raise ValueError(f"{run}: actual clip/auxiliary treatment differs from its panel")
    return {
        "cell_id": cell_id,
        "configuration_id": cell.get("configuration_id"),
        "seed": cell["seed"],
        "run_name": run,
        "wp": float(score),
        "source_binding_sha256": source_binding,
        "data_binding_sha256": data_binding,
        "fit_members_sha256": fit_members,
        "source_binding_scope": "receiving_panel.identity",
        "artifact_sha256": {
            "fit_receipt": file_sha256(receipt_path),
            "checkpoint": checkpoint_sha,
            "history": file_sha256(history_path),
            "provenance": file_sha256(provenance_path),
        },
        "artifact_paths": {
            "fit_receipt": str(receipt_path),
            "checkpoint": str(checkpoint),
            "history": str(history_path),
            "provenance": str(provenance_path),
        },
    }


def choose_arm(means: dict[str, float], control_mean: float) -> str | None:
    """Apply the frozen two-seed mean, tie, and control-margin rule."""
    if set(means) != {"clip5", "aux_late"}:
        raise ValueError("selection requires exactly the two frozen teacher arms")
    if any(not math.isfinite(value) for value in (*means.values(), control_mean)):
        raise ValueError("selection means must be finite")
    if abs(means["clip5"] - means["aux_late"]) <= TIE_TOLERANCE:
        winner = "clip5"
    else:
        winner = max(means, key=means.__getitem__)
    return winner if means[winner] >= control_mean + THRESHOLD else None


def select(panel_path: Path, authority: Path, admission: Path, control_panel_path: Path) -> Path:
    """Choose the best two-seed arm only when it clears the frozen control margin."""
    panel, panel_sha = admitted(panel_path, authority, admission)
    controls = read_json(control_panel_path)
    identity = cast(dict[str, object], panel["identity"])
    _check_shared_data_identity(panel, controls)
    split = load_campaign_split(
        Path(str(panel["paths"]["split_manifest"])), Path(str(panel["paths"]["project_root"]))
    )
    control_cells = [
        c
        for c in controls["cells"]
        if c.get("role") == "control"
        or c.get("configuration_id") in ("control", "gru256x2_control")
    ]
    candidates = {
        name: [c for c in panel["cells"] if c.get("configuration_id") == name]
        for name in ("clip5", "aux_late")
    }
    if (
        len(control_cells) != 2
        or {c.get("seed") for c in control_cells} != SEEDS
        or any(
            len(rows) != 2 or {c.get("seed") for c in rows} != SEEDS for rows in candidates.values()
        )
    ):
        raise ValueError("selection requires exact seed1001/1002 pairs for both arms and controls")
    control_rows = [
        teacher_result(controls, file_sha256(control_panel_path), c, split) for c in control_cells
    ]
    arm_rows = {
        name: [teacher_result(panel, panel_sha, c, split) for c in cells]
        for name, cells in candidates.items()
    }
    means = {name: sum(float(r["wp"]) for r in rows) / 2 for name, rows in arm_rows.items()}
    control_mean = sum(float(r["wp"]) for r in control_rows) / 2
    selected = choose_arm(means, control_mean)
    receipt = {
        "schema": "wnn.l4-objective-student-selection.v1",
        "status": "selected" if selected else "no_students",
        "objective_panel_sha256": panel_sha,
        "authority_sha256": file_sha256(authority),
        "admission_sha256": file_sha256(admission),
        "control_panel_sha256": file_sha256(control_panel_path),
        "identity": identity,
        "candidate_results": arm_rows,
        "control_results": control_rows,
        "candidate_mean_wp": means,
        "matched_control_mean_wp": control_mean,
        "threshold": THRESHOLD,
        "tie_tolerance": TIE_TOLERANCE,
        "selected_configuration": selected,
        "student_handoff": None
        if selected is None
        else {
            "teacher_cell_ids": [r["cell_id"] for r in arm_rows[selected]],
            "target_formula": "T_new=0.8*T_old+0.2*0.8125*mean(clip(each_teacher,-2,2))",
            "target_writer": "scripts/l4_teacher_capacity_20261104/stream_two_gru_target.py",
            "student_template_family": "wide_gru",
            "student_seeds": [1001, 1002],
            "required_gates": [
                "target-cache admission",
                "actual-shape student profiles",
                "independent profile review",
                "root confirms remaining fit authority",
            ],
        },
    }
    output = (
        Path(str(panel["paths"]["runtime_root"]))
        / "receipts/objective/OBJECTIVE_STUDENT_SELECTION_R1.json"
    )
    atomic_write_json(output, receipt, replace=False)
    return output


def _replace_arg(args: list[str], flag: str, value: str) -> None:
    """Replace one required flag value in an inherited frozen argument vector."""
    positions = [index for index, token in enumerate(args[:-1]) if token == flag]
    if len(positions) != 1:
        raise ValueError(f"student template must contain exactly one {flag}")
    args[positions[0] + 1] = value


def _profile_args(train_args: list[str], profile_name: str) -> list[str]:
    """Match the established bounded profile invocation without saving state."""
    args = train_args.copy()
    if "--save-training-state" in args:
        args.remove("--save-training-state")
    _replace_arg(args, "--name", profile_name)
    args.extend(["--profile-warmup", str(PROFILE_WARMUP), "--profile-steps", str(PROFILE_TIMED)])
    return args


def prepare(
    panel_path: Path,
    authority_path: Path,
    admission_path: Path,
    selection_path: Path,
    amendment_path: Path,
    continuation_panel_path: Path,
    control_panel_path: Path,
) -> Path:
    """Write selected teacher and two-student command handoff; launch nothing."""
    panel, panel_sha = admitted(panel_path, authority_path, admission_path)
    authority, amendment = read_json(authority_path), read_json(amendment_path)
    admission_sha, authority_sha = file_sha256(admission_path), file_sha256(authority_path)
    selection = read_json(selection_path)
    identity = cast(dict[str, object], panel["identity"])
    if (
        selection.get("schema") != "wnn.l4-objective-student-selection.v1"
        or selection.get("status") != "selected"
        or selection.get("objective_panel_sha256") != panel_sha
        or selection.get("authority_sha256") != authority_sha
        or selection.get("admission_sha256") != admission_sha
        or selection.get("identity") != identity
    ):
        raise ValueError("selection receipt does not bind this admitted objective panel")
    controls = read_json(control_panel_path)
    if selection.get("control_panel_sha256") != file_sha256(control_panel_path):
        raise ValueError("selection control panel bytes or chronological split changed")
    _check_shared_data_identity(panel, controls)
    split = load_campaign_split(
        Path(str(panel["paths"]["split_manifest"])), Path(str(panel["paths"]["project_root"]))
    )
    control_cells = [
        cell
        for cell in cast(list[dict[str, object]], controls["cells"])
        if cell.get("role") == "control"
        or cell.get("configuration_id") in ("control", "gru256x2_control")
    ]
    candidate_cells = {
        name: [
            cell
            for cell in cast(list[dict[str, object]], panel["cells"])
            if cell.get("configuration_id") == name
        ]
        for name in ("clip5", "aux_late")
    }
    if len(control_cells) != 2 or {cell.get("seed") for cell in control_cells} != SEEDS:
        raise ValueError("selection requires exact seed1001/1002 control cells")
    verified_controls = [
        teacher_result(controls, file_sha256(control_panel_path), cell, split)
        for cell in control_cells
    ]
    verified_candidates = {
        name: [teacher_result(panel, panel_sha, cell, split) for cell in cells]
        for name, cells in candidate_cells.items()
    }
    if (
        selection.get("control_results") != verified_controls
        or selection.get("candidate_results") != verified_candidates
    ):
        raise ValueError("selection receipt no longer matches the owned fit and score artifacts")
    means = cast(dict[str, float], selection.get("candidate_mean_wp"))
    control_mean = selection.get("matched_control_mean_wp")
    selected = selection.get("selected_configuration")
    actual_means = {
        name: sum(float(result["wp"]) for result in rows) / 2
        for name, rows in verified_candidates.items()
    }
    actual_control_mean = sum(float(result["wp"]) for result in verified_controls) / 2
    if (
        type(control_mean) not in (int, float)
        or means != actual_means
        or float(control_mean) != actual_control_mean
        or choose_arm(means, float(control_mean)) != selected
        or selected not in ("clip5", "aux_late")
    ):
        raise ValueError("selection receipt does not satisfy the frozen tie/control rule")
    deadline = authority.get("hard_deadline_epoch")
    if (
        amendment.get("schema") != "wnn.root-authority-amendment.v1"
        or amendment.get("status") != "approved_allocation_clarification"
        or amendment.get("parent_authority_sha256") != authority_sha
        or amendment.get("unchanged_max_new_scientific_fits") != 6
        or amendment.get("unchanged_hard_deadline_epoch") != deadline
        or amendment.get("teacher_gpu_indices") != [2, 3, 4, 5]
        or amendment.get("selected_target_writer_gpu_index") != 3
        or amendment.get("selected_target_writer_cpu_ids") != list(range(24, 32))
        or amendment.get("conditional_student_gpu_indices") != [0, 1]
        or amendment.get("conditional_student_cpu_ids") != list(range(16))
    ):
        raise ValueError("root allocation amendment does not bind the selected execution lanes")

    template, template_sha = continue_students.validate_panel(continuation_panel_path)
    template_paths = cast(dict[str, object], template["paths"])
    template_cells = cast(list[dict[str, object]], template["students"]["cells"])
    wide_cells = sorted(
        (cell for cell in template_cells if cell.get("family") == "wide_gru"),
        key=lambda cell: int(cell["seed"]),
    )
    if len(wide_cells) != 2 or {cell.get("seed") for cell in wide_cells} != SEEDS:
        raise ValueError("PR61 template must provide exactly two wide-GRU student controls")

    paths = cast(dict[str, object], panel["paths"])
    runtime_root, project_root = Path(str(paths["runtime_root"])), Path(str(paths["project_root"]))
    target_cache = project_root / "cache/l4_objective_selected_gru_transient"
    if target_cache.exists() or target_cache.is_symlink():
        raise FileExistsError(f"preserve existing target-cache path: {target_cache}")
    old_cache = Path(str(template_paths["old_cache"]))
    if not old_cache.is_dir() or old_cache.is_symlink():
        raise FileNotFoundError(f"qualified PR60 cache is unavailable: {old_cache}")
    teacher_rows = cast(dict[str, list[dict[str, object]]], selection["candidate_results"])[
        selected
    ]
    teacher_cells = [str(row["cell_id"]) for row in teacher_rows]
    if len(teacher_cells) != 2 or {int(row["seed"]) for row in teacher_rows} != SEEDS:
        raise ValueError("selected teacher family does not contain exact seed1001/1002 pair")

    python = str(paths["python"])
    source_root = Path(str(paths["source_root"]))
    taskset = "/usr/bin/taskset"
    target_script = source_root / "scripts/l4_teacher_capacity_20261104/stream_two_gru_target.py"
    train_script_path = source_root / "scripts/train_rnn.py"
    if not target_script.is_file() or not train_script_path.is_file():
        raise FileNotFoundError(
            "admitted source snapshot lacks the target writer or student trainer"
        )
    target_argv = [
        taskset,
        "-c",
        ",".join(map(str, amendment["selected_target_writer_cpu_ids"])),
        python,
        str(target_script),
        "--panel",
        str(panel_path.resolve()),
        "--cell-id",
        teacher_cells[0],
        "--cell-id",
        teacher_cells[1],
        "--old-cache",
        str(old_cache),
        "--output-cache",
        str(target_cache),
        "--device",
        "cuda:0",
        "--batch-size",
        "32",
    ]
    writer_gpu = next(
        cell["gpu_uuid"]
        for cell in panel["cells"]
        if cell["gpu_slot"] == amendment["selected_target_writer_gpu_index"]
    )
    train_script = str(train_script_path)
    students: list[dict[str, object]] = []
    amendment_uuids = cast(list[str], amendment["conditional_student_gpu_uuids"])
    authorized_cpus = set(cast(list[int], amendment["conditional_student_cpu_ids"]))
    for template_cell in wide_cells:
        seed = int(template_cell["seed"])
        gpu_index = seed - 1001
        if not 0 <= gpu_index < len(amendment_uuids):
            raise ValueError("student seed does not map to an amended GPU index")
        if template_cell.get("gpu_uuid") != amendment_uuids[gpu_index]:
            raise ValueError("PR61 student GPU UUID differs from the approved allocation amendment")
        run_name = f"l4obj_student_wide_gru_{selected}_s{seed}_20261104"
        train_args = list(cast(list[str], template_cell["train_args"]))
        _replace_arg(train_args, "--name", run_name)
        _replace_arg(train_args, "--distill-cache-root", str(target_cache))
        if "--checkpoint-epoch" not in train_args:
            if "--checkpoint-policy" not in train_args or "fixed" not in train_args:
                raise ValueError("template lacks the fixed checkpoint policy required for epoch 1")
            train_args.extend(["--checkpoint-epoch", "1"])
        elif train_args[train_args.index("--checkpoint-epoch") + 1] != "1":
            raise ValueError("fixed student endpoint must remain checkpoint epoch 1")
        cpus = list(cast(list[int], template_cell["cpu_affinity"]))
        if not set(cpus) <= authorized_cpus:
            raise ValueError("PR61 student CPU affinity escapes the approved CPU allocation")
        affinity = ",".join(map(str, cpus))
        fit_argv = [taskset, "-c", affinity, python, train_script, *train_args]
        profile_name = f"{run_name}_profile_r1"
        profile_argv = [
            taskset,
            "-c",
            affinity,
            python,
            train_script,
            *_profile_args(train_args, profile_name),
        ]
        students.append(
            {
                "cell_id": f"student_wide_gru_s{seed}",
                "family": "wide_gru",
                "seed": seed,
                "run_name": run_name,
                "gpu_index": gpu_index,
                "gpu_uuid": amendment_uuids[gpu_index],
                "cpu_affinity": cpus,
                "cpu_threads": template_cell["cpu_threads"],
                "execution_environment": {
                    "CUDA_VISIBLE_DEVICES": amendment_uuids[gpu_index],
                    "WNN_PROJECT_ROOT": str(project_root),
                    "WNN_THREADS": str(template_cell["cpu_threads"]),
                    "NVIDIA_TF32_OVERRIDE": "0",
                    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
                },
                "train_args": train_args,
                "profile_argv": profile_argv,
                "fit_argv": fit_argv,
            }
        )
    if sum(int(cell["cpu_threads"]) for cell in students) > 16:
        raise ValueError("conditional student cells exceed the amended 16-CPU allocation")
    allocated_cpus = [cpu for cell in students for cpu in cast(list[int], cell["cpu_affinity"])]
    if len(allocated_cpus) != len(set(allocated_cpus)) or set(allocated_cpus) != authorized_cpus:
        raise ValueError("PR61 student controls must partition the amended CPU IDs 0..15")
    output = runtime_root / "receipts/objective/OBJECTIVE_STUDENT_PREPARED_R1.json"
    packet = {
        "schema": "wnn.l4-objective-student-handoff.v1",
        "status": "prepared_no_launch",
        "objective_panel_path": str(panel_path.resolve()),
        "objective_panel_sha256": panel_sha,
        "authority_sha256": authority_sha,
        "admission_sha256": admission_sha,
        "allocation_amendment_path": str(amendment_path.resolve()),
        "allocation_amendment_sha256": file_sha256(amendment_path),
        "selection_receipt_path": str(selection_path.resolve()),
        "selection_receipt_sha256": file_sha256(selection_path),
        "continuation_template_path": str(continuation_panel_path.resolve()),
        "continuation_template_sha256": template_sha,
        "identity": identity,
        "selected_configuration": selected,
        "selected_teacher_cell_ids": teacher_cells,
        "target_formula": "T_new=0.8*T_old+0.2*0.8125*mean(clip(each_teacher,-2,2))",
        "target_cache": str(target_cache),
        "old_cache": str(old_cache),
        "target_writer_argv": target_argv,
        "target_writer_environment": {
            "CUDA_VISIBLE_DEVICES": writer_gpu,
            "NVIDIA_TF32_OVERRIDE": "0",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
        },
        "students": students,
        "required_external_gates": [
            "run target writer and independently admit its cache",
            "profile both exact student commands and bind actual receipts",
            "obtain independent profile review bound to the new phase panel/admission",
            "root confirms remaining deadline, disk and fit authority before launching either fit",
        ],
        "source_binding_scope": {
            "objective_teacher_source_binding_sha256": identity["source_binding_sha256"],
            "student_recipe_template_source_binding_sha256": template["identity"][
                "source_binding_sha256"
            ],
            "note": "These source identities remain separately bound; panel ancestry is not merged.",
        },
        "fit_count_reservation": {"campaign_total_max": 6, "teacher_fits": 4, "students_max": 2},
        "checkpoint_endpoint_epoch": 1,
        "new_fit_execution": False,
    }
    atomic_write_json(output, packet, replace=False)
    return output


def main() -> int:
    """Select eligible students or write their no-launch command packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "prepare"))
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--control-panel", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--amendment", type=Path)
    parser.add_argument("--continuation-panel", type=Path)
    args = parser.parse_args()
    if args.phase == "select":
        if args.control_panel is None:
            parser.error("select requires --control-panel")
        output = select(args.panel, args.authority, args.admission, args.control_panel)
    else:
        if any(
            value is None
            for value in (
                args.selection,
                args.amendment,
                args.continuation_panel,
                args.control_panel,
            )
        ):
            parser.error(
                "prepare requires --selection, --amendment, --continuation-panel and --control-panel"
            )
        output = prepare(
            args.panel,
            args.authority,
            args.admission,
            args.selection,
            args.amendment,
            args.continuation_panel,
            args.control_panel,
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
