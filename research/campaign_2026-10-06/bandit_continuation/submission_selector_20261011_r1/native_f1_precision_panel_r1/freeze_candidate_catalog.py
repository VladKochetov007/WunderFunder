"""Bind the exact fold1 checkpoints, fixed blends, and native Q8 packages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_PANEL_SHA256 = "4d68036453ca83545efc35be66936ab4f9f821502d2036b8d3b41f28fd194e4f"
EXPECTED_SPLIT_SHA256 = "31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae"
EXPECTED_TEACHER_CACHE_SHA256 = "a2f82d8dbc7bbbe2c7cf36e18e3b7be0b41f887aa14a742b43f616b060856779"


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def checkpoint_reference(run_ref: str) -> tuple[str, int]:
    """Split a run@ep reference into a run directory name and checkpoint epoch."""
    run_name, separator, epoch_text = run_ref.partition("@ep")
    if not separator or not epoch_text.isdigit():
        raise ValueError(f"expected a fixed run@ep reference: {run_ref}")
    return run_name, int(epoch_text)


def fixed_members_for_candidate(
    candidate: dict[str, object], candidates_by_id: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    """Resolve explicit members or inherit the source package's fixed ensemble."""
    member_plan = candidate.get("members")
    if member_plan is None:
        source_candidate = candidates_by_id.get(str(candidate.get("source_build_candidate")))
        if source_candidate is None or not isinstance(source_candidate.get("members"), list):
            raise ValueError(
                f"candidate has no fixed members or resolvable source candidate: "
                f"{candidate.get('candidate_id')}"
            )
        member_plan = source_candidate["members"]
    if not isinstance(member_plan, list):
        raise ValueError(f"candidate fixed members must be a list: {candidate.get('candidate_id')}")
    fixed: list[dict[str, object]] = []
    for member in member_plan:
        if not isinstance(member, dict) or not isinstance(member.get("run"), str):
            raise ValueError(
                f"candidate has a malformed fixed member: {candidate.get('candidate_id')}"
            )
        fixed.append({"run_reference": member["run"], "coefficient": member["coefficient"]})
    return fixed


def freeze_sources(
    project_root: Path,
    panel_path: Path,
    split_path: Path,
    teacher_cache_metadata_path: Path,
    plan: dict[str, object],
) -> dict[str, object]:
    """Verify fold1 membership exclusion and capture exact source ancestry pins."""
    if sha256_file(panel_path) != EXPECTED_PANEL_SHA256:
        raise ValueError("root-owned panel hash mismatch")
    if sha256_file(split_path) != EXPECTED_SPLIT_SHA256:
        raise ValueError("fold1 split hash mismatch")
    if sha256_file(teacher_cache_metadata_path) != EXPECTED_TEACHER_CACHE_SHA256:
        raise ValueError("strong-T3 teacher-cache metadata hash mismatch")
    panel = load_json(panel_path)
    split = load_json(split_path)
    selected_groups = set(panel["groups"])
    evaluation_groups = set(split["evaluation"]["valid"])
    if len(selected_groups) != 41 or len(evaluation_groups) != 457:
        raise ValueError("frozen panel/split group counts changed")
    if not selected_groups <= evaluation_groups:
        raise ValueError("selected panel contains a group outside fold1 evaluation membership")
    mapping = split["sequences"]["valid"]
    if {mapping[group]["component"] for group in selected_groups} != set(panel["components"]):
        raise ValueError("selected group/component mapping differs from panel receipt")

    run_refs = plan["source_runs"]
    if not isinstance(run_refs, dict):
        raise ValueError("candidate plan has no source-run map")
    source_records: dict[str, dict[str, object]] = {}
    for source_name, run_ref in run_refs.items():
        run_name, epoch = checkpoint_reference(run_ref)
        run_dir = project_root / "artifacts" / "runs" / run_name
        checkpoint_path = run_dir / f"model_ep{epoch}.pt"
        provenance_path = run_dir / "provenance.json"
        history_path = run_dir / "history.json"
        for path in (checkpoint_path, provenance_path, history_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        provenance = load_json(provenance_path)
        args = provenance.get("args")
        if not isinstance(args, dict):
            raise ValueError(f"{run_name}: provenance has no args")
        if provenance.get("split_manifest_sha256") != EXPECTED_SPLIT_SHA256:
            raise ValueError(f"{run_name}: fit used a different fold1 split")
        expected_evaluation = set(provenance.get("evaluation_row_groups", []))
        if expected_evaluation != evaluation_groups:
            raise ValueError(f"{run_name}: recorded evaluation groups differ from fold1 split")
        fit_row_groups = provenance.get("fit_row_groups")
        if not isinstance(fit_row_groups, dict) or "valid" not in fit_row_groups:
            raise ValueError(f"{run_name}: source-qualified valid fit membership is missing")
        fitted_valid_groups = set(fit_row_groups["valid"])
        if selected_groups & fitted_valid_groups:
            raise ValueError(f"{run_name}: selected panel overlaps fitted valid groups")
        checkpoint_digest_map = provenance.get("checkpoint_sha256")
        if not isinstance(checkpoint_digest_map, dict):
            raise ValueError(f"{run_name}: provenance checkpoint digests are missing")
        checkpoint_key = checkpoint_path.name
        actual_checkpoint_sha256 = sha256_file(checkpoint_path)
        if checkpoint_digest_map.get(checkpoint_key) != actual_checkpoint_sha256:
            raise ValueError(f"{run_name}: checkpoint digest differs from provenance")
        selected_epoch = provenance.get("selected_epoch")
        if selected_epoch is not None and selected_epoch != epoch:
            raise ValueError(f"{run_name}: provenance selects epoch {selected_epoch}, not {epoch}")

        distillation_cache = provenance.get("distillation_cache")
        if run_name == "vt_f1_dsq_dm_t3_a9_e6":
            if not isinstance(distillation_cache, dict):
                raise ValueError("H256 F1 prototype has no teacher-cache ancestry")
            if distillation_cache.get("metadata_sha256") != EXPECTED_TEACHER_CACHE_SHA256:
                raise ValueError("H256 F1 prototype points to a different teacher-cache receipt")
        record: dict[str, object] = {
            "run_reference": run_ref,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": actual_checkpoint_sha256,
            "provenance_path": str(provenance_path.resolve()),
            "provenance_sha256": sha256_file(provenance_path),
            "history_path": str(history_path.resolve()),
            "history_sha256": sha256_file(history_path),
            "seed": provenance.get("seed"),
            "selected_epoch": epoch,
            "model": {
                key: args.get(key)
                for key in ("arch", "hidden", "layers", "optim", "muon_scope", "sched", "loss")
            },
            "split_manifest_sha256": provenance["split_manifest_sha256"],
            "evaluation_valid_row_groups": len(expected_evaluation),
            "fitted_valid_row_groups": len(fitted_valid_groups),
            "selected_groups_fitted": 0,
            "propensity_checkpoint_sha256": provenance.get("propensity_checkpoint_sha256"),
            "distillation_cache": distillation_cache,
            "initial_checkpoint_sha256": provenance.get("initial_checkpoint_sha256"),
        }
        source_records[source_name] = record

    if source_records["teacher_s3"]["seed"] != 3 or source_records["teacher_s4"]["seed"] != 4:
        raise ValueError("H384 teacher seed identities changed")
    if source_records["teacher_s5"]["seed"] != 5:
        raise ValueError("H384 seed5 teacher identity changed")
    if source_records["h256_f1_prototype"]["seed"] != 3:
        raise ValueError("H256 F1 prototype is not the fixed seed3 run")

    student_run = source_records["h256_f1_prototype"]["run_reference"]
    student_run_name, student_epoch = checkpoint_reference(student_run)
    student_provenance = load_json(
        project_root / "artifacts" / "runs" / student_run_name / "provenance.json"
    )
    student_args = student_provenance["args"]
    if student_args.get("distill_alpha") != 0.9 or student_args.get("since_weight") not in (
        0,
        None,
    ):
        raise ValueError("H256 F1 prototype is not the declared alpha=.9/since=0 recipe")
    if student_args.get("checkpoint_epoch") != student_epoch:
        raise ValueError("H256 F1 prototype checkpoint epoch differs from its recipe")

    f1_parent_ref = plan["source_runs"].get("h256_parent")
    if not isinstance(f1_parent_ref, str):
        f1_parent_ref = "c20261003_combo_s3_f1@ep3"
    f1_parent_name, f1_parent_epoch = checkpoint_reference(f1_parent_ref)
    f1_parent_path = (
        project_root / "artifacts" / "runs" / f1_parent_name / f"model_ep{f1_parent_epoch}.pt"
    )
    f1_parent_sha = sha256_file(f1_parent_path)
    if student_provenance.get("initial_checkpoint_sha256") != f1_parent_sha:
        raise ValueError("H256 F1 prototype initializer differs from its declared F1 parent")

    comparison_context: dict[str, dict[str, object]] = {}
    context_runs = plan.get("comparison_context_runs")
    if not isinstance(context_runs, dict):
        raise ValueError("candidate plan has no H390 comparison-context run map")
    for context_name, run_ref in context_runs.items():
        run_name, epoch = checkpoint_reference(run_ref)
        run_dir = project_root / "artifacts" / "runs" / run_name
        checkpoint_path = run_dir / f"model_ep{epoch}.pt"
        provenance_path = run_dir / "provenance.json"
        history_path = run_dir / "history.json"
        for path in (checkpoint_path, provenance_path, history_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        provenance = load_json(provenance_path)
        checkpoint_map = provenance.get("checkpoint_sha256")
        if not isinstance(checkpoint_map, dict):
            raise ValueError(f"{run_name}: comparison-context checkpoint map is missing")
        checkpoint_sha = sha256_file(checkpoint_path)
        if checkpoint_map.get(checkpoint_path.name) != checkpoint_sha:
            raise ValueError(f"{run_name}: comparison-context checkpoint hash mismatch")
        comparison_context[context_name] = {
            "run_reference": run_ref,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "provenance_path": str(provenance_path.resolve()),
            "provenance_sha256": sha256_file(provenance_path),
            "history_path": str(history_path.resolve()),
            "history_sha256": sha256_file(history_path),
            "seed": provenance.get("seed"),
            "selected_epoch": epoch,
            "fit_scope": "full-data comparison context; its valid rows cannot support held-out claims",
        }
    if (
        student_provenance.get("initial_checkpoint_sha256")
        == comparison_context["h390_parent"]["checkpoint_sha256"]
    ):
        raise ValueError("F1 prototype and H390 all-data student share an initializer unexpectedly")

    return {
        "schema": "wnn.native_f1_precision_source_bindings.r1",
        "selection_panel": {
            "path": str(panel_path.resolve()),
            "sha256": sha256_file(panel_path),
            "selection_seed": panel["selection_seed"],
            "row_groups": panel["groups"],
            "components": panel["components"],
            "exposure": panel["evaluation_exposure"],
        },
        "fold1_split": {
            "path": str(split_path.resolve()),
            "sha256": sha256_file(split_path),
            "evaluation_valid_row_groups": len(evaluation_groups),
            "selected_panel_row_groups": len(selected_groups),
            "selected_panel_components": len(panel["components"]),
        },
        "teacher_cache_metadata": {
            "path": str(teacher_cache_metadata_path.resolve()),
            "sha256": sha256_file(teacher_cache_metadata_path),
            "semantics": (
                "Student homogeneous strong-T3 selection uses the float32 mean of teacher seeds "
                "3/4/5, then the declared .8125 family scale; metadata also records six source "
                "coefficients and must not be interpreted as the selected homogeneous target."
            ),
        },
        "sources": source_records,
        "comparison_context": comparison_context,
        "student_parent_qualification": {
            "f1_prototype_parent": {
                "run_reference": f1_parent_ref,
                "checkpoint_sha256": f1_parent_sha,
            },
            "f1_parent_run": f1_parent_ref,
            "h390_all_data_final": "vt_all_h390_ds_homogeneous_s3_r3@ep5",
            "h390_parent_run": "vt_all_fin_mu256p0e4_s3@ep3",
            "h390_parent_checkpoint_sha256": comparison_context["h390_parent"]["checkpoint_sha256"],
            "interpretation": (
                "The F1 student is an eligible cleanfold prototype on its own declared parent; "
                "it is not a matched heldout counterpart of the H390 all-data final."
            ),
        },
        "exclusion_result": {
            "all_candidate_source_fit_memberships_exclude_panel": True,
            "selected_rows_are_historically_exposed": True,
            "fresh_confirmation_claim": False,
        },
    }


def freeze_catalog(
    project_root: Path,
    packet_dir: Path,
    plan_path: Path,
    output_path: Path,
    panel_path: Path,
    split_path: Path,
    teacher_cache_metadata_path: Path,
) -> dict[str, object]:
    """Validate every frozen candidate artifact and emit the run catalog."""
    if output_path.exists():
        raise FileExistsError(output_path)
    plan = load_json(plan_path)
    sources = freeze_sources(
        project_root, panel_path, split_path, teacher_cache_metadata_path, plan
    )
    candidate_records: list[dict[str, object]] = []
    candidates_by_id = {item["candidate_id"]: item for item in plan["candidates"]}
    if len(candidates_by_id) != len(plan["candidates"]):
        raise ValueError("candidate IDs must be unique")
    for candidate in plan["candidates"]:
        package_path = (packet_dir / candidate["package_path"]).resolve(strict=True)
        receipt_path = (packet_dir / candidate["package_receipt_path"]).resolve(strict=True)
        receipt = load_json(receipt_path)
        if receipt.get("candidate") != candidate["candidate_id"]:
            raise ValueError(f"package receipt candidate mismatch: {candidate['candidate_id']}")
        if receipt.get("precision") != "q8vnni":
            raise ValueError(f"package is not Q8VNNI: {candidate['candidate_id']}")
        artifact = receipt.get("artifact")
        if not isinstance(artifact, dict):
            raise ValueError(f"package receipt has no artifact: {candidate['candidate_id']}")
        package_sha = sha256_file(package_path)
        if artifact.get("sha256") != package_sha:
            raise ValueError(f"package artifact hash mismatch: {candidate['candidate_id']}")
        if package_path.stat().st_size > candidate["max_zip_bytes"]:
            raise ValueError(
                f"candidate ZIP exceeds its declared size ceiling: {candidate['candidate_id']}"
            )
        fixed_members = receipt.get("fixed_members")
        if fixed_members is None:
            fixed_members = receipt.get("fixed_members")
        expected_members = fixed_members_for_candidate(candidate, candidates_by_id)
        if fixed_members != expected_members:
            raise ValueError(f"package fixed members differ from plan: {candidate['candidate_id']}")
        candidate_records.append(
            {
                "candidate_id": candidate["candidate_id"],
                "role": candidate["role"],
                "research_only": candidate.get("research_only", False),
                "research_only_reason": candidate.get("research_only_reason"),
                "prototype_qualification": candidate.get("prototype_qualification"),
                "precision": "q8vnni",
                "storage": candidate["storage"],
                "fixed_members": expected_members,
                "package_path": str(package_path),
                "package_sha256": package_sha,
                "package_bytes": package_path.stat().st_size,
                "package_receipt_path": str(receipt_path),
                "package_receipt_sha256": sha256_file(receipt_path),
                "predictions_path": str((packet_dir / candidate["predictions_path"]).resolve()),
                "inference_receipt_path": str(
                    (packet_dir / candidate["inference_receipt_path"]).resolve()
                ),
                "artifact_member_count": len(artifact.get("members", {})),
                "source_build_candidate": candidate["source_build_candidate"],
            }
        )
    catalog: dict[str, object] = {
        "schema": "wnn.native_f1_precision_candidate_catalog.r1",
        "producer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "source_bindings": sources,
        "candidates": candidate_records,
        "reference_candidate_id": "f1_teacher_s4_s5_q8_fp32",
        "inference_status": "PENDING_ROOT_ADMISSION",
        "grade_status": "PENDING_NATIVE_PREDICTIONS",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    return catalog


def main() -> None:
    """Write a source-qualified catalog for exact Q8 candidate packages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--teacher-cache-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = freeze_catalog(
        args.project_root,
        args.packet_dir,
        args.plan,
        args.output,
        args.panel,
        args.split,
        args.teacher_cache_metadata,
    )
    print(
        json.dumps(
            {"candidates": catalog["candidates"], "inference_status": catalog["inference_status"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
