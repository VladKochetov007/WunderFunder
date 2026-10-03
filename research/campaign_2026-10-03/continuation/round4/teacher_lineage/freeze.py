"""Summarize the saved bounded metadata audit, preserving unknown execution."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

OWN = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = json.loads((OWN / "lineage_map.json").read_text())
    hashes = json.loads((OWN / "provenance_hashes.json").read_text())
    rows = []
    for fold, entry in data["folds"].items():
        teachers = [node for name, node in entry["nodes"].items() if "since_s" in name]
        prop = next(node for name, node in entry["nodes"].items() if "_prop_" in name)
        rows.append(
            {
                "fold": int(fold),
                "evaluation_valid_groups": len(entry["evaluation_valid_groups"]),
                "evaluation_components": len(entry["evaluation_components"]),
                "linked_train_excluded": len(
                    entry["linked_train_groups_in_evaluation_components"]
                ),
                "teacher_fit_counts": teachers[0]["fit_counts"],
                "teacher_intersection_count": sum(
                    len(items)
                    for node in teachers
                    for items in node["reported_fit_component_intersections"].values()
                ),
                "propensity_valid_fit": prop["fit_counts"]["valid"],
                "propensity_intersection_count": sum(
                    len(items)
                    for items in prop["reported_fit_component_intersections"].values()
                ),
            }
        )
    checkpoint_records = [
        value
        for name, value in hashes.items()
        if name.endswith(".pt") and value["exists"]
    ]
    checks = {
        "graphs_match_all_manifests": all(
            e["graph_mapping_exact"] for e in data["folds"].values()
        ),
        "index_identities_match": all(
            all(e["index_identity_checks"].values()) for e in data["folds"].values()
        ),
        "all_teacher_and_propensity_reported_pools_exclude": all(
            n["reported_excludes_evaluation_components"]
            for e in data["folds"].values()
            for name, n in e["nodes"].items()
            if "combo" not in name
        ),
        "all_teacher_and_propensity_fixed_no_eval": all(
            n["fixed_epoch_record_consistent"]
            and n["no_eval_during_fit"]
            and all(row.get("evaluation") == "withheld" for row in n["history"])
            for e in data["folds"].values()
            for name, n in e["nodes"].items()
            if "combo" not in name
        ),
        "all_sidecar_checkpoint_bytes_and_order_match": all(
            s["checkpoint_actual_bytes_match"]
            and s["row_order_matches_index"]
            and s["run_split_match"]
            and s["split_manifest_matches"]
            for e in data["folds"].values()
            for s in e["sidecars"].values()
        ),
        "available_student_bindings_match": all(
            edge["bindings_match_current_sidecars"]
            for e in data["folds"].values()
            for edge in e["edges"]
            if edge["kind"] == "cached distillation"
            and not edge["student_not_available"]
        ),
        "all_recorded_checkpoint_hashes_match": all(
            c["matches_recorded"]
            for e in data["folds"].values()
            for n in e["nodes"].values()
            for c in n.get("checkpoints", {}).values()
        ),
        "metadata_stable_within_each_read": all(
            n.get("metadata_stable", True)
            for e in data["folds"].values()
            for n in e["nodes"].values()
        ),
        "audit_executed_source_matches": hashlib.sha256(
            (OWN / "audit.py").read_bytes()
        ).hexdigest()
        == data["audit_source_sha256"],
    }
    summary = {
        "folds": rows,
        "checks": checks,
        "checkpoint_files_hashed": len(checkpoint_records),
        "checkpoint_bytes_hashed": sum(c["bytes"] for c in checkpoint_records),
        "cost": data["cost"],
        "executed_teacher_membership_proven": False,
        "unknowns": [
            "teacher/propensity per-batch executed membership and historical full source identity",
            "embedded checkpoint metadata not parsed",
            "prediction cache bytes not rehashed",
            "combo fold3 not available; fold2 selected checkpoint not yet recorded",
            "undocumented ancestors or cross-run architecture/hyperparameter selection",
        ],
    }
    (OWN / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    state = json.loads((OWN / "STATE.json").read_text())
    state.update(
        phase="frozen",
        pending=[],
        last_failure=None,
        checks=checks,
        awaiting="controller-provided frozen builder review",
        cost=data["cost"],
        no_contamination_inferred_from_unknowns=True,
    )
    (OWN / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    own_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(OWN.iterdir())
        if p.is_file() and p.name not in {"HASHES.json", "freeze.log"}
    }
    (OWN / "HASHES.json").write_text(json.dumps(own_hashes, indent=2) + "\n")
    LOGGER.info("SUMMARY %s", json.dumps(summary))
    if not all(checks.values()):
        raise RuntimeError(
            "Metadata consistency check failed; inspect retained outputs"
        )


if __name__ == "__main__":
    main()
