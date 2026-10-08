"""Inventory small metadata only; no tensor loads or array/parquet reads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
OUT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    split_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold1.json"
    split = json.loads(split_path.read_text())
    expected_sha = sha(split_path)
    teachers = []
    for family, prefix, epoch, coefficient in (
        ("strong", "c20261003_combo", 3, 0.15),
        ("fullFT", "c20260929_ft_since", 0, 0.1),
        ("raw", "c20260929_since", 3, 0.125),
    ):
        for seed in (3, 4):
            run = f"{prefix}_s{seed}_f1"
            provenance_path = ROOT / "artifacts/runs" / run / "provenance.json"
            provenance = json.loads(provenance_path.read_text())
            checkpoint = provenance_path.parent / f"model_ep{epoch}.pt"
            fitting = provenance.get("fit_row_groups", {})
            evaluation_overlap = sorted(
                set(fitting.get("valid", [])) & set(split["evaluation"]["valid"])
            )
            exports = {}
            for source in ("train", "valid"):
                array = ROOT / f"cache/pred_{run}_{source}.npy"
                sidecar = array.with_suffix(".json")
                metadata = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
                exports[source] = {
                    "array_exists": array.is_file(),
                    "bytes": array.stat().st_size if array.is_file() else None,
                    "sidecar_sha256": sha(sidecar) if sidecar.is_file() else None,
                    "metadata_checkpoint_sha256": metadata.get("checkpoint_sha256"),
                    "metadata_epoch": metadata.get("epoch"),
                    "qualification": "metadata inventory only; actual array/producer checks required before cache construction",
                }
            teachers.append(
                {
                    "family": family,
                    "run": run,
                    "epoch": epoch,
                    "raw_sum_coefficient": coefficient,
                    "checkpoint_exists": checkpoint.is_file(),
                    "provenance_sha256": sha(provenance_path),
                    "split_binding_matches": provenance.get("split_manifest_sha256")
                    == expected_sha,
                    "evaluated_valid_overlap": evaluation_overlap,
                    "initializer": provenance["args"].get("init"),
                    "exports": exports,
                }
            )
    missing = []
    for seed in (3, 4):
        origin = ROOT / f"artifacts/runs/bd26_w384raw_s{seed}_f0/provenance.json"
        args = json.loads(origin.read_text())["args"]
        args.update(
            name=f"bd26_w384raw_s{seed}_f1",
            valid_fold=1,
            split_manifest=str(split_path),
            split_manifest_sha256=expected_sha,
        )
        missing.append(
            {
                "run": args["name"],
                "checkpoint_exists": (
                    ROOT / "artifacts/runs" / args["name"] / "model_ep3.pt"
                ).is_file(),
                "exact_counterpart_provenance_sha256": sha(origin),
                "frozen_draft_args": args,
                "family": "baseline",
                "raw_sum_coefficient": 0.03125,
                "fixed_epoch": 3,
                "admission": "pending root; two fresh GPU fits required, never substitute fold0 weights",
            }
        )
    result = {
        "status": "dependency_preparation_only",
        "split_sha256": expected_sha,
        "split_counts": split["counts"],
        "teachers": teachers,
        "missing_baseline_fits": missing,
        "propensity_run": "c20260929_prop_f1",
        "propensity_provenance_sha256": sha(
            ROOT / "artifacts/runs/c20260929_prop_f1/provenance.json"
        ),
        "ordered_delivery": [
            "Audit all teacher and propensity initializer ancestry against fold1 evaluation membership",
            "Prepare/admit two independent baseline384 seed3/4 fits with the exact draft counterpart args",
            "Export all eight teachers fulltrain/fullvalid with actual row-group/checkpoint/precision receipts; verify existing exports before reuse",
            "Build new fold1 four-family cache with raw sum .8125 and fixed coefficients; never relabel fold0",
            "Independent grader freezes official clip2 fixed-mask panel457 membership and comparison; fold1 historical exposure recorded",
            "Root admits matched seed4 student/backbone, seed3 GCorr finish and same half-blend endpoint after actual profile",
        ],
        "ownership": {
            "actor_delivery": "Sol continuation",
            "successor_builder": "Luna runner",
            "admission": "root",
            "metric_and_grade": "independent Luna grader",
        },
        "exposure": "fold1 historically inspected; robustness confirmation, not a fresh holdout",
        "science": "No public score extrapolation; no new fit/cache admitted by this inventory",
    }
    (OUT / "FOLD1_DEPENDENCY_INVENTORY_R1.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "evaluation_count": split["counts"]["evaluation_valid"],
                "existing_teacher_count": len(teachers),
                "missing_baseline_count": len(missing),
                "teacher_split_binding_all_match": all(
                    row["split_binding_matches"] for row in teachers
                ),
                "teacher_evaluation_overlap_count": sum(
                    len(row["evaluated_valid_overlap"]) for row in teachers
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
