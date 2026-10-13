#!/usr/bin/env python3
"""Small stdlib-only check of actual parent/teacher/propensity lineage receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
RUNS = MAIN / "artifacts/runs"
SPLIT_PATH = MAIN / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
EXPECTED_SPLIT = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_run(name: str) -> tuple[dict, dict]:
    payload = json.loads((RUNS / name / "provenance.json").read_text())
    return payload, payload["args"]


def identity_ok(run: dict, source: str, expected: set[int]) -> bool:
    recorded = run.get("fit_row_groups", {}).get(source)
    return recorded is not None and set(map(int, recorded)) == expected


def main() -> None:
    split_bytes = SPLIT_PATH.read_bytes()
    split_sha = hashlib.sha256(split_bytes).hexdigest()
    split = json.loads(split_bytes)
    fit_train = set(map(int, split["fit"]["train"]))
    fit_valid = set(map(int, split["fit"]["valid"]))
    eval_valid = set(map(int, split["evaluation"]["valid"]))

    parent, parent_args = read_run("c20261003_combo_s3_f0")
    teacher_records = parent_args.get("distill_provenance")
    if not isinstance(teacher_records, list):
        raise RuntimeError("actual parent args omit distill_provenance")
    teachers = {}
    for name in sorted({row["run"] for row in teacher_records}):
        run, args = read_run(name)
        artifact = RUNS / name / "model_ep3.pt"
        actual_sha = digest(artifact)
        recorded_sha = run.get("checkpoint_sha256", {}).get("model_ep3.pt")
        rows = [entry for entry in teacher_records if entry["run"] == name]
        bound_sha = {entry["checkpoint_sha256"] for entry in rows}
        teachers[name] = {
            "args": {key: args.get(key) for key in (
                "name", "seed", "init", "train_source", "valid_fold", "no_eval_during_fit",
                "checkpoint_policy", "checkpoint_epoch", "split_manifest_sha256",
            )},
            "actual_model_ep3_sha256": actual_sha,
            "provenance_model_ep3_sha256": recorded_sha,
            "parent_distill_bound_checkpoint_sha256": sorted(bound_sha),
            "parent_distill_record_rows": rows,
            "fit_train_exact_manifest": identity_ok(run, "train", fit_train),
            "fit_valid_exact_manifest": identity_ok(run, "valid", fit_valid),
            "evaluation_groups_exact_manifest": set(map(int, run.get("evaluation_row_groups", []))) == eval_valid,
        }
        if actual_sha != recorded_sha or bound_sha != {actual_sha}:
            raise RuntimeError(f"teacher checkpoint identity mismatch: {name}")
        if (args.get("init") or "") != "" or args.get("split_manifest_sha256") != split_sha:
            raise RuntimeError(f"teacher has unexpected initializer or split: {name}")
        if not (teachers[name]["fit_train_exact_manifest"] and teachers[name]["fit_valid_exact_manifest"] and teachers[name]["evaluation_groups_exact_manifest"]):
            raise RuntimeError(f"teacher membership mismatch: {name}")

    propensity, propensity_args = read_run("c20260929_prop_f0")
    propensity_file = RUNS / "c20260929_prop_f0" / "model.pt"
    propensity_sha = digest(propensity_file)
    prop_fit_valid = set(map(int, propensity.get("fit_row_groups", {}).get("valid", [])))
    prop_eval = set(map(int, propensity.get("evaluation_row_groups", [])))
    prop_validity = {
        "args": {key: propensity_args.get(key) for key in (
            "name", "seed", "init", "valid_fold", "no_eval_during_fit", "checkpoint_policy",
            "checkpoint_epoch", "split_manifest_sha256",
        )},
        "actual_model_sha256": propensity_sha,
        "provenance_model_sha256": propensity.get("checkpoint_sha256", {}).get("model.pt"),
        "parent_bound_sha256": parent.get("propensity_checkpoint_sha256"),
        "fit_valid_count": len(prop_fit_valid),
        "fit_valid_exact_manifest": prop_fit_valid == fit_valid,
        "evaluation_count": len(prop_eval),
        "evaluation_exact_manifest": prop_eval == eval_valid,
        "no_fit_eval_overlap": not bool(prop_fit_valid & prop_eval),
    }
    if not (propensity_sha == prop_validity["provenance_model_sha256"] == prop_validity["parent_bound_sha256"]):
        raise RuntimeError("propensity checkpoint hash mismatch")
    if (propensity_args.get("init") not in (None, "")
            or propensity_args.get("split_manifest_sha256") != split_sha
            or prop_fit_valid != fit_valid or prop_eval != eval_valid):
        raise RuntimeError("propensity lineage/membership mismatch")

    parent_model = RUNS / "c20261003_combo_s3_f0/model_ep3.pt"
    parent_model_sha = digest(parent_model)
    output = {
        "schema": "wnn.s09.independent-parent-lineage-review.r1",
        "split_manifest_sha256": split_sha,
        "parent": {
            "actual_model_ep3_sha256": parent_model_sha,
            "provenance_model_ep3_sha256": parent.get("checkpoint_sha256", {}).get("model_ep3.pt"),
            "args": {key: parent_args.get(key) for key in (
                "name", "seed", "init", "train_source", "valid_fold", "no_eval_during_fit",
                "checkpoint_policy", "checkpoint_epoch", "split_manifest_sha256", "distill",
                "propensity",
            )},
            "fit_train_count": len(parent["fit_row_groups"]["train"]),
            "fit_valid_count": len(parent["fit_row_groups"]["valid"]),
            "fit_train_exact_manifest": identity_ok(parent, "train", fit_train),
            "fit_valid_exact_manifest": identity_ok(parent, "valid", fit_valid),
            "evaluation_groups_exact_manifest": set(map(int, parent.get("evaluation_row_groups", []))) == eval_valid,
            "initial_checkpoint_sha256": parent.get("initial_checkpoint_sha256"),
            "teachers": teachers,
            "propensity": prop_validity,
        },
        "status": "pass",
        "limitations": [
            "Teacher prediction-array hashes are recorded by the actual parent args; their large payloads were not reread.",
            "Only JSON provenance and small model checkpoint files were inspected; no training data or cache arrays were opened.",
        ],
    }
    if parent_model_sha != output["parent"]["provenance_model_ep3_sha256"]:
        raise RuntimeError("parent checkpoint provenance hash mismatch")
    if not (output["parent"]["fit_train_exact_manifest"] and output["parent"]["fit_valid_exact_manifest"] and output["parent"]["evaluation_groups_exact_manifest"]):
        raise RuntimeError("parent membership mismatch")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
