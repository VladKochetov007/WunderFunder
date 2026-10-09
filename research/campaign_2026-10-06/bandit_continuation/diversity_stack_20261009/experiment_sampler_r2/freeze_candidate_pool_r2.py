#!/usr/bin/env python3
"""Bind the R1 recipe ideas to the immutable D02 R2 source closure."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(HERE))
import recipe_sampler as sampler  # noqa: E402


def canonical_json(value: object) -> str:
    """Return stable JSON for source and candidate receipts."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a nonempty JSONL candidate pool."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"candidate pool is empty: {path}")
    return rows


def make_source_record(spec: dict[str, Any]) -> dict[str, Any]:
    """Hash and describe the exact D02 R2 source closure used by every recipe."""
    code_root = (REPO_ROOT / spec["d02_source_root"]).resolve()
    expected_paths = sorted(spec["source_code_paths"])
    actual_paths = sorted(
        path.relative_to(REPO_ROOT).as_posix() for path in code_root.rglob("*") if path.is_file()
    )
    if actual_paths != expected_paths:
        raise ValueError("D02 source tree no longer matches the frozen 63-file closure")
    hashes = {relative: sampler.file_sha256(REPO_ROOT / relative) for relative in expected_paths}
    choices = sampler.load_cli_choices(
        [REPO_ROOT / relative for relative in spec["trainer_cli_choices"]]
    )
    choice_snapshot = {key: sorted(values) for key, values in sorted(choices.items())}
    return {
        "backend": "D02 R2 PyTorch trainer source closure",
        "source_root": spec["d02_source_root"],
        "source_commit": spec["d02_source_commit"],
        "code_payload_sha256": spec["d02_code_payload_sha256"],
        "files": expected_paths,
        "sha256_by_path": hashes,
        "binding_hash": sampler.sha256_json(hashes),
        "trainer_cli_choices": spec["trainer_cli_choices"],
        "trainer_cli_choice_snapshot": choice_snapshot,
        "optimizer_policy": {
            "implementation": "D02 MuonAdamW in src/wnn/optim.py",
            "orthogonalization_dtype_policy": "fp32",
            "selected_orthogonalization_dtype": "float32",
            "model_autocast_is_independent": True,
        },
        "schedule_policy": {
            "implementation": "src/wnn/training_dynamics.py:build_scheduler",
            "onecycle_warmup_frac": 0.05,
            "onecycle_div_factor": 25.0,
            "onecycle_final_div_factor": 100.0,
            "wsd_warmup_frac": 0.05,
            "wsd_decay_share": 0.2,
            "wsd_final_lr_fraction": 0.01,
            "sgdr_warmup_frac": 0.05,
            "sgdr_restarts": 1,
            "sgdr_final_lr_fraction": 0.01,
        },
    }


def align_schedule(recipe: dict[str, Any]) -> None:
    """Record the exact scheduler arguments implemented by D02 R2."""
    schedule = recipe["schedule"]
    params = schedule["parameters"]
    optimizer = recipe["optimizer"]
    if schedule["name"] == "onecycle":
        params.pop("pct_start", None)
        params["warmup_frac"] = 0.05
        params["final_div_factor"] = 100.0
        params["cycle_momentum"] = optimizer["name"] != "sgd"
        params["beta1_trajectory"] = (
            "OneCycleLR cycles beta1; record the observed per-update trace"
            if params["cycle_momentum"]
            else "fixed SGD momentum; OneCycleLR cycle_momentum=false"
        )
        schedule["implementation"] = (
            "wnn.training_dynamics.build_scheduler: OneCycleLR with D02 R2 warmup, "
            "divisor and final-divisor settings"
        )
    elif schedule["name"] == "wsd":
        params.update({"warmup_frac": 0.05, "decay_share": 0.2, "final_lr_fraction": 0.01})
        schedule["implementation"] = "wnn.training_dynamics.build_scheduler with wsd_lambda"
        params["beta1_trajectory"] = "constant trainer beta1=0.9 under LambdaLR"
    elif schedule["name"] == "sgdr":
        params.update({"warmup_frac": 0.05, "restarts": 1, "final_lr_fraction": 0.01})
        schedule["implementation"] = "wnn.training_dynamics.build_scheduler with sgdr_lambda"
        params["beta1_trajectory"] = "constant trainer beta1=0.9 under LambdaLR"
    if optimizer["name"] == "sgd":
        optimizer["beta1_policy"] = "fixed SGD momentum from recipe"
    elif schedule["name"] == "onecycle":
        optimizer["beta1_policy"] = "OneCycleLR beta1 cycle; trace every attempted update"
    else:
        optimizer["beta1_policy"] = "constant trainer beta1=0.9"


def freeze_pool() -> dict[str, Any]:
    """Create an identified pool successor and its source-closure receipt."""
    spec_path = HERE / "sampler_spec_r2.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    parent_pool = REPO_ROOT / spec["parent_candidate_pool_path"]
    parent_sha = hashlib.sha256(parent_pool.read_bytes()).hexdigest()
    if parent_sha != spec["parent_candidate_pool_sha256"]:
        raise ValueError("parent sampler R1 pool changed after R2 was identified")
    source = make_source_record(spec)
    candidates = read_jsonl(parent_pool)
    output_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        recipe = copy.deepcopy(candidate["recipe"])
        recipe["source"] = copy.deepcopy(source)
        recipe["implementation_dependency"] = {
            "state": "pending_recipe_aware_runner",
            "source_commit": spec["d02_source_commit"],
            "required_runner_entrypoint": f"{spec['d02_source_root']}/scripts/train_rnn.py",
            "blocker": (
                "No validated runner maps each full recipe to exact argv; the sealed D02 R2 GPU worker "
                "is a Muon-only WMSE/Huber pair worker."
            ),
            "required_checks": [
                "all four AdamW/Muon by WMSE/Huber cells use one matched source closure",
                "recipe loss, model, membership, update budget and schedule match emitted argv",
                "actual optimizer provenance records FP32 Muon orthogonalization where selected",
                "exact trainer and prediction parsers accept the bound argv",
            ],
        }
        recipe["runner_binding"] = {
            "entrypoint": f"{spec['d02_source_root']}/scripts/train_rnn.py",
            "launch_argv": None,
            "state": "pending_recipe_aware_runner",
            "note": "Code/source binding is not a runnable or admitted queue item.",
        }
        align_schedule(recipe)
        if recipe["optimizer"]["name"] == "muon":
            recipe["optimizer"]["muon_policy"] = {
                "scope": "rnn",
                "gatewise": "all",
                "orthogonalization_dtype_policy": "fp32",
                "selected_orthogonalization_dtype": "float32",
                "ns_steps": 5,
                "rms": 0.2,
                "lr_multiplier": 1.0,
            }
        recipe["identity_hashes"]["source_sha256"] = sampler.sha256_json(source)
        recipe["identity_hashes"]["data_sha256"] = sampler.sha256_json(
            {"data": recipe["data"], "sampling": recipe["sampling"]}
        )
        recipe["identity_hashes"]["ancestry_sha256"] = sampler.sha256_json(recipe["ancestry"])
        row = {
            "candidate_id": candidate["candidate_id"],
            "description": candidate.get("description", ""),
            "mechanism_tags": sorted(set(candidate.get("mechanism_tags", []))),
            "recipe": recipe,
        }
        row["recipe_hash"] = sampler.recipe_hash(recipe)
        output_rows.append(row)
    pool_path = HERE / spec["candidate_pool_path"]
    pool_bytes = ("".join(canonical_json(row) + "\n" for row in output_rows)).encode("utf-8")
    pool_path.write_bytes(pool_bytes)
    closure_receipt = {
        "schema": "wnn.diversity.d02-source-closure.v1",
        "state": "source_bound_runner_pending",
        "source": source,
        "parent_candidate_pool_path": spec["parent_candidate_pool_path"],
        "parent_candidate_pool_sha256": parent_sha,
        "candidate_pool_path": spec["candidate_pool_path"],
        "candidate_pool_sha256": hashlib.sha256(pool_bytes).hexdigest(),
        "candidate_count": len(output_rows),
        "fit_allowed": False,
        "admission_allowed": False,
    }
    receipt_path = HERE / "D02_SOURCE_CLOSURE_R1.json"
    receipt_path.write_text(json.dumps(closure_receipt, indent=2, sort_keys=True) + "\n")
    return closure_receipt


def main() -> int:
    """Freeze the source-bound pool without invoking a trainer or provider."""
    receipt = freeze_pool()
    print(
        json.dumps(
            {
                "candidate_count": receipt["candidate_count"],
                "candidate_pool_sha256": receipt["candidate_pool_sha256"],
                "source_binding_sha256": receipt["source"]["binding_hash"],
                "fit_allowed": receipt["fit_allowed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
