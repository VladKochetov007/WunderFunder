"""Create unsealed, hash-bound E464 head-fit draft specs without loading Torch."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
CAMPAIGN_ROOT = ROOT / "research/campaign_2026-10-04"
STAGING = CAMPAIGN_ROOT / "staging/e464"
PARENTS = {
    "h093_learned": "c20261004_h093_learned_s3_final",
    "h093_hash": "c20261004_h093_hash_s3_final",
    "h093_universal": "c20261004_h093_universal_s3_final",
    "h093_gru_budget": "c20261004_h093_gru_budget_s3_final",
    "h094_conditional": "c20261004_h094_conditional_s3_final",
    "h094_unconditional": "c20261004_h094_unconditional_s3_final",
    "h094_diagonal": "c20261004_h094_diagonal_s3_final",
    "h094_gru_budget": "c20261004_h094_gru_budget_s3_final",
}
EXPECTED_ARCH = {
    "h093_learned": "specialist", "h093_hash": "specialist",
    "h093_universal": "specialist", "h093_gru_budget": "gru",
    "h094_conditional": "prefix-adapter", "h094_unconditional": "prefix-adapter",
    "h094_diagonal": "prefix-adapter", "h094_gru_budget": "gru",
}
FROZEN = {
    "specialist": ("inp", "gru", "experts", "prefix_proj", "router"),
    "prefix-adapter": ("inp", "gru", "adapter_", "prefix_prior"),
    "gru": ("inp", "gru"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def descriptor(relative: str) -> dict[str, object]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def incumbent_trainer_argv(parent: str, child: str, freeze: tuple[str, ...]) -> list[str]:
    """Record the helper's incumbent recipe after its E464-only freeze/eval overrides."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import continuation_recipes

    source = continuation_recipes.head_args(child, f"{parent}@ep3", None)
    result = []
    index = 0
    while index < len(source):
        if source[index] == "--eval-source":
            index += 2
            continue
        if source[index] == "--freeze":
            result.extend(("--freeze", ",".join(freeze)))
            index += 2
            continue
        result.append(source[index])
        index += 1
    if "--freeze" not in result:
        result.extend(("--freeze", ",".join(freeze)))
    result.append("--no-eval-during-fit")
    return result


def main() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import continuation_recipes

    source_paths = sorted(set(continuation_recipes.source_paths(ROOT)) | {
        "research/campaign_2026-10-04/staging/e464/scripts/run_head_fit.py",
    })
    code = [descriptor(path) for path in source_paths]
    inputs_from_helper = continuation_recipes.input_paths(None, parent="placeholder@ep3")
    shared_inputs = [path for path in inputs_from_helper
                     if not path.startswith("artifacts/runs/placeholder/")]
    bundle = {
        "status": "draft_unsealed",
        "campaign": "E464 complete-architecture head-only continuation",
        "preregistration": "research/campaign_2026-10-04/E464_COMPLETE_ARCHITECTURE_PREREG.md",
        "fit_recipe": {
            "epochs": 1, "fixed_checkpoint_epoch": 0, "seed": 3,
            "optimizer": "AdamW", "learning_rate": 0.0005, "weight_decay": 0.01,
            "batch": 16, "chunk_rows": 2000, "loss": "gcorr",
            "mask_weighted": True, "evaluation_during_fit": "withheld",
            "train_source": "valid development groups only", "valid_fit_sequences": 1505,
            "excluded_valid_sequences": 368,
            "parent_auxiliary_targets": "inherit exactly from each parent checkpoint; preserve 23-output head",
        },
        "resource_plan": {
            "backend": "local single GPU, sequential; root admission required",
            "cpus": 8, "ram_bytes": 24 * 1024**3,
            "gpu_allocator_bytes": 4_200_000_000, "gpu_stop_bytes": 4_800_000_000,
            "worker_timeout_seconds": 1200,
            "estimated_total_gpu_hours": 1.0,
            "reserved_max_gpu_hours": 3.0,
            "admission_gate": "wait for E425/E426 and existing one-GPU queue terminal; reconcile cumulative 80-device-GPU-hour campaign budget before root admits",
        },
        "source_closure": code,
        "shared_input_paths": shared_inputs,
        "arms": [],
    }
    for arm, parent in PARENTS.items():
        parent_dir = ROOT / "artifacts/runs" / parent
        checkpoint_path = parent_dir / "model_ep3.pt"
        provenance_path = parent_dir / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        parent_code_spec_path = f"research/campaign_2026-10-04/specs/e424_{arm}_v2.sealed.json"
        parent_code_spec = json.loads((ROOT / parent_code_spec_path).read_text())
        parent_code_spec_sha = sha256_file(ROOT / parent_code_spec_path)
        parent_code_files = {item["path"]: item["sha256"] for item in parent_code_spec["code"]}
        source_mismatches = [path for path, expected_sha in parent_code_files.items()
                             if sha256_file(ROOT / path) != expected_sha]
        if source_mismatches:
            raise ValueError(f"{arm} current source differs from its sealed E424 parent closure: {source_mismatches[:8]}")
        parent_source_args = provenance.get("args", {})
        if provenance.get("run_identity_source_sha256") != parent_code_files.get("src/wnn/run_identity.py"):
            raise ValueError(f"{arm} run-identity provenance does not match its sealed E424 source")
        expected_model_source = "src/wnn/specialist.py" if parent_source_args.get("arch") in ("specialist", "prefix-adapter") else "src/wnn/nn.py"
        if (provenance.get("model_source_sha256") is not None
                and provenance["model_source_sha256"] != parent_code_files.get(expected_model_source)):
            raise ValueError(f"{arm} model-source provenance does not match its sealed E424 source")
        args = parent_source_args
        if args.get("arch", "gru") != EXPECTED_ARCH[arm]:
            raise ValueError(f"{arm} parent architecture differs from the preregistered map")
        if provenance.get("selected_epoch") != 3 or args.get("checkpoint_policy") != "fixed" or args.get("checkpoint_epoch") != 3:
            raise ValueError(f"{arm} parent is not the fixed epoch-3 checkpoint")
        groups = provenance.get("fit_row_groups", {}).get("valid", [])
        if len(groups) != 1505 or len(set(groups)) != 1505:
            raise ValueError(f"{arm} parent lacks the exact 1,505-sequence valid fit population")
        expected_from_provenance = provenance.get("checkpoint_sha256", {}).get("model_ep3.pt")
        checkpoint_sha = sha256_file(checkpoint_path)
        provenance_sha = sha256_file(provenance_path)
        if expected_from_provenance != checkpoint_sha:
            raise ValueError(f"{arm} checkpoint bytes disagree with provenance")
        child = f"c20261004_e464_{arm}_head_s3_final"
        if (ROOT / "artifacts/runs" / child).exists():
            raise FileExistsError(f"refusing to draft over existing child run: {child}")
        if args.get("aux_struct") != 0.3 or args.get("aux_next") != 0.3:
            raise ValueError(f"{arm} parent does not have the preregistered inherited auxiliary targets")
        freeze = FROZEN[EXPECTED_ARCH[arm]]
        trainer_argv = incumbent_trainer_argv(parent, child, freeze)
        audit = f"research/campaign_2026-10-04/staging/e464/audits/{child}.json"
        argv = [
            "{python}", "research/campaign_2026-10-04/staging/e464/scripts/run_head_fit.py",
            "--arm", arm, "--parent", parent, "--run-name", child,
            "--expected-parent-sha256", checkpoint_sha,
            "--expected-parent-provenance-sha256", provenance_sha,
            "--parent-code-spec", parent_code_spec_path,
            "--expected-parent-code-spec-sha256", parent_code_spec_sha,
            "--audit", audit,
        ]
        input_paths = sorted(set(shared_inputs + [
            f"artifacts/runs/{parent}/model_ep3.pt",
            f"artifacts/runs/{parent}/provenance.json",
            parent_code_spec_path,
        ]))
        draft = {
            "status": "draft_unsealed",
            "version": 1,
            "name": child,
            "arm": arm,
            "kind": "gpu",
            "parent": {
                "run": parent, "epoch": 3,
                "checkpoint_path": f"artifacts/runs/{parent}/model_ep3.pt",
                "checkpoint_sha256": checkpoint_sha,
                "provenance_path": f"artifacts/runs/{parent}/provenance.json",
                "provenance_sha256": provenance_sha,
                "parent_sealed_code_spec": parent_code_spec_path,
                "parent_sealed_code_spec_sha256": parent_code_spec_sha,
                "parent_sealed_code_aggregate_sha256": parent_code_spec.get("code_sha256"),
                "parent_valid_fit_count": 1505,
                "parent_arch": EXPECTED_ARCH[arm],
            },
            "architecture_scope": {
                "freeze_prefixes": list(freeze),
                "trainable_prefixes": list(
                    ("universal_head.", "expert_heads.") if EXPECTED_ARCH[arm] == "specialist"
                    else ("head.",)
                ),
                "audit": "all state_dict non-head parameters and every buffer bitwise equal to parent; head tensor shapes/dtypes fixed",
            },
            "argv": argv,
            "trainer_argv_contract": trainer_argv,
            "code": code,
            "inputs": input_paths,
            "artifacts": [
                f"artifacts/runs/{child}/model_ep0.pt",
                f"artifacts/runs/{child}/model.pt",
                f"artifacts/runs/{child}/provenance.json",
                f"artifacts/runs/{child}/history.json",
                f"artifacts/runs/{child}/training_state.pt",
                audit,
            ],
            "limits": {
                "cpus": 8, "ram_bytes": 24 * 1024**3,
                "timeout_seconds": 1200,
                "gpu_memory_bytes": 4_200_000_000,
                "gpu_stop_bytes": 4_800_000_000,
            },
            "environment": {
                "WNN_LOADER_WORKERS": "2",
                "WNN_LOADER_RAM_GB": "3",
                "WNN_PORTABLE_DATA_MANIFEST": "research/campaign_2026-10-04/portable_inputs.json",
            },
        }
        write_json(STAGING / "specs" / f"{child}.draft.json", draft)
        bundle["arms"].append({"arm": arm, "child": child, "parent": parent,
                               "parent_checkpoint_sha256": checkpoint_sha,
                               "parent_provenance_sha256": provenance_sha,
                               "parent_code_spec_sha256": parent_code_spec_sha,
                               "draft": f"research/campaign_2026-10-04/staging/e464/specs/{child}.draft.json"})
    write_json(STAGING / "specs" / "e464_bundle.draft.json", bundle)
    summary = {
        "status": "source_prepared_draft_unsealed",
        "registered_hypothesis": "E464 complete architecture head-only transfer for all eight E424 seed-3 fixed-epoch-3 parents",
        "preregistration_path": "research/campaign_2026-10-04/E464_COMPLETE_ARCHITECTURE_PREREG.md",
        "preregistration_sha256": sha256_file(CAMPAIGN_ROOT / "E464_COMPLETE_ARCHITECTURE_PREREG.md"),
        "builder_sha256": sha256_file(Path(__file__)),
        "wrapper_sha256": sha256_file(STAGING / "scripts/run_head_fit.py"),
        "synthetic_test_sha256": sha256_file(STAGING / "tests/test_e464_scope_audit.py"),
        "draft_bundle_sha256": sha256_file(STAGING / "specs/e464_bundle.draft.json"),
        "arms": bundle["arms"],
        "source_closure_files": len(code),
        "parent_fit_source_identity": "all current code paths match each parent's sealed E424 v2 source closure; source spec path/hash is bound into each draft",
        "validation": {
            "builder_imports": ["standard_library", "scripts/continuation_recipes.py"],
            "torch_imported_by_builder": False,
            "torch_loaded_for_parent_inspection": False,
            "tests_added_not_run": True,
            "training_or_prediction_started": False,
            "provider_calls_made": False,
        },
        "resource_gate": bundle["resource_plan"],
        "next_admission_gate": "root review and bounded CPU architecture/state audit, then verify E425/E426 and one-GPU queue terminal plus cumulative 80-device-GPU-hour budget before any local GPU fit",
    }
    write_json(STAGING / "E464_STAGING_SUMMARY.json", summary)
    print(json.dumps({"status": "drafts_created_unsealed", "arms": len(bundle["arms"]),
                      "source_files": len(code), "shared_inputs": len(shared_inputs),
                      "draft_root": str(STAGING / "specs")}, indent=2))


if __name__ == "__main__":
    main()
