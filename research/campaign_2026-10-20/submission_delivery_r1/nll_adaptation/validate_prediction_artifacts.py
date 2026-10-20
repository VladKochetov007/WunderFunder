"""Run the frozen prediction preflight or post-export audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prediction_artifact_validation import _absent, _base, _json, _pin, _sha
from prediction_export_audit import _export, _fit_args_by_name

RECIPE_SHA256 = "e34fa919ad49330bcb9ee8f06208da2346eda729d7462be49455d55078c00c25"
INPUT_VERIFICATION_SHA256 = "ef81bfeb79bcf9cf971e663c158ccff14a99e22aa0c0159a3ff623034634ccd5"


def _execution_records(root: Path, source_commit: str) -> tuple[Path, Path, dict[str, Any]]:
    """Check and return the admitted recipe and its frozen input receipt."""
    directory = root / "research/campaign_2026-10-20/submission_delivery_r1/nll_adaptation"
    recipe_path, verification_path = (
        directory / "recipe.json",
        directory / "input_verification.json",
    )
    if _sha(recipe_path) != RECIPE_SHA256 or _sha(verification_path) != INPUT_VERIFICATION_SHA256:
        raise ValueError("frozen recipe or input-verification receipt changed")
    verification = _json(verification_path)
    if (
        verification.get("recipe_sha256") != RECIPE_SHA256
        or verification.get("source_commit") != source_commit
        or verification.get("source_module_digests_match") is not True
    ):
        raise ValueError("input-verification receipt does not bind the frozen recipe/source")
    return recipe_path, verification_path, verification


def _producer_tree(worktree: Path, commit: str) -> dict[str, Any]:
    """Require pinned producer code and reject untracked Python files."""
    diff = subprocess.run(
        [
            "git",
            "-C",
            str(worktree),
            "diff",
            "--quiet",
            commit,
            "--",
            "src/wnn",
            "scripts",
            "utils.py",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if diff.returncode != 0:
        raise ValueError(f"producer code differs from {commit}: {diff.stderr.strip()}")
    untracked = subprocess.run(
        [
            "git",
            "-C",
            str(worktree),
            "ls-files",
            "--others",
            "--",
            "src/wnn",
            "scripts",
            "utils.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    python_files = [path for path in untracked if path.endswith(".py")]
    if python_files:
        raise ValueError(f"untracked producer Python files: {python_files}")
    return {
        "worktree": str(worktree),
        "base_commit": commit,
        "tracked_code_diff": False,
        "untracked_python": [],
    }


def _expected_ancestry(
    plan: dict[str, Any],
    recipe: dict[str, Any],
    verification: dict[str, Any],
    fit_args: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Resolve the exact parent, propensity, and target-cache provenance pins."""
    root = Path(plan["source_pins"]["project_root"])
    parent = next(item for item in plan["endpoints"] if item["id"] == "parent_cp5")
    inputs = {str(Path(item["path"])): item for item in verification["inputs"]}
    children = list(fit_args.values())
    propensities = {item["propensity"] for item in children}
    cache_roots = {item["distill_cache_root"] for item in children}
    if len(children) != 2 or len(propensities) != 1 or len(cache_roots) != 1:
        raise ValueError("matched child recipes must share one propensity and target cache")
    propensity_path = root / "artifacts/runs" / next(iter(propensities)) / "model.pt"
    propensity_pin = inputs.get(str(propensity_path))
    cache_root = Path(next(iter(cache_roots)))
    target_cache = recipe["target_cache"]
    metadata_path = cache_root / "metadata.json"
    metadata_pin = inputs.get(str(metadata_path))
    if (
        propensity_pin is None
        or metadata_pin is None
        or metadata_pin.get("sha256") != target_cache["metadata_sha256"]
    ):
        raise ValueError("input receipt omits the fitted ancestry or cache metadata")
    verified_files = {}
    for filename, digest in target_cache["target_sha256"].items():
        file_pin = inputs.get(str(cache_root / filename))
        if file_pin is None or file_pin.get("sha256") != digest:
            raise ValueError(f"input receipt target pin differs: {filename}")
        verified_files[Path(filename).stem] = digest
    return {
        "initial_checkpoint_sha256": parent["checkpoint_sha256"],
        "propensity_checkpoint_sha256": propensity_pin["sha256"],
        "distillation_cache": {
            "metadata_sha256": target_cache["metadata_sha256"],
            "verified_files": verified_files,
        },
    }


def main() -> None:
    """Run preflight or post-export audit and write a create-only receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "post-export"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--post-receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "post-export" and args.preflight_receipt is None:
        parser.error("--preflight-receipt is required for post-export")
    if args.phase == "preflight" and args.post_receipt is None:
        parser.error("--post-receipt is required for preflight")
    if args.phase == "post-export" and args.post_receipt is not None:
        parser.error("--post-receipt is only valid for preflight")
    receipt = args.receipt.resolve()
    try:
        if receipt.exists():
            raise FileExistsError(f"refusing to overwrite receipt: {receipt}")
        plan_path, eval_path = args.plan.resolve(), args.evaluation_manifest.resolve()
        plan_preview = _json(plan_path)
        root = Path(plan_preview["source_pins"]["project_root"])
        source_commit = plan_preview["source_pins"]["training_source_commit"]
        recipe_path, verification_path, verification = _execution_records(root, source_commit)
        plan, _, split, components, base = _base(plan_path, eval_path, verification)
        base["source_identities"].extend(
            [_pin(recipe_path, RECIPE_SHA256), _pin(verification_path, INPUT_VERIFICATION_SHA256)]
        )
        worktree = Path(plan["source_pins"]["predictor"]["path"]).parents[1]
        producer_tree = _producer_tree(worktree, source_commit)
        recipe = _json(recipe_path)
        fit_args = _fit_args_by_name(recipe)
        ancestry = _expected_ancestry(plan, recipe, verification, fit_args)
        audit_dir = Path(__file__).parent
        validator_files = [
            Path(__file__),
            audit_dir / "prediction_artifact_validation.py",
            audit_dir / "prediction_export_audit.py",
        ]
        validator_identity = {str(path.resolve()): _sha(path) for path in validator_files}
        payload: dict[str, Any] = {
            "schema_version": 1,
            "status": "PASS",
            "phase": args.phase,
            "created_utc": datetime.now(UTC).isoformat(),
            "validator_modules_sha256": validator_identity,
            "plan": {"path": str(plan_path), "sha256": _sha(plan_path)},
            "evaluation_manifest": {"path": str(eval_path), "sha256": _sha(eval_path)},
            "execution_recipe": {"path": str(recipe_path), "sha256": _sha(recipe_path)},
            "producer_execution_tree": producer_tree,
            **base,
        }
        if args.phase == "preflight":
            payload["new_output_paths_absent"] = _absent(plan, receipt, args.post_receipt.resolve())
            payload["completed_checkpoints"] = [
                _export(
                    root,
                    item,
                    split,
                    components,
                    plan["source_pins"]["split_manifest"]["sha256"],
                    expected_args=fit_args.get(item["run"]),
                    expected_ancestry=ancestry if item["run"] in fit_args else None,
                    check_predictions=False,
                )
                for item in plan["endpoints"]
                if item.get("argv")
            ]
        else:
            pre_path = args.preflight_receipt.resolve()
            pre = _json(pre_path)
            if (
                pre.get("status") != "PASS"
                or pre.get("phase") != "preflight"
                or pre.get("plan", {}).get("sha256") != payload["plan"]["sha256"]
                or pre.get("evaluation_manifest", {}).get("sha256")
                != payload["evaluation_manifest"]["sha256"]
                or pre.get("execution_recipe") != payload["execution_recipe"]
                or pre.get("source_identities") != payload["source_identities"]
                or pre.get("producer_execution_tree") != payload["producer_execution_tree"]
                or not pre.get("new_output_paths_absent")
                or pre.get("validator_modules_sha256") != validator_identity
            ):
                raise ValueError(
                    "preflight receipt does not bind this plan and evaluation manifest"
                )
            payload["preflight_receipt"] = {"path": str(pre_path), "sha256": _sha(pre_path)}
            prior = {row["id"]: row for row in pre.get("completed_checkpoints", [])}
            exports = []
            for item in plan["endpoints"]:
                if not item.get("argv"):
                    continue
                current = _export(
                    root,
                    item,
                    split,
                    components,
                    plan["source_pins"]["split_manifest"]["sha256"],
                    expected_args=fit_args.get(item["run"]),
                    expected_ancestry=ancestry if item["run"] in fit_args else None,
                )
                earlier = prior.get(item["id"], {})
                if (
                    any(
                        current.get(field, {}).get("sha256") != earlier.get(field, {}).get("sha256")
                        for field in ("checkpoint", "provenance")
                    )
                    or current.get("fit_args") != earlier.get("fit_args")
                    or current.get("lineage_pins") != earlier.get("lineage_pins")
                ):
                    raise ValueError(
                        f"{item['id']} checkpoint/provenance changed since pre-export binding"
                    )
                exports.append(current)
            payload["new_exports"] = exports
        receipt.parent.mkdir(parents=True, exist_ok=True)
        with receipt.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
    except (
        OSError,
        subprocess.SubprocessError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        AttributeError,
    ) as exc:
        print(
            json.dumps({"status": "FAIL", "phase": args.phase, "error": str(exc)}), file=sys.stderr
        )
        raise SystemExit(2) from exc
    print(json.dumps({"status": "PASS", "phase": args.phase, "receipt": str(receipt)}))


if __name__ == "__main__":
    main()
