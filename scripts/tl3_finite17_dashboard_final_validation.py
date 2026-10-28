from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

try:
    from scripts.tl3_finite17_dashboard_core import (
        SHA256,
        VALID_GROUPS,
        VALID_ROWS,
        BridgeError,
        _canonical_sha256,
        _read_json,
        _sha256_file,
    )
    from scripts.tl3_finite17_results_validation import _verify_run
except ModuleNotFoundError:
    from tl3_finite17_dashboard_core import (
        SHA256,
        VALID_GROUPS,
        VALID_ROWS,
        BridgeError,
        _canonical_sha256,
        _read_json,
        _sha256_file,
    )
    from tl3_finite17_results_validation import _verify_run

FINAL_SCORE_SCHEMA = "wnn.tl3-chronological-score.v1"


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _source_constants(path: Path) -> tuple[str, dict[str, float]]:
    """Read frozen PREFIX and FINAL_WEIGHTS constants without importing project code."""
    if path.is_symlink() or not path.is_file():
        raise BridgeError("fixed recipe source is missing or unsafe")
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, object] = {}
    for statement in module.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in {"PREFIX", "FINAL_WEIGHTS"}:
                    values[target.id] = ast.literal_eval(statement.value)
    prefix, raw_weights = values.get("PREFIX"), values.get("FINAL_WEIGHTS")
    if (
        not isinstance(prefix, str)
        or not isinstance(raw_weights, dict)
        or len(raw_weights) != 3
        or any(
            not isinstance(name, str) or not _finite(weight) for name, weight in raw_weights.items()
        )
        or any(float(weight) <= 0 for weight in raw_weights.values())
        or not math.isclose(sum(map(float, raw_weights.values())), 1.0, rel_tol=0, abs_tol=1e-12)
    ):
        raise BridgeError("fixed recipe source has malformed final members or weights")
    return prefix, {name: float(weight) for name, weight in raw_weights.items()}


def _verified_recipe(root: Path, plan: dict[str, Any]) -> tuple[list[str], dict[str, float]]:
    identity = plan.get("source_identity")
    files = identity.get("files") if isinstance(identity, dict) else None
    if not isinstance(files, dict) or identity.get("sha256") != _canonical_sha256(files):
        raise BridgeError("finite plan source identity file map is invalid")
    recipe_relative = "scripts/tl3_chronological_recipes.py"
    scorer_relative = "scripts/score_tl3_chronological.py"
    recipe_sha = files.get(recipe_relative)
    scorer_sha = files.get(scorer_relative)
    if (
        not isinstance(recipe_sha, str)
        or not SHA256.fullmatch(recipe_sha)
        or plan.get("recipe_sha256") != recipe_sha
        or not isinstance(scorer_sha, str)
        or not SHA256.fullmatch(scorer_sha)
        or plan.get("scorer_sha256") != scorer_sha
    ):
        raise BridgeError("finite plan omits the frozen recipe or scorer source hashes")
    recipe_path, scorer_path = root / recipe_relative, root / scorer_relative
    for path, expected in ((recipe_path, recipe_sha), (scorer_path, scorer_sha)):
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected:
            raise BridgeError(f"frozen source file changed: {path.name}")
    prefix, weights = _source_constants(recipe_path)
    run_weights = {f"{prefix}{name}": weight for name, weight in weights.items()}
    runs = list(run_weights)
    if plan.get("final_runs") != runs:
        raise BridgeError("finite plan final member order differs from its frozen recipe")
    return runs, run_weights


def _terminal_score(
    root: Path, state_dir: Path, plan: dict[str, Any], status: dict[str, Any]
) -> tuple[dict[str, Any], str] | None:
    if status.get("status") != "completed":
        return None
    if status.get("plan_sha256") != plan.get("sha256"):
        raise BridgeError("completed finite status differs from the frozen plan")
    fits_used = status.get("fits_used")
    if type(fits_used) is not int or not 1 <= fits_used <= 17:
        raise BridgeError("completed finite status has an invalid fit count")
    score_state = status.get("score")
    if not isinstance(score_state, dict) or score_state.get("state") != "complete":
        raise BridgeError("completed finite DAG has no verified fixed score stage")
    score_path = root / "final-score.json"
    if (
        score_path.is_symlink()
        or score_state.get("path") != str(score_path)
        or not score_path.is_file()
        or not isinstance(score_state.get("sha256"), str)
        or not SHA256.fullmatch(score_state["sha256"])
    ):
        raise BridgeError("fixed score artifact path or hash receipt is invalid")
    score_sha = _sha256_file(score_path)
    score = _read_json(score_path, 1 << 20)
    if score_sha != score_state["sha256"] or _sha256_file(score_path) != score_sha:
        raise BridgeError("fixed score artifact changed after driver verification")
    status_path, terminal_path = state_dir / "status.json", state_dir / "terminal.json"
    if status_path.is_symlink() or terminal_path.is_symlink():
        raise BridgeError("finite terminal receipts cannot be symlinks")
    disk_status = _read_json(status_path)
    terminal = _read_json(terminal_path)
    if (
        disk_status != status
        or terminal.get("schema") != "wnn.tl3-finite17-terminal.v1"
        or terminal.get("status") != "completed"
        or terminal.get("plan_sha256") != plan.get("sha256")
        or terminal.get("state_sha256") != _sha256_file(status_path)
        or terminal.get("fits_used") != fits_used
        or terminal.get("score") != score_state
    ):
        raise BridgeError("fixed score is not bound to the completed finite terminal receipt")
    return score, score_sha


def _verify_members(
    root: Path,
    plan: dict[str, Any],
    status: dict[str, Any],
    train: list[int],
    valid: list[int],
    runs: list[str],
    run_weights: dict[str, float],
    score: dict[str, Any],
) -> list[dict[str, Any]]:
    node_map = {
        str(node.get("run")): node for node in plan.get("nodes", []) if isinstance(node, dict)
    }
    states = status.get("nodes")
    exports = status.get("final_exports")
    score_members = score.get("members")
    if (
        not isinstance(states, dict)
        or not isinstance(exports, dict)
        or not isinstance(score_members, list)
        or len(score_members) != len(runs)
    ):
        raise BridgeError("finite score omits fixed member receipts")
    verified_members = []
    for run, member in zip(runs, score_members, strict=True):
        node = node_map.get(run)
        if node is None or not isinstance(member, dict):
            raise BridgeError("fixed score member differs from the finite plan")
        state = states.get(str(node["id"]))
        export = exports.get(run)
        verified = (
            _verify_run(root, plan, node, state, train, valid) if isinstance(state, dict) else None
        )
        if (
            verified is None
            or state.get("state") != "complete"
            or not isinstance(export, dict)
            or export.get("state") != "complete"
            or not _finite(member.get("weight"))
            or not math.isclose(float(member["weight"]), run_weights[run], rel_tol=0, abs_tol=1e-12)
            or member.get("epoch") != node.get("endpoint")
            or member.get("checkpoint_sha256") != verified["model_sha256"]
            or member.get("checkpoint_sha256") != export.get("checkpoint_sha256")
            or member.get("prediction_sha256") != export.get("sha256")
        ):
            raise BridgeError(f"{run}: final score member is not bound to its verified fit/export")
        prediction = root / "cache" / f"pred_{run}_valid.npy"
        sidecar = prediction.with_suffix(".json")
        if (
            prediction.is_symlink()
            or sidecar.is_symlink()
            or not prediction.is_file()
            or not sidecar.is_file()
            or _sha256_file(prediction) != member["prediction_sha256"]
            or _sha256_file(sidecar) != export.get("sidecar_sha256")
        ):
            raise BridgeError(f"{run}: fixed score prediction export changed")
        metadata = _read_json(sidecar, 1 << 20)
        if (
            metadata.get("run") != run
            or metadata.get("split") != "valid"
            or metadata.get("split_manifest_sha256") != plan["split"]["split_manifest_sha256"]
            or metadata.get("campaign_split_manifest_sha256")
            != plan["split"]["split_manifest_sha256"]
            or metadata.get("campaign_export_scope") != "evaluation"
            or metadata.get("feature_only") is not True
            or metadata.get("checkpoint_file") != "model.pt"
            or metadata.get("checkpoint_sha256") != member["checkpoint_sha256"]
            or metadata.get("prediction_sha256") != member["prediction_sha256"]
            or metadata.get("row_groups") != valid
        ):
            raise BridgeError(f"{run}: fixed score export metadata differs from official valid")
        verified_members.append(
            {
                "run": run,
                "weight": run_weights[run],
                "epoch": int(node["endpoint"]),
                "checkpoint_sha256": member["checkpoint_sha256"],
                "prediction_sha256": member["prediction_sha256"],
            }
        )
    return verified_members


def validate_final_score(
    root: Path,
    state_dir: Path,
    plan: dict[str, Any],
    status: dict[str, Any],
    train: list[int],
    valid: list[int],
) -> dict[str, Any] | None:
    """Require terminal driver, source, model, export, and score receipts to agree."""
    terminal = _terminal_score(root, state_dir, plan, status)
    if terminal is None:
        return None
    score, score_sha = terminal
    runs, run_weights = _verified_recipe(root, plan)
    if (
        score.get("schema") != FINAL_SCORE_SCHEMA
        or score.get("split_manifest_sha256") != plan["split"]["split_manifest_sha256"]
        or score.get("valid_sequences") != VALID_GROUPS
        or score.get("selected_rows") != VALID_ROWS
    ):
        raise BridgeError("fixed blend score does not cover the official full-valid identity")
    members = _verify_members(root, plan, status, train, valid, runs, run_weights, score)
    scores = score.get("scores")
    blend = scores.get("fixed_tl3_blend") if isinstance(scores, dict) else None
    if (
        not isinstance(blend, dict)
        or blend.get("metric_id") != "global_wp_clip2_v1"
        or blend.get("selected_rows") != VALID_ROWS
        or not all(_finite(blend.get(key)) for key in ("wp", "t0", "t1"))
        or not _finite(score.get("clean_gru_reference_wp"))
        or not _finite(score.get("blend_delta_to_clean_gru"))
        or not math.isclose(
            float(score["blend_delta_to_clean_gru"]),
            float(blend["wp"]) - float(score["clean_gru_reference_wp"]),
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise BridgeError(
            "fixed blend score is missing consistent official metrics or baseline delta"
        )
    return {
        "score": score,
        "score_sha256": score_sha,
        "run_weights": run_weights,
        "members": members,
        "blend": blend,
        "clean_gru_reference_wp": float(score["clean_gru_reference_wp"]),
        "delta_to_clean_gru": float(score["blend_delta_to_clean_gru"]),
    }
