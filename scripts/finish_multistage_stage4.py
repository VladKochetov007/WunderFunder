from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path
from types import ModuleType

PRIMARY_SHA = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
AUXILIARY_SHA = "26b448bad7f60612cea482f4abaf5f184e5a724228eef227d7e78a8155190075"
FAILURE = "only 23 of the admitted 24 trials were executed"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _read(path: Path, limit: int = 4_000_000) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("recovery requires a bounded regular JSON file")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("recovery input must be a JSON object")
    return value


def history_snapshot(database: Path, study_name: str, count: int) -> dict[str, object]:
    """Validate exact COMPLETE states and hash every original trial's scientific records."""
    if database.is_symlink() or not database.is_file():
        raise ValueError("original Optuna database must be a regular file")
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        studies = connection.execute(
            "SELECT study_id FROM studies WHERE study_name=?", (study_name,)
        ).fetchall()
        if len(studies) != 1:
            raise ValueError("recovery requires the exact original Optuna study")
        study_id = studies[0][0]
        trials = connection.execute(
            "SELECT * FROM trials WHERE study_id=? ORDER BY number", (study_id,)
        ).fetchall()
        columns = [row[1] for row in connection.execute("PRAGMA table_info(trials)")]
        records = [dict(zip(columns, row, strict=True)) for row in trials]
        if (
            len(records) != count
            or [row["number"] for row in records] != list(range(count))
            or any(row["state"] != "COMPLETE" for row in records)
        ):
            raise ValueError(f"recovery requires exactly {count} COMPLETE trials and zero others")
        original_ids = [row["trial_id"] for row in records[:23]]
        history: dict[str, object] = {"trials": trials[:23]}
        placeholders = ",".join("?" for _ in original_ids)
        for table in (
            "trial_values",
            "trial_params",
            "trial_user_attributes",
            "trial_system_attributes",
        ):
            history[table] = connection.execute(
                f"SELECT * FROM {table} WHERE trial_id IN ({placeholders}) ORDER BY 1", original_ids
            ).fetchall()
        attrs = connection.execute(
            f"SELECT key,value_json FROM trial_user_attributes WHERE trial_id IN ({placeholders})",
            original_ids,
        ).fetchall()
        if (
            sum(
                key == "search_role" and json.loads(value) == "convex_weighted_mse_warmstart"
                for key, value in attrs
            )
            != 1
        ):
            raise ValueError(
                "original warmstart must already be COMPLETE; no new queued trial is allowed"
            )
    return {"count": count, "original_23_sha256": _canonical(history), "original_23": history}


def _load_original(root: Path, request: dict[str, object]) -> ModuleType:
    sources = request.get("runner_sources")
    if not isinstance(sources, dict) or len(sources) < 19:
        raise ValueError("original runner request lacks its scientific source inventory")
    for relative, expected in sources.items():
        path = root / relative
        if (
            not path.resolve(strict=True).is_relative_to(root)
            or path.is_symlink()
            or _sha(path) != expected
        ):
            raise ValueError(f"original scientific source changed: {relative}")
    if any(name == "wnn" or name.startswith("wnn.") for name in sys.modules):
        raise ValueError("recovery must start in a fresh process before importing wnn")
    spec = importlib.util.spec_from_file_location(
        "original_stage4_runner", root / "scripts/multistage_stage4_runner.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot import the original runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import wnn

    if not Path(wnn.__file__).resolve().is_relative_to(root):
        raise ValueError("scientific imports resolved outside the original source")
    return module


def finish(args: argparse.Namespace) -> dict[str, object]:
    """Complete only the previously admitted 24th trial in the unchanged original study."""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    run_dir, source = (
        args.run_dir.resolve(strict=True),
        args.original_source_root.resolve(strict=True),
    )
    output = args.recovery_output.resolve()
    if output.exists() or output.is_symlink() or (run_dir / "stage4/ensemble_result.json").exists():
        raise FileExistsError("recovery output must be fresh and original result must be absent")
    request_path = run_dir / "stage4/runner_request.json"
    request = _read(request_path)
    if (
        _sha(request_path) != args.runner_request_sha256
        or request.get("optuna_trials") != 24
        or request.get("used_primary_valid") is not False
        or request.get("scope") != "inner_train_proxy"
    ):
        raise ValueError("recovery does not bind the original frozen 24-trial request")
    child = _read(args.failed_child_receipt)
    log = args.failure_log
    if log.is_symlink() or not log.is_file() or log.stat().st_size > 16_000_000:
        raise ValueError("original failure log must be bounded and regular")
    command = child.get("command")
    if (
        child.get("schema") != "wnn.multistage-owned-process.v1"
        or child.get("status") != "failed"
        or child.get("exit_code") != 1
        or child.get("timed_out") is not False
        or child.get("stage") != 4
        or not isinstance(command, list)
        or request.get("command") not in (command, command[1:])
        or FAILURE not in log.read_text(encoding="utf-8")
    ):
        raise ValueError(
            "recovery requires the original child exit1 caused by the exact 23-of-24 refusal"
        )
    if (
        Path(str(request["command"][0])).resolve(strict=True)
        != source / "scripts/multistage_stage4_runner.py"
    ):
        raise ValueError("recovery source root differs from the original OP16 command")
    elapsed = child.get("finished_at_epoch", 0) - child.get("started_at_epoch", 0)
    if not 0 < elapsed <= 1500:
        raise ValueError(
            "original elapsed time leaves insufficient room for the bounded300s recovery within1800s"
        )
    original = _load_original(source, request)
    manifest = original.read_json(run_dir / "run.json")
    _, admission = original.read_bound_campaign_reference(manifest.get("admission"), "admission")
    _, contract = original.read_bound_campaign_reference(manifest.get("contract"), "contract")
    data = admission["data"]
    data_root = Path(str(data["data_root"])).resolve(strict=True)
    auxiliary_path, _ = original.read_bound_campaign_reference(
        manifest.get("auxiliary_manifest"), "auxiliary split"
    )
    portable = args.portable_data_manifest.resolve(strict=True)
    os.environ["WNN_PORTABLE_DATA_MANIFEST"] = str(portable)
    split = original.load_campaign_split(auxiliary_path, data_root)
    if (
        _sha(run_dir / "run.json") != request.get("run_manifest_sha256")
        or manifest["admission"]["sha256"] != request.get("admission_sha256")
        or manifest["contract"]["sha256"] != request.get("contract_sha256")
        or _sha(portable) != request.get("portable_data_manifest_sha256")
        or split.sha256 != AUXILIARY_SHA
        or request.get("auxiliary_split_sha256") != AUXILIARY_SHA
        or split.primary_split_sha256 != PRIMARY_SHA
        or request.get("primary_split_sha256") != PRIMARY_SHA
        or split.evaluation_source != "train"
        or len(split.train_groups) != 8260
        or len(split.evaluation_groups) != 2065
        or admission["stage4"].get("optuna_trials", 24) != 24
    ):
        raise ValueError("recovery run/data/admission differs from the original fitting-only study")
    inventory = request.get("candidate_trial_results")
    if not isinstance(inventory, list) or len(inventory) != 8:
        raise ValueError("recovery requires exactly eight original candidates")
    candidates = original.load_finalized_candidates(
        run_dir,
        [f"{row['name']}={row['path']}" for row in inventory],
        AUXILIARY_SHA,
        PRIMARY_SHA,
        split,
        [["train", group] for group in split.train_groups],
        [["train", group] for group in split.evaluation_groups],
    )
    for bound, candidate in zip(inventory, candidates, strict=True):
        if any(
            bound[key] != candidate[target]
            for key, target in (
                ("name", "name"),
                ("stage", "stage"),
                ("family", "family"),
                ("trial_number", "trial_number"),
                ("sha256", "trial_result_sha256"),
                ("trial_request_sha256", "trial_request_sha256"),
                ("config_sha256", "config_sha256"),
                ("config_identity_sha256", "config_identity_sha256"),
            )
        ):
            raise ValueError("recovery candidate changed from its immutable original request")
    study_name = f"{manifest['experiment_id']}-stage4-wp"
    database = run_dir / "stage4/optuna.db"
    before = history_snapshot(database, study_name, 23)
    output.mkdir(parents=True, exist_ok=False)
    recovery_request = {
        "schema": "wnn.multistage-stage4-recovery-request.v1",
        "helper_sha256": _sha(Path(__file__)),
        "original_runner_request_sha256": _sha(request_path),
        "original_source_root": str(source),
        "original_sources": request["runner_sources"],
        "original_child_receipt_sha256": _sha(args.failed_child_receipt),
        "failure_log_sha256": _sha(log),
        "history_before": before,
        "additional_trials": 1,
        "total_trial_cap": 24,
        "outer_wall_cap_seconds": 300,
        "sampler": {
            "seed": int(manifest["sampler_seed"]) + 4,
            "rng_state": "restarted sampler RNG; not continuous RNG state",
        },
        "primary_valid_used": False,
        "new_fit_or_profile": False,
    }
    original.atomic_write_json(output / "recovery_request.json", recovery_request, replace=False)
    predictions = [candidate["predictions"] for candidate in candidates]
    quadratic, linear, constant, total_weight = original.accumulate_quadratic(
        data_root, split, predictions
    )
    warmstart = original.solve_simplex(quadratic, linear)
    warmstart_mse = float(
        (warmstart @ quadratic @ warmstart - 2 * warmstart @ linear + constant) / total_weight
    )
    weights, score, count = original.search_wp_weights(
        run_dir,
        data_root,
        split,
        predictions,
        [candidate["name"] for candidate in candidates],
        warmstart,
    )
    after = history_snapshot(database, study_name, 24)
    if (
        count != 24
        or before["original_23_sha256"] != after["original_23_sha256"]
        or _sha(request_path) != args.runner_request_sha256
    ):
        raise ValueError(
            "recovery changed original history/request or failed to finish exactly24 COMPLETE trials"
        )
    original.verify_finalized_candidate_inputs_unchanged(candidates)
    result = original.build_stage4_ensemble_result(
        candidates,
        weights,
        score,
        count,
        warmstart,
        warmstart_mse,
        AUXILIARY_SHA,
        PRIMARY_SHA,
        [["train", group] for group in split.evaluation_groups],
        contract["baseline"]["canonical_fresh_TL3_wp"],
    )
    destination = run_dir / "stage4/ensemble_result.json"
    original.atomic_write_json(destination, result, replace=False)
    original.export_stage4_trial_plots(run_dir)
    original.export_stage4_ensemble_plots(destination)
    original.append_event(
        run_dir / "events.jsonl",
        {
            "event": "stage4_ensemble_complete",
            "at_utc": original.utc_now(),
            "stage": 4,
            "result_path": str(destination),
            "result_sha256": _sha(destination),
            "objective_value": score["wp"],
            "status": "COMPLETE",
            "recovery_request_sha256": _sha(output / "recovery_request.json"),
        },
    )
    receipt = {
        "status": "complete",
        "result_path": str(destination),
        "result_sha256": _sha(destination),
        "original_history_preserved": True,
        "original_23_sha256": before["original_23_sha256"],
        "complete_trials": count,
        "recovery_request_sha256": _sha(output / "recovery_request.json"),
    }
    original.atomic_write_json(output / "recovery_receipt.json", receipt, replace=False)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "run-dir",
        "original-source-root",
        "portable-data-manifest",
        "failed-child-receipt",
        "failure-log",
        "recovery-output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--runner-request-sha256", required=True)
    print(json.dumps(finish(parser.parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
