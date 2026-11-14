#!/usr/bin/env python3
"""Verify and summarize the six sealed Kaggle R2 worker outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path, PurePosixPath

PACKET = Path(__file__).resolve().parent
RUNS = (
    "q14s_a50_s3",
    "q14s_a90_s3",
    "q14s_a50_s4",
    "q14s_a90_s4",
    "q14s_boot8_s3",
    "q14s_boot8_s4",
)
VARIANTS = {
    "q14s_a50_s3": (3, 0.5, 0),
    "q14s_a90_s3": (3, 0.9, 0),
    "q14s_a50_s4": (4, 0.5, 0),
    "q14s_a90_s4": (4, 0.9, 0),
    "q14s_boot8_s3": (3, 0.9, 8),
    "q14s_boot8_s4": (4, 0.9, 8),
}
SPLIT_SHA256 = "6d23a1b76f66d8001dac805d2b218a3b139e4eebe555f0c9b96fe1263ce186f0"
TEACHER = "tt29rep_fbc1k_s5"
METRIC = "global_wp_clip2_v1"
ALLOWED_ARG_VARIATION = {"name", "seed", "distill_alpha", "distill_boot_h"}
ALPHA90_MIN_MEAN_DELTA = 0.0005


class CollectionError(RuntimeError):
    """A present output contradicts the sealed recipe or its integrity records."""


def sha256_file(path: Path) -> str:
    """Hash a downloaded artifact as a stream."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, *, max_bytes: int = 32 << 20) -> dict | list:
    """Read bounded JSON evidence and make malformed files a clear collection failure."""
    try:
        size = path.stat().st_size
        if size > max_bytes:
            raise CollectionError(f"{path}: exceeds {max_bytes} byte read limit")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CollectionError(f"{path}: cannot read JSON: {error}") from error
    if not isinstance(value, (dict, list)):
        raise CollectionError(f"{path}: expected a JSON object or array")
    return value


def expected_specs() -> dict[str, dict]:
    """Load the six local sealed specs used to validate actual worker outputs."""
    specs = {}
    for name in RUNS:
        spec = read_json(PACKET / "specs" / f"{name}.json")
        if not isinstance(spec, dict) or spec.get("name") != name:
            raise CollectionError(f"sealed spec identity mismatch: {name}")
        specs[name] = spec
    return specs


def output_directory(output_root: Path) -> Path:
    """Accept either Kaggle's download folder or its contained ``results`` folder."""
    if (output_root / "results").is_dir():
        return output_root / "results"
    return output_root


def expected_cli_values(spec: dict, actual_args: dict) -> None:
    """Require every explicitly sealed CLI option to match saved run arguments."""
    argv = spec["argv"][3:]
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            raise CollectionError(f"{spec['name']}: unexpected argv token {token!r}")
        key, equal, inline = token[2:].partition("=")
        key = key.replace("-", "_")
        if equal:
            raw = inline
            has_value = True
        elif index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            raw = argv[index + 1]
            has_value = True
            index += 1
        else:
            raw = "true"
            has_value = False
        if key not in actual_args:
            raise CollectionError(f"{spec['name']}: saved args omit --{key.replace('_', '-')}")
        observed = actual_args[key]
        if not has_value:
            wanted = True
        elif isinstance(observed, bool):
            wanted = raw.lower() in {"1", "true", "yes"}
        elif isinstance(observed, int):
            wanted = int(raw)
        elif isinstance(observed, float):
            wanted = float(raw)
        else:
            wanted = raw
        # setup_lineage rewrites the CLI alias when a campaign split is supplied.
        if key == "eval_source" and "--split-manifest" in argv and raw == "dev":
            wanted = "vfold"
        if key == "split_manifest":
            wanted = str(PurePosixPath(spec["environment"]["WNN_PROJECT_ROOT"]) / raw)
        if observed != wanted:
            raise CollectionError(
                f"{spec['name']}: saved arg {key}={observed!r}, expected {wanted!r}"
            )
        index += 1


def artifact_files(run_dir: Path, result: dict, spec: dict) -> dict[str, dict]:
    """Verify the runner's complete per-file size and SHA-256 manifest."""
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict) or artifacts.get("missing"):
        raise CollectionError(f"{spec['name']}: result manifest reports missing artifacts")
    entries = artifacts.get("files")
    if not isinstance(entries, list):
        raise CollectionError(f"{spec['name']}: result manifest has no files list")
    by_path = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise CollectionError(f"{spec['name']}: malformed artifact entry")
        relative = entry.get("path")
        posix = PurePosixPath(relative or "")
        if not relative or posix.is_absolute() or ".." in posix.parts or "\\" in relative:
            raise CollectionError(f"{spec['name']}: unsafe artifact path {relative!r}")
        if relative in by_path:
            raise CollectionError(f"{spec['name']}: duplicate artifact path {relative}")
        by_path[relative] = entry
    expected = set(spec["artifacts"])
    if set(by_path) != expected:
        raise CollectionError(f"{spec['name']}: artifact manifest paths differ from sealed spec")
    verified = {}
    artifact_root = run_dir / "artifacts"
    for relative, entry in by_path.items():
        path = artifact_root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file():
            raise FileNotFoundError(path)
        if not path.resolve().is_relative_to(artifact_root.resolve()):
            raise CollectionError(f"{spec['name']}: artifact path escapes downloaded results")
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        if actual_size != entry.get("size") or actual_sha != entry.get("sha256"):
            raise CollectionError(f"{spec['name']}: artifact identity mismatch: {relative}")
        verified[relative] = {"size": actual_size, "sha256": actual_sha}
    return verified


def check_runtime(path: Path, name: str) -> dict:
    """Require the recorded T4 capability and effective no-TF32/no-autocast controls."""
    receipt = read_json(path, max_bytes=1 << 20)
    if not isinstance(receipt, dict):
        raise CollectionError(f"{name}: runtime receipt is not an object")
    valid = (
        receipt.get("run") == name
        and "T4" in str(receipt.get("device_name", ""))
        and receipt.get("device_capability") == [7, 5]
        and receipt.get("cuda_matmul_allow_tf32") is False
        and receipt.get("cudnn_allow_tf32") is False
        and receipt.get("requested_precision") == "fp32"
        and receipt.get("cuda_autocast_enabled_at_entry") is False
        and receipt.get("autocast_context") == "cuda disabled around entire trainer entrypoint"
        and receipt.get("cudnn_rnn_fp32_precision") in {"ieee", "legacy_flags_t4_no_tf32_hardware"}
    )
    if not valid:
        raise CollectionError(f"{name}: T4 strict-FP32 runtime receipt failed")
    return receipt


def check_provenance(
    path: Path, name: str, spec: dict, artifact_hashes: dict[str, dict]
) -> tuple[dict, dict]:
    """Check declared recipe pairs, clean split membership, and exact teacher cache ancestry."""
    provenance = read_json(path)
    if not isinstance(provenance, dict):
        raise CollectionError(f"{name}: provenance is not an object")
    args = provenance.get("args")
    if not isinstance(args, dict):
        raise CollectionError(f"{name}: provenance has no serialized args")
    expected_cli_values(spec, args)
    seed, alpha, boot_h = VARIANTS[name]
    if (
        args.get("name") != name
        or args.get("seed") != seed
        or args.get("epochs") != 4
        or args.get("checkpoint_policy") != "fixed"
        or args.get("checkpoint_epoch") != 3
        or args.get("precision") != "fp32"
        or args.get("distill") != TEACHER
        or args.get("distill_alpha") != alpha
        or args.get("distill_boot_h") != boot_h
        or provenance.get("selected_epoch") != 3
        or provenance.get("split_manifest_sha256") != SPLIT_SHA256
    ):
        raise CollectionError(f"{name}: saved recipe, fixed checkpoint, or split identity differs")
    fit_groups = provenance.get("fit_row_groups")
    train_groups = fit_groups.get("train") if isinstance(fit_groups, dict) else None
    valid_groups = fit_groups.get("valid") if isinstance(fit_groups, dict) else None
    evaluation_groups = provenance.get("evaluation_row_groups")
    split_path = (
        PACKET.parents[2] / "campaign_2026-10-29/teacher_track_r1/chronological_primary.json"
    )
    if sha256_file(split_path) != SPLIT_SHA256:
        raise CollectionError("local frozen split identity differs")
    split = read_json(split_path)
    if (
        not isinstance(train_groups, list)
        or len(train_groups) != 10_325
        or len(set(train_groups)) != 10_325
        or valid_groups != []
        or not isinstance(evaluation_groups, list)
        or len(evaluation_groups) != 1_873
        or len(set(evaluation_groups)) != 1_873
        or set(train_groups) != set(split["fit"]["train"])
        or set(evaluation_groups) != set(split["evaluation"]["valid"])
    ):
        raise CollectionError(f"{name}: train-only fit/evaluation membership differs")

    inputs = {item["path"]: item for item in spec["inputs"]}
    teacher_model = inputs[f"artifacts/runs/{TEACHER}/model.pt"]["sha256"]
    teacher_pred = inputs[f"cache/pred_{TEACHER}_train.npy"]["sha256"]
    bindings = args.get("distill_provenance")
    if (
        not isinstance(bindings, list)
        or len(bindings) != 1
        or not isinstance(bindings[0], dict)
        or bindings[0].get("run") != TEACHER
        or bindings[0].get("split") != "train"
        or bindings[0].get("scale") != 1.0
        or bindings[0].get("checkpoint_sha256") != teacher_model
        or bindings[0].get("prediction_sha256") != teacher_pred
    ):
        raise CollectionError(f"{name}: teacher checkpoint/prediction ancestry differs")
    checkpoint_hashes = provenance.get("checkpoint_sha256")
    for filename in ("model_ep3.pt", "model.pt"):
        relative = f"artifacts/runs/{name}/{filename}"
        if (
            not isinstance(checkpoint_hashes, dict)
            or checkpoint_hashes.get(filename) != artifact_hashes[relative]["sha256"]
        ):
            raise CollectionError(f"{name}: provenance does not bind {filename}")
    return provenance, args


def check_history(path: Path, name: str) -> tuple[list[dict], float]:
    """Require four chronological full-valid epochs and return the fixed epoch-3 score."""
    history = read_json(path, max_bytes=16 << 20)
    if not isinstance(history, list) or len(history) != 4:
        raise CollectionError(f"{name}: history must contain exactly four epochs")
    for epoch, row in enumerate(history):
        if not isinstance(row, dict) or row.get("epoch") != epoch or row.get("metric_id") != METRIC:
            raise CollectionError(f"{name}: history epoch/metric identity differs at {epoch}")
        score = row.get("weighted_pearson")
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not -1 <= score <= 1:
            raise CollectionError(f"{name}: invalid weighted_pearson at epoch {epoch}")
    return history, float(history[3]["weighted_pearson"])


def check_training_states(records: dict[str, dict]) -> dict[str, int]:
    """Load CPU training states only after every worker output passed collection preflight."""
    try:
        import torch
    except ImportError as error:
        raise CollectionError(
            f"PyTorch is required to inspect collected training states: {error}"
        ) from error

    counts = {}
    for name in RUNS:
        path = records[name]["paths"][f"artifacts/runs/{name}/training_state.pt"]
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except (
            OSError,
            RuntimeError,
            EOFError,
            ImportError,
            AttributeError,
            pickle.UnpicklingError,
        ) as error:
            raise CollectionError(f"{name}: cannot load trusted training state: {error}") from error
        if (
            not isinstance(state, dict)
            or state.get("boundary") != "epoch"
            or state.get("next_epoch") != 4
        ):
            raise CollectionError(
                f"{name}: training state is not the completed four-epoch boundary"
            )
        contract = state.get("contract")
        optimizer = state.get("optimizer")
        groups = optimizer.get("param_groups") if isinstance(optimizer, dict) else None
        if (
            not isinstance(contract, dict)
            or contract.get("epochs") != 4
            or not isinstance(groups, list)
            or not groups
            or any(
                not isinstance(group, dict) or group.get("strict_fp32") is not True
                for group in groups
            )
            or not any(group.get("muon") is True for group in groups)
        ):
            raise CollectionError(f"{name}: optimizer groups are not all strict FP32 Muon state")
        options = contract.get("options")
        provenance_args = records[name]["args"]
        excluded = {
            "name",
            "resume_state",
            "save_training_state",
            "stop_after_epoch",
            "split_manifest",
            "fit_groups_manifest",
        }
        expected_options = {
            key: value for key, value in provenance_args.items() if key not in excluded
        }
        if options != expected_options:
            raise CollectionError(f"{name}: saved continuation options differ from provenance args")
        code_hashes = {item["path"]: item["sha256"] for item in records[name]["spec"]["code"]}
        sources = contract.get("sources")
        if (
            not isinstance(sources, dict)
            or sources.get("training_dynamics.py")
            != code_hashes.get("src/wnn/training_dynamics.py")
            or sources.get("trainer") != code_hashes.get("scripts/train_rnn.py")
        ):
            raise CollectionError(
                f"{name}: training-state source identity differs from sealed code"
            )
        counts[name] = len(groups)
        del state
    return counts


def validate_config_pairs(records: dict[str, dict]) -> None:
    """Reject any serialized training-argument differences beyond the declared panel axes."""
    all_keys = set().union(*(record["args"] for record in records.values()))
    if any(set(record["args"]) != all_keys for record in records.values()):
        raise CollectionError("serialized run argument keys differ")
    varying = {
        key
        for key in all_keys
        if len({json.dumps(record["args"].get(key), sort_keys=True) for record in records.values()})
        > 1
    }
    if varying != ALLOWED_ARG_VARIATION:
        raise CollectionError(f"undeclared or missing config argument variation: {sorted(varying)}")


def build_complete_report(records: dict[str, dict], group_counts: dict[str, int]) -> dict:
    """Calculate paired contrasts and the predeclared alpha-0.9 screen rule."""
    scores = {name: records[name]["score"] for name in RUNS}
    alpha_deltas = {
        str(seed): scores[f"q14s_a90_s{seed}"] - scores[f"q14s_a50_s{seed}"] for seed in (3, 4)
    }
    boot_deltas = {
        str(seed): scores[f"q14s_boot8_s{seed}"] - scores[f"q14s_a90_s{seed}"] for seed in (3, 4)
    }
    alpha_mean = sum(alpha_deltas.values()) / 2
    return {
        "schema_version": 1,
        "status": "complete",
        "quality_claim": "not_established; exposed full-valid development evidence only",
        "metric": METRIC,
        "scores": scores,
        "paired_deltas": {
            "alpha_0_9_minus_0_5": {"by_seed": alpha_deltas, "mean": alpha_mean},
            "boot8_minus_alpha_0_9": {
                "by_seed": boot_deltas,
                "mean": sum(boot_deltas.values()) / 2,
            },
        },
        "alpha90_gate": {
            "rule": "mean paired delta >= 0.0005 and both seed deltas > 0",
            "threshold": ALPHA90_MIN_MEAN_DELTA,
            "seed_deltas": alpha_deltas,
            "mean_delta": alpha_mean,
            "both_seeds_positive": all(delta > 0 for delta in alpha_deltas.values()),
            "status": "met"
            if alpha_mean >= ALPHA90_MIN_MEAN_DELTA
            and all(delta > 0 for delta in alpha_deltas.values())
            else "not_met",
        },
        "optimizer_strict_fp32_group_counts": group_counts,
        "runs": {name: records[name]["public"] for name in RUNS},
        "scope_note": "Gate status evaluates the predeclared screen rule; it is not a promotion or leaderboard claim.",
    }


def collect(output_root: Path) -> tuple[dict, int]:
    """Validate one downloaded Kaggle output tree without grading partial campaigns."""
    report = {
        "schema_version": 1,
        "status": "incomplete",
        "quality_claim": "withheld until all six complete, verified outputs are present",
        "output_root": str(output_root),
        "scores": None,
        "paired_deltas": None,
        "alpha90_gate": {"status": "not_evaluated", "threshold": ALPHA90_MIN_MEAN_DELTA},
        "runs": {},
        "issues": [],
    }
    try:
        specs = expected_specs()
    except CollectionError as error:
        report["status"] = "failed"
        report["issues"].append(str(error))
        return report, 1
    results_root = output_directory(output_root)
    summary_path = results_root / "summary.json"
    failed = False
    try:
        summary = read_json(summary_path)
    except FileNotFoundError:
        summary = None
        report["issues"].append(f"missing runner summary: {summary_path}")
    except CollectionError as error:
        summary = None
        report["issues"].append(str(error))
        failed = summary_path.exists()

    incomplete = summary is None
    finished_by_name = {}
    if isinstance(summary, dict):
        if "incident" not in summary or summary.get("incident") is not None:
            failed = True
            report["issues"].append(
                f"runner incident: {summary.get('incident', '<missing field>')}"
            )
        finished = summary.get("finished")
        if not isinstance(finished, list):
            failed = True
            report["issues"].append("runner summary has no finished list")
            finished = []
        finished_by_name = {item.get("name"): item for item in finished if isinstance(item, dict)}
        if len(finished_by_name) != len(finished) or set(finished_by_name) - set(RUNS):
            failed = True
            report["issues"].append("runner summary has duplicate or unexpected run identities")
        not_started = summary.get("not_started")
        if not isinstance(not_started, list):
            failed = True
            report["issues"].append("runner summary has no valid not_started list")
            not_started = []
        if any(not isinstance(name, str) for name in not_started) or set(not_started) - set(RUNS):
            failed = True
            report["issues"].append("runner summary lists unexpected not_started identities")
        if not_started:
            incomplete = True
            report["issues"].append(f"workers not started: {not_started}")
    elif summary is not None:
        failed = True
        report["issues"].append("runner summary is not a JSON object")
    records = {}
    for name in RUNS:
        run_dir = results_root / name
        result_path = run_dir / "result.json"
        failure_path = run_dir / "failure.json"
        if not result_path.is_file():
            if failure_path.is_file():
                failed = True
                report["runs"][name] = {"status": "failed", "reason": "runner wrote failure.json"}
                report["issues"].append(f"{name}: worker failed")
            else:
                incomplete = True
                report["runs"][name] = {
                    "status": "incomplete",
                    "reason": "result.json not downloaded",
                }
            continue
        try:
            result = read_json(result_path, max_bytes=2 << 20)
            if not isinstance(result, dict) or result.get("name") != name:
                raise CollectionError(f"{name}: result identity mismatch")
            if result.get("status") != "complete" or result.get("returncode") != 0:
                raise CollectionError(f"{name}: runner result is {result.get('status')!r}")
            if summary is not None and finished_by_name.get(name) != result:
                raise CollectionError(f"{name}: per-run result differs from runner summary record")
            device = result.get("device")
            if not isinstance(device, dict) or "T4" not in str(device.get("name", "")):
                raise CollectionError(f"{name}: runner device record is not a T4")
            hashes = artifact_files(run_dir, result, specs[name])
            paths = {
                relative: run_dir / "artifacts" / Path(*PurePosixPath(relative).parts)
                for relative in hashes
            }
            worker_log = run_dir / "worker.log"
            if not worker_log.is_file():
                incomplete = True
                report["runs"][name] = {
                    "status": "incomplete",
                    "reason": "worker.log not downloaded",
                }
                continue
            with worker_log.open(encoding="utf-8", errors="replace") as stream:
                selected_fixed = any(line.strip() == "SELECTED_FIXED 3" for line in stream)
            if not selected_fixed:
                raise CollectionError(f"{name}: log lacks SELECTED_FIXED 3")
            runtime_path = paths[f"artifacts/runtime/{name}.json"]
            runtime = check_runtime(runtime_path, name)
            provenance_path = paths[f"artifacts/runs/{name}/provenance.json"]
            provenance, args = check_provenance(provenance_path, name, specs[name], hashes)
            history_path = paths[f"artifacts/runs/{name}/history.json"]
            history, score = check_history(history_path, name)
            records[name] = {
                "args": args,
                "history": history,
                "paths": paths,
                "provenance": provenance,
                "score": score,
                "spec": specs[name],
                "public": {
                    "status": "verified",
                    "score_epoch": 3,
                    "score": score,
                    "result_json_sha256": sha256_file(result_path),
                    "worker_log_sha256": sha256_file(worker_log),
                    "artifact_files": hashes,
                    "runtime": runtime,
                    "selected_fixed_epoch": 3,
                },
            }
            report["runs"][name] = records[name]["public"]
        except FileNotFoundError as error:
            incomplete = True
            report["runs"][name] = {
                "status": "incomplete",
                "reason": f"artifact not downloaded: {error.filename}",
            }
        except (
            CollectionError,
            OSError,
            json.JSONDecodeError,
            ValueError,
            TypeError,
            KeyError,
        ) as error:
            failed = True
            report["runs"][name] = {"status": "failed", "reason": str(error)}
            report["issues"].append(str(error))

    missing_summary_names = set(RUNS) - set(finished_by_name)
    if summary is not None and missing_summary_names:
        incomplete = True
        report["issues"].append(f"runs absent from runner summary: {sorted(missing_summary_names)}")
    if failed:
        for run in report["runs"].values():
            run.pop("score", None)
        report["status"] = "failed"
        return report, 1
    if incomplete or set(records) != set(RUNS):
        for run in report["runs"].values():
            run.pop("score", None)
        report["status"] = "incomplete"
        return report, 2

    try:
        validate_config_pairs(records)
        group_counts = check_training_states(records)
    except (
        CollectionError,
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        KeyError,
        pickle.UnpicklingError,
    ) as error:
        report["status"] = "failed"
        report["issues"].append(str(error))
        for run in report["runs"].values():
            run.pop("score", None)
        return report, 1
    complete_report = build_complete_report(records, group_counts)
    return complete_report, 0


def write_report(path: Path, report: dict) -> None:
    """Atomically write the machine-readable collection report when requested."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main() -> int:
    """Run collection checks and print a JSON report; noncomplete data never get scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_root", type=Path, help="Kaggle output download directory or its results/ child"
    )
    parser.add_argument("--report-json", type=Path, help="also write the JSON report to this path")
    args = parser.parse_args()
    report, status = collect(args.output_root)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.report_json is not None:
        write_report(args.report_json, report)
    return status


if __name__ == "__main__":
    sys.exit(main())
