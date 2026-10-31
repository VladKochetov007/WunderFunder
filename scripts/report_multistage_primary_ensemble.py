from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
SOURCE6_SHA256 = "8d97910b523931bb115c4a979c32ec1c37c7353a666bf0dcf6f22e5de246ed35"
SCORE_SCRIPT_SHA256 = "56502eb04c4cbe2495e78d6236ef354cb0820b1ecc63ab820610a804cfac89a3"
PRIMARY_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
AUXILIARY_SHA256 = "26b448bad7f60612cea482f4abaf5f184e5a724228eef227d7e78a8155190075"
TL3_WP = 0.6923774245652725
MEMBER_TIMEOUT_SECONDS = 900
OVERALL_TIMEOUT_SECONDS = 1_800


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str, limit: int = 32_000_000) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"{label} must be a bounded regular file")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _source6_info(root: Path) -> dict[str, object]:
    sys.path.insert(0, str(SCRIPTS))
    from report_multistage_primary_ensemble_worker import _source_identity, _verify_source6

    root = root.resolve(strict=True)
    identity = _source_identity(root)
    score = root / "scripts/score_sequence.py"
    info = {
        "root": str(root),
        "source_sha256": identity["sha256"],
        "score_script_sha256": _sha256(score),
    }
    _verify_source6(root, info)
    return info


def _candidate_refs(run_dir: Path, runner_request: dict[str, Any]) -> list[str]:
    inventory = runner_request.get("candidate_trial_results")
    if not isinstance(inventory, list) or len(inventory) != 8:
        raise ValueError("the frozen Stage4 request must bind exactly eight candidates")
    names, refs = set(), []
    for item in inventory:
        if not isinstance(item, dict) or item.get("stage") not in (1, 2, 3):
            raise ValueError("Stage4 candidate inventory has an invalid stage record")
        name, path = item.get("name"), item.get("path")
        if not isinstance(name, str) or not name or name in names or not isinstance(path, str):
            raise ValueError("Stage4 candidate names/paths are malformed or duplicated")
        candidate_source = Path(path)
        if candidate_source.is_symlink() or not candidate_source.is_file():
            raise ValueError("Stage4 candidate result must be a regular non-symlink file")
        candidate_path = candidate_source.resolve(strict=True)
        if not candidate_path.is_relative_to(run_dir) or _sha256(candidate_path) != item.get(
            "sha256"
        ):
            raise ValueError("Stage4 candidate result escapes the campaign or changed bytes")
        names.add(name)
        refs.append(f"{name}={candidate_path}")
    return refs


def prepare_request(args: argparse.Namespace) -> tuple[dict[str, Any], Path, str]:
    """Validate frozen Stage4 descendants and create a bounded primary score request."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    sys.path.insert(0, str(ROOT / "src"))
    from wnn.data import SEQUENCE_LENGTH, WARMUP
    from wnn.multistage_search import canonical_json_sha256, file_sha256
    from wnn.multistage_search_ensemble_finalized_inputs import (
        load_finalized_candidates,
        read_bound_campaign_reference,
        regular_campaign_file,
    )
    from wnn.splits import load_campaign_split

    run_dir = args.run_dir.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("primary ensemble output directory must be fresh")
    stage4_path = regular_campaign_file(args.stage4_result, "Stage4 result")
    if stage4_path != (run_dir / "stage4/ensemble_result.json").resolve(strict=True):
        raise ValueError("Stage4 result is not the campaign's frozen ensemble result")
    run_path = regular_campaign_file(run_dir / "run.json", "campaign run manifest")
    run = _read_json(run_path, "campaign run manifest")
    admission_path, admission = read_bound_campaign_reference(run.get("admission"), "admission")
    contract_path, contract = read_bound_campaign_reference(run.get("contract"), "contract")
    baseline = contract.get("baseline")
    if (
        not isinstance(baseline, dict)
        or baseline.get("canonical_fresh_TL3_wp") != TL3_WP
        or baseline.get("original_results")
        != "research/experiment_drafts/heterogeneous-tl3-20261030/results/R3/REPORT.json"
    ):
        raise ValueError("contract TL3 reference differs from the fixed report baseline")
    data = admission.get("data")
    if not isinstance(data, dict) or Path(str(data.get("data_root"))).resolve(
        strict=True
    ) != args.data_root.resolve(strict=True):
        raise ValueError("data root differs from the frozen Stage4 admission")
    primary_path = regular_campaign_file(args.primary_split_manifest, "primary split manifest")
    auxiliary_path = regular_campaign_file(
        args.auxiliary_split_manifest, "auxiliary split manifest"
    )
    portable_path = regular_campaign_file(args.portable_data_manifest, "portable data registry")
    os.environ["WNN_PORTABLE_DATA_MANIFEST"] = str(portable_path)
    primary, auxiliary = (
        load_campaign_split(primary_path, args.data_root),
        load_campaign_split(auxiliary_path, args.data_root),
    )
    auxiliary_ref = data.get("auxiliary_manifest")
    if (
        primary.sha256 != PRIMARY_SHA256
        or len(primary.train_groups) != 10_325
        or primary.valid_groups
        or len(primary.evaluation_groups) != 1_873
        or auxiliary.sha256 != AUXILIARY_SHA256
        or len(auxiliary.train_groups) != 8_260
        or len(auxiliary.evaluation_groups) != 2_065
        or auxiliary.primary_split_sha256 != primary.sha256
        or data.get("primary_split_sha256") != primary.sha256
        or not isinstance(auxiliary_ref, dict)
        or auxiliary_ref.get("sha256") != auxiliary.sha256
    ):
        raise ValueError("primary or auxiliary split differs from the frozen 10325/1873 protocol")
    frozen_aux_path, _ = read_bound_campaign_reference(
        run.get("auxiliary_manifest"), "auxiliary split"
    )
    if frozen_aux_path != auxiliary_path:
        raise ValueError("auxiliary split path differs from the frozen campaign reference")
    from report_multistage_primary_terminal import qualify_primary_fit_terminal

    terminal_qualification = qualify_primary_fit_terminal(run_dir)
    source6 = _source6_info(args.source6_root)
    if (
        run.get("evaluation_source_sha256") != SOURCE6_SHA256
        or run.get("evaluation_source_files_sha256") != SOURCE6_SHA256
        or run.get("evaluation_score_script_sha256") != SCORE_SCRIPT_SHA256
    ):
        raise ValueError("campaign evaluation source pins differ from Source6")
    runner_path = regular_campaign_file(
        stage4_path.parent / "runner_request.json", "Stage4 runner request"
    )
    runner = _read_json(runner_path, "Stage4 runner request")
    stage4 = _read_json(stage4_path, "Stage4 result")
    if (
        stage4.get("schema") != "wnn.multistage-stage4-ensemble.v1"
        or stage4.get("status") != "complete"
        or stage4.get("scope") != "inner_train_proxy"
        or stage4.get("primary_split_sha256") != primary.sha256
        or stage4.get("auxiliary_split_sha256") != auxiliary.sha256
        or stage4.get("evaluation_source") != "train"
        or runner.get("run_manifest_sha256") != file_sha256(run_path)
        or runner.get("admission_sha256") != run["admission"].get("sha256")
        or runner.get("contract_sha256") != run["contract"].get("sha256")
        or runner.get("portable_data_manifest_sha256") != file_sha256(portable_path)
        or runner.get("primary_split_sha256") != primary.sha256
        or runner.get("auxiliary_split_sha256") != auxiliary.sha256
        or runner.get("used_primary_valid") is not False
    ):
        raise ValueError("Stage4 result/request is not bound to the frozen train-only campaign")
    stage4_reference = stage4.get("fixed_primary_tl3_reference")
    if (
        not isinstance(stage4_reference, dict)
        or stage4_reference.get("wp") != TL3_WP
        or stage4_reference.get("used_for_weight_search") is not False
        or stage4_reference.get("directly_comparable_to_inner_proxy") is not False
    ):
        raise ValueError("Stage4 did not preserve TL3 as an unscored reference-only baseline")
    refs = _candidate_refs(run_dir, runner)
    expected_fit = [["train", group] for group in auxiliary.train_groups]
    expected_control = [["train", group] for group in auxiliary.evaluation_groups]
    candidates = load_finalized_candidates(
        run_dir, refs, auxiliary.sha256, primary.sha256, auxiliary, expected_fit, expected_control
    )
    ensemble = stage4.get("ensemble")
    members = ensemble.get("members") if isinstance(ensemble, dict) else None
    if not isinstance(members, list) or len(members) != 8 or len(candidates) != 8:
        raise ValueError("Stage4 must contain eight fully qualified final model members")
    expected_control = [["train", group] for group in auxiliary.evaluation_groups]
    if (
        stage4.get("evaluation_members") != expected_control
        or stage4.get("evaluation_mask") != "all_after_warmup AND row_index>=99"
        or stage4.get("selected_rows") != len(expected_control) * (SEQUENCE_LENGTH - WARMUP)
    ):
        raise ValueError("Stage4 optimization scope differs from the frozen auxiliary control")
    weights = [item.get("weight") for item in members if isinstance(item, dict)]
    if (
        len(weights) != 8
        or any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
            for weight in weights
        )
        or abs(sum(float(weight) for weight in weights) - 1.0) > 1e-10
    ):
        raise ValueError("frozen Stage4 weights are not a nonnegative unit simplex")
    candidate_records = []
    for index, (spec, candidate, member) in enumerate(
        zip(runner["candidate_trial_results"], candidates, members, strict=True)
    ):
        result_path = Path(str(candidate["trial_result_path"]))
        spec_request_source = Path(str(spec["trial_request_path"]))
        if spec_request_source.is_symlink() or not spec_request_source.is_file():
            raise ValueError("runner candidate request must be a regular non-symlink file")
        spec_request_path = spec_request_source.resolve(strict=True)
        candidate_request_path = Path(str(candidate["trial_request_path"])).resolve(strict=True)
        if (
            spec.get("name") != candidate["name"]
            or spec.get("stage") != candidate["stage"]
            or spec.get("trial_number") != candidate["trial_number"]
            or spec.get("family") != candidate["family"]
            or spec.get("config_sha256") != candidate["config_sha256"]
            or str(result_path) != str(Path(str(spec["path"])).resolve(strict=True))
            or spec.get("sha256") != candidate["trial_result_sha256"]
            or spec_request_path != candidate_request_path
            or spec.get("trial_request_sha256") != candidate["trial_request_sha256"]
            or member.get("trial_result_path") != str(result_path)
            or member.get("trial_result_sha256") != candidate["trial_result_sha256"]
            or member.get("trial_request_path") != candidate["trial_request_path"]
            or member.get("trial_request_sha256") != candidate["trial_request_sha256"]
            or member.get("checkpoint_sha256") != candidate["checkpoint_sha256"]
            or member.get("config_sha256") != candidate["config_sha256"]
            or member.get("config_identity_sha256") != candidate["config_identity_sha256"]
            or member.get("producer_identity_sha256") != candidate["producer_identity_sha256"]
            or member.get("prediction_manifest_sha256") != candidate["prediction_manifest_sha256"]
            or member.get("objective") != candidate["objective"]
            or member.get("stage") != candidate["stage"]
            or member.get("family") != candidate["family"]
            or member.get("trial_number") != candidate["trial_number"]
            or member.get("weight") != weights[index]
        ):
            raise ValueError("Stage4 member differs from its validated descendant artifacts")
        trial_request = _read_json(candidate_request_path, "candidate trial request")
        training_source_sha = trial_request.get("source_sha256")
        if (
            not isinstance(training_source_sha, str)
            or len(training_source_sha) != 64
            or any(character not in "0123456789abcdef" for character in training_source_sha)
        ):
            raise ValueError("candidate trial request omits its pinned training-source SHA")
        candidate_records.append(
            {
                "index": index,
                "name": candidate["name"],
                "stage": candidate["stage"],
                "trial_number": candidate["trial_number"],
                "family": candidate["family"],
                "weight": float(weights[index]),
                "inner_wp": candidate["objective"]["value"],
                "trial_result_path": str(result_path),
                "trial_result_sha256": candidate["trial_result_sha256"],
                "trial_request_path": candidate["trial_request_path"],
                "trial_request_sha256": candidate["trial_request_sha256"],
                "inner_objective_path": str(candidate["inner_objective_path"]),
                "config_path": candidate["config_path"],
                "config_sha256": candidate["config_sha256"],
                "config_identity_sha256": candidate["config_identity_sha256"],
                "checkpoint_path": str(candidate["checkpoint_path"]),
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "prediction_manifest_sha256": candidate["prediction_manifest_sha256"],
                "inner_objective_sha256": candidate["inner_objective_sha256"],
                "producer_identity_sha256": candidate["producer_identity_sha256"],
                "objective": candidate["objective"],
                "training_source_sha256": training_source_sha,
                "training_source_revision": trial_request.get("source_revision"),
                "gpu_uuid": args.gpu_uuid[index],
                "predictions_path": str(output / "member_predictions" / f"member-{index:02d}.npy"),
                "receipt_path": str(output / "member_receipts" / f"member-{index:02d}.json"),
            }
        )
    weight_sha = canonical_json_sha256([float(value) for value in weights])
    validator_files = (
        "src/wnn/multistage_search_ensemble_finalized_inputs.py",
        "src/wnn/multistage_search_ensemble_descendants.py",
        "src/wnn/multistage_search_ensemble_inputs.py",
        "src/wnn/multistage_search_reviewed_successor_validation.py",
        "src/wnn/multistage_search_successor_parent.py",
        "src/wnn/training/sequence_cache_ensemble_provenance.py",
        "src/wnn/training/sequence_cache_weighted_pool_lineage.py",
    )
    request = {
        "schema": "wnn.multistage-primary-ensemble-score-request.v1",
        "status": "validated_ready_for_fixed_weight_primary_scoring",
        "terminal_qualification": terminal_qualification,
        "run_manifest": {"path": str(run_path), "sha256": file_sha256(run_path)},
        "admission": {"path": str(admission_path), "sha256": file_sha256(admission_path)},
        "contract": {"path": str(contract_path), "sha256": file_sha256(contract_path)},
        "stage4_result": {"path": str(stage4_path), "sha256": file_sha256(stage4_path)},
        "stage4_runner_request": {"path": str(runner_path), "sha256": file_sha256(runner_path)},
        "stage4_runner_request_sha256": file_sha256(runner_path),
        "stage4_weights": [float(value) for value in weights],
        "weight_vector_sha256": weight_sha,
        "validator_sources": {item: file_sha256(ROOT / item) for item in validator_files},
        "source6": source6,
        "data_root": str(args.data_root.resolve(strict=True)),
        "primary_split": {
            "path": str(primary_path),
            "sha256": primary.sha256,
            "fit_train_sequences": 10_325,
            "fit_valid_sequences": 0,
            "evaluation_sequences": 1_873,
            "evaluation_members": [["valid", group] for group in primary.evaluation_groups],
        },
        "auxiliary_split": {
            "path": str(auxiliary_path),
            "sha256": auxiliary.sha256,
            "fit_sequences": 8_260,
            "control_sequences": 2_065,
        },
        "portable_data_registry": {
            "path": str(portable_path),
            "sha256": file_sha256(portable_path),
        },
        "output_dir": str(output),
        "candidates": candidate_records,
        "metric": "Source6 canonical pooled clipped WP, t0/t1 and clipped-target weighted MSE",
        "evaluation_mask": "is_scored AND row_index>=99",
        "selected_rows": 3_528_273,
        "primary_valid_used_for_weight_selection": False,
        "interpretation": "Report-only evaluation of the already-frozen Stage4 weights on exposed official primary valid.",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "member_predictions").mkdir()
    (output / "member_receipts").mkdir()
    (output / "logs").mkdir()
    request_path = output / "score_request.json"
    _write_new(request_path, request)
    return request, request_path, file_sha256(request_path)


def _terminate_owned(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _run_workers(
    args: argparse.Namespace, request: dict[str, Any], request_path: Path, request_sha: str
) -> dict[str, object]:
    start = time.monotonic()
    deadline = start + OVERALL_TIMEOUT_SECONDS
    source6 = Path(str(request["source6"]["root"]))
    worker_script = SCRIPTS / "report_multistage_primary_ensemble_worker.py"
    env_base = os.environ.copy()
    env_base.update(
        {
            "WNN_PORTABLE_DATA_MANIFEST": str(request["portable_data_registry"]["path"]),
            "WNN_PROJECT_ROOT": str(request["data_root"]),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "CUBLAS_EMULATE_SINGLE_PRECISION": "0",
        }
    )
    env_base["PYTHONPATH"] = os.pathsep.join(
        (str(source6 / "src"), str(args.dependency_root.resolve(strict=True)))
    )
    launched: list[dict[str, Any]] = []
    try:
        for index, _ in enumerate(request["candidates"]):
            log_path = Path(str(request["output_dir"])) / "logs" / f"member-{index:02d}.log"
            log = log_path.open("xb")
            env = env_base | {"CUDA_VISIBLE_DEVICES": str(args.gpu_uuid[index])}
            command = [
                args.python,
                str(worker_script),
                "--mode",
                "member",
                "--request",
                str(request_path),
                "--request-sha256",
                request_sha,
                "--member-index",
                str(index),
                "--gpu-uuid",
                str(args.gpu_uuid[index]),
            ]
            try:
                process = subprocess.Popen(
                    command, cwd=source6, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            except BaseException:
                log.close()
                raise
            launched.append(
                {
                    "index": index,
                    "process": process,
                    "log": log,
                    "log_path": log_path,
                    "command": command,
                    "started": time.monotonic(),
                }
            )
    except BaseException:
        for item in launched:
            _terminate_owned(item["process"])
            item["log"].close()
        raise
    timed_out: set[int] = set()
    while any(item["process"].poll() is None for item in launched):
        now = time.monotonic()
        for item in launched:
            proc = item["process"]
            elapsed = now - item["started"]
            if proc.poll() is None and (elapsed >= MEMBER_TIMEOUT_SECONDS or now >= deadline):
                timed_out.add(item["index"])
                _terminate_owned(proc)
        if time.monotonic() >= deadline:
            break
        time.sleep(0.2)
    if time.monotonic() >= deadline:
        for item in launched:
            if item["process"].poll() is None:
                timed_out.add(item["index"])
                _terminate_owned(item["process"])
    workers = []
    for item in launched:
        item["log"].close()
        candidate = request["candidates"][item["index"]]
        receipt_path = Path(str(candidate["receipt_path"]))
        status = (
            "complete"
            if item["process"].returncode == 0 and receipt_path.is_file()
            else "timeout"
            if item["index"] in timed_out
            else "failed"
        )
        row = {
            "member_index": item["index"],
            "status": status,
            "exit_code": item["process"].returncode,
            "elapsed_seconds": round(time.monotonic() - item["started"], 3),
            "gpu_uuid": candidate["gpu_uuid"],
            "log_path": str(item["log_path"]),
        }
        if status == "complete":
            row["receipt_path"] = str(receipt_path.resolve())
            row["receipt_sha256"] = _sha256(receipt_path)
            receipt = _read_json(receipt_path, "member score receipt")
            row["prediction_path"] = receipt["predictions"]["path"]
            row["prediction_sha256"] = receipt["predictions"]["sha256"]
        workers.append(row)
    if any(row["status"] != "complete" for row in workers):
        failure = {
            "schema": "wnn.multistage-primary-ensemble-failure.v1",
            "status": "partial_member_failure",
            "workers": workers,
            "deadline_seconds": OVERALL_TIMEOUT_SECONDS,
        }
        _write_new(Path(str(request["output_dir"])) / "score_failure.json", failure)
        raise RuntimeError(
            "one or more Source6 member score workers failed; no ensemble result was produced"
        )
    members = [
        {
            "receipt_path": row["receipt_path"],
            "receipt_sha256": row["receipt_sha256"],
            "prediction_sha256": row["prediction_sha256"],
        }
        for row in workers
    ]
    blend_request = {
        "schema": "wnn.multistage-primary-ensemble-blend-request.v1",
        "score_request_path": str(request_path),
        "score_request_sha256": request_sha,
        "members": members,
        "output_dir": request["output_dir"],
        "ensemble_predictions_path": str(
            Path(str(request["output_dir"])) / "ensemble_scored_predictions.npy"
        ),
    }
    blend_path = Path(str(request["output_dir"])) / "blend_request.json"
    _write_new(blend_path, blend_request)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("eight member scorers exhausted the outer report deadline")
    blend_log = Path(str(request["output_dir"])) / "logs/blend.log"
    blend_env = env_base | {"CUDA_VISIBLE_DEVICES": ""}
    blend_script = SCRIPTS / "report_multistage_primary_ensemble_blend.py"
    command = [
        args.python,
        str(blend_script),
        "--blend-request",
        str(blend_path),
        "--blend-request-sha256",
        _sha256(blend_path),
    ]
    with blend_log.open("xb") as log:
        try:
            completed = subprocess.run(
                command,
                cwd=source6,
                env=blend_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=remaining,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            failure = {
                "schema": "wnn.multistage-primary-ensemble-failure.v1",
                "status": "blend_timeout",
                "workers": workers,
                "deadline_seconds": OVERALL_TIMEOUT_SECONDS,
            }
            _write_new(Path(str(request["output_dir"])) / "score_failure.json", failure)
            raise TimeoutError("Source6 CPU blend exceeded the outer report deadline") from error
    report_path = Path(str(request["output_dir"])) / "ensemble_primary_report.json"
    if completed.returncode != 0 or not report_path.is_file():
        failure = {
            "schema": "wnn.multistage-primary-ensemble-failure.v1",
            "status": "blend_failed",
            "blend_exit_code": completed.returncode,
            "workers": workers,
            "log_path": str(blend_log),
        }
        _write_new(Path(str(request["output_dir"])) / "score_failure.json", failure)
        raise RuntimeError("Source6 blend failed; no complete ensemble report was produced")
    receipt = {
        "schema": "wnn.multistage-primary-ensemble-execution.v1",
        "status": "complete",
        "score_request_path": str(request_path),
        "score_request_sha256": request_sha,
        "workers": workers,
        "blend_command": command,
        "blend_exit_code": completed.returncode,
        "blend_log_path": str(blend_log),
        "report_path": str(report_path.resolve()),
        "report_sha256": _sha256(report_path),
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "member_timeout_seconds": MEMBER_TIMEOUT_SECONDS,
        "outer_timeout_seconds": OVERALL_TIMEOUT_SECONDS,
    }
    _write_new(Path(str(request["output_dir"])) / "execution_receipt.json", receipt)
    return _read_json(report_path, "primary ensemble report") | {"execution_receipt": receipt}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage4-result", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--primary-split-manifest", type=Path, required=True)
    parser.add_argument("--auxiliary-split-manifest", type=Path, required=True)
    parser.add_argument("--portable-data-manifest", type=Path, required=True)
    parser.add_argument("--source6-root", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", action="append", required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if (
        len(args.gpu_uuid) != 8
        or len(set(args.gpu_uuid)) != 8
        or any(not value.startswith("GPU-") for value in args.gpu_uuid)
    ):
        raise SystemExit("provide exactly eight distinct B300 GPU UUIDs")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        request, request_path, request_sha = prepare_request(args)
        result = _run_workers(args, request, request_path, request_sha)
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        TimeoutError,
    ) as error:
        raise SystemExit(f"Primary ensemble report refused or incomplete: {error}") from error
    sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
