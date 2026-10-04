"""Fail-closed E427 smoke → fit → fixed export → component score queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
STAGING = CAMPAIGN / "staging/e427_pipeline_v1"
RUNNER = CAMPAIGN / "runner"
OUTPUT = CAMPAIGN / "checks/e427_sequence_v1.json"
E464_UNIT = "wnn-e464-sequence-v1.service"
E464_SEQUENCE = CAMPAIGN / "checks/e464_sequence_v1.json"
MAX_GPU_SECONDS = 24 * 60 * 60
FIT_TIMEOUT = 3600
OBSERVATION_GRACE = 300
ARMS = ("aligned", "stale", "noop")
SMOKE_ATTEMPTS = {arm: f"e427-smoke-{arm}-s3-b32x2-v1" for arm in ARMS}
FIT_ATTEMPTS = {arm: f"e427-{arm}-s3-1ep-v2" for arm in ARMS}
EXPORT_ATTEMPTS = {arm: f"e427-predict-{arm}-fp32-v2" for arm in ARMS}
SCORE_ATTEMPT = "e427-score-component-panels-v1"
FIT_RUNS = {arm: f"c20261004_e427_{arm}_s3_1ep_v2" for arm in ARMS}
EXPECTED_GROUPS = list(range(1873))
EXPECTED_E464_FITS = (
    "e464-h093-learned-head-s3-final-v1", "e464-h093-hash-head-s3-final-v1",
    "e464-h093-universal-head-s3-final-v1", "e464-h093-gru-budget-head-s3-final-v1",
    "e464-h094-conditional-head-s3-final-v1", "e464-h094-unconditional-head-s3-final-v1",
    "e464-h094-diagonal-head-s3-final-v1", "e464-h094-gru-budget-head-s3-final-v1",
)
EXPECTED_E464_EXPORTS = tuple(
    f"e464-predict-{arm.replace('_', '-')}-fp32-v1"
    for arm in ("h093_learned", "h093_hash", "h093_universal", "h093_gru_budget",
                "h094_conditional", "h094_unconditional", "h094_diagonal", "h094_gru_budget")
)
EXPECTED_E464_SCORE = "e464-score-all-parent-head-fp32-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def guard() -> None:
    if (RUNNER / "PAUSE.json").exists():
        raise RuntimeError("campaign runner PAUSE.json requires review")
    if not (CAMPAIGN / "CONTINUATION_AUTHORIZATION_20261004.md").is_file():
        raise FileNotFoundError("continuation authorization record is missing")


def properties(unit: str) -> dict[str, str]:
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, "-p", "LoadState", "-p", "ActiveState",
         "-p", "SubState", "-p", "ExecMainStatus"],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=10,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def exact_artifacts(record: dict[str, Any], expected_paths: list[str]) -> None:
    entries = record.get("artifacts")
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise ValueError("runner success record has no authoritative artifact list")
    if {entry.get("path") for entry in entries} != set(expected_paths) or len(entries) != len(expected_paths):
        raise ValueError("runner artifacts do not exactly cover the declared outputs")
    for entry in entries:
        path = ROOT / entry["path"]
        if not path.is_file() or path.stat().st_size != entry.get("size"):
            raise ValueError(f"runner artifact path/size changed: {entry['path']}")
        if sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"runner artifact hash changed: {entry['path']}")


def require_success(attempt: str, *, kind: str | None = None) -> dict[str, Any]:
    path = RUNNER / "attempts" / f"{attempt}.json"
    record = read_object(path)
    if (record.get("id") != attempt or record.get("status") != "finished"
            or record.get("exit_code") != 0 or record.get("teardown_verified") is not True
            or record.get("failure") is not None
            or (kind is not None and record.get("kind") != kind)):
        raise RuntimeError(f"required attempt is not a successful terminal run: {attempt}")
    memory = record.get("final_memory_events", {})
    if any(type(memory.get(key, 0)) is not int or memory.get(key, 0) != 0
           for key in ("oom", "oom_kill", "oom_group_kill")):
        raise RuntimeError(f"required attempt reports a final OOM event: {attempt}")
    return record


def prediction_domain_is_exact(groups: object) -> bool:
    return (isinstance(groups, list) and len(groups) == 1873
            and all(type(group) is int for group in groups)
            and groups == EXPECTED_GROUPS)


def verify_recorded_artifacts(record: dict[str, Any], spec: dict[str, Any]) -> None:
    exact_artifacts(record, list(spec.get("artifacts", [])))


def validate_gpu_resources(record: dict[str, Any], spec: dict[str, Any]) -> None:
    limits = spec.get("limits", {})
    ram_peak = record.get("peak_ram_bytes")
    vram_peak = record.get("peak_observed_vram_bytes")
    if (record.get("kind") != "gpu" or type(ram_peak) is not int or type(vram_peak) is not int
            or ram_peak > limits.get("ram_bytes", 0)
            or vram_peak > limits.get("gpu_stop_bytes", 0)):
        raise ValueError("GPU attempt lacks truthful peak-memory evidence or exceeded its sealed limits")


def charged_gpu_seconds() -> float:
    names = list(SMOKE_ATTEMPTS.values()) + list(FIT_ATTEMPTS.values()) + list(EXPORT_ATTEMPTS.values())
    names.append(SCORE_ATTEMPT)
    total = 0.0
    for attempt in names:
        path = RUNNER / "attempts" / f"{attempt}.json"
        if path.is_file():
            record = read_object(path)
            if record.get("kind") == "gpu" and record.get("status") == "finished":
                total += float(record.get("charged_seconds", 0.0))
    return total


def reserve_gpu(attempt: str, spec: dict[str, Any], remaining_timeouts: int = 0) -> None:
    if spec.get("kind") != "gpu":
        return
    timeout = spec.get("limits", {}).get("timeout_seconds")
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("GPU job has no positive immutable timeout")
    for other in [*SMOKE_ATTEMPTS.values(), *FIT_ATTEMPTS.values(), *EXPORT_ATTEMPTS.values()]:
        if other == attempt:
            continue
        path = RUNNER / "attempts" / f"{other}.json"
        if path.is_file() and read_object(path).get("status") in ("reserved", "running"):
            raise RuntimeError(f"another E427 GPU attempt remains active: {other}")
    if charged_gpu_seconds() + timeout + remaining_timeouts > MAX_GPU_SECONDS:
        raise RuntimeError("charged E427 GPU time plus immutable future timeout reservations exceeds 24h")


def sealed_path(draft: Path) -> Path:
    return draft.with_name(draft.name.removesuffix(".draft.json") + ".sealed.json")


def validate_path_only(draft: dict[str, Any]) -> None:
    required = {"version", "name", "kind", "argv", "code", "inputs", "artifacts", "limits", "environment"}
    if set(draft) != required:
        raise ValueError("runtime execution spec is not canonical runner schema")
    for key in ("code", "inputs"):
        values = draft[key]
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError(f"execution spec {key} must contain root-relative paths")
        for value in values:
            path = Path(value)
            if not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
                raise ValueError(f"non-normalized {key} path: {value}")
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate execution spec {key} path")


def verify_sealed_matches_draft(draft_path: Path, sealed: dict[str, Any]) -> None:
    draft = read_object(draft_path)
    validate_path_only(draft)
    for key in ("version", "name", "kind", "argv", "artifacts", "limits", "environment"):
        if sealed.get(key) != draft.get(key):
            raise ValueError(f"sealed execution identity changed field {key}: {draft_path}")
    for key in ("code", "inputs"):
        if sorted(entry.get("path") for entry in sealed.get(key, [])) != sorted(draft[key]):
            raise ValueError(f"sealed spec {key} path closure differs from draft: {draft_path}")


def fit_spec(arm: str) -> Path:
    return STAGING / "execution_specs" / f"c20261004_e427_{arm}_s3_1ep_v2.draft.json"


def smoke_spec(arm: str) -> Path:
    return CAMPAIGN / f"staging/e427_smoke_review/specs/e427_smoke_{arm}_s3_b32x2_v1.draft.json"


def require_sealed_registered(draft: Path) -> tuple[Path, dict[str, Any]]:
    path = sealed_path(draft)
    if not path.is_file():
        raise FileNotFoundError(f"root must seal registered spec before queue admission: {path}")
    sealed = read_object(path)
    verify_sealed_matches_draft(draft, sealed)
    return path, sealed


def seal_after_gate(draft: Path) -> tuple[Path, dict[str, Any]]:
    """Seal a pinned downstream draft only after its future input files exist."""
    if not draft.is_file():
        raise FileNotFoundError(draft)
    validate_path_only(read_object(draft))
    path = sealed_path(draft)
    if not path.exists():
        subprocess.run([sys.executable, "scripts/run_campaign.py", "seal", str(draft), str(path)],
                       cwd=ROOT, check=True, timeout=120)
    sealed = read_object(path)
    verify_sealed_matches_draft(draft, sealed)
    for item in sealed.get("inputs", []):
        if not (ROOT / item["path"]).is_file():
            raise FileNotFoundError(f"downstream gate input missing: {item['path']}")
    return path, sealed


def launch_job(spec_path: Path, attempt: str, completed: list[str], phase: str,
               remaining_timeouts: int = 0) -> dict[str, Any]:
    guard()
    spec = read_object(spec_path)
    record_path = RUNNER / "attempts" / f"{attempt}.json"
    if record_path.exists():
        record = read_object(record_path)
        if record.get("id") != attempt or record.get("spec") != spec:
            raise ValueError(f"existing E427 attempt identity differs: {attempt}")
        if record.get("status") not in ("reserved", "running", "finished"):
            raise RuntimeError(f"no blind retry of stopped E427 attempt {attempt}")
        if record.get("status") == "finished":
            success = require_success(attempt, kind=spec["kind"])
            verify_recorded_artifacts(success, spec)
            if spec.get("kind") == "gpu":
                validate_gpu_resources(success, spec)
            if attempt not in completed:
                completed.append(attempt)
            return success
        reserve_gpu(attempt, spec, remaining_timeouts)
    else:
        if any((ROOT / item).exists() for item in spec["artifacts"]):
            raise FileExistsError(f"outputs exist without the exact runner record: {attempt}")
        reserve_gpu(attempt, spec, remaining_timeouts)
        subprocess.run([sys.executable, "scripts/run_campaign.py", "launch", str(spec_path),
                        "--attempt", attempt], cwd=ROOT, check=True, timeout=30,
                        capture_output=True, text=True)
    deadline = time.monotonic() + int(spec["limits"]["timeout_seconds"]) + OBSERVATION_GRACE
    while True:
        guard()
        record = read_object(record_path)
        if record.get("id") != attempt or record.get("spec") != spec:
            raise ValueError(f"attempt identity changed during {phase}: {attempt}")
        if record.get("status") == "finished":
            success = require_success(attempt, kind=spec["kind"])
            verify_recorded_artifacts(success, spec)
            if spec.get("kind") == "gpu":
                validate_gpu_resources(success, spec)
            if attempt not in completed:
                completed.append(attempt)
            save("running", completed, phase=phase, last_finished_attempt=attempt,
                 charged_gpu_seconds=charged_gpu_seconds())
            return success
        if record.get("status") not in ("reserved", "running"):
            raise RuntimeError(f"E427 {phase} stopped; no retry: {attempt}: {record.get('status')}")
        unit = record.get("unit")
        if not isinstance(unit, str) or not unit:
            raise ValueError(f"E427 runner record lacks exact worker unit: {attempt}")
        observed = properties(unit)
        if observed.get("LoadState") != "loaded" or observed.get("ActiveState") not in ("active", "activating"):
            raise RuntimeError(f"E427 worker handle disappeared; no retry: {attempt}: {observed}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"observation deadline elapsed for {attempt}; no retry")
        save("running", completed, phase=phase, active_attempt=attempt, observed=observed,
             charged_gpu_seconds=charged_gpu_seconds())
        time.sleep(5)


def save(status: str, completed: list[str], **details: object) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix(".json.tmp")
    temp.write_text(json.dumps({"schema": "wnn.e427-sequence-v1", "status": status,
                                "completed": completed, **details}, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(OUTPUT)


def validate_e464_payload(sequence: dict[str, Any], report: dict[str, Any],
                          report_sha256: str) -> None:
    if sequence.get("status") != "complete":
        raise ValueError(f"E464 has not completed successfully: {sequence.get('status')}")
    if sequence.get("completed") != [*EXPECTED_E464_FITS, *EXPECTED_E464_EXPORTS, EXPECTED_E464_SCORE]:
        raise ValueError("E464 completion sequence differs from the fixed eight-fit/export/score order")
    if sequence.get("paired_comparisons") != 16 or sequence.get("bootstrap_replicates") != 2000:
        raise ValueError("E464 paired-screen count/bootstrap differs from its fixed design")
    expected_report = "research/campaign_2026-10-04/staging/e464/results/e464_report_all_fp32_v1.json"
    if sequence.get("paired_score_path") != expected_report:
        raise ValueError("E464 completion names an unexpected score report")
    if sequence.get("paired_score_sha256") != report_sha256:
        raise ValueError("E464 completion does not bind its current paired-score artifact")
    if report.get("metric_id") != "global_wp_clip2_v1" or report.get("n_sequences") != 368:
        raise ValueError("E464 paired screen metric or declared historical-panel count changed")
    screen = report.get("screen", {})
    if screen.get("folds") != ["holdout"] or len(screen.get("comparisons", [])) != 16:
        raise ValueError("E464 report does not contain the exact holdout panel and sixteen contrasts")


def validate_e464_completion(sequence: dict[str, Any], report: dict[str, Any]) -> None:
    expected_report = "research/campaign_2026-10-04/staging/e464/results/e464_report_all_fp32_v1.json"
    if sequence.get("paired_score_path") != expected_report:
        raise ValueError("E464 completion names an unexpected score report")
    report_path = ROOT / expected_report
    validate_e464_payload(sequence, report, sha256_file(report_path))
    attempts = [*EXPECTED_E464_FITS, *EXPECTED_E464_EXPORTS, EXPECTED_E464_SCORE]
    for index, attempt in enumerate(attempts):
        record = require_success(attempt, kind="cpu" if attempt == EXPECTED_E464_SCORE else "gpu")
        if not isinstance(record.get("spec"), dict):
            raise ValueError(f"E464 attempt lacks its authoritative sealed spec: {attempt}")
        exact_artifacts(record, list(record["spec"].get("artifacts", [])))
        if record.get("kind") not in ("gpu", "cpu"):
            raise ValueError(f"E464 required attempt has invalid kind: {attempt}")


def validate_smoke_payload(value: dict[str, Any], arm: str) -> dict[str, Any]:
    marker = value.get("training_result")
    memory_fields = ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
                     "peak_host_rss_bytes", "host_rss_limit_bytes", "allocator_limit_bytes",
                     "stop_limit_bytes")
    if (value.get("schema") != "wnn.e427-production-smoke-record-v1"
            or value.get("arm") != arm or value.get("status") != "pass"
            or value.get("exit_code") != 0 or not isinstance(marker, dict)
            or marker.get("status") != "pass" or marker.get("arm") != arm
            or marker.get("completed_loader_batches") != 2
            or marker.get("optimizer_attempts") != 40
            or marker.get("optimizer_completed_updates", 0) + marker.get("amp_skipped_updates", 0) != 40
            or marker.get("ema_updates") != 40
            or marker.get("student_projection_audits") != 40
            or marker.get("ema_projection_audits") != 40
            or marker.get("loader_batch_shape") != [32, 20000, 211]
            or marker.get("initial_parent_cuda_parity", {}).get("parent_parity_pass") is not True
            or marker.get("initial_parent_cuda_parity", {}).get("causal_prefix_future_suffix_pass") is not True
            or any(type(marker.get(field)) is not int for field in memory_fields)
            or marker.get("allocator_limit_bytes") != 4200000000
            or marker.get("stop_limit_bytes") != 4800000000
            or marker.get("host_rss_limit_bytes") != 24 * (1 << 30)
            or marker.get("peak_cuda_allocated_bytes", 0) > 4200000000
            or marker.get("peak_cuda_reserved_bytes", 0) > 4800000000
            or marker.get("peak_host_rss_bytes", 0) > 24 * (1 << 30)
            or marker.get("failures") != []
            or value.get("model_checkpoint_written") is not False
            or value.get("scientific_fit") is not False
            or value.get("no_holdout_evaluation") is not True):
        raise ValueError(f"E427 {arm} full-shape smoke did not pass every registered gate")
    return marker


def wait_for_e464(completed: list[str], seen_active: bool) -> dict[str, Any]:
    while True:
        guard()
        observed = properties(E464_UNIT)
        if observed.get("LoadState") != "loaded":
            raise RuntimeError(f"required E464 service is missing: {observed}")
        if observed.get("ActiveState") in ("active", "activating", "deactivating"):
            seen_active = True
        if not E464_SEQUENCE.is_file():
            if seen_active and observed.get("ActiveState") not in ("active", "activating", "deactivating"):
                raise RuntimeError("E464 controller exited without its sequence report")
            save("waiting_for_e464", completed, predecessor=E464_UNIT,
                 predecessor_seen_active=seen_active, observed=observed)
            time.sleep(15)
            continue
        sequence = read_object(E464_SEQUENCE)
        if sequence.get("status") != "complete":
            if sequence.get("status") in ("waiting_for_e425", "waiting_for_e425_start",
                                           "waiting_for_e425_teardown", "running",
                                           "waiting_for_fit_specs", "fits_and_audits_complete"):
                if observed.get("ActiveState") not in ("active", "activating", "deactivating"):
                    raise RuntimeError(f"E464 sequence is unfinished but service is terminal: {observed}")
                save("waiting_for_e464", completed, predecessor=E464_UNIT,
                     predecessor_status=sequence.get("status"), observed=observed)
                time.sleep(15)
                continue
            raise RuntimeError(f"E464 controller stopped before completion: {sequence}")
        if observed.get("ActiveState") in ("active", "activating", "deactivating"):
            save("waiting_for_e464_teardown", completed, observed=observed)
            time.sleep(5)
            continue
        if observed.get("ExecMainStatus") != "0":
            raise RuntimeError(f"E464 service did not terminate successfully: {observed}")
        report_path = ROOT / sequence["paired_score_path"]
        validate_e464_completion(sequence, read_object(report_path))
        return sequence


def validate_stats() -> dict[str, Any]:
    stats_path = CAMPAIGN / "checks/e427_context_stats_s3_v1.json"
    parent_path = ROOT / "artifacts/runs/c20261004_combo_s3_final_r1/provenance.json"
    parent_checkpoint = ROOT / "artifacts/runs/c20261004_combo_s3_final_r1/model_ep3.pt"
    stats = read_object(stats_path)
    parent = read_object(parent_path)
    stats_record = require_success("e427-context-stats-s3-v1", kind="cpu")
    exact_artifacts(stats_record, ["research/campaign_2026-10-04/checks/e427_context_stats_s3_v1.json"])
    if (stats_record["artifacts"][0]["sha256"] != sha256_file(stats_path)
            or stats.get("format") != "wnn.e427-volume28-fit-stats"
            or stats.get("scoring_rows_used") is not False
            or stats.get("row_count") != 242_220_000
            or stats.get("parent_checkpoint_sha256") != sha256_file(parent_checkpoint)
            or stats.get("parent_provenance_sha256") != sha256_file(parent_path)
            or stats.get("parent_checkpoint_path") != "artifacts/runs/c20261004_combo_s3_final_r1/model_ep3.pt"):
        raise ValueError("fit-only E427 context statistics do not match exact parent lineage")
    for split in ("train", "valid"):
        groups = list(map(int, stats.get("fit_row_groups", {}).get(split, [])))
        parent_groups = list(map(int, parent.get("fit_row_groups", {}).get(split, [])))
        if len(groups) != len(set(groups)) or set(groups) != set(parent_groups):
            raise ValueError(f"E427 statistics {split} membership differs from fixed parent fit")
    if (len(stats["fit_row_groups"]["train"]) != 10606
            or len(stats["fit_row_groups"]["valid"]) != 1505
            or len(stats.get("evaluation_row_groups_excluded", [])) != 368):
        raise ValueError("E427 parent fit/excluded membership cardinalities changed")
    for rel, expected in stats.get("input_sha256", {}).items():
        if sha256_file(ROOT / rel) != expected:
            raise ValueError(f"E427 fit-only input changed: {rel}")
    return {"stats_sha256": sha256_file(stats_path),
            "parent_checkpoint_sha256": sha256_file(parent_checkpoint),
            "parent_provenance_sha256": sha256_file(parent_path),
            "fit_train_groups": 10606, "fit_valid_groups": 1505,
            "stats_excluded_groups": 368, "rows_scanned": stats["row_count"]}


def validate_smoke(arm: str) -> dict[str, Any]:
    attempt = SMOKE_ATTEMPTS[arm]
    record = require_success(attempt, kind="gpu")
    artifact = f"research/campaign_2026-10-04/checks/e427_smoke_{arm}_s3_b32x2_v1.json"
    exact_artifacts(record, [artifact])
    value = read_object(ROOT / artifact)
    marker = validate_smoke_payload(value, arm)
    return {"arm": arm, "attempt": attempt, "record_sha256": sha256_file(RUNNER / f"attempts/{attempt}.json"),
            "artifact_sha256": sha256_file(ROOT / artifact), "training_result": marker}


def validate_fit(arm: str) -> dict[str, Any]:
    run = FIT_RUNS[arm]
    record = require_success(FIT_ATTEMPTS[arm], kind="gpu")
    prov_path = ROOT / "artifacts/runs" / run / "provenance.json"
    ckpt_path = ROOT / "artifacts/runs" / run / "model_ep0.pt"
    prov = read_object(prov_path)
    args = prov.get("args", {})
    expected_args = {
        "name": run, "arch": "e427-volume-gate", "e427_arm": arm,
        "epochs": 1, "checkpoint_policy": "fixed", "checkpoint_epoch": 0,
        "init": "c20261004_combo_s3_final_r1@ep3", "train_source": "train+valid",
        "batch": 32, "chunk": 1000, "hidden": 256, "layers": 2, "seed": 3,
        "optim": "muon", "muon_scope": "rnn", "ema_teacher": 1,
        "ema_decay": 0.999, "eval_ema": True, "precision": "fp16",
        "no_eval_during_fit": True, "aux_struct": 0.3, "aux_next": 0.3,
        "loss": "wmse", "propensity": "prop_v1", "prop_power": 2.0,
        "prop_eps": 0.03, "valid_true_mask": 0.03,
        "distill": "ab67_final_since_s3,ab67_final_since_s4",
        "distill_alpha": 0.5, "distill_boot_h": 8,
        "since_weight": 32.0, "since_floor": 0.25,
        "no_eval_during_fit": True,
    }
    for key, value in expected_args.items():
        if args.get(key) != value:
            raise ValueError(f"{arm} fit changed preregistered trainer arg {key}: {args.get(key)!r}")
    if (prov.get("selected_epoch") != 0 or prov.get("e427_arm") != arm
            or prov.get("e427_context_stats_sha256") != sha256_file(CAMPAIGN / "checks/e427_context_stats_s3_v1.json")
            or prov.get("initial_checkpoint_sha256") != sha256_file(ROOT / "artifacts/runs/c20261004_combo_s3_final_r1/model_ep3.pt")
            or prov.get("checkpoint_sha256", {}).get("model_ep0.pt") != sha256_file(ckpt_path)
            or len(prov.get("evaluation_row_groups", [])) != 368
            or set(map(int, prov.get("evaluation_row_groups", [])))
            != set(map(int, read_object(CAMPAIGN / "checks/e427_context_stats_s3_v1.json")
                          .get("evaluation_row_groups_excluded", [])))
            or prov.get("split_manifest_sha256") is not None):
        raise ValueError(f"{arm} fixed checkpoint/provenance lineage differs")
    fit = prov.get("fit_row_groups", {})
    parent_provenance = read_object(ROOT / "artifacts/runs/c20261004_combo_s3_final_r1/provenance.json")
    expected_fit = parent_provenance.get("fit_row_groups", {})
    if (len(fit.get("train", [])) != 10606 or len(fit.get("valid", [])) != 1505
            or set(map(int, fit["train"])) != set(map(int, expected_fit.get("train", [])))
            or set(map(int, fit["valid"])) != set(map(int, expected_fit.get("valid", [])))):
        raise ValueError(f"{arm} exact continuation fit membership changed")
    if prov.get("normalization_identity") != parent_provenance.get("normalization_identity"):
        raise ValueError(f"{arm} fit did not preserve exact parent normalization")
    epochs = prov.get("update_diagnostics", [])
    if len(epochs) != 1 or epochs[0].get("epoch_index") != 0:
        raise ValueError(f"{arm} fixed fit does not have one epoch-0 update record")
    counts = epochs[0].get("epoch", {})
    if (counts.get("completed_batches") != 378 or counts.get("optimizer_attempts") != 7560
            or counts.get("ema_updates") != 7560
            or counts.get("optimizer_completed_updates", -1) + counts.get("amp_skipped_updates", -1) != 7560):
        raise ValueError(f"{arm} fixed one-epoch optimizer/EMA counters do not reconcile")
    audit = prov.get("e427_constraint_projection_audits", {})
    if (audit.get("student_optimizer_attempts_audited_this_process") != 7560
            or audit.get("ema_updates_audited_this_process") != 7560
            or audit.get("structural_zero_coordinates") != 4096
            or audit.get("post_attempt_and_post_ema_projection") is not True):
        raise ValueError(f"{arm} constraint projection audits do not reconcile to 7,560 attempts")
    history = json.loads((ROOT / "artifacts/runs" / run / "history.json").read_text())
    if len(history) != 1 or history[0].get("evaluation") != "withheld":
        raise ValueError(f"{arm} fit used evaluation during training")
    expected_paths = [f"artifacts/runs/{run}/{name}" for name in
                      ("model_ep0.pt", "model.pt", "provenance.json", "history.json", "training_state.pt")]
    exact_artifacts(record, expected_paths)
    return {"arm": arm, "run": run, "checkpoint_sha256": sha256_file(ckpt_path),
            "provenance_sha256": sha256_file(prov_path),
            "normalization_identity": prov.get("normalization_identity"),
            "fit_membership": {"train": 10606, "valid": 1505},
            "completed_batches": 378, "optimizer_attempts": 7560,
            "runner_attempt_sha256": sha256_file(RUNNER / f"attempts/{FIT_ATTEMPTS[arm]}.json")}


def validate_export(arm: str) -> dict[str, Any]:
    run = FIT_RUNS[arm]
    path = ROOT / "cache" / f"pred_{run}_valid.npy"
    side_path = path.with_suffix(".json")
    side = read_object(side_path)
    if (side.get("schema") != "wnn.e427-fixed-epoch0-fp32-prediction-v1"
            or side.get("run") != run or side.get("arm") != arm or side.get("epoch") != 0
            or side.get("checkpoint_policy") != "fixed" or side.get("inference_precision") != "fp32"
            or side.get("fit_time_evaluation") != "withheld"
            or not prediction_domain_is_exact(side.get("row_groups"))
            or side.get("checkpoint_sha256") != sha256_file(ROOT / f"artifacts/runs/{run}/model_ep0.pt")
            or side.get("provenance_sha256") != sha256_file(ROOT / f"artifacts/runs/{run}/provenance.json")
            or side.get("parent_checkpoint_sha256") != sha256_file(ROOT / "artifacts/runs/c20261004_combo_s3_final_r1/model_ep3.pt")
            or side.get("parent_provenance_sha256") != sha256_file(ROOT / "artifacts/runs/c20261004_combo_s3_final_r1/provenance.json")
            or side.get("context_stats_sha256") != sha256_file(CAMPAIGN / "checks/e427_context_stats_s3_v1.json")
            or side.get("prediction_sha256") != sha256_file(path)):
        raise ValueError(f"{arm} prediction sidecar failed exact lineage/domain/hash validation")
    import numpy as np

    pred = np.load(path, mmap_mode="r", allow_pickle=False)
    if pred.shape != (1873, 20000, 2) or pred.dtype != np.float32:
        raise ValueError(f"{arm} prediction has wrong complete FP32 shape")
    return {"arm": arm, "prediction_sha256": side["prediction_sha256"],
            "sidecar_sha256": sha256_file(side_path)}


def run_sequence(completed: list[str]) -> None:
    guard()
    e464 = wait_for_e464(completed, bool(read_object(OUTPUT).get("predecessor_seen_active")) if OUTPUT.exists() else False)
    stats_identity = validate_stats()
    save("waiting_for_smoke_specs", completed, e464_score_sha256=e464["paired_score_sha256"],
         context_stats=stats_identity)

    smoke_sealed = {}
    fit_sealed = {}
    for export_index, arm in enumerate(ARMS):
        smoke_path, smoke_spec_value = require_sealed_registered(smoke_spec(arm))
        if smoke_spec_value["kind"] != "gpu" or smoke_spec_value["limits"] != {
                "cpus": 8, "ram_bytes": 24 * (1 << 30), "timeout_seconds": 900,
                "gpu_memory_bytes": 4200000000, "gpu_stop_bytes": 4800000000}:
            raise ValueError(f"{arm} smoke spec limits changed")
        smoke_sealed[arm] = smoke_path
        fit_path, fit_spec_value = require_sealed_registered(fit_spec(arm))
        if fit_spec_value["limits"].get("timeout_seconds") != FIT_TIMEOUT:
            raise ValueError(f"{arm} fit immutable full-run timeout is not 3,600 seconds")
        fit_sealed[arm] = fit_path

    smoke_records = []
    for arm in ARMS:
        launch_job(smoke_sealed[arm], SMOKE_ATTEMPTS[arm], completed, f"production_smoke:{arm}")
        smoke_records.append(validate_smoke(arm))
    # No fit is started until all three full-shape, causal, parent-parity smokes passed.
    for arm in ARMS:
        validate_smoke(arm)
    if charged_gpu_seconds() + 3 * FIT_TIMEOUT > MAX_GPU_SECONDS:
        raise RuntimeError("full-fit phase requires 3h of remaining immutable timeout reserve")
    save("smokes_complete", completed, smokes=smoke_records,
         charged_gpu_seconds=charged_gpu_seconds(), reserved_fit_timeout_seconds=3 * FIT_TIMEOUT)

    fits = []
    for index, arm in enumerate(ARMS):
        reserve_left = (len(ARMS) - index - 1) * FIT_TIMEOUT
        launch_job(fit_sealed[arm], FIT_ATTEMPTS[arm], completed, f"fit:{arm}", reserve_left)
        fits.append(validate_fit(arm))
    normalization = [fit["normalization_identity"] for fit in fits]
    if not normalization[0] or any(value != normalization[0] for value in normalization[1:]):
        raise ValueError("E427 arms do not share the identical inherited normalization")
    save("fits_complete", completed, smokes=smoke_records, fits=fits,
         charged_gpu_seconds=charged_gpu_seconds())

    exports = []
    for arm in ARMS:
        for prior_arm in ARMS:
            validate_fit(prior_arm)
        draft = STAGING / "execution_specs" / f"e427_predict_{arm}_fp32_v2.draft.json"
        spec_path, spec = seal_after_gate(draft)
        expected_timeout = 1800
        if spec.get("kind") != "gpu" or spec.get("limits", {}).get("timeout_seconds") != expected_timeout:
            raise ValueError(f"{arm} epoch-0 export changed its bounded resource policy")
        attempt = EXPORT_ATTEMPTS[arm]
        launch_job(spec_path, attempt, completed, f"fp32_export:{arm}",
                   (len(ARMS) - export_index - 1) * 1800)
        record = require_success(attempt, kind="gpu")
        verify_recorded_artifacts(record, spec)
        exports.append(validate_export(arm))
    for arm in ARMS:
        validate_export(arm)
    save("exports_complete", completed, fits=fits, exports=exports,
         charged_gpu_seconds=charged_gpu_seconds())

    score_draft = STAGING / "execution_specs/e427_score_component_panels_v1.draft.json"
    score_path, score_spec = seal_after_gate(score_draft)
    if (score_spec.get("kind") != "cpu" or score_spec.get("limits", {}).get("cpus") != 1
            or score_spec.get("limits", {}).get("ram_bytes") != 2 * (1 << 30)):
        raise ValueError("E427 panel screen must remain one-CPU and 2-GiB bounded")
    launch_job(score_path, SCORE_ATTEMPT, completed, "component_panel_score")
    report_path = ROOT / "research/campaign_2026-10-04/staging/e427_pipeline_v1/results/e427_report_v1.json"
    report = read_object(report_path)
    if (report.get("metric_id") != "global_wp_clip2_v1" or report.get("n_boot") != 2000
            or set(report.get("panels", {})) != {
                "fold0_primary_historically_exposed", "stats_excluded_368_historically_exposed",
                "fold0_nonfit_94_diagnostic"}):
        raise ValueError("E427 component report is missing a preregistered panel or metric")
    save("complete", completed, fits=fits, exports=exports,
         paired_score_path=report_path.relative_to(ROOT).as_posix(),
         paired_score_sha256=sha256_file(report_path), charged_gpu_seconds=charged_gpu_seconds(),
         qualification="historically exposed; no holdout/generalization claim")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if OUTPUT.exists():
        previous = read_object(OUTPUT)
        if not args.resume:
            raise FileExistsError(f"E427 controller snapshot already exists: {OUTPUT}")
        if previous.get("status") in ("complete", "stopped"):
            raise RuntimeError(f"refusing blind resume of terminal E427 controller: {previous}")
        completed = list(previous.get("completed", []))
    else:
        if args.resume:
            raise FileNotFoundError("--resume requested without an E427 sequence snapshot")
        completed = []
    try:
        run_sequence(completed)
    except Exception as error:
        save("stopped", completed, reason=f"{type(error).__name__}: {error}",
             charged_gpu_seconds=charged_gpu_seconds())
        raise


if __name__ == "__main__":
    main()
