"""Durably queue the eight fixed E464 fits, then their exports and paired score."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from run_seed4_sequence import guard, properties, read_json, require_success

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
STAGING = CAMPAIGN / "staging/e464"
RUNNER = CAMPAIGN / "runner"
OUTPUT = CAMPAIGN / "checks/e464_sequence_v1.json"
PREDECESSOR = "wnn-e425-sequence-v1.service"
PREDECESSOR_SEQUENCE = CAMPAIGN / "checks/e425_sequence_v1.json"
FIT_GROUP_COUNT = 1505
FIT_BATCH = 16
ROWS_PER_SEQUENCE = 20_000
UPDATE_ROWS = 2_000
EXPECTED_BATCHES = FIT_GROUP_COUNT // FIT_BATCH
EXPECTED_UPDATES = EXPECTED_BATCHES * (ROWS_PER_SEQUENCE // UPDATE_ROWS)
MAX_E464_GPU_SECONDS = 3 * 60 * 60
OBSERVATION_GRACE_SECONDS = 300

FITS = (
    ("h093_learned", "e464-h093-learned-head-s3-final-v1"),
    ("h093_hash", "e464-h093-hash-head-s3-final-v1"),
    ("h093_universal", "e464-h093-universal-head-s3-final-v1"),
    ("h093_gru_budget", "e464-h093-gru-budget-head-s3-final-v1"),
    ("h094_conditional", "e464-h094-conditional-head-s3-final-v1"),
    ("h094_unconditional", "e464-h094-unconditional-head-s3-final-v1"),
    ("h094_diagonal", "e464-h094-diagonal-head-s3-final-v1"),
    ("h094_gru_budget", "e464-h094-gru-budget-head-s3-final-v1"),
)
EXPORTS = tuple(
    (arm, f"e464-predict-{arm.replace('_', '-')}-fp32-v1") for arm, _ in FITS
)
SCORE_ATTEMPT = "e464-score-all-parent-head-fp32-v1"
SCORE_DRAFT = STAGING / "specs/e464_score_all_parent_head_fp32_v1.draft.json"


def save(status: str, completed: list[str], **details: object) -> None:
    """Atomically snapshot controller progress; runner attempt files stay authoritative."""
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"status": status, "completed": completed, **details}, indent=2,
                    sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(OUTPUT)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def e464_guard() -> None:
    """Retain the campaign pause/authorization and arithmetic boundaries."""
    guard()
    if (RUNNER / "PAUSE.json").exists():
        raise RuntimeError("campaign runner PAUSE.json requires review")
    if (EXPECTED_BATCHES, EXPECTED_UPDATES) != (94, 940):
        raise RuntimeError("fixed E464 1,505-group recipe arithmetic changed")


def wait_for_e425(completed: list[str], was_seen_active: bool) -> dict[str, Any]:
    """Require the exact successful E425 service, terminal record, and three jobs."""
    required_attempts = (
        "e425-soft-smoke-2b-v1",
        "e425-soft-s3-final-v2",
        "e425-predict-c20261004-h106-soft-s3-final-fp32-v1",
    )
    seen_active = was_seen_active
    while True:
        e464_guard()
        observed = properties(PREDECESSOR)
        if observed.get("LoadState") == "not-found":
            raise RuntimeError(f"required E425 service is not installed: {PREDECESSOR}")
        if observed.get("ActiveState") in ("active", "activating", "deactivating"):
            seen_active = True
        if not PREDECESSOR_SEQUENCE.is_file():
            if seen_active and observed.get("ActiveState") not in (
                "active", "activating", "deactivating"
            ):
                raise RuntimeError("E425 service ended without its required sequence record")
            save("waiting_for_e425_start" if not seen_active else "waiting_for_e425",
                 completed, predecessor=PREDECESSOR, predecessor_seen_active=seen_active,
                 observed=observed)
            time.sleep(15)
            continue
        sequence = read_json_object(PREDECESSOR_SEQUENCE)
        status = sequence.get("status")
        if status == "fits_and_exports_complete":
            if observed.get("ActiveState") != "inactive":
                if observed.get("ActiveState") in ("active", "activating", "deactivating"):
                    save("waiting_for_e425_teardown", completed, predecessor=PREDECESSOR,
                         observed=observed)
                    time.sleep(5)
                    continue
                raise RuntimeError(f"E425 service is not terminal: {observed}")
            if observed.get("ExecMainStatus") != "0":
                raise RuntimeError(f"E425 service exited unsuccessfully: {observed}")
            for attempt in required_attempts:
                record = require_success(attempt)
                if record.get("id") != attempt:
                    raise ValueError(f"E425 attempt identity mismatch: {attempt}")
            return sequence
        if status not in ("waiting_for_e426", "waiting_for_e426_teardown", "running",
                          "waiting_for_e425", "waiting_for_seed4"):
            raise RuntimeError(f"E425 controller stopped before fits_and_exports_complete: {sequence}")
        if observed.get("ActiveState") not in ("active", "activating", "deactivating"):
            raise RuntimeError(f"E425 service handle disappeared before completion: {observed}")
        save("waiting_for_e425", completed, predecessor=PREDECESSOR,
             predecessor_status=status, predecessor_seen_active=seen_active, observed=observed)
        time.sleep(15)


def draft_path_for_fit(arm: str) -> Path:
    return STAGING / "execution_specs" / f"c20261004_e464_{arm}_head_s3_final.draft.json"


def sealed_path(draft: Path) -> Path:
    if not draft.name.endswith(".draft.json"):
        raise ValueError(f"expected .draft.json: {draft}")
    return draft.with_name(draft.name.removesuffix(".draft.json") + ".sealed.json")


def verify_sealed_matches_draft(draft_path: Path, sealed: dict[str, Any]) -> None:
    """Ensure root's sealed identity corresponds exactly to this registered draft."""
    draft = read_json_object(draft_path)
    required = {"version", "name", "kind", "argv", "code", "inputs", "artifacts",
                "limits", "environment"}
    if set(draft) != required:
        raise ValueError(f"unexpected draft schema: {draft_path}")
    for field in ("version", "name", "kind", "argv", "artifacts", "limits", "environment"):
        if sealed.get(field) != draft[field]:
            raise ValueError(f"root-sealed spec changed registered {field}: {draft_path}")
    for field in ("code", "inputs"):
        expected_paths = sorted(draft[field])
        actual_paths = sorted(item["path"] for item in sealed.get(field, []))
        if actual_paths != expected_paths:
            raise ValueError(f"root-sealed {field} paths differ from draft: {draft_path}")


def _flag(argv: list[str], name: str) -> str:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"missing scalar argument {name}") from error


def fit_audit_paths(arm: str) -> tuple[str, str, str]:
    child = f"c20261004_e464_{arm}_head_s3_final"
    return child, f"artifacts/runs/{child}/model_ep0.pt", \
        f"research/campaign_2026-10-04/staging/e464/audits/{child}.json"


def validate_fit_output(arm: str, sealed_fit: dict[str, Any]) -> dict[str, Any]:
    """Validate the fixed epoch-0 head audit and files before any export is sealed."""
    child, checkpoint_rel, audit_rel = fit_audit_paths(arm)
    argv = sealed_fit["argv"]
    parent = _flag(argv, "--parent")
    parent_sha = _flag(argv, "--expected-parent-sha256")
    parent_provenance_sha = _flag(argv, "--expected-parent-provenance-sha256")
    run_name = _flag(argv, "--run-name")
    if run_name != child or _flag(argv, "--arm") != arm:
        raise ValueError(f"sealed E464 fit identity differs for {arm}")
    checkpoint = ROOT / checkpoint_rel
    audit_path = ROOT / audit_rel
    provenance_path = ROOT / f"artifacts/runs/{child}/provenance.json"
    if not all(path.is_file() for path in (checkpoint, audit_path, provenance_path)):
        raise FileNotFoundError(f"{arm} fit outputs/audit are incomplete")
    audit = read_json_object(audit_path)
    checkpoint_sha = sha256_file(checkpoint)
    parent_run = read_json_object(ROOT / f"artifacts/runs/{parent}/provenance.json")
    if (
        audit.get("status") != "pass"
        or audit.get("arm") != arm
        or audit.get("parent_run") != parent
        or audit.get("child_run") != child
        or audit.get("parent_epoch") != 3
        or audit.get("parent_checkpoint_sha256") != parent_sha
        or audit.get("parent_provenance_sha256") != parent_provenance_sha
        or audit.get("parent_code_spec_sha256") != _flag(argv, "--expected-parent-code-spec-sha256")
        or audit.get("fixed_epoch") != 0
        or audit.get("child_checkpoint_sha256") != checkpoint_sha
        or audit.get("expected_completed_batches") != EXPECTED_BATCHES
        or audit.get("expected_optimizer_attempts") != EXPECTED_UPDATES
        or audit.get("non_head_state_mismatches") != []
        or audit.get("head_layout_mismatches") != []
        or audit.get("evaluation") != "withheld"
    ):
        raise ValueError(f"{arm} fixed epoch-0 audit failed identity/accounting checks")
    child_provenance = read_json_object(provenance_path)
    update_records = child_provenance.get("update_diagnostics")
    if not isinstance(update_records, list) or len(update_records) != 1:
        raise ValueError(f"{arm} provenance lacks its single fixed-epoch update record")
    counts = update_records[0].get("epoch", {})
    if (
        counts.get("completed_batches") != EXPECTED_BATCHES
        or counts.get("optimizer_attempts") != EXPECTED_UPDATES
        or counts.get("optimizer_completed_updates", 0) + counts.get("amp_skipped_updates", 0)
        != EXPECTED_UPDATES
    ):
        raise ValueError(f"{arm} optimizer counters do not reconcile to 940 attempts")
    if child_provenance.get("checkpoint_sha256", {}).get("model_ep0.pt") != checkpoint_sha:
        raise ValueError(f"{arm} child provenance does not bind model_ep0.pt")
    parent_checkpoint_rel = f"artifacts/runs/{parent}/model_ep3.pt"
    parent_checkpoint_sha_actual = sha256_file(ROOT / parent_checkpoint_rel)
    if parent_checkpoint_sha_actual != parent_sha:
        raise ValueError(f"{arm} fixed parent checkpoint changed")
    if parent_run.get("checkpoint_sha256", {}).get("model_ep3.pt") != parent_sha:
        raise ValueError(f"{arm} fixed parent provenance no longer binds model_ep3.pt")
    return {
        "arm": arm,
        "parent": parent,
        "child": child,
        "parent_checkpoint_sha256": parent_sha,
        "child_checkpoint_sha256": checkpoint_sha,
        "audit_sha256": sha256_file(audit_path),
        "audit_path": audit_rel,
        "checkpoint_path": checkpoint_rel,
        "completed_batches": EXPECTED_BATCHES,
        "optimizer_attempts": EXPECTED_UPDATES,
    }


def validate_all_fit_outputs(sealed_fits: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate every registered fit before the queue seals or launches any export."""
    if set(sealed_fits) != {arm for arm, _ in FITS}:
        raise ValueError("the fixed eight-arm fit set is incomplete")
    preflight = read_json_object(CAMPAIGN / "checks/e464_scope_preflight_v1/result.json")
    if preflight.get("status") != "pass" or len(preflight.get("arms", [])) != 8:
        raise ValueError("bounded eight-parent E464 preflight evidence is missing or failed")
    return [validate_fit_output(arm, sealed_fits[arm]) for arm, _ in FITS]


def seal_after_fit_gate(draft: Path, completed: list[str]) -> tuple[Path, dict[str, Any]]:
    """Seal an export/score only after caller validated every preceding artifact."""
    sealed_path_value = sealed_path(draft)
    if not draft.is_file():
        raise FileNotFoundError(draft)
    command = [sys.executable, "scripts/run_campaign.py"]
    if not sealed_path_value.exists():
        e464_guard()
        subprocess.run([*command, "seal", str(draft), str(sealed_path_value)],
                       cwd=ROOT, check=True, timeout=120)
    sealed = read_json_object(sealed_path_value)
    verify_sealed_matches_draft(draft, sealed)
    for item in sealed["inputs"]:
        if not (ROOT / item["path"]).is_file():
            raise FileNotFoundError(f"sealed downstream input missing: {item['path']}")
    return sealed_path_value, sealed


def validate_export_spec(arm: str, sealed: dict[str, Any], fit_record: dict[str, Any]) -> None:
    argv = sealed.get("argv")
    child = fit_record["child"]
    if (
        not isinstance(argv, list)
        or sealed.get("kind") != "gpu"
        or argv[1] != "research/campaign_2026-10-04/staging/e464/scripts/predict_e464_head.py"
        or argv[2:5] != [arm, child, "valid"]
        or _flag(argv, "--epoch") != "0"
        or _flag(argv, "--inference-precision") != "fp32"
        or _flag(argv, "--output-name") != child
    ):
        raise ValueError(f"export spec does not describe {arm}'s fixed epoch-0 FP32 predictor")
    required_inputs = {
        fit_record["checkpoint_path"],
        f"artifacts/runs/{child}/provenance.json",
        fit_record["audit_path"],
        "research/campaign_2026-10-04/staging/e464/specs/e464_predictor_source_closure.draft.json",
    }
    if not required_inputs <= {x["path"] for x in sealed["inputs"]}:
        raise ValueError(f"{arm} export does not bind checkpoint, provenance, audit and source closure")


def validate_export_output(arm: str, fit_record: dict[str, Any]) -> dict[str, Any]:
    child = fit_record["child"]
    prediction = ROOT / f"cache/pred_{child}_valid.npy"
    sidecar_path = prediction.with_suffix(".json")
    if not prediction.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"{arm} FP32 prediction output is incomplete")
    sidecar = read_json_object(sidecar_path)
    if (
        sidecar.get("run") != child
        or sidecar.get("parent_run") != fit_record["parent"]
        or sidecar.get("arm") != arm
        or sidecar.get("split") != "valid"
        or sidecar.get("epoch") != 0
        or sidecar.get("fixed_checkpoint_policy") != "fixed"
        or sidecar.get("checkpoint_sha256") != fit_record["child_checkpoint_sha256"]
        or sidecar.get("head_audit_sha256") != fit_record["audit_sha256"]
        or sidecar.get("head_audit_status") != "pass"
        or sidecar.get("inference_precision") != "fp32"
        or sidecar.get("model_outputs") != 23
        or sidecar.get("outputs_exported") != 2
        or sidecar.get("prediction_sha256") != sha256_file(prediction)
        or len(sidecar.get("row_groups", [])) != 1873
        or len(set(sidecar.get("row_groups", []))) != 1873
    ):
        raise ValueError(f"{arm} prediction sidecar does not bind the audited fixed export")
    return {"arm": arm, "run": child, "prediction_sha256": sidecar["prediction_sha256"],
            "sidecar_sha256": sha256_file(sidecar_path)}


def launch_job(
    sealed_path_value: Path,
    attempt: str,
    completed: list[str],
    *,
    phase: str,
) -> dict[str, Any]:
    """Launch or observe only this exact identity; never relaunch a stopped attempt."""
    e464_guard()
    sealed = read_json_object(sealed_path_value)
    record_path = RUNNER / f"attempts/{attempt}.json"
    if record_path.exists():
        record = read_json_object(record_path)
        if record.get("id") != attempt or record.get("spec") != sealed:
            raise ValueError(f"existing {phase} attempt identity differs: {attempt}")
        if record.get("status") not in ("reserved", "running", "finished"):
            raise RuntimeError(f"refusing blind retry of stopped {phase} attempt {attempt}: {record}")
    else:
        if any((ROOT / item).exists() for item in sealed["artifacts"]):
            raise FileExistsError(f"{phase} artifacts exist without matching attempt: {attempt}")
        subprocess.run(
            [sys.executable, "scripts/run_campaign.py", "launch", str(sealed_path_value),
             "--attempt", attempt],
            cwd=ROOT,
            check=True,
            timeout=30,
            capture_output=True,
            text=True,
        )

    deadline = time.monotonic() + int(sealed["limits"]["timeout_seconds"]) + OBSERVATION_GRACE_SECONDS
    while True:
        e464_guard()
        record = read_json_object(record_path)
        if record.get("id") != attempt or record.get("spec") != sealed:
            raise ValueError(f"{phase} attempt record changed identity: {attempt}")
        if record.get("status") == "finished":
            success = require_success(attempt)
            if success.get("spec") != sealed:
                raise ValueError(f"{phase} terminal runner spec differs: {attempt}")
            if attempt not in completed:
                completed.append(attempt)
            save("running", completed, phase=phase, last_finished_attempt=attempt)
            return success
        if record.get("status") not in ("reserved", "running"):
            raise RuntimeError(f"{phase} attempt stopped: {attempt}: {record.get('status')}")
        observed = properties(str(record["unit"]))
        if observed.get("ActiveState") not in ("active", "activating"):
            raise RuntimeError(f"{phase} unit disappeared; no retry: {attempt}: {observed}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"observation bound elapsed for {attempt}; no retry was started")
        save("running", completed, phase=phase, active_attempt=attempt, observed=observed,
             charged_e464_seconds=e464_charged_seconds())
        time.sleep(5)


def e464_charged_seconds() -> float:
    """Sum only terminal/reserved/running E464 job charges for the local reserve gate."""
    attempts = [attempt for _, attempt in FITS] + [attempt for _, attempt in EXPORTS]
    total = 0.0
    for attempt in attempts:
        path = RUNNER / f"attempts/{attempt}.json"
        if not path.is_file():
            continue
        record = read_json_object(path)
        if record.get("status") in ("reserved", "running"):
            total += float(record.get("elapsed_seconds", 0.0))
        else:
            total += float(record.get("charged_seconds", 0.0))
    return total


def enforce_gpu_reserve() -> None:
    if e464_charged_seconds() >= MAX_E464_GPU_SECONDS:
        raise RuntimeError("E464 reached its preregistered three-GPU-hour reserve; stop for review")


def validate_score_spec(sealed: dict[str, Any]) -> None:
    argv = sealed.get("argv")
    if (
        not isinstance(argv, list)
        or sealed.get("kind") != "cpu"
        or argv[1] != "research/campaign_2026-10-04/staging/e464/scripts/score_e464_all.py"
        or _flag(argv, "--output") != "research/campaign_2026-10-04/staging/e464/results/e464_report_all_fp32_v1.json"
    ):
        raise ValueError("E464 paired-score spec changed its fixed output or scorer")
    screen_path = ROOT / "research/campaign_2026-10-04/staging/e464/specs/e464_screen_all_fp32_v1.draft.json"
    screen = read_json_object(screen_path)
    if screen.get("n_boot") != 2000 or len(screen.get("comparisons", [])) != 16:
        raise ValueError("E464 score must retain all 16 preregistered paired comparisons")


def run_sequence(completed: list[str], predecessor_seen_active: bool) -> None:
    e464_guard()
    wait_for_e425(completed, predecessor_seen_active)
    save("waiting_for_fit_specs", completed, predecessor=PREDECESSOR)

    # Fit specs are root-sealed before the queue starts; this controller never seals or edits them.
    sealed_fits: dict[str, dict[str, Any]] = {}
    for arm, attempt in FITS:
        draft = draft_path_for_fit(arm)
        sealed_path_value = sealed_path(draft)
        if not sealed_path_value.is_file():
            raise FileNotFoundError(f"root must seal all eight fit specs before admission: {sealed_path_value}")
        sealed = read_json_object(sealed_path_value)
        verify_sealed_matches_draft(draft, sealed)
        enforce_gpu_reserve()
        sealed_fits[arm] = sealed
        launch_job(sealed_path_value, attempt, completed, phase=f"fit:{arm}")
        enforce_gpu_reserve()

    # Validate all eight fixed epoch-0 audits together before sealing any export.
    fit_records = validate_all_fit_outputs(sealed_fits)
    fit_by_arm = {item["arm"]: item for item in fit_records}
    save("fits_and_audits_complete", completed, fits=fit_records,
         actual_gpu_seconds=e464_charged_seconds())

    export_records: list[dict[str, Any]] = []
    for arm, attempt in EXPORTS:
        validate_all_fit_outputs(sealed_fits)
        draft = STAGING / "specs" / f"e464_predict_{arm}_fp32_v1.draft.json"
        sealed_export_path, sealed_export = seal_after_fit_gate(draft, completed)
        validate_export_spec(arm, sealed_export, fit_by_arm[arm])
        enforce_gpu_reserve()
        launch_job(sealed_export_path, attempt, completed, phase=f"export:{arm}")
        export_records.append(validate_export_output(arm, fit_by_arm[arm]))
        enforce_gpu_reserve()

    # No paired score is sealed or launched until all eight exports pass sidecar checks.
    for arm, _ in EXPORTS:
        validate_export_output(arm, fit_by_arm[arm])
    score_sealed_path, score_sealed = seal_after_fit_gate(SCORE_DRAFT, completed)
    validate_score_spec(score_sealed)
    launch_job(score_sealed_path, SCORE_ATTEMPT, completed, phase="paired_score")
    result_path = ROOT / "research/campaign_2026-10-04/staging/e464/results/e464_report_all_fp32_v1.json"
    if not result_path.is_file():
        raise FileNotFoundError("paired E464 report is missing despite successful score attempt")
    save("complete", completed, fits=fit_records, exports=export_records,
         paired_score_path=result_path.relative_to(ROOT).as_posix(),
         paired_score_sha256=sha256_file(result_path),
         paired_comparisons=16, bootstrap_replicates=2000,
         actual_e464_gpu_seconds=e464_charged_seconds())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if OUTPUT.exists():
        previous = read_json_object(OUTPUT)
        if not args.resume:
            raise FileExistsError(f"E464 controller snapshot already exists: {OUTPUT}")
        if previous.get("status") in ("complete", "stopped"):
            raise RuntimeError(f"refusing blind resume of terminal E464 controller: {previous}")
        completed = list(previous.get("completed", []))
        predecessor_seen_active = bool(previous.get("predecessor_seen_active", False))
    else:
        if args.resume:
            raise FileNotFoundError("--resume requested without an E464 controller snapshot")
        completed = []
        predecessor_seen_active = False
    try:
        run_sequence(completed, predecessor_seen_active)
    except Exception as error:
        save("stopped", completed, reason=f"{type(error).__name__}: {error}",
             actual_e464_gpu_seconds=e464_charged_seconds())
        raise


if __name__ == "__main__":
    main()
