"""Run the sealed E425 soft-route smoke and full fit after E426 completes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
import time
from pathlib import Path

from run_seed4_sequence import guard, properties, read_json, require_success

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
RUNNER = CAMPAIGN / "runner"
OUTPUT = CAMPAIGN / "checks/e425_sequence_v1.json"
PREDECESSOR = "wnn-e426-sequence-v1.service"
E426_ATTEMPTS = (
    "e426-smoke-shared-f0-s3-2b-v2",
    "e426-f0-constant-s3-4ep-v2",
    "e426-f0-aligned-s3-4ep-v2",
    "e426-f0-shuffled-s3-4ep-v2",
)
CONTROL_IDENTITY = CAMPAIGN / "checks/e425_control_identity_v1.json"
FULL_RUN = "c20261004_h106_soft_s3_final"
PREDICTION_RUN = f"{FULL_RUN}_fp32"
PREDICTION_ATTEMPT = "e425-predict-c20261004-h106-soft-s3-final-fp32-v1"
PREDICTION_NAME = "e425_predict_c20261004_h106_soft_s3_final_fp32_v1"
PREDICTION_DRAFT = CAMPAIGN / f"specs/{PREDICTION_NAME}.draft.json"
PREDICTION_SEALED = CAMPAIGN / f"specs/{PREDICTION_NAME}.sealed.json"
VALID_CONTROL_SIDECAR = ROOT / "cache/pred_c20261004_h093_learned_s3_final_valid.json"

ARMS = (
    (
        "e425_soft_smoke_2b_v1",
        "e425-soft-smoke-2b-v1",
        "c20261004_h106_soft_smoke_2b",
        1,
        2,
    ),
    (
        "e425_soft_s3_final_v2",
        "e425-soft-s3-final-v2",
        FULL_RUN,
        4,
        378,
    ),
)
E424_PREDICTION_TEMPLATE = CAMPAIGN / (
    "specs/e424_predict_c20261004_h093_learned_s3_final_v3.draft.json"
)


def save(status: str, completed: list[str], **details: object) -> None:
    """Atomically update this controller snapshot without replacing raw records."""
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"status": status, "completed": completed, **details}, indent=2) + "\n"
    )
    temporary.replace(OUTPUT)


def _sha256_file(path: Path) -> str:
    """Hash a file incrementally using only the standard library."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wait_for_e426(completed: list[str]) -> None:
    """Require exact E426 terminal evidence before starting any E425 attempt."""
    sequence_path = CAMPAIGN / "checks/e426_sequence_v1.json"
    while True:
        guard()
        observed = properties(PREDECESSOR)
        sequence = read_json(sequence_path)
        if sequence.get("status") == "fits_complete":
            if observed.get("ActiveState") != "inactive":
                if observed.get("ActiveState") in ("active", "activating", "deactivating"):
                    save("waiting_for_e426_teardown", completed, observed=observed)
                    time.sleep(5)
                    continue
                raise RuntimeError(f"E426 service did not reach inactive state: {observed}")
            if observed.get("ExecMainStatus") != "0":
                raise RuntimeError(f"E426 controller exited unsuccessfully: {observed}")
            for attempt in E426_ATTEMPTS:
                require_success(attempt)
            return
        if sequence.get("status") not in ("waiting_for_seed4", "running"):
            raise RuntimeError(f"E426 sequence stopped before fits_complete: {sequence}")
        if observed.get("ActiveState") not in ("active", "activating", "deactivating"):
            raise RuntimeError(f"E426 handle disappeared before fits_complete: {observed}")
        save("waiting_for_e426", completed, predecessor=PREDECESSOR, observed=observed)
        time.sleep(15)


def _flag_value(argv: list[str], flag: str) -> str | None:
    """Return a scalar option from a sealed argument vector."""
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def verify_training_spec(
    spec: dict, spec_name: str, model_name: str, epochs: int, expected_batches: int
) -> None:
    """Fail closed if a sealed smoke/full recipe differs from the E425 protocol."""
    argv = spec.get("argv")
    if not isinstance(argv, list) or spec.get("name") != spec_name:
        raise ValueError(f"Unexpected E425 sealed spec identity: {spec_name}")
    expected_flags = {
        "--name": model_name,
        "--train-source": "train+valid",
        "--loss": "wmse",
        "--arch": "specialist",
        "--specialist-route": "soft",
        "--hidden": "96",
        "--layers": "1",
        "--dropout": "0",
        "--expert-hidden": "32",
        "--num-experts": "4",
        "--prefix-rows": "99",
        "--hash-seed": "1729",
        "--seed": "3",
        "--batch": "32",
        "--chunk": "1000",
        "--epochs": str(epochs),
        "--lr": "0.001",
        "--wd": "0.01",
        "--checkpoint-policy": "fixed",
        "--checkpoint-epoch": str(epochs - 1),
        "--optim": "muon",
        "--muon-scope": "rnn",
        "--ema-teacher": "1",
        "--ema-decay": "0.999",
        "--since-weight": "32",
        "--since-floor": "0.25",
        "--distill": "ab67_final_since_s3,ab67_final_since_s4",
        "--distill-alpha": "0.5",
        "--distill-boot-h": "8",
        "--aux-struct": "0.3",
        "--aux-next": "0.3",
        "--propensity": "prop_v1",
    }
    for flag, expected in expected_flags.items():
        if _flag_value(argv, flag) != expected:
            raise ValueError(f"{spec_name}: {flag} must be {expected}")
    for flag in ("--full-valid", "--eval-ema", "--fast", "--save-training-state"):
        if flag not in argv:
            raise ValueError(f"{spec_name}: required combo flag is absent: {flag}")
    if "--eval-source" not in argv or _flag_value(argv, "--eval-source") != "holdout":
        raise ValueError(f"{spec_name}: evaluation source must remain holdout")
    if spec.get("limits", {}).get("gpu_memory_bytes") != 4_200_000_000:
        raise ValueError(f"{spec_name}: expected 4.2-GB allocator cap")
    if spec.get("limits", {}).get("gpu_stop_bytes") != 4_800_000_000:
        raise ValueError(f"{spec_name}: expected 4.8-GB GPU stop")
    if epochs == 1:
        if _flag_value(argv, "--max-batches") != "2" or "--no-eval-during-fit" not in argv:
            raise ValueError("E425 smoke must be one epoch, two batches, evaluation withheld")
    elif _flag_value(argv, "--max-batches") is not None:
        raise ValueError("E425 full fit cannot have a batch limit")
    artifact_names = spec.get("artifacts", [])
    expected_checkpoint = f"artifacts/runs/{model_name}/model_ep{epochs - 1}.pt"
    if expected_checkpoint not in artifact_names:
        raise ValueError(f"{spec_name}: fixed checkpoint is not declared")
    if expected_batches <= 0:
        raise ValueError("expected batch count must be positive")


def _validate_epoch_diagnostics(
    diagnostics: dict, epoch: int, attempts: int, batch_rows: int, chunk_rows: int
) -> None:
    """Validate soft-route telemetry without importing the model framework."""
    if diagnostics.get("epoch_index") != epoch:
        raise ValueError(f"soft-route epoch index mismatch: {diagnostics}")
    if diagnostics.get("optimizer_attempts") != attempts:
        raise ValueError(f"soft-route attempt count mismatch: {diagnostics}")
    expected_rows = attempts * batch_rows * chunk_rows
    if diagnostics.get("route_rows") != expected_rows:
        raise ValueError(f"soft-route row count mismatch: {diagnostics}")

    entropy = diagnostics.get("route_entropy_mean_nats")
    if not isinstance(entropy, (int, float)) or not math.isfinite(entropy):
        raise ValueError(f"nonfinite soft-route entropy: {diagnostics}")
    if not 0 <= entropy <= math.log(4) + 0.002:
        raise ValueError(f"soft-route entropy outside four-expert bounds: {entropy}")

    probabilities = diagnostics.get("route_probability_mean")
    if (
        not isinstance(probabilities, list)
        or len(probabilities) != 4
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in probabilities
        )
        or any(value < 0 for value in probabilities)
        or not math.isclose(sum(probabilities), 1.0, rel_tol=0, abs_tol=0.001)
    ):
        raise ValueError(f"invalid soft-route probability masses: {probabilities}")

    finite_steps = diagnostics.get("expert_prediction_gradient_finite_steps")
    nonfinite_steps = diagnostics.get("expert_prediction_gradient_nonfinite_steps")
    gradient_nonzero = diagnostics.get("expert_prediction_gradient_nonzero_steps")
    gradient_means = diagnostics.get("expert_prediction_gradient_l2_mean")
    update_means = diagnostics.get("expert_parameter_update_l2_mean")
    update_nonzero = diagnostics.get("expert_parameter_nonzero_update_steps")
    arrays = (
        finite_steps,
        nonfinite_steps,
        gradient_nonzero,
        gradient_means,
        update_means,
        update_nonzero,
    )
    if any(not isinstance(values, list) or len(values) != 4 for values in arrays):
        raise ValueError(f"soft-route per-expert diagnostics have invalid shape: {diagnostics}")
    for index in range(4):
        finite_count = finite_steps[index]
        nonfinite_count = nonfinite_steps[index]
        if (
            not isinstance(finite_count, int)
            or not isinstance(nonfinite_count, int)
            or finite_count < 0
            or nonfinite_count < 0
            or finite_count + nonfinite_count != attempts
        ):
            raise ValueError(
                f"expert {index} finite/nonfinite gradient counts do not sum to attempts"
            )
        if (
            not isinstance(gradient_nonzero[index], int)
            or not 0 <= gradient_nonzero[index] <= finite_count
        ):
            raise ValueError(f"expert {index} nonzero-gradient count is inconsistent")
        if finite_count == 0:
            if gradient_means[index] is not None:
                raise ValueError(
                    f"expert {index} must have null gradient mean with no finite samples"
                )
        elif (
            not isinstance(gradient_means[index], (int, float))
            or not math.isfinite(gradient_means[index])
            or gradient_means[index] < 0
        ):
            raise ValueError(f"expert {index} finite gradient mean is invalid")
        if (
            not isinstance(update_means[index], (int, float))
            or not math.isfinite(update_means[index])
            or update_means[index] < 0
            or not isinstance(update_nonzero[index], int)
            or not 0 <= update_nonzero[index] <= attempts
        ):
            raise ValueError(f"expert {index} measured update diagnostics are invalid")


def _validate_prediction_group_domains(
    predicted_groups: list[int], full_valid_groups: list[int], evaluation_groups: list[int]
) -> None:
    """Keep all-VALID exports distinct from the historical 368-group score panel."""
    if full_valid_groups != list(range(1873)) or predicted_groups != full_valid_groups:
        raise ValueError("prediction sidecar must cover the canonical 1,873 all-VALID groups")
    if (
        any(type(value) is not int for value in predicted_groups + evaluation_groups)
        or len(evaluation_groups) != 368
        or len(set(evaluation_groups)) != 368
        or not set(evaluation_groups).issubset(full_valid_groups)
    ):
        raise ValueError("provenance holdout must remain the distinct 368-group score panel")


def verify_controller_rounding_contracts() -> None:
    """Exercise the stdlib-only FP16 probability envelope and VALID/score split."""
    # The model stores softmax probabilities in FP16 during fast training. A
    # 0.001 mass and 0.002 entropy allowance conservatively bounds four-way
    # FP16 rounding (including subnormal quantization); these remain diagnostics.
    raw = (0.133, 0.177, 0.291, 0.399)
    probabilities = [struct.unpack("e", struct.pack("e", value))[0] for value in raw]
    entropy = -sum(value * math.log(max(value, 1e-12)) for value in probabilities)
    fixture = {
        "epoch_index": 0,
        "optimizer_attempts": 1,
        "route_rows": 1,
        "route_entropy_mean_nats": entropy,
        "route_probability_mean": probabilities,
        "expert_prediction_gradient_finite_steps": [1] * 4,
        "expert_prediction_gradient_nonfinite_steps": [0] * 4,
        "expert_prediction_gradient_nonzero_steps": [1] * 4,
        "expert_prediction_gradient_l2_mean": [1.0] * 4,
        "expert_parameter_update_l2_mean": [0.0] * 4,
        "expert_parameter_nonzero_update_steps": [0] * 4,
    }
    _validate_epoch_diagnostics(fixture, 0, 1, 1, 1)

    # The export must include all VALID groups even though only 368 historical
    # holdout groups feed the paired report. This catches accidental equality.
    full_valid = list(range(1873))
    historical_holdout = full_valid[:368]
    _validate_prediction_group_domains(full_valid, full_valid, historical_holdout)
    if full_valid == historical_holdout:
        raise AssertionError("full VALID and holdout fixtures unexpectedly coincide")
    malformed_panels = (
        historical_holdout[:-1] + [historical_holdout[0]],
        [False] + historical_holdout[1:],
        [1873] + historical_holdout[1:],
    )
    for panel in malformed_panels:
        try:
            _validate_prediction_group_domains(full_valid, full_valid, panel)
        except ValueError:
            continue
        raise AssertionError("malformed evaluation domain was accepted")


def validate_training_result(
    spec_name: str, model_name: str, epochs: int, batches_per_epoch: int
) -> None:
    """Validate counters, route telemetry, fixed checkpoint, and fit provenance."""
    run_dir = ROOT / "artifacts" / "runs" / model_name
    history = read_json(run_dir / "history.json")
    provenance = read_json(run_dir / "provenance.json")
    if len(history) != epochs:
        raise ValueError(f"{model_name}: expected {epochs} epochs, got {len(history)}")
    args = provenance.get("args", {})
    if args.get("specialist_route") != "soft" or args.get("arch") != "specialist":
        raise ValueError(f"{model_name}: provenance is not a soft specialist fit")
    if args.get("checkpoint_policy") != "fixed" or args.get("checkpoint_epoch") != epochs - 1:
        raise ValueError(f"{model_name}: checkpoint policy differs from sealed recipe")

    expected_attempts = batches_per_epoch * 20
    route_history = provenance.get("soft_route_diagnostics")
    if not isinstance(route_history, list) or len(route_history) != epochs:
        raise ValueError(f"{model_name}: missing per-epoch soft-route diagnostics")
    cumulative_attempts = 0
    cumulative_ema_updates = 0
    cumulative_completed_updates = 0
    cumulative_skips = 0
    cumulative_batches = 0
    for epoch, result in enumerate(history):
        if result.get("epoch") != epoch:
            raise ValueError(f"{model_name}: epoch order changed")
        if epochs == 1 and result.get("evaluation") != "withheld":
            raise ValueError(f"{model_name}: smoke evaluation must be withheld")
        update = result.get("update_diagnostics", {})
        counts = update.get("epoch", {})
        if (
            counts.get("completed_batches") != batches_per_epoch
            or counts.get("optimizer_attempts") != expected_attempts
            or counts.get("ema_updates") != expected_attempts
            or counts.get("optimizer_completed_updates", -1) + counts.get("amp_skipped_updates", -1)
            != expected_attempts
        ):
            raise ValueError(f"{model_name}: epoch update counters are inconsistent: {counts}")
        cumulative_attempts += counts["optimizer_attempts"]
        cumulative_ema_updates += counts["ema_updates"]
        cumulative_completed_updates += counts["optimizer_completed_updates"]
        cumulative_skips += counts["amp_skipped_updates"]
        cumulative_batches += counts["completed_batches"]
        expected_cumulative = {
            "optimizer_attempts": cumulative_attempts,
            "ema_updates": cumulative_ema_updates,
            "optimizer_completed_updates": cumulative_completed_updates,
            "amp_skipped_updates": cumulative_skips,
            "completed_batches": cumulative_batches,
        }
        if update.get("cumulative") != expected_cumulative:
            raise ValueError(f"{model_name}: epoch cumulative counters are inconsistent")
        if update.get("soft_route") != route_history[epoch]:
            raise ValueError(f"{model_name}: history/provenance route diagnostics differ")
        _validate_epoch_diagnostics(route_history[epoch], epoch, expected_attempts, 32, 1000)

    checkpoint = run_dir / f"model_ep{epochs - 1}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if epochs == 4:
        if provenance.get("selected_epoch") != 3:
            raise ValueError(f"{model_name}: fixed epoch 3 was not recorded as selected")
        if provenance.get("cumulative_updates") != history[-1]["update_diagnostics"].get(
            "cumulative"
        ):
            raise ValueError(f"{model_name}: cumulative history/provenance counters differ")
        validate_control_identity(provenance)
    elif provenance.get("selected_epoch") != 0:
        raise ValueError(f"{model_name}: fixed smoke checkpoint epoch 0 was not recorded")


def validate_control_identity(provenance: dict) -> None:
    """Require the soft fit to match learned/universal fold membership and normalization."""
    identity = read_json(CONTROL_IDENTITY)
    controls = identity["controls"]
    learned = controls["c20261004_h093_learned_s3_final"]
    universal = controls["c20261004_h093_universal_s3_final"]
    for run_name, control in (
        ("c20261004_h093_learned_s3_final", learned),
        ("c20261004_h093_universal_s3_final", universal),
    ):
        control_provenance_path = ROOT / f"artifacts/runs/{run_name}/provenance.json"
        if _sha256_file(control_provenance_path) != control["provenance_sha256"]:
            raise ValueError(f"{run_name}: provenance changed since the control identity audit")
        control_provenance = read_json(control_provenance_path)
        if (
            control_provenance.get("fit_membership_identity") != control["fit_membership_identity"]
            or control_provenance.get("normalization_identity") != control["normalization_identity"]
            or control_provenance.get("propensity_checkpoint_sha256")
            != control["propensity_checkpoint_sha256"]
        ):
            raise ValueError(f"{run_name}: current provenance differs from control identity audit")
    if learned["fit_membership_identity"] != universal["fit_membership_identity"]:
        raise ValueError("learned/universal control fit membership differs")
    if learned["normalization_identity"] != universal["normalization_identity"]:
        raise ValueError("learned/universal control normalization differs")
    for control_name, control in (("learned", learned), ("universal", universal)):
        if provenance.get("fit_membership_identity") != control["fit_membership_identity"]:
            raise ValueError(f"soft fit membership does not match {control_name} control")
        if provenance.get("normalization_identity") != control["normalization_identity"]:
            raise ValueError(f"soft normalization does not match {control_name} control")
        if (
            provenance.get("propensity_checkpoint_sha256")
            != control["propensity_checkpoint_sha256"]
        ):
            raise ValueError(f"soft propensity identity does not match {control_name} control")


def run_job(
    spec_name: str,
    attempt: str,
    completed: list[str],
    *,
    model_name: str | None = None,
    epochs: int | None = None,
    batches_per_epoch: int | None = None,
) -> None:
    """Consume one root-sealed spec and observe its exact attempt without retry."""
    guard()
    sealed_path = CAMPAIGN / f"specs/{spec_name}.sealed.json"
    if not sealed_path.is_file():
        raise FileNotFoundError(f"root must seal this spec before controller use: {sealed_path}")
    spec = read_json(sealed_path)
    if model_name is None:
        if spec_name != PREDICTION_NAME:
            raise ValueError(f"unexpected non-training E425 spec: {spec_name}")
        verify_sealed_prediction_spec()
    else:
        if epochs is None or batches_per_epoch is None:
            raise ValueError("training jobs require exact epoch and batch expectations")
        verify_training_spec(spec, spec_name, model_name, epochs, batches_per_epoch)
    record_path = RUNNER / f"attempts/{attempt}.json"
    if record_path.exists():
        record = read_json(record_path)
        if record.get("spec") != spec:
            raise ValueError(f"Sealed attempt identity changed: {attempt}")
        if record.get("status") == "finished":
            require_success(attempt)
            if model_name is None:
                validate_prediction()
            else:
                validate_training_result(spec_name, model_name, epochs, batches_per_epoch)
            if attempt not in completed:
                completed.append(attempt)
            save("running", completed)
            return
        if record.get("status") not in ("reserved", "running"):
            raise RuntimeError(f"Refusing to restart stopped attempt {attempt}: {record}")
    else:
        if model_name is None:
            prediction_paths = (
                ROOT / f"cache/pred_{PREDICTION_RUN}_valid.npy",
                ROOT / f"cache/pred_{PREDICTION_RUN}_valid.json",
                ROOT / f"cache/pred_{PREDICTION_RUN}_valid.npy.partial",
            )
            if any(path.exists() for path in prediction_paths):
                raise FileExistsError(
                    "E425 prediction outputs exist without the matching runner attempt"
                )
        subprocess.run(
            [
                sys.executable,
                "scripts/run_campaign.py",
                "launch",
                str(sealed_path),
                "--attempt",
                attempt,
            ],
            cwd=ROOT,
            check=True,
        )

    deadline = time.monotonic() + int(spec["limits"]["timeout_seconds"]) + 240
    while True:
        guard()
        record = read_json(record_path)
        if record.get("spec") != spec:
            raise ValueError(f"Attempt record no longer matches sealed spec: {attempt}")
        if record["status"] == "finished":
            require_success(attempt)
            break
        if record["status"] not in ("running", "reserved"):
            raise RuntimeError(f"Attempt stopped: {attempt}: {record['status']}")
        unit = record.get("unit")
        if not isinstance(unit, str) or not unit:
            raise RuntimeError(f"Attempt lacks its exact live service handle: {attempt}")
        observed = properties(unit)
        if observed.get("ActiveState") not in ("active", "activating"):
            raise RuntimeError(f"Attempt handle disappeared: {attempt}: {observed}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"Observation deadline for {attempt}; do not restart")
        save("running", completed, active_attempt=attempt, observed=observed)
        time.sleep(5)
    if model_name is None:
        validate_prediction()
    else:
        validate_training_result(spec_name, model_name, epochs, batches_per_epoch)
    if attempt not in completed:
        completed.append(attempt)
    save("running", completed)


def prediction_draft() -> dict[str, object]:
    """Create a predictor draft whose staged/root code matches its source closure."""
    template = read_json(E424_PREDICTION_TEMPLATE)
    stage = CAMPAIGN / "staging/e425"
    source_paths = set((ROOT / "src/wnn").rglob("*.py"))
    source_paths.update((stage / "src/wnn").rglob("*.py"))
    source_paths.update((stage / "scripts").glob("*.py"))
    source_paths.add(stage / "scripts/predict_e425_soft.py")
    root_utils = ROOT / "utils.py"
    if root_utils.is_file():
        source_paths.add(root_utils)
    if any(not path.is_file() for path in source_paths):
        raise FileNotFoundError("E425 prediction source closure is incomplete")
    prediction_code = {path.relative_to(ROOT).as_posix() for path in source_paths}
    draft = dict(template)
    draft["name"] = PREDICTION_NAME
    draft["argv"] = [
        "{python}",
        "research/campaign_2026-10-04/staging/e425/scripts/predict_e425_soft.py",
        FULL_RUN,
        "valid",
        "--epoch",
        "3",
        "--inference-precision",
        "fp32",
        "--output-name",
        PREDICTION_RUN,
    ]
    draft["code"] = sorted(prediction_code)
    draft["inputs"] = [
        "datasets/valid.parquet",
        "cache/valid_targets.npz",
        VALID_CONTROL_SIDECAR.relative_to(ROOT).as_posix(),
        f"artifacts/runs/{FULL_RUN}/model_ep3.pt",
        f"artifacts/runs/{FULL_RUN}/provenance.json",
    ]
    draft["artifacts"] = [
        f"cache/pred_{PREDICTION_RUN}_valid.npy",
        f"cache/pred_{PREDICTION_RUN}_valid.json",
    ]
    return draft


def create_prediction_draft() -> None:
    """Create and seal the reviewed export draft after fit provenance validates."""
    checkpoint = ROOT / f"artifacts/runs/{FULL_RUN}/model_ep3.pt"
    provenance = ROOT / f"artifacts/runs/{FULL_RUN}/provenance.json"
    if not checkpoint.is_file() or not provenance.is_file():
        raise FileNotFoundError("E425 fixed epoch-3 checkpoint/provenance is not present")
    validate_training_result(ARMS[1][2], FULL_RUN, 4, 378)
    if PREDICTION_DRAFT.exists():
        if read_json(PREDICTION_DRAFT) != prediction_draft():
            raise ValueError(f"Prediction draft changed after creation: {PREDICTION_DRAFT}")
    else:
        draft = prediction_draft()
        with PREDICTION_DRAFT.open("x") as target:
            target.write(json.dumps(draft, indent=2) + "\n")
    if not PREDICTION_SEALED.exists():
        subprocess.run(
            [
                sys.executable,
                "scripts/run_campaign.py",
                "seal",
                str(PREDICTION_DRAFT),
                str(PREDICTION_SEALED),
            ],
            cwd=ROOT,
            check=True,
        )
    verify_sealed_prediction_spec()


def verify_sealed_prediction_spec() -> dict:
    """Check the root-sealed export contract before consuming it."""
    sealed = read_json(PREDICTION_SEALED)
    expected = prediction_draft()
    for key in ("version", "name", "kind", "argv", "artifacts", "limits", "environment"):
        if sealed.get(key) != expected.get(key):
            raise ValueError(f"sealed prediction differs from generated E425 draft in {key}")
    if sorted(item["path"] for item in sealed.get("code", [])) != sorted(expected["code"]):
        raise ValueError("sealed prediction code closure differs from generated E425 draft")
    if sorted(item["path"] for item in sealed.get("inputs", [])) != sorted(expected["inputs"]):
        raise ValueError("sealed prediction input paths differ from generated E425 draft")
    sealed_code = {item["path"]: item for item in sealed.get("code", [])}
    for relative_path in expected["code"]:
        path = ROOT / relative_path
        entry = sealed_code.get(relative_path)
        if entry is None or entry.get("sha256") != _sha256_file(path):
            raise ValueError(f"sealed prediction source hash changed: {relative_path}")
        if entry.get("size") != path.stat().st_size:
            raise ValueError(f"sealed prediction source size changed: {relative_path}")
    checkpoint_path = ROOT / f"artifacts/runs/{FULL_RUN}/model_ep3.pt"
    checkpoint_entry = next(
        (
            item
            for item in sealed["inputs"]
            if item["path"] == checkpoint_path.relative_to(ROOT).as_posix()
        ),
        None,
    )
    if checkpoint_entry is None or checkpoint_entry.get("sha256") != _sha256_file(checkpoint_path):
        raise ValueError("sealed prediction does not pin the actual fixed epoch-3 checkpoint")
    control_entry = next(
        (
            item
            for item in sealed["inputs"]
            if item["path"] == VALID_CONTROL_SIDECAR.relative_to(ROOT).as_posix()
        ),
        None,
    )
    if control_entry is None or control_entry.get("sha256") != _sha256_file(VALID_CONTROL_SIDECAR):
        raise ValueError("sealed prediction does not pin the all-VALID control sidecar")
    return sealed


def validate_prediction() -> None:
    """Check FP32 sidecar identity, complete VALID groups, and byte hashes."""
    pred_path = ROOT / f"cache/pred_{PREDICTION_RUN}_valid.npy"
    sidecar_path = pred_path.with_suffix(".json")
    if not pred_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("E425 FP32 prediction pair is incomplete")
    sidecar = read_json(sidecar_path)
    provenance = read_json(ROOT / f"artifacts/runs/{FULL_RUN}/provenance.json")
    valid_control = read_json(VALID_CONTROL_SIDECAR)
    expected_checkpoint = _sha256_file(ROOT / f"artifacts/runs/{FULL_RUN}/model_ep3.pt")
    if (
        sidecar.get("run") != FULL_RUN
        or sidecar.get("split") != "valid"
        or sidecar.get("epoch") != 3
        or sidecar.get("gate_mode") != "soft"
        or sidecar.get("specialist_route") != "soft"
        or sidecar.get("inference_precision") != "fp32"
        or sidecar.get("checkpoint_sha256") != expected_checkpoint
        or sidecar.get("prediction_sha256") != _sha256_file(pred_path)
        or sidecar.get("model_source_sha256") != provenance.get("model_source_sha256")
    ):
        raise ValueError("E425 prediction sidecar does not match the fixed soft FP32 model")
    _validate_prediction_group_domains(
        sidecar.get("row_groups", []),
        valid_control.get("row_groups", []),
        provenance.get("evaluation_row_groups", []),
    )
    closure = sidecar.get("code_source_closure", {})
    expected_files = []
    for relative_path in sorted(prediction_draft()["code"]):
        expected_files.append({"path": relative_path, "sha256": _sha256_file(ROOT / relative_path)})
    closure_digest = hashlib.sha256(
        json.dumps(expected_files, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    if (
        closure.get("schema") != "e425-prediction-source-closure-v1"
        or closure.get("files") != expected_files
        or closure.get("sha256") != closure_digest
    ):
        raise ValueError("E425 prediction source closure differs from the sealed source files")


def main() -> None:
    """Wait for E426, consume E425 smoke/full specs, then stage prediction export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if OUTPUT.exists() and not args.resume:
        raise FileExistsError(OUTPUT)
    completed = read_json(OUTPUT)["completed"] if OUTPUT.exists() else []
    try:
        verify_controller_rounding_contracts()
        wait_for_e426(completed)
        for spec_name, attempt, model_name, epochs, batches in ARMS:
            run_job(
                spec_name,
                attempt,
                completed,
                model_name=model_name,
                epochs=epochs,
                batches_per_epoch=batches,
            )
        create_prediction_draft()
        verify_sealed_prediction_spec()
        run_job(
            PREDICTION_NAME,
            PREDICTION_ATTEMPT,
            completed,
        )
        validate_prediction()
        save(
            "fits_and_exports_complete",
            completed,
            paired_score="pending",
            prediction_run=PREDICTION_RUN,
        )
    except Exception as error:
        save("stopped", completed, reason=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
