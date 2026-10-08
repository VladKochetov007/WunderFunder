#!/usr/bin/env python3
"""Run the gated E540 V6 AMP4096 profile, paired fits, and frozen-group exports."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import resource
import struct
import sys
import time
from pathlib import Path
from typing import TextIO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from e540_cli_preflight import normalize_entry_argv, parse_project_cli  # noqa: E402, I001

PACKET_ID = "E540_AMP4096_V6"
AMP_INIT_SCALE = 4096.0
MAX_WALL_SECONDS = 5_200
PROJECTION_MARGIN_SECONDS = 300
RUNNER_FINALIZATION_RESERVE_SECONDS = 300
PROFILE_WARMUP_BATCHES = 1
PROFILE_TIMED_BATCHES = 3
UPDATES_PER_OUTER_BATCH = math.ceil(20_000 / 256)
EXPECTED_PROFILE_ATTEMPTS = (PROFILE_WARMUP_BATCHES + PROFILE_TIMED_BATCHES) * UPDATES_PER_OUTER_BATCH
ALLOCATOR_LIMIT_BYTES = 4_300_000_000
RUN_NAMES = {
    "linear": "e540v6-linear-s540-amp4096",
    "ple": "e540v6-ple-s540-amp4096",
}
AUTHORIZATION_PATH = Path(
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/"
    "E540_V6_ROOT_CONDITIONAL_AUTHORIZATION.json"
)
SPLIT_MANIFEST = Path("research/campaign_2026-09-29/artifacts/splits/fold0.json")
SUBSET_MANIFEST = Path(
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/E540_FIT_SUBSET_V5.json"
)
PORTABLE_INPUTS = Path("research/campaign_2026-10-04/portable_inputs.json")
MOUNTED_DATA_SOURCES = (
    "datasets/train.parquet",
    "datasets/valid.parquet",
    "cache/train_targets.npz",
    "cache/valid_targets.npz",
)
RUNTIME = ROOT / "e540_cohort"


def write_json(path: Path, value: object) -> None:
    """Atomically persist a finite receipt before advancing a cohort phase."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    """Hash checkpoints and predictions in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Tee(io.TextIOBase):
    """Keep worker output in the provider log and a phase-specific local receipt."""

    def __init__(self, target: TextIO, capture: io.StringIO) -> None:
        self.target = target
        self.capture = capture

    def write(self, value: str) -> int:
        self.capture.write(value)
        written = self.target.write(value)
        self.target.flush()
        return written

    def flush(self) -> None:
        self.target.flush()
        self.capture.flush()


def parse_trainer_output(output: str) -> dict[str, object]:
    """Extract the trainer's own finite timing and AMP-update evidence."""
    found: dict[str, object] = {}
    for line in output.splitlines():
        for prefix, key in (
            ("PROFILE ", "profile"),
            ("PROFILE_UPDATE_EVIDENCE ", "update_evidence"),
        ):
            if line.startswith(prefix):
                found[key] = json.loads(line[len(prefix) :])
    if set(found) != {"profile", "update_evidence"}:
        raise ValueError("trainer did not emit both profile and AMP update evidence")
    return found


def validate_profile(record: dict[str, object]) -> tuple[float, dict[str, object]]:
    """Reject a profile unless its timed outer steps and every AMP update are valid."""
    profile = record.get("profile")
    evidence = record.get("update_evidence")
    if not isinstance(profile, dict) or not isinstance(evidence, dict):
        raise ValueError("profile receipt is missing trainer metrics")
    step_wall = profile.get("step_wall")
    if isinstance(step_wall, bool) or not isinstance(step_wall, (int, float)):
        raise ValueError("profile step_wall is not numeric")
    if not math.isfinite(float(step_wall)) or float(step_wall) <= 0:
        raise ValueError("profile step_wall is not finite and positive")
    if (
        evidence.get("optimizer_attempts") != EXPECTED_PROFILE_ATTEMPTS
        or evidence.get("optimizer_completed_updates") != EXPECTED_PROFILE_ATTEMPTS
        or evidence.get("amp_skipped_updates") != 0
        or evidence.get("loss_checks") != EXPECTED_PROFILE_ATTEMPTS
        or evidence.get("loss_finite") is not True
        or evidence.get("gradient_finite") is not True
        or evidence.get("valid_for_throughput") is not True
    ):
        raise ValueError("profile must complete all 316 finite optimizer updates with zero AMP skips")
    return float(step_wall), evidence


def expected_fit_updates(train_groups: int, valid_groups: int, batch: int = 2) -> int:
    """Count the fixed-drop-last outer batches and 256-row recurrent updates."""
    outer_batches = (train_groups + valid_groups) // batch
    return outer_batches * UPDATES_PER_OUTER_BATCH


def validate_numeric_checkpoint(
    mode: str,
    numeric: dict[str, object],
    subset: dict[str, object],
    model_state: dict[str, object],
) -> dict[str, object]:
    """Bind fitted preprocessing to the exact fit members and stored PLE buffers."""
    import numpy as np

    fit_groups = subset["selected_groups"]
    if (
        numeric.get("mode") != mode
        or numeric.get("n_features") != 112
        or numeric.get("n_intervals") != 8
        or numeric.get("mapped_input_width") != 1107
        or numeric.get("split_manifest_sha256") != subset.get("split_manifest_sha256")
        or numeric.get("fit_members") != fit_groups
        or numeric.get("fit_members_sha256") != subset.get("selected_members_sha256")
    ):
        raise ValueError("checkpoint numeric preprocessing is not bound to the frozen fit membership")
    if mode == "linear":
        if numeric.get("control_definition") != "same K+1 slots; only raw bypass coordinate is active":
            raise ValueError("linear control numeric-map definition differs from E540 V6")
        return {
            "mode": mode,
            "fit_members_sha256": numeric["fit_members_sha256"],
            "mapped_input_width": numeric["mapped_input_width"],
        }
    if mode != "ple":
        raise ValueError(f"unsupported E540 numeric checkpoint mode: {mode}")
    if numeric.get("fit_member_count") != sum(map(len, fit_groups.values())):
        raise ValueError("PLE knot provenance has the wrong fitting sequence count")

    def array(value: object) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    knot_value = model_state.get("numeric_feature_map.knots")
    count_value = model_state.get("numeric_feature_map.knot_counts")
    fitted_value = model_state.get("numeric_feature_map.knots_fitted")
    if knot_value is None or count_value is None or fitted_value is None:
        raise ValueError("PLE checkpoint is missing its fixed knot buffers")
    knots = array(knot_value).astype("<f4", copy=False)
    counts = array(count_value).astype("<i8", copy=False)
    fitted = bool(array(fitted_value).item())
    if (
        knots.shape != (112, 9)
        or counts.shape != (112,)
        or not np.isfinite(knots).all()
        or np.any(counts < 1)
        or np.any(counts > 9)
        or not fitted
    ):
        raise ValueError("PLE checkpoint knot buffers are malformed or were not installed")
    knot_digest = hashlib.sha256(knots.tobytes() + counts.tobytes()).hexdigest()
    if numeric.get("knot_sha256") != knot_digest:
        raise ValueError("PLE checkpoint knot-buffer digest differs from its preprocessing receipt")
    return {
        "mode": mode,
        "fit_members_sha256": numeric["fit_members_sha256"],
        "mapped_input_width": numeric["mapped_input_width"],
        "fit_member_count": numeric["fit_member_count"],
        "samples_per_sequence": numeric.get("samples_per_sequence"),
        "candidate_sample_rows": numeric.get("candidate_sample_rows"),
        "retained_sample_rows": numeric.get("retained_sample_rows"),
        "knot_sha256": knot_digest,
        "knots_fitted": fitted,
    }


def elapsed_projection_seconds(
    elapsed: float,
    profiles: dict[str, dict[str, object]],
    train_group_count: int,
    valid_group_count: int,
    evaluation_group_count: int,
) -> float:
    """Project remaining work; elapsed already includes completed profile durations."""
    fit_seconds = 0.0
    prediction_seconds = 0.0
    for item in profiles.values():
        outer_wall = float(item["step_wall_ms"]) / 1000.0
        fit_batches = (train_group_count + valid_group_count) // 2
        fit_seconds += fit_batches * outer_wall * 1.35
        prediction_seconds += math.ceil(evaluation_group_count / 16) * outer_wall * 0.75
    return elapsed + fit_seconds + prediction_seconds + PROJECTION_MARGIN_SECONDS


def worker_wall_budget_seconds(
    kernel_deadline_monotonic: float, *, now_monotonic: float
) -> int:
    """Cap the worker at 5,200s while preserving runtime for runner finalization."""
    remaining = math.floor(
        kernel_deadline_monotonic - now_monotonic - RUNNER_FINALIZATION_RESERVE_SECONDS
    )
    return max(0, min(MAX_WALL_SECONDS, remaining))


def trainer_argv(mode: str, *, profile: bool) -> list[str]:
    """Return the same explicit V6 recipe in both arms, adding profile diagnostics only."""
    argv = [
        "scripts/train_rnn.py",
        "--device", "cuda",
        "--name", RUN_NAMES[mode],
        "--numeric-map", mode,
        "--ple-intervals", "8",
        "--seed", "540",
        "--train-source", "train+valid",
        "--split-manifest", SPLIT_MANIFEST.as_posix(),
        "--fit-subset-manifest", SUBSET_MANIFEST.as_posix(),
        "--hidden", "128",
        "--layers", "1",
        "--dropout", "0",
        "--batch", "2",
        "--chunk", "256",
        "--epochs", "1",
        "--loss", "wmse",
        "--loss-norm", "chunk",
        "--optim", "adamw",
        "--lr", "0.001",
        "--wd", "0.01",
        "--sched", "onecycle",
        "--precision", "fp16",
        "--amp-init-scale", "4096",
        "--batch-order", "seeded",
        "--checkpoint-policy", "fixed",
        "--checkpoint-epoch", "0",
        "--no-eval-during-fit",
        "--deterministic",
        "--distill", "",
        "--distill-cache-root", "",
        "--distill-family-weight", "0",
        "--propensity", "",
        "--prop-input", "",
        "--prop-power", "1",
        "--prop-eps", "0",
        "--aux-next", "0",
        "--aux-tier", "0",
        "--aux-mask", "0",
        "--aux-sign", "0",
        "--aux-quote", "0",
        "--aux-struct", "0",
        "--aux", "",
        "--aux-weight", "0",
    ]
    if profile:
        argv.extend(("--profile-warmup", str(PROFILE_WARMUP_BATCHES)))
        argv.extend(("--profile-steps", str(PROFILE_TIMED_BATCHES)))
    return argv


def predict_argv(mode: str) -> list[str]:
    """Export only the frozen evaluation groups through feature-only FP32 reads."""
    return [
        "scripts/predict_run.py",
        RUN_NAMES[mode],
        "valid",
        "--epoch", "0",
        "--inference-precision", "fp32",
        "--output-name", f"{RUN_NAMES[mode]}-eval",
        "--fit-subset-manifest", SUBSET_MANIFEST.as_posix(),
    ]


def call_main(module: object, script_name: str, argv: list[str], log_path: Path) -> str:
    """Call a project entry point in-process to reuse validated input digests."""
    capture = io.StringIO()
    old_argv = sys.argv
    try:
        cli_arguments = normalize_entry_argv(script_name, argv)
        sys.argv = [script_name, *cli_arguments]
        with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(
            Tee(log, capture)
        ):
            module.main()
    finally:
        sys.argv = old_argv
    return capture.getvalue()


def validate_recipe_cli() -> list[dict[str, object]]:
    """Parse the real profile, fit and prediction CLIs before mounted-data hashing."""
    trainer_path = ROOT / "scripts/train_rnn.py"
    dynamics_path = ROOT / "src/wnn/training_dynamics.py"
    predictor_path = ROOT / "scripts/predict_run.py"
    checks: list[dict[str, object]] = []
    for mode in ("linear", "ple"):
        for stage in ("profile", "fit"):
            logged = trainer_argv(mode, profile=stage == "profile")
            cli = normalize_entry_argv("train_rnn.py", logged)
            parsed = parse_project_cli(
                trainer_path, cli, parser_name="ap", dynamics_path=dynamics_path
            )
            expected_steps = PROFILE_TIMED_BATCHES if stage == "profile" else 0
            if (
                parsed.name != RUN_NAMES[mode]
                or parsed.numeric_map != mode
                or parsed.amp_init_scale != AMP_INIT_SCALE
                or parsed.loss != "wmse"
                or parsed.loss_norm != "chunk"
                or parsed.profile_steps != expected_steps
                or (stage == "profile" and parsed.profile_warmup != PROFILE_WARMUP_BATCHES)
            ):
                raise ValueError(f"E540 {mode} {stage} CLI differs from the frozen recipe")
            checks.append({
                "entry_point": "scripts/train_rnn.py",
                "mode": mode,
                "stage": stage,
                "status": "parsed",
                "script_token_removed": logged[1:] == cli,
            })
        logged = predict_argv(mode)
        cli = normalize_entry_argv("predict_run.py", logged)
        parsed = parse_project_cli(predictor_path, cli, parser_name="parser")
        if (
            parsed.name != RUN_NAMES[mode]
            or parsed.split != "valid"
            or parsed.epoch != 0
            or parsed.inference_precision != "fp32"
            or parsed.fit_subset_manifest != SUBSET_MANIFEST
        ):
            raise ValueError(f"E540 {mode} prediction CLI differs from the frozen recipe")
        checks.append({
            "entry_point": "scripts/predict_run.py",
            "mode": mode,
            "stage": "prediction",
            "status": "parsed",
            "script_token_removed": logged[1:] == cli,
        })
    return checks


def validate_conditional_authorization(
    receipt_path: Path, expected: dict[str, object]
) -> dict[str, object]:
    """Require root's packet-bound fit authorization before hashing mounted data."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "schema": "wnn.e540-conditional-fit-authorization.v1",
        "owner": "root",
        "status": "approved",
        "authorized": True,
        "conditional_fit_after_profile_gate": True,
    }
    for key, value in required.items():
        if receipt.get(key) != value:
            raise ValueError(f"root conditional authorization missing valid {key}")
    for key in (
        "packet_id",
        "code_manifest_sha256",
        "recipe_sha256",
        "subset_manifest_sha256",
        "kernel_process_wall_seconds_max",
        "worker_wall_seconds_max",
    ):
        if receipt.get(key) != expected.get(key):
            raise ValueError(f"root conditional authorization does not bind {key}")
    authorization_id = receipt.get("authorization_id")
    if not isinstance(authorization_id, str) or not authorization_id.strip():
        raise ValueError("root conditional authorization requires a nonempty authorization_id")
    return receipt


def _conditional_authorization_contract() -> dict[str, object]:
    """Load expected identity from the runner's hash-bound spec environment."""
    fields = {
        "packet_id": "WNN_E540_PACKET_ID",
        "code_manifest_sha256": "WNN_E540_CODE_MANIFEST_SHA256",
        "recipe_sha256": "WNN_E540_RECIPE_SHA256",
        "subset_manifest_sha256": "WNN_E540_SUBSET_SHA256",
    }
    contract: dict[str, object] = {}
    for field, environment_name in fields.items():
        value = os.environ.get(environment_name)
        if not value:
            raise RuntimeError(f"missing required authorization setting {environment_name}")
        contract[field] = value
    if contract["packet_id"] != PACKET_ID:
        raise RuntimeError("conditional-authorization packet ID differs from the worker")
    if os.environ.get("WNN_E540_AUTHORIZATION_PATH") != AUTHORIZATION_PATH.as_posix():
        raise RuntimeError("conditional-authorization receipt path differs from the worker")
    for field, environment_name in (
        ("kernel_process_wall_seconds_max", "WNN_E540_KERNEL_WALL_SECONDS"),
        ("worker_wall_seconds_max", "WNN_E540_WORKER_WALL_SECONDS"),
    ):
        raw = os.environ.get(environment_name)
        if raw is None or not raw.isdecimal():
            raise RuntimeError(f"invalid required authorization setting {environment_name}")
        contract[field] = int(raw)
    return contract


def resident_memory_bytes() -> int:
    """Return this process' peak resident set size in portable Linux units."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * (1024 if sys.platform != "darwin" else 1)


def verify_mounted_data() -> None:
    """Hash all deferred external inputs once and persist observations before Torch import."""
    deferred = {
        value
        for value in os.environ.get("WNN_DEFERRED_IDENTITY_PATHS", "").split(",")
        if value
    }
    if deferred != set(MOUNTED_DATA_SOURCES):
        raise ValueError("worker external-input identity scope differs from the frozen packet")
    split = json.loads((ROOT / SPLIT_MANIFEST).read_text(encoding="utf-8"))
    portable_path = ROOT / PORTABLE_INPUTS
    portable = json.loads(portable_path.read_text(encoding="utf-8"))
    receipt_path = RUNTIME / "input_identity.json"
    receipt: dict[str, object] = {
        "packet_id": PACKET_ID,
        "status": "verifying",
        "portable_manifest_sha256": sha256_file(portable_path),
        "sources": [],
        "method": "streaming sha256 through wnn.portability.verify_portable_source; no parquet footer or NPZ arrays loaded",
    }
    write_json(receipt_path, receipt)
    from wnn.portability import verify_portable_source

    for name in MOUNTED_DATA_SOURCES:
        identity = split["sources"][name]
        entry = portable["sources"][name]
        if entry.get("original_identity") != identity:
            raise ValueError(f"portable registry disagrees with the frozen split for {name}")
        observed: dict[str, object] = {
            "path": name,
            "expected": {"size": identity["bytes"], "sha256": entry["sha256"]},
            "observed": {"size": None, "sha256": None},
            "status": "hashing",
        }
        receipt["sources"].append(observed)

        def persist_observation(size: int, digest: str | None) -> None:
            observed["observed"] = {"size": size, "sha256": digest}
            observed["status"] = (
                "match" if size == identity["bytes"] and digest == entry["sha256"] else "mismatch"
            )
            write_json(receipt_path, receipt)

        digest = verify_portable_source(
            ROOT / name,
            name,
            identity,
            portable_path,
            observation_callback=persist_observation,
        )
        if digest != entry["sha256"]:
            raise ValueError(f"mounted external input failed its portable digest: {name}")
    receipt["status"] = "verified"
    write_json(receipt_path, receipt)


def validate_frozen_membership(
    subset: dict[str, object], expected: dict[str, object], campaign: object
) -> dict[str, object]:
    """Verify exact seeded component selection and component-complete fit membership."""
    import numpy as np

    required = (
        "split_manifest_sha256",
        "selection_seed",
        "selected_component_count",
        "eligible_component_count",
        "selected_components",
        "selected_component_sha256",
        "selected_groups",
        "selected_group_counts",
        "selected_group_sha256",
        "selected_members_sha256",
        "evaluation_groups",
        "evaluation_group_count",
        "evaluation_membership_sha256",
    )
    for key in required:
        if subset.get(key) != expected.get(key):
            raise ValueError(f"mounted E540 V6 subset differs from deterministic selection: {key}")
    if subset.get("selected_component_count") != 512 or subset.get("evaluation_group_count") != 468:
        raise ValueError("E540 V6 requires exactly 512 fitting components and 468 evaluation groups")
    if subset.get("selection_seed") != 540:
        raise ValueError("E540 V6 component selection seed differs from the frozen recipe")

    train_groups = list(map(int, subset["selected_groups"]["train"]))
    valid_groups = list(map(int, subset["selected_groups"]["valid"]))
    evaluation_groups = list(map(int, subset["evaluation_groups"]))
    selected_components = list(map(int, subset["selected_components"]))
    if (
        train_groups != sorted(set(train_groups))
        or valid_groups != sorted(set(valid_groups))
        or evaluation_groups != sorted(set(evaluation_groups))
        or selected_components != sorted(set(selected_components))
    ):
        raise ValueError("E540 V6 group and component lists must be sorted and unique")
    campaign.assert_fit_groups(train_groups, valid_groups)
    if evaluation_groups != sorted(map(int, campaign.evaluation_groups)):
        raise ValueError("E540 V6 evaluation membership differs from the frozen split")

    component_of = {
        (split, int(group)): int(component)
        for split, group, component in campaign.components
    }
    fitting_component_ids = {
        component_of[("train", group)] for group in train_groups
    } | {component_of[("valid", group)] for group in valid_groups}
    if fitting_component_ids != set(selected_components):
        raise ValueError("E540 V6 selected component list differs from its fitting groups")
    for split, selected in (("train", train_groups), ("valid", valid_groups)):
        complete = sorted(
            group
            for (member_split, group), component in component_of.items()
            if member_split == split and component in fitting_component_ids
        )
        if selected != complete:
            raise ValueError(f"E540 V6 fitting groups are not component-complete for {split}")
        digest = hashlib.sha256(np.asarray(selected, dtype="<i8").tobytes()).hexdigest()
        if subset["selected_group_sha256"].get(split) != digest:
            raise ValueError(f"E540 V6 {split} fitting group digest mismatch")

    component_digest = hashlib.sha256(
        np.asarray(selected_components, dtype="<i8").tobytes()
    ).hexdigest()
    evaluation_digest = hashlib.sha256(
        np.asarray(evaluation_groups, dtype="<i8").tobytes()
    ).hexdigest()
    return {
        "status": "verified_before_profile_and_fit",
        "protocol": "first 512 components from seed-540 permutation of sorted eligible component IDs",
        "selection_seed": 540,
        "eligible_component_count": int(subset["eligible_component_count"]),
        "selected_component_count": len(selected_components),
        "realized_component_fraction": len(selected_components) / int(subset["eligible_component_count"]),
        "selected_component_sha256": component_digest,
        "fit_group_counts": {"train": len(train_groups), "valid": len(valid_groups)},
        "fit_group_sha256": dict(subset["selected_group_sha256"]),
        "fit_members_sha256": subset["selected_members_sha256"],
        "evaluation_group_count": len(evaluation_groups),
        "evaluation_membership_sha256": evaluation_digest,
        "split_manifest_sha256": campaign.sha256,
        "component_complete": True,
        "fit_evaluation_components_disjoint": True,
    }


def verify_frozen_membership(
    data_root: Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Recompute and persist the exact fit/evaluation membership before Torch import."""
    import sys as worker_sys

    worker_sys.path.insert(0, str(ROOT / "scripts"))
    worker_sys.path.insert(0, str(ROOT / "src"))
    from prepare_e540_fit_subset import build_manifest
    from wnn.splits import load_campaign_split

    source_root = ROOT if data_root is None else data_root.resolve(strict=True)
    subset = json.loads((ROOT / SUBSET_MANIFEST).read_text(encoding="utf-8"))
    split_path = ROOT / SPLIT_MANIFEST
    campaign = load_campaign_split(split_path, source_root)
    expected = build_manifest(
        split_path,
        None,
        540,
        source_root,
        component_count=512,
    )
    receipt = validate_frozen_membership(subset, expected, campaign)
    receipt["subset_manifest"] = SUBSET_MANIFEST.as_posix()
    receipt["subset_manifest_sha256"] = hashlib.sha256(
        (ROOT / SUBSET_MANIFEST).read_bytes()
    ).hexdigest()
    if data_root is None:
        write_json(RUNTIME / "membership.json", receipt)
    return subset, receipt


def run() -> None:
    """Execute profiles, gate the paired fit, and export only the fixed 468 evaluation groups."""
    started = time.monotonic()
    deadline_value = os.environ.get("WNN_KERNEL_DEADLINE_MONOTONIC")
    if deadline_value is None:
        raise RuntimeError("E540 V6 requires the bounded Kaggle kernel bootstrap")
    try:
        kernel_deadline = float(deadline_value)
    except ValueError as error:
        raise RuntimeError("invalid Kaggle kernel wall deadline") from error
    if not math.isfinite(kernel_deadline):
        raise RuntimeError("invalid Kaggle kernel wall deadline")
    worker_wall_seconds = worker_wall_budget_seconds(
        kernel_deadline, now_monotonic=started
    )
    if worker_wall_seconds < 1:
        raise TimeoutError("kernel deadline leaves no worker time after finalization reserve")
    RUNTIME.mkdir(parents=True, exist_ok=False)
    preflight_path = RUNTIME / "startup_preflight.json"
    preflight: dict[str, object] = {"packet_id": PACKET_ID, "status": "running"}
    write_json(preflight_path, preflight)
    try:
        cli_checks = validate_recipe_cli()
        preflight["cli_checks"] = cli_checks
        preflight["status"] = "cli_passed_authorization_pending_before_data_hashing"
        write_json(preflight_path, preflight)
        authorization = validate_conditional_authorization(
            ROOT / AUTHORIZATION_PATH, _conditional_authorization_contract()
        )
    except BaseException as error:
        preflight.update({
            "status": "failed_before_mounted_data_hashing",
            "failure": f"{type(error).__name__}: {error}",
        })
        write_json(preflight_path, preflight)
        raise
    preflight.update({
        "status": "passed_before_mounted_data_hashing",
        "authorization_id": authorization["authorization_id"],
    })
    write_json(preflight_path, preflight)
    receipts = RUNTIME / "receipts"
    logs = RUNTIME / "logs"
    logs.mkdir()
    for mode, run_name in RUN_NAMES.items():
        write_json(
            receipts / f"{mode}.json",
            {
                "mode": mode,
                "run_name": run_name,
                "status": "not_started",
                "profile_amp_initial_scale": AMP_INIT_SCALE,
                "fit_amp_initial_scale": AMP_INIT_SCALE,
            },
        )
    cohort = {
        "packet_id": PACKET_ID,
        "status": "profile_pending",
        "fit_admitted_by_packet": False,
        "provider_review_required": True,
        "maximum_worker_seconds": worker_wall_seconds,
        "declared_worker_ceiling_seconds": MAX_WALL_SECONDS,
        "runner_finalization_reserve_seconds": RUNNER_FINALIZATION_RESERVE_SECONDS,
        "amp_initial_scale": AMP_INIT_SCALE,
        "phase": "input_identity_preflight_completed_by_kaggle_gpu_runner",
        "started_unix": time.time(),
        "arm_receipts": {},
    }
    write_json(RUNTIME / "cohort.json", cohort)
    # Keep the large raw-data hashes in this process: the trainer's portable
    # verifier reuses the same exact-stat cache when the fit starts.
    verify_mounted_data()
    subset, membership = verify_frozen_membership()
    cohort["membership"] = "membership.json"
    cohort["phase"] = "membership_verified_before_profile_and_fit"
    selected_sequence_count = sum(subset["selected_group_counts"].values())
    loader_membership = {
        "status": "derived_from_frozen_batch_size_and_drop_last_contract; actual_indices_not_captured",
        "selected_sequence_count": selected_sequence_count,
        "batch_size": 2,
        "drop_last": True,
        "expected_consumed_sequence_count_per_epoch": selected_sequence_count
        - selected_sequence_count % 2,
        "expected_dropped_sequence_count_per_epoch": selected_sequence_count % 2,
        "dropped_sequence_identity": None,
        "paired_order_status": "same seed and sampler configuration; realized order equivalence is not independently verified",
    }
    cohort["loader_membership"] = loader_membership
    for mode in RUN_NAMES:
        arm_path = receipts / f"{mode}.json"
        arm = json.loads(arm_path.read_text(encoding="utf-8"))
        arm["selected_fit_group_counts"] = subset["selected_group_counts"]
        arm["selected_fit_sequence_count"] = selected_sequence_count
        arm["loader_membership"] = loader_membership
        write_json(arm_path, arm)
    write_json(RUNTIME / "cohort.json", cohort)
    # Torch is deliberately imported only after input identities and exact membership pass.
    import torch

    import scripts.predict_run as predict_run
    import scripts.train_rnn as train_rnn

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("E540 V6 requires exactly one visible CUDA device")
    device_name = torch.cuda.get_device_name(0)
    if "T4" not in device_name:
        raise RuntimeError(f"E540 V6 packet requires the reviewed Kaggle T4, observed {device_name}")
    hardware = {
        "device_name": device_name,
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "python": sys.version,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
    }
    write_json(RUNTIME / "hardware.json", hardware)
    profiles: dict[str, dict[str, object]] = {}
    for mode in ("linear", "ple"):
        arm = {
            "mode": mode,
            "run_name": RUN_NAMES[mode],
            "status": "profile_running",
            "profile_amp_initial_scale": AMP_INIT_SCALE,
            "fit_amp_initial_scale": AMP_INIT_SCALE,
            "argv_profile": trainer_argv(mode, profile=True),
            "argv_fit": trainer_argv(mode, profile=False),
            "argv_prediction": predict_argv(mode),
        }
        write_json(receipts / f"{mode}.json", arm)
        cohort["phase"] = f"profile_{mode}"
        write_json(RUNTIME / "cohort.json", cohort)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        phase_start = time.monotonic()
        try:
            output = call_main(
                train_rnn,
                "train_rnn.py",
                trainer_argv(mode, profile=True),
                logs / f"{mode}-profile.log",
            )
            stats = parse_trainer_output(output)
            step_wall_ms, evidence = validate_profile(stats)
            profile_wall = time.monotonic() - phase_start
            peak_reserved = int(torch.cuda.max_memory_reserved())
            peak_allocated = int(torch.cuda.max_memory_allocated())
            if peak_reserved > ALLOCATOR_LIMIT_BYTES:
                raise ValueError("profile allocator reservation exceeds the 4.3 GB operating target")
            record = {
                **stats,
                "step_wall_ms": step_wall_ms,
                "profile_wall_seconds": profile_wall,
                "profile_outer_batches": PROFILE_WARMUP_BATCHES + PROFILE_TIMED_BATCHES,
                "updates_per_outer_batch": UPDATES_PER_OUTER_BATCH,
                "profile_update_evidence": evidence,
                "cuda_peak_reserved_bytes": peak_reserved,
                "cuda_peak_allocated_bytes": peak_allocated,
                "process_peak_rss_bytes": resident_memory_bytes(),
                "amp_initial_scale": AMP_INIT_SCALE,
            }
            profiles[mode] = record
            arm.update({"status": "profile_pass", "profile": record})
            write_json(receipts / f"{mode}.json", arm)
        except BaseException as error:
            arm.update({"status": "profile_failed", "error": f"{type(error).__name__}: {error}"})
            write_json(receipts / f"{mode}.json", arm)
            cohort.update({"status": "profile_failed", "phase": f"profile_{mode}_failed"})
            write_json(RUNTIME / "cohort.json", cohort)
            raise
        cohort["arm_receipts"][mode] = f"receipts/{mode}.json"
        write_json(RUNTIME / "cohort.json", cohort)

    train_groups = subset["selected_groups"]["train"]
    valid_groups = subset["selected_groups"]["valid"]
    evaluation_groups = subset["evaluation_groups"]
    if (
        subset.get("selected_component_count") != 512
        or subset.get("evaluation_group_count") != 468
        or len(evaluation_groups) != 468
    ):
        raise ValueError("the mounted E540 V6 membership is not the frozen 512/468 contract")
    projected_seconds = elapsed_projection_seconds(
        time.monotonic() - started,
        profiles,
        len(train_groups),
        len(valid_groups),
        len(evaluation_groups),
    )
    total_fit_updates = expected_fit_updates(len(train_groups), len(valid_groups))
    gate = {
        "status": "pass" if projected_seconds <= worker_wall_seconds else "reject",
        "worker_elapsed_seconds_at_gate": time.monotonic() - started,
        "projected_total_worker_seconds": projected_seconds,
        "maximum_worker_seconds": worker_wall_seconds,
        "fitting_group_counts": {"train": len(train_groups), "valid": len(valid_groups)},
        "loader_membership": loader_membership,
        "membership_receipt": "membership.json",
        "fitting_component_count": membership["selected_component_count"],
        "fitting_component_sha256": membership["selected_component_sha256"],
        "fitting_group_sha256": membership["fit_group_sha256"],
        "fitting_members_sha256": membership["fit_members_sha256"],
        "evaluation_group_count": len(evaluation_groups),
        "expected_fit_optimizer_attempts_per_arm": total_fit_updates,
        "profile_attempts_per_arm": EXPECTED_PROFILE_ATTEMPTS,
        "profiles": {
            mode: {
                "step_wall_ms": profiles[mode]["step_wall_ms"],
                "peak_reserved_bytes": profiles[mode]["cuda_peak_reserved_bytes"],
                "amp_skipped_updates": profiles[mode]["profile_update_evidence"]["amp_skipped_updates"],
            }
            for mode in profiles
        },
        "rule": "profile both arms, require zero AMP skips and <=4.3GB allocator reservation, then conservatively project all fit and prediction work inside the cap",
    }
    write_json(RUNTIME / "profile_gate.json", gate)
    cohort["profile_gate"] = "profile_gate.json"
    if gate["status"] != "pass":
        cohort.update({"status": "profile_gate_rejected", "phase": "profile_gate_rejected"})
        write_json(RUNTIME / "cohort.json", cohort)
        raise RuntimeError(
            f"profile projection exceeds the {worker_wall_seconds}-second worker cap"
        )
    cohort.update({"status": "fit_running", "phase": "fit_linear"})
    write_json(RUNTIME / "cohort.json", cohort)

    for mode in ("linear", "ple"):
        arm_path = receipts / f"{mode}.json"
        arm = json.loads(arm_path.read_text(encoding="utf-8"))
        arm["status"] = "fit_running"
        write_json(arm_path, arm)
        phase_start = time.monotonic()
        try:
            output = call_main(
                train_rnn,
                "train_rnn.py",
                trainer_argv(mode, profile=False),
                logs / f"{mode}-fit.log",
            )
            fit_seconds = time.monotonic() - phase_start
            checkpoint_path = ROOT / "artifacts" / "runs" / RUN_NAMES[mode] / "model_ep0.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint_args = checkpoint.get("args", {})
            provenance = checkpoint.get("provenance", {})
            diagnostics = provenance.get("update_diagnostics", [])
            update_record = diagnostics[0] if len(diagnostics) == 1 else {}
            update = update_record.get("epoch", {})
            numeric = provenance.get("numeric_preprocessing", {})
            numeric_identity = validate_numeric_checkpoint(
                mode, numeric, subset, checkpoint.get("model", {})
            )
            fit_members = provenance.get("fit_row_groups", {})
            if (
                checkpoint.get("epoch") != 0
                or checkpoint.get("n_in") != 211
                or checkpoint_args.get("epochs") != 1
                or checkpoint_args.get("checkpoint_policy") != "fixed"
                or checkpoint_args.get("checkpoint_epoch") != 0
                or checkpoint_args.get("batch") != 2
                or checkpoint_args.get("chunk") != 256
                or checkpoint_args.get("loss") != "wmse"
                or checkpoint_args.get("loss_norm") != "chunk"
                or checkpoint_args.get("precision") != "fp16"
                or checkpoint_args.get("numeric_map") != mode
                or checkpoint_args.get("amp_init_scale") != AMP_INIT_SCALE
                or checkpoint_args.get("fit_subset_manifest_sha256")
                != hashlib.sha256((ROOT / SUBSET_MANIFEST).read_bytes()).hexdigest()
                or provenance.get("split_manifest_sha256")
                != subset.get("split_manifest_sha256")
                or fit_members.get("train") != train_groups
                or fit_members.get("valid") != valid_groups
                or numeric.get("mapped_input_width") != 1107
                or update.get("optimizer_attempts") != total_fit_updates
                or update.get("optimizer_completed_updates") != total_fit_updates
                or update.get("amp_skipped_updates") != 0
                or checkpoint_args.get("distill") != ""
                or checkpoint_args.get("distill_cache_root") != ""
                or checkpoint_args.get("propensity") != ""
                or checkpoint_args.get("prop_input") != ""
                or any(
                    checkpoint_args.get(key) != 0
                    for key in (
                        "aux_next", "aux_tier", "aux_mask", "aux_sign", "aux_quote",
                        "aux_struct", "distill_family_weight",
                    )
                )
            ):
                raise ValueError("fit checkpoint does not satisfy the fixed V5 epoch/AMP/update contract")
            arm.update(
                {
                    "status": "fit_pass",
                    "fit_wall_seconds": fit_seconds,
                    "fit_optimizer_diagnostics": update,
                    "checkpoint": {
                        "path": checkpoint_path.relative_to(ROOT).as_posix(),
                        "size": checkpoint_path.stat().st_size,
                        "sha256": sha256_file(checkpoint_path),
                        "epoch": checkpoint.get("epoch"),
                        "amp_initial_scale": checkpoint_args.get("amp_init_scale"),
                    },
                    "numeric_preprocessing": numeric_identity,
                    "fit_log": f"logs/{mode}-fit.log",
                }
            )
            write_json(arm_path, arm)
        except BaseException as error:
            arm.update({"status": "fit_failed", "error": f"{type(error).__name__}: {error}"})
            write_json(arm_path, arm)
            cohort.update({"status": "fit_failed", "phase": f"fit_{mode}_failed"})
            write_json(RUNTIME / "cohort.json", cohort)
            raise
        cohort["phase"] = "fit_ple" if mode == "linear" else "prediction_linear"
        write_json(RUNTIME / "cohort.json", cohort)
        if time.monotonic() - started > worker_wall_seconds:
            raise TimeoutError("E540 V6 worker exceeded its dynamically reduced deadline")

    for mode in ("linear", "ple"):
        arm_path = receipts / f"{mode}.json"
        arm = json.loads(arm_path.read_text(encoding="utf-8"))
        arm["status"] = "prediction_running"
        write_json(arm_path, arm)
        phase_start = time.monotonic()
        try:
            call_main(
                predict_run,
                "predict_run.py",
                predict_argv(mode),
                logs / f"{mode}-prediction.log",
            )
            prediction = ROOT / "cache" / f"pred_{RUN_NAMES[mode]}-eval_valid.npy"
            sidecar = prediction.with_suffix(".json")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            membership_bytes = struct.pack(f"<{len(evaluation_groups)}q", *evaluation_groups)
            expected_membership = hashlib.sha256(membership_bytes).hexdigest()
            prediction_digest = sha256_file(prediction)
            if (
                metadata.get("run") != RUN_NAMES[mode]
                or metadata.get("split") != "valid"
                or metadata.get("epoch") != 0
                or metadata.get("checkpoint_sha256") != arm["checkpoint"]["sha256"]
                or metadata.get("prediction_sha256") != prediction_digest
                or metadata.get("row_groups") != evaluation_groups
                or metadata.get("evaluation_membership_sha256") != expected_membership
                or metadata.get("fit_subset_manifest_sha256")
                != hashlib.sha256((ROOT / SUBSET_MANIFEST).read_bytes()).hexdigest()
                or metadata.get("inference_precision") != "fp32"
            ):
                raise ValueError("prediction sidecar differs from the frozen feature-only 468 group contract")
            arm.update(
                {
                    "status": "prediction_pass",
                    "prediction_wall_seconds": time.monotonic() - phase_start,
                    "prediction": {
                        "path": prediction.relative_to(ROOT).as_posix(),
                        "size": prediction.stat().st_size,
                        "sha256": prediction_digest,
                        "sidecar": sidecar.relative_to(ROOT).as_posix(),
                        "row_group_count": len(evaluation_groups),
                        "membership_sha256": expected_membership,
                        "inference_precision": "fp32",
                        "reader": "features only; target and is_scored columns not requested",
                    },
                    "prediction_log": f"logs/{mode}-prediction.log",
                }
            )
            write_json(arm_path, arm)
        except BaseException as error:
            arm.update({"status": "prediction_failed", "error": f"{type(error).__name__}: {error}"})
            write_json(arm_path, arm)
            cohort.update({"status": "prediction_failed", "phase": f"prediction_{mode}_failed"})
            write_json(RUNTIME / "cohort.json", cohort)
            raise
        cohort["phase"] = "prediction_ple" if mode == "linear" else "complete"
        write_json(RUNTIME / "cohort.json", cohort)
        if time.monotonic() - started > worker_wall_seconds:
            raise TimeoutError("E540 V6 worker exceeded its dynamically reduced deadline")
    cohort.update(
        {
            "status": "complete_ungraded",
            "phase": "complete",
            "fit_admitted_by_packet": True,
            "grade_owner": "root independent grader",
            "graded_members": 468,
            "public_forecast": None,
            "wall_seconds": time.monotonic() - started,
        }
    )
    write_json(RUNTIME / "cohort.json", cohort)


def main() -> None:
    """Start the explicitly scoped E540 V6 warm cohort after controller review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        run()
    except BaseException as error:
        if RUNTIME.is_dir():
            failure = f"{type(error).__name__}: {error}"
            identity_path = RUNTIME / "input_identity.json"
            if identity_path.is_file():
                identity = json.loads(identity_path.read_text(encoding="utf-8"))
                if identity.get("status") != "verified":
                    identity.update({"status": "failed", "failure": failure})
                    write_json(identity_path, identity)
            cohort_path = RUNTIME / "cohort.json"
            if cohort_path.is_file():
                cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
                if cohort.get("status") != "complete_ungraded":
                    cohort.update({"status": "failed", "failure": failure})
                    write_json(cohort_path, cohort)
            for mode in RUN_NAMES:
                receipt_path = RUNTIME / "receipts" / f"{mode}.json"
                if not receipt_path.is_file():
                    continue
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if receipt.get("status") == "not_started":
                    receipt["status"] = "not_started_due_to_cohort_failure"
                    receipt["failure"] = failure
                    write_json(receipt_path, receipt)
        raise


if __name__ == "__main__":
    main()
