#!/usr/bin/env python3
"""Reuse the verified V8 linear control; profile, fit and export E540 V9 PLE."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import resource
import shutil
import struct
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from e540_cli_preflight import normalize_entry_argv, parse_project_cli  # noqa: E402, I001

PACKET_ID = "E540_FP32_V9_REUSE_CONTROL"
INERT_SCALER_SCALE_ARGUMENT = 1.0
MAX_WALL_SECONDS = 5_200
PROJECTION_MARGIN_SECONDS = 300
RUNNER_FINALIZATION_RESERVE_SECONDS = 300
PROFILE_WARMUP_BATCHES = 1
PROFILE_TIMED_BATCHES = 3
UPDATES_PER_OUTER_BATCH = math.ceil(20_000 / 256)
EXPECTED_PROFILE_ATTEMPTS = (
    PROFILE_WARMUP_BATCHES + PROFILE_TIMED_BATCHES
) * UPDATES_PER_OUTER_BATCH
ALLOCATOR_LIMIT_BYTES = 4_300_000_000
RUN_NAMES = {
    "linear": "e540v8-linear-s540-fp32",
    "ple": "e540v9-ple-s540-fp32",
}
AUTHORIZATION_PATH = Path(
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/"
    "E540_V9_ROOT_CONDITIONAL_AUTHORIZATION.json"
)
CONTROL_DIRECTORY = Path("e540_control")
CONTROL_MANIFEST_PATH = CONTROL_DIRECTORY / "control-manifest.json"
CONTROL_PROVENANCE_PATH = CONTROL_DIRECTORY / "provenance.json"
CONTROL_HISTORY_PATH = CONTROL_DIRECTORY / "history.json"
CONTROL_MODEL_PATH = CONTROL_DIRECTORY / "model.pt"
CONTROL_EPOCH0_PATH = CONTROL_DIRECTORY / "model_ep0.pt"
V8_ORIGIN = {
    "source_commit": "942178ef647fd4f49c90d1f6846dbd927cfab4fa",
    "source_review_sha256": "34c2d33457035bf338d3191ae01b590f8593c9b31f226f99902558023a4052b4",
    "worker_sha256": "2a76f211620640b896f404d69817f991c7c35f7ec3c75129878aa388593efe39",
    "code_manifest_sha256": "c1396bb5fd29a0d4c85216bdfa63e3b9e84204b3369aeb98cd1634a80e1ed8fc",
    "recipe_sha256": "00f5ddfaa7d97f147d97d7cc4d7d2ec15a976c052985be12026686cfb266dedb",
    "conditional_authorization_sha256": "4823742074306346b934ac0474e029553318ad40df8ee2c20ecc79d55e0444a0",
    "terminal_receipt_sha256": "93a6956dffc9f12fd480b0e9374be9e67654767456f6a7edbfe51e4f680ad5f9",
    "result_sha256": "88a415b843966c3f8f1ef4e4551759f9698bdc5658479cfb6ca1c0814dc16493",
    "provenance_sha256": "ef342abcc11b94eabed66bd4d1e29009fc645079bb17bb95f5f63d83b7e82794",
}
V8_TRAINING_SOURCE_HASHES = {
    "scripts/train_rnn.py": "31b72e53ae81a771d6cff42d8a48a4c06cdc523fabc8c7d00e51d20710942853",
    "src/wnn/nn.py": "874fbf1ab5756c7dabe9e388d9d619c0a29c218d28db36a7f8747957df453181",
    "src/wnn/run_identity.py": "7033ea007b9c7dcf498d3ab1e31aaeacb5725b7009fe719e9071c05b99272a76",
    "src/wnn/numeric_ple.py": "229a7e0f032cd65bbf6700fa9372c08016b20e4719cb2f9eddd540ff8c468720",
}
POST_SNAPSHOT_PROVENANCE_FIELDS = {"checkpoint_sha256", "selected_epoch", "fitting_seconds"}
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
    """Extract trainer timing and update evidence for a disabled-AMP profile."""
    found: dict[str, object] = {}
    for line in output.splitlines():
        for prefix, key in (
            ("PROFILE ", "profile"),
            ("PROFILE_UPDATE_EVIDENCE ", "update_evidence"),
        ):
            if line.startswith(prefix):
                found[key] = json.loads(line[len(prefix) :])
    if set(found) != {"profile", "update_evidence"}:
        raise ValueError("trainer did not emit both profile and update evidence")
    return found


def validate_profile(record: dict[str, object]) -> tuple[float, dict[str, object]]:
    """Require exact FP32 update, loss, and pre-clip gradient evidence."""
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
        or evidence.get("grad_scaler_enabled") is not False
        or evidence.get("gradient_stats_steps") != EXPECTED_PROFILE_ATTEMPTS
        or evidence.get("gradient_stats_nonfinite_steps") != 0
        or evidence.get("gradient_check_basis") != "complete pre-clip GradStats gradient norms"
        or evidence.get("valid_for_throughput") is not True
    ):
        raise ValueError(
            "profile must complete 316 finite FP32 updates with complete finite GradStats"
        )
    return float(step_wall), evidence


def expected_fit_updates(train_groups: int, valid_groups: int, batch: int = 2) -> int:
    """Count the fixed-drop-last outer batches and 256-row recurrent updates."""
    outer_batches = (train_groups + valid_groups) // batch
    return outer_batches * UPDATES_PER_OUTER_BATCH


def validate_finite_tensors(
    named_tensors: Mapping[str, object], is_finite: Callable[[object], bool]
) -> int:
    """Reject empty or nonfinite model tensors and return the checked count."""
    if not named_tensors:
        raise ValueError("checkpoint contains no model tensors to validate")
    for name, tensor in named_tensors.items():
        if not is_finite(tensor):
            raise ValueError(f"checkpoint tensor is nonfinite: {name}")
    return len(named_tensors)


def checkpoint_named_tensors(
    checkpoint: Mapping[str, object], *, snapshot_name: str
) -> dict[str, object]:
    """Collect model and fixed readout tensors from one exported snapshot."""
    state = checkpoint.get("model")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{snapshot_name} checkpoint has no model state mapping")
    named_tensors = {f"{snapshot_name}.model.{name}": value for name, value in state.items()}
    readout = checkpoint.get("readout")
    if readout is not None:
        if not isinstance(readout, (list, tuple)):
            raise ValueError(f"{snapshot_name} checkpoint readout is malformed")
        named_tensors.update(
            {f"{snapshot_name}.readout.{index}": value for index, value in enumerate(readout)}
        )
    return named_tensors


def validate_finite_prediction_array(
    predictions: object, expected_shape: tuple[int, int, int]
) -> dict[str, object]:
    """Require an exact-shape FP32 prediction array with only finite values."""
    import numpy as np

    shape = tuple(getattr(predictions, "shape", ()))
    dtype = getattr(predictions, "dtype", None)
    if shape != expected_shape or dtype != np.dtype(np.float32):
        raise ValueError("prediction output has the wrong shape or is not FP32")
    for start in range(0, expected_shape[0], 8):
        if not np.isfinite(predictions[start : start + 8]).all():
            raise ValueError("prediction output contains nonfinite values")
    return {"shape": list(shape), "dtype": str(dtype), "finite": True}


def validate_fit_update_contract(update: Mapping[str, object], expected_attempts: int) -> None:
    """Require exact FP32 update counts and complete finite fit GradStats."""
    gradient_stats = update.get("grad_stats")
    if not isinstance(gradient_stats, Mapping):
        raise ValueError("fit update diagnostics are missing GradStats")
    if (
        update.get("optimizer_attempts") != expected_attempts
        or update.get("optimizer_completed_updates") != expected_attempts
        or update.get("amp_skipped_updates") != 0
        or gradient_stats.get("steps") != expected_attempts
        or gradient_stats.get("nonfinite_steps") != 0
    ):
        raise ValueError("fit update diagnostics violate the exact finite FP32 update contract")


def fit_update_from_provenance_record(
    update_record: Mapping[str, object],
) -> dict[str, object]:
    """Combine trainer epoch counters with their sibling GradStats receipt."""
    epoch = update_record.get("epoch")
    if not isinstance(epoch, Mapping):
        raise ValueError("fit update diagnostics are missing epoch counters")
    return {**epoch, "grad_stats": update_record.get("grad_stats")}


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
        raise ValueError(
            "checkpoint numeric preprocessing is not bound to the frozen fit membership"
        )
    if mode == "linear":
        if (
            numeric.get("control_definition")
            != "same K+1 slots; only raw bypass coordinate is active"
        ):
            raise ValueError("linear control numeric-map definition differs from E540 V8")
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
    ple_profile: Mapping[str, object],
    train_group_count: int,
    valid_group_count: int,
    evaluation_group_count: int,
) -> float:
    """Project one PLE fit and two 468-group exports from the fresh PLE profile."""
    outer_wall = float(ple_profile["step_wall_ms"]) / 1000.0
    fit_batches = (train_group_count + valid_group_count) // 2
    fit_seconds = fit_batches * outer_wall * 1.35
    prediction_seconds = 2 * math.ceil(evaluation_group_count / 16) * outer_wall * 0.75
    return elapsed + fit_seconds + prediction_seconds + PROJECTION_MARGIN_SECONDS


def worker_wall_budget_seconds(kernel_deadline_monotonic: float, *, now_monotonic: float) -> int:
    """Cap the worker at 5,200s while preserving runtime for runner finalization."""
    remaining = math.floor(
        kernel_deadline_monotonic - now_monotonic - RUNNER_FINALIZATION_RESERVE_SECONDS
    )
    return max(0, min(MAX_WALL_SECONDS, remaining))


def trainer_argv(mode: str, *, profile: bool) -> list[str]:
    """Return the explicit original FP32 recipe for the PLE candidate."""
    argv = [
        "scripts/train_rnn.py",
        "--device",
        "cuda",
        "--name",
        RUN_NAMES[mode],
        "--numeric-map",
        mode,
        "--ple-intervals",
        "8",
        "--seed",
        "540",
        "--train-source",
        "train+valid",
        "--split-manifest",
        SPLIT_MANIFEST.as_posix(),
        "--fit-subset-manifest",
        SUBSET_MANIFEST.as_posix(),
        "--hidden",
        "128",
        "--layers",
        "1",
        "--dropout",
        "0",
        "--batch",
        "2",
        "--chunk",
        "256",
        "--epochs",
        "1",
        "--loss",
        "wmse",
        "--loss-norm",
        "chunk",
        "--optim",
        "adamw",
        "--lr",
        "0.001",
        "--wd",
        "0.01",
        "--sched",
        "onecycle",
        "--precision",
        "fp32",
        "--amp-init-scale",
        "1",
        "--grad-stats",
        "--batch-order",
        "seeded",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "0",
        "--no-eval-during-fit",
        "--deterministic",
        "--distill",
        "",
        "--distill-cache-root",
        "",
        "--distill-family-weight",
        "0",
        "--propensity",
        "",
        "--prop-input",
        "",
        "--prop-power",
        "1",
        "--prop-eps",
        "0",
        "--aux-next",
        "0",
        "--aux-tier",
        "0",
        "--aux-mask",
        "0",
        "--aux-sign",
        "0",
        "--aux-quote",
        "0",
        "--aux-struct",
        "0",
        "--aux",
        "",
        "--aux-weight",
        "0",
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
        "--epoch",
        "0",
        "--inference-precision",
        "fp32",
        "--output-name",
        f"{RUN_NAMES[mode]}-eval",
        "--fit-subset-manifest",
        SUBSET_MANIFEST.as_posix(),
    ]


def call_main(module: object, script_name: str, argv: list[str], log_path: Path) -> str:
    """Call a project entry point in-process to reuse validated input digests."""
    capture = io.StringIO()
    old_argv = sys.argv
    try:
        cli_arguments = normalize_entry_argv(script_name, argv)
        sys.argv = [script_name, *cli_arguments]
        with (
            log_path.open("w", encoding="utf-8") as log,
            contextlib.redirect_stdout(Tee(log, capture)),
        ):
            module.main()
    finally:
        sys.argv = old_argv
    return capture.getvalue()


def validate_recipe_cli() -> list[dict[str, object]]:
    """Parse the PLE fit/profile and both prediction commands before data hashing."""
    trainer_path = ROOT / "scripts/train_rnn.py"
    dynamics_path = ROOT / "src/wnn/training_dynamics.py"
    predictor_path = ROOT / "scripts/predict_run.py"
    checks: list[dict[str, object]] = []
    expected_common = {
        "device": "cuda",
        "seed": 540,
        "train_source": "train+valid",
        "hidden": 128,
        "layers": 1,
        "dropout": 0.0,
        "batch": 2,
        "chunk": 256,
        "ple_intervals": 8,
        "epochs": 1,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 0,
        "no_eval_during_fit": True,
        "deterministic": True,
        "batch_order": "seeded",
        "optim": "adamw",
        "lr": 0.001,
        "wd": 0.01,
        "sched": "onecycle",
        "precision": "fp32",
        "amp_init_scale": INERT_SCALER_SCALE_ARGUMENT,
        "grad_stats": True,
        "loss": "wmse",
        "loss_norm": "chunk",
        "distill": "",
        "distill_cache_root": "",
        "distill_family_weight": 0.0,
        "propensity": "",
        "prop_input": "",
        "prop_power": 1.0,
        "prop_eps": 0.0,
        "aux_next": 0.0,
        "aux_tier": 0.0,
        "aux_mask": 0.0,
        "aux_sign": 0.0,
        "aux_quote": 0.0,
        "aux_struct": 0.0,
        "aux": "",
        "aux_weight": 0.0,
    }
    for stage in ("profile", "fit"):
        logged = trainer_argv("ple", profile=stage == "profile")
        cli = normalize_entry_argv("train_rnn.py", logged)
        parsed = parse_project_cli(trainer_path, cli, parser_name="ap", dynamics_path=dynamics_path)
        expected_steps = PROFILE_TIMED_BATCHES if stage == "profile" else 0
        parsed_common = {key: getattr(parsed, key, None) for key in expected_common}
        if (
            parsed.name != RUN_NAMES["ple"]
            or parsed.numeric_map != "ple"
            or parsed_common != expected_common
            or parsed.profile_steps != expected_steps
            or (stage == "profile" and parsed.profile_warmup != PROFILE_WARMUP_BATCHES)
        ):
            raise ValueError(f"E540 PLE {stage} CLI differs from the frozen recipe")
        checks.append(
            {
                "entry_point": "scripts/train_rnn.py",
                "mode": "ple",
                "stage": stage,
                "status": "parsed",
                "script_token_removed": logged[1:] == cli,
            }
        )
    for mode in ("linear", "ple"):
        logged = predict_argv(mode)
        cli = normalize_entry_argv("predict_run.py", logged)
        parsed = parse_project_cli(predictor_path, cli, parser_name="parser")
        if (
            parsed.name != RUN_NAMES[mode]
            or parsed.split != "valid"
            or parsed.epoch != 0
            or parsed.inference_precision != "fp32"
            or parsed.fit_subset_manifest != SUBSET_MANIFEST
            or parsed.output_name != f"{RUN_NAMES[mode]}-eval"
            or parsed.gate_mode != "soft"
        ):
            raise ValueError(f"E540 {mode} prediction CLI differs from the frozen recipe")
        checks.append(
            {
                "entry_point": "scripts/predict_run.py",
                "mode": mode,
                "stage": "prediction",
                "status": "parsed",
                "script_token_removed": logged[1:] == cli,
            }
        )
    return checks


def same_group_membership(left: object, right: object) -> bool:
    """Compare ordered group lists as exact unique membership, allowing source row order."""
    if not isinstance(left, list) or not isinstance(right, list):
        return False
    if any(type(value) is not int for value in (*left, *right)):
        return False
    return len(left) == len(right) == len(set(left)) and set(left) == set(right)


def validate_conditional_authorization(
    receipt_path: Path, expected: dict[str, object]
) -> dict[str, object]:
    """Require root's packet/control-bound PLE fit authorization before data hashing."""
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
        "control_manifest_sha256",
        "control_args_sha256",
        "control_model_sha256",
        "control_epoch0_sha256",
        "control_provenance_sha256",
    ):
        if receipt.get(key) != expected.get(key):
            raise ValueError(f"root conditional authorization does not bind {key}")
    authorization_id = receipt.get("authorization_id")
    if not isinstance(authorization_id, str) or not authorization_id.strip():
        raise ValueError("root conditional authorization requires a nonempty authorization_id")
    return receipt


def _conditional_authorization_contract() -> dict[str, object]:
    """Load packet, subset and reused-control identity from the runner's frozen spec."""
    fields = {
        "packet_id": "WNN_E540_PACKET_ID",
        "code_manifest_sha256": "WNN_E540_CODE_MANIFEST_SHA256",
        "recipe_sha256": "WNN_E540_RECIPE_SHA256",
        "subset_manifest_sha256": "WNN_E540_SUBSET_SHA256",
        "control_manifest_sha256": "WNN_E540_CONTROL_MANIFEST_SHA256",
        "control_args_sha256": "WNN_E540_CONTROL_ARGS_SHA256",
        "control_model_sha256": "WNN_E540_CONTROL_MODEL_SHA256",
        "control_epoch0_sha256": "WNN_E540_CONTROL_EPOCH0_SHA256",
        "control_provenance_sha256": "WNN_E540_CONTROL_PROVENANCE_SHA256",
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


def canonical_json_sha256(value: object) -> str:
    """Hash JSON metadata with one stable encoding for packet/control comparisons."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_control_provenance(
    sibling_provenance: Mapping[str, object],
    control_manifest: Mapping[str, object],
    subset: Mapping[str, object],
    *,
    expected_manifest_sha256: str,
    expected_args_sha256: str,
) -> dict[str, object]:
    """Validate the completed V8 control's real sibling provenance before data hashing."""
    producer = control_manifest.get("producer")
    files = control_manifest.get("files")
    if not isinstance(producer, Mapping) or not isinstance(files, Mapping):
        raise ValueError("reused V8 control manifest is malformed")
    if (
        control_manifest.get("schema") != "wnn.e540-v8-linear-control.v1"
        or len(expected_manifest_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_manifest_sha256)
        or control_manifest.get("args_sha256") != expected_args_sha256
        or producer.get("source_commit") != V8_ORIGIN["source_commit"]
        or producer.get("source_review_sha256") != V8_ORIGIN["source_review_sha256"]
        or producer.get("worker_sha256") != V8_ORIGIN["worker_sha256"]
        or producer.get("code_manifest_sha256") != V8_ORIGIN["code_manifest_sha256"]
        or producer.get("recipe_sha256") != V8_ORIGIN["recipe_sha256"]
        or producer.get("conditional_authorization_sha256")
        != V8_ORIGIN["conditional_authorization_sha256"]
        or producer.get("terminal_receipt_sha256") != V8_ORIGIN["terminal_receipt_sha256"]
        or producer.get("result_sha256") != V8_ORIGIN["result_sha256"]
        or producer.get("provenance_sha256") != V8_ORIGIN["provenance_sha256"]
    ):
        raise ValueError("reused control source, recipe, or manifest identity differs from V8")
    if set(files) != {"model.pt", "model_ep0.pt", "provenance.json", "history.json"}:
        raise ValueError("reused control manifest contains an unexpected asset set")

    args = sibling_provenance.get("args")
    if not isinstance(args, Mapping) or canonical_json_sha256(dict(args)) != expected_args_sha256:
        raise ValueError("reused V8 control trainer arguments differ from the frozen identity")
    if (
        args.get("name") != RUN_NAMES["linear"]
        or args.get("seed") != 540
        or args.get("numeric_map") != "linear"
        or args.get("precision") != "fp32"
        or args.get("loss") != "wmse"
        or args.get("loss_norm") != "chunk"
        or args.get("epochs") != 1
        or args.get("checkpoint_policy") != "fixed"
        or args.get("checkpoint_epoch") != 0
        or args.get("fit_subset_manifest_sha256") != control_manifest.get("subset_manifest_sha256")
    ):
        raise ValueError("reused V8 control arguments do not match the fixed recipe")

    if (
        sibling_provenance.get("seed") != 540
        or sibling_provenance.get("selected_epoch") != 0
        or sibling_provenance.get("fit_subset_manifest_sha256")
        != control_manifest.get("subset_manifest_sha256")
        or sibling_provenance.get("fit_subset_members_sha256")
        != subset.get("selected_members_sha256")
        or sibling_provenance.get("split_manifest_sha256") != subset.get("split_manifest_sha256")
        or sibling_provenance.get("fit_row_groups") != subset.get("selected_groups")
        or not same_group_membership(
            sibling_provenance.get("evaluation_row_groups"), subset.get("evaluation_groups")
        )
    ):
        raise ValueError("reused V8 control provenance differs from the frozen 512/468 membership")

    numeric = sibling_provenance.get("numeric_preprocessing")
    if not isinstance(numeric, Mapping):
        raise ValueError("reused V8 control numeric-preprocessing receipt is missing")
    if (
        numeric.get("mode") != "linear"
        or numeric.get("n_features") != 112
        or numeric.get("base_input_width") != 211
        or numeric.get("mapped_input_width") != 1107
        or numeric.get("n_intervals") != 8
        or numeric.get("fit_members") != subset.get("selected_groups")
        or numeric.get("fit_members_sha256") != subset.get("selected_members_sha256")
    ):
        raise ValueError("reused V8 linear preprocessing receipt differs from frozen membership")

    source_fields = {
        "scripts/train_rnn.py": sibling_provenance.get("trainer_source_sha256"),
        "src/wnn/nn.py": sibling_provenance.get("model_source_sha256"),
        "src/wnn/run_identity.py": sibling_provenance.get("run_identity_source_sha256"),
        "src/wnn/numeric_ple.py": numeric.get("module_source_sha256"),
    }
    if control_manifest.get("training_source_hashes") != V8_TRAINING_SOURCE_HASHES:
        raise ValueError("reused V8 manifest training-source map differs from reviewed source")
    for relative, expected in V8_TRAINING_SOURCE_HASHES.items():
        if source_fields.get(relative) != expected or sha256_file(ROOT / relative) != expected:
            raise ValueError(f"reused V8 control source identity differs: {relative}")

    diagnostics = sibling_provenance.get("update_diagnostics")
    if (
        not isinstance(diagnostics, list)
        or len(diagnostics) != 1
        or not isinstance(diagnostics[0], Mapping)
    ):
        raise ValueError("reused V8 control has no single epoch-0 update diagnostic")
    update = fit_update_from_provenance_record(diagnostics[0])
    validate_fit_update_contract(update, 72_996)
    fit_rows = sibling_provenance.get("fit_row_groups")
    evaluation_rows = sibling_provenance.get("evaluation_row_groups")
    expected_group_counts = (
        {key: len(value) for key, value in fit_rows.items()}
        if isinstance(fit_rows, Mapping)
        else None
    )
    expected_update_record = {
        "epoch": diagnostics[0].get("epoch"),
        "cumulative": diagnostics[0].get("cumulative"),
        "sample_order_identity": diagnostics[0].get("sample_order_identity"),
        "normalization_identity": diagnostics[0].get("normalization_identity"),
        "grad_stats": diagnostics[0].get("grad_stats"),
    }
    if (
        control_manifest.get("run_name") != RUN_NAMES["linear"]
        or control_manifest.get("subset_manifest_sha256")
        != sibling_provenance.get("fit_subset_manifest_sha256")
        or control_manifest.get("fit_members_sha256") != subset.get("selected_members_sha256")
        or control_manifest.get("split_manifest_sha256") != subset.get("split_manifest_sha256")
        or control_manifest.get("fit_group_counts") != expected_group_counts
        or control_manifest.get("evaluation_group_count")
        != (len(evaluation_rows) if isinstance(evaluation_rows, list) else None)
        or control_manifest.get("update_diagnostics") != expected_update_record
    ):
        raise ValueError("reused V8 manifest differs from the frozen fit/update/membership receipt")
    cumulative = sibling_provenance.get("cumulative_updates")
    if (
        not isinstance(cumulative, Mapping)
        or cumulative.get("optimizer_completed_updates") != 72_996
    ):
        raise ValueError("reused V8 control cumulative update count is not exact")
    checkpoint_hashes = sibling_provenance.get("checkpoint_sha256")
    if not isinstance(checkpoint_hashes, Mapping) or set(checkpoint_hashes) != {
        "model.pt",
        "model_ep0.pt",
    }:
        raise ValueError("sibling provenance must name both fixed control snapshots")
    for filename in ("model.pt", "model_ep0.pt"):
        value = checkpoint_hashes.get(filename)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("sibling provenance contains an invalid checkpoint digest")
        expected_file = files.get(filename)
        if not isinstance(expected_file, Mapping) or expected_file.get("sha256") != value:
            raise ValueError("control manifest and sibling provenance checkpoint hashes differ")

    return {
        "status": "verified",
        "run_name": args["name"],
        "args_sha256": expected_args_sha256,
        "fit_subset_manifest_sha256": sibling_provenance["fit_subset_manifest_sha256"],
        "fit_subset_members_sha256": sibling_provenance["fit_subset_members_sha256"],
        "fit_group_counts": {
            key: len(value) for key, value in sibling_provenance["fit_row_groups"].items()
        },
        "evaluation_group_count": len(sibling_provenance["evaluation_row_groups"]),
        "optimizer_completed_updates": update["optimizer_completed_updates"],
        "amp_skipped_updates": update["amp_skipped_updates"],
        "gradient_stats_steps": update["grad_stats"]["steps"],
        "gradient_stats_nonfinite_steps": update["grad_stats"]["nonfinite_steps"],
        "sample_order_identity": diagnostics[0]["sample_order_identity"],
        "checkpoint_sha256": dict(checkpoint_hashes),
        "source_sha256": source_fields,
    }


def verify_control_assets(
    subset: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Hash tiny private control files and bind their metadata before data identity reads."""
    manifest_path = ROOT / CONTROL_MANIFEST_PATH
    provenance_path = ROOT / CONTROL_PROVENANCE_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected_manifest = os.environ.get("WNN_E540_CONTROL_MANIFEST_SHA256", "")
    expected_args = os.environ.get("WNN_E540_CONTROL_ARGS_SHA256", "")
    if sha256_file(manifest_path) != expected_manifest:
        raise ValueError("reused control manifest hash differs from the frozen packet")
    expected_files = {
        "model.pt": (
            ROOT / CONTROL_MODEL_PATH,
            os.environ.get("WNN_E540_CONTROL_MODEL_SHA256", ""),
        ),
        "model_ep0.pt": (
            ROOT / CONTROL_EPOCH0_PATH,
            os.environ.get("WNN_E540_CONTROL_EPOCH0_SHA256", ""),
        ),
        "provenance.json": (
            provenance_path,
            os.environ.get("WNN_E540_CONTROL_PROVENANCE_SHA256", ""),
        ),
        "history.json": (ROOT / CONTROL_HISTORY_PATH, None),
    }
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, Mapping):
        raise ValueError("reused control manifest has no file identity map")
    for filename, (path, bound_hash) in expected_files.items():
        entry = manifest_files.get(filename)
        if not isinstance(entry, Mapping):
            raise ValueError(f"reused control asset is missing its {filename} descriptor")
        observed_hash = sha256_file(path)
        if (
            path.stat().st_size != entry.get("size_bytes")
            or observed_hash != entry.get("sha256")
            or (bound_hash is not None and observed_hash != bound_hash)
        ):
            raise ValueError(f"reused control asset failed its bound identity: {filename}")
    if sha256_file(provenance_path) != expected_files["provenance.json"][1]:
        raise ValueError("reused control sibling provenance hash differs from the frozen packet")
    summary = validate_control_provenance(
        provenance,
        manifest,
        subset,
        expected_manifest_sha256=expected_manifest,
        expected_args_sha256=expected_args,
    )
    return manifest, provenance, summary


def validate_checkpoint_snapshot_metadata(
    checkpoint: Mapping[str, object], sibling_provenance: Mapping[str, object]
) -> None:
    """Enforce the trainer's real checkpoint/sibling-provenance serialization contract."""
    args = checkpoint.get("args")
    embedded = checkpoint.get("provenance")
    sibling_args = sibling_provenance.get("args")
    if not isinstance(args, Mapping) or args != sibling_args:
        raise ValueError("checkpoint args differ from sibling trainer provenance")
    if not isinstance(embedded, Mapping):
        raise ValueError("checkpoint-embedded provenance is missing")
    expected_embedded = {
        key: value
        for key, value in sibling_provenance.items()
        if key not in POST_SNAPSHOT_PROVENANCE_FIELDS
    }
    if dict(embedded) != expected_embedded:
        raise ValueError("checkpoint-embedded provenance differs from sibling trainer provenance")


def validate_history(
    history: object,
    sibling_provenance: Mapping[str, object],
    *,
    expected_updates: int,
) -> dict[str, object]:
    """Check the fixed epoch-0 history and finite training duration without evaluation labels."""
    if not isinstance(history, list) or len(history) != 1 or not isinstance(history[0], Mapping):
        raise ValueError("fixed epoch-0 history must contain exactly one record")
    row = history[0]
    seconds = row.get("train_seconds")
    if (
        row.get("epoch") != 0
        or row.get("evaluation") != "withheld"
        or isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(float(seconds))
        or float(seconds) <= 0
    ):
        raise ValueError("fixed epoch-0 history is malformed or evaluated held-out data")
    fitting_seconds = sibling_provenance.get("fitting_seconds")
    if (
        isinstance(fitting_seconds, bool)
        or not isinstance(fitting_seconds, (int, float))
        or not math.isfinite(float(fitting_seconds))
        or not math.isclose(float(seconds), float(fitting_seconds), rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError("history training duration differs from sibling provenance")
    update = row.get("update_diagnostics")
    if not isinstance(update, Mapping):
        raise ValueError("fixed epoch-0 history omits update diagnostics")
    provenance_rows = sibling_provenance.get("update_diagnostics")
    if not isinstance(provenance_rows, list) or len(provenance_rows) != 1:
        raise ValueError("sibling provenance omits the single fixed epoch update")
    expected = provenance_rows[0]
    if not isinstance(expected, Mapping):
        raise ValueError("sibling update diagnostics are malformed")
    for key in (
        "epoch",
        "cumulative",
        "sample_order_identity",
        "normalization_identity",
        "grad_stats",
    ):
        if update.get(key) != expected.get(key):
            raise ValueError(f"history and provenance differ in {key}")
    merged = fit_update_from_provenance_record(
        {"epoch": update.get("epoch"), "grad_stats": update.get("grad_stats")}
    )
    validate_fit_update_contract(merged, expected_updates)
    return {
        "epoch": 0,
        "train_seconds": float(seconds),
        "updates": expected_updates,
        "evaluation": "withheld",
    }


def resident_memory_bytes() -> int:
    """Return this process' peak resident set size in portable Linux units."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * (1024 if sys.platform != "darwin" else 1)


def verify_mounted_data() -> None:
    """Hash all deferred external inputs once and persist observations before Torch import."""
    deferred = {
        value for value in os.environ.get("WNN_DEFERRED_IDENTITY_PATHS", "").split(",") if value
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

        def persist_observation(
            size: int,
            digest: str | None,
            *,
            observed: dict[str, object] = observed,
            identity: dict[str, object] = identity,
            entry: dict[str, object] = entry,
        ) -> None:
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
            raise ValueError(f"mounted E540 V8 subset differs from deterministic selection: {key}")
    if subset.get("selected_component_count") != 512 or subset.get("evaluation_group_count") != 468:
        raise ValueError(
            "E540 V8 requires exactly 512 fitting components and 468 evaluation groups"
        )
    if subset.get("selection_seed") != 540:
        raise ValueError("E540 V8 component selection seed differs from the frozen recipe")

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
        raise ValueError("E540 V8 group and component lists must be sorted and unique")
    campaign.assert_fit_groups(train_groups, valid_groups)
    if evaluation_groups != sorted(map(int, campaign.evaluation_groups)):
        raise ValueError("E540 V8 evaluation membership differs from the frozen split")

    component_of = {
        (split, int(group)): int(component) for split, group, component in campaign.components
    }
    fitting_component_ids = {component_of[("train", group)] for group in train_groups} | {
        component_of[("valid", group)] for group in valid_groups
    }
    if fitting_component_ids != set(selected_components):
        raise ValueError("E540 V8 selected component list differs from its fitting groups")
    for split, selected in (("train", train_groups), ("valid", valid_groups)):
        complete = sorted(
            group
            for (member_split, group), component in component_of.items()
            if member_split == split and component in fitting_component_ids
        )
        if selected != complete:
            raise ValueError(f"E540 V8 fitting groups are not component-complete for {split}")
        digest = hashlib.sha256(np.asarray(selected, dtype="<i8").tobytes()).hexdigest()
        if subset["selected_group_sha256"].get(split) != digest:
            raise ValueError(f"E540 V8 {split} fitting group digest mismatch")

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
        "realized_component_fraction": len(selected_components)
        / int(subset["eligible_component_count"]),
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
    """Validate the reused control, then profile/fit PLE and export both frozen arms."""
    started = time.monotonic()
    deadline_value = os.environ.get("WNN_KERNEL_DEADLINE_MONOTONIC")
    if deadline_value is None:
        raise RuntimeError("E540 V9 requires the bounded Kaggle kernel bootstrap")
    try:
        kernel_deadline = float(deadline_value)
    except ValueError as error:
        raise RuntimeError("invalid Kaggle kernel wall deadline") from error
    if not math.isfinite(kernel_deadline):
        raise RuntimeError("invalid Kaggle kernel wall deadline")
    worker_wall_seconds = worker_wall_budget_seconds(kernel_deadline, now_monotonic=started)
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
        subset_preview = json.loads((ROOT / SUBSET_MANIFEST).read_text(encoding="utf-8"))
        control_manifest, control_provenance, control_summary = verify_control_assets(
            subset_preview
        )
        control_history = json.loads((ROOT / CONTROL_HISTORY_PATH).read_text(encoding="utf-8"))
        history_summary = validate_history(
            control_history, control_provenance, expected_updates=72_996
        )
    except BaseException as error:
        preflight.update(
            {
                "status": "failed_before_mounted_data_hashing",
                "failure": f"{type(error).__name__}: {error}",
            }
        )
        write_json(preflight_path, preflight)
        raise
    preflight.update(
        {
            "status": "control_assets_verified_before_mounted_data_hashing",
            "authorization_id": authorization["authorization_id"],
            "control_manifest_sha256": os.environ["WNN_E540_CONTROL_MANIFEST_SHA256"],
            "control_args_sha256": control_summary["args_sha256"],
            "control_checkpoint_sha256": control_summary["checkpoint_sha256"],
            "control_history": history_summary,
        }
    )
    write_json(preflight_path, preflight)

    receipts = RUNTIME / "receipts"
    logs = RUNTIME / "logs"
    logs.mkdir()
    control_receipt: dict[str, object] = {
        "mode": "linear_reused_control",
        "run_name": RUN_NAMES["linear"],
        "status": "control_metadata_pass",
        "source_packet_id": "E540_FP32_V8",
        "source_commit": V8_ORIGIN["source_commit"],
        "checkpoint": control_summary["checkpoint_sha256"],
        "provenance_sha256": os.environ["WNN_E540_CONTROL_PROVENANCE_SHA256"],
        "history": history_summary,
        "fit_diagnostics": {
            key: control_summary[key]
            for key in (
                "optimizer_completed_updates",
                "amp_skipped_updates",
                "gradient_stats_steps",
                "gradient_stats_nonfinite_steps",
                "fit_group_counts",
                "evaluation_group_count",
                "sample_order_identity",
            )
        },
        "prediction_argv": predict_argv("linear"),
    }
    write_json(receipts / "linear-control.json", control_receipt)
    ple_receipt: dict[str, object] = {
        "mode": "ple",
        "run_name": RUN_NAMES["ple"],
        "status": "not_started",
        "precision": "fp32",
        "grad_scaler_enabled": False,
        "amp_init_scale_argument": INERT_SCALER_SCALE_ARGUMENT,
        "amp_init_scale_applied": False,
        "argv_profile": trainer_argv("ple", profile=True),
        "argv_fit": trainer_argv("ple", profile=False),
        "argv_prediction": predict_argv("ple"),
    }
    write_json(receipts / "ple.json", ple_receipt)
    cohort: dict[str, object] = {
        "packet_id": PACKET_ID,
        "status": "input_identity_pending",
        "fit_admitted_by_packet": False,
        "provider_review_required": True,
        "maximum_worker_seconds": worker_wall_seconds,
        "declared_worker_ceiling_seconds": MAX_WALL_SECONDS,
        "runner_finalization_reserve_seconds": RUNNER_FINALIZATION_RESERVE_SECONDS,
        "precision": "fp32",
        "grad_scaler_enabled": False,
        "amp_init_scale_argument": INERT_SCALER_SCALE_ARGUMENT,
        "amp_init_scale_applied": False,
        "phase": "authorized_control_assets_verified",
        "started_unix": time.time(),
        "control_receipt": "receipts/linear-control.json",
        "ple_receipt": "receipts/ple.json",
        "v8_control_reused": True,
    }
    write_json(RUNTIME / "cohort.json", cohort)

    verify_mounted_data()
    subset, membership = verify_frozen_membership()
    train_groups = subset["selected_groups"]["train"]
    valid_groups = subset["selected_groups"]["valid"]
    evaluation_groups = subset["evaluation_groups"]
    selected_sequence_count = len(train_groups) + len(valid_groups)
    if (
        subset.get("selected_component_count") != 512
        or subset.get("evaluation_group_count") != 468
        or len(evaluation_groups) != 468
        or selected_sequence_count != 1_849
    ):
        raise ValueError("mounted V9 membership differs from the frozen 512/468 contract")
    loader_membership = {
        "status": "derived_from_frozen_batch_size_and_drop_last_contract; actual_indices_not_captured",
        "selected_sequence_count": selected_sequence_count,
        "batch_size": 2,
        "drop_last": True,
        "expected_consumed_sequence_count_per_epoch": selected_sequence_count
        - selected_sequence_count % 2,
        "expected_dropped_sequence_count_per_epoch": selected_sequence_count % 2,
        "dropped_sequence_identity": None,
        "paired_order_status": "will_compare stored control sampler identity to observed PLE identity",
    }
    cohort.update(
        {
            "phase": "membership_verified_before_profile_and_fit",
            "membership": "membership.json",
            "membership_receipt": membership,
            "loader_membership": loader_membership,
        }
    )
    write_json(RUNTIME / "cohort.json", cohort)

    import torch

    import scripts.predict_run as predict_run
    import scripts.train_rnn as train_rnn

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("E540 V9 requires exactly one visible CUDA device")
    device_name = torch.cuda.get_device_name(0)
    if "T4" not in device_name:
        raise RuntimeError(f"E540 V9 requires the reviewed Kaggle T4, observed {device_name}")
    hardware = {
        "device_name": device_name,
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "python": sys.version,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
    }
    write_json(RUNTIME / "hardware.json", hardware)

    control_dir = ROOT / "artifacts" / "runs" / RUN_NAMES["linear"]
    if control_dir.exists():
        raise FileExistsError("refusing to overwrite a preexisting V8 control run directory")
    control_checkpoint = torch.load(
        ROOT / CONTROL_EPOCH0_PATH, map_location="cpu", weights_only=False
    )
    control_model = torch.load(ROOT / CONTROL_MODEL_PATH, map_location="cpu", weights_only=False)
    validate_checkpoint_snapshot_metadata(control_checkpoint, control_provenance)
    validate_checkpoint_snapshot_metadata(control_model, control_provenance)
    if (
        control_checkpoint.get("epoch") != 0
        or control_model.get("epoch") != 0
        or control_checkpoint.get("n_in") != 211
        or control_model.get("n_in") != 211
        or control_checkpoint.get("args") != control_provenance.get("args")
        or control_model.get("args") != control_provenance.get("args")
    ):
        raise ValueError("reused V8 model snapshots are not the exact fixed epoch-0 control")
    control_state = control_checkpoint.get("model")
    selected_control_state = control_model.get("model")
    if (
        not isinstance(control_state, Mapping)
        or not isinstance(selected_control_state, Mapping)
        or control_state.keys() != selected_control_state.keys()
        or any(
            not torch.equal(control_state[name], selected_control_state[name])
            for name in control_state
        )
    ):
        raise ValueError("reused V8 model.pt differs from its fixed epoch-0 snapshot")
    control_readout = control_checkpoint.get("readout")
    selected_control_readout = control_model.get("readout")
    control_readout_matches = (control_readout is None and selected_control_readout is None) or (
        isinstance(control_readout, (list, tuple))
        and isinstance(selected_control_readout, (list, tuple))
        and len(control_readout) == len(selected_control_readout)
        and all(
            torch.equal(left, right)
            for left, right in zip(control_readout, selected_control_readout, strict=True)
        )
    )
    if not control_readout_matches:
        raise ValueError("reused V8 readout differs between model.pt and model_ep0.pt")
    control_tensors = checkpoint_named_tensors(control_checkpoint, snapshot_name="control_epoch0")
    control_tensors.update(checkpoint_named_tensors(control_model, snapshot_name="control_model"))
    control_tensor_count = validate_finite_tensors(
        control_tensors,
        lambda tensor: torch.is_tensor(tensor) and bool(torch.isfinite(tensor).all().item()),
    )
    control_numeric = control_provenance.get("numeric_preprocessing")
    if not isinstance(control_numeric, dict) or not isinstance(control_state, dict):
        raise ValueError("reused V8 control numeric or model state is malformed")
    validate_numeric_checkpoint("linear", control_numeric, subset, control_state)
    control_dir.mkdir(parents=True, exist_ok=False)
    for source, filename in (
        (ROOT / CONTROL_MODEL_PATH, "model.pt"),
        (ROOT / CONTROL_EPOCH0_PATH, "model_ep0.pt"),
        (ROOT / CONTROL_PROVENANCE_PATH, "provenance.json"),
        (ROOT / CONTROL_HISTORY_PATH, "history.json"),
    ):
        shutil.copyfile(source, control_dir / filename)
    control_receipt.update(
        {
            "status": "control_pass",
            "parameter_finiteness": {
                "status": "finite",
                "tensor_count": control_tensor_count,
                "includes": "both V8 model snapshots and fixed readout tensors",
            },
            "numeric_preprocessing": {
                "mode": "linear",
                "fit_members_sha256": control_numeric["fit_members_sha256"],
                "mapped_input_width": control_numeric["mapped_input_width"],
            },
            "copied_checkpoint_paths": [
                (control_dir / "model.pt").relative_to(ROOT).as_posix(),
                (control_dir / "model_ep0.pt").relative_to(ROOT).as_posix(),
            ],
        }
    )
    write_json(receipts / "linear-control.json", control_receipt)
    cohort.update({"phase": "control_validated", "status": "ple_profile_pending"})
    write_json(RUNTIME / "cohort.json", cohort)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ple_receipt["status"] = "profile_running"
    write_json(receipts / "ple.json", ple_receipt)
    cohort["phase"] = "profile_ple"
    write_json(RUNTIME / "cohort.json", cohort)
    phase_start = time.monotonic()
    try:
        profile_output = call_main(
            train_rnn,
            "train_rnn.py",
            trainer_argv("ple", profile=True),
            logs / "ple-profile.log",
        )
        profile_stats = parse_trainer_output(profile_output)
        step_wall_ms, profile_evidence = validate_profile(profile_stats)
        profile_record = {
            **profile_stats,
            "step_wall_ms": step_wall_ms,
            "profile_wall_seconds": time.monotonic() - phase_start,
            "profile_outer_batches": PROFILE_WARMUP_BATCHES + PROFILE_TIMED_BATCHES,
            "updates_per_outer_batch": UPDATES_PER_OUTER_BATCH,
            "profile_update_evidence": profile_evidence,
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "process_peak_rss_bytes": resident_memory_bytes(),
            "grad_scaler_enabled": False,
            "amp_init_scale_argument": INERT_SCALER_SCALE_ARGUMENT,
            "amp_init_scale_applied": False,
        }
        if profile_record["cuda_peak_reserved_bytes"] > ALLOCATOR_LIMIT_BYTES:
            raise ValueError("PLE profile allocator reservation exceeds the 4.3 GB target")
        ple_receipt.update({"status": "profile_pass", "profile": profile_record})
        write_json(receipts / "ple.json", ple_receipt)
    except BaseException as error:
        ple_receipt.update(
            {"status": "profile_failed", "error": f"{type(error).__name__}: {error}"}
        )
        write_json(receipts / "ple.json", ple_receipt)
        cohort.update({"status": "profile_failed", "phase": "profile_ple_failed"})
        write_json(RUNTIME / "cohort.json", cohort)
        raise

    total_fit_updates = expected_fit_updates(len(train_groups), len(valid_groups))
    projected_seconds = elapsed_projection_seconds(
        time.monotonic() - started,
        profile_record,
        len(train_groups),
        len(valid_groups),
        len(evaluation_groups),
    )
    gate = {
        "status": "pass" if projected_seconds <= worker_wall_seconds else "reject",
        "worker_elapsed_seconds_at_gate": time.monotonic() - started,
        "projected_total_worker_seconds": projected_seconds,
        "maximum_worker_seconds": worker_wall_seconds,
        "fit_group_counts": {"train": len(train_groups), "valid": len(valid_groups)},
        "loader_membership": loader_membership,
        "membership_receipt": "membership.json",
        "fitting_component_count": membership["selected_component_count"],
        "fitting_component_sha256": membership["selected_component_sha256"],
        "fitting_group_sha256": membership["fit_group_sha256"],
        "fitting_members_sha256": membership["fit_members_sha256"],
        "evaluation_group_count": len(evaluation_groups),
        "expected_ple_fit_optimizer_attempts": total_fit_updates,
        "profile_attempts": EXPECTED_PROFILE_ATTEMPTS,
        "profile": {
            "step_wall_ms": step_wall_ms,
            "peak_reserved_bytes": profile_record["cuda_peak_reserved_bytes"],
            "optimizer_completed_updates": profile_evidence["optimizer_completed_updates"],
            "gradient_stats_nonfinite_steps": profile_evidence["gradient_stats_nonfinite_steps"],
            "grad_scaler_enabled": profile_evidence["grad_scaler_enabled"],
        },
        "control": {
            "reused_v8_checkpoint_sha256": control_summary["checkpoint_sha256"],
            "checkpoint_and_provenance_verified": True,
            "control_tensor_count": control_tensor_count,
        },
        "rule": (
            "profile only PLE; require 316 complete finite FP32 updates and complete finite GradStats, "
            "<=4.3GB allocator reservation, and projected PLE fit plus both 468-group exports inside the cap"
        ),
    }
    write_json(RUNTIME / "profile_gate.json", gate)
    cohort["profile_gate"] = "profile_gate.json"
    if gate["status"] != "pass":
        cohort.update({"status": "profile_gate_rejected", "phase": "profile_gate_rejected"})
        write_json(RUNTIME / "cohort.json", cohort)
        raise RuntimeError(
            f"PLE profile projection exceeds the {worker_wall_seconds}-second worker cap"
        )
    cohort.update({"phase": "fit_ple", "status": "fit_running"})
    write_json(RUNTIME / "cohort.json", cohort)

    ple_receipt["status"] = "fit_running"
    write_json(receipts / "ple.json", ple_receipt)
    fit_started = time.monotonic()
    try:
        fit_output = call_main(
            train_rnn,
            "train_rnn.py",
            trainer_argv("ple", profile=False),
            logs / "ple-fit.log",
        )
        del fit_output
        fit_seconds = time.monotonic() - fit_started
        ple_run_dir = ROOT / "artifacts" / "runs" / RUN_NAMES["ple"]
        checkpoint_path = ple_run_dir / "model_ep0.pt"
        model_path = ple_run_dir / "model.pt"
        provenance_path = ple_run_dir / "provenance.json"
        history_path = ple_run_dir / "history.json"
        sibling = json.loads(provenance_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        selected_checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        validate_checkpoint_snapshot_metadata(checkpoint, sibling)
        validate_checkpoint_snapshot_metadata(selected_checkpoint, sibling)
        checkpoint_args = checkpoint.get("args", {})
        expected_ple_args = dict(control_provenance["args"])
        expected_ple_args.update({"name": RUN_NAMES["ple"], "numeric_map": "ple"})
        if (
            checkpoint.get("epoch") != 0
            or selected_checkpoint.get("epoch") != 0
            or checkpoint.get("n_in") != 211
            or selected_checkpoint.get("n_in") != 211
            or checkpoint_args != expected_ple_args
            or selected_checkpoint.get("args") != expected_ple_args
            or checkpoint_args.get("fit_subset_manifest_sha256")
            != os.environ["WNN_E540_SUBSET_SHA256"]
        ):
            raise ValueError("PLE checkpoint args differ from the exact V8 control recipe")
        if (
            sibling.get("fit_row_groups") != subset["selected_groups"]
            or not same_group_membership(
                sibling.get("evaluation_row_groups"), subset.get("evaluation_groups")
            )
            or sibling.get("fit_subset_manifest_sha256") != os.environ["WNN_E540_SUBSET_SHA256"]
            or sibling.get("fit_subset_members_sha256") != subset["selected_members_sha256"]
            or sibling.get("split_manifest_sha256") != subset.get("split_manifest_sha256")
        ):
            raise ValueError("PLE fit provenance differs from the frozen 512/468 membership")
        source_hashes = {
            "scripts/train_rnn.py": sibling.get("trainer_source_sha256"),
            "src/wnn/nn.py": sibling.get("model_source_sha256"),
            "src/wnn/run_identity.py": sibling.get("run_identity_source_sha256"),
        }
        numeric = sibling.get("numeric_preprocessing")
        if isinstance(numeric, Mapping):
            source_hashes["src/wnn/numeric_ple.py"] = numeric.get("module_source_sha256")
        for relative, expected in V8_TRAINING_SOURCE_HASHES.items():
            if source_hashes.get(relative) != expected or sha256_file(ROOT / relative) != expected:
                raise ValueError(f"PLE trainer source identity differs from V8 recipe: {relative}")
        diagnostics = sibling.get("update_diagnostics")
        if (
            not isinstance(diagnostics, list)
            or len(diagnostics) != 1
            or not isinstance(diagnostics[0], Mapping)
        ):
            raise ValueError("PLE checkpoint has no single epoch-0 update diagnostic")
        update = fit_update_from_provenance_record(diagnostics[0])
        validate_fit_update_contract(update, total_fit_updates)
        history = json.loads(history_path.read_text(encoding="utf-8"))
        history_summary_ple = validate_history(history, sibling, expected_updates=total_fit_updates)
        if diagnostics[0].get("sample_order_identity") != control_summary[
            "sample_order_identity"
        ] or diagnostics[0].get("normalization_identity") != control_provenance[
            "update_diagnostics"
        ][0].get("normalization_identity"):
            raise ValueError("PLE sample order or normalization differs from the reused control")
        checkpoint_hashes = sibling.get("checkpoint_sha256", {})
        if checkpoint_hashes.get("model_ep0.pt") != sha256_file(
            checkpoint_path
        ) or checkpoint_hashes.get("model.pt") != sha256_file(model_path):
            raise ValueError("PLE checkpoint files do not match sibling provenance.json hashes")
        ple_state = checkpoint.get("model")
        selected_ple_state = selected_checkpoint.get("model")
        if (
            not isinstance(ple_state, Mapping)
            or not isinstance(selected_ple_state, Mapping)
            or ple_state.keys() != selected_ple_state.keys()
            or any(not torch.equal(ple_state[key], selected_ple_state[key]) for key in ple_state)
        ):
            raise ValueError("PLE model.pt differs from the fixed epoch-0 checkpoint")
        readout = checkpoint.get("readout")
        selected_readout = selected_checkpoint.get("readout")
        readout_matches = (readout is None and selected_readout is None) or (
            isinstance(readout, (list, tuple))
            and isinstance(selected_readout, (list, tuple))
            and len(readout) == len(selected_readout)
            and all(
                torch.equal(left, right)
                for left, right in zip(readout, selected_readout, strict=True)
            )
        )
        if not readout_matches:
            raise ValueError("PLE fixed readout differs between model.pt and model_ep0.pt")
        named_tensors = checkpoint_named_tensors(checkpoint, snapshot_name="ple_epoch0")
        named_tensors.update(
            checkpoint_named_tensors(selected_checkpoint, snapshot_name="ple_model")
        )
        tensor_count = validate_finite_tensors(
            named_tensors,
            lambda tensor: torch.is_tensor(tensor) and bool(torch.isfinite(tensor).all().item()),
        )
        if not isinstance(numeric, dict) or not isinstance(ple_state, dict):
            raise ValueError("PLE numeric preprocessing or model state is malformed")
        numeric_identity = validate_numeric_checkpoint("ple", numeric, subset, ple_state)
        ple_receipt.update(
            {
                "status": "fit_pass",
                "fit_wall_seconds": fit_seconds,
                "fit_optimizer_diagnostics": update,
                "history": history_summary_ple,
                "sample_order_matches_reused_control": True,
                "parameter_finiteness": {
                    "status": "finite",
                    "tensor_count": tensor_count,
                    "includes": "both PLE snapshots and fixed readout tensors",
                },
                "checkpoint": {
                    "path": checkpoint_path.relative_to(ROOT).as_posix(),
                    "size": checkpoint_path.stat().st_size,
                    "sha256": sha256_file(checkpoint_path),
                    "model_sha256": sha256_file(model_path),
                    "provenance_sha256": sha256_file(provenance_path),
                    "epoch": 0,
                    "precision": "fp32",
                    "grad_scaler_enabled": False,
                    "amp_init_scale_argument": checkpoint_args.get("amp_init_scale"),
                    "amp_init_scale_applied": False,
                },
                "numeric_preprocessing": numeric_identity,
                "fit_log": "logs/ple-fit.log",
            }
        )
        write_json(receipts / "ple.json", ple_receipt)
    except BaseException as error:
        ple_receipt.update({"status": "fit_failed", "error": f"{type(error).__name__}: {error}"})
        write_json(receipts / "ple.json", ple_receipt)
        cohort.update({"status": "fit_failed", "phase": "fit_ple_failed"})
        write_json(RUNTIME / "cohort.json", cohort)
        raise
    cohort.update({"phase": "prediction_linear_control", "status": "prediction_running"})
    write_json(RUNTIME / "cohort.json", cohort)

    for mode in ("linear", "ple"):
        arm_receipt_path = receipts / ("linear-control.json" if mode == "linear" else "ple.json")
        arm = json.loads(arm_receipt_path.read_text(encoding="utf-8"))
        arm["status"] = "prediction_running"
        write_json(arm_receipt_path, arm)
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
            import numpy as np

            prediction_array = np.load(prediction, mmap_mode="r", allow_pickle=False)
            prediction_finiteness = validate_finite_prediction_array(
                prediction_array, (len(evaluation_groups), 20_000, 2)
            )
            membership_bytes = struct.pack(f"<{len(evaluation_groups)}q", *evaluation_groups)
            expected_membership = hashlib.sha256(membership_bytes).hexdigest()
            prediction_digest = sha256_file(prediction)
            checkpoint_digest = (
                control_summary["checkpoint_sha256"]["model_ep0.pt"]
                if mode == "linear"
                else ple_receipt["checkpoint"]["sha256"]
            )
            if (
                metadata.get("run") != RUN_NAMES[mode]
                or metadata.get("split") != "valid"
                or metadata.get("epoch") != 0
                or metadata.get("checkpoint_sha256") != checkpoint_digest
                or metadata.get("prediction_sha256") != prediction_digest
                or metadata.get("row_groups") != evaluation_groups
                or metadata.get("evaluation_membership_sha256") != expected_membership
                or metadata.get("fit_subset_manifest_sha256")
                != hashlib.sha256((ROOT / SUBSET_MANIFEST).read_bytes()).hexdigest()
                or metadata.get("inference_precision") != "fp32"
            ):
                raise ValueError(
                    "prediction sidecar differs from the frozen feature-only 468-group contract"
                )
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
                        "finiteness": prediction_finiteness,
                        "reader": "features only; target and is_scored columns not requested",
                    },
                    "prediction_log": f"logs/{mode}-prediction.log",
                }
            )
            write_json(arm_receipt_path, arm)
        except BaseException as error:
            arm.update({"status": "prediction_failed", "error": f"{type(error).__name__}: {error}"})
            write_json(arm_receipt_path, arm)
            cohort.update({"status": "prediction_failed", "phase": f"prediction_{mode}_failed"})
            write_json(RUNTIME / "cohort.json", cohort)
            raise
        cohort["phase"] = "prediction_ple" if mode == "linear" else "complete"
        write_json(RUNTIME / "cohort.json", cohort)
        if time.monotonic() - started > worker_wall_seconds:
            raise TimeoutError("E540 V9 worker exceeded its dynamically reduced deadline")

    cohort.update(
        {
            "status": "complete_ungraded",
            "phase": "complete",
            "fit_admitted_by_packet": True,
            "grade_owner": "root independent grader",
            "graded_members": len(evaluation_groups),
            "scientific_conclusion": "not included in worker; root grades frozen predictions",
            "v8_linear_control_reused": True,
            "ple_independently_fitted": True,
        }
    )
    write_json(RUNTIME / "cohort.json", cohort)


def main() -> None:
    """Start the explicitly scoped E540 V8 FP32 warm cohort after root review."""
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
