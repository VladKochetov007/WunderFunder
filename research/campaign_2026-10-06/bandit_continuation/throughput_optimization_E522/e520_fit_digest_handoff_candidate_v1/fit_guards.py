"""Pure fail-closed validation for the E520 scratch-pair fit packet."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

EXPERIMENT_ID = "E520_SCRATCH_PAIR_FIT_V1"
ACTOR_IDS = {"weak_diversity_kaggle_actor", "e520_cloud_delivery"}
REQUIRED_ARTIFACTS = (
    "FIT_STAGE_CONFIG.json",
    "FIT_STAGE_README.md",
    "E520_FIT_STAGE_PLAN.json",
    "FIT_OUTPUT_RECEIPT_SCHEMA.json",
    "E520_GRADER_SPEC.json",
    "fit_worker.py",
    "fit_kernel.py",
    "fit_guards.py",
    "prepare_fit_packet.py",
    "evaluate_e520_pair.py",
    "support/export_guards.py",
    "support/input_identity.py",
    "support/verified_digest_handoff.py",
    "source_archive.tar",
    "tests/test_fit_guards.py",
    "tests/test_fit_packet_builder.py",
    "tests/test_training_target_semantics.py",
    "tests/test_digest_handoff.py",
)
MAX_WORKER_WALL_SECONDS = 10_200
WMSE_TRAINING_TARGET_SEMANTICS = {
    "source_archive_sha256": "78f9e5c3d001b8a0254ed535846f07b3b9a906adba2a270f7ddc1ab6606081fc",
    "source_file": "src/wnn/nn.py",
    "source_file_sha256": "9293f8f4e2a2f3bd9f0157f8f83f12a6a758608dd03c7e141c290d82d4d5aeca",
    "source_symbol": "weighted_mse_loss",
    "source_function_sha256": "0d12688ff3b16ba0d0d58f57bc276aa3a822b553a6843cdebdb9804690be4869",
    "clip_constant_name": "METRIC_CLIP",
    "clip_constant_value": 2.0,
    "target_transform": "target.float().clamp(-METRIC_CLIP, METRIC_CLIP)",
    "weight_formula": "abs(clipped_target) * valid[..., None]",
    "loss_formula": "sum(weight * (pred.float() - clipped_target) ** 2) / max(sum(weight), 1e-8)",
    "normalization_scope": "per trainer chunk; --loss-norm chunk calls weighted_mse_loss on that chunk and normalizes by that chunk weight sum",
    "prediction_clipping_during_training_loss": False,
    "auxiliary_loss_terms": False,
    "distinct_from_official_metric": "Training clips targets and weights by abs(clipped target); global_wp_clip2_v1 additionally clips predictions and applies the official scored/prediction mask.",
}


class FitGateError(RuntimeError):
    """Raised when a frozen fit packet or its admission evidence is incomplete."""


def validate_training_target_semantics(config: dict[str, Any], packet_root: Path) -> None:
    """Fail closed unless the WMSE contract and its staged plan are exact.

    Args:
        config: Loaded fit-stage configuration.
        packet_root: Root containing the staged fit-stage plan.

    Raises:
        FitGateError: If the objective flags, semantic contract, or plan binding differ.
    """
    recipe = config.get("recipe")
    if (
        not isinstance(recipe, dict)
        or recipe.get("loss") != "wmse"
        or recipe.get("loss_norm") != "chunk"
        or recipe.get("training_target_semantics") != WMSE_TRAINING_TARGET_SEMANTICS
    ):
        raise FitGateError("WMSE training-target semantics differ from the pinned trainer")

    plan_name = config.get("fit_stage_plan_path")
    plan_sha = config.get("fit_stage_plan_sha256")
    relative_plan = PurePosixPath(plan_name) if isinstance(plan_name, str) else None
    if (
        relative_plan is None
        or not relative_plan.parts
        or relative_plan.is_absolute()
        or ".." in relative_plan.parts
        or relative_plan.as_posix() != plan_name
    ):
        raise FitGateError("fit-stage plan path is missing or unsafe")
    plan_path = packet_root.joinpath(*relative_plan.parts)
    if plan_path.is_symlink() or not plan_path.is_file() or sha256_file(plan_path) != plan_sha:
        raise FitGateError("fit-stage plan bytes differ from their configured SHA-256")
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FitGateError("fit-stage plan is not valid JSON") from error
    if not isinstance(plan, dict):
        raise FitGateError("fit-stage plan root must be a JSON object")
    plan_recipe = plan.get("recipe")
    plan_revision = plan.get("fit_stage_candidate_v2")
    if not isinstance(plan_recipe, dict) or not isinstance(plan_revision, dict):
        raise FitGateError("fit-stage plan lacks its recipe or candidate revision object")
    handoff_contract = config.get("execution_optimization")
    if (
        not isinstance(handoff_contract, dict)
        or handoff_contract != plan.get("execution_optimization")
        or handoff_contract.get("allowed_sources")
        != ["datasets/train.parquet", "datasets/valid.parquet"]
        or handoff_contract.get("portable_verifier_integration")
        != "wrap only pinned wnn.portability._content_digest; unchanged verify_portable_source metadata/registry comparisons and all split membership checks remain active; a stat/allowlist miss calls the pinned original hash"
    ):
        raise FitGateError("digest-handoff execution contract differs between config and plan")
    if (
        plan_recipe.get("training_target_semantics") != WMSE_TRAINING_TARGET_SEMANTICS
        or plan_revision.get("revision") != config.get("candidate_revision")
        or plan_revision.get("parent_plan_path") != config.get("parent_plan_path")
        or plan_revision.get("parent_plan_sha256") != config.get("parent_plan_sha256")
        or config.get("fit_stage_plan_sha256") != sha256_file(plan_path)
    ):
        raise FitGateError("fit-stage plan does not bind the pinned WMSE target semantics")


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 digest for in-memory receipt bytes."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, block_bytes: int = 1 << 20) -> str:
    """Hash a file in bounded blocks without materializing its contents."""
    if block_bytes <= 0:
        raise ValueError("block_bytes must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash a JSON value using a stable canonical representation."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def artifact_hashes(packet_root: Path) -> dict[str, str]:
    """Hash the exact reviewed source set and reject missing or non-regular files."""
    result: dict[str, str] = {}
    for relative in REQUIRED_ARTIFACTS:
        path = packet_root / relative
        if path.is_symlink() or not path.is_file():
            raise FitGateError(f"review artifact is missing or not a regular file: {relative}")
        result[relative] = sha256_file(path)
    return result


def _receipt(payload: bytes, expected_sha256: str, label: str) -> dict[str, Any]:
    if len(expected_sha256) != 64 or sha256_bytes(payload) != expected_sha256:
        raise FitGateError(f"{label} receipt bytes do not match their externally pinned SHA-256")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise FitGateError(f"{label} receipt is not valid JSON") from error
    if not isinstance(value, dict):
        raise FitGateError(f"{label} receipt root must be an object")
    return value


def validate_review(
    payload: bytes,
    expected_sha256: str,
    hashes: dict[str, str],
) -> dict[str, Any]:
    """Require a separate recipe/grader reviewer to approve every exact artifact."""
    review = _receipt(payload, expected_sha256, "independent review")
    reviewer = review.get("reviewer", {})
    if (
        review.get("experiment_id") != EXPERIMENT_ID
        or review.get("scope") != "scratch-pair-fit-and-streaming-export"
        or review.get("status") != "approved"
        or reviewer.get("role") != "independent-recipe-grader"
        or not reviewer.get("id")
        or reviewer.get("id") in ACTOR_IDS
        or review.get("reviewed_hashes") != hashes
    ):
        raise FitGateError("independent review does not approve this exact E520 fit artifact set")
    return review


def validate_controller_admission(
    payload: bytes,
    expected_sha256: str,
    *,
    review_sha256: str,
    hashes: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Require a current root fit admission, successful profile, grader, and live budget."""
    admission = _receipt(payload, expected_sha256, "controller admission")
    issuer = admission.get("issuer", {})
    live = admission.get("live_admission", {})
    profile = admission.get("profile_gate", {})
    evaluation = admission.get("evaluation_gate", {})
    worker_wall = live.get("worker_wall_cap_seconds")
    quota_reservation = live.get("quota_reservation_seconds")
    if (
        admission.get("experiment_id") != EXPERIMENT_ID
        or admission.get("scope") != "serial-seed3-seed4-fit-and-export"
        or admission.get("status") != "approved"
        or admission.get("fit_enabled") is not True
        or issuer.get("role") != "controller"
        or not issuer.get("id")
        or issuer.get("id") in ACTOR_IDS
        or admission.get("external_review_sha256") != review_sha256
        or admission.get("reviewed_hashes") != hashes
        or admission.get("recipe_sha256") != canonical_sha256(config["recipe"])
        or admission.get("input_static_identity_sha256")
        != canonical_sha256(config["input_identity"]["static_hashes"])
        or profile.get("status") != "completed"
        or not _is_sha256(profile.get("receipt_sha256"))
        or evaluation.get("grader_status") != "frozen"
        or not _is_sha256(evaluation.get("grader_script_sha256"))
        or evaluation.get("anchor_report_sha256")
        != config["evaluation"]["anchor_evaluation_report_sha256"]
        or not _is_sha256(live.get("quota_receipt_sha256"))
        or not _is_sha256(live.get("campaign_cap_receipt_sha256"))
        or not _is_sha256(live.get("resource_receipt_sha256"))
        or live.get("provider_runtime_cap_seconds") is None
        or not isinstance(worker_wall, int)
        or worker_wall <= 0
        or worker_wall
        > min(MAX_WORKER_WALL_SECONDS, config["resource_policy"]["max_worker_wall_seconds"])
        or not isinstance(quota_reservation, (int, float))
        or quota_reservation < worker_wall
        or live.get("account_reserve_fraction", 0.0)
        < config["resource_policy"]["account_reserve_fraction"]
        or live.get("available_quota_after_reserve_seconds", 0) < quota_reservation
        or live.get("campaign_headroom_seconds", 0) < quota_reservation
        or live.get("visible_gpu_count") != 1
        or live.get("parallel_fits") is not False
        or live.get("fit_seeds") != config["recipe"]["seeds"]
        or live.get("fit_order") != config["resource_policy"]["fit_order"]
        or live.get("current_input_identity_required") is not True
    ):
        raise FitGateError("controller fit admission is stale, incomplete, or mismatched")
    return admission


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def trainer_argv(seed: int, run_name: str) -> list[str]:
    """Return the exact fixed recipe argv for one seed; no optional recipe knobs."""
    if seed not in (3, 4):
        raise ValueError("E520 admits only preregistered seeds 3 and 4")
    expected = f"e520_raw_baseline_s{seed}_f0"
    if run_name != expected:
        raise ValueError(f"run name must be {expected}")
    return [
        "python",
        "scripts/train_rnn.py",
        "--name",
        run_name,
        "--device",
        "cuda",
        "--arch",
        "gru",
        "--cell",
        "gru",
        "--hidden",
        "128",
        "--layers",
        "2",
        "--dropout",
        "0",
        "--seed",
        str(seed),
        "--raw-only",
        "--loss",
        "wmse",
        "--loss-norm",
        "chunk",
        "--train-source",
        "train+valid",
        "--train-seqs",
        "10607",
        "--split-manifest",
        "research/campaign_2026-09-29/artifacts/splits/fold0.json",
        "--valid-fold",
        "0",
        "--epochs",
        "4",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "3",
        "--no-eval-during-fit",
        "--batch",
        "16",
        "--chunk",
        "2000",
        "--lr",
        "0.001",
        "--wd",
        "0.01",
        "--optim",
        "adamw",
        "--sched",
        "onecycle",
        "--precision",
        "fp16",
        "--init",
        "",
        "--distill",
        "",
        "--distill-boot-h",
        "0",
        "--propensity",
        "",
        "--prop-input",
        "",
        "--prop-power",
        "1",
        "--prop-eps",
        "0",
        "--valid-true-mask",
        "-1",
        "--ema-teacher",
        "0",
        "--aux-struct",
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
        "--aux",
        "",
        "--aux-weight",
        "0",
        "--swa-frac",
        "0",
        "--fast",
    ]


def validate_prediction_chunk(
    values: Any,
    *,
    expected_shape: tuple[int, int, int],
    require_finite: bool = True,
) -> None:
    """Validate one small prediction chunk before writing it to a memmap."""
    import numpy as np

    if not isinstance(values, np.ndarray):
        raise TypeError("prediction chunk must be a NumPy array")
    if values.dtype != np.dtype("float32"):
        raise FitGateError(f"prediction chunk dtype must be float32, got {values.dtype}")
    if values.shape != expected_shape or len(expected_shape) != 3 or expected_shape[2] != 2:
        raise FitGateError(f"prediction chunk shape {values.shape} != {expected_shape}")
    if require_finite and not bool(np.isfinite(values).all()):
        raise FitGateError("prediction chunk contains non-finite values")


def stream_memmap_for_test(
    output_path: Path,
    chunks: list[tuple[int, int, Any]],
    *,
    rows: int,
    sequence_rows: int,
    chunk_rows: int,
) -> dict[str, Any]:
    """Write ordered synthetic chunks atomically; production fixes 20k/4k at caller."""
    import numpy as np

    if min(rows, sequence_rows, chunk_rows) <= 0 or sequence_rows % chunk_rows:
        raise ValueError("test writer dimensions must be positive and exactly chunkable")
    final = output_path
    partial = final.with_name(final.name + ".partial")
    if final.exists() or partial.exists():
        raise FileExistsError("refusing to overwrite an existing prediction or partial output")
    final.parent.mkdir(parents=True, exist_ok=True)
    mapped: Any | None = None
    mapped = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float32, shape=(rows, sequence_rows, 2)
    )
    time_blocks = sequence_rows // chunk_rows
    coverage = np.zeros((rows, time_blocks), dtype=np.bool_)
    try:
        for row_start, time_start, values in chunks:
            if not isinstance(row_start, int) or row_start < 0 or row_start >= rows:
                raise FitGateError("synthetic exporter row start is invalid")
            batch = values.shape[0] if hasattr(values, "shape") else 0
            row_stop = row_start + batch
            time_stop = time_start + chunk_rows
            validate_prediction_chunk(
                values,
                expected_shape=(batch, chunk_rows, 2),
            )
            if row_stop > rows or not 0 <= time_start < sequence_rows:
                raise FitGateError("synthetic exporter slice lies outside output shape")
            if time_start % chunk_rows or time_stop > sequence_rows:
                raise FitGateError(
                    "synthetic exporter chunk is not aligned to the fixed chunk grid"
                )
            block = time_start // chunk_rows
            if coverage[row_start:row_stop, block].any():
                raise FitGateError("synthetic exporter repeated an output slice")
            mapped[row_start:row_stop, time_start:time_stop, :] = values
            coverage[row_start:row_stop, block] = True
        if not coverage.all():
            raise FitGateError("synthetic exporter did not cover every output slice exactly once")
        mapped.flush()
        mapped._mmap.close()
        del mapped
        mapped = None
        with partial.open("rb") as stream:
            import os

            os.fsync(stream.fileno())
        os.replace(partial, final)
        return {
            "path": str(final),
            "shape": [rows, sequence_rows, 2],
            "dtype": "float32",
            "file_bytes": final.stat().st_size,
        }
    except Exception:
        if mapped is not None:
            mapped._mmap.close()
            del mapped
        raise
