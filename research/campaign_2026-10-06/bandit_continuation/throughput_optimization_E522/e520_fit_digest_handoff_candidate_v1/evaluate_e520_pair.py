"""Independently grade the frozen E520 seed pair against the fixed E511 C1 pair.

This grader is a one-endpoint fold-0 development screen. It cannot choose a
checkpoint, seed subset, pair weight, mask, or metric after seeing results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve()
for parent in ROOT.parents:
    if (parent / "src" / "wnn" / "gmetric.py").is_file():
        ROOT = parent
        break
else:
    raise RuntimeError("cannot find repository root containing src/wnn/gmetric.py")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_bandit_continuation as campaign_eval  # noqa: E402

from wnn.evaluation import paired_component_bootstrap, paired_component_moments  # noqa: E402
from wnn.gmetric import METRIC_ID, WARMUP, GlobalAccumulator  # noqa: E402
from wnn.splits import file_sha256, load_campaign_split  # noqa: E402

SPEC_PATH = Path(
    "research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/scratch_pair_fit_candidate_v1/E520_GRADER_SPEC.json"
)
CONFIG_PATH = Path(
    "research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/scratch_pair_fit_candidate_v1/FIT_STAGE_CONFIG.json"
)
TARGET_ROWS = 20_000
SEQUENCES_PER_BATCH = 4
EXPECTED_TARGET_ROWS = 1_873
EXPECTED_EVALUATION_GROUPS = 468
EXPECTED_COMPONENTS = 141


class GraderInputError(RuntimeError):
    """Raised when a fixed input or candidate artifact differs from the grader spec."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GraderInputError(f"cannot read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise GraderInputError(f"expected JSON object: {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise GraderInputError(f"missing {label}: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise GraderInputError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return observed


def _validate_spec(
    repository_root: Path, config_path: Path, spec_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check the root-frozen comparison, grader dependencies, and own source pins."""
    spec = _read_json(spec_path)
    config = _read_json(config_path)
    if (
        spec.get("format") != "fixed-pair-contrast-grader-v1"
        or spec.get("id") != "E520-raw-scratch-pair-vs-E511-C1V2-fixed-pair"
        or spec.get("status") != "root-frozen-before-E520-fit-scoring"
        or spec.get("metric", {}).get("id") != METRIC_ID
        or spec.get("metric", {}).get("warmup_steps_excluded") != WARMUP
        or spec.get("metric", {}).get("bootstrap")
        != {
            "replicates": 2_000,
            "seed": 271_828,
            "unit": "frozen connected evaluation component",
            "centered": True,
            "interpretation": "conditional on these fixed checkpoints; excludes training-seed and public uncertainty",
        }
    ):
        raise GraderInputError("E520 grader specification differs from the root-frozen endpoint")
    if config.get("experiment_id") != "E520_SCRATCH_PAIR_FIT_V1":
        raise GraderInputError("fit-stage configuration names a different experiment")
    pinned_config = config.get("grader_artifacts", {})
    if (
        pinned_config.get("spec_path") != str(SPEC_PATH)
        or pinned_config.get("spec_sha256") != file_sha256(spec_path)
        or pinned_config.get("script_path")
        != "research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/scratch_pair_fit_candidate_v1/evaluate_e520_pair.py"
        or pinned_config.get("script_sha256") != file_sha256(Path(__file__))
    ):
        raise GraderInputError(
            "grader code or spec differs from the separately frozen E520 grader hashes"
        )
    for relative, expected in spec["dependencies"].items():
        _require_hash(repository_root / relative, str(expected), f"grader dependency {relative}")
    _require_hash(
        repository_root / spec["split"]["path"],
        str(spec["split"]["sha256"]),
        "fold-0 split manifest",
    )
    return spec, config


def _blend_equal_pair(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Average raw float32 members in float64, then return float32 without clipping."""
    left, right = np.asarray(first, dtype=np.float32), np.asarray(second, dtype=np.float32)
    if left.shape != right.shape:
        raise GraderInputError("pair members do not have identical shapes")
    return ((left.astype(np.float64) + right.astype(np.float64)) * 0.5).astype(np.float32)


def _official_mask(is_scored: np.ndarray) -> np.ndarray:
    """Apply exactly the registered prediction-need and organizer scoring masks."""
    scored = np.asarray(is_scored, dtype=bool)
    if scored.ndim != 2 or scored.shape[1] != TARGET_ROWS:
        raise GraderInputError("is_scored must contain complete 20,000-step sequences")
    need_prediction = np.arange(TARGET_ROWS) >= WARMUP
    return scored & need_prediction[None, :]


def _load_targets(
    repository_root: Path, spec: dict[str, Any], split: Any
) -> tuple[list[int], Any, np.ndarray, np.ndarray]:
    """Validate the frozen target cache/index and return labels, mask, and row order."""
    import numpy as np

    target_path = repository_root / spec["split"]["valid_target_cache_path"]
    _require_hash(
        target_path, str(spec["split"]["valid_target_cache_sha256"]), "valid target cache"
    )
    with np.load(target_path, allow_pickle=False) as cache:
        row_groups = list(map(int, cache["row_group"]))
        targets = np.asarray(cache["y"], dtype=np.float32)
        is_scored = np.asarray(cache["mask"], dtype=bool)
        classes = np.asarray(cache["market_class"], dtype=np.int64)
    if campaign_eval._list_sha256(row_groups) != spec["split"]["valid_target_row_groups_sha256"]:
        raise GraderInputError("valid target row order differs from the fixed E511 anchor digest")
    if len(row_groups) != EXPECTED_TARGET_ROWS:
        raise GraderInputError("valid target cache row count differs from the frozen contract")
    expected_groups = campaign_eval._expected_target_groups(
        split.valid_groups, split.evaluation_groups
    )
    campaign_eval._validate_target_index(row_groups, targets, is_scored, classes, expected_groups)
    evaluation_groups = list(map(int, split.evaluation_groups))
    position = {group: index for index, group in enumerate(row_groups)}
    if len(evaluation_groups) != EXPECTED_EVALUATION_GROUPS or any(
        group not in position for group in evaluation_groups
    ):
        raise GraderInputError("frozen fold-0 evaluation groups are incomplete")
    indices = np.asarray([position[group] for group in evaluation_groups], dtype=np.int64)
    return row_groups, targets, is_scored, indices


def _bind_repository_portable_manifest(repository_root: Path, spec: dict[str, Any]) -> Path:
    """Set the split verifier registry only after checking its frozen content hash."""
    os.environ.pop("WNN_PORTABLE_DATA_MANIFEST", None)
    relative = "research/campaign_2026-10-04/portable_inputs.json"
    registry = repository_root / relative
    expected = spec["input_identity"]["asset_hashes"][relative]
    _require_hash(registry, str(expected), "repository portable input registry")
    os.environ["WNN_PORTABLE_DATA_MANIFEST"] = str(registry)
    return registry


def _load_anchor_members(
    repository_root: Path,
    spec: dict[str, Any],
    row_groups: list[int],
) -> dict[int, np.memmap]:
    """Bind both C1-v2 anchor checkpoint/provenance/prediction files to E511's report."""
    report_path = repository_root / spec["anchor"]["evaluation_report_path"]
    _require_hash(
        report_path,
        str(spec["anchor"]["evaluation_report_sha256"]),
        "E511 anchor evaluation report",
    )
    report = _read_json(report_path)
    _validate_anchor_report(report, spec)
    anchors: dict[int, np.memmap] = {}
    for seed in (3, 4):
        expected = spec["anchor"]["members"][str(seed)]
        run = str(expected["run"])
        current = report.get("members", {}).get(run, {})
        for field in (
            "epoch",
            "checkpoint_sha256",
            "prediction_sha256",
            "provenance_sha256",
            "split_manifest_sha256",
            "row_groups_sha256",
            "seed",
        ):
            if current.get(field) != expected[field]:
                raise GraderInputError(f"E511 report member field changed: {run}/{field}")
        run_dir = repository_root / "artifacts" / "runs" / run
        checkpoint_path = run_dir / f"model_ep{expected['epoch']}.pt"
        provenance_path = run_dir / "provenance.json"
        _require_hash(checkpoint_path, str(expected["checkpoint_sha256"]), f"E511 {run} checkpoint")
        _require_hash(provenance_path, str(expected["provenance_sha256"]), f"E511 {run} provenance")
        prediction_path = repository_root / "cache" / f"pred_{run}_valid.npy"
        sidecar_path = prediction_path.with_suffix(".json")
        sidecar = _read_json(sidecar_path)
        prediction, _ = campaign_eval._validate_prediction_artifact(
            prediction_path,
            sidecar,
            run=run,
            epoch=int(expected["epoch"]),
            split_sha256=str(spec["split"]["sha256"]),
            checkpoint_sha256=str(expected["checkpoint_sha256"]),
            row_groups=row_groups,
            shape=(len(row_groups), TARGET_ROWS, 2),
        )
        if file_sha256(prediction_path) != expected["prediction_sha256"]:
            raise GraderInputError(f"E511 {run} prediction differs from its frozen report")
        anchors[seed] = prediction
    return anchors


def _validate_anchor_report(report: dict[str, Any], spec: dict[str, Any]) -> None:
    """Require the report's exact frozen grader, masks, pair text, and data exposure."""
    checks = (
        (report.get("metric_id"), METRIC_ID),
        (
            report.get("grader_spec", {}).get("sha256"),
            spec["anchor"]["reference_grader_spec_sha256"],
        ),
        (
            report.get("grader_source_sha256"),
            spec["anchor"]["reference_grader_script_sha256"],
        ),
        (report.get("split", {}).get("sha256"), spec["split"]["sha256"]),
        (
            report.get("targets", {}).get("sha256"),
            spec["split"]["valid_target_cache_sha256"],
        ),
        (
            report.get("targets", {}).get("row_groups_sha256"),
            spec["split"]["valid_target_row_groups_sha256"],
        ),
        (report.get("selection"), spec["metric"]["selection"]),
        (report.get("pair_blend"), spec["anchor"]["pair_blend"]),
        (report.get("data_exposure"), spec["anchor"]["data_exposure"]),
    )
    if any(observed != expected for observed, expected in checks):
        raise GraderInputError(
            "E511 anchor report does not match the frozen E520 comparison contract"
        )


def _validate_fit_input_identity(
    candidate_root: Path, fit_receipt: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    """Rehash durable fit input receipt and require all content/fold bindings."""
    identity_path = candidate_root / "e520_fit" / "receipts" / "input_identity.json"
    identity_bytes = identity_path.read_bytes()
    identity_sha = hashlib.sha256(identity_bytes).hexdigest()
    identity = _read_json(identity_path)
    portable = identity.get("observed", {}).get("portable_manifest_binding", {})
    required_hashes = {
        "datasets/train.parquet": spec["input_identity"]["train_parquet_sha256"],
        "datasets/valid.parquet": spec["input_identity"]["valid_parquet_sha256"],
        "cache/valid_targets.npz": spec["split"]["valid_target_cache_sha256"],
        **spec["input_identity"]["asset_hashes"],
    }
    assets_manifest = identity.get("observed", {}).get("assets-manifest.json", {})
    target_orders = fit_receipt.get("target_orders", {})
    target_index_matches = all(
        identity.get("target_index_hashes", {}).get(name)
        == target_orders.get(name, {}).get("target_index_sha256")
        for name in ("train", "valid")
    )
    if (
        fit_receipt.get("input_identity_sha256") != identity_sha
        or fit_receipt.get("input_identity") != identity
        or fit_receipt.get("input_identity", {}).get("status") != "matched"
        or fit_receipt.get("input_identity", {}).get("split_sha256") != spec["split"]["sha256"]
        or identity.get("status") != "matched"
        or identity.get("split_sha256") != spec["split"]["sha256"]
        or identity.get("portable_manifest_env") != portable.get("staged_path")
        or portable.get("sha256") != spec["input_identity"]["portable_registry_sha256"]
        or assets_manifest.get("sha256") != spec["input_identity"]["assets_manifest_sha256"]
        or assets_manifest.get("bytes") != spec["input_identity"]["assets_manifest_bytes"]
        or not target_index_matches
        or any(
            identity.get("observed", {}).get(name, {}).get("sha256") != expected
            for name, expected in required_hashes.items()
        )
        or fit_receipt.get("source_archive_sha256") != spec["candidate"]["source_archive_sha256"]
        or fit_receipt.get("score_started") is not False
    ):
        raise GraderInputError("candidate fit input identity receipt is missing or mismatched")
    return identity


def _validate_candidate_review_receipts(candidate_root: Path, fit_receipt: dict[str, Any]) -> None:
    """Authenticate copied review/admission receipts and their exact artifact map."""
    receipt_root = candidate_root / "e520_fit" / "receipts"
    review_path = receipt_root / "external_review.json"
    admission_path = receipt_root / "controller_admission.json"
    packet_meta_path = receipt_root / "packet_meta.json"
    for path, key, label in (
        (review_path, "external_review_sha256", "external review"),
        (admission_path, "controller_admission_sha256", "controller admission"),
    ):
        _require_hash(path, str(fit_receipt.get(key)), label)
    review = _read_json(review_path)
    admission = _read_json(admission_path)
    packet_meta = _read_json(packet_meta_path)
    artifact_hashes = fit_receipt.get("candidate_artifact_hashes")
    actor_ids = {"weak_diversity_kaggle_actor", "e520_cloud_delivery"}
    reviewer = review.get("reviewer", {})
    issuer = admission.get("issuer", {})
    grader_sha = file_sha256(Path(__file__))
    if (
        review.get("experiment_id") != "E520_SCRATCH_PAIR_FIT_V1"
        or review.get("scope") != "scratch-pair-fit-and-streaming-export"
        or review.get("status") != "approved"
        or reviewer.get("role") != "independent-recipe-grader"
        or not reviewer.get("id")
        or reviewer.get("id") in actor_ids
        or review.get("reviewed_hashes") != artifact_hashes
        or admission.get("experiment_id") != "E520_SCRATCH_PAIR_FIT_V1"
        or admission.get("status") != "approved"
        or admission.get("fit_enabled") is not True
        or issuer.get("role") != "controller"
        or not issuer.get("id")
        or issuer.get("id") in actor_ids
        or admission.get("external_review_sha256") != fit_receipt.get("external_review_sha256")
        or admission.get("reviewed_hashes") != artifact_hashes
        or packet_meta.get("artifact_hashes") != artifact_hashes
        or packet_meta.get("external_review_sha256") != fit_receipt.get("external_review_sha256")
        or packet_meta.get("controller_admission_sha256")
        != fit_receipt.get("controller_admission_sha256")
        or packet_meta.get("grader_script_sha256") != grader_sha
        or fit_receipt.get("grader_script_sha256") != grader_sha
        or packet_meta.get("score_enabled") is not False
        or admission.get("evaluation_gate", {}).get("grader_status") != "frozen"
        or admission.get("evaluation_gate", {}).get("grader_script_sha256") != grader_sha
        or admission.get("profile_gate", {}).get("status") != "completed"
        or admission.get("profile_gate", {}).get("receipt_sha256")
        != fit_receipt.get("profile_receipt_sha256")
    ):
        raise GraderInputError("copied review/admission receipts do not bind this fit and grader")


def _validate_candidate_member(
    candidate_root: Path,
    fit_receipt: dict[str, Any],
    spec: dict[str, Any],
    config: dict[str, Any],
    seed: int,
    row_groups: list[int],
    split: Any,
) -> tuple[np.memmap, dict[str, Any]]:
    """Bind a candidate epoch-3 checkpoint and streamed valid export to its fit receipt."""
    run = f"e520_raw_baseline_s{seed}_f0"
    members = [item for item in fit_receipt.get("members", []) if item.get("seed") == seed]
    if len(members) != 1 or members[0].get("run_name") != run:
        raise GraderInputError(f"candidate fit receipt lacks exactly one seed-{seed} member")
    member = members[0]
    if (
        member.get("selected_epoch") != 3
        or member.get("input_identity_sha256") != fit_receipt.get("input_identity_sha256")
        or member.get("source_archive_sha256") != spec["candidate"]["source_archive_sha256"]
        or member.get("trainer_source_sha256") != spec["candidate"]["trainer_source_sha256"]
    ):
        raise GraderInputError(f"candidate {run} member receipt disagrees with the fixed ancestry")
    expected_argv = config.get("fit_commands", {}).get(run)
    if (
        not isinstance(expected_argv, list)
        or member.get("fit_argv") != expected_argv
        or member.get("fit_argv_sha256") != _canonical_sha256(expected_argv)
    ):
        raise GraderInputError(f"candidate {run} trainer command differs from the frozen recipe")
    run_dir = candidate_root / "e520_source" / "artifacts" / "runs" / run
    checkpoint_path = run_dir / "model.pt"
    epoch_checkpoint_path = run_dir / "model_ep3.pt"
    provenance_path = run_dir / "provenance.json"
    for path, key, label in (
        (checkpoint_path, "checkpoint_sha256", f"{run} selected checkpoint"),
        (epoch_checkpoint_path, "epoch_checkpoint_sha256", f"{run} epoch-3 checkpoint"),
        (provenance_path, "trainer_provenance_sha256", f"{run} trainer provenance"),
    ):
        _require_hash(path, str(member[key]), label)
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    provenance = _read_json(provenance_path)
    args = checkpoint.get("args", {})
    fixed = {
        "name": run,
        "seed": seed,
        "arch": "gru",
        "cell": "gru",
        "hidden": 128,
        "layers": 2,
        "dropout": 0.0,
        "raw_only": True,
        "loss": "wmse",
        "loss_norm": "chunk",
        "train_source": "train+valid",
        "train_seqs": 10_607,
        "valid_fold": 0,
        "epochs": 4,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 3,
        "no_eval_during_fit": True,
        "batch": 16,
        "chunk": 2_000,
        "lr": 0.001,
        "wd": 0.01,
        "optim": "adamw",
        "sched": "onecycle",
        "precision": "fp16",
        "init": "",
        "distill": "",
        "distill_boot_h": 0,
        "propensity": "",
        "prop_input": "",
        "prop_power": 1.0,
        "prop_eps": 0.0,
        "valid_true_mask": -1,
        "ema_teacher": 0.0,
        "aux_struct": 0.0,
        "aux_next": 0.0,
        "aux_tier": 0.0,
        "aux_mask": 0.0,
        "aux_sign": 0.0,
        "aux_quote": 0.0,
        "aux_weight": 0.0,
        "swa_frac": 0.0,
        "fast": True,
    }
    mismatched = {
        key: (args.get(key), expected)
        for key, expected in fixed.items()
        if args.get(key) != expected
    }
    if (
        checkpoint.get("epoch") != 3
        or member.get("selected_epoch") != 3
        or provenance.get("selected_epoch") != 3
        or mismatched
        or campaign_eval._normalized(provenance.get("args")) != campaign_eval._normalized(args)
        or provenance.get("split_manifest_sha256") != spec["split"]["sha256"]
    ):
        raise GraderInputError(
            f"candidate checkpoint/provenance is not the fixed recipe: {mismatched}"
        )
    fit = provenance.get("fit_row_groups", {})
    train_groups = set(map(int, fit.get("train", [])))
    valid_groups = set(map(int, fit.get("valid", [])))
    split.assert_fit_groups(train_groups, valid_groups)
    if len(train_groups) != 10_535 or len(valid_groups) != 1_405 or 528 in train_groups:
        raise GraderInputError("candidate fit membership is incomplete or contaminated")
    pred_path = candidate_root / "e520_fit" / "predictions" / f"{run}_valid.npy"
    sidecar_path = pred_path.with_suffix(".json")
    export = member.get("exports", {}).get("valid", {})
    _require_hash(sidecar_path, str(export.get("sidecar_sha256")), f"{run} valid export sidecar")
    sidecar = _read_json(sidecar_path)
    prediction_sha = file_sha256(pred_path)
    if (
        sidecar.get("status") != "complete"
        or sidecar.get("run_name") != run
        or sidecar.get("seed") != seed
        or sidecar.get("split") != "valid"
        or sidecar.get("selected_epoch") != 3
        or sidecar.get("shape") != [len(row_groups), TARGET_ROWS, 2]
        or sidecar.get("dtype") != "float32"
        or sidecar.get("row_group_order_sha256") != spec["split"]["valid_target_row_groups_sha256"]
        or sidecar.get("target_cache_sha256") != spec["split"]["valid_target_cache_sha256"]
        or sidecar.get("split_manifest_sha256") != spec["split"]["sha256"]
        or sidecar.get("checkpoint_sha256") != member.get("checkpoint_sha256")
        or sidecar.get("trainer_provenance_sha256") != member.get("trainer_provenance_sha256")
        or sidecar.get("trainer_argv_sha256") != member.get("fit_argv_sha256")
        or sidecar.get("input_identity_sha256") != fit_receipt.get("input_identity_sha256")
        or sidecar.get("source_archive_sha256") != fit_receipt.get("source_archive_sha256")
        or sidecar.get("trainer_source_sha256") != spec["candidate"]["trainer_source_sha256"]
        or sidecar.get("target_index_sha256")
        != fit_receipt.get("target_orders", {}).get("valid", {}).get("target_index_sha256")
        or sidecar.get("prediction_sha256") != prediction_sha
        or sidecar.get("official_metric_clipping_applied") is not False
        or sidecar.get("finite_checked_per_chunk") is not True
    ):
        raise GraderInputError(f"candidate valid prediction binding differs for {run}")
    prediction = np.load(pred_path, mmap_mode="r", allow_pickle=False)
    if prediction.shape != (len(row_groups), TARGET_ROWS, 2) or prediction.dtype != np.float32:
        raise GraderInputError(f"candidate valid prediction shape/dtype differs for {run}")
    del checkpoint
    return prediction, {
        "seed": seed,
        "run_name": run,
        "selected_epoch": 3,
        "checkpoint_sha256": member["checkpoint_sha256"],
        "epoch_checkpoint_sha256": member["epoch_checkpoint_sha256"],
        "trainer_provenance_sha256": member["trainer_provenance_sha256"],
        "prediction_sha256": prediction_sha,
        "prediction_sidecar_sha256": export["sidecar_sha256"],
        "row_group_order_sha256": sidecar["row_group_order_sha256"],
        "target_index_sha256": sidecar.get("target_index_sha256"),
    }


def _grade(
    repository_root: Path,
    candidate_root: Path,
    output_dir: Path,
    *,
    spec_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Compute the one frozen pair-vs-anchor contrast and descriptive seed context."""
    repository_root = repository_root.resolve(strict=True)
    candidate_root = candidate_root.resolve(strict=True)
    spec, config = _validate_spec(repository_root, config_path, spec_path)
    _bind_repository_portable_manifest(repository_root, spec)
    split = load_campaign_split(repository_root / spec["split"]["path"], repository_root)
    if split.fold != 0 or len(split.evaluation_groups) != EXPECTED_EVALUATION_GROUPS:
        raise GraderInputError("expected exactly the frozen 468-group fold-0 evaluation set")
    row_groups, targets, is_scored, evaluation_indices = _load_targets(repository_root, spec, split)
    fit_receipt_path = candidate_root / "e520_fit" / "receipts" / "fit_pair_receipt.json"
    fit_receipt = _read_json(fit_receipt_path)
    if (
        fit_receipt.get("experiment_id") != "E520_SCRATCH_PAIR_FIT_V1"
        or fit_receipt.get("status") != "complete"
        or fit_receipt.get("fit_started") is not True
        or fit_receipt.get("score_started") is not False
        or fit_receipt.get("retry_count") != 0
        or len(fit_receipt.get("members", [])) != 2
        or fit_receipt.get("input_identity", {}).get("status") != "matched"
        or fit_receipt.get("input_identity", {}).get("split_sha256") != spec["split"]["sha256"]
        or fit_receipt.get("source_archive_sha256") != spec["candidate"]["source_archive_sha256"]
        or fit_receipt.get("trainer_source_sha256") != spec["candidate"]["trainer_source_sha256"]
        or fit_receipt.get("anchor_report_sha256") != spec["anchor"]["evaluation_report_sha256"]
        or fit_receipt.get("grader_script_sha256") != file_sha256(Path(__file__))
    ):
        raise GraderInputError(
            "candidate fit receipt is incomplete or not an unscored successful pair"
        )
    _validate_candidate_review_receipts(candidate_root, fit_receipt)
    _validate_fit_input_identity(candidate_root, fit_receipt, spec)
    validated_candidates = {
        seed: _validate_candidate_member(
            candidate_root, fit_receipt, spec, config, seed, row_groups, split
        )
        for seed in (3, 4)
    }
    candidate_members = {seed: value[0] for seed, value in validated_candidates.items()}
    candidate_bindings = {str(seed): value[1] for seed, value in validated_candidates.items()}
    anchors = _load_anchor_members(repository_root, spec, row_groups)
    groups = list(map(int, split.evaluation_groups))
    group_to_component = {
        int(group): int(component)
        for source, group, component in split.components
        if source == "valid"
    }
    if any(group not in group_to_component for group in groups):
        raise GraderInputError("evaluation group lacks a frozen validation component")
    components = np.asarray([group_to_component[group] for group in groups], dtype=np.int64)
    selected_mask = _official_mask(is_scored[evaluation_indices])
    candidate_moments = np.empty((len(groups), 2, 6), dtype=np.float64)
    anchor_moments = np.empty_like(candidate_moments)
    candidate_seed_scores = {seed: GlobalAccumulator() for seed in (3, 4)}
    anchor_seed_scores = {seed: GlobalAccumulator() for seed in (3, 4)}
    for start in range(0, len(groups), SEQUENCES_PER_BATCH):
        stop = min(start + SEQUENCES_PER_BATCH, len(groups))
        positions = evaluation_indices[start:stop]
        y = np.asarray(targets[positions], dtype=np.float32)
        mask = selected_mask[start:stop]
        c3 = np.asarray(candidate_members[3][positions], dtype=np.float32)
        c4 = np.asarray(candidate_members[4][positions], dtype=np.float32)
        a3 = np.asarray(anchors[3][positions], dtype=np.float32)
        a4 = np.asarray(anchors[4][positions], dtype=np.float32)
        candidate_pair = _blend_equal_pair(c3, c4)
        anchor_pair = _blend_equal_pair(a3, a4)
        cand, anchor = paired_component_moments(y, candidate_pair, anchor_pair, mask)
        candidate_moments[start:stop] = cand
        anchor_moments[start:stop] = anchor
        candidate_seed_scores[3].add(y, c3, mask)
        candidate_seed_scores[4].add(y, c4, mask)
        anchor_seed_scores[3].add(y, a3, mask)
        anchor_seed_scores[4].add(y, a4, mask)
    bootstrap = spec["metric"]["bootstrap"]
    comparison = paired_component_bootstrap(
        candidate_moments,
        anchor_moments,
        components,
        n_boot=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]),
        centered=bool(bootstrap["centered"]),
    )
    if comparison["n_components"] != EXPECTED_COMPONENTS:
        raise GraderInputError("frozen component count differs from the grader contract")
    candidate_context = {str(seed): candidate_seed_scores[seed].result() for seed in (3, 4)}
    anchor_context = {str(seed): anchor_seed_scores[seed].result() for seed in (3, 4)}
    report = {
        "grader_id": spec["id"],
        "grader_spec": {
            "path": str(spec_path),
            "sha256": file_sha256(spec_path),
        },
        "grader_source_sha256": file_sha256(Path(__file__)),
        "fit_stage_config_sha256": file_sha256(config_path),
        "fit_receipt_sha256": file_sha256(fit_receipt_path),
        "candidate": {
            "experiment_id": fit_receipt["experiment_id"],
            "members": {str(seed): candidate_bindings[str(seed)] for seed in (3, 4)},
            "pair": "equal 0.5/0.5 raw seed-3/seed-4 mean at fixed zero-based epoch 3",
            "individual_seed_context": candidate_context,
        },
        "anchor": {
            "evaluation_report_path": spec["anchor"]["evaluation_report_path"],
            "evaluation_report_sha256": spec["anchor"]["evaluation_report_sha256"],
            "members": spec["anchor"]["members"],
            "pair": "equal 0.5/0.5 raw fixed epoch-1 E511 C1-v2 seed-3/seed-4 mean",
            "individual_seed_context": anchor_context,
        },
        "metric_id": METRIC_ID,
        "split": {
            "fold": split.fold,
            "sha256": split.sha256,
            "evaluation_groups": groups,
            "evaluation_group_count": len(groups),
            "exposed_development_data": True,
        },
        "targets": {
            "path": str(repository_root / spec["split"]["valid_target_cache_path"]),
            "sha256": spec["split"]["valid_target_cache_sha256"],
            "row_groups_sha256": campaign_eval._list_sha256(row_groups),
            "row_group_order": "cache/valid_targets.npz row_group order",
        },
        "selection": spec["metric"]["selection"],
        "warmup_steps_excluded": WARMUP,
        "pair_blend": spec["metric"]["primary_candidate_pair"],
        "primary_comparison": {
            "candidate_pair_vs_anchor_pair": comparison,
            "interpretation": "paired connected-component bootstrap conditional on fixed checkpoints; fold-0 is exposed development evidence only",
        },
        "selected_rows": int(selected_mask.sum()),
        "component_count": int(len(np.unique(components))),
        "bootstrap": bootstrap,
        "data_exposure": spec["data_exposure"],
        "post_outcome_selection": False,
        "public_forecast": None,
    }
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite grader output: {output_dir}")
    output_dir.mkdir(parents=True)
    report_partial = output_dir / "report.json.partial"
    report_partial.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_partial.replace(output_dir / "report.json")
    component_receipt = {
        "split_manifest_sha256": split.sha256,
        "selection": spec["metric"]["selection"],
        "components": [
            {"evaluation_group": group, "component": int(component)}
            for group, component in zip(groups, components, strict=True)
        ],
    }
    (output_dir / "components.json").write_text(
        json.dumps(component_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    """Run the frozen E520 pair endpoint against the existing E511 anchor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=ROOT / SPEC_PATH)
    parser.add_argument("--config", type=Path, default=ROOT / CONFIG_PATH)
    args = parser.parse_args()
    result = _grade(
        args.repository_root,
        args.candidate_root,
        args.output_dir,
        spec_path=args.spec,
        config_path=args.config,
    )
    print(json.dumps(result["primary_comparison"], indent=2))


if __name__ == "__main__":
    main()
