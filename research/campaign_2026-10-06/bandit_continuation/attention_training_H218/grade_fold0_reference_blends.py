"""Independently grade preregistered H218 blends after root artifact admission."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import TypeAlias, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
H218_DIR = ROOT / "research/campaign_2026-10-06/bandit_continuation/attention_training_H218"
PREREG = H218_DIR / "H218_FOLD0_REFERENCE_BLEND_PREREG_20261008_R1.json"
AUTHORIZATION = H218_DIR / "H218_FOLD0_REFERENCE_BLEND_GRADE_ADMISSION_20261008_R1.json"
SPLIT_PATH = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
TARGET_PATH = ROOT / "cache/valid_targets.npz"
SOURCE_FAMILIES = ROOT / "research/campaign_2026-10-06/bandit_continuation/SOURCE_FAMILIES.json"
SCORER_PATH = ROOT / "research/campaign_2026-09-30/audit/campaign_score.py"
GMETRIC_PATH = ROOT / "src/wnn/gmetric.py"
H217_REFERENCE_HELPERS = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/vast/H217_A3_GRADE_TCN_REFERENCE_BLEND_V1.py"
)
REPORT = H218_DIR / "H218_FOLD0_REFERENCE_BLEND_20261008_R1.json"
MARKDOWN = H218_DIR / "H218_FOLD0_REFERENCE_BLEND_20261008_R1.md"
EXPECTED_PREREG_SHA256 = "6c52f156344c5f8d084440044d980ae8d6a7b0b8ef9f97f6dcd48c249148ae73"
EXPORT_CONTRACT = H218_DIR / "H218_FOLD0_EVAL_EXPORT_CONTRACT_V1_20261008.json"
EXPECTED_EXPORT_CONTRACT_SHA256 = "47fafae83e7cda7c9568b73350f5f8d47b281c27adc56820e350e87b6571a652"
EXPECTED_SOURCE_LOCK_SHA256 = "64696558572c4afacc99224cfa0c6a72e94b99c57cdfc04e49ef4f2e21849a64"
EXPECTED_H217_HELPER_SHA256 = "ec5b94408a7670e727718d70d2e49dbb5225f5b3c6a3fdbd6fc83b8efbfede3a"
EXPECTED_SCORER_SHA256 = "b6377a0e1d67030115c6d20a64ee00dc97ef15eaad7a806408a76ef70558ab39"
EXPECTED_GMETRIC_SHA256 = "2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa"
ALPHAS = (0.0, 0.1, 0.25, 0.5, 1.0)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 271828
QualifiedGroup: TypeAlias = tuple[str, int]
AlignedRow: TypeAlias = tuple[QualifiedGroup, int, int, int]


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return cast(dict[str, object], value)


def _int_groups(value: object, name: str) -> list[int]:
    if not isinstance(value, list) or any(type(group) is not int or group < 0 for group in value):
        raise ValueError(f"{name} must be a list of nonnegative integer row groups")
    groups = cast(list[int], value)
    if len(groups) != len(set(groups)):
        raise ValueError(f"{name} contains duplicate row groups")
    return groups


def source_group_index(source: str, row_groups: Sequence[int]) -> dict[QualifiedGroup, int]:
    """Index ordered rows by source-qualified IDs, never bare row-group integers."""
    if source not in {"train", "valid"}:
        raise ValueError(f"unsupported dataset source: {source}")
    if any(type(group) is not int or group < 0 for group in row_groups):
        raise ValueError("row groups must be nonnegative integer IDs")
    keys = [(source, group) for group in row_groups]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate ({source}, row_group) identity")
    return {key: index for index, key in enumerate(keys)}


def align_evaluation_rows(
    evaluation_groups: Sequence[int],
    candidate_groups: Sequence[int],
    reference_groups: Sequence[int],
    target_groups: Sequence[int],
) -> list[AlignedRow]:
    """Join candidate, reference, and targets through valid-source row identities."""
    expected_keys = [("valid", group) for group in evaluation_groups]
    if len(expected_keys) != len(set(expected_keys)):
        raise ValueError("evaluation groups contain duplicate qualified identities")
    expected = set(expected_keys)
    candidate_index = source_group_index("valid", candidate_groups)
    reference_index = source_group_index("valid", reference_groups)
    target_index = source_group_index("valid", target_groups)
    if set(candidate_index) != expected:
        raise ValueError(
            "candidate prediction membership differs from the frozen evaluation groups"
        )
    if not expected.issubset(reference_index):
        raise ValueError("reference prediction is missing one or more evaluation groups")
    if not expected.issubset(target_index):
        raise ValueError("target cache is missing one or more evaluation groups")
    return [
        (key, candidate_index[key], reference_index[key], target_index[key])
        for key in expected_keys
    ]


def validate_candidate_fit_membership(
    split: dict[str, object], provenance: dict[str, object], evaluation_groups: Sequence[int]
) -> dict[str, object]:
    """Require exact source-qualified fit membership and component exclusion."""
    expected_fit_raw = _mapping(split.get("fit"), "split.fit")
    actual_fit_raw = _mapping(provenance.get("fit_row_groups"), "provenance.fit_row_groups")
    expected_fit = {
        source: set(_int_groups(groups, f"split.fit.{source}"))
        for source, groups in expected_fit_raw.items()
    }
    actual_fit = {
        source: set(_int_groups(groups, f"provenance.fit_row_groups.{source}"))
        for source, groups in actual_fit_raw.items()
    }
    if actual_fit != expected_fit:
        raise ValueError("candidate fit membership differs from the frozen source-qualified split")

    evaluation_keys = {("valid", group) for group in evaluation_groups}
    fit_keys = {(source, group) for source, groups in actual_fit.items() for group in groups}
    if fit_keys & evaluation_keys:
        raise ValueError("candidate fit includes a source-qualified evaluation group")

    sequences = _mapping(split.get("sequences"), "split.sequences")
    component_of: dict[QualifiedGroup, int] = {}
    for source, nodes_obj in sequences.items():
        if not isinstance(nodes_obj, list):
            raise ValueError(f"split.sequences.{source} must be a list")
        for node_obj in nodes_obj:
            node = _mapping(node_obj, f"split.sequences.{source}[]")
            group = node.get("row_group")
            component = node.get("component")
            if type(group) is not int or type(component) is not int:
                raise ValueError("split sequence nodes require integer row_group and component")
            key = (source, group)
            if key in component_of:
                raise ValueError(f"duplicate source-qualified split node: {key}")
            component_of[key] = component

    missing = (fit_keys | evaluation_keys) - component_of.keys()
    if missing:
        raise ValueError(f"split lacks component labels for {len(missing)} source-qualified groups")
    fit_components = {component_of[key] for key in fit_keys}
    evaluation_components = {component_of[key] for key in evaluation_keys}
    overlap = fit_components & evaluation_components
    if overlap:
        raise ValueError(f"candidate fit/evaluation component overlap ({len(overlap)})")

    recorded_evaluation = _int_groups(
        provenance.get("evaluation_row_groups"), "provenance.evaluation_row_groups"
    )
    if set(recorded_evaluation) != {group for _, group in evaluation_keys}:
        raise ValueError("candidate provenance evaluation groups differ from the frozen fold")
    return {
        "fit_groups_by_source": {source: len(groups) for source, groups in actual_fit.items()},
        "evaluation_groups": len(evaluation_keys),
        "fit_evaluation_source_qualified_overlap": 0,
        "fit_evaluation_component_overlap": 0,
        "evaluation_components": len(evaluation_components),
    }


def validate_prediction_array(
    prediction: np.ndarray, expected_rows: int, expected_timesteps: int = 20_000
) -> None:
    """Validate a memory-mapped FP32 prediction array without loading it all."""
    if prediction.shape != (expected_rows, expected_timesteps, 2):
        raise ValueError("prediction array has the wrong row/sequence/target shape")
    if prediction.dtype != np.float32:
        raise ValueError("prediction array must be float32")
    for index in range(expected_rows):
        if not np.isfinite(prediction[index]).all():
            raise ValueError(f"prediction array contains a nonfinite value in row {index}")


def raw_blend(reference: np.ndarray, candidate: np.ndarray, alpha: float) -> np.ndarray:
    """Mix raw predictions before the frozen metric applies its clipping."""
    if reference.shape != candidate.shape:
        raise ValueError("blend member shapes differ")
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("blend weight must be finite and in [0, 1]")
    return (1.0 - alpha) * reference + alpha * candidate


def score_predeclared_blends(
    targets: np.ndarray,
    masks: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    aligned_rows: Sequence[AlignedRow],
    components: np.ndarray,
    scorer: ModuleType,
    pair_helpers: ModuleType,
    *,
    n_boot: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Score every fixed blend using the frozen moments and component bootstrap."""
    if len(aligned_rows) == 0 or components.shape != (len(aligned_rows),):
        raise ValueError("aligned evaluation rows and component labels must be nonempty and equal")
    baseline_moments: list[np.ndarray] = []
    blend_moments: dict[float, list[np.ndarray]] = {alpha: [] for alpha in ALPHAS}
    pair_moments: list[list[np.ndarray]] = [[], []]
    selected_rows = 0

    for _, candidate_index, reference_index, target_index in aligned_rows:
        y = np.asarray(targets[target_index], dtype=np.float32)
        mask = np.asarray(masks[target_index], dtype=bool)
        candidate_rows = np.asarray(candidate[candidate_index], dtype=np.float32)
        reference_rows = np.asarray(reference[reference_index], dtype=np.float32)
        if y.shape != (candidate_rows.shape[0], 2) or reference_rows.shape != candidate_rows.shape:
            raise ValueError("candidate/reference/target sequence shapes do not align")
        if not np.isfinite(candidate_rows).all() or not np.isfinite(reference_rows).all():
            raise ValueError("nonfinite candidate or reference prediction sequence")
        if mask.shape != (y.shape[0],):
            raise ValueError("target scored mask has the wrong sequence length")
        need_prediction = np.arange(y.shape[0]) >= 99
        selected = need_prediction & mask
        if not np.isfinite(y[selected]).all():
            raise ValueError("nonfinite target value on the official scored/prediction mask")
        selected_rows += int(selected.sum())
        baseline_moments.append(
            np.stack(
                [
                    scorer.moments(y[:, target], reference_rows[:, target], selected)
                    for target in range(2)
                ]
            )
        )
        for alpha in ALPHAS:
            blend = raw_blend(reference_rows, candidate_rows, alpha)
            blend_moments[alpha].append(
                np.stack(
                    [
                        scorer.moments(y[:, target], blend[:, target], selected)
                        for target in range(2)
                    ]
                )
            )
        for target in range(2):
            pair_moments[target].append(
                pair_helpers.weighted_pair_moments(
                    candidate_rows[:, target], reference_rows[:, target], y[:, target], selected
                )
            )

    baseline_array = np.stack(baseline_moments)
    baseline_score = scorer.pooled(baseline_array)
    blends: list[dict[str, object]] = []
    for alpha in ALPHAS:
        moments = np.stack(blend_moments[alpha])
        score = scorer.pooled(moments)
        bootstrap = scorer.paired_component_bootstrap(
            moments,
            baseline_array,
            components,
            n_boot=n_boot,
            seed=bootstrap_seed,
            centered=True,
        )
        blends.append(
            {
                "alpha_on_h218": alpha,
                "score": score,
                "delta_vs_reference": {
                    name: score[name] - baseline_score[name] for name in ("t0", "t1", "wp")
                },
                "paired_component_bootstrap_vs_reference": bootstrap,
            }
        )
    pair_stats = {
        f"t{target}": pair_helpers.summarize_pair_moments(scorer, np.stack(pair_moments[target]))
        for target in range(2)
    }
    return {
        "reference_baseline_official_score": baseline_score,
        "predeclared_blends_all_reported": blends,
        "pair_prediction_covariance_correlation": pair_stats,
        "selected_rows": selected_rows,
        "group_count": len(aligned_rows),
    }


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repo_file(root: Path, entry: object, name: str) -> tuple[Path, dict[str, object]]:
    item = _mapping(entry, f"authorization.candidate_artifacts.{name}")
    relative = item.get("path")
    expected = item.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"authorization artifact {name} needs a repo path and SHA256")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"authorization artifact {name} is missing or outside the repository")
    if sha256_file(path) != expected:
        raise ValueError(f"authorization artifact {name} SHA256 mismatch")
    return path, item


def preregistered_fit_attempt_id(prereg: dict[str, object]) -> str:
    """Read the fit identity from the frozen preregistration's top-level field."""
    fit_attempt_id = prereg.get("fit_attempt_id")
    if not isinstance(fit_attempt_id, str) or not fit_attempt_id:
        raise ValueError("pre-registration lacks its top-level fit_attempt_id")
    return fit_attempt_id


def validate_attempt_terminal_record(
    receipt: dict[str, object], expected_attempt_id: str, receipt_name: str
) -> None:
    """Require the actual bandit attempt-record terminal fields and identity."""
    observed_id = receipt.get("id", receipt.get("attempt_id"))
    if (
        observed_id != expected_attempt_id
        or receipt.get("status") != "finished"
        or receipt.get("exit_status") != 0
        or receipt.get("phase") != "terminal"
    ):
        raise ValueError(f"{receipt_name} is not the successful terminal bandit attempt record")


def validate_root_terminal_attestation(
    approval: dict[str, object],
    receipt: dict[str, object],
    expected_fit_attempt_id: str,
    expected_record_id: str,
    receipt_name: str,
) -> None:
    """Pair root's normalized success attestation with the raw bandit receipt."""
    if (
        approval.get("fit_attempt_id") != expected_fit_attempt_id
        or approval.get("exit_code") != 0
        or approval.get("terminal_status") != "success"
        or not isinstance(approval.get("checkpoint_sha256"), str)
    ):
        raise ValueError(f"root admission does not attest successful {receipt_name}")
    validate_attempt_terminal_record(receipt, expected_record_id, receipt_name)


def validate_export_sidecar(
    sidecar: dict[str, object],
    expected_fields: dict[str, object],
    expected_groups: Sequence[int],
) -> None:
    """Check the H218 exporter's emitted metadata before admitting its array."""
    required_fields = {
        "schema": "h218-fold0-feature-only-prediction-v1",
        "independent_grading_only": True,
        **expected_fields,
    }
    for name, value in required_fields.items():
        if sidecar.get(name) != value:
            raise ValueError(f"candidate export sidecar field differs from preregistration: {name}")
    if sidecar.get("labels_read") is not False or sidecar.get("targets_read") is not False:
        raise ValueError("candidate exporter read scoring labels or targets")
    groups = _int_groups(
        sidecar.get("evaluation_row_groups"), "candidate sidecar evaluation_row_groups"
    )
    if groups != list(expected_groups):
        raise ValueError("candidate predictions are not in the exact sorted evaluation membership")


def validate_root_admission(
    root: Path, authorization_path: Path, script_path: Path, prereg_path: Path
) -> tuple[dict[str, object], dict[str, tuple[Path, dict[str, object]]]]:
    """Require root approval bound to this grader and the actual delivered artifacts."""
    if not authorization_path.is_file():
        raise FileNotFoundError("root grade admission is absent; actual scoring is not authorized")
    authorization = _mapping(json.loads(authorization_path.read_text()), "grade admission")
    if (
        authorization.get("schema") != "h218-fold0-reference-blend-grade-admission-v1"
        or authorization.get("status") != "approved"
        or authorization.get("approved_by") != "root"
        or authorization.get("preregistration_sha256") != EXPECTED_PREREG_SHA256
        or authorization.get("grader_sha256") != sha256_file(script_path)
        or authorization.get("preregistration_record_id")
        != "H218_FOLD0_REFERENCE_BLEND_PREREG_20261008_R1"
        or sha256_file(prereg_path) != EXPECTED_PREREG_SHA256
    ):
        raise ValueError("root grade admission does not bind the frozen preregistration/grader")
    artifacts_obj = _mapping(authorization.get("candidate_artifacts"), "candidate_artifacts")
    required = {
        "fit_terminal_receipt",
        "export_terminal_receipt",
        "checkpoint",
        "provenance",
        "prediction",
        "sidecar",
        "source_lock",
    }
    if set(artifacts_obj) != required:
        raise ValueError("root grade admission has an incomplete or unexpected artifact set")
    artifacts = {name: _repo_file(root, artifacts_obj[name], name) for name in sorted(required)}
    prereg = _mapping(json.loads(prereg_path.read_text()), "pre-registration")
    fit_id = preregistered_fit_attempt_id(prereg)
    export_approval = artifacts["export_terminal_receipt"][1]
    for name in ("fit_terminal_receipt", "export_terminal_receipt"):
        approval = artifacts[name][1]
        expected_record_id = (
            fit_id if name == "fit_terminal_receipt" else approval.get("attempt_id")
        )
        if not isinstance(expected_record_id, str) or not expected_record_id:
            raise ValueError(f"root admission lacks the raw attempt id for {name}")
        receipt = _mapping(json.loads(artifacts[name][0].read_text()), name)
        validate_root_terminal_attestation(approval, receipt, fit_id, expected_record_id, name)
    if any(
        not isinstance(export_approval.get(name), str)
        for name in ("prediction_sha256", "sidecar_sha256", "source_lock_sha256")
    ):
        raise ValueError("root admission export attestation is missing delivered artifact hashes")
    return authorization, artifacts


def validate_source_lock(source_lock_path: Path, sidecar: dict[str, object]) -> dict[str, str]:
    """Rehash the reviewed H218 inference closure and bind the export sidecar."""
    lock = _mapping(json.loads(source_lock_path.read_text()), "source lock")
    files = _mapping(lock.get("files"), "source lock files")
    actual: dict[str, str] = {}
    for relative, expected in files.items():
        if not isinstance(expected, str):
            raise ValueError(f"source lock hash is malformed: {relative}")
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"locked source is missing or outside the repository: {relative}")
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(f"locked source changed after H218 export: {relative}")
        actual[relative] = digest
    expected_source_sha = {**actual, "source_lock": sha256_file(source_lock_path)}
    sidecar_source_sha = _mapping(sidecar.get("source_sha256"), "export sidecar source_sha256")
    if sidecar_source_sha != expected_source_sha:
        raise ValueError(
            "candidate export sidecar source hashes differ from the reviewed source lock"
        )
    return actual


def _check_frozen_inputs(
    prereg: dict[str, object],
) -> tuple[dict[str, object], ModuleType, ModuleType]:
    """Validate fixed public inputs and load the existing scorer/reference helpers."""
    design = _mapping(prereg.get("comparison_design"), "comparison_design")
    split_record = _mapping(design.get("split_manifest"), "split_manifest")
    target_record = _mapping(design.get("target_cache"), "target_cache")
    reference = _mapping(prereg.get("reference"), "reference")
    scoring_sources = _mapping(prereg.get("scoring_sources"), "scoring_sources")
    expected = {
        SPLIT_PATH: split_record.get("sha256"),
        TARGET_PATH: target_record.get("sha256"),
        SOURCE_FAMILIES: _mapping(reference.get("ancestry_manifest"), "ancestry_manifest").get(
            "sha256"
        ),
        H217_REFERENCE_HELPERS: EXPECTED_H217_HELPER_SHA256,
        SCORER_PATH: scoring_sources.get("research/campaign_2026-09-30/audit/campaign_score.py"),
        GMETRIC_PATH: scoring_sources.get("src/wnn/gmetric.py"),
    }
    for path, digest in expected.items():
        if not path.is_file() or not isinstance(digest, str) or sha256_file(path) != digest:
            raise ValueError(f"frozen grading input is absent or changed: {path.relative_to(ROOT)}")
    if sha256_file(SCORER_PATH) != EXPECTED_SCORER_SHA256:
        raise ValueError("campaign scorer SHA256 differs from the registered implementation")
    if sha256_file(GMETRIC_PATH) != EXPECTED_GMETRIC_SHA256:
        raise ValueError("official metric implementation SHA256 differs from preregistration")
    scorer = _load_module("h218_frozen_campaign_score", SCORER_PATH)
    if getattr(scorer, "METRIC_ID", None) != "global_wp_clip2_v1":
        raise ValueError("frozen campaign scorer has the wrong metric identity")
    helper = _load_module("h218_h217_reference_helpers", H217_REFERENCE_HELPERS)
    split = _mapping(json.loads(SPLIT_PATH.read_text()), "fold-0 split")
    return split, scorer, helper


def _validate_candidate(
    prereg: dict[str, object],
    split: dict[str, object],
    artifacts: dict[str, tuple[Path, dict[str, object]]],
) -> tuple[np.ndarray, dict[str, object], dict[str, object]]:
    """Bind the fixed epoch-3 checkpoint, source-qualified provenance and export."""
    candidate = _mapping(prereg.get("candidate"), "candidate")
    candidate_path = candidate.get("prediction_path")
    sidecar_path = candidate.get("prediction_sidecar_path")
    checkpoint_path = candidate.get("checkpoint_path")
    if not all(isinstance(value, str) for value in (candidate_path, sidecar_path, checkpoint_path)):
        raise ValueError("pre-registration lacks fixed candidate paths")
    expected_paths = {
        "prediction": candidate_path,
        "sidecar": sidecar_path,
        "checkpoint": checkpoint_path,
        "provenance": str(Path(checkpoint_path).parent / "provenance.json"),
        "source_lock": "research/campaign_2026-10-06/bandit_continuation/attention_training_H218/H218_FOLD0_EXPORT_SOURCE_LOCK_V1_20261008.json",
    }
    for name, expected_path in expected_paths.items():
        actual = str(artifacts[name][0].relative_to(ROOT))
        if actual != expected_path:
            raise ValueError(f"root admission {name} path differs from frozen H218 contract")
    checkpoint_sha = artifacts["checkpoint"][1]["sha256"]
    prediction_sha = artifacts["prediction"][1]["sha256"]
    sidecar = _mapping(json.loads(artifacts["sidecar"][0].read_text()), "candidate sidecar")
    provenance = _mapping(
        json.loads(artifacts["provenance"][0].read_text()), "candidate provenance"
    )
    source_lock_path = artifacts["source_lock"][0]
    if not isinstance(checkpoint_sha, str) or not isinstance(prediction_sha, str):
        raise ValueError("root admission checkpoint/prediction hashes are malformed")

    split_hash = _mapping(
        _mapping(prereg.get("comparison_design"), "comparison_design").get("split_manifest"),
        "split_manifest",
    ).get("sha256")
    run_name = Path(checkpoint_path).parent.name
    fit_id = preregistered_fit_attempt_id(prereg)
    fit_approval = artifacts["fit_terminal_receipt"][1]
    export_approval = artifacts["export_terminal_receipt"][1]
    if (
        fit_approval.get("checkpoint_sha256") != checkpoint_sha
        or export_approval.get("checkpoint_sha256") != checkpoint_sha
        or export_approval.get("prediction_sha256") != prediction_sha
        or export_approval.get("sidecar_sha256") != artifacts["sidecar"][1]["sha256"]
        or export_approval.get("source_lock_sha256") != artifacts["source_lock"][1]["sha256"]
    ):
        raise ValueError("fit/export terminal receipts do not bind the exact H218 artifacts")
    expected_sidecar = {
        "run": run_name,
        "fit_attempt_id": fit_id,
        "epoch": candidate.get("fixed_epoch_zero_based"),
        "checkpoint_sha256": checkpoint_sha,
        "prediction_sha256": prediction_sha,
        "prediction_path": candidate_path,
        "prediction_shape": candidate.get("shape"),
        "prediction_dtype": "float32",
        "split_manifest_sha256": split_hash,
        "evaluation_group_count": 468,
        "evaluation_group_ids_sha256": candidate.get("row_groups_sha256"),
        "inference_precision": "fp32",
        "official_score_computed": False,
    }
    validate_export_sidecar(
        sidecar,
        expected_sidecar,
        _int_groups(
            _mapping(split.get("evaluation"), "split.evaluation").get("valid"),
            "split.evaluation.valid",
        ),
    )
    if not isinstance(split_hash, str) or provenance.get("split_manifest_sha256") != split_hash:
        raise ValueError("candidate provenance does not bind the frozen split")
    if provenance.get("selected_epoch") != 3:
        raise ValueError("candidate provenance did not select fixed epoch 3")
    checkpoint_hashes = _mapping(provenance.get("checkpoint_sha256"), "checkpoint_sha256")
    if checkpoint_hashes.get("model_ep3.pt") != checkpoint_sha:
        raise ValueError(
            "candidate provenance does not bind the delivered fixed epoch-3 checkpoint"
        )
    args = _mapping(provenance.get("args"), "candidate provenance args")
    expected_args = {
        "name": run_name,
        "arch": "block-memory",
        "hidden": 128,
        "layers": 2,
        "block_heads": 4,
        "block_rows": 256,
        "batch": 4,
        "chunk": 2000,
        "seed": 7,
        "train_source": "train+valid",
        "valid_fold": 0,
        "epochs": 4,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 3,
        "no_eval_during_fit": True,
        "privileged_mode": "none",
    }
    if any(args.get(name) != value for name, value in expected_args.items()):
        raise ValueError("candidate provenance arguments differ from the frozen H218 recipe")
    if args.get("init") not in (None, "", "none"):
        raise ValueError("candidate H218 checkpoint is not from fresh scratch weights")

    evaluation_groups = _int_groups(
        _mapping(split.get("evaluation"), "split.evaluation").get("valid"),
        "split.evaluation.valid",
    )
    if len(evaluation_groups) != 468 or evaluation_groups != sorted(evaluation_groups):
        raise ValueError("fold-0 evaluation must be the exact sorted 468-group panel")
    candidate_groups = _int_groups(
        sidecar.get("evaluation_row_groups"), "candidate sidecar evaluation_row_groups"
    )
    digest = hashlib.sha256(np.asarray(candidate_groups, dtype="<i8").tobytes()).hexdigest()
    if digest != candidate.get("row_groups_sha256"):
        raise ValueError("candidate row-group digest differs from preregistration")
    membership = validate_candidate_fit_membership(split, provenance, evaluation_groups)
    if sha256_file(EXPORT_CONTRACT) != EXPECTED_EXPORT_CONTRACT_SHA256:
        raise ValueError("H218 export contract changed after independent source review")
    export_contract = _mapping(json.loads(EXPORT_CONTRACT.read_text()), "export contract")
    lock_hash = _mapping(export_contract.get("export"), "export contract.export").get(
        "source_lock_sha256"
    )
    if (
        lock_hash != EXPECTED_SOURCE_LOCK_SHA256
        or not isinstance(lock_hash, str)
        or sha256_file(source_lock_path) != lock_hash
    ):
        raise ValueError("H218 source-lock bytes differ from the frozen export contract")
    source_hashes = validate_source_lock(source_lock_path, sidecar)
    prediction = np.load(artifacts["prediction"][0], mmap_mode="r", allow_pickle=False)
    validate_prediction_array(prediction, 468)
    if sidecar.get("prediction_bytes") != artifacts["prediction"][0].stat().st_size:
        raise ValueError("candidate sidecar prediction byte count differs from the delivered file")
    return (
        prediction,
        sidecar,
        {
            "checkpoint_sha256": checkpoint_sha,
            "prediction_sha256": prediction_sha,
            "provenance_sha256": artifacts["provenance"][1]["sha256"],
            "fit_terminal_receipt_sha256": artifacts["fit_terminal_receipt"][1]["sha256"],
            "export_terminal_receipt_sha256": artifacts["export_terminal_receipt"][1]["sha256"],
            "sidecar_sha256": artifacts["sidecar"][1]["sha256"],
            "source_lock_sha256": lock_hash,
            "source_file_count": len(source_hashes),
            "membership": membership,
        },
    )


def _score(root_approval: Path) -> dict[str, object]:
    """Run the authorized comparison and write the frozen all-weights report."""
    if REPORT.exists() or MARKDOWN.exists():
        raise FileExistsError("H218 blend grade output already exists; refusing overwrite")
    prereg_sha = sha256_file(PREREG)
    if prereg_sha != EXPECTED_PREREG_SHA256:
        raise ValueError("H218 blend preregistration changed")
    prereg = _mapping(json.loads(PREREG.read_text()), "pre-registration")
    authorization, artifacts = validate_root_admission(ROOT, root_approval, Path(__file__), PREREG)
    split, scorer, helpers = _check_frozen_inputs(prereg)

    reference = _mapping(prereg.get("reference"), "reference")
    reference_prediction_path = ROOT / str(reference["prediction_path"])
    reference_sidecar_path = ROOT / str(reference["sidecar_path"])
    reference_checkpoint_path = ROOT / str(reference["checkpoint_path"])
    reference_provenance_path = ROOT / str(reference["provenance_path"])
    for path, key in (
        (reference_prediction_path, "prediction_sha256"),
        (reference_sidecar_path, "sidecar_sha256"),
        (reference_checkpoint_path, "checkpoint_sha256"),
        (reference_provenance_path, "provenance_sha256"),
    ):
        if not path.is_file() or sha256_file(path) != reference.get(key):
            raise ValueError(
                f"own reference artifact changed or is missing: {path.relative_to(ROOT)}"
            )
    reference_sidecar = _mapping(
        json.loads(reference_sidecar_path.read_text()), "reference sidecar"
    )
    if (
        reference_sidecar.get("run") != reference.get("run")
        or reference_sidecar.get("epoch") != reference.get("epoch_zero_based")
        or reference_sidecar.get("checkpoint_sha256") != reference.get("checkpoint_sha256")
        or reference_sidecar.get("prediction_sha256") != reference.get("prediction_sha256")
        or reference_sidecar.get("split_manifest_sha256") != reference.get("split_manifest_sha256")
    ):
        raise ValueError("own reference prediction sidecar differs from preregistration")
    lineage = helpers.verify_reference_lineage(split, reference_sidecar)
    if lineage["source_families_sha256"] != _mapping(
        reference.get("ancestry_manifest"), "reference ancestry_manifest"
    ).get("sha256"):
        raise ValueError("own reference source-family lineage differs from preregistration")
    reference_groups = _int_groups(
        reference_sidecar.get("row_groups"), "reference sidecar row_groups"
    )
    reference_digest = hashlib.sha256(
        np.asarray(reference_groups, dtype="<i8").tobytes()
    ).hexdigest()
    if reference_digest != reference.get("row_groups_sha256"):
        raise ValueError("reference row-group digest differs from preregistration")

    candidate_prediction, candidate_sidecar, candidate_binding = _validate_candidate(
        prereg, split, artifacts
    )
    scorer_targets = scorer.mmap_stored_npz(TARGET_PATH)
    target_groups = _int_groups(scorer_targets["row_group"].tolist(), "target-cache row_group")
    valid_nodes = _mapping(split.get("sequences"), "split.sequences").get("valid")
    if not isinstance(valid_nodes, list):
        raise ValueError("fold-0 split has no valid-source sequence rows")
    all_valid_groups = [
        _mapping(node, "split.sequences.valid[]").get("row_group") for node in valid_nodes
    ]
    valid_groups = _int_groups(all_valid_groups, "split.sequences.valid row groups")
    if set(target_groups) != set(valid_groups) or set(reference_groups) != set(valid_groups):
        raise ValueError("reference/target cache rows do not cover exact valid-source membership")
    if len(reference_groups) != 1873:
        raise ValueError(
            "own reference prediction is not the registered full 1,873-row valid export"
        )
    expected_eval_groups = _int_groups(
        _mapping(split.get("evaluation"), "split.evaluation").get("valid"),
        "split.evaluation.valid",
    )
    alignment = align_evaluation_rows(
        expected_eval_groups,
        _int_groups(candidate_sidecar.get("evaluation_row_groups"), "candidate groups"),
        reference_groups,
        target_groups,
    )
    reference_prediction = np.load(reference_prediction_path, mmap_mode="r", allow_pickle=False)
    validate_prediction_array(reference_prediction, len(reference_groups))
    component_map = {
        (source, int(node["row_group"])): int(node["component"])
        for source, nodes in _mapping(split.get("sequences"), "split.sequences").items()
        for node in cast(list[dict[str, object]], nodes)
    }
    components = np.asarray([component_map[key] for key, _, _, _ in alignment], dtype=np.int64)
    score_result = score_predeclared_blends(
        scorer_targets["y"],
        scorer_targets["mask"],
        candidate_prediction,
        reference_prediction,
        alignment,
        components,
        scorer,
        helpers,
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "record_id": "H218_FOLD0_REFERENCE_BLEND_20261008_R1",
        "created_utc": datetime.now(UTC).isoformat(),
        "grade_status": "conditional_exposed_discovery_comparison",
        "metric_id": "global_wp_clip2_v1",
        "preregistration_sha256": prereg_sha,
        "evaluation": {
            "fold": 0,
            "evaluation_groups": 468,
            "source_qualified_join": "(valid, row_group); candidate-only 468 rows join the full-valid reference and target cache by source-qualified identity",
            "selected_rows": score_result["selected_rows"],
            "split_manifest_sha256": _mapping(
                _mapping(prereg.get("comparison_design"), "comparison_design").get(
                    "split_manifest"
                ),
                "split_manifest",
            ).get("sha256"),
            "selection": "need_prediction AND is_scored; need_prediction is row index >= 99",
            "exposure": "exposed discovery fold; no fresh confirmation",
        },
        "candidate": {
            **_mapping(prereg.get("candidate"), "candidate"),
            **candidate_binding,
        },
        "reference": reference,
        "reference_baseline_official_score": score_result["reference_baseline_official_score"],
        "predeclared_blends_all_reported": score_result["predeclared_blends_all_reported"],
        "pair_prediction_covariance_correlation": {
            "definition": _mapping(prereg.get("comparison_design"), "comparison_design").get(
                "pair_covariance_and_correlation"
            ),
            "per_target": score_result["pair_prediction_covariance_correlation"],
        },
        "lineage_validation": lineage,
        "uncertainty_and_limits": {
            "bootstrap": "paired component bootstrap over 468 frozen fold-0 groups, 2,000 replicates, seed 271828",
            "conditional_on_fixed_predictions": True,
            "training_seed_uncertainty_estimated": False,
            "public_transfer_estimated": False,
            "blend_weight_selection": "none; all five preregistered weights are reported",
            "all_data_comparison": None,
            "public_forecast": None,
            "comparability_limits": prereg.get("comparability_limits"),
        },
        "verified_artifacts": {
            "grader_sha256": sha256_file(Path(__file__)),
            "grade_admission_sha256": sha256_file(root_approval),
            "pre_registration_sha256": prereg_sha,
            "candidate": candidate_binding,
            "reference_prediction_sha256": sha256_file(reference_prediction_path),
            "reference_sidecar_sha256": sha256_file(reference_sidecar_path),
            "target_cache_sha256": sha256_file(TARGET_PATH),
            "split_manifest_sha256": sha256_file(SPLIT_PATH),
            "source_families_sha256": sha256_file(SOURCE_FAMILIES),
            "frozen_campaign_scorer_sha256": sha256_file(SCORER_PATH),
            "official_metric_source_sha256": sha256_file(GMETRIC_PATH),
            "h217_reference_helpers_sha256": sha256_file(H217_REFERENCE_HELPERS),
        },
        "method": "Frozen campaign scorer; target arrays are memory-mapped from stored NPZ members; candidate/reference predictions are NPY memory maps. No Parquet, NPZ decompression, fit, GPU, provider action, all-data comparison, or public forecast.",
        "root_authorization": {
            "status": authorization.get("status"),
            "approved_by": authorization.get("approved_by"),
        },
    }
    lines = [
        "# H218 fold-0 reference blends",
        "",
        "This conditional comparison covers the same 468 exposed fold-0 groups for H218 and the own seed-3 GRU reference. The candidate contains only those 468 groups; its rows are joined to the reference's 1,873-row valid export and the target cache by `(valid, row_group)`, not by shared array position.",
        "",
        f"Reference baseline: WP {cast(dict[str, float], score_result['reference_baseline_official_score'])['wp']:.9f}.",
        "",
        "| H218 weight | WP | t0 | t1 | delta vs reference | paired component 95% CI |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for entry_obj in cast(list[dict[str, object]], score_result["predeclared_blends_all_reported"]):
        score = cast(dict[str, float], entry_obj["score"])
        delta = cast(dict[str, float], entry_obj["delta_vs_reference"])
        bootstrap = cast(dict[str, object], entry_obj["paired_component_bootstrap_vs_reference"])
        ci = cast(list[float], cast(dict[str, object], bootstrap)["ci95"])
        lines.append(
            f"| {cast(float, entry_obj['alpha_on_h218']):.2f} | {score['wp']:.9f} | "
            f"{score['t0']:.9f} | {score['t1']:.9f} | {delta['wp']:+.9f} | "
            f"[{ci[0]:+.9f}, {ci[1]:+.9f}] |"
        )
    pair_stats = cast(
        dict[str, dict[str, float]], score_result["pair_prediction_covariance_correlation"]
    )
    lines.extend(
        [
            "",
            "Pair prediction relationship on the official scored/prediction mask and clipped-target magnitude weights:",
            "",
            *[
                f"- {target}: weighted covariance {stats['weighted_population_covariance']:.9g}; "
                f"weighted correlation {stats['weighted_pearson_correlation']:.9f}"
                for target, stats in pair_stats.items()
            ],
            "",
            "All fixed weights are reported without selecting a winner. The interval is conditional on fixed predictions and exposed fold-0 components; it does not estimate training-seed or public-transfer uncertainty. This is not a public forecast or all-data comparison.",
        ]
    )
    with REPORT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with MARKDOWN.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return result


def main() -> int:
    """Run only after root admission binds the final H218 artifacts and this grader."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, default=AUTHORIZATION)
    args = parser.parse_args()
    result = _score(args.authorization.resolve())
    print(
        json.dumps(
            {
                "report": str(REPORT.relative_to(ROOT)),
                "grade_status": result["grade_status"],
                "evaluation_groups": 468,
                "blend_weights_reported": list(ALPHAS),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
