"""Report-only scoring of an auxiliary-trained endpoint on exposed primary valid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from wnn.data import WARMUP
from wnn.gmetric import GlobalAccumulator
from wnn.splits import (
    CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
    MULTISTAGE_AUXILIARY_SPLIT_SCHEMA,
    CampaignSplit,
    file_sha256,
    load_campaign_split,
)
from wnn.training.sequence_cache_ensemble_provenance import common_provenance
from wnn.training.sequence_data import SequenceRecords
from wnn.training.sequence_fit_utils import SequenceDataset, collate_sequences
from wnn.training.sequence_monitor_identity import membership_sha256
from wnn.training.sequence_precision import configure_precision_mode
from wnn.training.sequence_predict import load_endpoint, predict_sequence_batch

PRIMARY_SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
AUXILIARY_SPLIT_SHA256 = "26b448bad7f60612cea482f4abaf5f184e5a724228eef227d7e78a8155190075"
PRIMARY_TRAIN_COUNT = 10_325
AUXILIARY_FIT_COUNT = 8_260
AUXILIARY_CONTROL_COUNT = 2_065
PRIMARY_VALID_COUNT = 1_873
PRIMARY_SCORED_ROW_COUNT = 3_528_273
TL3_REFERENCE_WP = 0.6923774245652725
REPORT_SCHEMA = "wnn.multistage-primary-report.v1"
CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "research/experiment_drafts/multistage-optuna-b300-20261031/CONTRACT.json"
)


class _StreamingMetrics:
    """Accumulate the canonical pooled WP and clipped-target weighted MSE."""

    def __init__(self) -> None:
        self.wp = GlobalAccumulator()
        self.squared_error = np.zeros(2, dtype=np.float64)
        self.target_weight = np.zeros(2, dtype=np.float64)

    def add(self, target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> None:
        """Add one source-ordered sequence batch without retaining its full predictions."""
        y = np.asarray(target, dtype=np.float32)
        p = np.asarray(prediction, dtype=np.float32)
        selected = np.asarray(mask, dtype=bool)
        self.wp.add(y, p, selected)
        clipped = np.clip(y.astype(np.float64), -2.0, 2.0)
        weights = np.abs(clipped) * selected[..., None]
        difference = p.astype(np.float64) - clipped
        self.squared_error += np.sum(weights * np.square(difference), axis=(0, 1))
        self.target_weight += np.sum(weights, axis=(0, 1))

    def result(self) -> dict[str, float | int | str]:
        """Return pooled WP and target-weighted MSE using the canonical reductions."""
        wp = self.wp.result()
        per_target = np.divide(
            self.squared_error,
            self.target_weight,
            out=np.zeros(2, dtype=np.float64),
            where=self.target_weight > 0,
        )
        total_weight = float(self.target_weight.sum())
        return {
            "metric_id": wp["metric_id"],
            "wp": float(wp["wp"]),
            "t0": float(wp["t0"]),
            "t1": float(wp["t1"]),
            "selected_rows": int(wp["selected_rows"]),
            "wmse": (float(self.squared_error.sum() / total_weight) if total_weight > 0 else 0.0),
            "wmse_t0": float(per_target[0]),
            "wmse_t1": float(per_target[1]),
        }


def validate_auxiliary_report_membership(
    data: dict[str, Any],
    config: dict[str, Any],
    primary: CampaignSplit,
    auxiliary: CampaignSplit,
    primary_sources: dict[str, Any],
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Require a label-only auxiliary fit wholly contained in official train."""
    if (
        primary.schema != CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA
        or primary.valid_groups
        or primary.sha256 != PRIMARY_SPLIT_SHA256
        or auxiliary.schema != MULTISTAGE_AUXILIARY_SPLIT_SCHEMA
        or auxiliary.sha256 != AUXILIARY_SPLIT_SHA256
        or auxiliary.valid_groups
        or auxiliary.primary_split_sha256 != primary.sha256
        or len(primary.train_groups) != PRIMARY_TRAIN_COUNT
        or len(primary.evaluation_groups) != PRIMARY_VALID_COUNT
        or len(auxiliary.train_groups) != AUXILIARY_FIT_COUNT
        or len(auxiliary.evaluation_groups) != AUXILIARY_CONTROL_COUNT
        or primary.evaluation_source != "valid"
        or primary.evaluation_mask != "is_scored"
        or auxiliary.evaluation_source != "train"
        or auxiliary.evaluation_role != "control"
        or auxiliary.evaluation_mask != "all_after_warmup"
    ):
        raise ValueError("primary or auxiliary split differs from the frozen B300 report protocol")

    fit_members = [("train", group) for group in auxiliary.train_groups]
    control_members = [("train", group) for group in auxiliary.evaluation_groups]
    primary_train = {("train", group) for group in primary.train_groups}
    if (
        len(primary.train_groups) != len(set(primary.train_groups))
        or not set(fit_members).issubset(primary_train)
        or not set(control_members).issubset(primary_train)
        or set(fit_members).intersection(control_members)
        or set(fit_members).union(control_members) != primary_train
    ):
        raise ValueError("auxiliary fit/control cohorts do not partition official train")

    model_config = config.get("model")
    if (
        not isinstance(model_config, dict)
        or config.get("precision_mode") != "fp32"
        or type(model_config.get("hint_dim")) is not int
        or model_config["hint_dim"] != 0
        or type(config.get("kd_weight")) not in (int, float)
        or config["kd_weight"] != 0
        or type(config.get("hint_weight")) not in (int, float)
        or config["hint_weight"] != 0
    ):
        raise ValueError("primary report requires a direct strict-FP32 labels-only endpoint")
    expected_fit = [list(member) for member in fit_members]
    expected_control = [list(member) for member in control_members]
    expected_sources = {key: value for key, value in primary_sources.items()}
    normalization = data.get("normalization")
    fingerprints = (
        normalization.get("source_fingerprints") if isinstance(normalization, dict) else None
    )
    if (
        data.get("split_sha256") != auxiliary.sha256
        or data.get("primary_split_sha256") != primary.sha256
        or data.get("fit_members") != expected_fit
        or data.get("evaluation_members") != expected_control
        or data.get("evaluation_role") != "control"
        or data.get("evaluation_source") != "train"
        or data.get("evaluation_mask") != "all_after_warmup"
        or data.get("training_mask") != "all_after_warmup"
        or data.get("mask") != "all_after_warmup AND row_index>=99"
        or data.get("sources") != expected_sources
        or data.get("teacher") is not None
        or data.get("initializer") is not None
        or data.get("hint_members") is not None
        or not isinstance(fingerprints, dict)
        or set(fingerprints) != {"datasets/train.parquet"}
    ):
        raise ValueError(
            "checkpoint ancestry or fit/control membership violates the report protocol"
        )
    return fit_members, control_members


def _validate_endpoint(
    checkpoint: Path,
    payload: dict[str, Any],
    primary: CampaignSplit,
    auxiliary: CampaignSplit,
    primary_sources: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Validate fixed-final ancestry and return the registered producer identity."""
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("report checkpoint must be a regular fixed-fit model file")
    config, data, provenance = (
        payload.get("config"),
        payload.get("data"),
        payload.get("provenance"),
    )
    if (
        not isinstance(config, dict)
        or not isinstance(data, dict)
        or not isinstance(provenance, dict)
    ):
        raise ValueError("report checkpoint is missing config, data or provenance")
    epochs = config.get("epochs")
    if (
        payload.get("checkpoint_schema") != "wnn.sequence-fit.v1"
        or type(epochs) is not int
        or payload.get("completed_epochs") != epochs
        or payload.get("epoch") != epochs - 1
        or provenance.get("fixed_checkpoint_policy") != "configured-final-epoch"
    ):
        raise ValueError("primary report accepts only a fixed configured-final endpoint")
    fit_members, _ = validate_auxiliary_report_membership(
        data, config, primary, auxiliary, primary_sources
    )
    checkpoint_sha256 = file_sha256(checkpoint)
    producer = {
        "checkpoint_sha256": checkpoint_sha256,
        "split_sha256": auxiliary.sha256,
        "fit_members": [list(member) for member in fit_members],
        "teacher": data.get("teacher"),
        "config": config,
        "provenance": provenance,
    }
    common_provenance(producer, provenance)
    return producer, checkpoint_sha256


def _load_contract(path: Path) -> tuple[dict[str, Any], str]:
    """Load the frozen experiment contract that supplies the exposed TL3 reference."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_000_000:
        raise ValueError("experiment contract must be a small regular file")
    raw = path.read_bytes()
    contract = json.loads(raw)
    if (
        not isinstance(contract, dict)
        or contract.get("schema") != "wnn.multistage-optuna-contract.v1"
        or contract.get("experiment_id") != "E-MULTISTAGE-OPTUNA-B300-20261031-R1"
    ):
        raise ValueError("unsupported multistage experiment contract")
    policy, baseline = contract.get("data_policy"), contract.get("baseline")
    if (
        not isinstance(policy, dict)
        or policy.get("primary_split_sha256") != PRIMARY_SPLIT_SHA256
        or policy.get("primary_fit_train_sequences") != PRIMARY_TRAIN_COUNT
        or policy.get("primary_fit_valid_sequences") != 0
        or policy.get("primary_eval_valid_sequences") != PRIMARY_VALID_COUNT
        or policy.get("primary_eval_scored_rows") != PRIMARY_SCORED_ROW_COUNT
        or not isinstance(baseline, dict)
        or type(baseline.get("canonical_fresh_TL3_wp")) not in (int, float)
        or not math.isfinite(float(baseline["canonical_fresh_TL3_wp"]))
        or baseline["canonical_fresh_TL3_wp"] != TL3_REFERENCE_WP
        or baseline.get("original_results")
        != "research/experiment_drafts/heterogeneous-tl3-20261030/results/R3/REPORT.json"
        or baseline.get("exposure")
        != "development-exposed full official valid; no hidden/public forecast"
    ):
        raise ValueError("contract data policy or TL3 baseline differs from the report target")
    return contract, hashlib.sha256(raw).hexdigest()


def _portable_registry_sha256() -> tuple[str, str]:
    """Return the current portable registry path and exact receipt digest."""
    registry_value = os.environ.get("WNN_PORTABLE_DATA_MANIFEST")
    if not registry_value:
        raise ValueError("primary reporting requires the verified portable data registry")
    registry = Path(registry_value)
    if registry.is_symlink() or not registry.is_file():
        raise ValueError("portable data registry must be a regular file")
    return str(registry.resolve()), file_sha256(registry)


def _score_primary_valid(
    model: torch.nn.Module,
    dataset: SequenceRecords,
    chunk_rows: int,
    batch_size: int,
    device: torch.device,
    prediction_temporary: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Stream all official valid sequences through the canonical batched predictor."""
    if not dataset.members or chunk_rows < 1 or batch_size < 1:
        raise ValueError(
            "primary report requires nonempty membership and positive batch/chunk sizes"
        )
    if dataset.training_mask != "is_scored":
        raise ValueError("primary valid scoring must use the official is_scored mask")
    metrics = _StreamingMetrics()
    mask_digest = hashlib.sha256()
    selected_index_digest = hashlib.sha256()
    prediction_map = None
    if prediction_temporary is not None:
        prediction_map = np.lib.format.open_memmap(
            prediction_temporary,
            mode="w+",
            dtype=np.float32,
            shape=(PRIMARY_SCORED_ROW_COUNT, 2),
        )
    output_offset = 0
    per_sequence_counts: list[dict[str, int]] = []
    try:
        for batch_start in range(0, len(dataset), batch_size):
            indices = list(range(batch_start, min(batch_start + batch_size, len(dataset))))
            batch = collate_sequences(cast(SequenceDataset, dataset), indices)
            targets = batch["y"].numpy()
            masks = batch["mask"].numpy()
            prediction = predict_sequence_batch(
                model, batch["x"], chunk_rows=chunk_rows, device=device
            )[..., :2]
            metrics.add(targets, prediction, masks)
            for offset, index in enumerate(indices):
                row_group = dataset.members[index][1]
                mask = np.asarray(masks[offset], dtype=bool)
                if np.any(mask[:WARMUP]):
                    raise ValueError("primary evaluation mask selected a warmup row")
                packed = np.packbits(mask, bitorder="little")
                mask_digest.update(packed.tobytes())
                row_indices = np.flatnonzero(mask).astype("<u4", copy=False)
                selected_index_digest.update(np.asarray([row_group], dtype="<i8").tobytes())
                selected_index_digest.update(row_indices.tobytes())
                count = int(len(row_indices))
                per_sequence_counts.append({"row_group": row_group, "selected_rows": count})
                if prediction_map is not None:
                    stop = output_offset + count
                    prediction_map[output_offset:stop] = prediction[offset, mask]
                output_offset += count
    finally:
        if prediction_map is not None:
            prediction_map.flush()
            del prediction_map

    result = metrics.result()
    if (
        result["selected_rows"] != PRIMARY_SCORED_ROW_COUNT
        or output_offset != PRIMARY_SCORED_ROW_COUNT
    ):
        raise ValueError("primary scoring did not cover all 3,528,273 official scored rows")
    prediction_receipt = {
        "sequence_order": "official primary evaluation membership order; source then row_group",
        "row_order": "ascending row_index among is_scored rows after warmup row 99",
        "mask": "is_scored AND row_index>=99",
        "mask_packed_bitorder": "little",
        "membership_sha256": membership_sha256(dataset.members),
        "mask_sha256": mask_digest.hexdigest(),
        "selected_row_index_sha256": selected_index_digest.hexdigest(),
        "selected_rows": output_offset,
        "per_sequence_selected_rows": per_sequence_counts,
    }
    return result, prediction_receipt


def create_primary_report(
    checkpoint: Path,
    data_root: Path,
    primary_manifest: Path,
    auxiliary_manifest: Path,
    output: Path,
    *,
    contract_path: Path = CONTRACT_PATH,
    predictions_output: Path | None = None,
    batch_size: int = 1,
    device_name: str = "cpu",
    cpu_threads: int = 1,
) -> dict[str, Any]:
    """Score a frozen auxiliary endpoint on primary valid without exposing an HPO objective."""
    if batch_size < 1 or cpu_threads < 1:
        raise ValueError("batch size and CPU thread count must be positive")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace primary report: {output}")
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("report checkpoint must be a regular fixed-fit model file")
    if primary_manifest.is_symlink() or auxiliary_manifest.is_symlink():
        raise ValueError("split manifests must be regular, non-symlink files")
    if predictions_output is not None and (
        predictions_output.exists()
        or predictions_output.is_symlink()
        or predictions_output.resolve() == output.resolve()
    ):
        raise FileExistsError(
            f"refusing to replace or alias prediction output: {predictions_output}"
        )
    contract, contract_sha256 = _load_contract(contract_path)
    primary = load_campaign_split(primary_manifest, data_root)
    auxiliary = load_campaign_split(auxiliary_manifest, data_root)
    primary_manifest_data = json.loads(primary_manifest.read_text(encoding="utf-8"))
    primary_sources = primary_manifest_data.get("sources")
    if not isinstance(primary_sources, dict):
        raise ValueError("primary split is missing its exact source identity mapping")

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA report device requested but CUDA is unavailable")
    configure_precision_mode("fp32")
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    model, payload = load_endpoint(checkpoint, device)
    producer, checkpoint_sha256 = _validate_endpoint(
        checkpoint, payload, primary, auxiliary, primary_sources
    )
    data = payload["data"]
    fit_members = [("train", group) for group in auxiliary.train_groups]
    control_members = [("train", group) for group in auxiliary.evaluation_groups]
    eval_members = [("valid", group) for group in primary.evaluation_groups]
    input_mode = payload["config"]["model"].get("input_feature_mode", "raw112")
    dataset = SequenceRecords(
        data_root,
        eval_members,
        0,
        training_mask="is_scored",
        input_feature_mode=input_mode,
    )
    dataset.validate_source_schema()

    output.parent.mkdir(parents=True, exist_ok=True)
    prediction_temporary = None
    if predictions_output is not None:
        predictions_output.parent.mkdir(parents=True, exist_ok=True)
        prediction_temporary = predictions_output.with_name(predictions_output.name + ".partial")
        if prediction_temporary.exists() or prediction_temporary.is_symlink():
            raise FileExistsError(f"prediction staging path already exists: {prediction_temporary}")
    try:
        metrics, prediction_receipt = _score_primary_valid(
            model,
            dataset,
            int(payload["config"]["chunk_rows"]),
            batch_size,
            device,
            prediction_temporary,
        )
    except BaseException:
        if prediction_temporary is not None:
            prediction_temporary.unlink(missing_ok=True)
        raise
    if metrics["selected_rows"] != PRIMARY_SCORED_ROW_COUNT:
        raise ValueError("primary selected-row count differs from the frozen official mask")
    prediction_record = None
    if predictions_output is not None and prediction_temporary is not None:
        prediction_temporary.replace(predictions_output)
        prediction_record = {
            "path": str(predictions_output.resolve()),
            "sha256": file_sha256(predictions_output),
            "dtype": "float32",
            "shape": [PRIMARY_SCORED_ROW_COUNT, 2],
            **prediction_receipt,
        }

    registry_path, registry_sha256 = _portable_registry_sha256()
    identity = producer["provenance"]["identity"]
    report = {
        "schema": REPORT_SCHEMA,
        "report_type": "primary_development_exposed_report_only",
        "objective_eligible": False,
        "checkpoint_selection_allowed": False,
        "interpretation": (
            "Frozen auxiliary-trained endpoint scored once on full official valid; "
            "validation is development-exposed and is not an Optuna objective."
        ),
        "model": {
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_schema": payload["checkpoint_schema"],
            "completed_epochs": payload["completed_epochs"],
            "config": payload["config"],
            "config_sha256": identity["config_sha256"],
            "run_identity_sha256": hashlib.sha256(
                json.dumps(
                    identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "data_identity_sha256": identity["data_sha256"],
            "source_files_sha256": identity["source_sha256"],
            "producer": producer,
            "provenance": payload["provenance"],
        },
        "fit": {
            "split_manifest": str(auxiliary_manifest.resolve()),
            "split_sha256": auxiliary.sha256,
            "primary_split_sha256": primary.sha256,
            "sequence_count": len(fit_members),
            "membership_sha256": membership_sha256(fit_members),
            "members": [list(member) for member in fit_members],
            "control_sequence_count": len(control_members),
            "control_membership_sha256": membership_sha256(control_members),
            "control_members": [list(member) for member in control_members],
            "teacher": data.get("teacher"),
            "initializer": data.get("initializer"),
            "normalization": data["normalization"],
        },
        "evaluation": {
            "split_manifest": str(primary_manifest.resolve()),
            "split_sha256": primary.sha256,
            "source": "valid",
            "role": "primary_valid",
            "mask": "is_scored AND row_index>=99",
            "sequence_count": len(eval_members),
            "membership_sha256": membership_sha256(eval_members),
            "members": [list(member) for member in eval_members],
            "selected_rows": PRIMARY_SCORED_ROW_COUNT,
            "metrics": metrics,
            "prediction_artifact": prediction_record,
        },
        "comparison_reference": {
            "name": "canonical fresh TL3",
            "wp": float(contract["baseline"]["canonical_fresh_TL3_wp"]),
            "trained_sequence_count": PRIMARY_TRAIN_COUNT,
            "source": contract["baseline"]["original_results"],
            "exposure": contract["baseline"]["exposure"],
            "measured_in_this_report": False,
            "contract_path": str(contract_path.resolve()),
            "contract_sha256": contract_sha256,
        },
        "portable_data_registry": {"path": registry_path, "sha256": registry_sha256},
        "runtime": {"device": str(device), "batch_size": batch_size, "cpu_threads": cpu_threads},
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main() -> None:
    """Run a report-only evaluation of a frozen endpoint on full primary valid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--primary-split-manifest", type=Path, required=True)
    parser.add_argument("--auxiliary-split-manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    try:
        create_primary_report(
            args.checkpoint,
            args.data_root,
            args.primary_split_manifest,
            args.auxiliary_split_manifest,
            args.output,
            contract_path=args.contract,
            predictions_output=args.predictions_output,
            batch_size=args.batch_size,
            device_name=args.device,
            cpu_threads=args.cpu_threads,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(f"primary report refused: {error}")


if __name__ == "__main__":
    main()
