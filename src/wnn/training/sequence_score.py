from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wnn.data import SEQUENCE_LENGTH, WARMUP, SequenceFile
from wnn.evaluation import paired_component_moments
from wnn.gmetric import GlobalAccumulator, _merge_moments
from wnn.splits import (
    CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
    MULTISTAGE_AUXILIARY_SPLIT_SCHEMA,
    SEQUENCE_INNER_SPLIT_SCHEMA,
    CampaignSplit,
    file_sha256,
    load_campaign_split,
)
from wnn.training.sequence_cache import validate_export_schema, validate_producer
from wnn.training.sequence_epoch_prediction import EPOCH_KIND, validate_epoch_producer
from wnn.training.sequence_monitor_identity import membership_sha256
from wnn.training.sequence_provenance import canonical_sha256


def score_export(
    root: Path,
    split_manifest: Path,
    predictions: Path,
    output: Path,
    *,
    report_only_primary_valid: bool = False,
) -> None:
    """Score every held-out row once with the canonical pooled official metric."""
    split = load_campaign_split(split_manifest, root)
    manifest_path = predictions / "manifest.json"
    if manifest_path.stat().st_size > 32_000_000:
        raise ValueError("prediction manifest exceeds bounded metadata size")
    meta = json.loads(manifest_path.read_text())
    if meta.get("schema") != "sequence-predictions-v1" or meta.get("complete") is not True:
        raise ValueError("score requires a completed held-out prediction export")
    if meta["split_sha256"] != split.sha256:
        raise ValueError("prediction split hash mismatch")
    validate_export_schema(meta, "evaluation")
    fitting = [("train", g) for g in split.train_groups]
    if split.schema != CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA:
        fitting += [("valid", g) for g in split.valid_groups]
    epoch_evaluation = meta["producer"].get("kind") == EPOCH_KIND
    if epoch_evaluation:
        if split.schema == CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA:
            if (
                not report_only_primary_valid
                or meta["producer"].get("primary_valid_report_only") is not True
            ):
                raise ValueError(
                    "chronological retained epochs require explicit report-only primary-valid mode"
                )
            validate_epoch_producer(
                meta["producer"], split.sha256, fitting, primary_report_only=True
            )
        elif split.schema in (SEQUENCE_INNER_SPLIT_SCHEMA, MULTISTAGE_AUXILIARY_SPLIT_SCHEMA):
            if report_only_primary_valid:
                raise ValueError("report-only primary mode requires the chronological split")
            validate_epoch_producer(meta["producer"], split.sha256, fitting)
        else:
            raise ValueError("retained epoch reports require inner or chronological membership")
    else:
        if report_only_primary_valid:
            raise ValueError("report-only primary mode requires a retained epoch export")
        expected_sources = None
        if split.path.is_file():
            split_record = json.loads(split.path.read_text(encoding="utf-8"))
            raw_sources = split_record.get("sources") if isinstance(split_record, dict) else None
            expected_sources = raw_sources if isinstance(raw_sources, dict) else None
        validate_producer(
            meta["producer"],
            split.sha256,
            fitting,
            evaluation_members=[
                (split.evaluation_source, group) for group in split.evaluation_groups
            ],
            primary_split_sha256=split.primary_split_sha256,
            expected_sources=expected_sources,
        )
    members = [(entry["split"], entry["row_group"]) for entry in meta["sequences"]]
    expected = [(split.evaluation_source, group) for group in split.evaluation_groups]
    if len(set(members)) != len(members) or set(members) != set(expected):
        raise ValueError("held-out prediction coverage is incomplete or duplicated")
    if split.schema == MULTISTAGE_AUXILIARY_SPLIT_SCHEMA and members != expected:
        raise ValueError("auxiliary prediction membership is not in its canonical source order")
    source = SequenceFile(root / f"datasets/{split.evaluation_source}.parquet")
    accumulator = GlobalAccumulator()
    centered_by_component: dict[int, np.ndarray] = {}
    component_map = {(s, g): c for s, g, c in split.components}
    for entry in meta["sequences"]:
        path = predictions / entry["file"]
        if path.parent.resolve() != predictions.resolve() or path.is_symlink():
            raise ValueError("prediction file escapes export root")
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("prediction hash mismatch")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.dtype != np.float32 or values.shape != (SEQUENCE_LENGTH, 2 + meta["hint_dim"]):
            raise ValueError("prediction shape/dtype mismatch")
        if not np.isfinite(values).all():
            raise ValueError("nonfinite prediction export")
        seq = source.read(entry["row_group"])
        if seq.targets is None:
            raise ValueError("evaluation labels unavailable")
        if split.evaluation_mask == "all_after_warmup":
            mask = np.ones(SEQUENCE_LENGTH, dtype=bool)
        elif split.evaluation_mask == "is_scored" and seq.scored is not None:
            mask = seq.scored.copy()
        else:
            raise ValueError("evaluation scoring mask is unavailable")
        mask[:WARMUP] = False
        accumulator.add(
            seq.targets,
            values[:, :2],
            mask,
            group=component_map[(split.evaluation_source, entry["row_group"])],
        )
        if mask.any():
            moments, _ = paired_component_moments(
                seq.targets[None], values[None, :, :2], values[None, :, :2], mask[None]
            )
            component = component_map[(split.evaluation_source, entry["row_group"])]
            previous = centered_by_component.get(component, np.zeros((2, 6), dtype=np.float64))
            centered_by_component[component] = np.stack(
                [_merge_moments(previous[k], moments[0, k]) for k in range(2)]
            )
    receipt = {
        "score": accumulator.result(),
        "split_sha256": split.sha256,
        "predictions_manifest_sha256": file_sha256(manifest_path),
        "producer": meta["producer"],
        "sequences": len(members),
        "component_raw_sums": {str(k): v.tolist() for k, v in accumulator.groups.items()},
        "component_centered_moments": {
            str(k): v.tolist() for k, v in centered_by_component.items()
        },
        "interpretation": (
            "Report-only full official-valid metric for this retained epoch; no checkpoint selection; validation is exposed."
            if report_only_primary_valid
            else "Inner development metric for epoch selection; not outer confirmation or public score."
            if epoch_evaluation
            else "Exposed fold; fixed endpoint, not checkpoint selection or public score."
        ),
        "report_only_primary_valid": report_only_primary_valid,
    }
    if split.schema == MULTISTAGE_AUXILIARY_SPLIT_SCHEMA:
        receipt = _inner_objective_receipt(
            split,
            meta,
            manifest_path,
            members,
            accumulator.result(),
            len(members) * (SEQUENCE_LENGTH - WARMUP),
        )
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _inner_objective_receipt(
    split: CampaignSplit,
    predictions: dict[str, object],
    manifest_path: Path,
    members: list[tuple[str, int]],
    score: dict[str, float | int | str],
    selected_rows: int,
) -> dict[str, object]:
    """Bind an auxiliary WP objective to exact control, fit, model, and config identities."""
    producer = predictions.get("producer")
    if not isinstance(producer, dict):
        raise ValueError("inner objective producer identity is malformed")
    fitting, config = producer.get("fit_members"), producer.get("config")
    producer_provenance = producer.get("provenance")
    model_sha256 = producer.get("checkpoint_sha256")
    split_sha256 = predictions.get("split_sha256")
    primary_sha256 = split.primary_split_sha256
    if (
        not isinstance(fitting, list)
        or not isinstance(config, dict)
        or not isinstance(producer_provenance, dict)
        or not isinstance(model_sha256, str)
        or not isinstance(split_sha256, str)
        or not isinstance(primary_sha256, str)
    ):
        raise ValueError("inner objective is missing parent, fit, config, or checkpoint identity")
    identity = producer_provenance.get("identity")
    identity_data = identity.get("data") if isinstance(identity, dict) else None
    expected_fit = [["train", group] for group in split.train_groups]
    expected_evaluation = [list(member) for member in members]
    if (
        not isinstance(identity, dict)
        or not isinstance(identity_data, dict)
        or producer.get("split_sha256") != split.sha256
        or fitting != expected_fit
        or identity.get("data_sha256") != canonical_sha256(identity_data)
        or identity.get("config_sha256") != canonical_sha256(config)
        or identity_data.get("split_sha256") != split.sha256
        or identity_data.get("primary_split_sha256") != primary_sha256
        or identity_data.get("fit_members") != fitting
        or identity_data.get("evaluation_members") != expected_evaluation
    ):
        raise ValueError("inner objective producer run identity does not bind this auxiliary split")
    metrics = {key: float(score[key]) for key in ("wp", "t0", "t1")}
    fit_members = [(str(member[0]), int(member[1])) for member in fitting]
    return {
        "schema": "wnn.multistage-inner-objective.v1",
        "status": "complete",
        "split_sha256": split_sha256,
        "primary_split_sha256": primary_sha256,
        "prediction_manifest_sha256": file_sha256(manifest_path),
        "model_sha256": model_sha256,
        "config_sha256": canonical_sha256(config),
        "fit_members_sha256": membership_sha256(fit_members),
        "evaluation_members_sha256": membership_sha256(members),
        "sequence_count": len(members),
        "selected_rows": selected_rows,
        "metric_id": score["metric_id"],
        "objective": {
            "name": "wp",
            "value": metrics["wp"],
            "t0": metrics["t0"],
            "t1": metrics["t1"],
            "reduction": "mean_of_two_pooled_clipped_weighted_Pearson_targets",
        },
        "evaluation": {
            "role": "control",
            "scope": "inner_train_proxy",
            "source": "train",
            "row_mask": "all_after_warmup AND row_index>=99",
            "is_scored_used": False,
            "members": [list(member) for member in members],
        },
        "interpretation": "selection proxy only; not chronological validation or public score",
        "producer": {
            "checkpoint_sha256": model_sha256,
            "split_sha256": split_sha256,
            "fit_members_sha256": membership_sha256(fit_members),
            "provenance": {
                "identity": identity,
                "identity_sha256": canonical_sha256(identity),
            },
        },
        "prediction_producer": producer,
    }


def main() -> None:
    """Grade an immutable endpoint export without importing or executing a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-only-primary-valid", action="store_true")
    args = parser.parse_args()
    score_export(
        args.data_root,
        args.split_manifest,
        args.predictions,
        args.output,
        report_only_primary_valid=args.report_only_primary_valid,
    )
