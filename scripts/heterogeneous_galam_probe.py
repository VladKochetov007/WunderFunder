"""Reuse the original scored-only GALAM probe as an explicitly evaluation-only source."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from attrs import define, field

from scripts.heterogeneous_ensemble_contract import (
    JsonObject,
    PredictionSource,
    pinned_file,
    read_object,
)
from wnn.splits import CampaignSplit
from wnn.training.sequence_epoch_prediction import digest, epoch_admission

PROBE_SCRIPT_SHA256 = "514d3aae37d828f115396cedea37f4afd8506519da998225e7d74638b9c79e1d"
PROBE_CELL = "g2-galam-lstm128"
NATIVE_EPOCH = 2


@define
class ScoredProbeSource(PredictionSource):
    """Expose original raw scored predictions while marking all other rows unobserved.

    Zero placeholders exist only in this scorer's temporary in-memory representation.
    They are never exported as complete model predictions or used as fitting targets.
    """

    selected_positions: dict[int, np.ndarray] = field(factory=dict)
    offsets: dict[int, tuple[int, int]] = field(factory=dict)

    def sequence(self, group: int) -> np.ndarray:
        """Place the observed raw outputs at their verified official scoring positions."""
        if self.bank is None:
            raise ValueError("scored probe has no observed predictions")
        start, stop = self.offsets[group]
        observed = np.asarray(self.bank[start:stop])
        if observed.dtype != np.float32 or not np.isfinite(observed).all():
            raise ValueError("scored probe contains nonfinite or non-FP32 observations")
        result = np.zeros((20_000, 2), dtype=np.float32)
        result[self.selected_positions[group]] = observed
        return result


def scored_row_layout(
    validation: Path, groups: list[int]
) -> tuple[dict[int, np.ndarray], dict[int, tuple[int, int]], str, int]:
    """Reproduce the original probe's exact source-qualified mask-order hash."""
    parquet = pq.ParquetFile(validation)
    selected_positions, offsets = {}, {}
    membership_hash = hashlib.sha256()
    count = 0
    for group in groups:
        table = parquet.read_row_group(group, columns=["is_scored"], use_threads=False)
        if table.num_rows != 20_000:
            raise ValueError("scored probe requires complete20000-row source sequences")
        mask = table["is_scored"].to_numpy().astype(bool, copy=True)
        mask[:99] = False
        selected = np.flatnonzero(mask).astype("<i4")
        membership_hash.update(np.asarray(("valid", group), dtype="U16").tobytes())
        membership_hash.update(selected.tobytes())
        selected_positions[group] = selected
        offsets[group] = count, count + len(selected)
        count += len(selected)
    return selected_positions, offsets, membership_hash.hexdigest(), count


def validate_probe_receipts(
    record: JsonObject,
    receipt: JsonObject,
    plan: JsonObject,
    result: JsonObject,
    split: CampaignSplit,
    admission: JsonObject,
) -> None:
    """Join the original frozen plan, terminal result, selected epoch and raw export."""
    endpoint = receipt.get("endpoint", {})
    if (
        record.get("run") != PROBE_CELL
        or record.get("epoch") != NATIVE_EPOCH
        or endpoint.get("cell") != PROBE_CELL
        or endpoint.get("epoch") != NATIVE_EPOCH + 1
        or endpoint.get("checkpoint_sha256") != record["checkpoint"]["sha256"]
        or endpoint.get("epoch_index_sha256") != record.get("epoch_index_sha256")
        or endpoint not in plan.get("endpoints", [])
    ):
        raise ValueError("scored probe checkpoint/role/native-versus-human epoch mismatch")
    if (
        plan.get("schema") != "tabdata.galam-fixed-blend-discovery.v1"
        or plan.get("split_sha256") != split.sha256
        or plan.get("members") != len(split.evaluation_groups)
        or result.get("plan_sha256") != record["plan"]["sha256"]
        or result.get("script_sha256") != PROBE_SCRIPT_SHA256
        or result.get("scientific_fits_started") != 0
        or result.get("training_processes_changed") is not False
    ):
        raise ValueError("scored probe plan/terminal/source identity mismatch")
    matches = [item for item in result.get("receipts", []) if item.get("endpoint") == endpoint]
    if len(matches) != 1 or any(matches[0].get(key) != value for key, value in receipt.items()):
        raise ValueError("scored probe receipt does not match the original terminal record")
    if receipt.get("prediction_sha256") != record["predictions"]["sha256"]:
        raise ValueError("scored probe predictions differ from their producer receipt")
    identity = admission["index"]["identity"]
    data = identity.get("data", {})
    if (
        admission["entry"]["epoch"] != NATIVE_EPOCH
        or identity.get("source_sha256") != plan.get("training_source_sha256")
        or digest(identity.get("source_files")) != identity.get("source_sha256")
        or digest(data) != identity.get("data_sha256")
        or data.get("split_sha256") != split.sha256
        or data.get("fit_members") != [["train", group] for group in split.train_groups]
        or data.get("evaluation_members") != [["valid", group] for group in split.evaluation_groups]
        or data.get("teacher") is not None
        or data.get("hint_mode") != "aligned"
        or data.get("hint_members") is not None
    ):
        raise ValueError("scored probe checkpoint has different source/fitting ancestry")


def load_probe(
    record: JsonObject, base: Path, split: CampaignSplit, validation: Path
) -> ScoredProbeSource:
    """Admit the original full-valid scored-only artifact without any new inference."""
    receipt_path = pinned_file(record.get("metadata"), base)
    plan_path = pinned_file(record.get("plan"), base)
    result_path = pinned_file(record.get("result"), base)
    checkpoint = pinned_file(record.get("checkpoint"), base)
    index = pinned_file(record.get("epoch_index"), base)
    predictions = pinned_file(record.get("predictions"), base)
    receipt, plan, result = (
        read_object(receipt_path),
        read_object(plan_path),
        read_object(result_path),
    )
    admission = epoch_admission(checkpoint, index, record.get("epoch_index_sha256"))
    validate_probe_receipts(record, receipt, plan, result, split, admission)
    positions, offsets, row_hash, count = scored_row_layout(
        validation, list(split.evaluation_groups)
    )
    if (
        receipt.get("row_order_sha256") != row_hash
        or result.get("row_order_sha256") != row_hash
        or receipt.get("scored_rows") != count
        or result.get("scored_rows") != count
        or plan.get("expected_scored_rows") != count
        or receipt.get("metrics", {}).get("selected_rows") != count
    ):
        raise ValueError("scored probe mask/order/count differs from actual official valid")
    bank = np.load(predictions, mmap_mode="r", allow_pickle=False)
    if bank.dtype != np.float32 or bank.shape != (count, 2):
        raise ValueError("scored probe must retain exactly its original FP32 scored-row shape")
    return ScoredProbeSource(
        record, receipt_path, checkpoint, {}, bank, selected_positions=positions, offsets=offsets
    )
