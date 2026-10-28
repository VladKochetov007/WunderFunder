from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from wnn.gmetric import GlobalAccumulator
from wnn.splits import file_sha256
from wnn.training.sequence_fit_utils import collate_sequences
from wnn.training.sequence_models import detach_state
from wnn.training.sequence_records import SequenceRecords


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: SequenceRecords,
    chunk_rows: int,
    batch_size: int,
    device: torch.device,
    output: Path,
    check_deadline: Callable[[], None],
) -> dict[str, object]:
    """Export every official-valid prediction and pool the unchanged clip2 metric."""
    if min(chunk_rows, batch_size, len(dataset)) < 1:
        raise ValueError("evaluation requires positive chunk, batch and cohort sizes")
    model.eval()
    output.mkdir()
    accumulator = GlobalAccumulator()
    entries: list[dict[str, object]] = []
    selected_rows = 0
    for start in range(0, len(dataset), batch_size):
        check_deadline()
        indices = list(range(start, min(start + batch_size, len(dataset))))
        batch = collate_sequences(dataset, indices)
        state = None
        parts = []
        for position in range(0, batch["x"].shape[1], chunk_rows):
            check_deadline()
            prediction, _, state = model(
                batch["x"][:, position : position + chunk_rows].to(device), state
            )
            state = detach_state(state)
            parts.append(prediction)
        predicted = torch.cat(parts, dim=1).float().cpu().numpy()
        if not np.isfinite(predicted).all():
            raise ValueError("nonfinite prediction in BN diagnosis")
        mask = batch["mask"].numpy()
        accumulator.add(batch["y"].numpy(), predicted, mask)
        selected_rows += int(mask.sum())
        for offset, index in enumerate(indices):
            source, group = dataset.members[index]
            target = output / f"{source}-{group}.npy"
            with target.open("xb") as stream:
                np.save(stream, predicted[offset], allow_pickle=False)
            entries.append(
                {"member": [source, group], "file": target.name, "sha256": file_sha256(target)}
            )
    if selected_rows <= 0 or len(entries) != len(dataset):
        raise ValueError("empty or incomplete full-valid evaluation")
    return {
        "metric": "global_wp_clip2_v1",
        "scores": accumulator.result(),
        "selected_rows": selected_rows,
        "sequence_count": len(dataset),
        "mask": "is_scored AND row_index>=99",
        "predictions": entries,
    }
