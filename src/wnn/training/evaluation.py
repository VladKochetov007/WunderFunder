from __future__ import annotations

# Historical checkpoint records and experimental backend/batch states have
# heterogeneous layouts; Any is retained only at those compatibility boundaries.
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from wnn.data import market_class
from wnn.metric import block_aggregate_score, block_scores, score
from wnn.outcomes import readout
from wnn.training.environment import TrainingEnvironment
from wnn.training.inputs import build_inputs


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    x: torch.Tensor,
    chunk: int,
    device: torch.device,
    *,
    environment: TrainingEnvironment | None = None,
) -> np.ndarray:
    """Predict."""
    environment = environment if environment is not None else TrainingEnvironment()
    model.eval()
    inputs = build_inputs(x.to(device, non_blocking=True), environment=environment)
    state, outs = (None, [])
    for s in range(0, inputs.shape[1], chunk):
        with torch.autocast(
            device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda" and (not getattr(model, "_inference_fp32", False)),
        ):
            out, state = environment.forward_chunk(model, inputs, slice(s, s + chunk), state)
        outs.append(
            readout(out, *environment.readout.tables, tilted=environment.readout.tilted)
            if environment.readout.tables is not None
            else out[..., :2].float()
        )
    return torch.cat(outs, dim=1).cpu().numpy()


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    chunk: int,
    device: torch.device,
    *,
    environment: TrainingEnvironment | None = None,
) -> dict[str, Any]:
    """Evaluate."""
    environment = environment if environment is not None else TrainingEnvironment()
    preds, ys, masks, classes = ([], [], [], [])
    for x, y, m in loader:
        preds.append(
            predict(model, x, chunk, device, environment=environment)
            if environment.prediction is None
            else environment.prediction(model, x, chunk, device)
        )
        ys.append(y.numpy())
        selected = m.numpy().copy()
        selected[:, : environment.warmup] = False
        masks.append(selected)
        classes += [market_class(t) for t in y.numpy()]
    p, y, m = (np.concatenate(preds), np.concatenate(ys), np.concatenate(masks))
    res = score(y, p, m)
    legacy = (
        block_aggregate_score(y, p, m)
        if res["eligible_blocks"]
        else {"weighted_pearson": None, "t0": None, "t1": None}
    )
    res.update(
        global_t0=res["t0"],
        global_t1=res["t1"],
        per_seq_wp=legacy["weighted_pearson"],
        block_t0=legacy["t0"],
        block_t1=legacy["t1"],
    )
    res["frac_abs_pred_gt2"] = float((np.abs(p[m]) > 2).mean())
    per = block_scores(y, p, m).mean(axis=1)
    cls = np.array(classes)
    res["by_class"] = {
        int(c): score(y[cls == c], p[cls == c], m[cls == c])["weighted_pearson"]
        for c in np.unique(cls)
    }
    res["block_by_class"] = {
        int(c): round(float(np.nanmean(per[cls == c])), 4)
        if np.isfinite(per[cls == c]).any()
        else None
        for c in np.unique(cls)
    }
    return res
