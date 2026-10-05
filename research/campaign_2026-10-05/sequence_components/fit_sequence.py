"""Fixed exposed-cohort sequence screen; no checkpoint or hyperparameter selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from components import AdditiveSpline, DiagonalSSM, KernelAttention, PointwiseMLPControl
from torch import nn

from wnn.gmetric import _centered_moments, _wp_from_moments
from wnn.resources import configure_torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
FEATURES = [
    "i0_p0",
    "i0_p11",
    "i0_v0",
    "i0_v11",
    "i0_dp0",
    "i0_dv0",
    "i1_p0",
    "i1_p11",
    "i1_v0",
    "i1_v11",
    "i1_dp0",
    "i1_dv0",
    "a0",
    "a1",
    "a2",
    "a3",
    "a4",
    "a5",
    "a6",
    "a7",
]
ARMS = (
    "gru_mlp",
    "gru_spline",
    "ssm_fixed_mlp",
    "ssm_learned_mlp",
    "ssm_learned_spline",
    "gru_attention_mlp",
    "gru_pointwise_mlp",
    "ssm_attention_mlp",
    "ssm_pointwise_mlp",
    "ssm_attention_spline",
)


def weighted_pearson(target: np.ndarray, prediction: np.ndarray) -> float:
    """Use the existing official scorer's centered clipped moments unchanged."""
    y = np.clip(np.asarray(target, dtype=np.float32), -2, 2).astype(np.float64)
    p = np.clip(np.asarray(prediction, dtype=np.float32), -2, 2).astype(np.float64)
    if y.shape != p.shape or not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("unaligned or nonfinite selected metric inputs")
    return _wp_from_moments(_centered_moments(y, p))


def causal_features(raw: np.ndarray) -> np.ndarray:
    """Embed observed levels, previous-row changes, and observable event age."""
    difference = np.diff(raw, axis=0, prepend=raw[:1])
    event = np.any(difference[:, :2] != 0, axis=1)
    index = np.arange(len(raw))
    last = np.maximum.accumulate(np.where(event, index, 0))
    return np.nan_to_num(
        np.column_stack([raw, difference, np.log1p(index - last), event]),
        nan=0,
        posinf=0,
        neginf=0,
    ).astype(np.float32)


def detach_state(state: object) -> object:
    """Truncate the gradient, keeping streaming inference state across chunks."""
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, tuple):
        return tuple(detach_state(value) for value in state)
    if state is None:
        return None
    raise TypeError(f"unknown recurrent state type: {type(state)}")


class SequenceModel(nn.Module):
    """Compose a causal embedding, memory, and independently controlled head."""

    def __init__(self, arm: str, input_size: int = 42, width: int = 32) -> None:
        super().__init__()
        self.arm = arm
        self.embedding = nn.Sequential(nn.Linear(input_size, width), nn.Tanh())
        self.memory = (
            nn.GRU(width, width, batch_first=True)
            if arm.startswith("gru")
            else DiagonalSSM(width, width, learn_decay="fixed" not in arm)
        )
        self.attention = KernelAttention(width, 8, 8) if "attention" in arm else None
        self.pointwise = (
            PointwiseMLPControl(width, 8, parameter_budget=3 * (width + 1) * 8 + 1)
            if "pointwise" in arm
            else None
        )
        readout_size = width + (
            8 if self.attention is not None or self.pointwise is not None else 0
        )
        # The spline is a residual over the same linear anchor. For each memory
        # family the MLP and spline heads have closely matched parameter counts.
        self.anchor = nn.Linear(readout_size, 2)
        if arm.endswith("spline"):
            self.correction = AdditiveSpline(readout_size, 2, knots=7, knot_min=-1, knot_max=1)
        else:
            hidden = max(1, round(readout_size * 14 / (readout_size + 3)))
            self.correction = nn.Sequential(
                nn.Linear(readout_size, hidden), nn.Tanh(), nn.Linear(hidden, 2)
            )
            nn.init.zeros_(self.correction[-1].weight)
            nn.init.zeros_(self.correction[-1].bias)

    def forward(self, x: torch.Tensor, state: object = None) -> tuple[torch.Tensor, object]:
        """Advance fixed-size state using current and earlier features only."""
        memory_state, attention_state = (None, None) if state is None else state
        encoded = self.embedding(x)
        memory, memory_state = self.memory(encoded, memory_state)
        if self.attention is not None:
            retrieved, attention_state = self.attention(encoded, attention_state)
            memory = torch.cat([memory, retrieved], dim=-1)
        elif self.pointwise is not None:
            memory = torch.cat([memory, torch.tanh(self.pointwise(encoded))], dim=-1)
        prediction = self.anchor(memory) + self.correction(memory)
        return prediction, (memory_state, attention_state)


def load_cohort() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Reuse the frozen diagnostic cohort and fit normalization on fit rows only."""
    manifest = json.loads((HERE.parent / "target_mechanics/cohort.json").read_text())
    nodes = manifest["cohort"]
    if len({n["row_group"] for n in nodes}) != len(nodes):
        raise ValueError("duplicate row groups in the frozen cohort")
    if len({n["component"] for n in nodes}) != len(nodes):
        raise ValueError("duplicate components in the frozen cohort")
    fit_components = {n["component"] for n in nodes if n["role"] == "fit"}
    test_components = {n["component"] for n in nodes if n["role"] == "test"}
    if fit_components & test_components:
        raise ValueError("component overlap in this experiment's fit/test split")
    pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    xs, ys, need_masks, score_masks = [], [], [], []
    for node in nodes:
        table = pf.read_row_group(
            node["row_group"],
            columns=[*FEATURES, "t0", "t1", "need_prediction", "is_scored"],
            use_threads=False,
        )
        raw = np.column_stack([table[c].to_numpy() for c in FEATURES])
        xs.append(causal_features(raw))
        ys.append(np.column_stack([table[c].to_numpy() for c in ("t0", "t1")]))
        need = table["need_prediction"].to_numpy().astype(bool)
        need_masks.append(need)
        score_masks.append(need & table["is_scored"].to_numpy().astype(bool))
    x, y = np.stack(xs), np.clip(np.stack(ys), -2, 2).astype(np.float32)
    if x.shape[1] != 20_000 or not np.isfinite(y).all():
        raise ValueError("expected complete sequences with finite labels")
    fit = np.array([n["role"] == "fit" for n in nodes])
    mean = x[fit].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = np.maximum(x[fit].std(axis=(0, 1), dtype=np.float64), 0.001).astype(np.float32)
    x = np.clip((x - mean) / scale, -8, 8)
    manifest["normalization"] = {"fit_mean": mean.tolist(), "fit_scale": scale.tolist()}
    return manifest, x, y, np.stack(need_masks), np.stack(score_masks), nodes


def smoke(device: torch.device) -> dict:
    """Probe the actual batch/chunk shape and an optimizer step for every arm."""
    receipt = {}
    for arm in ARMS:
        torch.manual_seed(271828)
        x = torch.randn(4, 256, 42, device=device)
        target = torch.randn(4, 256, 2, device=device).clamp(-2, 2)
        model = SequenceModel(arm).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        prediction, state = model(x)
        loss = (target.abs() * (prediction - target).square()).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        if not torch.isfinite(prediction).all():
            raise ValueError(f"nonfinite predictions: {arm}")
        with torch.no_grad():
            model(x, detach_state(state))
        receipt[arm] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "loss": loss.item(),
        }
    return receipt


def run_arm(
    arm: str,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    score: np.ndarray,
    nodes: list[dict],
    device: torch.device,
    output: Path,
) -> dict:
    """Train exactly one full epoch, then score a single fixed checkpoint."""
    torch.manual_seed(271828)
    model = SequenceModel(arm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    fit_ids = [i for i, n in enumerate(nodes) if n["role"] == "fit"]
    rng = np.random.default_rng(271828)
    rng.shuffle(fit_ids)
    steps, start = 0, time.monotonic()
    for offset in range(0, len(fit_ids), 4):
        ids = fit_ids[offset : offset + 4]
        state = None
        for position in range(0, x.shape[1], 256):
            xx = torch.from_numpy(x[ids, position : position + 256]).to(device)
            yy = torch.from_numpy(y[ids, position : position + 256]).to(device)
            mm = torch.from_numpy(need[ids, position : position + 256]).to(device)[..., None]
            optimizer.zero_grad(set_to_none=True)
            pred, state = model(xx, state)
            weights = yy.abs() * mm
            mass = weights.sum(dim=(0, 1))
            loss = ((weights * (pred - yy).square()).sum(dim=(0, 1)) / mass.clamp_min(1e-12)).mean()
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite loss: {arm}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            state = detach_state(state)
            steps += 1
    fit_seconds = time.monotonic() - start
    torch.save(model.state_dict(), output / f"{arm}.pt")
    model.eval()
    all_y, all_p, by_sequence = [], [], []
    with torch.no_grad():
        for i, node in enumerate(nodes):
            if node["role"] != "test":
                continue
            state, predictions = None, []
            for position in range(0, x.shape[1], 256):
                xx = torch.from_numpy(x[i : i + 1, position : position + 256]).to(device)
                pred, state = model(xx, state)
                predictions.append(pred[0].cpu().numpy())
            prediction = np.concatenate(predictions)
            if not np.isfinite(prediction).all():
                raise ValueError(f"nonfinite evaluation prediction: {arm}")
            yy, pp = y[i][score[i]], prediction[score[i]]
            all_y.append(yy)
            all_p.append(pp)
            by_sequence.append(
                {
                    "row_group": node["row_group"],
                    "scored_rows": len(yy),
                    "wp": [weighted_pearson(yy[:, j], pp[:, j]) for j in range(2)],
                }
            )
    target, prediction = np.concatenate(all_y), np.concatenate(all_p)
    wp = [weighted_pearson(target[:, j], prediction[:, j]) for j in range(2)]
    np.savez_compressed(
        output / f"{arm}_selected_predictions.npz", targets=target, predictions=prediction
    )
    return {
        "arm": arm,
        "parameters": sum(p.numel() for p in model.parameters()),
        "steps": steps,
        "fit_seconds": fit_seconds,
        "wp": float(np.mean(wp)),
        "target_wp": wp,
        "scored_rows": len(target),
        "sequences": by_sequence,
    }


def main() -> None:
    """Execute only the chosen probe or fixed all-arm experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda":
        configure_torch()
    if args.smoke:
        result = smoke(device)
    else:
        manifest, x, y, need, score, nodes = load_cohort()
        (args.output / "cohort.json").write_text(json.dumps(manifest, indent=2))
        result = {
            "metric": "global_wp_clip2_v1",
            "historical_exposure": manifest["historical_exposure"],
            "scope": "small full-sequence one-epoch models; mechanism screen, no public forecast",
            "cohort_sha256": hashlib.sha256(
                (HERE.parent / "target_mechanics/cohort.json").read_bytes()
            ).hexdigest(),
            "arms": [],
        }
        for arm in ARMS:
            item = run_arm(arm, x, y, need, score, nodes, device, args.output)
            result["arms"].append(item)
            print(json.dumps(item), flush=True)
            (args.output / "result.json").write_text(json.dumps(result, indent=2))
    result["source_sha256"] = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in ("fit_sequence.py", "components.py")
    }
    if device.type == "cuda":
        result["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    (args.output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
