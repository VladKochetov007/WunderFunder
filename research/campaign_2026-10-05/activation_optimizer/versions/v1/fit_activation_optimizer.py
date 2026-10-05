"""Fixed 8x2 activation/optimizer screen on the E467 exposed cohort.

This is a draft trainer. It does not launch itself through the campaign runner;
use the parent's reviewed, content-bound resource spec before any full fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

from wnn.gmetric import _centered_moments, _wp_from_moments
from wnn.optim import MuonAdamW, muon_param_groups
from wnn.resources import configure_torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
COHORT_PATH = ROOT / "research/campaign_2026-10-05/target_mechanics/cohort.json"
DATA_PATH = ROOT / "datasets/valid.parquet"
SEED = 271828
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
    *[f"a{i}" for i in range(8)],
]
ACTIVATIONS = (
    "tanh",
    "gelu",
    "relu",
    "leaky_relu_001",
    "gelu_sine_01",
    "gelu_sine_cosine_01",
    "gelu_sinc_05",
    "split_gelu_leaky_relu",
)
OPTIMIZERS = ("adamw", "muon_adamw")
ARMS = tuple(f"{activation}__{optimizer}" for activation in ACTIVATIONS for optimizer in OPTIMIZERS)


class PointwiseActivation(nn.Module):
    """Parameter-free activation choices; every choice preserves channel width."""

    def __init__(self, name: str) -> None:
        super().__init__()
        if name not in ACTIVATIONS:
            raise ValueError(f"unknown activation: {name}")
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "tanh":
            return torch.tanh(x)
        if self.name == "gelu":
            return torch.nn.functional.gelu(x)
        if self.name == "relu":
            return torch.relu(x)
        if self.name == "leaky_relu_001":
            return torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        if self.name == "gelu_sine_01":
            return torch.nn.functional.gelu(x) + 0.1 * torch.sin(x)
        if self.name == "gelu_sine_cosine_01":
            return torch.nn.functional.gelu(x) + 0.1 * torch.sin(x) + 0.1 * (torch.cos(x) - 1.0)
        if self.name == "gelu_sinc_05":
            return torch.nn.functional.gelu(x) * (1.0 + 0.5 * torch.sinc(x))
        if self.name == "split_gelu_leaky_relu":
            split = x.shape[-1] // 2
            return torch.cat(
                (
                    torch.nn.functional.gelu(x[..., :split]),
                    torch.nn.functional.leaky_relu(x[..., split:], negative_slope=0.01),
                ),
                dim=-1,
            )
        raise AssertionError("activation dispatch is exhaustive")


class ActivationGRU(nn.Module):
    """E467 GRU+MLP layout with one shared pointwise activation choice."""

    def __init__(self, activation: str, input_size: int = 42, width: int = 32) -> None:
        super().__init__()
        self.activation_name = activation
        self.embedding = nn.Linear(input_size, width)
        self.embedding_activation = PointwiseActivation(activation)
        # Keep this attribute name for the existing, audited Muon gate selector.
        self.gru = nn.GRU(width, width, batch_first=True)
        self.anchor = nn.Linear(width, 2)
        hidden = max(1, round(width * 14 / (width + 3)))
        self.correction_hidden = nn.Linear(width, hidden)
        self.correction_activation = PointwiseActivation(activation)
        self.correction_output = nn.Linear(hidden, 2)
        nn.init.zeros_(self.correction_output.weight)
        nn.init.zeros_(self.correction_output.bias)

    def forward(
        self, x: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.embedding_activation(self.embedding(x))
        memory, final_state = self.gru(encoded, state)
        residual = self.correction_output(
            self.correction_activation(self.correction_hidden(memory))
        )
        return self.anchor(memory) + residual, final_state


def make_optimizer(model: ActivationGRU, name: str) -> torch.optim.Optimizer:
    """Use equal AdamW settings and the repository's GRU-only Muon partition."""
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999)
        )
    if name != "muon_adamw":
        raise ValueError(f"unknown optimizer: {name}")
    groups = muon_param_groups(model, scope="rnn")
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assigned = [id(p) for group in groups for p in group["params"]]
    if len(assigned) != len(set(assigned)) or set(assigned) != trainable:
        raise ValueError("Muon/AdamW groups must be a disjoint exact trainable-parameter partition")
    muon_groups = [group for group in groups if group.get("muon")]
    selected_names = {
        name
        for name, param in model.named_parameters()
        if any(id(param) == id(candidate) for group in muon_groups for candidate in group["params"])
    }
    expected = {"gru.weight_ih_l0", "gru.weight_hh_l0"}
    if selected_names != expected or len(muon_groups) != 1 or muon_groups[0].get("blocks") != 3:
        raise ValueError(f"Muon must cover only three-block GRU gate matrices: {selected_names}")
    return MuonAdamW(groups, lr=0.001, weight_decay=0.01, betas=(0.9, 0.999))


def weighted_pearson(target: np.ndarray, prediction: np.ndarray) -> float:
    y = np.clip(np.asarray(target, dtype=np.float32), -2, 2).astype(np.float64)
    p = np.clip(np.asarray(prediction, dtype=np.float32), -2, 2).astype(np.float64)
    if y.shape != p.shape or not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("unaligned or nonfinite selected metric inputs")
    return _wp_from_moments(_centered_moments(y, p))


def causal_features(raw: np.ndarray) -> np.ndarray:
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


def load_cohort(
    include_test: bool,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Load the frozen 24-fit/16-test cohort; preflight can exclude test rows entirely."""
    manifest = json.loads(COHORT_PATH.read_text())
    nodes = [node for node in manifest["cohort"] if include_test or node["role"] == "fit"]
    if not nodes or len({node["row_group"] for node in nodes}) != len(nodes):
        raise ValueError("empty cohort or duplicate row groups")
    fit_components = {node["component"] for node in nodes if node["role"] == "fit"}
    test_components = {node["component"] for node in nodes if node["role"] == "test"}
    if fit_components & test_components:
        raise ValueError("component overlap in fit/test split")
    parquet = pq.ParquetFile(DATA_PATH)
    xs, ys, need_masks, score_masks = [], [], [], []
    for node in nodes:
        table = parquet.read_row_group(
            node["row_group"],
            columns=[*FEATURES, "t0", "t1", "need_prediction", "is_scored"],
            use_threads=False,
        )
        raw = np.column_stack([table[column].to_numpy() for column in FEATURES])
        xs.append(causal_features(raw))
        ys.append(np.column_stack([table[column].to_numpy() for column in ("t0", "t1")]))
        need = table["need_prediction"].to_numpy().astype(bool)
        need_masks.append(need)
        score_masks.append(need & table["is_scored"].to_numpy().astype(bool))
    x = np.stack(xs)
    y = np.clip(np.stack(ys), -2, 2).astype(np.float32)
    if x.shape[1:] != (20_000, 42) or not np.isfinite(y).all():
        raise ValueError(f"unexpected cohort shape or nonfinite labels: {x.shape}, {y.shape}")
    fit = np.array([node["role"] == "fit" for node in nodes])
    mean = x[fit].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale = np.maximum(x[fit].std(axis=(0, 1), dtype=np.float64), 0.001).astype(np.float32)
    x = np.clip((x - mean) / scale, -8, 8)
    manifest["normalization"] = {"fit_mean": mean.tolist(), "fit_scale": scale.tolist()}
    if not include_test:
        manifest["cohort"] = nodes
    return manifest, x, y, np.stack(need_masks), np.stack(score_masks), nodes


def loss_for_batch(
    prediction: torch.Tensor, target: torch.Tensor, need: torch.Tensor
) -> torch.Tensor:
    """Unchanged E467 per-target absolute-label-weighted MSE objective."""
    weights = target.abs() * need[..., None]
    mass = weights.sum(dim=(0, 1))
    return (
        (weights * (prediction - target).square()).sum(dim=(0, 1)) / mass.clamp_min(1e-12)
    ).mean()


def ensure_finite_parameters(model: nn.Module, arm: str) -> None:
    """Fail immediately if an optimizer creates a nonfinite parameter."""
    flags = torch.stack([torch.isfinite(parameter).all() for parameter in model.parameters()])
    if not bool(flags.all().item()):
        raise ValueError(f"optimizer produced nonfinite parameters in {arm}")


def train_update(
    model: ActivationGRU,
    optimizer: torch.optim.Optimizer,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    ids: list[int],
    device: torch.device,
    duty: object = None,
) -> float:
    """Run one 4-sequence by 256-row optimizer update on the first chunk."""
    xx = torch.from_numpy(x[ids, :256]).to(device)
    yy = torch.from_numpy(y[ids, :256]).to(device)
    mm = torch.from_numpy(need[ids, :256]).to(device)
    optimizer.zero_grad(set_to_none=True)
    prediction, _ = model(xx)
    loss = loss_for_batch(prediction, yy, mm)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite training loss")
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    ensure_finite_parameters(model, "CPU preflight")
    if duty is not None:
        torch.cuda.synchronize()
        duty.tick()
    return float(loss.detach().cpu())


def run_preflight(x: np.ndarray, y: np.ndarray, need: np.ndarray, nodes: list[dict]) -> dict:
    """One real B4/chunk256 update per arm, using fit rows only."""
    fit_ids = [index for index, node in enumerate(nodes) if node["role"] == "fit"]
    np.random.default_rng(SEED).shuffle(fit_ids)
    ids = fit_ids[:4]
    receipts = []
    for arm in ARMS:
        activation, optimizer_name = arm.split("__")
        torch.manual_seed(SEED)
        model = ActivationGRU(activation)
        optimizer = make_optimizer(model, optimizer_name)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        loss = train_update(model, optimizer, x, y, need, ids, torch.device("cpu"))
        updated = any(
            not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True)
        )
        finite = all(torch.isfinite(parameter).all().item() for parameter in model.parameters())
        if not updated or not finite:
            raise ValueError(f"real-data optimizer preflight failed for {arm}")
        receipts.append(
            {
                "arm": arm,
                "fit_row_groups": [nodes[index]["row_group"] for index in ids],
                "batch": 4,
                "chunk_rows": 256,
                "actual_optimizer_updates": 1,
                "loss": loss,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "parameters_changed": updated,
                "finite_parameters": finite,
            }
        )
    return {"kind": "fit-only real-data CPU preflight", "seed": SEED, "arms": receipts}


def run_arm(
    arm: str,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    score: np.ndarray,
    nodes: list[dict],
    device: torch.device,
    output: Path,
    duty: object = None,
) -> dict:
    activation, optimizer_name = arm.split("__")
    torch.manual_seed(SEED)
    model = ActivationGRU(activation).to(device)
    optimizer = make_optimizer(model, optimizer_name)
    fit_ids = [index for index, node in enumerate(nodes) if node["role"] == "fit"]
    np.random.default_rng(SEED).shuffle(fit_ids)
    steps, started = 0, time.monotonic()
    for offset in range(0, len(fit_ids), 4):
        ids = fit_ids[offset : offset + 4]
        if len(ids) != 4:
            raise ValueError("the fixed E467 recipe requires batches of exactly four")
        state = None
        for position in range(0, x.shape[1], 256):
            xx = torch.from_numpy(x[ids, position : position + 256]).to(device)
            yy = torch.from_numpy(y[ids, position : position + 256]).to(device)
            mm = torch.from_numpy(need[ids, position : position + 256]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, state = model(xx, state)
            loss = loss_for_batch(prediction, yy, mm)
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite loss in {arm}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            ensure_finite_parameters(model, arm)
            state = state.detach()
            if duty is not None:
                torch.cuda.synchronize()
                duty.tick()
            steps += 1
    fit_seconds = time.monotonic() - started
    if steps != 474:
        raise ValueError(f"expected exactly 474 updates, got {steps}")
    torch.save(model.state_dict(), output / f"{arm}.pt")
    model.eval()
    all_targets, all_predictions, per_sequence = [], [], []
    with torch.no_grad():
        for index, node in enumerate(nodes):
            if node["role"] != "test":
                continue
            state, predictions = None, []
            for position in range(0, x.shape[1], 256):
                prediction, state = model(
                    torch.from_numpy(x[index : index + 1, position : position + 256]).to(device),
                    state,
                )
                predictions.append(prediction[0].cpu().numpy())
            sequence_prediction = np.concatenate(predictions)
            selected = score[index]
            target_selected, prediction_selected = y[index][selected], sequence_prediction[selected]
            all_targets.append(target_selected)
            all_predictions.append(prediction_selected)
            per_sequence.append(
                {
                    "row_group": node["row_group"],
                    "scored_rows": len(target_selected),
                    "wp": [
                        weighted_pearson(target_selected[:, j], prediction_selected[:, j])
                        for j in range(2)
                    ],
                }
            )
    targets, predictions = np.concatenate(all_targets), np.concatenate(all_predictions)
    target_wp = [weighted_pearson(targets[:, j], predictions[:, j]) for j in range(2)]
    np.savez_compressed(
        output / f"{arm}_selected_predictions.npz", targets=targets, predictions=predictions
    )
    return {
        "arm": arm,
        "activation": activation,
        "optimizer": optimizer_name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "steps": steps,
        "fit_seconds": fit_seconds,
        "metric": "global_wp_clip2_v1",
        "wp": float(np.mean(target_wp)),
        "target_wp": target_wp,
        "scored_rows": len(targets),
        "sequences": per_sequence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preflight-real",
        action="store_true",
        help="fit-only real cohort, one B4/chunk256 update per arm",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output}")
    args.output.mkdir(parents=True)
    device = torch.device("cpu" if args.preflight_real else args.device)
    duty = configure_torch() if device.type == "cuda" else None
    manifest, x, y, need, score, nodes = load_cohort(include_test=not args.preflight_real)
    if args.preflight_real:
        result = run_preflight(x, y, need, nodes)
        result["cohort_sha256"] = hashlib.sha256(COHORT_PATH.read_bytes()).hexdigest()
    else:
        (args.output / "cohort.json").write_text(json.dumps(manifest, indent=2))
        result = {
            "kind": "fixed activation-by-optimizer screen",
            "seed": SEED,
            "cohort_sha256": hashlib.sha256(COHORT_PATH.read_bytes()).hexdigest(),
            "metric": "global_wp_clip2_v1",
            "exposure": "24 fit / 16 exposed diagnostic test components; mechanism screen only",
            "arms": [],
        }
        for arm in ARMS:
            item = run_arm(arm, x, y, need, score, nodes, device, args.output, duty)
            result["arms"].append(item)
            print(json.dumps(item), flush=True)
            (args.output / "result.json").write_text(json.dumps(result, indent=2))
    result["source_sha256"] = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in ("fit_activation_optimizer.py", "tests.py")
    }
    if device.type == "cuda":
        result["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    (args.output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
