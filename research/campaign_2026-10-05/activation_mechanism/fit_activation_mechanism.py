"""Five-formula activation mechanism screen using the frozen E470 recipe.

This module reuses E470's loader, loss, metric and optimizer grouping without
editing that trainer. It is a draft runner; full fits require a parent-reviewed
content seal and tracked launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
sys.path.insert(0, str(E470_DIR))

import fit_activation_optimizer as e470  # noqa: E402

from wnn.resources import configure_torch  # noqa: E402

SEED = 271828
CHUNK_ROWS = 256
ACTIVATIONS = (
    "gelu",
    "gelu_plus_sine_01",
    "gelu_plus_linear_01",
    "scaled_gelu_11",
    "jax_tanh_gelu_plus_sine_01",
)
OPTIMIZERS = ("adamw", "muon_adamw")
ARMS = tuple(f"{activation}__{optimizer}" for activation in ACTIVATIONS for optimizer in OPTIMIZERS)


class MechanismActivation(nn.Module):
    """Parameter-free activation functions isolating the proposed contrasts."""

    def __init__(self, name: str) -> None:
        super().__init__()
        if name not in ACTIVATIONS:
            raise ValueError(f"unknown activation: {name}")
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gelu = F.gelu(x)
        if self.name == "gelu":
            return gelu
        if self.name == "gelu_plus_sine_01":
            return gelu + 0.1 * torch.sin(x)
        if self.name == "gelu_plus_linear_01":
            return gelu + 0.1 * x
        if self.name == "scaled_gelu_11":
            return 1.1 * gelu
        if self.name == "jax_tanh_gelu_plus_sine_01":
            return F.gelu(x, approximate="tanh") + 0.1 * torch.sin(x)
        raise AssertionError("activation dispatch is exhaustive")


class MechanismGRU(e470.ActivationGRU):
    """E470 layout with one matched activation choice at embedding and head."""

    def __init__(self, activation: str, input_size: int = 42, width: int = 32) -> None:
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation: {activation}")
        super().__init__("gelu", input_size=input_size, width=width)
        self.activation_name = activation
        self.embedding_activation = MechanismActivation(activation)
        self.correction_activation = MechanismActivation(activation)


def file_hash(path: Path) -> str:
    """Return the SHA-256 digest of a file using bounded reads."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_update(
    model: MechanismGRU,
    optimizer: torch.optim.Optimizer,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    ids: list[int],
    device: torch.device,
) -> float:
    """Run one real B4×256 fit-only update and reject nonfinite values."""
    xx = torch.from_numpy(x[ids, :CHUNK_ROWS]).to(device)
    yy = torch.from_numpy(y[ids, :CHUNK_ROWS]).to(device)
    mm = torch.from_numpy(need[ids, :CHUNK_ROWS]).to(device)
    optimizer.zero_grad(set_to_none=True)
    prediction, _ = model(xx)
    if not torch.isfinite(prediction).all():
        raise ValueError("nonfinite activation-mechanism prediction")
    loss = e470.loss_for_batch(prediction, yy, mm)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite activation-mechanism loss")
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    e470.ensure_finite_parameters(model, model.activation_name)
    return float(loss.detach().cpu())


def run_preflight(x: np.ndarray, y: np.ndarray, need: np.ndarray, nodes: list[dict]) -> dict:
    """Make one CPU update per arm from only the first shuffled fit batch."""
    fit_ids = [index for index, node in enumerate(nodes) if node["role"] == "fit"]
    np.random.default_rng(SEED).shuffle(fit_ids)
    ids = fit_ids[:4]
    row_groups = [nodes[index]["row_group"] for index in ids]
    results = []
    for activation in ACTIVATIONS:
        for optimizer_name in OPTIMIZERS:
            torch.manual_seed(SEED)
            model = MechanismGRU(activation)
            optimizer = e470.make_optimizer(model, optimizer_name)
            before = [parameter.detach().clone() for parameter in model.parameters()]
            loss = train_update(model, optimizer, x, y, need, ids, torch.device("cpu"))
            changed = any(
                not torch.equal(old, new)
                for old, new in zip(before, model.parameters(), strict=True)
            )
            if not changed:
                raise ValueError(
                    f"CPU preflight made no parameter update in {activation}/{optimizer_name}"
                )
            results.append(
                {
                    "arm": f"{activation}__{optimizer_name}",
                    "fit_row_groups": row_groups,
                    "batch": 4,
                    "chunk_rows": CHUNK_ROWS,
                    "optimizer_updates": 1,
                    "loss": loss,
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "parameters_changed": changed,
                    "parameters_finite": all(
                        torch.isfinite(parameter).all().item() for parameter in model.parameters()
                    ),
                }
            )
    return {"kind": "fit-only real-data CPU activation-mechanism preflight", "arms": results}


def run_arm(
    arm: str,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    score: np.ndarray,
    nodes: list[dict],
    device: torch.device,
    output: Path,
    duty: object,
) -> dict:
    """Fit one fixed arm with the unchanged E470 sequence recipe and scorer."""
    activation, optimizer_name = arm.split("__")
    torch.manual_seed(SEED)
    model = MechanismGRU(activation).to(device)
    optimizer = e470.make_optimizer(model, optimizer_name)
    fit_ids = [index for index, node in enumerate(nodes) if node["role"] == "fit"]
    np.random.default_rng(SEED).shuffle(fit_ids)
    steps = 0
    started = time.monotonic()
    for batch_start in range(0, len(fit_ids), 4):
        ids = fit_ids[batch_start : batch_start + 4]
        if len(ids) != 4:
            raise ValueError("the fixed E470 recipe requires complete batches of four")
        state = None
        for position in range(0, x.shape[1], CHUNK_ROWS):
            xx = torch.from_numpy(x[ids, position : position + CHUNK_ROWS]).to(device)
            yy = torch.from_numpy(y[ids, position : position + CHUNK_ROWS]).to(device)
            mm = torch.from_numpy(need[ids, position : position + CHUNK_ROWS]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, state = model(xx, state)
            if not torch.isfinite(prediction).all():
                raise ValueError(f"nonfinite prediction in {arm}")
            loss = e470.loss_for_batch(prediction, yy, mm)
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite loss in {arm}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            e470.ensure_finite_parameters(model, arm)
            state = state.detach()
            if duty is not None:
                torch.cuda.synchronize()
                duty.tick()
            steps += 1
    fit_seconds = time.monotonic() - started
    if steps != 474:
        raise ValueError(f"expected 474 optimizer updates, got {steps}")
    torch.save(model.state_dict(), output / f"{arm}.pt")

    model.eval()
    targets, predictions, sequences = [], [], []
    with torch.no_grad():
        for index, node in enumerate(nodes):
            if node["role"] != "test":
                continue
            state, sequence_parts = None, []
            for position in range(0, x.shape[1], CHUNK_ROWS):
                prediction, state = model(
                    torch.from_numpy(x[index : index + 1, position : position + CHUNK_ROWS]).to(
                        device
                    ),
                    state,
                )
                sequence_parts.append(prediction[0].cpu().numpy())
            sequence_prediction = np.concatenate(sequence_parts)
            if not np.isfinite(sequence_prediction).all():
                raise ValueError(f"nonfinite full-sequence prediction in {arm}")
            selected = score[index]
            target_selected = y[index][selected]
            prediction_selected = sequence_prediction[selected]
            targets.append(target_selected)
            predictions.append(prediction_selected)
            sequences.append(
                {
                    "row_group": node["row_group"],
                    "scored_rows": len(target_selected),
                    "wp": [
                        e470.weighted_pearson(
                            target_selected[:, column], prediction_selected[:, column]
                        )
                        for column in range(2)
                    ],
                }
            )
    target, prediction = np.concatenate(targets), np.concatenate(predictions)
    target_wp = [
        e470.weighted_pearson(target[:, column], prediction[:, column]) for column in range(2)
    ]
    np.savez_compressed(
        output / f"{arm}_selected_predictions.npz", targets=target, predictions=prediction
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
        "scored_rows": len(target),
        "sequences": sequences,
    }


def main() -> None:
    """Run a fit-only preflight or the parent-authorized fixed six-sequence recipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-real", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output}")
    args.output.mkdir(parents=True)
    device = torch.device("cpu" if args.preflight_real else args.device)
    duty = configure_torch() if device.type == "cuda" else None
    torch.set_num_threads(1)
    manifest, x, y, need, score, nodes = e470.load_cohort(include_test=not args.preflight_real)
    if args.preflight_real:
        result = run_preflight(x, y, need, nodes)
    else:
        (args.output / "cohort.json").write_text(json.dumps(manifest, indent=2))
        result = {
            "kind": "fixed activation-mechanism by optimizer screen",
            "seed": SEED,
            "metric": "global_wp_clip2_v1",
            "scope": "24 fit / 16 exposed diagnostic components; mechanism screen only",
            "arms": [],
        }
        for arm in ARMS:
            item = run_arm(arm, x, y, need, score, nodes, device, args.output, duty)
            result["arms"].append(item)
            print(json.dumps(item), flush=True)
            (args.output / "result.json").write_text(json.dumps(result, indent=2))
    result["cohort_sha256"] = file_hash(e470.COHORT_PATH)
    result["dataset_sha256"] = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
    result["recipe_source_sha256"] = file_hash(E470_DIR / "fit_activation_optimizer.py")
    result["source_sha256"] = {
        name: file_hash(HERE / name)
        for name in ("fit_activation_mechanism.py", "tests.py")
        if (HERE / name).is_file()
    }
    if device.type == "cuda":
        result["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    (args.output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
