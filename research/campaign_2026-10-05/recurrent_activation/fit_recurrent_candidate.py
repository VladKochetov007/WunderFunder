"""Fixed six-arm GRU candidate-activation screen using a scripted custom scan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
sys.path.insert(0, str(E470_DIR))
import fit_activation_optimizer as e470  # noqa: E402

from wnn.resources import configure_torch  # noqa: E402

SEED = 271828
WIDTH = 32
INPUT_SIZE = 42
CHUNK_ROWS = 256
ALPHAS = (("m10", -0.1), ("zero", 0.0), ("p10", 0.1))
OPTIMIZERS = ("adamw", "muon_adamw")
ARMS = tuple(f"{alpha_name}__{optimizer}" for alpha_name, _ in ALPHAS for optimizer in OPTIMIZERS)
DATASET_SHA256 = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"


@torch.jit.script
def candidate_from_tanh(u: Tensor, alpha: float) -> Tensor:
    """Bounded candidate perturbation n=u+alpha*sin(pi*u), for u=tanh(v)."""
    return u + alpha * torch.sin(math.pi * u)


@torch.jit.script
def scripted_candidate_scan(
    x: Tensor,
    h0: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Tensor,
    bias_hh: Tensor,
    alpha: float,
) -> tuple[Tensor, Tensor]:
    """Single-layer PyTorch-GRU equation scan with a modified candidate only.

    Gate ordering is reset, update, candidate. Like nn.GRU, the hidden affine
    candidate bias is inside the reset multiplication: r * (W_hn h + b_hn).
    The input projection is vectorized over time; the hidden recurrence is a
    TorchScript loop so no Python work occurs per row.
    """
    input_gates = F.linear(x, weight_ih, bias_ih)
    outputs = torch.jit.annotate(list[Tensor], [])
    hidden = h0
    for step in range(x.size(1)):
        input_reset, input_update, input_candidate = input_gates[:, step, :].chunk(3, dim=1)
        hidden_gates = F.linear(hidden, weight_hh, bias_hh)
        hidden_reset, hidden_update, hidden_candidate = hidden_gates.chunk(3, dim=1)
        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        u = torch.tanh(input_candidate + reset * hidden_candidate)
        candidate = candidate_from_tanh(u, alpha)
        hidden = (1.0 - update) * candidate + update * hidden
        outputs.append(hidden)
    return torch.stack(outputs, dim=1), hidden


class CandidateGRUModel(nn.Module):
    """E470 GRU+MLP with a custom candidate equation and unchanged state API."""

    def __init__(self, alpha: float, input_size: int = INPUT_SIZE, width: int = WIDTH) -> None:
        super().__init__()
        if alpha not in (-0.1, 0.0, 0.1):
            raise ValueError("alpha must be one of -0.1, 0.0, +0.1")
        self.alpha = float(alpha)
        self.embedding = nn.Linear(input_size, width)
        self.embedding_activation = nn.Tanh()
        # Preserve model.gru parameter names and initialization layout for the
        # existing Muon selector: gru.weight_ih_l0 / gru.weight_hh_l0.
        self.gru = nn.GRU(width, width, batch_first=True)
        self.anchor = nn.Linear(width, 2)
        hidden = max(1, round(width * 14 / (width + 3)))
        self.correction_hidden = nn.Linear(width, hidden)
        self.correction_activation = nn.Tanh()
        self.correction_output = nn.Linear(hidden, 2)
        nn.init.zeros_(self.correction_output.weight)
        nn.init.zeros_(self.correction_output.bias)

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.embedding.in_features:
            raise ValueError("x must have shape (batch, time, input_size)")
        encoded = self.embedding_activation(self.embedding(x))
        batch = encoded.shape[0]
        if state is None:
            hidden = encoded.new_zeros((batch, self.gru.hidden_size))
        else:
            if state.shape != (1, batch, self.gru.hidden_size):
                raise ValueError("GRU state must have shape (1, batch, hidden_size)")
            hidden = state[0]
        memory, final_hidden = scripted_candidate_scan(
            encoded,
            hidden,
            self.gru.weight_ih_l0,
            self.gru.weight_hh_l0,
            self.gru.bias_ih_l0,
            self.gru.bias_hh_l0,
            self.alpha,
        )
        residual = self.correction_output(
            self.correction_activation(self.correction_hidden(memory))
        )
        return self.anchor(memory) + residual, final_hidden.unsqueeze(0)


def alpha_for_arm(arm: str) -> float:
    name, optimizer = arm.split("__")
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer in arm: {arm}")
    for alpha_name, alpha in ALPHAS:
        if name == alpha_name:
            return alpha
    raise ValueError(f"unknown alpha arm: {arm}")


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)


def make_optimizer(model: CandidateGRUModel, name: str) -> torch.optim.Optimizer:
    """Reuse E470 optimizer settings and exact GRU gate-matrix partition."""
    return e470.make_optimizer(model, name)


def train_chunk_update(
    model: CandidateGRUModel,
    optimizer: torch.optim.Optimizer,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    ids: list[int],
    start: int = 0,
) -> float:
    xx = torch.from_numpy(x[ids, start : start + CHUNK_ROWS])
    yy = torch.from_numpy(y[ids, start : start + CHUNK_ROWS])
    mm = torch.from_numpy(need[ids, start : start + CHUNK_ROWS])
    optimizer.zero_grad(set_to_none=True)
    prediction, _ = model(xx)
    loss = e470.loss_for_batch(prediction, yy, mm)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite candidate-loss preflight")
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    e470.ensure_finite_parameters(model, f"candidate alpha={model.alpha}")
    return float(loss.detach())


def run_preflight(x: np.ndarray, y: np.ndarray, need: np.ndarray, nodes: list[dict]) -> dict:
    fit_ids = [index for index, node in enumerate(nodes) if node["role"] == "fit"]
    np.random.default_rng(SEED).shuffle(fit_ids)
    ids = fit_ids[:4]
    receipts = []
    for arm in ARMS:
        alpha = alpha_for_arm(arm)
        optimizer_name = arm.split("__")[1]
        set_seed()
        model = CandidateGRUModel(alpha)
        optimizer = make_optimizer(model, optimizer_name)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        loss = train_chunk_update(model, optimizer, x, y, need, ids)
        changed = any(
            not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True)
        )
        if not changed:
            raise ValueError(f"CPU preflight did not update parameters for {arm}")
        receipts.append(
            {
                "arm": arm,
                "alpha": alpha,
                "optimizer": optimizer_name,
                "fit_row_groups": [nodes[index]["row_group"] for index in ids],
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
    return {"kind": "fit-only real-data CPU update preflight", "seed": SEED, "arms": receipts}


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
    alpha = alpha_for_arm(arm)
    optimizer_name = arm.split("__")[1]
    set_seed()
    model = CandidateGRUModel(alpha).to(device)
    optimizer = make_optimizer(model, optimizer_name)
    fit_ids = [index for index, node in enumerate(nodes) if node["role"] == "fit"]
    np.random.default_rng(SEED).shuffle(fit_ids)
    started = time.monotonic()
    steps = 0
    for batch_start in range(0, len(fit_ids), 4):
        ids = fit_ids[batch_start : batch_start + 4]
        if len(ids) != 4:
            raise ValueError("the fixed E470 recipe requires batches of exactly four")
        state = None
        for position in range(0, x.shape[1], CHUNK_ROWS):
            xx = torch.from_numpy(x[ids, position : position + CHUNK_ROWS]).to(device)
            yy = torch.from_numpy(y[ids, position : position + CHUNK_ROWS]).to(device)
            mm = torch.from_numpy(need[ids, position : position + CHUNK_ROWS]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, state = model(xx, state)
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
        raise ValueError(f"expected exactly 474 optimizer updates, got {steps}")
    torch.save(model.state_dict(), output / f"{arm}.pt")
    model.eval()
    targets, predictions, by_sequence = [], [], []
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
            yy = y[index][selected]
            pp = sequence_prediction[selected]
            targets.append(yy)
            predictions.append(pp)
            by_sequence.append(
                {
                    "row_group": node["row_group"],
                    "scored_rows": len(yy),
                    "wp": [e470.weighted_pearson(yy[:, j], pp[:, j]) for j in range(2)],
                }
            )
    target, prediction = np.concatenate(targets), np.concatenate(predictions)
    target_wp = [e470.weighted_pearson(target[:, j], prediction[:, j]) for j in range(2)]
    np.savez_compressed(
        output / f"{arm}_selected_predictions.npz", targets=target, predictions=prediction
    )
    return {
        "arm": arm,
        "alpha": alpha,
        "optimizer": optimizer_name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "steps": steps,
        "fit_seconds": fit_seconds,
        "metric": "global_wp_clip2_v1",
        "wp": float(np.mean(target_wp)),
        "target_wp": target_wp,
        "scored_rows": len(target),
        "sequences": by_sequence,
    }


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-real", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output}")
    args.output.mkdir(parents=True)
    device = torch.device("cpu" if args.preflight_real else args.device)
    duty = configure_torch() if device.type == "cuda" else None
    manifest, x, y, need, score, nodes = e470.load_cohort(include_test=not args.preflight_real)
    if args.preflight_real:
        result = run_preflight(x, y, need, nodes)
    else:
        (args.output / "cohort.json").write_text(json.dumps(manifest, indent=2))
        result = {
            "kind": "fixed recurrent candidate activation screen",
            "seed": SEED,
            "cohort_sha256": hashlib.sha256(e470.COHORT_PATH.read_bytes()).hexdigest(),
            "dataset_sha256": DATASET_SHA256,
            "metric": "global_wp_clip2_v1",
            "scope": "24 fit / 16 exposed diagnostic components; mechanism screen only",
            "custom_scan": "TorchScript candidate recurrence; PyTorch GRU reset-after-hidden affine and bias placement",
            "arms": [],
        }
        for arm in ARMS:
            item = run_arm(arm, x, y, need, score, nodes, device, args.output, duty)
            result["arms"].append(item)
            print(json.dumps(item), flush=True)
            (args.output / "result.json").write_text(json.dumps(result, indent=2))
    result["recipe_source_sha256"] = file_hash(E470_DIR / "fit_activation_optimizer.py")
    result["source_sha256"] = {
        filename: file_hash(HERE / filename)
        for filename in ("fit_recurrent_candidate.py", "tests.py", "gpu_probe.py")
        if (HERE / filename).is_file()
    }
    result["cohort_sha256"] = hashlib.sha256(e470.COHORT_PATH.read_bytes()).hexdigest()
    result["dataset_sha256"] = DATASET_SHA256
    if device.type == "cuda":
        result["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    (args.output / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
