"""Twelve-fit activation-placement screen on the incumbent GRU topology.

This trainer imports E470's fixed 24-fit cohort, loss, and scorer helpers, but
loads fit-role rows only. It never opens test row groups. Scores and planned
contrasts are left to the independent parent-owned grader.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E470_DIR))

import fit_activation_optimizer as e470  # noqa: E402

from wnn.nn import RNNModel  # noqa: E402
from wnn.resources import configure_torch  # noqa: E402

SEEDS = (271828, 271829)
HIDDEN = 64
FEATURES = 42
EXPECTED_PARAMETERS = 61_058
CHUNK_ROWS = 256
UPDATES_PER_FIT = 474
TELEMETRY_STEPS = (0, 46, 236, 473)
TELEMETRY_PROTOCOL = {
    "steps_zero_based": list(TELEMETRY_STEPS),
    "families": ["inp", "gru", "head"],
    "gradient": "element-weighted RMS before global norm clipping",
    "parameter_update": "element-weighted RMS of optimizer parameter delta",
    "fit_rows_only": True,
    "descriptive_only": True,
}
ADAMW_LR = 0.001
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
ACTIVATION_FORMULAS = {
    "gelu": "F.gelu(x, approximate='none')",
    "sine": "F.gelu(x, approximate='none') + 0.1*sin(x)",
    "slope": "F.gelu(x, approximate='none') + 0.1*x",
}
MODELS = (
    ("base", "gelu", "gelu"),
    ("sine_embed", "sine", "gelu"),
    ("slope_embed", "slope", "gelu"),
    ("sine_head", "gelu", "sine"),
    ("slope_head", "gelu", "slope"),
    ("sine_both", "sine", "sine"),
)
ARMS = tuple((seed, name, embedding, head) for seed in SEEDS for name, embedding, head in MODELS)
PLANNED_CONTRASTS = (
    "sine_embed - base",
    "slope_embed - base",
    "sine_embed - slope_embed",
    "sine_head - base",
    "slope_head - base",
    "sine_head - slope_head",
    "sine_both - sine_embed - sine_head + base",
)
DATASET_SHA256 = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"


class PlacementActivation(nn.Module):
    """Parameter-free exact GELU and matched-slope pointwise choices."""

    def __init__(self, name: str) -> None:
        super().__init__()
        if name not in ACTIVATION_FORMULAS:
            raise ValueError(f"unknown activation formula: {name}")
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gelu = F.gelu(x, approximate="none")
        if self.name == "gelu":
            return gelu
        if self.name == "sine":
            return gelu + 0.1 * torch.sin(x)
        if self.name == "slope":
            return gelu + 0.1 * x
        raise AssertionError("activation dispatch is exhaustive")


class PlacementRNN(RNNModel):
    """Two-layer RNNModel topology with explicit embedding/head activations."""

    def __init__(
        self,
        embedding_activation: str,
        head_activation: str,
        n_in: int = FEATURES,
        hidden: int = HIDDEN,
    ) -> None:
        super().__init__(n_in=n_in, hidden=hidden, layers=2, dropout=0.0, n_out=2)
        self.embedding_activation_name = embedding_activation
        self.head_activation_name = head_activation
        self.embedding_activation = PlacementActivation(embedding_activation)
        self.head[1] = PlacementActivation(head_activation)

    def encode(self, xn: torch.Tensor) -> torch.Tensor:
        return self.embedding_activation(self.inp(xn))


@contextlib.contextmanager
def full_ieee_float32() -> Iterator[None]:
    """Temporarily force FP32 IEEE math for CUDA matmul and cuDNN RNN alike."""
    backends = torch.backends
    if not torch.cuda.is_available():
        with torch.autocast(device_type="cpu", enabled=False):
            yield
        return

    required = (
        hasattr(backends, "fp32_precision"),
        hasattr(backends.cuda.matmul, "fp32_precision"),
        hasattr(backends.cudnn, "fp32_precision"),
        hasattr(backends.cudnn.rnn, "fp32_precision"),
    )
    if not all(required):
        raise RuntimeError("this screen requires PyTorch per-backend fp32_precision controls")
    settings = (
        (backends, "fp32_precision"),
        (backends.cuda.matmul, "fp32_precision"),
        (backends.cudnn, "fp32_precision"),
        (backends.cudnn.rnn, "fp32_precision"),
    )
    previous = [getattr(owner, name) for owner, name in settings]
    try:
        for owner, name in settings:
            setattr(owner, name, "ieee")
        with torch.autocast(device_type="cuda", enabled=False):
            yield
    finally:
        for (owner, name), value in zip(settings, previous, strict=True):
            setattr(owner, name, value)


def precision_metadata() -> dict:
    """Describe active internal arithmetic policy for saved fit receipts."""
    if not torch.cuda.is_available():
        return {"device_math": "CPU FP32", "autocast": False}
    return {
        "device_math": "FP32 IEEE",
        "autocast": False,
        "torch.backends.fp32_precision": torch.backends.fp32_precision,
        "torch.backends.cuda.matmul.fp32_precision": torch.backends.cuda.matmul.fp32_precision,
        "torch.backends.cudnn.fp32_precision": torch.backends.cudnn.fp32_precision,
        "torch.backends.cudnn.rnn.fp32_precision": torch.backends.cudnn.rnn.fp32_precision,
    }


def file_hash(path: Path) -> str:
    """Hash a source or data file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_optimizer(model: PlacementRNN) -> torch.optim.AdamW:
    """Build the fixed AdamW optimizer used for every one of the twelve fits."""
    return torch.optim.AdamW(
        model.parameters(), lr=ADAMW_LR, weight_decay=WEIGHT_DECAY, betas=BETAS
    )


def make_seed_states() -> dict[int, dict[str, torch.Tensor]]:
    """Create one baseline initialization per seed for every matched arm."""
    states: dict[int, dict[str, torch.Tensor]] = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        baseline = PlacementRNN("gelu", "gelu")
        count = sum(parameter.numel() for parameter in baseline.parameters())
        if count != EXPECTED_PARAMETERS:
            raise ValueError(f"expected {EXPECTED_PARAMETERS} parameters, got {count}")
        states[seed] = {key: value.detach().clone() for key, value in baseline.state_dict().items()}
    return states


def initialize_arm(
    seed: int,
    embedding_activation: str,
    head_activation: str,
    initial_state: dict[str, torch.Tensor],
) -> PlacementRNN:
    """Load the same full baseline state into a parameter-free activation variant."""
    model = PlacementRNN(embedding_activation, head_activation)
    model.load_state_dict(initial_state, strict=True)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EXPECTED_PARAMETERS:
        raise ValueError(f"expected {EXPECTED_PARAMETERS} parameters, got {count}")
    return model


def fit_one_update(
    model: PlacementRNN,
    optimizer: torch.optim.Optimizer,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    ids: list[int],
    device: torch.device,
) -> tuple[float, dict]:
    """Apply one B4×256 fit-only optimizer update with finite checks."""
    xx = torch.from_numpy(x[ids, :CHUNK_ROWS]).to(device=device, dtype=torch.float32)
    yy = torch.from_numpy(y[ids, :CHUNK_ROWS]).to(device=device, dtype=torch.float32)
    mm = torch.from_numpy(need[ids, :CHUNK_ROWS]).to(device)
    if not (torch.isfinite(xx).all() and torch.isfinite(yy).all()):
        raise ValueError("nonfinite raw fit-only probe batch")
    optimizer.zero_grad(set_to_none=True)
    prediction, _ = model(xx)
    if not torch.isfinite(prediction).all():
        raise ValueError("nonfinite model prediction")
    loss = e470.loss_for_batch(prediction, yy, mm)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite fit loss")
    loss.backward()
    telemetry = apply_optimizer_update(
        model, optimizer, parameter_families(model), 0, model.embedding_activation_name
    )
    if telemetry is None:
        raise AssertionError("preflight update must be sampled at update zero")
    return float(loss.detach().cpu()), telemetry


def parameter_families(model: PlacementRNN) -> dict[str, list[nn.Parameter]]:
    """Group all trainable parameters by their unchanged baseline module names."""
    families = {"inp": [], "gru": [], "head": []}
    for name, parameter in model.named_parameters():
        family = name.split(".", maxsplit=1)[0]
        if family not in families:
            raise ValueError(f"unassigned trainable parameter: {name}")
        families[family].append(parameter)
    if any(not parameters for parameters in families.values()):
        raise ValueError("every telemetry family must contain trainable parameters")
    if sum(len(parameters) for parameters in families.values()) != len(list(model.parameters())):
        raise ValueError("telemetry family partition is incomplete")
    return families


def family_rms(values: dict[str, list[torch.Tensor]]) -> dict[str, float]:
    """Compute the element-weighted RMS for each parameter family."""
    result = {}
    for family, tensors in values.items():
        if not tensors:
            raise ValueError(f"empty telemetry family: {family}")
        squared_sum = sum(tensor.detach().double().square().sum() for tensor in tensors)
        element_count = sum(tensor.numel() for tensor in tensors)
        result[family] = float(torch.sqrt(squared_sum / element_count).cpu())
    return result


def apply_optimizer_update(
    model: PlacementRNN,
    optimizer: torch.optim.Optimizer,
    families: dict[str, list[nn.Parameter]],
    update_index: int,
    owner: str,
) -> dict | None:
    """Clip, update, verify finiteness, and collect only fixed-step telemetry."""
    sampled = update_index in TELEMETRY_STEPS
    preclip_gradient_rms = None
    parameter_before = None
    if sampled:
        gradients = {
            family: [parameter.grad for parameter in parameters if parameter.grad is not None]
            for family, parameters in families.items()
        }
        if any(len(gradients[family]) != len(families[family]) for family in families):
            raise ValueError("sampled update has a missing family gradient")
        preclip_gradient_rms = family_rms(gradients)
        parameter_before = {
            family: [parameter.detach().clone() for parameter in parameters]
            for family, parameters in families.items()
        }
    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    e470.ensure_finite_parameters(model, owner)
    if not sampled:
        return None
    if parameter_before is None or preclip_gradient_rms is None:
        raise AssertionError("sampled update telemetry snapshots were not captured")
    update_delta = {
        family: [
            parameter.detach() - before
            for parameter, before in zip(families[family], parameter_before[family], strict=True)
        ]
        for family in families
    }
    return {
        "update_index_zero_based": update_index,
        "preclip_gradient_rms": preclip_gradient_rms,
        "parameter_update_rms": family_rms(update_delta),
    }


def run_preflight(
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    score: np.ndarray,
    nodes: list[dict],
    initial_states: dict[int, dict[str, torch.Tensor]],
) -> dict:
    """Make one CPU update for each planned arm using fit row groups only."""
    if x.shape != (24, 20_000, FEATURES):
        raise ValueError(f"expected exact (24, 20000, 42) fit feature array, got {x.shape}")
    if y.shape != (24, 20_000, 2) or need.shape != (24, 20_000) or score.shape != need.shape:
        raise ValueError("unexpected fit label, need-mask or score-mask shape")
    if len(nodes) != 24 or {node["role"] for node in nodes} != {"fit"}:
        raise ValueError("CPU preflight must load exactly the 24 fit row groups")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("nonfinite fit-only cohort arrays")

    results = []
    for seed in SEEDS:
        fit_ids = list(range(len(nodes)))
        np.random.default_rng(seed).shuffle(fit_ids)
        ids = fit_ids[:4]
        row_groups = [nodes[index]["row_group"] for index in ids]
        for name, embedding_activation, head_activation in MODELS:
            model = initialize_arm(
                seed, embedding_activation, head_activation, initial_states[seed]
            )
            optimizer = make_optimizer(model)
            before = [parameter.detach().clone() for parameter in model.parameters()]
            loss, telemetry = fit_one_update(model, optimizer, x, y, need, ids, torch.device("cpu"))
            changed = any(
                not torch.equal(old, new)
                for old, new in zip(before, model.parameters(), strict=True)
            )
            finite = all(torch.isfinite(parameter).all().item() for parameter in model.parameters())
            if not changed or not finite:
                raise ValueError(f"preflight update failed for seed={seed}/{name}")
            results.append(
                {
                    "seed": seed,
                    "arm": name,
                    "embedding_activation": embedding_activation,
                    "head_activation": head_activation,
                    "fit_row_groups": row_groups,
                    "batch": 4,
                    "chunk_rows": CHUNK_ROWS,
                    "optimizer_updates": 1,
                    "loss": loss,
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "parameters_changed": changed,
                    "parameters_finite": finite,
                    "telemetry": telemetry,
                }
            )
    return {"kind": "fit-only CPU placement preflight", "arms": results}


def run_fit(
    seed: int,
    name: str,
    embedding_activation: str,
    head_activation: str,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    nodes: list[dict],
    device: torch.device,
    output: Path,
    initial_state: dict[str, torch.Tensor],
) -> dict:
    """Train one seed/placement arm, without reading test groups or labels."""
    fit_ids = list(range(len(nodes)))
    np.random.default_rng(seed).shuffle(fit_ids)
    if len(fit_ids) != 24 or len(fit_ids) % 4:
        raise ValueError("expected 24 fit groups and B4 batches")
    model = initialize_arm(seed, embedding_activation, head_activation, initial_state)
    model = model.to(device).float()
    optimizer = make_optimizer(model)
    steps, started = 0, time.monotonic()
    families = parameter_families(model)
    telemetry = []
    for batch_start in range(0, len(fit_ids), 4):
        ids = fit_ids[batch_start : batch_start + 4]
        state = None
        for position in range(0, x.shape[1], CHUNK_ROWS):
            xx = torch.from_numpy(x[ids, position : position + CHUNK_ROWS]).to(
                device=device, dtype=torch.float32
            )
            yy = torch.from_numpy(y[ids, position : position + CHUNK_ROWS]).to(
                device=device, dtype=torch.float32
            )
            mm = torch.from_numpy(need[ids, position : position + CHUNK_ROWS]).to(device)
            if not (torch.isfinite(xx).all() and torch.isfinite(yy).all()):
                raise ValueError(f"nonfinite raw training batch in seed={seed}/{name}")
            optimizer.zero_grad(set_to_none=True)
            prediction, state = model(xx, state)
            if not torch.isfinite(prediction).all():
                raise ValueError(f"nonfinite prediction in seed={seed}/{name}")
            loss = e470.loss_for_batch(prediction, yy, mm)
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite loss in seed={seed}/{name}")
            loss.backward()
            update_index = steps
            step_telemetry = apply_optimizer_update(
                model, optimizer, families, update_index, f"seed={seed}/{name}"
            )
            if step_telemetry is not None:
                step_telemetry.update(
                    {
                        "fit_row_groups": [nodes[index]["row_group"] for index in ids],
                        "chunk_start": position,
                    }
                )
                telemetry.append(step_telemetry)
            state = state.detach()
            steps += 1
    elapsed = time.monotonic() - started
    if steps != UPDATES_PER_FIT:
        raise ValueError(f"expected {UPDATES_PER_FIT} updates, got {steps}")
    if tuple(item["update_index_zero_based"] for item in telemetry) != TELEMETRY_STEPS:
        raise ValueError("fixed telemetry step sample is incomplete")
    checkpoint = output / f"seed{seed}_{name}.pt"
    torch.save(model.cpu().state_dict(), checkpoint)
    return {
        "seed": seed,
        "arm": name,
        "embedding_activation": embedding_activation,
        "head_activation": head_activation,
        "activation_formulas": {
            "embedding": ACTIVATION_FORMULAS[embedding_activation],
            "head": ACTIVATION_FORMULAS[head_activation],
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "updates": steps,
        "fit_seconds": elapsed,
        "fit_row_groups": [nodes[index]["row_group"] for index in fit_ids],
        "telemetry_sampling_rule": {
            "steps_zero_based": list(TELEMETRY_STEPS),
            "scope": "actual training update, fit rows only, same fixed indices for every arm",
            "families": ["inp", "gru", "head"],
            "gradient": "element-weighted RMS before global norm clipping",
            "parameter_update": "element-weighted RMS of optimizer parameter delta",
        },
        "telemetry": telemetry,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": file_hash(checkpoint),
    }


def main() -> None:
    """Fit all twelve planned models or run their fit-only CPU preflight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-real", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output}")
    args.output.mkdir(parents=True)
    device = torch.device("cpu" if args.preflight_real else args.device)
    torch.set_num_threads(1)
    if device.type == "cuda":
        configure_torch()
    manifest, x, y, need, score, nodes = e470.load_cohort(include_test=False)
    if len(nodes) != 24 or any(node["role"] != "fit" for node in nodes):
        raise ValueError("trainer must load fit rows only")
    if x.shape != (24, 20_000, FEATURES):
        raise ValueError(f"expected exact feature count/shape, got {x.shape}")
    initial_states = make_seed_states()

    result = {
        "kind": "twelve-fit localized activation placement screen",
        "status": "preflight" if args.preflight_real else "fitting",
        "fit_only": True,
        "test_rows_loaded": False,
        "seeds": list(SEEDS),
        "planned_arms": [
            {
                "seed": seed,
                "arm": name,
                "embedding_activation": embedding,
                "head_activation": head,
            }
            for seed, name, embedding, head in ARMS
        ],
        "planned_contrasts": list(PLANNED_CONTRASTS),
        "architecture": "RNNModel topology: input GELU -> 2-layer nn.GRU -> concat recurrent+embedding -> GELU head",
        "hidden": HIDDEN,
        "trainable_parameters_per_arm": EXPECTED_PARAMETERS,
        "input_features": FEATURES,
        "batch": 4,
        "chunk_rows": CHUNK_ROWS,
        "updates_per_fit": UPDATES_PER_FIT,
        "telemetry_protocol": TELEMETRY_PROTOCOL,
        "optimizer": {
            "name": "AdamW",
            "lr": ADAMW_LR,
            "weight_decay": WEIGHT_DECAY,
            "betas": list(BETAS),
            "schedule": "constant",
            "gradient_clip": 1.0,
        },
        "cohort_sha256": file_hash(e470.COHORT_PATH),
        "dataset_sha256": DATASET_SHA256,
        "recipe_source_sha256": file_hash(E470_DIR / "fit_activation_optimizer.py"),
        "source_sha256": {
            name: file_hash(HERE / name)
            for name in ("fit_activation_placement.py", "tests.py", "gpu_probe.py")
            if (HERE / name).is_file()
        },
    }
    (args.output / "fit_only_cohort.json").write_text(json.dumps(manifest, indent=2))
    if args.preflight_real:
        result["arms"] = run_preflight(x, y, need, score, nodes, initial_states)["arms"]
        result["status"] = "preflight_passed"
    else:
        with full_ieee_float32():
            result["precision"] = precision_metadata()
            result["arms"] = []
            for seed, name, embedding, head in ARMS:
                item = run_fit(
                    seed,
                    name,
                    embedding,
                    head,
                    x,
                    y,
                    need,
                    nodes,
                    device,
                    args.output,
                    initial_states[seed],
                )
                result["arms"].append(item)
                result["status"] = "fitting"
                (args.output / "fit_manifest.json").write_text(json.dumps(result, indent=2))
                print(json.dumps(item), flush=True)
        if len(result["arms"]) != 12:
            raise ValueError("all twelve predeclared models must finish before handoff")
        result["status"] = "all_12_fit_arms_complete_test_unread"
    (args.output / "fit_manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
