"""Fit-only E477 head-mixture by dropout screen on the E474 H64 topology."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E470_DIR))
sys.path.insert(0, str(E474_DIR))

import fit_activation_optimizer as e470  # noqa: E402
import fit_activation_placement as e474  # noqa: E402

from wnn.resources import configure_torch  # noqa: E402

SEEDS = (271828, 271829)
EXPECTED_PARAMETERS = 61_058
CHUNK_ROWS = 256
UPDATES_PER_FIT = 474
ACTIVATIONS = ("gelu", "channel_split", "same_channel_mix")
DROPOUTS = (0.0, 0.1)
ARMS = tuple(
    (seed, activation, p) for seed in SEEDS for activation in ACTIVATIONS for p in DROPOUTS
)
CONTRASTS = (
    "split_p0 - gelu_p0",
    "mix_p0 - gelu_p0",
    "split_p0 - mix_p0",
    "split_p01 - gelu_p01",
    "mix_p01 - gelu_p01",
    "split_p01 - mix_p01",
    "(mix_p01 - mix_p0) - (gelu_p01 - gelu_p0)",
)
E470_SOURCE_SHA256 = "871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9"
E474_SOURCE_SHA256 = "40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068"
DATASET_SHA256 = e474.DATASET_SHA256


class ReadoutActivationDropout(nn.Module):
    """Parameter-free head activation followed immediately by dropout."""

    def __init__(self, activation: str, dropout: float, hidden: int = 64) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown readout activation: {activation}")
        if dropout not in DROPOUTS:
            raise ValueError(f"unplanned dropout probability: {dropout}")
        if activation == "channel_split" and hidden % 2:
            raise ValueError("channel-split readout requires even hidden width")
        self.activation_name = activation
        self.dropout_probability = dropout
        self.dropout = nn.Dropout(p=dropout)
        self.hidden = hidden

    def activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "gelu":
            return F.gelu(x, approximate="none")
        if self.activation_name == "channel_split":
            split = self.hidden // 2
            return torch.cat(
                (
                    F.gelu(x[..., :split], approximate="none"),
                    F.leaky_relu(x[..., split:], negative_slope=0.01),
                ),
                dim=-1,
            )
        return 0.5 * F.gelu(x, approximate="none") + 0.5 * F.leaky_relu(x, negative_slope=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activate(x))


class HeadMixtureRNN(e474.PlacementRNN):
    """E474 model with only its readout activation and dropout varied."""

    def __init__(self, activation: str, dropout: float) -> None:
        super().__init__(embedding_activation="gelu", head_activation="gelu")
        self.head_activation_name = activation
        self.head[1] = ReadoutActivationDropout(activation, dropout, hidden=64)
        self.dropout_probability = dropout


def check_parent_sources() -> None:
    """Fail closed if either frozen parent trainer changed after preregistration."""
    actual_e470 = e474.file_hash(E470_DIR / "fit_activation_optimizer.py")
    actual_e474 = e474.file_hash(E474_DIR / "fit_activation_placement.py")
    if actual_e470 != E470_SOURCE_SHA256 or actual_e474 != E474_SOURCE_SHA256:
        raise ValueError(f"parent source hash mismatch: E470={actual_e470}, E474={actual_e474}")


def make_initial_states() -> dict[int, dict[str, torch.Tensor]]:
    """Reuse the E474 per-seed baseline initialization byte-for-byte."""
    return e474.make_seed_states()


def initialize_arm(
    activation: str,
    dropout: float,
    state: dict[str, torch.Tensor],
) -> HeadMixtureRNN:
    """Clone the baseline tensor state, replacing only parameter-free modules."""
    model = HeadMixtureRNN(activation, dropout)
    model.load_state_dict(state, strict=True)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EXPECTED_PARAMETERS:
        raise ValueError(f"expected {EXPECTED_PARAMETERS} parameters, got {count}")
    return model


def arm_name(activation: str, dropout: float) -> str:
    suffix = "p0" if dropout == 0 else "p01"
    return f"{activation}_{suffix}"


def run_cpu_preflight(
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    nodes: list[dict],
    states: dict[int, dict[str, torch.Tensor]],
) -> list[dict]:
    """Perform one real fit-only B4x256 optimizer step for each fixed arm."""
    if x.shape != (24, 20_000, 42) or y.shape != (24, 20_000, 2):
        raise ValueError("preflight requires the exact fit-only 24 x 20k cohort")
    if len(nodes) != 24 or any(node["role"] != "fit" for node in nodes):
        raise ValueError("preflight must only receive the 24 fit row groups")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("nonfinite fit-only preflight arrays")
    result = []
    for seed in SEEDS:
        order = np.arange(24)
        np.random.default_rng(seed).shuffle(order)
        ids = order[:4].tolist()
        groups = [nodes[index]["row_group"] for index in ids]
        for activation in ACTIVATIONS:
            for dropout in DROPOUTS:
                torch.manual_seed(seed)
                model = initialize_arm(activation, dropout, states[seed])
                optimizer = e474.make_optimizer(model)
                before = [parameter.detach().clone() for parameter in model.parameters()]
                loss, telemetry = e474.fit_one_update(
                    model, optimizer, x, y, need, ids, torch.device("cpu")
                )
                changed = any(
                    not torch.equal(old, new)
                    for old, new in zip(before, model.parameters(), strict=True)
                )
                finite = all(
                    bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters()
                )
                if not changed or not finite:
                    raise ValueError(f"invalid CPU update for {seed}/{activation}/{dropout}")
                result.append(
                    {
                        "seed": seed,
                        "arm": arm_name(activation, dropout),
                        "activation": activation,
                        "head_dropout": dropout,
                        "fit_row_groups": groups,
                        "updates": 1,
                        "loss": loss,
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "parameters_changed": changed,
                        "parameters_finite": finite,
                        "telemetry_step0": telemetry,
                    }
                )
    return result


def run_fit(
    seed: int,
    activation: str,
    dropout: float,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    nodes: list[dict],
    device: torch.device,
    output: Path,
    initial_state: dict[str, torch.Tensor],
) -> dict:
    """Train one predeclared arm with E474 batching, loss and telemetry."""
    order = np.arange(24)
    np.random.default_rng(seed).shuffle(order)
    torch.manual_seed(seed)
    model = initialize_arm(activation, dropout, initial_state).to(device).float()
    optimizer = e474.make_optimizer(model)
    families = e474.parameter_families(model)
    telemetry = []
    step = 0
    started = time.monotonic()
    for offset in range(0, 24, 4):
        ids = order[offset : offset + 4].tolist()
        state = None
        for position in range(0, 20_000, CHUNK_ROWS):
            xx = torch.from_numpy(x[ids, position : position + CHUNK_ROWS]).to(
                device=device, dtype=torch.float32
            )
            yy = torch.from_numpy(y[ids, position : position + CHUNK_ROWS]).to(
                device=device, dtype=torch.float32
            )
            mm = torch.from_numpy(need[ids, position : position + CHUNK_ROWS]).to(device)
            if not (torch.isfinite(xx).all() and torch.isfinite(yy).all()):
                raise ValueError(f"nonfinite raw fit batch at {seed}/{activation}/{dropout}")
            optimizer.zero_grad(set_to_none=True)
            prediction, state = model(xx, state)
            if not torch.isfinite(prediction).all():
                raise ValueError(f"nonfinite prediction at {seed}/{activation}/{dropout}")
            loss = e470.loss_for_batch(prediction, yy, mm)
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite loss at {seed}/{activation}/{dropout}")
            loss.backward()
            stats = e474.apply_optimizer_update(
                model, optimizer, families, step, f"{seed}/{activation}/{dropout}"
            )
            if stats is not None:
                stats.update(
                    {
                        "fit_row_groups": [nodes[index]["row_group"] for index in ids],
                        "chunk_start": position,
                    }
                )
                telemetry.append(stats)
            state = state.detach()
            step += 1
    if step != UPDATES_PER_FIT or [x["update_index_zero_based"] for x in telemetry] != [
        0,
        46,
        236,
        473,
    ]:
        raise ValueError(f"unexpected update/telemetry counts for {seed}/{activation}/{dropout}")
    checkpoint = output / f"seed{seed}_{arm_name(activation, dropout)}.pt"
    torch.save(model.cpu().state_dict(), checkpoint)
    return {
        "seed": seed,
        "arm": arm_name(activation, dropout),
        "activation": activation,
        "head_dropout": dropout,
        "activation_formula": {
            "gelu": "GELU(x)",
            "channel_split": "concat(GELU(x[:32]), LeakyReLU_0.01(x[32:]))",
            "same_channel_mix": "0.5*GELU(x) + 0.5*LeakyReLU_0.01(x)",
        }[activation],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "updates": step,
        "fit_seconds": time.monotonic() - started,
        "fit_row_groups": [nodes[index]["row_group"] for index in order],
        "telemetry": telemetry,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": e474.file_hash(checkpoint),
    }


def main() -> None:
    """Run the source-bound CPU preflight or twelve fit-only models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-real", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)
    check_parent_sources()
    device = torch.device("cpu" if args.preflight_real else args.device)
    if device.type == "cuda":
        configure_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    states = make_initial_states()
    manifest, x, y, need, _, nodes = e470.load_cohort(include_test=False)
    if len(nodes) != 24 or any(node["role"] != "fit" for node in nodes):
        raise ValueError("E477 must load fit row groups only")
    if x.shape != (24, 20_000, 42) or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError(f"invalid fit-only cohort arrays: {x.shape}")
    result = {
        "kind": "E477 readout activation by dropout screen",
        "status": "preflight" if args.preflight_real else "fitting",
        "fit_only": True,
        "test_rows_loaded": False,
        "seeds": list(SEEDS),
        "activations": list(ACTIVATIONS),
        "dropout_probabilities": list(DROPOUTS),
        "arms": [],
        "planned_contrasts_within_seed": list(CONTRASTS),
        "architecture": "E474 RNNModel H64 2-layer GRU; vary only head activation and post-activation dropout",
        "parameters_per_arm": EXPECTED_PARAMETERS,
        "updates_per_fit": UPDATES_PER_FIT,
        "batch": 4,
        "chunk_rows": CHUNK_ROWS,
        "optimizer": {
            "name": "AdamW",
            "lr": 0.001,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "schedule": "constant",
            "clip_norm": 1.0,
        },
        "telemetry_sampling": {
            "step_indices_zero_based": [0, 46, 236, 473],
            "families": ["inp", "gru", "head"],
            "gradient": "preclip element-weighted RMS",
            "update": "parameter delta RMS",
            "descriptive_only": True,
        },
        "cohort_sha256": e474.file_hash(e470.COHORT_PATH),
        "dataset_sha256": DATASET_SHA256,
        "e470_source_sha256": E470_SOURCE_SHA256,
        "e474_source_sha256": E474_SOURCE_SHA256,
        "source_sha256": {
            name: e474.file_hash(HERE / name)
            for name in ("fit_activation_head_mixture.py", "tests.py", "gpu_probe.py")
            if (HERE / name).exists()
        },
    }
    (args.output / "fit_only_cohort.json").write_text(json.dumps(manifest, indent=2))
    if args.preflight_real:
        result["arms"] = run_cpu_preflight(x, y, need, nodes, states)
        result["status"] = "preflight_passed"
    else:
        with e474.full_ieee_float32():
            result["precision"] = e474.precision_metadata()
            for seed, activation, dropout in ARMS:
                result["arms"].append(
                    run_fit(
                        seed,
                        activation,
                        dropout,
                        x,
                        y,
                        need,
                        nodes,
                        device,
                        args.output,
                        states[seed],
                    )
                )
                (args.output / "fit_manifest.json").write_text(json.dumps(result, indent=2))
        if len(result["arms"]) != 12:
            raise ValueError("all twelve fit arms must finish before handoff")
        result["status"] = "all_12_fit_arms_complete_test_unread"
    (args.output / "fit_manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
