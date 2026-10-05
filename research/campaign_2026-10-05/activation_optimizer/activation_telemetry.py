"""Bounded fit-only activation telemetry for the completed E470 checkpoints.

This script performs inference only. It follows the trainer's fit-only
normalization and fixed first shuffled B4, carries GRU state through 256-row
chunks, and never opens diagnostic test row groups or updates parameters.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
RUN = HERE / "runs/sixteen_v2"
RESULT_PATH = RUN / "result.json"
OUTPUT_JSON = HERE / "ACTIVATION_TELEMETRY.json"
EDU = HERE / "eda"
SEED = 271828
CHUNK_ROWS = 256
MAX_DERIVATIVE_SAMPLES = 50_000
MAX_PLOT_SAMPLES = 2_000
EXPECTED_FIRST_GROUPS = [1284, 1793, 38, 1869]

# Permit direct script execution while reusing exactly the screened modules
# and fit-only cohort loader. Do not edit the fit implementation here.
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sequence_components"))
import fit_activation_optimizer as trainer  # noqa: E402


def activation_value(name: str, x: torch.Tensor) -> torch.Tensor:
    """Evaluate the same scalar activation formulas used in the fit module."""
    if name == "tanh":
        return torch.tanh(x)
    if name == "gelu":
        return torch.nn.functional.gelu(x)
    if name == "relu":
        return torch.relu(x)
    if name == "leaky_relu_001":
        return torch.nn.functional.leaky_relu(x, negative_slope=0.01)
    if name == "gelu_sine_01":
        return torch.nn.functional.gelu(x) + 0.1 * torch.sin(x)
    if name == "gelu_sine_cosine_01":
        return torch.nn.functional.gelu(x) + 0.1 * torch.sin(x) + 0.1 * (torch.cos(x) - 1.0)
    if name == "gelu_sinc_05":
        return torch.nn.functional.gelu(x) * (1.0 + 0.5 * torch.sinc(x))
    raise ValueError(f"not a scalar activation: {name}")


def scalar_curve(name: str, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a pointwise curve and its derivative for a bounded grid."""
    values = x.detach().clone().requires_grad_(True)
    output = activation_value(name, values)
    derivative = torch.autograd.grad(output.sum(), values)[0]
    return output.detach(), derivative.detach()


def split_curve(branch: str, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Plot the two channel-split functions as separate scalar branches."""
    values = x.detach().clone().requires_grad_(True)
    if branch == "split GELU channel":
        output = torch.nn.functional.gelu(values)
    elif branch == "split LeakyReLU channel":
        output = torch.nn.functional.leaky_relu(values, negative_slope=0.01)
    else:
        raise ValueError(branch)
    derivative = torch.autograd.grad(output.sum(), values)[0]
    return output.detach(), derivative.detach()


def save_activation_curves(path: Path) -> None:
    grid = torch.linspace(-6, 6, 1201, dtype=torch.float64)
    curves = [
        ("tanh", "tanh"),
        ("gelu", "GELU (PyTorch exact)"),
        ("relu", "ReLU"),
        ("leaky_relu_001", "LeakyReLU(0.01)"),
        ("gelu_sine_01", "GELU + 0.1 sin"),
        ("gelu_sine_cosine_01", "GELU + sine + centered cosine"),
        ("gelu_sinc_05", "GELU × (1 + 0.5 sinc)"),
    ]
    colors = plt.get_cmap("tab10").colors
    figure, (axis_value, axis_derivative) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for index, (name, label) in enumerate(curves):
        value, derivative = scalar_curve(name, grid)
        axis_value.plot(
            grid.numpy(), value.numpy(), label=label, color=colors[index], linewidth=1.8
        )
        axis_derivative.plot(
            grid.numpy(), derivative.numpy(), label=label, color=colors[index], linewidth=1.8
        )
    for index, branch in enumerate(("split GELU channel", "split LeakyReLU channel"), start=7):
        value, derivative = split_curve(branch, grid)
        axis_value.plot(
            grid.numpy(), value.numpy(), label=branch, color=colors[index], linewidth=1.8
        )
        axis_derivative.plot(
            grid.numpy(), derivative.numpy(), label=branch, color=colors[index], linewidth=1.8
        )
    axis_value.axhline(0, color="black", linewidth=0.7, alpha=0.5)
    axis_value.axvline(0, color="black", linewidth=0.7, alpha=0.5)
    axis_value.set_ylabel("Activation f(x)")
    axis_value.set_title("Pointwise activations used in the fixed screen")
    axis_value.grid(alpha=0.2)
    axis_value.legend(ncol=2, fontsize=8, frameon=False)
    axis_derivative.axhline(0, color="black", linewidth=0.7, alpha=0.5)
    axis_derivative.axvline(0, color="black", linewidth=0.7, alpha=0.5)
    axis_derivative.set_xlabel("Preactivation x")
    axis_derivative.set_ylabel("Derivative f′(x)")
    axis_derivative.grid(alpha=0.2)
    axis_derivative.legend(ncol=2, fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def stats(values: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if not flat.size or not np.isfinite(flat).all():
        raise ValueError("telemetry capture is empty or nonfinite")
    return {
        "count": int(flat.size),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "rms": float(np.sqrt(np.mean(flat * flat))),
        "fraction_abs_gt_3": float(np.mean(np.abs(flat) > 3)),
        "fraction_abs_gt_1": float(np.mean(np.abs(flat) > 1)),
        "fraction_zero": float(np.mean(flat == 0)),
    }


def sampled_indices(count: int, maximum: int) -> np.ndarray:
    size = min(count, maximum)
    return np.linspace(0, count - 1, size, dtype=np.int64)


def activation_derivative_sample(
    activation: str, preactivation: np.ndarray, maximum: int = MAX_DERIVATIVE_SAMPLES
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Sample actual preactivations and differentiate with the correct channel rule."""
    if preactivation.ndim != 3:
        raise ValueError(f"expected B×T×C activation input, got {preactivation.shape}")
    flat = preactivation.reshape(-1)
    indices = sampled_indices(flat.size, maximum)
    sampled = flat[indices].astype(np.float64, copy=False)
    if activation == "split_gelu_leaky_relu":
        channels = np.tile(
            np.arange(preactivation.shape[-1]), preactivation.shape[0] * preactivation.shape[1]
        )
        selected_channels = channels[indices]
        split = preactivation.shape[-1] // 2
        x = torch.tensor(sampled, dtype=torch.float64, requires_grad=True)
        gelu_branch = torch.nn.functional.gelu(x)
        leaky_branch = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        output = torch.where(torch.from_numpy(selected_channels < split), gelu_branch, leaky_branch)
    else:
        x = torch.tensor(sampled, dtype=torch.float64, requires_grad=True)
        output = activation_value(activation, x)
    derivative = torch.autograd.grad(output.sum(), x)[0].detach().numpy()
    summary = {
        "sample_count": int(len(derivative)),
        "min": float(np.min(derivative)),
        "max": float(np.max(derivative)),
        "mean": float(np.mean(derivative)),
        "rms": float(np.sqrt(np.mean(derivative * derivative))),
        "quantiles": {
            str(q): float(np.quantile(derivative, q / 100)) for q in (1, 5, 25, 50, 75, 95, 99)
        },
    }
    return sampled.astype(np.float32, copy=False), summary


def load_model(checkpoint: Path, activation: str) -> trainer.ActivationGRU:
    model = trainer.ActivationGRU(activation)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def capture_site(
    model: trainer.ActivationGRU,
    module: nn.Module,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run full 20k streams in chunks while retaining only the two hooked sites."""
    chunks: list[tuple[np.ndarray, np.ndarray]] = []

    def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        chunks.append((inputs[0].detach().cpu().numpy(), output.detach().cpu().numpy()))

    handle = module.register_forward_hook(hook)
    state = None
    try:
        with torch.inference_mode():
            for start in range(0, x.shape[1], CHUNK_ROWS):
                chunk = torch.from_numpy(x[:, start : start + CHUNK_ROWS])
                _, state = model(chunk, state)
    finally:
        handle.remove()
    pre = np.concatenate([item[0] for item in chunks], axis=1)
    post = np.concatenate([item[1] for item in chunks], axis=1)
    return pre, post


def save_fit_preactivation_summary(
    samples: dict[tuple[str, str, str], np.ndarray], path: Path
) -> None:
    activation_names = trainer.ACTIVATIONS
    colors = {"adamw": "#2864a5", "muon_adamw": "#d17a00"}
    figure, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    for axis, site in zip(axes, ("embedding", "correction_head"), strict=True):
        data, positions, facecolors = [], [], []
        for index, activation in enumerate(activation_names):
            for offset, optimizer in ((-0.18, "adamw"), (0.18, "muon_adamw")):
                key = (activation, optimizer, site)
                data.append(samples[key])
                positions.append(index + offset)
                facecolors.append(colors[optimizer])
        boxes = axis.boxplot(
            data,
            positions=positions,
            widths=0.30,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            manage_ticks=False,
        )
        for box, color in zip(boxes["boxes"], facecolors, strict=True):
            box.set_facecolor(color)
            box.set_alpha(0.55)
        axis.axhline(0, color="black", linewidth=0.7, alpha=0.5)
        axis.set_ylabel("Activation preinput")
        axis.set_title(
            "Embedding linear output" if site == "embedding" else "Correction hidden linear output"
        )
        axis.set_ylim(-6, 6)
        axis.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(np.arange(len(activation_names)))
    axes[-1].set_xticklabels(activation_names, rotation=35, ha="right")
    axes[0].plot([], [], color=colors["adamw"], linewidth=6, alpha=0.55, label="AdamW")
    axes[0].plot([], [], color=colors["muon_adamw"], linewidth=6, alpha=0.55, label="MuonAdamW")
    axes[0].legend(frameon=False, ncol=2)
    figure.suptitle("Fit-only preactivation distributions on four fixed sequences")
    figure.text(
        0.5,
        0.015,
        "Boxes show sampled interquartile ranges; whiskers show the 5th–95th percentiles. Full ranges, RMS, and |x|>3 fractions are in the JSON.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    torch.set_num_threads(1)
    if OUTPUT_JSON.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT_JSON}")
    result = json.loads(RESULT_PATH.read_text())
    if result.get("seed") != SEED or len(result.get("arms", [])) != 16:
        raise ValueError("expected the completed 16-arm seed-271828 result")

    manifest, x, y, need, score, nodes = trainer.load_cohort(include_test=False)
    del y, need, score
    fit_ids = [index for index, node in enumerate(nodes) if node["role"] == "fit"]
    np.random.default_rng(SEED).shuffle(fit_ids)
    fit_ids = fit_ids[:4]
    row_groups = [nodes[index]["row_group"] for index in fit_ids]
    if row_groups != EXPECTED_FIRST_GROUPS:
        raise ValueError(
            f"fit cohort/order drift: expected {EXPECTED_FIRST_GROUPS}, got {row_groups}"
        )
    fit_x = x[fit_ids]
    del x

    telemetry = []
    plot_samples: dict[tuple[str, str, str], np.ndarray] = {}
    checkpoint_hashes = {}
    result_arms = {arm["arm"]: arm for arm in result["arms"]}
    for arm in trainer.ARMS:
        activation, optimizer = arm.split("__")
        checkpoint = RUN / f"{arm}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_hashes[arm] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        model = load_model(checkpoint, activation)
        per_site = {}
        for site, module in (
            ("embedding", model.embedding_activation),
            ("correction_head", model.correction_activation),
        ):
            pre, post = capture_site(model, module, fit_x)
            sampled, derivative = activation_derivative_sample(activation, pre)
            per_site[site] = {
                "shape": [int(value) for value in pre.shape],
                "preactivation": stats(pre),
                "activation_output": stats(post),
                "derivative_on_sampled_fit_preactivations": derivative,
            }
            plot_samples[(activation, optimizer, site)] = sampled_indices_copy(
                pre, MAX_PLOT_SAMPLES
            )
            del pre, post, sampled
        telemetry.append(
            {
                "arm": arm,
                "activation": activation,
                "optimizer": optimizer,
                "wp_from_completed_fit": result_arms[arm]["wp"],
                "checkpoint_sha256": checkpoint_hashes[arm],
                "sites": per_site,
            }
        )
        del model

    EDU.mkdir(parents=True, exist_ok=True)
    save_activation_curves(EDU / "activation_curves.png")
    save_fit_preactivation_summary(plot_samples, EDU / "fit_pre_activation_summary.png")
    output = {
        "title": "Fit-only activation telemetry for completed E470 checkpoints",
        "scope": {
            "training": "none; inference-only checkpoint telemetry",
            "training_seed": SEED,
            "optimizer_arms": list(trainer.OPTIMIZERS),
            "activations": list(trainer.ACTIVATIONS),
            "fit_sequences": 4,
            "row_groups": row_groups,
            "rows_per_sequence": 20_000,
            "chunk_rows": CHUNK_ROWS,
            "chunks_per_sequence": 79,
            "state": "GRU state carried across all chunks; reset between row groups",
            "cohort_role_accessed": "fit only; no test row groups opened",
            "normalization": manifest["normalization"],
            "derivative_sample_cap_per_site_arm": MAX_DERIVATIVE_SAMPLES,
            "derivative_sample_method": "deterministic evenly spaced indices over captured fit preactivations",
            "plot_sample_cap_per_site_arm": MAX_PLOT_SAMPLES,
            "activation_inputs_captured": "forward hooks on embedding_activation and correction_activation",
            "split_semantics": "derivatives use GELU for the first floor(C/2) channels and LeakyReLU(0.01) for remaining channels, matching the implemented concatenation",
        },
        "telemetry_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cohort_sha256": hashlib.sha256(trainer.COHORT_PATH.read_bytes()).hexdigest(),
        "fit_checkpoint_result_sha256": hashlib.sha256(RESULT_PATH.read_bytes()).hexdigest(),
        "checkpoint_sha256": checkpoint_hashes,
        "arms": telemetry,
        "interpretation_limit": "Activation occupancy and derivatives are descriptive for four fit sequences and saved one-seed checkpoints. They do not establish a causal mechanism, market periodicity, or performance on independent components.",
        "artifacts": [
            "research/campaign_2026-10-05/activation_optimizer/eda/activation_curves.png",
            "research/campaign_2026-10-05/activation_optimizer/eda/fit_pre_activation_summary.png",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps({"output": str(OUTPUT_JSON), "arms": len(telemetry), "row_groups": row_groups})
    )


def sampled_indices_copy(values: np.ndarray, maximum: int) -> np.ndarray:
    flat = values.reshape(-1)
    return flat[sampled_indices(flat.size, maximum)].astype(np.float32, copy=True)


if __name__ == "__main__":
    main()
