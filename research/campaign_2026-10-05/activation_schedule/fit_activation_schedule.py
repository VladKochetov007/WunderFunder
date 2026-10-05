"""E476 activation x optimizer x LR-protocol screen, source-only until root admission.

The model, parameter layout, exact GELU/sine formulas, fit-only data helpers,
objective, and 24-group cohort are inherited from E474/E470. This file adds
only the optimizer and deterministic per-update LR protocol factor.
"""

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
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E474_DIR))
sys.path.insert(0, str(E470_DIR))

import fit_activation_optimizer as e470  # noqa: E402
import fit_activation_placement as e474  # noqa: E402

from wnn.optim import MuonAdamW, muon_param_groups  # noqa: E402
from wnn.resources import configure_torch  # noqa: E402

SEED = 271828
UPDATES = 474
WARMUP_UPDATES = 47
COSINE_UPDATES = UPDATES - WARMUP_UPDATES
BASE_LR = 0.0001
PEAK_LR = 0.001
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
GRADIENT_CLIP = 1.0
SAMPLE_STEPS = (0, 46, 236, 473)
ACTIVATIONS = ("base", "sine_embed")
OPTIMIZERS = ("adamw", "muon_adamw")
SCHEDULES = ("constant_peak", "warmcos_47_427", "constant_mean_warmcos")
EXPECTED_PARAMETERS = 61_058
EXPECTED_FEATURES = 42
EXPECTED_GROUP_SHAPE = (24, 20_000, EXPECTED_FEATURES)
EXPECTED_DATASET_SHA256 = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"


def file_hash(path: Path) -> str:
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def e474_lr_trace() -> np.ndarray:
    """Return E474's exact source-defined per-update trace (constant 1e-3)."""
    # E474's make_optimizer sets lr=0.001, and run_fit never changes it.
    return np.full(UPDATES, e474.ADAMW_LR, dtype=np.float64)


def make_lr_traces() -> dict[str, np.ndarray]:
    """Build deterministic length-474 traces before any model update."""
    warm = np.linspace(BASE_LR, PEAK_LR, WARMUP_UPDATES, dtype=np.float64)
    phase = np.arange(COSINE_UPDATES, dtype=np.float64) / (COSINE_UPDATES - 1)
    cosine = BASE_LR + (PEAK_LR - BASE_LR) * 0.5 * (1.0 + np.cos(np.pi * phase))
    warmcos = np.concatenate((warm, cosine))
    mean_lr = math.fsum(warmcos.tolist()) / UPDATES
    traces = {
        "constant_peak": np.full(UPDATES, PEAK_LR, dtype=np.float64),
        "warmcos_47_427": warmcos,
        "constant_mean_warmcos": np.full(UPDATES, mean_lr, dtype=np.float64),
    }
    if any(trace.shape != (UPDATES,) or not np.isfinite(trace).all() for trace in traces.values()):
        raise ValueError("each LR protocol must be a finite 474-update trace")
    if not (
        traces["warmcos_47_427"][0] == BASE_LR
        and traces["warmcos_47_427"][46] == PEAK_LR
        and traces["warmcos_47_427"][47] == PEAK_LR
        and traces["warmcos_47_427"][-1] == BASE_LR
    ):
        raise ValueError("warm/cosine endpoints do not match the preregistered boundaries")
    if not np.allclose(
        traces["warmcos_47_427"].sum(), traces["constant_mean_warmcos"].sum(), rtol=0, atol=1e-15
    ):
        raise ValueError(
            "constant-mean protocol does not match the exact warm/cosine arithmetic area"
        )
    return traces


def model_for(activation: str, initial_state: dict[str, torch.Tensor]) -> e474.PlacementRNN:
    """Instantiate the requested E474 activation with common initialization."""
    if activation == "base":
        embedding = head = "gelu"
    elif activation == "sine_embed":
        embedding, head = "sine", "gelu"
    else:
        raise ValueError(f"unknown activation: {activation}")
    model = e474.initialize_arm(SEED, embedding, head, initial_state)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EXPECTED_PARAMETERS:
        raise ValueError(f"expected {EXPECTED_PARAMETERS} trainable parameters, got {count}")
    return model


def make_optimizer(name: str, model: e474.PlacementRNN) -> torch.optim.Optimizer:
    """Build AdamW or recurrent-only MuonAdamW with 3 GRU gate blocks."""
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY, betas=BETAS
        )
    if name == "muon_adamw":
        groups = muon_param_groups(model, scope="rnn")
        recurrent = next(group for group in groups if group.get("muon", False))
        matrices = [parameter for parameter in recurrent["params"]]
        expected = {"gru.weight_ih_l0", "gru.weight_hh_l0", "gru.weight_ih_l1", "gru.weight_hh_l1"}
        names = {
            name
            for name, parameter in model.named_parameters()
            if any(parameter is p for p in matrices)
        }
        if names != expected or recurrent.get("blocks") != 3:
            raise ValueError(
                f"Muon must cover only the four 3-gate GRU matrices; got {sorted(names)}"
            )
        return MuonAdamW(groups, lr=PEAK_LR, weight_decay=WEIGHT_DECAY, betas=BETAS)
    raise ValueError(f"unknown optimizer: {name}")


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Apply one scalar LR to every optimizer parameter group."""
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def parameter_family(name: str) -> str:
    """Map named trainable parameters to a stable telemetry family."""
    if name.startswith("inp."):
        return "input_projection"
    if name.startswith("gru."):
        return "recurrent_core"
    if name.startswith("head.0."):
        return "hidden_head"
    if name.startswith("head.2."):
        return "output_head"
    raise ValueError(f"unclassified parameter family: {name}")


def shuffled_fit_order(nodes: list[dict]) -> list[int]:
    """Return the exact fixed-seed group traversal shared by every factorial cell."""
    if len(nodes) != 24 or any(node["role"] != "fit" for node in nodes):
        raise ValueError("expected exactly 24 fit-role row groups")
    order = list(range(len(nodes)))
    np.random.default_rng(SEED).shuffle(order)
    return order


def tensor_rms(values: list[torch.Tensor]) -> float:
    count = sum(value.numel() for value in values)
    if count == 0:
        return 0.0
    squared = sum(float(value.detach().double().square().sum().item()) for value in values)
    return math.sqrt(squared / count)


def reset_cuda_peak(device: torch.device) -> None:
    """Start a per-cell peak-memory window when running on CUDA."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def cuda_peak_bytes(device: torch.device) -> dict[str, int | None]:
    """Return per-cell peak allocated/reserved CUDA bytes, or null on CPU."""
    if device.type != "cuda":
        return {
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
        }
    return {
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def update_once(
    model: e474.PlacementRNN,
    optimizer: torch.optim.Optimizer,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    ids: list[int],
    device: torch.device,
    lr: float,
    step: int,
    telemetry: bool,
) -> tuple[float, float | None, dict[str, float] | None]:
    """One fit-only B4x256 update; telemetry uses preclip gradient and delta RMS."""
    xx = torch.from_numpy(x[ids, : e474.CHUNK_ROWS]).to(device=device, dtype=torch.float32)
    yy = torch.from_numpy(y[ids, : e474.CHUNK_ROWS]).to(device=device, dtype=torch.float32)
    mm = torch.from_numpy(need[ids, : e474.CHUNK_ROWS]).to(device)
    if not (torch.isfinite(xx).all() and torch.isfinite(yy).all()):
        raise ValueError("nonfinite fit-only probe batch")
    set_lr(optimizer, lr)
    optimizer.zero_grad(set_to_none=True)
    prediction, _ = model(xx)
    if not torch.isfinite(prediction).all():
        raise ValueError("nonfinite model prediction")
    loss = e470.loss_for_batch(prediction, yy, mm)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite fit loss")
    loss.backward()
    preclip_norm = float(
        torch.linalg.vector_norm(
            torch.stack(
                [
                    torch.linalg.vector_norm(parameter.grad.detach(), 2)
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
            ),
            2,
        ).item()
    )
    if not math.isfinite(preclip_norm):
        raise ValueError("nonfinite global gradient norm")
    nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP, error_if_nonfinite=True)
    before = None
    if telemetry and step in SAMPLE_STEPS:
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer.step()
    e470.ensure_finite_parameters(model, f"step={step}")
    if before is None:
        return float(loss.detach().cpu()), None, None
    deltas: dict[str, list[torch.Tensor]] = {}
    for name, parameter in model.named_parameters():
        family = parameter_family(name)
        deltas.setdefault(family, []).append(parameter.detach() - before[name])
    return (
        float(loss.detach().cpu()),
        preclip_norm,
        {family: tensor_rms(values) for family, values in sorted(deltas.items())},
    )


def one_update_matrix(
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    nodes: list[dict],
    initial_state: dict[str, torch.Tensor],
    device: torch.device,
    traces: dict[str, np.ndarray],
) -> list[dict]:
    """Fit-only one-update preflight for every planned factorial cell."""
    if x.shape != EXPECTED_GROUP_SHAPE:
        raise ValueError(f"expected exact fit feature shape {EXPECTED_GROUP_SHAPE}; got {x.shape}")
    if y.shape != (24, 20_000, 2) or need.shape != (24, 20_000):
        raise ValueError("fit labels or need_prediction mask have an unexpected shape")
    if len(nodes) != 24 or {node["role"] for node in nodes} != {"fit"}:
        raise ValueError("preflight must load exactly the 24 fit row groups")
    order = shuffled_fit_order(nodes)
    ids = order[:4]
    row_groups = [nodes[index]["row_group"] for index in ids]
    results = []
    for activation in ACTIVATIONS:
        for optimizer_name in OPTIMIZERS:
            for schedule in SCHEDULES:
                model = model_for(activation, initial_state).to(device).float()
                optimizer = make_optimizer(optimizer_name, model)
                parameters_before = [p.detach().clone() for p in model.parameters()]
                reset_cuda_peak(device)
                loss, grad_norm, update_rms = update_once(
                    model,
                    optimizer,
                    x,
                    y,
                    need,
                    ids,
                    device,
                    float(traces[schedule][0]),
                    0,
                    telemetry=True,
                )
                changed = [
                    not torch.equal(before, after)
                    for before, after in zip(parameters_before, model.parameters(), strict=True)
                ]
                if not any(changed):
                    raise ValueError(
                        f"fit-only preflight update changed no parameters: "
                        f"{activation}/{optimizer_name}/{schedule}"
                    )
                results.append(
                    {
                        "activation": activation,
                        "optimizer": optimizer_name,
                        "schedule": schedule,
                        "fit_row_groups": row_groups,
                        "optimizer_updates": 1,
                        "loss": loss,
                        "gradient_norm_preclip": grad_norm,
                        "parameter_update_rms": update_rms,
                        "parameters": sum(parameter.numel() for parameter in model.parameters()),
                        "parameters_changed": True,
                        "changed_parameter_tensors": sum(changed),
                        "device": str(device),
                        **cuda_peak_bytes(device),
                    }
                )
    if len(results) != 12:
        raise AssertionError(f"expected 12 factorial cells, got {len(results)}")
    return results


def fit_cell(
    activation: str,
    optimizer_name: str,
    schedule: str,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    nodes: list[dict],
    initial_state: dict[str, torch.Tensor],
    device: torch.device,
    traces: dict[str, np.ndarray],
    output: Path,
) -> dict:
    """Run one exact 474-update fit arm over only the preregistered fit cohort."""
    order = shuffled_fit_order(nodes)
    model = model_for(activation, initial_state).to(device).float()
    optimizer = make_optimizer(optimizer_name, model)
    initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    reset_cuda_peak(device)
    telemetry = []
    step, started = 0, time.monotonic()
    for batch_start in range(0, len(order), 4):
        ids = order[batch_start : batch_start + 4]
        state = None
        for position in range(0, x.shape[1], e474.CHUNK_ROWS):
            xx = torch.from_numpy(x[ids, position : position + e474.CHUNK_ROWS]).to(
                device=device, dtype=torch.float32
            )
            yy = torch.from_numpy(y[ids, position : position + e474.CHUNK_ROWS]).to(
                device=device, dtype=torch.float32
            )
            mm = torch.from_numpy(need[ids, position : position + e474.CHUNK_ROWS]).to(device)
            if not (torch.isfinite(xx).all() and torch.isfinite(yy).all()):
                raise ValueError(f"nonfinite raw fit batch at update {step}")
            lr = float(traces[schedule][step])
            optimizer.zero_grad(set_to_none=True)
            prediction, state = model(xx, state)
            if not torch.isfinite(prediction).all():
                raise ValueError(f"nonfinite prediction at update {step}")
            loss = e470.loss_for_batch(prediction, yy, mm)
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite fit loss at update {step}")
            loss.backward()
            preclip_norm = float(
                torch.linalg.vector_norm(
                    torch.stack(
                        [
                            torch.linalg.vector_norm(parameter.grad.detach(), 2)
                            for parameter in model.parameters()
                            if parameter.grad is not None
                        ]
                    ),
                    2,
                ).item()
            )
            if not math.isfinite(preclip_norm):
                raise ValueError(f"nonfinite gradient norm at update {step}")
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP, error_if_nonfinite=True)
            before = None
            if step in SAMPLE_STEPS:
                before = {
                    name: parameter.detach().clone() for name, parameter in model.named_parameters()
                }
            set_lr(optimizer, lr)
            optimizer.step()
            e470.ensure_finite_parameters(model, f"update={step}")
            if before is not None:
                delta_groups: dict[str, list[torch.Tensor]] = {}
                for name, parameter in model.named_parameters():
                    delta_groups.setdefault(parameter_family(name), []).append(
                        parameter.detach() - before[name]
                    )
                telemetry.append(
                    {
                        "step_zero_based": step,
                        "lr": lr,
                        "gradient_norm_preclip": preclip_norm,
                        "parameter_update_rms": {
                            family: tensor_rms(values)
                            for family, values in sorted(delta_groups.items())
                        },
                        "fit_loss": float(loss.detach().cpu()),
                    }
                )
            state = state.detach()
            step += 1
    if step != UPDATES or [entry["step_zero_based"] for entry in telemetry] != list(SAMPLE_STEPS):
        raise ValueError(f"wrong updates/telemetry: updates={step}, samples={telemetry}")
    changed = [
        not torch.equal(before, after)
        for before, after in zip(initial_parameters, model.parameters(), strict=True)
    ]
    if not any(changed):
        raise ValueError(
            f"all fit parameters were unchanged: {activation}/{optimizer_name}/{schedule}"
        )
    elapsed = time.monotonic() - started
    checkpoint = output / f"{activation}__{optimizer_name}__{schedule}.pt"
    torch.save(model.cpu().state_dict(), checkpoint)
    return {
        "activation": activation,
        "optimizer": optimizer_name,
        "schedule": schedule,
        "updates": step,
        "fit_seconds": elapsed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_changed": True,
        "changed_parameter_tensors": sum(changed),
        **cuda_peak_bytes(device),
        "fit_row_groups": [nodes[index]["row_group"] for index in order],
        "lr_trace_sha256": hashlib.sha256(traces[schedule].astype("<f8").tobytes()).hexdigest(),
        "telemetry": telemetry,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": file_hash(checkpoint),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preflight-only", action="store_true", help="one fit-only update per factorial cell"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output}")
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda":
        configure_torch()
    traces = make_lr_traces()
    trace_receipt = {
        "updates": UPDATES,
        "reference": "E474 make_optimizer lr=0.001, no scheduler; exact source trace is constant",
        "e474_source_reference_trace": e474_lr_trace().tolist(),
        "e474_source_reference_trace_sha256": hashlib.sha256(
            e474_lr_trace().astype("<f8").tobytes()
        ).hexdigest(),
        "protocols": {
            name: {
                "values": trace.tolist(),
                "mean": float(trace.mean()),
                "sum": float(trace.sum()),
                "sha256": hashlib.sha256(trace.astype("<f8").tobytes()).hexdigest(),
                "same_as_constant_peak": bool(np.array_equal(trace, traces["constant_peak"])),
            }
            for name, trace in traces.items()
        },
        "warmcos_definition": "0..46 linear inclusive 1e-4 to 1e-3; 47..473 cosine inclusive 1e-3 to 1e-4",
        "interpretation": "constant_mean_warmcos is the arithmetic mean of the exact frozen 474-step warmcos_47_427 trace; warmcos versus this arm isolates schedule shape at equal arithmetic LR area, but not equal optimizer-update area.",
    }
    (args.output / "lr_traces.json").write_text(json.dumps(trace_receipt, indent=2) + "\n")
    _manifest, x, y, need, _score, nodes = e470.load_cohort(include_test=False)
    if len(nodes) != 24 or any(node["role"] != "fit" for node in nodes):
        raise ValueError("E476 must load exactly the 24 E470 fit row groups")
    if x.shape != EXPECTED_GROUP_SHAPE:
        raise ValueError(f"unexpected E470 feature shape: {x.shape}")
    if file_hash(e470.DATA_PATH) != EXPECTED_DATASET_SHA256:
        raise ValueError("the frozen E470 dataset hash changed")
    initial_state = e474.make_seed_states()[SEED]
    result = {
        "experiment": "E476 activation x optimizer x learning-rate protocol factorial",
        "status": "preflight_only" if args.preflight_only else "fitting",
        "fit_only": True,
        "exposed_test_groups_loaded": False,
        "cohort_rows": 24,
        "cohort_sha256": file_hash(e470.COHORT_PATH),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "seed": SEED,
        "activations": list(ACTIVATIONS),
        "optimizers": list(OPTIMIZERS),
        "schedules": list(SCHEDULES),
        "parameters_per_cell": EXPECTED_PARAMETERS,
        "batch": 4,
        "chunk_rows": 256,
        "updates_per_fit": UPDATES,
        "weight_decay": WEIGHT_DECAY,
        "betas": list(BETAS),
        "gradient_clip": GRADIENT_CLIP,
        "sampling_steps_zero_based": list(SAMPLE_STEPS),
        "optimizer_details": {
            "adamw": "all parameters; betas=(0.9,0.999), wd=0.01",
            "muon_adamw": "Muon only on gru.weight_ih_l0, weight_hh_l0, weight_ih_l1, weight_hh_l1; 3 blocks each; AdamW on input/head/biases",
        },
        "lr_trace_receipt": "lr_traces.json",
        "source_sha256": {
            str(path.relative_to(ROOT)): file_hash(path)
            for path in (
                HERE / "fit_activation_schedule.py",
                HERE / "tests.py",
                HERE / "PREREG.md",
                E474_DIR / "fit_activation_placement.py",
                E470_DIR / "fit_activation_optimizer.py",
                ROOT / "src/wnn/optim.py",
                ROOT / "src/wnn/nn.py",
            )
        },
        "source_admission": "draft only; requires independent source review, root-owned grading and actual-data CUDA probe before any full fits",
    }
    if args.preflight_only:
        result["device"] = str(device)
        with e474.full_ieee_float32():
            result["precision"] = e474.precision_metadata()
            result["preflight_cells"] = one_update_matrix(
                x, y, need, nodes, initial_state, device, traces
            )
        result["status"] = "preflight_passed"
    else:
        with e474.full_ieee_float32():
            result["precision"] = e474.precision_metadata()
            result["completed_cells"] = []
            for activation in ACTIVATIONS:
                for optimizer_name in OPTIMIZERS:
                    for schedule in SCHEDULES:
                        cell = fit_cell(
                            activation,
                            optimizer_name,
                            schedule,
                            x,
                            y,
                            need,
                            nodes,
                            initial_state,
                            device,
                            traces,
                            args.output,
                        )
                        result["completed_cells"].append(cell)
                        (args.output / "fit_manifest.json").write_text(
                            json.dumps(result, indent=2) + "\n"
                        )
        if len(result["completed_cells"]) != 12:
            raise ValueError("all 12 preregistered cells must finish before handoff")
        result["status"] = "all_12_complete_test_unread"
    (args.output / "fit_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
