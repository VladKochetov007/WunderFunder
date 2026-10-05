"""E479 bounded learnable GRU candidate-shape study (fit rows only)."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E470_DIR))
sys.path.insert(0, str(E474_DIR))

import fit_activation_optimizer as e470  # noqa: E402
import fit_activation_placement as e474  # noqa: E402

from wnn.optim import MuonAdamW, muon_param_groups  # noqa: E402
from wnn.resources import configure_torch  # noqa: E402

SEEDS = (271828, 271829)
OPTIMIZERS = ("adamw", "muon_adamw")
CONDITIONS = ("fixed0", "fixed01", "learned0", "learned01")
ARMS = tuple(
    (seed, optimizer, condition)
    for seed in SEEDS
    for optimizer in OPTIMIZERS
    for condition in CONDITIONS
)
FEATURES = 42
HIDDEN = 64
CHUNK_ROWS = 256
UPDATES_PER_FIT = 474
FIXED_TELEMETRY_STEPS = (0, 46, 236, 473)
EXPECTED_FIXED_PARAMETERS = 61_058
EXPECTED_LEARNED_PARAMETERS = 61_060
MUON_NAMES = {"gru.weight_ih_l0", "gru.weight_hh_l0", "gru.weight_ih_l1", "gru.weight_hh_l1"}
ALPHA_MAX = 0.2
ALPHA_INIT_POSITIVE_RAW = math.atanh(0.5)
CLIP_NORM = 1.0
LR = 0.001
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
DATASET_SHA256 = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
E470_SOURCE_SHA256 = "871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9"
E474_SOURCE_SHA256 = "40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068"

CONTRASTS = (
    "learned0-fixed0 within each seed and optimizer",
    "learned01-fixed01 within each seed and optimizer",
    "fixed01-fixed0 within each seed and optimizer",
    "learned01-learned0 within each seed and optimizer",
    "MuonAdamW-AdamW within each seed and candidate condition",
    "(learned0-fixed0)_Muon-(learned0-fixed0)_AdamW within each seed",
    "(learned01-fixed01)_Muon-(learned01-fixed01)_AdamW within each seed",
)


@torch.jit.script
def candidate_from_tanh_tensor(u: Tensor, alpha: Tensor) -> Tensor:
    """Apply bounded candidate warp while preserving gradients to scalar alpha."""
    return u + alpha * torch.sin(math.pi * u)


@torch.jit.script
def scripted_adaptive_scan(
    x: Tensor,
    h0: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Tensor,
    bias_hh: Tensor,
    alpha: Tensor,
) -> tuple[Tensor, Tensor]:
    """Scan one GRU layer using PyTorch's reset-after candidate bias placement."""
    input_gates = torch.nn.functional.linear(x, weight_ih, bias_ih)
    outputs = torch.jit.annotate(list[Tensor], [])
    hidden = h0
    for step in range(x.size(1)):
        input_reset, input_update, input_candidate = input_gates[:, step, :].chunk(3, dim=1)
        hidden_gates = torch.nn.functional.linear(hidden, weight_hh, bias_hh)
        hidden_reset, hidden_update, hidden_candidate = hidden_gates.chunk(3, dim=1)
        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        u = torch.tanh(input_candidate + reset * hidden_candidate)
        candidate = candidate_from_tanh_tensor(u, alpha)
        hidden = (1.0 - update) * candidate + update * hidden
        outputs.append(hidden)
    return torch.stack(outputs, dim=1), hidden


class AdaptiveCandidateRNN(e474.PlacementRNN):
    """E474 model with fixed or layer-specific differentiable bounded alpha."""

    def __init__(self, condition: str) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"unknown E479 condition: {condition}")
        super().__init__(embedding_activation="gelu", head_activation="gelu")
        self.condition = condition
        if condition.startswith("learned"):
            initial = 0.0 if condition == "learned0" else ALPHA_INIT_POSITIVE_RAW
            self.raw_alpha = nn.Parameter(torch.full((2,), initial, dtype=torch.float32))
            self.register_buffer("fixed_alpha", torch.zeros(2), persistent=False)
        else:
            fixed = 0.0 if condition == "fixed0" else 0.1
            self.register_parameter("raw_alpha", None)
            self.register_buffer("fixed_alpha", torch.full((2,), fixed), persistent=False)

    def effective_alpha(self) -> Tensor:
        """Return live bounded coefficients, with no scalar conversion or detach."""
        if self.raw_alpha is None:
            return self.fixed_alpha
        return ALPHA_MAX * torch.tanh(self.raw_alpha)

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != FEATURES:
            raise ValueError("x must have shape (batch, time, 42)")
        encoded = self.encode((x - self.mean) / self.std)
        batch = encoded.shape[0]
        if state is None:
            hidden_state = encoded.new_zeros((2, batch, HIDDEN))
        else:
            if state.shape != (2, batch, HIDDEN):
                raise ValueError("GRU state must have shape (2, batch, 64)")
            hidden_state = state
        alpha = self.effective_alpha()
        layer_input = encoded
        final_states: list[Tensor] = []
        for layer in range(2):
            memory, final_hidden = scripted_adaptive_scan(
                layer_input,
                hidden_state[layer],
                getattr(self.gru, f"weight_ih_l{layer}"),
                getattr(self.gru, f"weight_hh_l{layer}"),
                getattr(self.gru, f"bias_ih_l{layer}"),
                getattr(self.gru, f"bias_hh_l{layer}"),
                alpha[layer],
            )
            layer_input = memory
            final_states.append(final_hidden)
        recurrent_state = torch.stack(final_states, dim=0)
        prediction = self.head(torch.cat((layer_input, encoded), dim=-1))
        return prediction, recurrent_state


def seed_states() -> dict[int, dict[str, Tensor]]:
    """Build original E474 initialization bytes before attaching alpha scalars."""
    return e474.make_seed_states()


def check_frozen_sources() -> dict[str, str]:
    """Fail closed if the inherited E470 recipe or E474 topology changed."""
    paths = {
        "e470": (E470_DIR / "fit_activation_optimizer.py", E470_SOURCE_SHA256),
        "e474": (E474_DIR / "fit_activation_placement.py", E474_SOURCE_SHA256),
    }
    actual = {name: e474.file_hash(path) for name, (path, _expected) in paths.items()}
    for name, (_path, expected) in paths.items():
        if actual[name] != expected:
            raise ValueError(f"frozen source changed: {name}={actual[name]}")
    return actual


def initialize_arm(condition: str, base_state: dict[str, Tensor]) -> AdaptiveCandidateRNN:
    """Load all baseline tensors identically, then retain planned alpha init."""
    model = AdaptiveCandidateRNN(condition)
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    allowed_missing = {"raw_alpha"} if condition.startswith("learned") else set()
    if set(missing) != allowed_missing or unexpected:
        raise ValueError(f"baseline transfer mismatch: missing={missing}, unexpected={unexpected}")
    actual = sum(parameter.numel() for parameter in model.parameters())
    expected = (
        EXPECTED_LEARNED_PARAMETERS
        if condition.startswith("learned")
        else EXPECTED_FIXED_PARAMETERS
    )
    if actual != expected:
        raise ValueError(f"parameter-count mismatch for {condition}: {actual} != {expected}")
    return model


def make_optimizer(model: AdaptiveCandidateRNN, name: str) -> torch.optim.Optimizer:
    """Use exact incumbent optimizer partitions and a no-decay learned-alpha group."""
    alpha_params = [] if model.raw_alpha is None else [model.raw_alpha]
    alpha_ids = {id(parameter) for parameter in alpha_params}
    if name == "adamw":
        ordinary = [parameter for parameter in model.parameters() if id(parameter) not in alpha_ids]
        groups = [{"params": ordinary, "weight_decay": WEIGHT_DECAY}]
        if alpha_params:
            groups.append({"params": alpha_params, "weight_decay": 0.0})
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            groups, lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY
        )
    elif name == "muon_adamw":
        groups = muon_param_groups(model, scope="rnn")
        for group in groups:
            if not group.get("muon"):
                group["params"] = [
                    parameter for parameter in group["params"] if id(parameter) not in alpha_ids
                ]
        groups = [group for group in groups if group.get("muon") or group["params"]]
        if alpha_params:
            groups.append({"params": alpha_params, "weight_decay": 0.0})
        optimizer = MuonAdamW(groups, lr=LR, weight_decay=WEIGHT_DECAY, betas=BETAS)
        selected = {
            parameter_name
            for parameter_name, parameter in model.named_parameters()
            if any(
                id(parameter) == id(candidate)
                for group in optimizer.param_groups
                if group.get("muon")
                for candidate in group["params"]
            )
        }
        muon_groups = [group for group in optimizer.param_groups if group.get("muon")]
        if selected != MUON_NAMES or len(muon_groups) != 1 or muon_groups[0].get("blocks") != 3:
            raise ValueError(f"wrong Muon selection: {selected}")
    else:
        raise ValueError(f"unknown optimizer: {name}")
    all_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    expected_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != expected_ids:
        raise ValueError("optimizer groups are not a disjoint exact trainable partition")
    for group in optimizer.param_groups:
        if any(id(parameter) in alpha_ids for parameter in group["params"]) and (
            group["weight_decay"] != 0.0 or group.get("muon", False)
        ):
            raise ValueError("learned alpha requires its own zero-decay AdamW group")
    return optimizer


def original_families(model: AdaptiveCandidateRNN) -> dict[str, list[nn.Parameter]]:
    """Group only incumbent parameters for registered family telemetry."""
    families = {"inp": [], "gru": [], "head": []}
    for name, parameter in model.named_parameters():
        if name == "raw_alpha":
            continue
        family = name.split(".", maxsplit=1)[0]
        if family not in families:
            raise ValueError(f"unassigned incumbent parameter: {name}")
        families[family].append(parameter)
    if sum(len(values) for values in families.values()) != len(list(model.parameters())) - int(
        model.raw_alpha is not None
    ):
        raise ValueError("baseline telemetry partition is incomplete")
    return families


def one_update(
    model: AdaptiveCandidateRNN,
    optimizer: torch.optim.Optimizer,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    ids: list[int],
    update_index: int,
    label: str,
) -> tuple[float, dict]:
    """Run one finite B4×256 update and collect fixed-step mechanism telemetry."""
    xx = torch.from_numpy(x[ids, :CHUNK_ROWS]).float()
    yy = torch.from_numpy(y[ids, :CHUNK_ROWS]).float()
    mm = torch.from_numpy(need[ids, :CHUNK_ROWS])
    if not (torch.isfinite(xx).all() and torch.isfinite(yy).all()):
        raise ValueError("nonfinite raw fit-only update input")
    optimizer.zero_grad(set_to_none=True)
    prediction, state = model(xx)
    if not (torch.isfinite(prediction).all() and torch.isfinite(state).all()):
        raise ValueError(f"nonfinite prediction/state: {label}")
    loss = e470.loss_for_batch(prediction, yy, mm)
    if not torch.isfinite(loss):
        raise ValueError(f"nonfinite fit loss: {label}")
    loss.backward()
    alpha_grad = None
    alpha_before = None
    if model.raw_alpha is not None:
        if model.raw_alpha.grad is None or not torch.isfinite(model.raw_alpha.grad).all():
            raise ValueError(f"missing/nonfinite alpha gradient: {label}")
        alpha_grad = model.raw_alpha.grad.detach().clone()
        alpha_before = model.raw_alpha.detach().clone()
    telemetry = e474.apply_optimizer_update(
        model, optimizer, original_families(model), update_index, label
    )
    if model.raw_alpha is not None:
        if telemetry is None:
            telemetry = {"update_index_zero_based": update_index}
        telemetry["raw_alpha"] = model.raw_alpha.detach().cpu().tolist()
        telemetry["effective_alpha"] = model.effective_alpha().detach().cpu().tolist()
        telemetry["raw_alpha_gradient"] = (
            alpha_grad.cpu().tolist() if alpha_grad is not None else None
        )
        telemetry["raw_alpha_update"] = (model.raw_alpha.detach() - alpha_before).cpu().tolist()
    elif telemetry is not None:
        telemetry["raw_alpha"] = None
        telemetry["effective_alpha"] = model.effective_alpha().detach().cpu().tolist()
        telemetry["raw_alpha_gradient"] = None
        telemetry["raw_alpha_update"] = None
    return float(loss.detach()), telemetry or {}


def run_cpu_preflight(
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    nodes: list[dict],
    states: dict[int, dict[str, Tensor]],
) -> list[dict]:
    """Check one actual fit-only update for every one of the 16 registered cells."""
    if x.shape != (24, 20_000, FEATURES) or y.shape != (24, 20_000, 2):
        raise ValueError(f"unexpected fit-only shape: {x.shape}, {y.shape}")
    if len(nodes) != 24 or any(node["role"] != "fit" for node in nodes):
        raise ValueError("E479 preflight must contain exactly 24 fit rows")
    results = []
    for seed, optimizer_name, condition in ARMS:
        order = np.arange(24)
        np.random.default_rng(seed).shuffle(order)
        ids = order[:4].tolist()
        torch.manual_seed(seed)
        model = initialize_arm(condition, states[seed])
        optimizer = make_optimizer(model, optimizer_name)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        loss, telemetry = one_update(
            model, optimizer, x, y, need, ids, 0, f"{seed}/{optimizer_name}/{condition}"
        )
        changed = [
            not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True)
        ]
        if not any(changed):
            raise ValueError(f"no parameters changed for {seed}/{optimizer_name}/{condition}")
        if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise ValueError(f"nonfinite updated parameter for {seed}/{optimizer_name}/{condition}")
        if condition.startswith("learned"):
            gradient = torch.tensor(telemetry["raw_alpha_gradient"])
            delta = torch.tensor(telemetry["raw_alpha_update"])
            if not torch.isfinite(gradient).all() or not (gradient.abs() > 0).any():
                raise ValueError("learned alpha gradient must be finite and nonzero")
            if not (delta.abs() > 0).any():
                raise ValueError("learned alpha must change in the CPU probe")
        results.append(
            {
                "seed": seed,
                "optimizer": optimizer_name,
                "condition": condition,
                "fit_row_groups": [nodes[index]["row_group"] for index in ids],
                "batch": 4,
                "chunk_rows": CHUNK_ROWS,
                "updates": 1,
                "loss": loss,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "changed_parameter_tensors": sum(changed),
                "telemetry_step0": telemetry,
            }
        )
    return results


def run_fit(
    seed: int,
    optimizer_name: str,
    condition: str,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    nodes: list[dict],
    base_state: dict[str, Tensor],
    device: torch.device,
    output: Path,
) -> dict:
    """Fit one full preregistered cell using fit-role groups only."""
    order = np.arange(24)
    np.random.default_rng(seed).shuffle(order)
    torch.manual_seed(seed)
    model = initialize_arm(condition, base_state).to(device).float()
    optimizer = make_optimizer(model, optimizer_name)
    families = original_families(model)
    telemetry: list[dict] = []
    update = 0
    started = time.monotonic()
    for batch_start in range(0, 24, 4):
        ids = order[batch_start : batch_start + 4].tolist()
        if len(ids) != 4:
            raise ValueError("E479 requires batches of exactly four fit row groups")
        state: Tensor | None = None
        for row_start in range(0, 20_000, CHUNK_ROWS):
            xx = torch.from_numpy(x[ids, row_start : row_start + CHUNK_ROWS]).to(
                device=device, dtype=torch.float32
            )
            yy = torch.from_numpy(y[ids, row_start : row_start + CHUNK_ROWS]).to(
                device=device, dtype=torch.float32
            )
            mm = torch.from_numpy(need[ids, row_start : row_start + CHUNK_ROWS]).to(device)
            if not (torch.isfinite(xx).all() and torch.isfinite(yy).all()):
                raise ValueError(f"nonfinite raw fit batch: {seed}/{optimizer_name}/{condition}")
            optimizer.zero_grad(set_to_none=True)
            prediction, state = model(xx, state)
            if not (torch.isfinite(prediction).all() and torch.isfinite(state).all()):
                raise ValueError(f"nonfinite prediction/state: {seed}/{optimizer_name}/{condition}")
            loss = e470.loss_for_batch(prediction, yy, mm)
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite loss: {seed}/{optimizer_name}/{condition}")
            loss.backward()
            alpha_gradient = None
            alpha_before = None
            if model.raw_alpha is not None:
                if model.raw_alpha.grad is None or not torch.isfinite(model.raw_alpha.grad).all():
                    raise ValueError("learned alpha gradient missing or nonfinite")
                alpha_gradient = model.raw_alpha.grad.detach().clone()
                alpha_before = model.raw_alpha.detach().clone()
            stats = e474.apply_optimizer_update(
                model, optimizer, families, update, f"{seed}/{optimizer_name}/{condition}"
            )
            if update in FIXED_TELEMETRY_STEPS:
                if stats is None:
                    raise ValueError("required fixed telemetry sample was not captured")
                stats.update(
                    {
                        "condition": condition,
                        "optimizer": optimizer_name,
                        "seed": seed,
                        "fit_row_groups": [nodes[index]["row_group"] for index in ids],
                        "chunk_start": row_start,
                        "effective_alpha": model.effective_alpha().detach().cpu().tolist(),
                        "raw_alpha": None
                        if model.raw_alpha is None
                        else model.raw_alpha.detach().cpu().tolist(),
                        "raw_alpha_gradient": None
                        if alpha_gradient is None
                        else alpha_gradient.cpu().tolist(),
                        "raw_alpha_update": (
                            None
                            if alpha_before is None
                            else (model.raw_alpha.detach() - alpha_before).cpu().tolist()
                        ),
                        "loss": float(loss.detach()),
                        "state_abs_max": float(state.detach().abs().max()),
                    }
                )
                telemetry.append(stats)
            state = state.detach()
            update += 1
    if (
        update != UPDATES_PER_FIT
        or tuple(item["update_index_zero_based"] for item in telemetry) != FIXED_TELEMETRY_STEPS
    ):
        raise ValueError(f"wrong update/telemetry count for {seed}/{optimizer_name}/{condition}")
    checkpoint_name = f"seed{seed}_{optimizer_name}_{condition}.pt"
    checkpoint_path = output / checkpoint_name
    torch.save(model.cpu().state_dict(), checkpoint_path)
    return {
        "seed": seed,
        "optimizer": optimizer_name,
        "condition": condition,
        "updates": update,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "fit_seconds": time.monotonic() - started,
        "fit_row_groups": [nodes[index]["row_group"] for index in order],
        "telemetry": telemetry,
        "checkpoint": checkpoint_name,
        "checkpoint_sha256": e474.file_hash(checkpoint_path),
    }


def main() -> None:
    """Run source-bound fit-only preflight; full fits require parent admission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--preflight-real", action="store_true")
    choice.add_argument("--run-fits", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output}")
    if args.preflight_real and args.device != "cpu":
        raise ValueError("real-data CPU preflight requires --device cpu")
    args.output.mkdir(parents=True)
    parent_source_hashes = check_frozen_sources()
    torch.set_num_threads(1)
    states = seed_states()
    manifest, x, y, need, _score, nodes = e470.load_cohort(include_test=False)
    if len(nodes) != 24 or any(node["role"] != "fit" for node in nodes):
        raise ValueError("fit-only loader contract failed")
    fit_manifest = {
        **manifest,
        "cohort": [node for node in manifest["cohort"] if node["role"] == "fit"],
    }
    if len(fit_manifest["cohort"]) != 24:
        raise ValueError("saved cohort metadata must contain only the 24 fit-role entries")
    result = {
        "kind": "E479 learned per-layer bounded GRU candidate-shape preflight",
        "status": "preflight_running",
        "fit_only": True,
        "test_rows_loaded": False,
        "seeds": list(SEEDS),
        "optimizers": list(OPTIMIZERS),
        "conditions": list(CONDITIONS),
        "planned_fit_count": len(ARMS),
        "architecture": "42->GELU projection->2x H64 reset-after GRU->concat state/embedding->GELU head",
        "candidate": "u=tanh(v); n=u+alpha_l*sin(pi*u); learned alpha_l=.2*tanh(raw_l)",
        "parameters": {"fixed": EXPECTED_FIXED_PARAMETERS, "learned": EXPECTED_LEARNED_PARAMETERS},
        "optimizer": {
            "lr": LR,
            "weight_decay_original_parameters": WEIGHT_DECAY,
            "learned_alpha_weight_decay": 0.0,
            "betas": list(BETAS),
            "schedule": "constant",
            "clip_norm": CLIP_NORM,
            "muon_matrices": sorted(MUON_NAMES),
            "muon_blocks_per_matrix": 3,
        },
        "updates_per_full_fit": UPDATES_PER_FIT,
        "batch": 4,
        "chunk_rows": CHUNK_ROWS,
        "telemetry_steps_zero_based": list(FIXED_TELEMETRY_STEPS),
        "contrasts": list(CONTRASTS),
        "parent_source_sha256": parent_source_hashes,
        "source_sha256": {
            name: e474.file_hash(HERE / name)
            for name in (
                "fit_recurrent_activation_adaptive.py",
                "tests.py",
                "gpu_probe.py",
            )
        },
        "cohort_sha256": e474.file_hash(e470.COHORT_PATH),
        "dataset_sha256": DATASET_SHA256,
        "fit_only_cohort": fit_manifest,
        "arms": [],
    }
    if args.preflight_real:
        result["arms"] = run_cpu_preflight(x, y, need, nodes, states)
        result["status"] = "preflight_passed"
    else:
        if args.device == "cuda":
            configure_torch()
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
        device = torch.device(args.device)
        with e474.full_ieee_float32():
            result["precision"] = e474.precision_metadata()
            result["arms"] = [
                run_fit(
                    seed,
                    optimizer_name,
                    condition,
                    x,
                    y,
                    need,
                    nodes,
                    states[seed],
                    device,
                    args.output,
                )
                for seed, optimizer_name, condition in ARMS
            ]
        result["status"] = "all_16_fit_arms_complete_test_unread"
    (args.output / "fit_manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
