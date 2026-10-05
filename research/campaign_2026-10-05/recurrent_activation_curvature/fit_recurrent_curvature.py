"""Fit the preregistered E481 three-shape GRU screen on fit-role rows only."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E470_DIR))
sys.path.insert(0, str(E474_DIR))

import fit_activation_optimizer as e470  # noqa: E402
import fit_activation_placement as e474  # noqa: E402

from wnn.nn import RNNModel  # noqa: E402
from wnn.optim import MuonAdamW, muon_param_groups  # noqa: E402
from wnn.resources import configure_torch  # noqa: E402

SEEDS = (271828, 271829, 271830)
OPTIMIZERS = ("adamw", "muon_adamw")
SHAPES = ("baseline", "sine", "cubic")
ARMS = tuple(
    (seed, optimizer, shape) for seed in SEEDS for optimizer in OPTIMIZERS for shape in SHAPES
)
FEATURES, HIDDEN, CHUNK_ROWS, UPDATES = 42, 64, 256, 474
N_PARAMETERS = 61_058
TELEMETRY_STEPS = (0, 46, 236, 473)
MUON_NAMES = {"gru.weight_ih_l0", "gru.weight_hh_l0", "gru.weight_ih_l1", "gru.weight_hh_l1"}
LR, WD, BETAS, CLIP_NORM = 0.001, 0.01, (0.9, 0.999), 1.0
DATASET_SHA256 = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
E470_SHA256 = "871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9"
E474_SHA256 = "40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068"
COHORT_SHA256 = "068103c71663bc445aab943e314c5be1d9a094f7f83456fdb406fbdc1012a6f2"
SHAPE_CODE = {"baseline": 0, "sine": 1, "cubic": 2}


@torch.jit.script
def candidate_transform(u: Tensor, shape_code: int) -> Tensor:
    """Apply baseline, bounded sine, or matched-center cubic candidate map."""
    if shape_code == 0:
        return u
    if shape_code == 1:
        return u + 0.1 * torch.sin(math.pi * u)
    if shape_code == 2:
        return u + (0.1 * math.pi) * u * (1.0 - u * u)
    raise ValueError("unregistered candidate shape code")


@torch.jit.script
def scripted_shape_scan(
    x: Tensor,
    h0: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Tensor,
    bias_hh: Tensor,
    shape_code: int,
) -> tuple[Tensor, Tensor]:
    """Reset-after PyTorch GRU scan with input-affine r/z/n gate order."""
    input_affine = torch.nn.functional.linear(x, weight_ih, bias_ih)
    outputs = torch.jit.annotate(list[Tensor], [])
    hidden = h0
    for step in range(x.size(1)):
        input_reset, input_update, input_candidate = input_affine[:, step, :].chunk(3, dim=1)
        hidden_reset, hidden_update, hidden_candidate = torch.nn.functional.linear(
            hidden, weight_hh, bias_hh
        ).chunk(3, dim=1)
        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        u = torch.tanh(input_candidate + reset * hidden_candidate)
        candidate = candidate_transform(u, shape_code)
        hidden = (1.0 - update) * candidate + update * hidden
        outputs.append(hidden)
    return torch.stack(outputs, dim=1), hidden


class CurvatureRNN(e474.PlacementRNN):
    """E478 topology with a fixed candidate formula and no added parameters."""

    def __init__(self, shape: str):
        if shape not in SHAPE_CODE:
            raise ValueError(f"unregistered shape {shape}")
        super().__init__(embedding_activation="gelu", head_activation="gelu")
        self.shape_name = shape
        self.shape_code = SHAPE_CODE[shape]

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != FEATURES:
            raise ValueError("expected input shape [batch,time,42]")
        encoded = self.encode((x - self.mean) / self.std)
        batch = encoded.shape[0]
        if state is None:
            hidden_state = encoded.new_zeros((2, batch, HIDDEN))
        else:
            if state.shape != (2, batch, HIDDEN):
                raise ValueError("expected recurrent state shape [2,batch,64]")
            hidden_state = state
        layer_input = encoded
        finals = []
        for layer in range(2):
            layer_input, final = scripted_shape_scan(
                layer_input,
                hidden_state[layer],
                getattr(self.gru, f"weight_ih_l{layer}"),
                getattr(self.gru, f"weight_hh_l{layer}"),
                getattr(self.gru, f"bias_ih_l{layer}"),
                getattr(self.gru, f"bias_hh_l{layer}"),
                self.shape_code,
            )
            finals.append(final)
        prediction = self.head(torch.cat((layer_input, encoded), dim=-1))
        return prediction, torch.stack(finals)


def check_parent_sources() -> dict[str, str]:
    sources = {
        "e470": (E470_DIR / "fit_activation_optimizer.py", E470_SHA256),
        "e474": (E474_DIR / "fit_activation_placement.py", E474_SHA256),
    }
    result = {name: e474.file_hash(path) for name, (path, _expected) in sources.items()}
    for name, (_path, expected) in sources.items():
        if result[name] != expected:
            raise ValueError(f"frozen source mismatch: {name}={result[name]}")
    return result


def seed_states() -> dict[int, dict[str, Tensor]]:
    states = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        baseline = RNNModel(FEATURES, hidden=HIDDEN, layers=2, dropout=0.0, n_out=2)
        if sum(p.numel() for p in baseline.parameters()) != N_PARAMETERS:
            raise ValueError(f"baseline parameter count changed for seed {seed}")
        states[seed] = {key: value.detach().clone() for key, value in baseline.state_dict().items()}
    return states


def initialize_arm(shape: str, state: dict[str, Tensor]) -> CurvatureRNN:
    model = CurvatureRNN(shape)
    model.load_state_dict(state, strict=True)
    if sum(p.numel() for p in model.parameters()) != N_PARAMETERS:
        raise ValueError(f"parameter count changed for {shape}")
    return model


def make_optimizer(model: CurvatureRNN, optimizer_name: str) -> torch.optim.Optimizer:
    if optimizer_name == "adamw":
        optimizer: torch.optim.Optimizer = e474.make_optimizer(model)
    elif optimizer_name == "muon_adamw":
        groups = muon_param_groups(model, scope="rnn")
        ids = [id(p) for group in groups for p in group["params"]]
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError("Muon/AdamW groups are not an exact disjoint parameter partition")
        muon_groups = [group for group in groups if group.get("muon")]
        selected = {
            name
            for name, parameter in model.named_parameters()
            if any(
                id(parameter) == id(candidate)
                for group in muon_groups
                for candidate in group["params"]
            )
        }
        if selected != MUON_NAMES or len(muon_groups) != 1 or muon_groups[0].get("blocks") != 3:
            raise ValueError(f"Muon selection differs: {selected}")
        optimizer = MuonAdamW(groups, lr=LR, weight_decay=WD, betas=BETAS)
    else:
        raise ValueError(f"unregistered optimizer {optimizer_name}")
    for group in optimizer.param_groups:
        if group["lr"] != LR or group["weight_decay"] != WD:
            raise ValueError("optimizer LR/weight decay differs from E481")
    return optimizer


def one_update(
    model: CurvatureRNN,
    optimizer: torch.optim.Optimizer,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    ids: list[int],
    device: torch.device,
    owner: str,
) -> tuple[float, dict]:
    """One fit-only B4×256 update with a global clip and finite checks."""
    xx = torch.from_numpy(x[ids, :CHUNK_ROWS]).to(device=device, dtype=torch.float32)
    yy = torch.from_numpy(y[ids, :CHUNK_ROWS]).to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(need[ids, :CHUNK_ROWS]).to(device=device)
    if not (torch.isfinite(xx).all() and torch.isfinite(yy).all()):
        raise ValueError(f"nonfinite raw fit input: {owner}")
    optimizer.zero_grad(set_to_none=True)
    prediction, state = model(xx)
    if not (torch.isfinite(prediction).all() and torch.isfinite(state).all()):
        raise ValueError(f"nonfinite prediction or state: {owner}")
    loss = e470.loss_for_batch(prediction, yy, mask)
    if not torch.isfinite(loss):
        raise ValueError(f"nonfinite fit loss: {owner}")
    loss.backward()
    telemetry = e474.apply_optimizer_update(
        model, optimizer, e474.parameter_families(model), 0, owner
    )
    if telemetry is None:
        raise ValueError("update zero telemetry was not captured")
    return float(loss.detach().cpu()), telemetry


def run_cpu_preflight(
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    nodes: list[dict],
    initial_states: dict[int, dict[str, Tensor]],
) -> list[dict]:
    """One actual fit-only CPU update for every registered experiment cell."""
    if (
        x.shape != (24, 20_000, FEATURES)
        or y.shape != (24, 20_000, 2)
        or need.shape != (24, 20_000)
    ):
        raise ValueError(f"unexpected fit-only arrays: {x.shape}, {y.shape}, {need.shape}")
    if len(nodes) != 24 or any(n["role"] != "fit" for n in nodes):
        raise ValueError("CPU preflight requires exactly 24 fit-role groups")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("nonfinite fit-only arrays")
    rows = []
    for seed, optimizer_name, shape in ARMS:
        order = np.arange(24)
        np.random.default_rng(seed).shuffle(order)
        ids = order[:4].tolist()
        torch.manual_seed(seed)
        model = initialize_arm(shape, initial_states[seed])
        optimizer = make_optimizer(model, optimizer_name)
        before = [p.detach().clone() for p in model.parameters()]
        loss, telemetry = one_update(
            model,
            optimizer,
            x,
            y,
            need,
            ids,
            torch.device("cpu"),
            f"{seed}/{optimizer_name}/{shape}",
        )
        changed = [not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True)]
        if not any(changed) or not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise ValueError(
                f"nonfinite or no-change preflight update: {seed}/{optimizer_name}/{shape}"
            )
        rows.append(
            {
                "seed": seed,
                "optimizer": optimizer_name,
                "shape": shape,
                "fit_row_groups": [nodes[i]["row_group"] for i in ids],
                "batch": 4,
                "chunk_rows": CHUNK_ROWS,
                "updates": 1,
                "loss": loss,
                "parameters": sum(p.numel() for p in model.parameters()),
                "changed_parameter_tensors": sum(changed),
                "telemetry_step0": telemetry,
            }
        )
    return rows


def run_fit(
    seed: int,
    optimizer_name: str,
    shape: str,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    nodes: list[dict],
    state_dict: dict[str, Tensor],
    device: torch.device,
    output: Path,
) -> dict:
    order = np.arange(24)
    np.random.default_rng(seed).shuffle(order)
    torch.manual_seed(seed)
    model = initialize_arm(shape, state_dict).to(device).float()
    optimizer = make_optimizer(model, optimizer_name)
    families = e474.parameter_families(model)
    telemetry, update = [], 0
    started = time.monotonic()
    for batch_start in range(0, 24, 4):
        ids = order[batch_start : batch_start + 4].tolist()
        recurrent_state = None
        for row_start in range(0, 20_000, CHUNK_ROWS):
            xx = torch.from_numpy(x[ids, row_start : row_start + CHUNK_ROWS]).to(
                device=device, dtype=torch.float32
            )
            yy = torch.from_numpy(y[ids, row_start : row_start + CHUNK_ROWS]).to(
                device=device, dtype=torch.float32
            )
            mask = torch.from_numpy(need[ids, row_start : row_start + CHUNK_ROWS]).to(device=device)
            if not (torch.isfinite(xx).all() and torch.isfinite(yy).all()):
                raise ValueError(f"nonfinite fit batch: {seed}/{optimizer_name}/{shape}")
            optimizer.zero_grad(set_to_none=True)
            prediction, recurrent_state = model(xx, recurrent_state)
            if not (torch.isfinite(prediction).all() and torch.isfinite(recurrent_state).all()):
                raise ValueError(f"nonfinite prediction/state: {seed}/{optimizer_name}/{shape}")
            loss = e470.loss_for_batch(prediction, yy, mask)
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite loss: {seed}/{optimizer_name}/{shape}")
            loss.backward()
            stats = e474.apply_optimizer_update(
                model, optimizer, families, update, f"{seed}/{optimizer_name}/{shape}"
            )
            if update in TELEMETRY_STEPS:
                if stats is None:
                    raise ValueError("required telemetry sample missing")
                stats.update(
                    {
                        "seed": seed,
                        "optimizer": optimizer_name,
                        "shape": shape,
                        "fit_row_groups": [nodes[i]["row_group"] for i in ids],
                        "chunk_start": row_start,
                        "loss": float(loss.detach()),
                        "state_abs_max": float(recurrent_state.detach().abs().max()),
                    }
                )
                telemetry.append(stats)
            recurrent_state = recurrent_state.detach()
            update += 1
    if update != UPDATES or [x["update_index_zero_based"] for x in telemetry] != list(
        TELEMETRY_STEPS
    ):
        raise ValueError(f"wrong cadence for {seed}/{optimizer_name}/{shape}: {update}")
    checkpoint = f"seed{seed}_{optimizer_name}_{shape}.pt"
    checkpoint_path = output / checkpoint
    torch.save(model.cpu().state_dict(), checkpoint_path)
    return {
        "seed": seed,
        "optimizer": optimizer_name,
        "shape": shape,
        "updates": update,
        "parameters": sum(p.numel() for p in model.parameters()),
        "fit_seconds": time.monotonic() - started,
        "fit_row_groups": [nodes[int(i)]["row_group"] for i in order],
        "telemetry": telemetry,
        "checkpoint": checkpoint,
        "checkpoint_sha256": e474.file_hash(checkpoint_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--preflight-real", action="store_true")
    operation.add_argument("--run-fits", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    if args.preflight_real and args.device != "cpu":
        raise ValueError("real-data CPU preflight must use CPU")
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    parents = check_parent_sources()
    states = seed_states()
    manifest, x, y, need, _score, nodes = e470.load_cohort(include_test=False)
    if len(nodes) != 24 or {n["role"] for n in nodes} != {"fit"}:
        raise ValueError("loader did not return exactly the 24 fit groups")
    fit_manifest = {**manifest, "cohort": [n for n in manifest["cohort"] if n["role"] == "fit"]}
    result = {
        "kind": "E481 matched-center recurrent candidate-shape screen",
        "status": "preflight_running",
        "fit_only": True,
        "test_rows_loaded": False,
        "seeds": list(SEEDS),
        "optimizers": list(OPTIMIZERS),
        "shapes": list(SHAPES),
        "planned_fit_count": 18,
        "architecture": "42->GELU projection->2x H64 reset-after GRU->concat state/embedding->GELU head",
        "candidate_formulas": {
            "baseline": "u",
            "sine": "u+0.1*sin(pi*u)",
            "cubic": "u+0.1*pi*u*(1-u*u)",
        },
        "parameters_per_arm": N_PARAMETERS,
        "optimizer": {
            "lr": LR,
            "weight_decay": WD,
            "betas": list(BETAS),
            "schedule": "constant",
            "clip_norm": CLIP_NORM,
            "muon_matrices": sorted(MUON_NAMES),
            "muon_blocks_per_matrix": 3,
        },
        "updates_per_full_fit": UPDATES,
        "batch": 4,
        "chunk_rows": CHUNK_ROWS,
        "telemetry_steps_zero_based": list(TELEMETRY_STEPS),
        "parent_source_sha256": parents,
        "source_sha256": {
            p.name: e474.file_hash(p)
            for p in (HERE / "fit_recurrent_curvature.py", HERE / "tests.py", HERE / "gpu_probe.py")
        },
        "dataset_sha256": DATASET_SHA256,
        "cohort_sha256": e474.file_hash(e470.COHORT_PATH),
        "fit_only_cohort": fit_manifest,
        "arms": [],
    }
    if args.preflight_real:
        result["arms"] = run_cpu_preflight(x, y, need, nodes, states)
        result["status"] = "cpu_fit_only_one_update_per_arm_passed_test_unread"
    else:
        if args.device != "cuda":
            raise ValueError("full E481 fits require CUDA admission")
        configure_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        with e474.full_ieee_float32():
            result["precision"] = e474.precision_metadata()
            result["arms"] = [
                run_fit(
                    s, o, shape, x, y, need, nodes, states[s], torch.device("cuda"), args.output
                )
                for s, o, shape in ARMS
            ]
        result["status"] = "all_18_fit_arms_complete_test_unread"
    (args.output / "fit_manifest.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "status": result["status"],
                "arms": len(result["arms"]),
                "test_rows_loaded": False,
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
