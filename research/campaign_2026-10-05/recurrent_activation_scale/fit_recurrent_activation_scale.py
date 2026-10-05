"""Fit-only E478 bounded recurrent-candidate transfer screen."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
E471_DIR = ROOT / "research/campaign_2026-10-05/recurrent_activation"
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E470_DIR))
sys.path.insert(0, str(E471_DIR))
sys.path.insert(0, str(E474_DIR))

import fit_activation_optimizer as e470  # noqa: E402
import fit_activation_placement as e474  # noqa: E402
from fit_recurrent_candidate import scripted_candidate_scan  # noqa: E402

from wnn.nn import RNNModel  # noqa: E402
from wnn.optim import MuonAdamW, muon_param_groups  # noqa: E402
from wnn.resources import configure_torch  # noqa: E402

SEEDS = (271828, 271829, 271830)
ALPHAS = (0.0, 0.1)
HIDDEN = 64
FEATURES = 42
EXPECTED_PARAMETERS = 61_058
CHUNK_ROWS = 256
UPDATES_PER_FIT = 474
MUON_NAMES = {
    "gru.weight_ih_l0",
    "gru.weight_hh_l0",
    "gru.weight_ih_l1",
    "gru.weight_hh_l1",
}
OUTPUT_ATOL = 2e-5
OUTPUT_RTOL = 2e-4
GRAD_ATOL = 5e-5
GRAD_RTOL = 5e-4
UPDATE_ATOL = 1e-4
UPDATE_RTOL = 1e-3
E470_SOURCE_SHA256 = "871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9"
E471_SOURCE_SHA256 = "998bceb024b5c29e81e73efc5d0ebe434d53c7356cfcb947f7ca36dd625426cf"
E474_SOURCE_SHA256 = "40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068"


class BoundedCandidateRNN(e474.PlacementRNN):
    """Actual E474 H64 topology with only the two GRU candidate maps changed."""

    def __init__(self, alpha: float) -> None:
        if alpha not in ALPHAS:
            raise ValueError(f"unplanned candidate alpha: {alpha}")
        super().__init__(embedding_activation="gelu", head_activation="gelu")
        self.alpha = float(alpha)

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
        layer_input = encoded
        final_states = []
        for layer in range(2):
            weight_ih = getattr(self.gru, f"weight_ih_l{layer}")
            weight_hh = getattr(self.gru, f"weight_hh_l{layer}")
            bias_ih = getattr(self.gru, f"bias_ih_l{layer}")
            bias_hh = getattr(self.gru, f"bias_hh_l{layer}")
            layer_output, final_hidden = scripted_candidate_scan(
                layer_input,
                hidden_state[layer],
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                self.alpha,
            )
            layer_input = layer_output
            final_states.append(final_hidden)
        recurrent_state = torch.stack(final_states, dim=0)
        prediction = self.head(torch.cat((layer_input, encoded), dim=-1))
        return prediction, recurrent_state


def check_parent_sources() -> dict[str, str]:
    """Require the frozen E470/E471-scan/E474 helpers used by this screen."""
    sources = {
        "e470": (E470_DIR / "fit_activation_optimizer.py", E470_SOURCE_SHA256),
        "e471_scan": (E471_DIR / "fit_recurrent_candidate.py", E471_SOURCE_SHA256),
        "e474": (E474_DIR / "fit_activation_placement.py", E474_SOURCE_SHA256),
    }
    actual = {name: e474.file_hash(path) for name, (path, _) in sources.items()}
    for name, (_, expected) in sources.items():
        if actual[name] != expected:
            raise ValueError(f"frozen parent source changed: {name}={actual[name]}")
    return actual


def make_initial_states() -> dict[int, dict[str, Tensor]]:
    """Clone actual GELU RNNModel parameters once per seed for both alphas."""
    states = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        baseline = RNNModel(FEATURES, hidden=HIDDEN, layers=2, dropout=0.0, n_out=2)
        count = sum(parameter.numel() for parameter in baseline.parameters())
        if count != EXPECTED_PARAMETERS:
            raise ValueError(f"live RNNModel parameter count changed to {count}")
        states[seed] = {key: value.detach().clone() for key, value in baseline.state_dict().items()}
    return states


def initialize_arm(alpha: float, state: dict[str, Tensor]) -> BoundedCandidateRNN:
    """Load the common incumbent-topology state into the candidate scan."""
    model = BoundedCandidateRNN(alpha)
    model.load_state_dict(state, strict=True)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EXPECTED_PARAMETERS:
        raise ValueError(f"candidate parameter count differs: {count}")
    return model


def make_muon_optimizer(model: BoundedCandidateRNN) -> MuonAdamW:
    """Muon exactly the four stacked GRU matrices; use AdamW for the rest."""
    groups = muon_param_groups(model, scope="rnn")
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assigned = [id(parameter) for group in groups for parameter in group["params"]]
    if len(assigned) != len(set(assigned)) or set(assigned) != trainable:
        raise ValueError("Muon/AdamW groups must be an exact disjoint trainable partition")
    muon_groups = [group for group in groups if group.get("muon")]
    selected = {
        name
        for name, parameter in model.named_parameters()
        if any(
            id(parameter) == id(candidate) for group in muon_groups for candidate in group["params"]
        )
    }
    if selected != MUON_NAMES or len(muon_groups) != 1 or muon_groups[0].get("blocks") != 3:
        raise ValueError(f"unexpected Muon matrix selection: {selected}")
    return MuonAdamW(groups, lr=0.001, weight_decay=0.01, betas=(0.9, 0.999))


def telemetry_step(
    model: BoundedCandidateRNN,
    optimizer: MuonAdamW,
    update_index: int,
    owner: str,
) -> dict:
    """Apply E474's fixed telemetry and clipped update helper."""
    stats = e474.apply_optimizer_update(
        model, optimizer, e474.parameter_families(model), update_index, owner
    )
    if stats is None:
        raise ValueError("preflight telemetry update must be one of the fixed sample steps")
    return stats


def run_cpu_preflight(
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    nodes: list[dict],
    initial_states: dict[int, dict[str, Tensor]],
) -> list[dict]:
    """Apply one real fit-only B4x256 Muon update to each seed/alpha arm."""
    if x.shape != (24, 20_000, FEATURES) or y.shape != (24, 20_000, 2):
        raise ValueError("unexpected fit-only preflight data shape")
    if len(nodes) != 24 or any(node["role"] != "fit" for node in nodes):
        raise ValueError("preflight may only receive fit row groups")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("nonfinite fit-only preflight arrays")
    receipts = []
    for seed in SEEDS:
        order = np.arange(24)
        np.random.default_rng(seed).shuffle(order)
        ids = order[:4].tolist()
        groups = [nodes[index]["row_group"] for index in ids]
        for alpha in ALPHAS:
            torch.manual_seed(seed)
            model = initialize_arm(alpha, initial_states[seed])
            optimizer = make_muon_optimizer(model)
            before = [parameter.detach().clone() for parameter in model.parameters()]
            xx = torch.from_numpy(x[ids, :CHUNK_ROWS])
            yy = torch.from_numpy(y[ids, :CHUNK_ROWS])
            mm = torch.from_numpy(need[ids, :CHUNK_ROWS])
            optimizer.zero_grad(set_to_none=True)
            prediction, state = model(xx)
            if not (torch.isfinite(prediction).all() and torch.isfinite(state).all()):
                raise ValueError(f"nonfinite CPU preflight forward for {seed}/alpha{alpha}")
            loss = e470.loss_for_batch(prediction, yy, mm)
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite CPU preflight loss for {seed}/alpha{alpha}")
            loss.backward()
            stats = telemetry_step(model, optimizer, 0, f"{seed}/alpha{alpha}")
            params = list(model.parameters())
            changed = any(
                not torch.equal(old, new) for old, new in zip(before, params, strict=True)
            )
            finite = all(bool(torch.isfinite(parameter).all().item()) for parameter in params)
            if not changed or not finite:
                raise ValueError(f"CPU preflight did not update finite parameters: {seed}/{alpha}")
            receipts.append(
                {
                    "seed": seed,
                    "alpha": alpha,
                    "fit_row_groups": groups,
                    "batch": 4,
                    "chunk_rows": CHUNK_ROWS,
                    "updates": 1,
                    "loss": float(loss.detach()),
                    "parameters": sum(parameter.numel() for parameter in params),
                    "parameters_changed": changed,
                    "parameters_finite": finite,
                    "telemetry_step0": stats,
                }
            )
    return receipts


def run_fit(
    seed: int,
    alpha: float,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    nodes: list[dict],
    device: torch.device,
    output: Path,
    initial_state: dict[str, Tensor],
) -> dict:
    """Train one of the fixed Muon-only alpha arms, without test rows."""
    order = np.arange(24)
    np.random.default_rng(seed).shuffle(order)
    torch.manual_seed(seed)
    model = initialize_arm(alpha, initial_state).to(device).float()
    optimizer = make_muon_optimizer(model)
    families = e474.parameter_families(model)
    telemetry = []
    step = 0
    started = time.monotonic()
    for batch_start in range(0, 24, 4):
        ids = order[batch_start : batch_start + 4].tolist()
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
                raise ValueError(f"nonfinite raw fit batch for {seed}/alpha{alpha}")
            optimizer.zero_grad(set_to_none=True)
            prediction, state = model(xx, state)
            if not (torch.isfinite(prediction).all() and torch.isfinite(state).all()):
                raise ValueError(f"nonfinite prediction/state for {seed}/alpha{alpha}")
            loss = e470.loss_for_batch(prediction, yy, mm)
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite loss for {seed}/alpha{alpha}")
            loss.backward()
            stats = e474.apply_optimizer_update(
                model, optimizer, families, step, f"{seed}/alpha{alpha}"
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
    if step != UPDATES_PER_FIT or [item["update_index_zero_based"] for item in telemetry] != [
        0,
        46,
        236,
        473,
    ]:
        raise ValueError(f"unexpected fit/telemetry step count for {seed}/alpha{alpha}")
    checkpoint = output / f"seed{seed}_alpha{alpha:.1f}.pt"
    torch.save(model.cpu().state_dict(), checkpoint)
    return {
        "seed": seed,
        "alpha": alpha,
        "optimizer": "MuonAdamW; Muon on four GRU matrices, AdamW on others",
        "muon_matrices": sorted(MUON_NAMES),
        "muon_gate_blocks": 3,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "updates": step,
        "fit_seconds": time.monotonic() - started,
        "fit_row_groups": [nodes[index]["row_group"] for index in order],
        "telemetry": telemetry,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": e474.file_hash(checkpoint),
    }


def main() -> None:
    """Run source checks and fit-only preflight, or the parent-admitted full fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-real", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)
    parent_hashes = check_parent_sources()
    device = torch.device("cpu" if args.preflight_real else args.device)
    if device.type == "cuda":
        configure_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    initial_states = make_initial_states()
    manifest, x, y, need, _, nodes = e470.load_cohort(include_test=False)
    if len(nodes) != 24 or any(node["role"] != "fit" for node in nodes):
        raise ValueError("E478 may only load fit rows")
    if x.shape != (24, 20_000, FEATURES) or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError(f"invalid fit-only cohort arrays: {x.shape}")
    result = {
        "kind": "E478 bounded recurrent candidate transfer to H64 incumbent topology",
        "status": "preflight" if args.preflight_real else "fitting",
        "fit_only": True,
        "test_rows_loaded": False,
        "seeds": list(SEEDS),
        "alphas": list(ALPHAS),
        "arms": [],
        "parameters_per_arm": EXPECTED_PARAMETERS,
        "architecture": "42->GELU linear projection->two H64 GRU layers->concat recurrent/embedding->GELU head",
        "candidate": "u=tanh(v); n=u+alpha*sin(pi*u); h'=(1-z)*n+z*h",
        "updates_per_fit": UPDATES_PER_FIT,
        "batch": 4,
        "chunk_rows": CHUNK_ROWS,
        "optimizer": {
            "name": "MuonAdamW",
            "muon_matrices": sorted(MUON_NAMES),
            "muon_blocks_per_matrix": 3,
            "adamw_for": "all other trainable parameters",
            "lr": 0.001,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "schedule": "constant",
            "clip_norm": 1.0,
        },
        "parity_tolerances": {
            "output_state": {"atol": OUTPUT_ATOL, "rtol": OUTPUT_RTOL},
            "input_parameter_gradients": {"atol": GRAD_ATOL, "rtol": GRAD_RTOL},
            "post_optimizer_parameters": {"atol": UPDATE_ATOL, "rtol": UPDATE_RTOL},
        },
        "telemetry_sampling": {
            "step_indices_zero_based": list(e474.TELEMETRY_STEPS),
            "families": ["inp", "gru", "head"],
            "gradient": "preclip element-weighted RMS",
            "update": "parameter delta RMS",
            "descriptive_only": True,
        },
        "cohort_sha256": e474.file_hash(e470.COHORT_PATH),
        "dataset_sha256": e474.DATASET_SHA256,
        "parent_source_sha256": parent_hashes,
        "source_sha256": {
            name: e474.file_hash(HERE / name)
            for name in ("fit_recurrent_activation_scale.py", "tests.py", "gpu_probe.py")
            if (HERE / name).is_file()
        },
    }
    (args.output / "fit_only_cohort.json").write_text(json.dumps(manifest, indent=2))
    if args.preflight_real:
        result["arms"] = run_cpu_preflight(x, y, need, nodes, initial_states)
        result["status"] = "preflight_passed"
    else:
        with e474.full_ieee_float32():
            result["precision"] = e474.precision_metadata()
            for seed in SEEDS:
                for alpha in ALPHAS:
                    result["arms"].append(
                        run_fit(
                            seed,
                            alpha,
                            x,
                            y,
                            need,
                            nodes,
                            device,
                            args.output,
                            initial_states[seed],
                        )
                    )
                    (args.output / "fit_manifest.json").write_text(json.dumps(result, indent=2))
        if len(result["arms"]) != 6:
            raise ValueError("all six preregistered fits must finish before handoff")
        result["status"] = "all_6_fit_arms_complete_test_unread"
    (args.output / "fit_manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
