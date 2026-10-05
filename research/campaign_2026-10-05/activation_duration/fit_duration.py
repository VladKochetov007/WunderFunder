"""E485 fit-only recurrent candidate duration experiment."""

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

SEEDS = (271836, 271837, 271838)
ALPHAS = (0.0, 0.1)
HIDDEN = 128
FEATURES = 42
EXPECTED_PARAMETERS = 236_802
CHUNK_ROWS = 256
ROWS_PER_GROUP = 20_000
BATCH_SIZE = 4
UPDATES_PER_EPOCH = math.ceil(ROWS_PER_GROUP / CHUNK_ROWS) * (24 // BATCH_SIZE)
EPOCHS = 4
SNAPSHOT_EPOCHS = (1, 2, 4)
UPDATES_PER_TRAJECTORY = UPDATES_PER_EPOCH * EPOCHS
MUON_NAMES = {
    f"gru.{prefix}_l{layer}" for prefix in ("weight_ih", "weight_hh") for layer in range(2)
}
TELEMETRY_STEPS = (0, 46, 236, 473)
CLIP_NORM = 1.0
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
E470_SOURCE_SHA256 = "871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9"
E471_SOURCE_SHA256 = "998bceb024b5c29e81e73efc5d0ebe434d53c7356cfcb947f7ca36dd625426cf"
E474_SOURCE_SHA256 = "40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068"
WNN_NN_SHA256 = "388af04570f0a3944159bf86c1a234dc55c78b951804c38ef59fa314f92f3db3"
WNN_OPTIM_SHA256 = "620569a752fb611adf69976f1dbed1b7882e641672c58175005f5d3f819d2991"
WNN_RESOURCES_SHA256 = "06a624ad963dfc2b0c75ce717278de5f1bb85f48518a99e519b697a371d4cf09"
DESIGN_SHA256 = "bcbf17ec7b632909e9e8792cdf11436822dd4dcac5326befa506c8dcb96e70cc"
CRITIC_SHA256 = "7e5a2457636febfbec13f9cff7d4205757e0cb7de4efbfe9b2deb679481bbdeb"
UPDATE_GATES_SHA256 = "52d547abd2da11cd712e5973b2ef1ac2eca70b449cbd143abf8141bea221d164"


class DurationCandidateRNN(e474.PlacementRNN):
    """H128 incumbent topology with a fixed candidate-only sine perturbation."""

    def __init__(self, alpha: float) -> None:
        if alpha not in ALPHAS:
            raise ValueError(f"unregistered alpha: {alpha}")
        super().__init__("gelu", "gelu", hidden=HIDDEN)
        self.alpha = float(alpha)

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != FEATURES:
            raise ValueError("input must have shape (batch, time, 42)")
        encoded = self.encode((x - self.mean) / self.std)
        batch = encoded.shape[0]
        if state is None:
            hidden_state = encoded.new_zeros((2, batch, HIDDEN))
        else:
            if state.shape != (2, batch, HIDDEN):
                raise ValueError("state must have shape (2, batch, 128)")
            hidden_state = state
        layer_input = encoded
        final_states: list[Tensor] = []
        for layer in range(2):
            layer_output, final_hidden = scripted_candidate_scan(
                layer_input,
                hidden_state[layer],
                getattr(self.gru, f"weight_ih_l{layer}"),
                getattr(self.gru, f"weight_hh_l{layer}"),
                getattr(self.gru, f"bias_ih_l{layer}"),
                getattr(self.gru, f"bias_hh_l{layer}"),
                self.alpha,
            )
            layer_input = layer_output
            final_states.append(final_hidden)
        recurrent_state = torch.stack(final_states, dim=0)
        prediction = self.head(torch.cat((layer_input, encoded), dim=-1))
        return prediction, recurrent_state


def parent_source_hashes() -> dict[str, str]:
    """Check the frozen cohort, scan, and precision/update helpers."""
    files = {
        "e470": (E470_DIR / "fit_activation_optimizer.py", E470_SOURCE_SHA256),
        "e471_scan": (E471_DIR / "fit_recurrent_candidate.py", E471_SOURCE_SHA256),
        "e474": (E474_DIR / "fit_activation_placement.py", E474_SOURCE_SHA256),
        "wnn.nn": (ROOT / "src/wnn/nn.py", WNN_NN_SHA256),
        "wnn.optim": (ROOT / "src/wnn/optim.py", WNN_OPTIM_SHA256),
        "wnn.resources": (ROOT / "src/wnn/resources.py", WNN_RESOURCES_SHA256),
    }
    actual = {name: e474.file_hash(path) for name, (path, _) in files.items()}
    for name, (_, expected) in files.items():
        if actual[name] != expected:
            raise ValueError(f"frozen parent source changed: {name}={actual[name]}")
    design_hash = e474.file_hash(HERE / "E485_DESIGN.md")
    critic_hash = e474.file_hash(HERE / "SPEC_CRITIC.md")
    gate_hash = e474.file_hash(HERE / "GPU_PROBE_UPDATE_GATES.md")
    if (
        design_hash != DESIGN_SHA256
        or critic_hash != CRITIC_SHA256
        or gate_hash != UPDATE_GATES_SHA256
    ):
        raise ValueError("registered E485 design, blind critique or update gates changed")
    return actual


def make_initial_states() -> dict[int, dict[str, Tensor]]:
    """Create one native RNNModel initialization per seed for both shapes."""
    states: dict[int, dict[str, Tensor]] = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = RNNModel(FEATURES, hidden=HIDDEN, layers=2, dropout=0.0, n_out=2)
        if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
            raise ValueError("native model parameter count changed")
        states[seed] = {name: value.detach().clone() for name, value in model.state_dict().items()}
    return states


def initialize_arm(alpha: float, state: dict[str, Tensor]) -> DurationCandidateRNN:
    """Load the shared native initialization into a parameter-free variant."""
    model = DurationCandidateRNN(alpha)
    model.load_state_dict(state, strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
        raise ValueError("candidate parameter count differs from registered H128 model")
    return model


def make_optimizer(model: DurationCandidateRNN) -> MuonAdamW:
    """Use Muon momentum .9 on four GRU matrices and fixed AdamW elsewhere."""
    groups = muon_param_groups(model, scope="rnn")
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assigned = [id(parameter) for group in groups for parameter in group["params"]]
    if len(assigned) != len(set(assigned)) or set(assigned) != trainable:
        raise ValueError("optimizer groups must exactly partition trainable parameters")
    muon_groups = [group for group in groups if group.get("muon")]
    names = {
        name
        for name, parameter in model.named_parameters()
        if any(id(parameter) == id(item) for group in muon_groups for item in group["params"])
    }
    if names != MUON_NAMES or len(muon_groups) != 1 or muon_groups[0].get("blocks") != 3:
        raise ValueError(f"unexpected recurrent Muon assignment: {names}")
    return MuonAdamW(
        groups,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=BETAS,
    )


def canonical_epoch_order(seed: int, epoch_index: int) -> np.ndarray:
    """Shuffle indices in the cohort's canonical 24-fit-role order."""
    if epoch_index not in range(EPOCHS):
        raise ValueError(f"invalid epoch index: {epoch_index}")
    order = np.arange(24)
    np.random.default_rng(seed + epoch_index).shuffle(order)
    return order


def optimizer_update(
    model: DurationCandidateRNN,
    optimizer: MuonAdamW,
    update_index: int,
    epoch_index: int,
    owner: str,
) -> tuple[bool, dict | None]:
    """Check raw gradients, clip globally, step once, and report fixed samples."""
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if len(gradients) != len(list(model.parameters())):
        raise ValueError("every trainable parameter must receive a gradient")
    if not all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients):
        raise ValueError(f"nonfinite raw gradient in {owner}")
    raw_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(gradient.detach(), 2) for gradient in gradients]), 2
    )
    if not torch.isfinite(raw_norm):
        raise ValueError(f"nonfinite global gradient norm in {owner}")
    clipped = bool(raw_norm.item() > CLIP_NORM)
    sample_index = update_index % UPDATES_PER_EPOCH
    sampled = sample_index in TELEMETRY_STEPS
    families = e474.parameter_families(model)
    gradient_rms = None
    parameter_before = None
    if sampled:
        gradient_rms = e474.family_rms(
            {
                family: [parameter.grad for parameter in parameters]
                for family, parameters in families.items()
            }
        )
        parameter_before = {
            family: [parameter.detach().clone() for parameter in parameters]
            for family, parameters in families.items()
        }
    nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM, error_if_nonfinite=True)
    optimizer.step()
    e470.ensure_finite_parameters(model, owner)
    telemetry = None
    if sampled:
        if gradient_rms is None or parameter_before is None:
            raise AssertionError("sampled telemetry snapshots are missing")
        updates = {
            family: [
                parameter.detach() - previous
                for parameter, previous in zip(
                    families[family], parameter_before[family], strict=True
                )
            ]
            for family in families
        }
        telemetry = {
            "global_update_index_zero_based": update_index,
            "epoch_index_zero_based": epoch_index,
            "epoch_step_index_zero_based": sample_index,
            "preclip_gradient_rms": gradient_rms,
            "parameter_update_rms": e474.family_rms(updates),
        }
    if not all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters()):
        raise ValueError(f"optimizer produced nonfinite parameters in {owner}")
    return clipped, telemetry


def fit_only_preflight(
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    nodes: list[dict],
    initial_states: dict[int, dict[str, Tensor]],
) -> list[dict]:
    """Make one actual B4×256 update for each seed and fixed shape."""
    if x.shape != (24, ROWS_PER_GROUP, FEATURES):
        raise ValueError(f"unexpected fit array: {x.shape}")
    if y.shape != (24, ROWS_PER_GROUP, 2) or need.shape != (24, ROWS_PER_GROUP):
        raise ValueError("unexpected labels or mask shape")
    if len(nodes) != 24 or any(node["role"] != "fit" for node in nodes):
        raise ValueError("preflight must contain only the canonical 24 fit roles")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("nonfinite fit-only cohort inputs")
    receipts: list[dict] = []
    for seed in SEEDS:
        ids = canonical_epoch_order(seed, 0)[:BATCH_SIZE].tolist()
        for alpha in ALPHAS:
            model = initialize_arm(alpha, initial_states[seed])
            optimizer = make_optimizer(model)
            xx = torch.from_numpy(x[ids, :CHUNK_ROWS]).float()
            yy = torch.from_numpy(y[ids, :CHUNK_ROWS]).float()
            mask = torch.from_numpy(need[ids, :CHUNK_ROWS])
            if not (torch.isfinite(xx).all() and torch.isfinite(yy).all()):
                raise ValueError("nonfinite raw preflight batch")
            optimizer.zero_grad(set_to_none=True)
            prediction, state = model(xx)
            if not (torch.isfinite(prediction).all() and torch.isfinite(state).all()):
                raise ValueError(f"nonfinite preflight forward at {seed}/alpha={alpha}")
            loss = e470.loss_for_batch(prediction, yy, mask)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite fit-only loss")
            loss.backward()
            before = [parameter.detach().clone() for parameter in model.parameters()]
            clipped, telemetry = optimizer_update(
                model, optimizer, 0, 0, f"preflight/{seed}/alpha={alpha}"
            )
            changed = any(
                not torch.equal(old, new)
                for old, new in zip(before, model.parameters(), strict=True)
            )
            if not changed:
                raise ValueError(f"preflight optimizer made no change: {seed}/alpha={alpha}")
            receipts.append(
                {
                    "seed": seed,
                    "alpha": alpha,
                    "batch": BATCH_SIZE,
                    "chunk_rows": CHUNK_ROWS,
                    "updates": 1,
                    "fit_row_groups": [nodes[index]["row_group"] for index in ids],
                    "loss": float(loss.detach()),
                    "clip_applied": clipped,
                    "parameters_changed": changed,
                    "parameters_finite": True,
                    "telemetry": telemetry,
                }
            )
    return receipts


def save_checkpoint(
    model: DurationCandidateRNN, output: Path, seed: int, alpha: float, epoch: int
) -> tuple[str, str]:
    """Write an immutable CPU checkpoint for one registered trajectory snapshot."""
    path = output / f"seed{seed}_alpha{alpha:.1f}_epoch{epoch}.pt"
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    torch.save(state, path)
    return path.name, e474.file_hash(path)


def run_trajectory(
    seed: int,
    alpha: float,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    nodes: list[dict],
    device: torch.device,
    output: Path,
    initial_state: dict[str, Tensor],
    active_precision: dict,
) -> dict:
    """Train four continuous epochs and save the three registered snapshots."""
    model = initialize_arm(alpha, initial_state).to(device).float()
    optimizer = make_optimizer(model)
    records: list[dict] = []
    global_step = 0
    total_clipped = 0
    started = time.monotonic()
    for epoch_index in range(EPOCHS):
        order = canonical_epoch_order(seed, epoch_index)
        epoch_groups: list[str] = []
        clipped_this_epoch = 0
        epoch_telemetry: list[dict] = []
        for batch_start in range(0, 24, BATCH_SIZE):
            ids = order[batch_start : batch_start + BATCH_SIZE].tolist()
            epoch_groups.extend(nodes[index]["row_group"] for index in ids)
            state: Tensor | None = None
            for position in range(0, ROWS_PER_GROUP, CHUNK_ROWS):
                xx = torch.from_numpy(x[ids, position : position + CHUNK_ROWS]).to(
                    device=device, dtype=torch.float32
                )
                yy = torch.from_numpy(y[ids, position : position + CHUNK_ROWS]).to(
                    device=device, dtype=torch.float32
                )
                mask = torch.from_numpy(need[ids, position : position + CHUNK_ROWS]).to(
                    device=device
                )
                if not (torch.isfinite(xx).all() and torch.isfinite(yy).all()):
                    raise ValueError(f"nonfinite raw fit batch at {seed}/{alpha}/{global_step}")
                optimizer.zero_grad(set_to_none=True)
                prediction, state = model(xx, state)
                if not (torch.isfinite(prediction).all() and torch.isfinite(state).all()):
                    raise ValueError(f"nonfinite forward at {seed}/{alpha}/{global_step}")
                loss = e470.loss_for_batch(prediction, yy, mask)
                if not torch.isfinite(loss):
                    raise ValueError(f"nonfinite loss at {seed}/{alpha}/{global_step}")
                loss.backward()
                clipped, telemetry = optimizer_update(
                    model,
                    optimizer,
                    global_step,
                    epoch_index,
                    f"fit/{seed}/alpha={alpha}",
                )
                clipped_this_epoch += int(clipped)
                if telemetry is not None:
                    telemetry.update(
                        {
                            "row_groups": [nodes[index]["row_group"] for index in ids],
                            "chunk_start": position,
                        }
                    )
                    epoch_telemetry.append(telemetry)
                state = state.detach()
                global_step += 1
        if len(epoch_groups) != 24 or len(set(epoch_groups)) != 24:
            raise ValueError("epoch order must contain each fit role exactly once")
        if global_step != (epoch_index + 1) * UPDATES_PER_EPOCH:
            raise ValueError("unexpected continuous trajectory update count")
        total_clipped += clipped_this_epoch
        snapshot: dict | None = None
        if epoch_index + 1 in SNAPSHOT_EPOCHS:
            filename, digest = save_checkpoint(model, output, seed, alpha, epoch_index + 1)
            snapshot = {"checkpoint": filename, "sha256": digest}
        records.append(
            {
                "epoch": epoch_index + 1,
                "updates_this_epoch": UPDATES_PER_EPOCH,
                "total_updates": global_step,
                "fit_row_groups_in_order": epoch_groups,
                "clipped_updates": clipped_this_epoch,
                "cumulative_clipped_updates": total_clipped,
                "telemetry": epoch_telemetry,
                "snapshot": snapshot,
            }
        )
        write_progress_manifest(
            output, seed, alpha, records, total_clipped, global_step, active_precision
        )
    if global_step != UPDATES_PER_TRAJECTORY or len(records) != EPOCHS:
        raise ValueError("trajectory did not complete all four epochs")
    return {
        "seed": seed,
        "alpha": alpha,
        "status": "finished",
        "parameters": EXPECTED_PARAMETERS,
        "updates": global_step,
        "clipped_updates": total_clipped,
        "fit_seconds": time.monotonic() - started,
        "epochs": records,
    }


def write_progress_manifest(
    output: Path,
    seed: int,
    alpha: float,
    epochs: list[dict],
    clipped: int,
    updates: int,
    active_precision: dict,
) -> None:
    """Persist per-trajectory progress after each completed epoch."""
    path = output / f"seed{seed}_alpha{alpha:.1f}_progress.json"
    payload = {
        "seed": seed,
        "alpha": alpha,
        "completed_epochs": len(epochs),
        "updates": updates,
        "clipped_updates": clipped,
        "active_precision": active_precision,
        "epochs": epochs,
    }
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    """Run six fit-only preflight updates or all six continuous trajectories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-real", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    device = torch.device("cpu" if args.preflight_real else args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        configure_torch()
    hashes = parent_source_hashes()
    initial_states = make_initial_states()
    cohort, x, y, need, _, nodes = e470.load_cohort(include_test=False)
    if len(nodes) != 24 or any(node["role"] != "fit" for node in nodes):
        raise ValueError("E485 can load only the 24 fit-role row groups")
    if x.shape != (24, ROWS_PER_GROUP, FEATURES):
        raise ValueError(f"unexpected fit cohort shape: {x.shape}")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("nonfinite fit cohort data")
    result = {
        "kind": "E485 recurrent activation duration screen",
        "status": "preflight" if args.preflight_real else "fitting",
        "fit_only": True,
        "test_rows_loaded": False,
        "seeds": list(SEEDS),
        "alphas": list(ALPHAS),
        "hidden": HIDDEN,
        "parameters_per_trajectory": EXPECTED_PARAMETERS,
        "epochs": EPOCHS,
        "snapshots_per_trajectory": list(SNAPSHOT_EPOCHS),
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "batch_size": BATCH_SIZE,
        "chunk_rows": CHUNK_ROWS,
        "candidate": "u=tanh(v); n=u+alpha*sin(pi*u); h'=(1-z)*n+z*h",
        "optimizer": {
            "name": "MuonAdamW",
            "muon_names": sorted(MUON_NAMES),
            "muon_blocks": 3,
            "muon_beta1": 0.9,
            "adam_betas": list(BETAS),
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "schedule": "constant",
            "global_clip_norm": CLIP_NORM,
            "state_continues_across_epochs": True,
            "cuda_muon_orthogonalization": "existing wnn.optim BF16 implementation",
        },
        "shuffle": "canonical np.arange(24), default_rng(seed+epoch_index).shuffle",
        "cohort_sha256": e474.file_hash(e470.COHORT_PATH),
        "dataset_sha256": e474.DATASET_SHA256,
        "parent_source_sha256": hashes,
        "design_sha256": DESIGN_SHA256,
        "blind_critic_sha256": CRITIC_SHA256,
        "probe_update_gates_sha256": UPDATE_GATES_SHA256,
        "source_sha256": {
            name: e474.file_hash(HERE / name)
            for name in (
                "fit_duration.py",
                "tests.py",
                "gpu_probe.py",
                "GPU_PROBE_UPDATE_GATES.md",
            )
            if (HERE / name).is_file()
        },
        "cohort_manifest": cohort,
        "trajectories": [],
    }
    (args.output / "fit_only_cohort.json").write_text(json.dumps(cohort, indent=2))
    if args.preflight_real:
        result["preflight_arms"] = fit_only_preflight(x, y, need, nodes, initial_states)
        result["status"] = "preflight_passed"
    else:
        with e474.full_ieee_float32():
            result["active_precision"] = e474.precision_metadata()
            for seed in SEEDS:
                for alpha in ALPHAS:
                    trajectory = run_trajectory(
                        seed,
                        alpha,
                        x,
                        y,
                        need,
                        nodes,
                        device,
                        args.output,
                        initial_states[seed],
                        result["active_precision"],
                    )
                    result["trajectories"].append(trajectory)
                    (args.output / "fit_manifest.json").write_text(json.dumps(result, indent=2))
        if len(result["trajectories"]) != 6:
            raise ValueError("all six registered trajectories must finish")
        result["status"] = "all_six_trajectories_complete_test_unread"
    (args.output / "fit_manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
