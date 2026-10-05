"""Fit the preregistered E483 isolated recurrent-Muon-momentum comparison."""

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
from torch import Tensor, nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
E481_DIR = ROOT / "research/campaign_2026-10-05/recurrent_activation_curvature"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E470_DIR))
sys.path.insert(0, str(E474_DIR))
sys.path.insert(0, str(E481_DIR))

import fit_activation_optimizer as e470  # noqa: E402
import fit_activation_placement as e474  # noqa: E402
import fit_recurrent_curvature as e481  # noqa: E402

from wnn.nn import RNNModel  # noqa: E402
from wnn.optim import MuonAdamW, muon_param_groups  # noqa: E402
from wnn.resources import configure_torch  # noqa: E402

SEEDS = (271831, 271832, 271833)
SHAPES = ("baseline", "sine")
MOMENTA = (0.8, 0.9, 0.95)
ARMS = tuple((seed, shape, beta) for seed in SEEDS for shape in SHAPES for beta in MOMENTA)
FEATURES, HIDDEN, CHUNK_ROWS, UPDATES = 42, 64, 256, 474
N_PARAMETERS = 61_058
TELEMETRY_STEPS = (0, 46, 236, 473)
MUON_NAMES = {
    "gru.weight_ih_l0",
    "gru.weight_hh_l0",
    "gru.weight_ih_l1",
    "gru.weight_hh_l1",
}
LR, WD, BETAS = 0.001, 0.01, (0.9, 0.999)
DESIGN_SHA256 = "fa5aa9ec2a7f897f7ed31f7a99d9ee0b8a681cce73146ebd367d729f6a1e8288"
SPEC_CRITIC_SHA256 = "9cd04e02fc0f4d5a0dd324d531f8d38d6f48261e3d1f87fd4aeddfc754da0c28"
E481_SHA256 = "3d97d2d07b116b3cf04b365c33e3d3caed2c96c35108bd28f3a6ff63b54d65cc"
E470_SHA256 = "871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9"
E474_SHA256 = "40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068"
DATASET_SHA256 = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
COHORT_SHA256 = "068103c71663bc445aab943e314c5be1d9a094f7f83456fdb406fbdc1012a6f2"
WNN_NN_SHA256 = "388af04570f0a3944159bf86c1a234dc55c78b951804c38ef59fa314f92f3db3"
WNN_OPTIM_SHA256 = "620569a752fb611adf69976f1dbed1b7882e641672c58175005f5d3f819d2991"
WNN_RESOURCES_SHA256 = "06a624ad963dfc2b0c75ce717278de5f1bb85f48518a99e519b697a371d4cf09"


def sha256_file(path: Path) -> str:
    """Hash a source or artifact in bounded-size blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_parent_sources() -> dict[str, str]:
    """Fail closed unless the pinned E470/E474/E481 training sources match."""
    expected = {
        "e470": (E470_DIR / "fit_activation_optimizer.py", E470_SHA256),
        "e474": (E474_DIR / "fit_activation_placement.py", E474_SHA256),
        "e481": (E481_DIR / "fit_recurrent_curvature.py", E481_SHA256),
        "wnn.nn": (ROOT / "src/wnn/nn.py", WNN_NN_SHA256),
        "wnn.optim": (ROOT / "src/wnn/optim.py", WNN_OPTIM_SHA256),
        "wnn.resources": (ROOT / "src/wnn/resources.py", WNN_RESOURCES_SHA256),
    }
    actual = {name: sha256_file(path) for name, (path, _digest) in expected.items()}
    for name, (_path, digest) in expected.items():
        if actual[name] != digest:
            raise ValueError(f"frozen parent source changed: {name}={actual[name]}")
    if sha256_file(HERE / "E483_DESIGN.md") != DESIGN_SHA256:
        raise ValueError("registered E483 design changed")
    if sha256_file(HERE / "SPEC_CRITIC.md") != SPEC_CRITIC_SHA256:
        raise ValueError("independent E483 design critique changed")
    if sha256_file(e470.COHORT_PATH) != COHORT_SHA256:
        raise ValueError("fit cohort manifest changed")
    if sha256_file(e470.DATA_PATH) != DATASET_SHA256:
        raise ValueError("frozen training dataset bytes changed")
    return actual


def make_seed_states() -> dict[int, dict[str, Tensor]]:
    """Create one common E481 initialization per new E483 seed."""
    states: dict[int, dict[str, Tensor]] = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = RNNModel(FEATURES, hidden=HIDDEN, layers=2, dropout=0.0, n_out=2)
        if sum(parameter.numel() for parameter in model.parameters()) != N_PARAMETERS:
            raise ValueError(f"E481 architecture changed at initialization seed {seed}")
        states[seed] = {name: value.detach().clone() for name, value in model.state_dict().items()}
    hashes = {seed: state_hash(state) for seed, state in states.items()}
    if len(set(hashes.values())) != len(SEEDS):
        raise ValueError("the three registered seed initializations are not distinct")
    return states


def state_hash(state: dict[str, Tensor]) -> str:
    """Hash a state dict by stable key, dtype, shape and tensor bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def optimizer_group_receipt(
    model: nn.Module, optimizer: torch.optim.Optimizer, beta: float
) -> list[dict]:
    """Validate and describe the exact Muon/AdamW parameter partition."""
    named = {id(parameter): name for name, parameter in model.named_parameters()}
    expected_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    actual_ids = [
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    ]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise ValueError("optimizer groups are not an exact disjoint trainable-parameter partition")
    receipts: list[dict] = []
    muon_groups = 0
    for group in optimizer.param_groups:
        names = sorted(named[id(parameter)] for parameter in group["params"])
        is_muon = bool(group.get("muon", False))
        expected_betas = (beta, BETAS[1]) if is_muon else BETAS
        if tuple(group["betas"]) != expected_betas:
            raise ValueError(f"effective betas leaked or changed: {names[:2]} {group['betas']}")
        if group["lr"] != LR or group["weight_decay"] != WD:
            raise ValueError("optimizer LR or weight decay differs from E483")
        if is_muon:
            muon_groups += 1
            if set(names) != MUON_NAMES or group.get("blocks") != 3:
                raise ValueError(f"Muon selection must be exactly four GRU matrices: {names}")
            if group.get("ns_steps") != 5 or group.get("rms") != 0.2:
                raise ValueError("Muon Newton–Schulz steps or scale changed")
        receipts.append(
            {
                "muon": is_muon,
                "names": names,
                "betas": list(group["betas"]),
                "lr": group["lr"],
                "weight_decay": group["weight_decay"],
                "blocks": group.get("blocks", 1),
                "ns_steps": group.get("ns_steps"),
                "rms": group.get("rms"),
            }
        )
    if muon_groups != 1:
        raise ValueError(f"expected one recurrent Muon group, got {muon_groups}")
    return receipts


def make_optimizer(model: nn.Module, beta: float) -> torch.optim.Optimizer:
    """Construct MuonAdamW with beta changed only on the recurrent-matrix group."""
    if beta not in MOMENTA:
        raise ValueError(f"unregistered recurrent Muon beta: {beta}")
    groups = muon_param_groups(model, scope="rnn")
    if len(groups) != 2:
        raise ValueError(f"expected exactly two optimizer families, got {len(groups)}")
    for group in groups:
        if group.get("muon", False):
            group["betas"] = (beta, BETAS[1])
        else:
            group["betas"] = BETAS
    optimizer = MuonAdamW(groups, lr=LR, weight_decay=WD, betas=BETAS)
    optimizer_group_receipt(model, optimizer, beta)
    return optimizer


def initialize_arm(shape: str, initial_state: dict[str, Tensor]) -> nn.Module:
    """Load an E481 registered baseline/sine candidate from a common state dict."""
    if shape not in SHAPES:
        raise ValueError(f"unregistered E483 shape: {shape}")
    model = e481.initialize_arm(shape, initial_state)
    if sum(parameter.numel() for parameter in model.parameters()) != N_PARAMETERS:
        raise ValueError("E481 model parameter count changed")
    return model


def apply_optimizer_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    families: dict[str, list[nn.Parameter]],
    update_index: int,
    owner: str,
) -> dict:
    """Apply frozen global clipping plus MuonAdamW and return exact preclip norm."""
    sampled = update_index in TELEMETRY_STEPS
    gradient_rms = None
    before = None
    if sampled:
        gradients = {
            family: [parameter.grad for parameter in parameters if parameter.grad is not None]
            for family, parameters in families.items()
        }
        if any(len(gradients[key]) != len(families[key]) for key in families):
            raise ValueError(f"missing sampled gradient for {owner}")
        gradient_rms = e474.family_rms(gradients)
        before = {
            family: [parameter.detach().clone() for parameter in parameters]
            for family, parameters in families.items()
        }
    preclip_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    norm_value = float(preclip_norm.detach().cpu())
    if not math.isfinite(norm_value):
        raise ValueError(f"nonfinite preclip gradient norm: {owner}")
    optimizer.step()
    e470.ensure_finite_parameters(model, owner)
    telemetry: dict[str, object] = {
        "update_index_zero_based": update_index,
        "preclip_gradient_norm": norm_value,
        "clipped": norm_value > 1.0,
    }
    if not sampled:
        return telemetry
    if before is None or gradient_rms is None:
        raise AssertionError("sampled telemetry was not captured")
    delta = {
        family: [
            parameter.detach() - old
            for parameter, old in zip(families[family], before[family], strict=True)
        ]
        for family in families
    }
    telemetry["preclip_gradient_rms"] = gradient_rms
    telemetry["parameter_update_rms"] = e474.family_rms(delta)
    return telemetry


def one_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    ids: list[int],
    device: torch.device,
    owner: str,
) -> tuple[float, dict]:
    """Run one fit-only B4×256 update with finite checks."""
    xx = torch.from_numpy(x[ids, :CHUNK_ROWS]).to(device=device, dtype=torch.float32)
    yy = torch.from_numpy(y[ids, :CHUNK_ROWS]).to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(need[ids, :CHUNK_ROWS]).to(device=device)
    if not torch.isfinite(xx).all() or not torch.isfinite(yy).all():
        raise ValueError(f"nonfinite raw fit input: {owner}")
    optimizer.zero_grad(set_to_none=True)
    prediction, state = model(xx)
    if not torch.isfinite(prediction).all() or not torch.isfinite(state).all():
        raise ValueError(f"nonfinite prediction/state: {owner}")
    loss = e470.loss_for_batch(prediction, yy, mask)
    if not torch.isfinite(loss):
        raise ValueError(f"nonfinite loss: {owner}")
    loss.backward()
    telemetry = apply_optimizer_update(model, optimizer, e474.parameter_families(model), 0, owner)
    return float(loss.detach().cpu()), telemetry


def load_fit_cohort() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Load and validate only the sealed 24-component fit cohort."""
    manifest, x, y, need, _score, nodes = e470.load_cohort(include_test=False)
    if (
        x.shape != (24, 20_000, FEATURES)
        or y.shape != (24, 20_000, 2)
        or need.shape != (24, 20_000)
        or len(nodes) != 24
        or any(node["role"] != "fit" for node in nodes)
    ):
        raise ValueError("loader did not return exactly 24 fit-only groups")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite fit-only arrays")
    return manifest, x, y, need, nodes


def run_cpu_preflight(
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    nodes: list[dict],
    states: dict[int, dict[str, Tensor]],
) -> list[dict]:
    """Make one actual fit-only CPU B4×256 update in each registered cell."""
    receipts = []
    for seed, shape, beta in ARMS:
        order = np.arange(24)
        np.random.default_rng(seed).shuffle(order)
        ids = order[:4].tolist()
        torch.manual_seed(seed)
        model = initialize_arm(shape, states[seed])
        optimizer = make_optimizer(model, beta)
        group_receipt = optimizer_group_receipt(model, optimizer, beta)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        loss, telemetry = one_update(
            model,
            optimizer,
            x,
            y,
            need,
            ids,
            torch.device("cpu"),
            f"{seed}/{shape}/muon_beta{beta}",
        )
        changed = [
            not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True)
        ]
        if not any(changed) or not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise ValueError(f"CPU preflight update failed: {seed}/{shape}/{beta}")
        receipts.append(
            {
                "seed": seed,
                "shape": shape,
                "muon_beta1": beta,
                "row_groups": [nodes[int(index)]["row_group"] for index in ids],
                "batch": 4,
                "chunk_rows": CHUNK_ROWS,
                "updates": 1,
                "loss": loss,
                "changed_parameter_tensors": sum(changed),
                "parameters": N_PARAMETERS,
                "optimizer_groups": group_receipt,
                "telemetry_step0": telemetry,
            }
        )
    if len(receipts) != 18:
        raise ValueError("CPU preflight did not cover all eighteen cells")
    return receipts


def run_fit(
    seed: int,
    shape: str,
    beta: float,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    nodes: list[dict],
    initial_state: dict[str, Tensor],
    device: torch.device,
    output: Path,
) -> dict:
    """Run one fixed 474-update fit arm on fit-role sequences only."""
    order = np.arange(24)
    np.random.default_rng(seed).shuffle(order)
    torch.manual_seed(seed)
    model = initialize_arm(shape, initial_state).to(device=device, dtype=torch.float32)
    optimizer = make_optimizer(model, beta)
    groups = optimizer_group_receipt(model, optimizer, beta)
    families = e474.parameter_families(model)
    telemetry: list[dict] = []
    clipping_count = 0
    update = 0
    started = time.monotonic()
    for batch_start in range(0, 24, 4):
        ids = order[batch_start : batch_start + 4].tolist()
        state = None
        for row_start in range(0, 20_000, CHUNK_ROWS):
            xx = torch.from_numpy(x[ids, row_start : row_start + CHUNK_ROWS]).to(
                device=device, dtype=torch.float32
            )
            yy = torch.from_numpy(y[ids, row_start : row_start + CHUNK_ROWS]).to(
                device=device, dtype=torch.float32
            )
            mask = torch.from_numpy(need[ids, row_start : row_start + CHUNK_ROWS]).to(device)
            if not torch.isfinite(xx).all() or not torch.isfinite(yy).all():
                raise ValueError(f"nonfinite fit input at {seed}/{shape}/{beta}/{row_start}")
            optimizer.zero_grad(set_to_none=True)
            prediction, state = model(xx, state)
            if not torch.isfinite(prediction).all() or not torch.isfinite(state).all():
                raise ValueError(f"nonfinite prediction/state at {seed}/{shape}/{beta}/{row_start}")
            loss = e470.loss_for_batch(prediction, yy, mask)
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite loss at {seed}/{shape}/{beta}/{row_start}")
            loss.backward()
            stats = apply_optimizer_update(
                model, optimizer, families, update, f"{seed}/{shape}/muon_beta{beta}"
            )
            clipping_count += int(stats["clipped"])
            if update in TELEMETRY_STEPS:
                stats.update(
                    {
                        "seed": seed,
                        "shape": shape,
                        "muon_beta1": beta,
                        "fit_row_groups": [nodes[index]["row_group"] for index in ids],
                        "chunk_start": row_start,
                        "loss": float(loss.detach().cpu()),
                        "state_abs_max": float(state.detach().abs().max().cpu()),
                    }
                )
                telemetry.append(stats)
            state = state.detach()
            update += 1
    if (
        update != UPDATES
        or tuple(item["update_index_zero_based"] for item in telemetry) != TELEMETRY_STEPS
    ):
        raise ValueError(f"wrong update/telemetry cadence: {seed}/{shape}/{beta}/{update}")
    if not 0 <= clipping_count <= UPDATES:
        raise ValueError(f"invalid clip count: {clipping_count}")
    checkpoint = output / f"seed{seed}_{shape}_momentum{beta}.pt"
    torch.save(model.cpu().state_dict(), checkpoint)
    return {
        "seed": seed,
        "shape": shape,
        "muon_beta1": beta,
        "updates": update,
        "clipped_updates": clipping_count,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "fit_seconds": time.monotonic() - started,
        "fit_row_groups": [nodes[int(index)]["row_group"] for index in order],
        "optimizer_groups": groups,
        "telemetry_steps_zero_based": list(TELEMETRY_STEPS),
        "telemetry": telemetry,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def run_probe(
    output: Path,
    manifest: dict,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    nodes: list[dict],
    states: dict[int, dict[str, Tensor]],
) -> dict:
    """Run all 18 actual B4×256 CUDA updates plus frozen parity gates."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite probe output: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA probe requested without CUDA")
    configure_torch()
    torch.set_num_threads(1)
    check_parent_sources()
    torch.cuda.reset_peak_memory_stats()
    arms = []
    with e474.full_ieee_float32():
        parity = run_native_and_optimizer_parity(torch.device("cuda"), states[SEEDS[0]])
        for seed, shape, beta in ARMS:
            order = np.arange(24)
            np.random.default_rng(seed).shuffle(order)
            ids = order[:4].tolist()
            torch.manual_seed(seed)
            model = initialize_arm(shape, states[seed]).to(device="cuda", dtype=torch.float32)
            optimizer = make_optimizer(model, beta)
            groups = optimizer_group_receipt(model, optimizer, beta)
            before = [parameter.detach().clone() for parameter in model.parameters()]
            loss, telemetry = one_update(
                model,
                optimizer,
                x,
                y,
                need,
                ids,
                torch.device("cuda"),
                f"probe/{seed}/{shape}/{beta}",
            )
            changed = [
                not torch.equal(old, p) for old, p in zip(before, model.parameters(), strict=True)
            ]
            if not any(changed) or not all(
                torch.isfinite(p).all().item() for p in model.parameters()
            ):
                raise ValueError(
                    f"CUDA probe cell failed finite/change gate: {seed}/{shape}/{beta}"
                )
            arms.append(
                {
                    "seed": seed,
                    "shape": shape,
                    "muon_beta1": beta,
                    "row_groups": [nodes[index]["row_group"] for index in ids],
                    "batch": 4,
                    "chunk_rows": CHUNK_ROWS,
                    "updates": 1,
                    "loss": loss,
                    "changed_parameter_tensors": sum(changed),
                    "parameters": N_PARAMETERS,
                    "optimizer_groups": groups,
                    "telemetry_step0": telemetry,
                }
            )
            del model, optimizer
            torch.cuda.synchronize()
    result = {
        "kind": "E483 all-cell actual CUDA update probe",
        "status": "all_18_cells_finite_changed",
        "fit_only": True,
        "test_rows_loaded": False,
        "dataset_sha256": DATASET_SHA256,
        "cohort_sha256": sha256_file(e470.COHORT_PATH),
        "parent_source_sha256": check_parent_sources(),
        "source_sha256": {
            path.name: sha256_file(path)
            for path in (HERE / "fit_momentum.py", HERE / "tests.py", HERE / "gpu_probe.py")
        },
        "fit_cohort_manifest_sha256": sha256_file(e470.COHORT_PATH),
        "fit_row_group_count": len(nodes),
        "fit_manifest_hash": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        "arms": arms,
        "precision": e474.precision_metadata(),
        "device": torch.cuda.get_device_name(),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "torch_version": torch.__version__,
        "parity_gates": parity,
    }
    if len(arms) != 18:
        raise ValueError("CUDA probe omitted one or more cells")
    output.mkdir(parents=True)
    (output / "probe.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def run_native_and_optimizer_parity(device: torch.device, state: dict[str, Tensor]) -> dict:
    """Check alpha-zero native GRU parity and exact beta-0.9 clipped-step parity."""
    torch.manual_seed(8741)
    custom = initialize_arm("baseline", state).to(device=device, dtype=torch.float32)
    native = RNNModel(FEATURES, hidden=HIDDEN, layers=2, dropout=0.0, n_out=2).to(
        device=device, dtype=torch.float32
    )
    native.load_state_dict(state, strict=True)
    x_custom = torch.randn(2, 37, FEATURES, device=device, requires_grad=True)
    x_native = x_custom.detach().clone().requires_grad_(True)
    pred_custom, final_custom = custom(x_custom)
    pred_native, final_native = native(x_native)
    torch.testing.assert_close(pred_custom, pred_native, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(final_custom, final_native, rtol=2e-4, atol=2e-5)
    (pred_custom.square().mean() + final_custom.square().mean()).backward()
    (pred_native.square().mean() + final_native.square().mean()).backward()
    torch.testing.assert_close(x_custom.grad, x_native.grad, rtol=5e-4, atol=5e-5)
    for (name_a, parameter_a), (name_b, parameter_b) in zip(
        custom.named_parameters(), native.named_parameters(), strict=True
    ):
        if name_a != name_b:
            raise ValueError("native and scripted model parameter order differs")
        torch.testing.assert_close(parameter_a.grad, parameter_b.grad, rtol=5e-4, atol=5e-5)

    left = initialize_arm("baseline", state).to(device=device, dtype=torch.float32)
    right = initialize_arm("baseline", state).to(device=device, dtype=torch.float32)
    left_optimizer = make_optimizer(left, 0.9)
    right_optimizer = make_optimizer(right, 0.9)
    sample = torch.randn(2, 37, FEATURES, device=device)
    target = torch.randn(2, 37, 2, device=device)
    left_optimizer.zero_grad(set_to_none=True)
    right_optimizer.zero_grad(set_to_none=True)
    prediction, final = left(sample)
    loss = torch.nn.functional.mse_loss(prediction, target) + 0.01 * final.square().mean()
    loss.backward()
    for left_parameter, right_parameter in zip(left.parameters(), right.parameters(), strict=True):
        right_parameter.grad = left_parameter.grad.detach().clone()
    local = apply_optimizer_update(left, left_optimizer, {}, 1, "parity/local")
    e474.apply_optimizer_update(right, right_optimizer, {}, 1, "parity/frozen")
    if any(
        not torch.equal(a, b) for a, b in zip(left.parameters(), right.parameters(), strict=True)
    ):
        raise ValueError("local clipped MuonAdamW update differs from frozen E474 helper")
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "device": device.type,
        "native_gru": "output_state_gradient_close",
        "optimizer_step": "exact_beta_0.9_clipped_muonadamw_parameter_match",
        "preclip_norm": local["preclip_gradient_norm"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--preflight-real", action="store_true")
    operation.add_argument("--run-fits", action="store_true")
    operation.add_argument("--gpu-probe", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    if args.preflight_real and args.device != "cpu":
        raise ValueError("real-data CPU preflight must run on CPU")
    if args.run_fits and args.device != "cuda":
        raise ValueError("full fits require separately admitted CUDA")
    torch.set_num_threads(1)
    parents = check_parent_sources()
    states = make_seed_states()
    manifest, x, y, need, nodes = load_fit_cohort()
    if args.preflight_real:
        result = {
            "kind": "E483 fit-only CPU one-update preflight",
            "status": "cpu_fit_only_one_update_per_arm_passed_test_unread",
            "design_sha256": DESIGN_SHA256,
            "blind_critic_sha256": SPEC_CRITIC_SHA256,
            "fit_only": True,
            "test_rows_loaded": False,
            "seeds": list(SEEDS),
            "shapes": list(SHAPES),
            "muon_beta1": list(MOMENTA),
            "planned_fit_count": len(ARMS),
            "batch": 4,
            "chunk_rows": CHUNK_ROWS,
            "updates_per_full_fit": UPDATES,
            "parameters_per_arm": N_PARAMETERS,
            "parent_source_sha256": parents,
            "source_sha256": {
                p.name: sha256_file(p)
                for p in (HERE / "fit_momentum.py", HERE / "tests.py", HERE / "gpu_probe.py")
            },
            "dataset_sha256": DATASET_SHA256,
            "cohort_sha256": sha256_file(e470.COHORT_PATH),
            "initial_state_sha256": {
                str(seed): state_hash(state) for seed, state in states.items()
            },
            "fit_manifest": {
                **manifest,
                "cohort": [node for node in manifest["cohort"] if node["role"] == "fit"],
            },
            "arms": run_cpu_preflight(x, y, need, nodes, states),
        }
    elif args.gpu_probe:
        result = run_probe(args.output, manifest, x, y, need, nodes, states)
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
        return
    else:
        configure_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        args.output.mkdir(parents=True)
        with e474.full_ieee_float32():
            arms = [
                run_fit(
                    seed,
                    shape,
                    beta,
                    x,
                    y,
                    need,
                    nodes,
                    states[seed],
                    torch.device("cuda"),
                    args.output,
                )
                for seed, shape, beta in ARMS
            ]
        result = {
            "kind": "E483 full fit matrix",
            "status": "all_18_fit_arms_complete_test_unread",
            "design_sha256": DESIGN_SHA256,
            "blind_critic_sha256": SPEC_CRITIC_SHA256,
            "fit_only": True,
            "test_rows_loaded": False,
            "seeds": list(SEEDS),
            "shapes": list(SHAPES),
            "muon_beta1": list(MOMENTA),
            "planned_fit_count": len(ARMS),
            "updates_per_fit": UPDATES,
            "dataset_sha256": DATASET_SHA256,
            "cohort_sha256": sha256_file(e470.COHORT_PATH),
            "fit_only_cohort": {
                **manifest,
                "cohort": [node for node in manifest["cohort"] if node["role"] == "fit"],
            },
            "parent_source_sha256": parents,
            "source_sha256": {
                p.name: sha256_file(p)
                for p in (HERE / "fit_momentum.py", HERE / "tests.py", HERE / "gpu_probe.py")
            },
            "initial_state_sha256": {
                str(seed): state_hash(state) for seed, state in states.items()
            },
            "arms": arms,
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "fit_manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False))
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
