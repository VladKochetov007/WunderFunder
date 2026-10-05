"""Fit the preregistered E488 continuous three-epoch context-mixture trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
E477_DIR = ROOT / "research/campaign_2026-10-05/activation_head_mixture"
E484_DIR = ROOT / "research/campaign_2026-10-05/activation_context_mixture"
sys.path[:0] = [str(E470_DIR), str(E474_DIR), str(E477_DIR), str(E484_DIR), str(ROOT / "src")]

import fit_activation_optimizer as e470  # noqa: E402
import fit_activation_placement as e474  # noqa: E402
import fit_context_mixture_v2 as e484  # noqa: E402

from wnn.resources import configure_torch  # noqa: E402

SEEDS = (271839, 271840, 271841)
ARMS = ("gelu", "fixed_mix", "static_mix", "recurrent_mix")
CONTRASTS_PER_SEED = (
    "recurrent_1 - static_1",
    "recurrent_1 - fixed_1",
    "static_1 - fixed_1",
    "fixed_1 - gelu_1",
    "recurrent_3 - static_3",
    "recurrent_3 - fixed_3",
    "static_3 - fixed_3",
    "fixed_3 - gelu_3",
    "(recurrent_3 - static_3) - (recurrent_1 - static_1)",
)
EPOCHS = 3
UPDATES_PER_EPOCH = 474
TOTAL_UPDATES = EPOCHS * UPDATES_PER_EPOCH
CHUNK_ROWS = 256
TELEMETRY_STEPS = (0, 46, 236, 473, 947, 1421)
CLIP_NORM, LR, WEIGHT_DECAY, BETAS = 1.0, 0.001, 0.01, (0.9, 0.999)
BASE_PARAMETERS, GATED_PARAMETERS = 61_058, 61_122
DESIGN_SHA256 = "3764a095504af6da66a4faa908a47ff5dd3f94266249d5bea7c07cf611148824"
CRITIC_SHA256 = "ccc5caba6eef86b928e31cc78ce65602ad528a6549b0017a3d91ffcf1a3aa3e1"
CLARIFICATIONS_SHA256 = "d382f867bd8c889c5c63f40228ba1133e0f52d2a9e396860fbf15009fced0a9c"
E484_ACTOR_SHA256 = "2dec963a966ba30e1dc3deed40472fd2d0129a2d24fa07585a3a7ed4e2e1dcbc"
E470_SHA256 = "871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9"
E474_SHA256 = "40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068"
E477_SHA256 = "2d4b1b103dd31912fea7ab24a63e6ce176e532dca8d0dfa2cdb32aebd9e567d7"
WNN_NN_SHA256 = "388af04570f0a3944159bf86c1a234dc55c78b951804c38ef59fa314f92f3db3"
WNN_RESOURCES_SHA256 = "06a624ad963dfc2b0c75ce717278de5f1bb85f48518a99e519b697a371d4cf09"
DATASET_SHA256 = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
COHORT_SHA256 = "068103c71663bc445aab943e314c5be1d9a094f7f83456fdb406fbdc1012a6f2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def state_sha256(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def check_sources() -> dict[str, str]:
    frozen = {
        "e470": (E470_DIR / "fit_activation_optimizer.py", E470_SHA256),
        "e474": (E474_DIR / "fit_activation_placement.py", E474_SHA256),
        "e477": (E477_DIR / "fit_activation_head_mixture.py", E477_SHA256),
        "e484_actor": (E484_DIR / "fit_context_mixture_v2.py", E484_ACTOR_SHA256),
        "wnn.nn": (ROOT / "src/wnn/nn.py", WNN_NN_SHA256),
        "wnn.resources": (ROOT / "src/wnn/resources.py", WNN_RESOURCES_SHA256),
    }
    actual = {name: sha256_file(path) for name, (path, _expected) in frozen.items()}
    for name, (_path, expected) in frozen.items():
        if not expected or actual[name] != expected:
            raise ValueError(f"frozen E488 parent source mismatch: {name}={actual[name]}")
    docs = {
        "design": (HERE / "E488_DESIGN.md", DESIGN_SHA256),
        "critic": (HERE / "SPEC_CRITIC.md", CRITIC_SHA256),
        "clarifications": (HERE / "CONTROLLER_CLARIFICATIONS.md", CLARIFICATIONS_SHA256),
    }
    for name, (path, expected) in docs.items():
        if not expected or sha256_file(path) != expected:
            raise ValueError(f"registered E488 {name} identity mismatch")
    if (
        sha256_file(e470.DATA_PATH) != DATASET_SHA256
        or sha256_file(e470.COHORT_PATH) != COHORT_SHA256
    ):
        raise ValueError("frozen E488 dataset/cohort identity mismatch")
    return actual


def make_initial_states() -> dict[int, dict[str, Tensor]]:
    states: dict[int, dict[str, Tensor]] = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        base = e474.PlacementRNN("gelu", "gelu")
        if sum(parameter.numel() for parameter in base.parameters()) != BASE_PARAMETERS:
            raise ValueError("E474 common trunk parameter count changed")
        states[seed] = {name: value.detach().clone() for name, value in base.state_dict().items()}
    if len({state_sha256(state) for state in states.values()}) != len(SEEDS):
        raise ValueError("E488 fresh seed initializations collided")
    return states


def load_fit_data() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    manifest, x, y, need, _score, nodes = e470.load_cohort(include_test=False)
    if x.shape != (24, 20_000, 42) or y.shape != (24, 20_000, 2) or need.shape != (24, 20_000):
        raise ValueError("E488 fit loader must return exact 24×20k fit-only tensors")
    if len(nodes) != 24 or any(node["role"] != "fit" for node in nodes):
        raise ValueError("E488 loader returned a non-fit role")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("E488 fit cohort has nonfinite values")
    normalization = manifest.get("normalization", {})
    mean = np.asarray(normalization.get("fit_mean"), dtype=np.float32)
    scale = np.asarray(normalization.get("fit_scale"), dtype=np.float32)
    if (
        mean.shape != (42,)
        or scale.shape != (42,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale < 0.001)
    ):
        raise ValueError("E488 inherited fit-only normalization is invalid")
    return manifest, x, y, need, nodes


def make_arm(arm: str, initial: dict[str, Tensor]):
    if arm not in ARMS:
        raise ValueError(f"unregistered E488 arm {arm}")
    model = e484.initialize_arm(arm, initial)
    expected = GATED_PARAMETERS if arm in ("static_mix", "recurrent_mix") else BASE_PARAMETERS
    if sum(parameter.numel() for parameter in model.parameters()) != expected:
        raise ValueError(f"wrong E488 {arm} parameter count")
    e484.check_common_parameter_bytes(model, initial)
    return model


def epoch_order(seed: int, epoch_index: int) -> np.ndarray:
    if epoch_index not in (0, 1, 2):
        raise ValueError("epoch index must be zero-based 0, 1, or 2")
    order = np.arange(24)
    np.random.default_rng(seed + epoch_index).shuffle(order)
    return order


def apply_duration_update(model, optimizer, families, step: int, owner: str) -> dict | None:
    """Match E474 clipping/AdamW exactly, with E488's preregistered sample steps."""
    sampled = step in TELEMETRY_STEPS
    before = None
    gradient_rms = None
    if sampled:
        gradients = {
            family: [parameter.grad for parameter in parameters if parameter.grad is not None]
            for family, parameters in families.items()
        }
        if any(len(gradients[key]) != len(families[key]) for key in families):
            raise ValueError(f"missing gradient family at E488 sample step {step}: {owner}")
        gradient_rms = e474.family_rms(gradients)
        before = {
            family: [parameter.detach().clone() for parameter in parameters]
            for family, parameters in families.items()
        }
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), CLIP_NORM, error_if_nonfinite=True
    )
    optimizer.step()
    e470.ensure_finite_parameters(model, owner)
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, Tensor) and not torch.isfinite(value).all():
                raise ValueError(f"optimizer produced nonfinite state in {owner}")
    if not sampled:
        return None
    assert before is not None and gradient_rms is not None
    delta = {
        family: [
            parameter.detach() - original
            for parameter, original in zip(families[family], before[family], strict=True)
        ]
        for family in families
    }
    return {
        "update_index_zero_based": step,
        "preclip_gradient_rms": gradient_rms,
        "preclip_global_norm": float(total_norm.detach().cpu()),
        "clip_factor": min(1.0, CLIP_NORM / (float(total_norm.detach().cpu()) + 1e-6)),
        "parameter_update_rms": e474.family_rms(delta),
        "head_update_rms": e474.family_rms({"head": delta["head"]}),
    }


def run_chunk_update(model, optimizer, families, x, y, need, ids, row_start, device, owner):
    xx = torch.from_numpy(x[ids, row_start : row_start + CHUNK_ROWS]).to(
        device=device, dtype=torch.float32
    )
    yy = torch.from_numpy(y[ids, row_start : row_start + CHUNK_ROWS]).to(
        device=device, dtype=torch.float32
    )
    mask = torch.from_numpy(need[ids, row_start : row_start + CHUNK_ROWS]).to(device=device)
    if not torch.isfinite(xx).all() or not torch.isfinite(yy).all():
        raise ValueError(f"nonfinite E488 input in {owner}")
    optimizer.zero_grad(set_to_none=True)
    prediction, recurrent_state, gate, context = model.forward_details(
        xx, None if row_start == 0 else model._e488_state
    )
    if not torch.isfinite(prediction).all() or not torch.isfinite(recurrent_state).all():
        raise ValueError(f"nonfinite prediction/state in {owner}")
    if gate is not None and not torch.isfinite(gate).all():
        raise ValueError(f"nonfinite gate in {owner}")
    if context is not None and not torch.isfinite(context).all():
        raise ValueError(f"nonfinite context in {owner}")
    loss = e470.loss_for_batch(prediction, yy, mask)
    if not torch.isfinite(loss):
        raise ValueError(f"nonfinite loss in {owner}")
    loss.backward()
    # Current global step is attached by the caller before invoking this helper.
    step = int(model._e488_step)
    stats = apply_duration_update(model, optimizer, families, step, owner)
    stats = {} if stats is None else stats
    stats["gate"] = e484.gate_summary(gate)
    coefficient = getattr(model.head[1], "gate_coeff", None)
    stats["gate_coefficient_rms"] = (
        None if coefficient is None else float(coefficient.detach().square().mean().sqrt().cpu())
    )
    e470.ensure_finite_parameters(model, owner)
    model._e488_state = recurrent_state.detach()
    return float(loss.detach().cpu()), recurrent_state.detach(), stats


def optimizer_description(model, optimizer) -> dict[str, Any]:
    if len(optimizer.param_groups) != 1:
        raise ValueError("E488 AdamW groups differ from E474")
    group = optimizer.param_groups[0]
    if (group["lr"], group["weight_decay"], tuple(group["betas"])) != (LR, WEIGHT_DECAY, BETAS):
        raise ValueError("E488 AdamW hyperparameters differ from preregistration")
    return {
        "name": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "betas": list(BETAS),
        "groups": 1,
    }


def run_preflight(x, y, need, states):
    records = []
    for seed in SEEDS:
        ids = epoch_order(seed, 0)[:4].tolist()
        for arm in ARMS:
            torch.manual_seed(seed)
            model = make_arm(arm, states[seed]).train()
            optimizer = e484.make_optimizer(model)
            before = [p.detach().clone() for p in model.parameters()]
            model._e488_state = None
            model._e488_step = 0
            loss, state, stats = run_chunk_update(
                model,
                optimizer,
                e484.parameter_families(model),
                x,
                y,
                need,
                ids,
                0,
                torch.device("cpu"),
                f"{seed}/{arm}/preflight",
            )
            changed = sum(
                not torch.equal(old, new)
                for old, new in zip(before, model.parameters(), strict=True)
            )
            if changed == 0 or not torch.isfinite(state).all():
                raise ValueError(f"E488 preflight update did not change finite state: {seed}/{arm}")
            records.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "loss": loss,
                    "changed_parameter_tensors": changed,
                    "state_shape": list(state.shape),
                    "telemetry": stats,
                }
            )
    if len(records) != 12:
        raise ValueError("E488 preflight omitted a cell")
    return records


def train_trajectory(seed, arm, x, y, need, nodes, initial, device, output):
    torch.manual_seed(seed)
    model = make_arm(arm, initial).to(device=device, dtype=torch.float32).train()
    optimizer = e484.make_optimizer(model)
    opt_receipt = optimizer_description(model, optimizer)
    families = e484.parameter_families(model)
    trajectory_started = time.monotonic()
    telemetry: list[dict[str, Any]] = []
    epoch_orders: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    progress = {
        "seed": seed,
        "arm": arm,
        "status": "running",
        "epochs_completed": 0,
        "global_updates": 0,
    }
    atomic_json(output / f"progress_seed{seed}_{arm}.json", progress)
    global_step = 0
    for epoch_index in range(EPOCHS):
        order = epoch_order(seed, epoch_index)
        epoch_orders.append(
            {
                "epoch_index_zero_based": epoch_index,
                "row_groups": [nodes[int(index)]["row_group"] for index in order],
            }
        )
        for batch_index, batch_start in enumerate(range(0, 24, 4)):
            ids = order[batch_start : batch_start + 4].tolist()
            model._e488_state = None
            for row_start in range(0, 20_000, CHUNK_ROWS):
                model._e488_step = global_step
                loss, state, stats = run_chunk_update(
                    model,
                    optimizer,
                    families,
                    x,
                    y,
                    need,
                    ids,
                    row_start,
                    device,
                    f"{seed}/{arm}/e{epoch_index}/b{batch_index}/r{row_start}",
                )
                if global_step in TELEMETRY_STEPS:
                    stats.update(
                        {
                            "global_update_zero_based": global_step,
                            "epoch_index_zero_based": epoch_index,
                            "batch_index_zero_based": batch_index,
                            "fit_row_groups": [nodes[int(index)]["row_group"] for index in ids],
                            "chunk_start": row_start,
                            "loss": loss,
                            "state_abs_max": float(state.abs().max().cpu()),
                            "gradient_rms_preclip": stats.get("preclip_global_norm"),
                        }
                    )
                    telemetry.append(stats)
                global_step += 1
                if global_step != (
                    epoch_index * UPDATES_PER_EPOCH + batch_index * 79 + row_start // CHUNK_ROWS + 1
                ):
                    raise AssertionError("E488 update cadence accounting drifted")
        if global_step != (epoch_index + 1) * UPDATES_PER_EPOCH:
            raise ValueError(f"E488 epoch update count mismatch for {seed}/{arm}/{epoch_index}")
        if epoch_index in (0, 2):
            epoch_number = epoch_index + 1
            checkpoint = output / f"seed{seed}_{arm}_epoch{epoch_number}.pt"
            temporary = checkpoint.with_suffix(".pt.tmp")
            torch.save(model.cpu().state_dict(), temporary)
            os.replace(temporary, checkpoint)
            checkpoint_record = {
                "epoch_number": epoch_number,
                "global_updates": global_step,
                "path": checkpoint.name,
                "sha256": sha256_file(checkpoint),
                "state_sha256": state_sha256(model.state_dict()),
            }
            checkpoints.append(checkpoint_record)
            model.to(device=device)
            atomic_json(
                output / f"checkpoint_seed{seed}_{arm}_epoch{epoch_number}.json", checkpoint_record
            )
        progress = {
            "seed": seed,
            "arm": arm,
            "status": f"epoch_{epoch_index + 1}_complete",
            "epochs_completed": epoch_index + 1,
            "global_updates": global_step,
            "checkpoints": checkpoints,
            "optimizer_state_steps": optimizer_state_steps(optimizer),
            "telemetry": telemetry,
            "epoch_orders": epoch_orders,
        }
        if optimizer_state_steps(optimizer) != [global_step]:
            raise ValueError(
                f"E488 optimizer moments did not continue to step {global_step}: {seed}/{arm}"
            )
        atomic_json(output / f"progress_seed{seed}_{arm}.json", progress)
    if (
        global_step != TOTAL_UPDATES
        or tuple(t["global_update_zero_based"] for t in telemetry) != TELEMETRY_STEPS
    ):
        raise ValueError(f"E488 wrong trajectory cadence: {seed}/{arm}")
    receipt = {
        **progress,
        "status": "complete",
        "fit_seconds": time.monotonic() - trajectory_started,
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "total_updates": global_step,
        "optimizer": opt_receipt,
        "telemetry_steps_zero_based": list(TELEMETRY_STEPS),
        "test_rows_loaded": False,
    }
    atomic_json(output / f"trajectory_seed{seed}_{arm}.json", receipt)
    return receipt


def optimizer_state_steps(optimizer):
    return sorted(
        {int(state["step"].item()) for state in optimizer.state.values() if "step" in state}
    )


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
    if args.run_fits and args.device != "cuda":
        raise ValueError("E488 full trajectories require separately admitted CUDA")
    if args.preflight_real and args.device != "cpu":
        raise ValueError("E488 fit-only preflight must run on CPU")
    torch.set_num_threads(1)
    sources = check_sources()
    states = make_initial_states()
    manifest, x, y, need, nodes = load_fit_data()
    if args.preflight_real:
        output = {
            "kind": "E488 fit-only real-data preflight",
            "status": "passed",
            "parent_source_sha256": sources,
            "actor_source_sha256": {
                path.name: sha256_file(path)
                for path in (
                    HERE / "fit_duration.py",
                    HERE / "tests_duration.py",
                    HERE / "gpu_probe_duration.py",
                )
            },
            "seed_initial_state_sha256": {
                str(seed): state_sha256(state) for seed, state in states.items()
            },
            "fit_only_cohort": {
                **manifest,
                "cohort": [node for node in manifest["cohort"] if node["role"] == "fit"],
            },
            "cells": run_preflight(x, y, need, states),
            "cells_complete": 12,
            "test_rows_loaded": False,
        }
        args.output.mkdir(parents=True)
        atomic_json(args.output / "fit_manifest.json", output)
        print(
            json.dumps(
                {
                    "status": output["status"],
                    "cells": 12,
                    "test_rows_loaded": False,
                    "output": str(args.output),
                },
                indent=2,
            ),
            flush=True,
        )
        return
    configure_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    args.output.mkdir(parents=True)
    records = []
    try:
        with e474.full_ieee_float32():
            precision = e474.precision_metadata()
            for seed in SEEDS:
                for arm in ARMS:
                    records.append(
                        train_trajectory(
                            seed,
                            arm,
                            x,
                            y,
                            need,
                            nodes,
                            states[seed],
                            torch.device("cuda"),
                            args.output,
                        )
                    )
            if len(records) != 12 or any(row["status"] != "complete" for row in records):
                raise ValueError("E488 requires twelve complete continuous trajectories")
            result = {
                "kind": "E488 continuous context-mixture duration study",
                "status": "all_12_trajectories_complete_test_unread",
                "design_sha256": DESIGN_SHA256,
                "parent_source_sha256": sources,
                "actor_source_sha256": {
                    path.name: sha256_file(path)
                    for path in (
                        HERE / "fit_duration.py",
                        HERE / "tests_duration.py",
                        HERE / "gpu_probe_duration.py",
                    )
                },
                "seed_initial_state_sha256": {
                    str(seed): state_sha256(state) for seed, state in states.items()
                },
                "fit_only_cohort": {
                    **manifest,
                    "cohort": [node for node in manifest["cohort"] if node["role"] == "fit"],
                },
                "seeds": list(SEEDS),
                "arms": list(ARMS),
                "epochs": EPOCHS,
                "updates_per_epoch": UPDATES_PER_EPOCH,
                "total_updates_per_trajectory": TOTAL_UPDATES,
                "registered_contrasts_per_seed": list(CONTRASTS_PER_SEED),
                "precision": precision,
                "trajectories": records,
                "test_rows_loaded": False,
            }
            atomic_json(args.output / "fit_manifest.json", result)
    except Exception as exc:
        atomic_json(
            args.output / "FIT_FAILURE.json",
            {
                "status": "partial_failure",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
                "completed_trajectories": records,
                "complete": False,
                "test_rows_loaded": False,
            },
        )
        raise
    print(
        json.dumps(
            {
                "status": "all_12_trajectories_complete_test_unread",
                "trajectories": len(records),
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
