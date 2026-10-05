"""Callable fit-only E484 actual B4×256 CUDA probe; do not launch without admission."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_optimizer"))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_placement"))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_head_mixture"))

import fit_context_mixture_v2 as actor  # noqa: E402

from wnn.resources import configure_torch  # noqa: E402


def run(output: Path) -> dict:
    """Validate native-reference parity and finite changed updates for all 12 arms."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite E484 probe output: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("E484 CUDA probe requested without CUDA")
    sources = actor.check_sources()
    configure_torch()
    torch.set_num_threads(1)
    states = actor.make_initial_states()
    manifest, x, y, need, nodes = actor.load_fit_cohort()
    torch.cuda.reset_peak_memory_stats()
    receipts = []
    with actor.e474.full_ieee_float32():
        precision = actor.e474.precision_metadata()
        parity = actor.check_reference_parity(torch.device("cuda"), states[actor.SEEDS[0]])
        for seed in actor.SEEDS:
            order = np.arange(24)
            np.random.default_rng(seed).shuffle(order)
            ids = order[:4].tolist()
            for arm in actor.ARMS:
                torch.manual_seed(seed)
                model = actor.initialize_arm(arm, states[seed]).to(
                    device="cuda", dtype=torch.float32
                )
                actor.check_common_parameter_bytes(model, states[seed])
                optimizer = actor.make_optimizer(model)
                before = [parameter.detach().clone() for parameter in model.parameters()]
                loss, telemetry = actor.one_update(
                    model,
                    optimizer,
                    x,
                    y,
                    need,
                    ids,
                    torch.device("cuda"),
                    f"probe/{seed}/{arm}",
                )
                changed = [
                    not torch.equal(old, new)
                    for old, new in zip(before, model.parameters(), strict=True)
                ]
                if not any(changed) or not all(
                    torch.isfinite(parameter).all().item() for parameter in model.parameters()
                ):
                    raise ValueError(f"E484 probe cell failed finite/change gate: {seed}/{arm}")
                receipts.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "fit_row_groups": [nodes[int(index)]["row_group"] for index in ids],
                        "batch": 4,
                        "chunk_rows": actor.CHUNK_ROWS,
                        "updates": 1,
                        "loss": loss,
                        "changed_parameter_tensors": sum(changed),
                        "parameters": sum(parameter.numel() for parameter in model.parameters()),
                        "optimizer": actor.optimizer_description(model, optimizer),
                        "telemetry_step0": telemetry,
                    }
                )
                del model, optimizer
                torch.cuda.synchronize()
    if len(receipts) != 12:
        raise ValueError("E484 CUDA probe omitted registered cells")
    result = {
        "kind": "E484 actual CUDA fit-only update probe",
        "status": "all_12_cells_finite_changed",
        "fit_only": True,
        "test_rows_loaded": False,
        "dataset_sha256": actor.DATASET_SHA256,
        "cohort_sha256": actor.file_sha256(actor.e470.COHORT_PATH),
        "parent_source_sha256": sources,
        "actor_source_sha256": {
            path.name: actor.file_sha256(path)
            for path in (
                HERE / "fit_context_mixture_v2.py",
                HERE / "tests_v2.py",
                HERE / "gpu_probe_v2.py",
            )
        },
        "fit_manifest_hash": actor.hashlib.sha256(
            json.dumps(actor.fit_only_manifest(manifest), sort_keys=True).encode()
        ).hexdigest(),
        "fit_group_count": len(nodes),
        "arms": receipts,
        "precision": precision,
        "device": torch.cuda.get_device_name(),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "parity": parity,
        "torch_version": torch.__version__,
    }
    output.mkdir(parents=True)
    temporary = output / "probe.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "probe.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
