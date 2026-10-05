"""Callable 18-cell E481 CUDA update probe; source only until separately admitted."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_optimizer"))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_placement"))
sys.path.insert(0, str(ROOT / "src"))
import fit_activation_optimizer as e470  # noqa: E402
import fit_activation_placement as e474  # noqa: E402
from fit_recurrent_curvature import (  # noqa: E402
    ARMS,
    DATASET_SHA256,
    FEATURES,
    N_PARAMETERS,
    SHAPES,
    check_parent_sources,
    initialize_arm,
    make_optimizer,
    one_update,
    seed_states,
)

from wnn.resources import configure_torch  # noqa: E402


def run(output: Path) -> dict:
    """Run all cells on a B4×256 fit batch with strict finite/change checks."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite probe directory: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    configure_torch()
    torch.set_num_threads(1)
    parents = check_parent_sources()
    manifest, x, y, need, _score, nodes = e470.load_cohort(include_test=False)
    if (
        x.shape != (24, 20_000, FEATURES)
        or len(nodes) != 24
        or any(n["role"] != "fit" for n in nodes)
    ):
        raise ValueError("probe loader did not return exactly the 24 fit-only sequences")
    states = seed_states()
    torch.cuda.reset_peak_memory_stats()
    arms = []
    with e474.full_ieee_float32():
        for seed, optimizer_name, shape in ARMS:
            order = np.arange(24)
            np.random.default_rng(seed).shuffle(order)
            ids = order[:4].tolist()
            torch.manual_seed(seed)
            model = initialize_arm(shape, states[seed]).to(device="cuda", dtype=torch.float32)
            optimizer = make_optimizer(model, optimizer_name)
            before = [p.detach().clone() for p in model.parameters()]
            loss, telemetry = one_update(
                model,
                optimizer,
                x,
                y,
                need,
                ids,
                torch.device("cuda"),
                f"{seed}/{optimizer_name}/{shape}",
            )
            changed = [
                not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True)
            ]
            if not any(changed) or not all(
                torch.isfinite(p).all().item() for p in model.parameters()
            ):
                raise ValueError(
                    f"probe update was nonfinite or unchanged: {seed}/{optimizer_name}/{shape}"
                )
            if sum(p.numel() for p in model.parameters()) != N_PARAMETERS:
                raise ValueError("probe model parameter count changed")
            arms.append(
                {
                    "seed": seed,
                    "optimizer": optimizer_name,
                    "shape": shape,
                    "fit_row_groups": [nodes[i]["row_group"] for i in ids],
                    "batch": 4,
                    "chunk_rows": 256,
                    "updates": 1,
                    "loss": loss,
                    "changed_parameter_tensors": sum(changed),
                    "parameters": N_PARAMETERS,
                    "telemetry_step0": telemetry,
                }
            )
            del model, optimizer
            torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    if len(arms) != 18:
        raise ValueError("probe omitted one or more cells")
    output.mkdir(parents=True)
    result = {
        "kind": "E481 all-cell CUDA update probe",
        "status": "all_18_cells_finite_changed",
        "fit_only": True,
        "test_rows_loaded": False,
        "dataset_sha256": DATASET_SHA256,
        "cohort_sha256": e474.file_hash(e470.COHORT_PATH),
        "parent_source_sha256": parents,
        "source_sha256": {
            name: e474.file_hash(HERE / name)
            for name in ("fit_recurrent_curvature.py", "tests.py", "gpu_probe.py")
        },
        "shapes": list(SHAPES),
        "arms": arms,
        "precision": e474.precision_metadata(),
        "device": torch.cuda.get_device_name(),
        "peak_allocated_bytes": int(peak_allocated),
        "peak_reserved_bytes": int(peak_reserved),
        "torch_version": torch.__version__,
    }
    (output / "probe.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "status": result["status"],
                "arms": len(arms),
                "peak_allocated_bytes": int(peak_allocated),
                "peak_reserved_bytes": int(peak_reserved),
                "output": str(output),
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
