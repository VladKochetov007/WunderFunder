from __future__ import annotations

import argparse
import json
import math


def trunk_estimate(layers: int, width: int, heads: int, ff_width: int) -> dict[str, object]:
    """Count dense MHA/SwiGLU trunk matrices; exclude stems, norms and auxiliary branches."""
    parameters = layers * (4 * width**2 + 3 * width * ff_width)
    return {
        "layers": layers,
        "width": width,
        "heads": heads,
        "ff_width": ff_width,
        "trunk_parameters": parameters,
        "fp32_parameter_gradient_momentum_gb": parameters * 12 / 1e9,
        "fp32_parameter_gradient_two_moments_gb": parameters * 16 / 1e9,
        "dense_score_map_one_sequence_4096_gb": 4 * heads * 4096**2 / 1e9,
        "dense_score_map_one_sequence_20000_gb": 4 * heads * 20000**2 / 1e9,
    }


def duration_scenarios(rows: int, batch: int, gpus: int) -> list[dict[str, object]]:
    """Compute conditional fit cost; seconds/update are examples, never measured throughput."""
    scenarios: list[dict[str, object]] = []
    for epochs in (12, 24, 48):
        updates = math.ceil(rows * epochs / batch)
        for seconds in (0.5, 1.0, 2.0, 5.0):
            wall_hours = updates * seconds / 3600
            scenarios.append(
                {
                    "epochs": epochs,
                    "approximate_updates": updates,
                    "hypothetical_seconds_per_update": seconds,
                    "fit_wall_hours_excluding_delivery": wall_hours,
                    "fit_gpu_hours_excluding_delivery": wall_hours * gpus,
                }
            )
    return scenarios


def main() -> None:
    """Print analytical memory and conditional runtime accounting without loading data or CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fitting-rows", type=int, default=200_000_000)
    parser.add_argument("--new-rows-per-update", type=int, default=65_536)
    parser.add_argument("--gpus", type=int, default=8)
    args = parser.parse_args()
    if min(args.fitting_rows, args.new_rows_per_update, args.gpus) <= 0:
        parser.error("row counts and GPU count must be positive")
    result = {
        "status": "analytical_only_not_a_profile_or_memory_admission",
        "limits": [
            "excludes nontrunk parameters, activations, workspace and communication buffers",
            "dense score map is one tensor, not total attention or peak device memory",
            "runtime seconds/update are hypothetical, not B300 estimates",
            "row batching, padding and sequence boundaries determine actual update counts",
            "multiply fit costs by actual fits; include validation, export and students separately",
        ],
        "fitting_rows_scenario": args.fitting_rows,
        "new_rows_per_update_scenario": args.new_rows_per_update,
        "gpus_per_fit_scenario": args.gpus,
        "trunks": [
            trunk_estimate(*shape)
            for shape in (
                (16, 1024, 16, 2816),
                (24, 1536, 24, 4096),
                (32, 2048, 32, 5632),
                (40, 2560, 40, 6912),
                (48, 3072, 48, 8192),
            )
        ],
        "duration_scenarios": duration_scenarios(
            args.fitting_rows, args.new_rows_per_update, args.gpus
        ),
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
