"""One fit-only production-batch update per fixed mechanism arm."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import fit_activation_mechanism as experiment
import numpy as np
import torch

from wnn.resources import configure_torch


def main() -> None:
    """Check finite changed parameters and memory before full fits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite probe result")
    configure_torch()
    torch.set_num_threads(1)
    _, x, y, need, _, nodes = experiment.e470.load_cohort(include_test=False)
    ids = list(range(len(nodes)))
    np.random.default_rng(experiment.SEED).shuffle(ids)
    ids = ids[:4]
    records = []
    for arm in experiment.ARMS:
        activation, optimizer_name = arm.split("__")
        torch.manual_seed(experiment.SEED)
        model = experiment.MechanismGRU(activation).cuda()
        optimizer = experiment.e470.make_optimizer(model, optimizer_name)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        started = time.monotonic()
        loss = experiment.train_update(model, optimizer, x, y, need, ids, torch.device("cuda"))
        torch.cuda.synchronize()
        changed = any(
            not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True)
        )
        if not changed:
            raise ValueError(f"probe did not update {arm}")
        records.append(
            {
                "arm": arm,
                "loss": loss,
                "update_seconds": time.monotonic() - started,
                "parameters_changed": True,
            }
        )
        del optimizer, model, before
    result = {
        "scope": "Ten fit-only B4x256 GPU updates, no test scoring or full fit",
        "fit_row_groups": [nodes[index]["row_group"] for index in ids],
        "arms": records,
        "source_sha256": {
            name: experiment.file_hash(Path(__file__).parent / name)
            for name in ("gpu_probe.py", "fit_activation_mechanism.py", "tests.py")
        },
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    args.output.mkdir(parents=True)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
