from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from fit_activation_optimizer import (
    ARMS,
    COHORT_PATH,
    SEED,
    ActivationGRU,
    load_cohort,
    make_optimizer,
    train_update,
)

from wnn.resources import configure_torch


def main() -> None:
    """Run one real B4/T256 update per frozen arm without evaluation rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("probe output must be a fresh identity")
    args.output.mkdir(parents=True)
    duty = configure_torch()
    device = torch.device("cuda")
    _, x, y, need, _, nodes = load_cohort(include_test=False)
    fit_ids = [i for i, node in enumerate(nodes) if node["role"] == "fit"]
    np.random.default_rng(SEED).shuffle(fit_ids)
    results = []
    for arm in ARMS:
        activation, optimizer_name = arm.split("__")
        torch.manual_seed(SEED)
        model = ActivationGRU(activation).to(device)
        optimizer = make_optimizer(model, optimizer_name)
        loss = train_update(model, optimizer, x, y, need, fit_ids[:4], device, duty)
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise ValueError(f"nonfinite updated parameters: {arm}")
        results.append({"arm": arm, "loss": loss, "updates": 1, "finite": True})
        del model, optimizer
        torch.cuda.empty_cache()
    result = {
        "probe_arms": results,
        "fit_only": True,
        "row_groups": [nodes[i]["row_group"] for i in fit_ids[:4]],
        "cohort_sha256": hashlib.sha256(COHORT_PATH.read_bytes()).hexdigest(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
