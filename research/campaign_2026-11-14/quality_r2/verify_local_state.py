"""Bind completed local strict-FP32 endpoints to saved optimizer state and source."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

RUNS = Path("/home/vlad/development/wunder/research_tmp/chrono_root/artifacts/runs")
SOURCE = Path("/home/vlad/development/wunder/research_tmp/q14_strict_source")
OUT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    """Hash a small completed artifact or source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(run: str) -> dict:
    """Require four epochs, fixed epoch3, genuine strict Muon groups and matching source."""
    folder = RUNS / run
    provenance = json.loads((folder / "provenance.json").read_text())
    history = json.loads((folder / "history.json").read_text())
    state = torch.load(folder / "training_state.pt", map_location="cpu", weights_only=False)
    assert len(history) == 4 and [x["epoch"] for x in history] == list(range(4))
    assert (
        provenance["selected_epoch"] == 3
        and state["next_epoch"] == 4
        and state["boundary"] == "epoch"
    )
    assert sha256(folder / "model_ep3.pt") == provenance["checkpoint_sha256"]["model_ep3.pt"]
    assert state["contract"]["epochs"] == 4
    groups = state["optimizer"]["param_groups"]
    assert (
        groups and all(x["strict_fp32"] is True for x in groups) and any(x["muon"] for x in groups)
    )
    expected = {
        "training_dynamics.py": sha256(SOURCE / "src/wnn/training_dynamics.py"),
        "optim.py": sha256(SOURCE / "src/wnn/optim.py"),
        "trainer": sha256(SOURCE / "scripts/train_rnn.py"),
    }
    assert all(state["contract"]["sources"][key] == value for key, value in expected.items())
    dtypes = set()
    for values in state["optimizer"]["state"].values():
        for value in values.values():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                dtypes.add(str(value.dtype))
    assert dtypes == {"torch.float32"}
    result = {
        "run": run,
        "status": "verified_completed_strict_state",
        "epoch": 3,
        "score": history[3]["weighted_pearson"],
        "model_ep3_sha256": sha256(folder / "model_ep3.pt"),
        "state_sha256": sha256(folder / "training_state.pt"),
        "strict_fp32_groups": len(groups),
        "optimizer_floating_state_dtypes": sorted(dtypes),
        "source_sha256": expected,
        "scope": "actual completed local run; group/state/source verification complements the prior live CUDA factory precision test",
    }
    (OUT / f"{run}.state_check.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    for name in sys.argv[1:]:
        print(json.dumps(check(name)))
