"""Future controller-only CUDA entry, fail closed before reading scientific tensors.

Any is required at the heterogeneous v4 serialization/provenance boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from protocol import HERE, V4, cap, execute, tensor_identity


def own_hashes() -> dict[str, str]:
    """Bind only frozen executable/preregistered interface files."""
    return {
        name: cap.sha(HERE / name)
        for name in (
            "PREREG.json",
            "PREREG_WARM_CLOCK_CLARIFICATION.json",
            "protocol.py",
            "run_real.py",
            "controls.py",
        )
    }


def warm_counts(payload: dict[str, Any]) -> tuple[int, int]:
    """Verify actual warm attempts/completions from clocks and every Adam moment clock."""
    clock = payload["clock"]
    skips = clock["skips"]
    if not isinstance(skips, torch.Tensor) or skips.dtype != torch.int64 or skips.ndim != 0:
        raise ValueError("UNKNOWN warm skip type")
    count = int(skips)
    complete = 4 - count
    if (
        count < 0
        or complete < 1
        or clock["attempts"] != 4
        or clock["step_count"] != 4
        or clock["ema_updates"] != 4
        or payload["ema_n"] != 4
        or payload["scheduler"]["last_epoch"] != 4
    ):
        raise ValueError("UNKNOWN warm attempted/completed clock")
    adam_steps = [
        state["step"] for state in payload["optimizer"]["state"].values() if "step" in state
    ]
    if not adam_steps or any(type(step) is not int or step != complete for step in adam_steps):
        raise ValueError("UNKNOWN completed updates disagree with genuine warm Adam state")
    return complete, count


def trajectory_check(payload: dict[str, Any], trace: dict[str, Any]) -> None:
    """Require initial None/four-attempt source evidence and verified completed clocks."""
    complete, skips = warm_counts(payload)
    expected = {
        "initial_student_h": None,
        "initial_ema_h": None,
        "warm_starts": [0, 1000, 2000, 3000],
        "completed_warm_updates": complete,
        "warm_skips": skips,
        "entrypoint": "runner.run_probe(mode='real')",
        "capture_runner_sha256": cap.sha(V4 / "runner.py"),
        "fit_panel_source_sha256": cap.sha(V4 / "fit_panel.py"),
        "adapter_sha256": cap.sha(V4 / "live_capture.py"),
        "source": payload["source"],
        "inputs_identity": tensor_identity(payload["batch"]["inputs"]),
        "mean_identity": tensor_identity(payload["student"]["mean"]),
        "std_identity": tensor_identity(payload["student"]["std"]),
        "panel_sha256": payload["metadata"].get("panel_sha256"),
    }
    if any(key not in trace or trace[key] != value for key, value in expected.items()):
        raise ValueError("UNKNOWN initial-None/four-warm-chunk source trajectory")
    if (
        payload["metadata"].get("evidence_kind") != "real_fit_capture"
        or payload["metadata"].get("s") != 4000
        or payload["args"]["chunk"] != 1000
    ):
        raise ValueError("UNKNOWN four prefix attempts at 4000")


def authorize(receipt_path: Path | None, pre: Path, post: Path, trace: Path) -> dict[str, Any]:
    """Check separate controller admission before tensor load/private CUDA validation."""
    if receipt_path is None:
        raise ValueError(
            "UNAPPROVED: later independent controller default-v4 gate and state-drift GPU approval required"
        )
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("status") != "approved"
        or receipt.get("default_v4_gate") != "PASS"
        or receipt.get("independent_state_drift_review") != "PASS"
        or receipt.get("reviewer") in (None, "", "state_drift", "round6-state-drift")
        or receipt.get("candidate_sha256") != own_hashes()
        or receipt.get("capture_sha256")
        != {"pre": cap.sha(pre), "post": cap.sha(post), "trajectory": cap.sha(trace)}
    ):
        raise ValueError("UNAPPROVED controller/source/capture/trajectory identity")
    import runner as v4_runner

    v4_runner.verify_review(Path(receipt["v4_review_receipt"]))
    v4_runner.runtime_guard()
    if not torch.cuda.is_available():
        raise ValueError("actual CUDA runtime required; no CPU CUDA restore")
    return receipt


def main() -> None:
    """Read only future explicitly approved real captures; execute one binary comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if not options.output.resolve().is_relative_to(HERE.resolve()):
        raise ValueError("output must stay inside state_drift")
    authorize(options.receipt, options.pre, options.post, options.trajectory)
    pre = torch.load(options.pre, map_location="cpu", weights_only=False)
    if pre["execution"]["device"] != "cuda":
        raise ValueError("real comparison is CUDA-only; CPU proxies not admitted")
    trajectory_check(pre, json.loads(options.trajectory.read_text()))
    # v4 validates CPU/CUDA RNG on installed private engines before any global setter.
    # Match the source capture configuration before backend/private-RNG validation.
    # Actual settings must still exactly match recorded execution in v4.validate.
    cap.trainer_module().configure_torch()
    cap.validate(pre)
    post = torch.load(options.post, map_location="cpu", weights_only=False)
    before = {
        str(path): cap.sha(path)
        for path in (options.pre, options.post, options.trajectory, options.receipt)
    }
    result = execute(pre, post, options.output, options.receipt)
    if before != {name: cap.sha(Path(name)) for name in before}:
        raise ValueError("scientific input preservation failed")
    (options.output / "input_hashes.json").write_text(json.dumps(before, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
