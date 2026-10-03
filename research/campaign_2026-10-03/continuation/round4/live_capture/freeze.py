"""Freeze local evidence, byte estimates and a pending independent review request."""

from __future__ import annotations

import json
import math

import torch
from live_capture import HERE, ROOT, identities, sha, tensor_bytes


def main() -> None:
    result = json.loads((HERE / "cpu_smoke_final/results.json").read_text())
    old_schema = json.loads(
        (
            ROOT
            / "research/campaign_2026-10-03/continuation/round3/objective_state_gate/export_schema.json"
        ).read_text()
    )
    shapes = old_schema["model_tensor_schema"]
    p = sum(
        math.prod(item["shape"]) for name, item in shapes.items() if name not in {"mean", "std"}
    )
    recurrent = sum(
        math.prod(item["shape"]) for name, item in shapes.items() if name.startswith("gru.weight_")
    )
    model = sum(math.prod(item["shape"]) * 4 for item in shapes.values())
    optimizer = recurrent * 4 + (p - recurrent) * 8
    batch = 32 * 20000 * (211 * 4 + 4 * 4 + 5 * 4 + 4 * 8)
    metadata = 32 * 20000 * 5
    hidden = 2 * 2 * 32 * 256 * 2
    pre = batch + metadata + hidden + 2 * model + optimizer
    post = 2 * model + optimizer + hidden + 3 * p * 4
    measured = {}
    for phase in ("pre", "post"):
        value = torch.load(
            HERE / "cpu_smoke_final/normal" / f"{phase}.pt", weights_only=False, map_location="cpu"
        )
        measured[phase + "_tensor_bytes"] = tensor_bytes(value)
        measured[phase + "_file_bytes"] = (
            (HERE / "cpu_smoke_final/normal" / f"{phase}.pt").stat().st_size
        )
    sizes = {
        "measured_CPU": measured,
        "architecture_derived_GPU": {
            "parameter_count": p,
            "one_model_tensor_bytes": model,
            "warm_muon_adam_moment_bytes_excluding_integer_clocks": optimizer,
            "both_incoming_fp16_hidden_bytes": hidden,
            "full_batch_loss_input_bytes": batch,
            "extra_prop_and_scored_metadata_bytes": metadata,
            "pre_tensor_bytes_approx": pre,
            "post_tensor_bytes_approx": post,
            "four_component_gradient_bytes": 4 * p * 4,
            "two_real_captures_plus_components_bytes_approx": 2 * (pre + post) + 4 * p * 4,
            "known_missing_historical_model_optimizer_hidden_bytes_at_least": 2 * model
            + optimizer
            + hidden,
        },
        "missing_historical_state": "Cannot recover genuine historical tensors from EMA export; full loader/trajectory state size unknown. No bytes fabricated.",
        "exclusions": [
            "RNG small tensors",
            "integer optimizer clocks",
            "scalar losses",
            "torch/Python/ZIP overhead",
            "activations/workspace",
            "CUDA context",
        ],
    }
    (HERE / "state_bytes.json").write_text(json.dumps(sizes, indent=2) + "\n")
    names = (
        "runner.py",
        "live_capture.py",
        "fit_panel.py",
        "fit_panel.json",
        "PREREG.txt",
        "source_binding.json",
        "trainer.patch",
    )
    request = {
        "status": "pending",
        "queue_finished": False,
        "reviewer": None,
        "candidate_sha256": {name: sha(HERE / name) for name in names},
        "source": identities(),
        "command": "research/campaign_2026-10-03/continuation/round4/live_capture/gpu_smoke_review_only.sh CONTROLLER_RECEIPT.json",
        "wall_limit_seconds": 590,
        "GPU_hours_upper_bound": 590 / 3600,
        "actor_may_not_approve": True,
    }
    (HERE / "review_request.json").write_text(json.dumps(request, indent=2) + "\n")
    state = {
        "phase": "frozen_for_independent_controller_review",
        "requested_model": "gpt-6.1-sol medium",
        "role": "round4-capture",
        "write_scope": str(HERE.relative_to(ROOT)),
        "CPU_tests": 25,
        "CPU_smoke_seconds": result["seconds"],
        "CPU_max_delta": {
            key: result[key]
            for key in (
                "normal_max_delta",
                "repeat_max_delta",
                "skip_max_delta",
                "uninstrumented_default_max_delta",
            )
        },
        "real_rows_identified": True,
        "panel_rows": "fit_panel.json",
        "data_scoring": False,
        "GPU_launched": False,
        "GPU_parity_established": False,
        "real_fit_parity_established": False,
        "candidate_approved": False,
        "live_edits": False,
        "queue_changes": False,
        "source": identities(),
        "failures_retained": [
            "logs/ruff_attempt1.log",
            "failure_controls/disabled_scaler_injection/pre.pt",
            "cpu_smoke_attempt1",
            "cpu_smoke_attempt2",
        ],
        "last_failure": "Initial Ruff violations corrected; deliberate disabled-scaler injection failure retained; no resource failure",
        "pending": [
            "independent controller source/schema review",
            "queue completion established independently",
            "controller-authored hash-bound approval receipt",
            "authorized <=590s CUDA smoke",
            "default replay gate before any estimator intervention",
        ],
        "estimated_GPU_cost": request["GPU_hours_upper_bound"],
    }
    (HERE / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    (HERE / "results.json").write_text(
        json.dumps({"checks": state, "smoke": result, "state_bytes": sizes}, indent=2) + "\n"
    )
    hashes = {
        str(path.relative_to(HERE)): sha(path)
        for path in HERE.rglob("*")
        if path.is_file()
        and path.name not in {"artifact_hashes.json", "freeze.log"}
        and not any(part.startswith("test_tmp") for part in path.relative_to(HERE).parts)
    }
    (HERE / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(
        json.dumps(
            {"CPU_deltas": state["CPU_max_delta"], "state_bytes": sizes, "approval": "pending"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
