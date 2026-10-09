"""Independently check the collected D02 producer artifacts before scoring."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009"


def read(path: Path) -> dict:
    """Read an artifact whose schema is checked at its consumption site."""
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    """Hash actual artifact bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_finite(value: object) -> None:
    """Check every tensor in an actual checkpoint or optimizer state."""
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for item in value.values():
            check_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            check_finite(item)


def main() -> None:
    """Validate fixed epoch, causal gates, controls, membership and predictions."""
    output = BASE / "cloud_delivery/d02_r5/terminal_collect_r1/attempt_001/outputs/out"
    control = BASE / "cloud_delivery/gpu_loss_pair_r5_export/terminal_collect_r1/attempt_001/outputs/out"
    run = read(output / "gpu_pair_run_receipt.json")
    assert run["state"] == "complete"
    split_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    split = read(split_path)
    groups = split["evaluation"]["valid"]
    components = {
        (source, int(row["row_group"])): int(row["component"])
        for source, rows in split["sequences"].items()
        for row in rows
    }
    evaluation_components = {components["valid", int(group)] for group in groups}
    reference_schedule = read(control / "shared/schedule_trace.json")
    arms = {}
    for name in ("wmse", "huber_d1"):
        arm = output / "arms" / name
        provenance = read(arm / "model/provenance.json")
        args, fit = provenance["args"], provenance["fit_row_groups"]
        assert args["optim"] == "muon"
        assert args["loss"] == ("huber" if name == "huber_d1" else "wmse")
        assert not args.get("init") and not args.get("distill")
        assert len(fit["train"]) == 350 and not fit["valid"]
        fitting_components = {
            components[source, int(group)] for source, rows in fit.items() for group in rows
        }
        assert not fitting_components & evaluation_components
        assert provenance["split_manifest_sha256"] == sha(split_path)
        for filename in ("initial_state.npz", "normalization.npz", "sample_orders.json", "profile_inputs.npz"):
            assert sha(arm / "prefit" / filename) == sha(control / "shared" / filename)
        schedule = read(arm / "schedule_trace.json")
        assert len(schedule["rows"]) == len(reference_schedule["rows"]) == 220
        for actual, expected in zip(schedule["rows"], reference_schedule["rows"], strict=True):
            assert {k: v for k, v in actual.items() if k != "optimizer_groups"} == {
                k: v for k, v in expected.items() if k != "optimizer_groups"
            }
            assert len(actual["optimizer_groups"]) == 2 and len(expected["optimizer_groups"]) == 1
            for group in actual["optimizer_groups"]:
                reference = expected["optimizer_groups"][0]
                assert (group["lr"], group["beta1"]) == (reference["lr"], reference["beta1"])
        trace = [json.loads(row) for row in (arm / "update_trace.jsonl").read_text().splitlines()]
        assert len(trace) == 220 and all(row["gradient_finite"] and not row["scaler_skipped"] for row in trace)
        model = torch.load(arm / "model/model.pt", map_location="cpu", weights_only=False)
        epoch = torch.load(arm / "model/model_ep1.pt", map_location="cpu", weights_only=False)
        state = torch.load(arm / "model/training_state.pt", map_location="cpu", weights_only=False)
        for value in (model, epoch, state):
            check_finite(value)
        assert model["model"].keys() == epoch["model"].keys()
        assert all(torch.equal(value, epoch["model"][key]) for key, value in model["model"].items())
        sidecar = read(arm / "predictions_valid.json")
        prediction = np.load(arm / "predictions_valid.npy", mmap_mode="r", allow_pickle=False)
        assert prediction.shape == (468, 20000, 2) and prediction.dtype == np.float32
        assert np.isfinite(prediction).all()
        assert sidecar["epoch"] == 1 and sidecar["row_groups"] == groups
        assert sidecar["checkpoint_sha256"] == sha(arm / "model/model.pt")
        assert sidecar["prediction_sha256"] == sha(arm / "predictions_valid.npy")
        gate = run["device_profile_gate"]["arms"]["huber" if name == "huber_d1" else name]
        assert gate["state"] == "passed" and gate["reload"]["status"] == "pass"
        assert gate["profile"]["evidence"]["optimizer_completed_updates"] == 10
        arms[name] = {
            "checkpoint_sha256": sha(arm / "model/model.pt"),
            "fixed_epoch1_sha256": sha(arm / "model/model_ep1.pt"),
            "prediction_sha256": sidecar["prediction_sha256"],
            "provenance_sha256": sha(arm / "model/provenance.json"),
            "traces": {
                filename: {"path": str((arm / filename).relative_to(ROOT)), "sha256": sha(arm / filename)}
                for filename in ("schedule_trace.json", "update_trace.jsonl")
            },
            "completed_updates": len(trace),
            "fit_components": len(fitting_components),
        }
    receipt = {
        "schema": "wnn.root-d02-r5-terminal-review.r1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "pass",
        "checker_sha256": sha(Path(__file__)),
        "collection_integrity_sha256": sha(BASE / "cloud_delivery/d02_r5/ROOT_COLLECTION_INTEGRITY_R1.json"),
        "operator_receipt_sha256": sha(BASE / "cloud_delivery/d02_r5/GPU_D02_R5_TERMINAL_OPERATOR_RECEIPT_R1.json"),
        "run_receipt_sha256": sha(output / "gpu_pair_run_receipt.json"),
        "arms": arms,
        "schedule_qualification": "Raw traces differ because Muon records two optimizer groups; all 220 step/LR/beta1 values match AdamW semantically. Preserve raw hashes and use an identified grader successor.",
        "checkpoint_qualification": "Prediction sidecars bind model.pt; its tensors equal fixed model_ep1.pt exactly. Archive hashes differ.",
        "scientific_result": "pending independent frozen11endpoint grade",
        "training_wall_seconds": run["fit_results"],
        "worker_wall_seconds": read(output.parent / "final_status.json")["elapsed_seconds"],
    }
    target = BASE / "ROOT_D02_R5_TERMINAL_REVIEW_R1.json"
    with target.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "pass", "receipt_sha256": sha(target)}))


if __name__ == "__main__":
    main()
