from __future__ import annotations

import math
from pathlib import Path

from recipe import ROOT, STATE, argv_for, legacy, read, sha, write


def validate(job: dict) -> dict:
    """Reject incomplete, unbound, nonfinite or wrong-membership fit endpoints."""
    work = STATE / "work" / job["id"]
    output = work / "artifacts/runs" / job["id"]
    paths = [
        output / n
        for n in [
            "model.pt",
            "model_ep5.pt",
            "history.json",
            "provenance.json",
            "training_state.pt",
        ]
    ]
    if any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError("Fit lacks complete owned artifacts")
    provenance = read(output / "provenance.json")
    args = provenance["args"]
    request = read(work / "fit_request.json")
    if request["train_argv"] != argv_for(job, False):
        raise ValueError("Fit request differs from frozen argv")
    subset = read(Path(job["subset"]["path"]))
    split_path = ROOT / "splits" / f"fold{job['fold']}.json"
    expected = dict(
        name=job["id"],
        seed=job["seed"],
        hidden=512,
        layers=1,
        epochs=6,
        checkpoint_epoch=5,
        checkpoint_policy="fixed",
        precision="fp32",
        fit_subset_manifest=job["subset"]["path"],
        fit_subset_experiment_id="TABDATA-SHORT4H-20261022-R1",
        split_manifest_sha256=sha(split_path),
        fit_subset_manifest_sha256=job["subset"]["sha256"],
    )
    modes = {
        "group": ("group_age", "none"),
        "field": ("field_age", "none"),
        "moments": ("causal_innovations", "newest"),
        "innovations": ("causal_innovations", "none"),
    }
    mode, control = modes[job["arm"]]
    expected.update(
        research_features=mode,
        research_feature_control=control,
        optim="adamw",
        lr=0.0001,
        beta2=0.999,
        wd=0.01,
        sched="onecycle",
        warmup_frac=0.05,
        batch=8,
        chunk=2000,
        loss="wmse",
        loss_norm="chunk",
        dropout=0.0,
        no_eval_during_fit=True,
        save_training_state=True,
        deterministic=True,
        init="",
        distill="",
        propensity="",
        swa_frac=0.0,
        batch_order="seeded",
    )
    for key, value in expected.items():
        if args.get(key) != value:
            raise ValueError(f"Provenance differs: {key}")
    if (
        provenance["selected_epoch"] != 5
        or provenance["fit_row_groups"] != subset["selected_groups"]
    ):
        raise ValueError("Selected endpoint or fitting membership differs")
    import json

    history = json.loads((output / "history.json").read_text())
    expected_batches = (job["subset"]["train"] + job["subset"]["valid"]) // 8
    for epoch, record in enumerate(history):
        counters = record["update_diagnostics"]["epoch"]
        if record["epoch"] != epoch or counters["completed_batches"] != expected_batches:
            raise ValueError("Unexpected epoch or completed batch count")
        if (
            counters["optimizer_completed_updates"] != 10 * expected_batches
            or counters["amp_skipped_updates"] != 0
        ):
            raise ValueError("Wrong update count or skipped updates")
        if record["update_diagnostics"]["grad_stats"]["nonfinite_steps"] != 0 or not math.isfinite(
            record["train_seconds"]
        ):
            raise ValueError("Nonfinite training state")
    if len(history) != 6:
        raise ValueError("Incomplete six-epoch history")
    import torch

    checkpoint = torch.load(output / "model_ep5.pt", map_location="cpu", weights_only=False)
    model = checkpoint["model"]
    if any(
        t.is_floating_point() and (t.dtype != torch.float32 or not torch.isfinite(t).all())
        for t in model.values()
    ):
        raise ValueError("Non-FP32 or nonfinite model state")

    def inspect_tensors(value: object) -> None:
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() and (
                value.dtype != torch.float32 or not torch.isfinite(value).all()
            ):
                raise ValueError("Nonfinite or non-FP32 optimizer/training-state tensor")
        elif isinstance(value, dict):
            for item in value.values():
                inspect_tensors(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                inspect_tensors(item)

    inspect_tensors(
        torch.load(output / "training_state.pt", map_location="cpu", weights_only=False)
    )
    identity = legacy(ROOT)["checkpoint_identity"].verify_checkpoint_pair(
        output / "model_ep5.pt",
        output / "model.pt",
        expected_epoch=5,
        expected_input_width=323 if job["mechanism"] == "age" else 291,
    )
    for name in ("model_ep5.pt", "model.pt"):
        if provenance["checkpoint_sha256"][name] != sha(output / name):
            raise ValueError("Checkpoint differs from original provenance")
    fit = dict(
        schema="tabdata-short4h-fit-integrity.v1",
        job=job,
        checkpoint_sha256=sha(output / "model_ep5.pt"),
        request_sha256=sha(work / "fit_request.json"),
        provenance_sha256=sha(output / "provenance.json"),
        history_sha256=sha(output / "history.json"),
        successful_updates=60 * expected_batches,
        checkpoint_payload_sha256=identity.payload_sha256,
        training_state_sha256=sha(output / "training_state.pt"),
        expected_args=expected,
    )
    write(work / "FIT.json", fit, fresh=True)
    return fit
