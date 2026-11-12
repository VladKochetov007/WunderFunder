from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from transition_support_s4 import copy_prefix, pinned_json, same_payload, write_json_new
from wnn.training.sequence_fit_resume import load_resume
from wnn.training.sequence_monitor import SequenceMonitor

from wnn.training.sequence_checkpoint import load_training_state
from wnn.training.sequence_config import load_config
from wnn.training.sequence_distributed import DistributedContext, initialize_distributed
from wnn.training.sequence_fit_utils import RankJsonlLog, make_optimizer, total_updates
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_monitor_progress import close_monitor_progress
from wnn.training.sequence_optimizer_telemetry import optimizer_contract
from wnn.training.sequence_provenance import (
    canonical_sha256,
    file_sha256,
    run_identity,
    runtime_identity,
    source_identity,
)


def test_actual_canonical_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore the actual state/model/optimizer/journals without a forward or optimizer update."""
    spec = pinned_json(json.loads(os.environ["WNN_GRU_RECOVERY_PROBE"]))
    output, scratch = Path(spec["output"]), Path(spec["scratch"])
    proof = pinned_json(
        {"path": str(output / "transition_proof.json"), "sha256": spec["proof_sha256"]}
    )
    identity = proof["continuation_identity"]
    request = proof["request"]
    config = load_config(Path(request["new_config"]["path"]))
    original = output / "training_state.pt"
    original_sha = file_sha256(original)
    if original_sha != proof["continuation_checkpoint_sha256"] or scratch.exists():
        raise ValueError("probe requires untouched migrated state and a fresh scratch directory")
    scratch.mkdir(parents=True)
    target = scratch / "training_state.pt"
    target.write_bytes(original.read_bytes())
    cpu = spec["device"] == "cpu"
    if cpu:
        assert not torch.cuda.is_initialized() and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    context = (
        DistributedContext(0, 0, 1, torch.device("cpu"), False) if cpu else initialize_distributed()
    )
    state = load_training_state(target, identity, context)
    offsets = {
        "rank0.jsonl": state["rank_log_offsets"][0],
        "metrics.jsonl": state["metrics_offset"],
        "monitor_events.jsonl": state["monitor_offsets"]["events"],
        "monitor_evaluations.jsonl": state["monitor_offsets"]["evaluations"],
    }
    for name, offset in offsets.items():
        copy_prefix(output / name, scratch / name, offset)
    model = build_sequence_model(config["model"]).to(context.device, dtype=torch.float32)
    duration = total_updates(config, len(identity["data"]["fit_members"]), 1)
    optimizer, scheduler = make_optimizer(model, config, duration)
    if not cpu:
        root = Path(request["new_source_root"])
        actual = run_identity(
            config,
            identity["data"],
            source_identity(root, root / "scripts/train_sequence.py"),
            runtime_identity(context),
            1,
        )
        actual["model_parameters"] = {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        actual["optimizer"] = optimizer_contract(optimizer, config, duration)
        if actual != identity:
            raise ValueError(
                "real GPU resume probe differs from exact S4 profile/runtime/source identity"
            )
    plan = identity["data"]["monitoring_plan"]
    monitor = SequenceMonitor(
        scratch,
        config["monitoring"],
        [tuple(v) for v in plan["train_eval_members"]],
        [tuple(v) for v in identity["data"]["evaluation_members"]],
        identity["data"]["split_sha256"],
        canonical_sha256(identity),
        resume=True,
        input_feature_mode=plan.get("input_feature_mode", "raw112"),
        evaluation_role=plan.get("evaluation_role", "valid"),
        evaluation_mask=plan.get("evaluation_mask", "is_scored"),
    )
    rank_log = RankJsonlLog(scratch / "rank0.jsonl", resume=True)
    cuda_calls = []
    if cpu:

        def capture_cuda_rng(value: torch.Tensor, device: object = None) -> None:
            """CPU-only test seam; compare opaque serialized CUDA bytes without applying them."""
            assert value.dtype == torch.uint8
            assert same_payload(value, state["rank_rng_states"][0]["torch_cuda"])
            assert device == context.device
            cuda_calls.append(value.clone())

        monkeypatch.setattr(torch.cuda, "set_rng_state", capture_cuda_rng)
    try:
        result = load_resume(
            target, identity, model, optimizer, scheduler, context, rank_log, scratch, monitor, True
        )
        assert result == (state["next_epoch"], state["global_step"], state["history"])
        assert same_payload({k: v.cpu() for k, v in model.state_dict().items()}, state["model"])

        # Loaded optimizer tensors may reside on the admitted GPU; compare serialized CPU copies.
        def cpu_payload(value: object) -> object:
            if isinstance(value, torch.Tensor):
                return value.cpu()
            if isinstance(value, dict):
                return {k: cpu_payload(v) for k, v in value.items()}
            if isinstance(value, list):
                return [cpu_payload(v) for v in value]
            if isinstance(value, tuple):
                return tuple(cpu_payload(v) for v in value)
            return value

        assert same_payload(cpu_payload(optimizer.state_dict()), state["optimizer"])
        assert same_payload(scheduler.state_dict(), state["scheduler"])
        rng = state["rank_rng_states"][0]
        assert random.getstate() == rng["python"]
        assert same_payload(np.random.get_state(), rng["numpy"])  # noqa: NPY002 - compare saved legacy RNG
        assert torch.equal(torch.get_rng_state(), rng["torch_cpu"])
        assert len(cuda_calls) == (1 if cpu else 0)
        if cpu:
            assert not torch.cuda.is_initialized()
        if not cpu:
            assert torch.equal(torch.cuda.get_rng_state(context.device), rng["torch_cuda"])
        for name, offset in offsets.items():
            assert (scratch / name).stat().st_size == offset
        assert monitor.offsets() == state["monitor_offsets"]
        assert file_sha256(original) == original_sha
        write_json_new(
            Path(spec["receipt"]),
            {
                "schema": "wnn.gru-canonical-resume-probe.v2",
                "status": "passed",
                "device": spec["device"],
                "cuda_rng_setter_stubbed": cpu,
                "real_cuda_resume_exercised": not cpu,
                "forward_calls": 0,
                "optimizer_updates": 0,
                "next_epoch": state["next_epoch"],
                "global_step": state["global_step"],
                "checkpoint_sha256": original_sha,
                "transition_proof_sha256": spec["proof_sha256"],
            },
        )
    finally:
        rank_log.close()
        close_monitor_progress(monitor)
