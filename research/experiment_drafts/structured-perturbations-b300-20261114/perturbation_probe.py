"""Server-only B300 admission for causal structured perturbations; never a quality score."""

from __future__ import annotations

import datetime as dt
import os
import platform
from pathlib import Path
from typing import Any

import torch
from probe_invariants import check_history, check_input, require
from torch import Tensor

from wnn.training.sequence_config import load_config
from wnn.training.sequence_models import build_sequence_model, detach_state
from wnn.training.sequence_precision import forward_context

ARMS = ("clean", "stale_iid", "stale_burst", "history_iid", "history_burst")
SEED = 208401
MODE = "bf16_bf16x9"
CONTRACT = {
    "seed": SEED,
    "hidden": 768,
    "heads": 12,
    "layers": 1,
    "context_rows": 128,
    "rows": 192,
    "chunk_rows": 64,
    "compile_blocks": True,
    "attention_backend": "efficient",
    "precision_mode": MODE,
    "stream_max_abs": 0.03125,
    "eval_arm_max_abs": 0.0,
    "training_rng_replay_max_abs": 0.0,
}


def measure_models(configs: Path, device: torch.device) -> list[dict[str, object]]:
    """Require neutral evaluation, streaming causality and compiled training replay."""
    torch.manual_seed(SEED)
    x = torch.randn(1, 192, 112, device=device)
    changed = x.clone()
    changed[:, 128:] += 5
    baseline: Tensor | None = None
    baseline_training: Tensor | None = None
    baseline_first_chunk: Tensor | None = None
    parameters: dict[str, Tensor] | None = None
    outcomes = []
    for arm in ARMS:
        settings = load_config(configs / f"{arm}_s{SEED}.json")["model"].copy()
        settings.update(layers=1, context_rows=128)
        torch.manual_seed(SEED)
        model = build_sequence_model(settings).to(device).eval()
        if parameters is None:
            parameters = {k: v.detach().clone() for k, v in model.state_dict().items()}
        current = model.state_dict()
        require(
            current.keys() == parameters.keys()
            and all(torch.equal(current[name], value) for name, value in parameters.items()),
            "same-seed initialization differs across arms",
        )
        with torch.inference_mode(), forward_context(MODE, "cuda"):
            output = model(x, None)[0].float()
            future = model(changed, None)[0].float()
            state = None
            chunks = []
            for start in range(0, 192, 64):
                prediction, _, state = model(x[:, start : start + 64], state)
                chunks.append(prediction.float())
            stream_error = float((output - torch.cat(chunks, dim=1)).abs().max())
            if baseline is None:
                baseline = output
            eval_error = float((output - baseline).abs().max())
            require(torch.equal(output[:, :128], future[:, :128]), "model future leakage")
            require(bool((output[:, 128:] != future[:, 128:]).any()), "insensitive future probe")
            require(torch.equal(output, model(x, None)[0].float()), "model reset differs")
            require(bool(torch.isfinite(output).all()), "nonfinite evaluation")
            require(stream_error <= 0.03125 and eval_error == 0, "eval contract failed")
        # One ordinary warm call builds the graph; replay uses the exact saved CUDA RNG.
        model.train()
        with forward_context(MODE, "cuda"):
            first_chunk, _, cache = model(x[:, :96], None)
            if baseline_first_chunk is None:
                baseline_first_chunk = first_chunk.detach().clone()
            require(torch.equal(first_chunk, baseline_first_chunk), "empty-cache behavior changed")
            cache = detach_state(cache)
            warm = model(x[:, 96:], cache)[0]
        del warm
        rng = torch.cuda.get_rng_state(device)
        with forward_context(MODE, "cuda"):
            first = model(x[:, 96:], cache)[0]
        if baseline_training is None:
            baseline_training = first.detach().clone()
        treatment_error = float((first.float() - baseline_training.float()).abs().max())
        require(
            treatment_error > 0 if arm.startswith("history_") else treatment_error == 0,
            "configured history treatment is inactive or a control changed",
        )
        torch.cuda.set_rng_state(rng, device)
        with forward_context(MODE, "cuda"):
            replay = model(x[:, 96:], cache)[0]
        replay_error = float((first.float() - replay.float()).abs().max())
        require(replay_error == 0, "compiled stochastic replay differs")
        first.float().square().mean().backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        require(
            all(g is not None and bool(torch.isfinite(g).all()) for g in grads), "gradient failure"
        )
        outcomes.append(
            {
                "arm": arm,
                "stream_max_abs": stream_error,
                "eval_arm_max_abs": eval_error,
                "training_rng_replay_max_abs": replay_error,
                "gradients_finite": True,
                "same_seed_initialization_equal": True,
                "training_treatment_max_abs": treatment_error,
            }
        )
        del model, first, replay, cache
    return outcomes


def run_probes(configs: Path, qualification: dict[str, Any], owner: str) -> dict[str, Any]:
    """Measure a bounded one-layer compiled geometry and independent corruption invariants."""
    if platform.system() != "Linux" or torch.cuda.device_count() != 1 or not owner.strip():
        raise ValueError("probe requires Linux, one allocated GPU and an owner")
    prop = torch.cuda.get_device_properties(0)
    if "B300" not in prop.name or not torch.cuda.is_bf16_supported():
        raise ValueError("probe requires a BF16-capable B300")
    device = torch.device("cuda:0")
    receipt = {
        "schema": "wnn.structured-perturbations.probe.v1",
        "status": "passed",
        "qualification": qualification,
        "contract": CONTRACT,
        "input": check_input(device),
        "history": check_history(device),
        "models": measure_models(configs, device),
        "owner": owner,
        "observed_at": dt.datetime.now(dt.UTC).isoformat(),
        "host": platform.node(),
        "device": {
            "name": prop.name,
            "uuid": str(prop.uuid),
            "visible": os.environ["CUDA_VISIBLE_DEVICES"],
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
        },
    }
    validate_probe(receipt, qualification)
    return receipt


def validate_probe(receipt: dict[str, Any], qualification: dict[str, Any]) -> None:
    """Require the complete immutable admission contract for this source and split."""
    if (
        receipt.get("schema") != "wnn.structured-perturbations.probe.v1"
        or receipt.get("status") != "passed"
        or receipt.get("qualification") != qualification
        or receipt.get("contract") != CONTRACT
        or not receipt.get("owner", "").strip()
        or "B300" not in receipt["device"]["name"]
        or [r["arm"] for r in receipt["models"]] != list(ARMS)
    ):
        raise ValueError("missing or stale perturbation probe")
    for mode in ("none", "iid", "burst"):
        if receipt["input"][mode] != {
            "masked_rows": 0 if mode == "none" else 4992,
            "prefix_exact": True,
        }:
            raise ValueError("input invariant receipt failed")
    for row in receipt["models"]:
        if not (
            0 <= row["stream_max_abs"] <= 0.03125
            and row["eval_arm_max_abs"] == row["training_rng_replay_max_abs"] == 0
            and row["gradients_finite"] is True
            and row["same_seed_initialization_equal"] is True
            and (
                row["training_treatment_max_abs"] > 0
                if row["arm"].startswith("history_")
                else row["training_treatment_max_abs"] == 0
            )
        ):
            raise ValueError("model invariant receipt failed")
    for mode in ("iid", "burst"):
        result = receipt["history"]["attention"][mode]
        if not (
            0 <= result["math_max_abs"] <= 0.03125
            and result["future_prefix_exact"] is True
            and result["dropped_key_invariance"] is True
            and result["retained_key_sensitivity"] is True
        ):
            raise ValueError("history invariant receipt failed")
