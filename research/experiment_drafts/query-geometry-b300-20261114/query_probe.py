"""One-B300 numerical admission of the query geometry packet; no scientific fits."""

from __future__ import annotations

import datetime as dt
import os
import platform
from pathlib import Path
from typing import Any

import torch
from query_invariants import require, run_invariants
from torch import Tensor

from wnn.training.sequence_config import load_config
from wnn.training.sequence_models import build_sequence_model, detach_state
from wnn.training.sequence_precision import forward_context

# Native qualification and receipt boundaries contain heterogeneous nested metadata.
ARMS = ("plain", "angular", "radial")
SEED = 209401
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
    "angle_radians": 0.35,
    "eval_arm_max_abs": 0.0,
    "stream_max_abs": 0.03125,
    "rng_replay_max_abs": 0.0,
    "algebra_relative_tolerance": 2e-6,
}


def measure_models(configs: Path, device: torch.device) -> list[dict[str, object]]:
    """Exercise compiled integration, causal eval, direct K/V preservation and gradients."""
    torch.manual_seed(SEED)
    x = torch.randn(1, 192, 112, device=device)
    changed = x.clone()
    changed[:, 128:] += 5
    baseline_eval: Tensor | None = None
    baseline_train: Tensor | None = None
    baseline_keys: tuple[Tensor, ...] | None = None
    baseline_values: tuple[Tensor, ...] | None = None
    parameters: dict[str, Tensor] | None = None
    outcomes = []
    for arm in ARMS:
        settings = load_config(configs / f"{arm}_s{SEED}.json")["model"].copy()
        settings.update(layers=1, context_rows=128)
        torch.manual_seed(SEED)
        model = build_sequence_model(settings).to(device).eval()
        current = model.state_dict()
        if parameters is None:
            parameters = {name: value.detach().clone() for name, value in current.items()}
        require(
            current.keys() == parameters.keys()
            and all(torch.equal(current[name], value) for name, value in parameters.items()),
            "same-seed initial parameters differ",
        )
        with torch.inference_mode(), forward_context(MODE, "cuda"):
            # Exclude first compilation/autotune from the model-path RNG assertion.
            model(x, None)
            rng = torch.cuda.get_rng_state(device)
            output = model(x, None)[0].float()
            require(
                torch.equal(rng, torch.cuda.get_rng_state(device)), "eval consumed perturbation RNG"
            )
            if baseline_eval is None:
                baseline_eval = output
            eval_error = float((output - baseline_eval).abs().max())
            future = model(changed, None)[0].float()
            require(torch.equal(output[:, :128], future[:, :128]), "evaluation future leakage")
            require(bool((output[:, 128:] != future[:, 128:]).any()), "future probe insensitive")
            require(torch.equal(output, model(x, None)[0].float()), "evaluation reset differs")
            chunks, state = [], None
            for start in range(0, 192, 64):
                prediction, _, state = model(x[:, start : start + 64], state)
                chunks.append(prediction.float())
            stream_error = float((torch.cat(chunks, dim=1) - output).abs().max())
            require(eval_error == 0 and stream_error <= 0.03125, "clean eval or streaming mismatch")
        model.train()
        with forward_context(MODE, "cuda"):
            # Warm compiled train path before testing an exact saved RNG replay.
            model(x, None)
        rng = torch.cuda.get_rng_state(device)
        with forward_context(MODE, "cuda"):
            first, _, cache = model(x, None)
        if baseline_train is None:
            baseline_train = first.detach().clone()
            baseline_keys = tuple(t.detach().clone() for t in cache.keys)
            baseline_values = tuple(t.detach().clone() for t in cache.values)
        treatment_error = float((first.float() - baseline_train.float()).abs().max())
        require(
            treatment_error == 0 if arm == "plain" else treatment_error > 0, "treatment is inactive"
        )
        # In one layer the cached K/V come from the same input projection, before Q corruption.
        require(
            all(torch.equal(a, b) for a, b in zip(cache.keys, baseline_keys, strict=True))
            and all(torch.equal(a, b) for a, b in zip(cache.values, baseline_values, strict=True)),
            "query corruption altered direct one-layer K/V",
        )
        torch.cuda.set_rng_state(rng, device)
        with forward_context(MODE, "cuda"):
            replay = model(x, None)[0]
        replay_error = float((first.float() - replay.float()).abs().max())
        require(replay_error == 0, "warm compiled RNG replay differs")
        torch.cuda.set_rng_state(rng, device)
        with forward_context(MODE, "cuda"):
            future_train = model(changed, None)[0]
        require(torch.equal(first[:, :128], future_train[:, :128]), "training future leakage")
        first.float().square().mean().backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        require(
            all(g is not None and bool(torch.isfinite(g).all()) for g in grads), "gradient failure"
        )
        with torch.no_grad(), forward_context(MODE, "cuda"):
            continued = model(x[:, :64], detach_state(cache))[0]
        require(bool(torch.isfinite(continued).all()), "cached training continuation failed")
        outcomes.append(
            {
                "arm": arm,
                "eval_arm_max_abs": eval_error,
                "stream_max_abs": stream_error,
                "training_treatment_max_abs": treatment_error,
                "rng_replay_max_abs": replay_error,
                "same_seed_initialization_equal": True,
                "direct_kv_equal": True,
                "future_prefix_exact": True,
                "gradients_finite": True,
                "continuation_finite": True,
            }
        )
        del model, first, replay, cache, future_train
    return outcomes


def run_probes(configs: Path, qualification: dict[str, Any], owner: str) -> dict[str, Any]:
    """Run bounded independent invariants on one allocated B300 in the native environment."""
    if platform.system() != "Linux" or torch.cuda.device_count() != 1 or not owner.strip():
        raise ValueError("probe requires Linux, one allocated GPU and an owner")
    prop = torch.cuda.get_device_properties(0)
    if "B300" not in prop.name or not torch.cuda.is_bf16_supported():
        raise ValueError("probe requires a BF16-capable B300")
    device = torch.device("cuda:0")
    receipt = {
        "schema": "wnn.query-geometry.probe.v1",
        "status": "passed",
        "qualification": qualification,
        "contract": CONTRACT,
        "invariants": run_invariants(device),
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
    """Bind every required numerical observation to this exact packet/source/split."""
    if (
        receipt.get("schema") != "wnn.query-geometry.probe.v1"
        or receipt.get("status") != "passed"
        or receipt.get("qualification") != qualification
        or receipt.get("contract") != CONTRACT
        or not receipt.get("owner", "").strip()
        or "B300" not in receipt["device"]["name"]
        or [r["arm"] for r in receipt["models"]] != list(ARMS)
    ):
        raise ValueError("missing or stale query-geometry probe")
    for mode in ("angular", "radial"):
        row = receipt["invariants"][mode]
        for field in (
            "mean_error",
            "energy_error",
            "displacement_error",
            "norm_error",
            "dot_error",
            "direction_error",
        ):
            require(0 <= row[field] <= 2e-6, "algebra receipt failed")
        for field in (
            "bf16_cast_exact",
            "rng_replay_exact",
            "future_prefix_exact",
            "row_shared_signs",
            "input_unchanged",
            "gradient_finite_nonzero",
        ):
            require(row[field] is True, "invariant receipt failed")
    for row in receipt["models"]:
        require(
            row["eval_arm_max_abs"] == row["rng_replay_max_abs"] == 0
            and 0 <= row["stream_max_abs"] <= 0.03125
            and (
                row["training_treatment_max_abs"] == 0
                if row["arm"] == "plain"
                else row["training_treatment_max_abs"] > 0
            ),
            "model numerical receipt failed",
        )
        for field in (
            "same_seed_initialization_equal",
            "direct_kv_equal",
            "future_prefix_exact",
            "gradients_finite",
            "continuation_finite",
        ):
            require(row[field] is True, "model invariant receipt failed")
