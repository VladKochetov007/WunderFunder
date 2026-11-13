"""Bounded B300 numerical qualification of the two proposed streaming geometries."""

from __future__ import annotations

import datetime as dt
import math
import os
import platform
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from wnn.training.sequence_config import load_config
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_precision import forward_context

HEADS = (12, 6)
WINDOW = 16
ROWS = 64
CHUNK = 8
CUT = 32
SEED = 207301
MODE = "bf16_bf16x9"
# Chunk shapes can change accumulation with BF16. Same-shape comparisons should be exact.
MAX_CHUNK_ERROR = 0.03125
MIN_SIGNAL = 0.125
CONTRACT = {
    "seed": SEED,
    "hidden": 768,
    "layers": 1,
    "context_rows": WINDOW,
    "sequence_rows": ROWS,
    "chunk_rows": CHUNK,
    "perturb_after": CUT,
    "attention_backend": "efficient",
    "compile_blocks": True,
    "precision_mode": MODE,
    "max_chunk_error": MAX_CHUNK_ERROR,
    "max_prefix_or_reset_error": 0.0,
    "min_signal_per_output": MIN_SIGNAL,
}


def checked(metrics: dict[str, Any]) -> bool:
    """Establish a nontrivial per-target perturbation and stable predictions."""
    if not metrics.get("cache_verified") or not metrics.get("all_outputs_finite"):
        return False
    values = [
        metrics[k]
        for k in (
            "stream_max_abs",
            "future_prefix_max_abs",
            "reset_max_abs",
            "future_effect_min_across_targets",
            "output_span_min_across_targets",
        )
    ]
    if not all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in values):
        return False
    chunk, prefix, reset, future_effect, span = values
    return (
        chunk <= MAX_CHUNK_ERROR and prefix == reset == 0 and min(future_effect, span) >= MIN_SIGNAL
    )


def measure(config_path: Path, heads: int, inputs: Tensor, changed: Tensor) -> dict[str, Any]:
    """Compare one full call to a rolling cache and a future-only intervention."""
    settings = load_config(config_path)["model"].copy()
    settings.update(layers=1, context_rows=WINDOW)
    if (
        settings["hidden"],
        settings["heads"],
        settings["precision_mode"],
        settings["attention_backend"],
        settings["compile_blocks"],
    ) != (768, heads, MODE, "efficient", True):
        raise ValueError("probe geometry, backend or precision changed")
    torch.manual_seed(SEED)
    model = build_sequence_model(settings).to(inputs.device).eval()

    def finite_output(prediction: Tensor) -> Tensor:
        if prediction.dtype != torch.bfloat16 or not bool(torch.isfinite(prediction).all()):
            raise ValueError("expected finite BF16 predictions")
        return prediction.float()

    with torch.inference_mode(), forward_context(MODE, "cuda"):
        reference = finite_output(model(inputs, None)[0])
        perturbed = finite_output(model(changed, None)[0])
        state = None
        streaming = []
        for start in range(0, ROWS, CHUNK):
            output, _, state = model(inputs[:, start : start + CHUNK], state)
            streaming.append(finite_output(output))
            stop = start + CHUNK
            shape = (1, heads, min(stop, WINDOW - 1), 768 // heads)
            if state.next_position != stop or len(state.keys) != 1 or len(state.values) != 1:
                raise ValueError("rolling state layer count or absolute position changed")
            for tensor in (state.keys[0], state.values[0]):
                if (
                    tensor.shape != shape
                    or tensor.dtype != torch.float32
                    or not bool(torch.isfinite(tensor).all())
                ):
                    raise ValueError("invalid rolling KV shape, dtype or contents")
        reset = finite_output(model(inputs, None)[0])
    delta = (perturbed - reference).abs()
    return {
        "heads": heads,
        "head_dim": 768 // heads,
        "cache_verified": True,
        "all_outputs_finite": True,
        "stream_max_abs": (torch.cat(streaming, dim=1) - reference).abs().max().item(),
        "future_prefix_max_abs": delta[:, :CUT].max().item(),
        "future_effect_min_across_targets": delta[:, CUT:].amax(dim=1).min().item(),
        "output_span_min_across_targets": (reference.amax(dim=1) - reference.amin(dim=1))
        .min()
        .item(),
        "reset_max_abs": (reset - reference).abs().max().item(),
    }


def run_probes(configs: Path, qualification: dict[str, Any], owner: str) -> dict[str, Any]:
    """Exercise the native compiled efficient attention implementation on one allocated B300."""
    if platform.system() != "Linux" or torch.cuda.device_count() != 1:
        raise ValueError("probe needs Linux and exactly one allocated CUDA device")
    prop = torch.cuda.get_device_properties(0)
    if "B300" not in prop.name or not torch.cuda.is_bf16_supported():
        raise ValueError("probe requires a BF16-capable B300")
    device = torch.device("cuda:0")
    generator = torch.Generator().manual_seed(SEED)
    inputs = torch.randn(1, ROWS, 112, generator=generator).to(device)
    changed = inputs.clone()
    changed[:, CUT:] = torch.randn(1, ROWS - CUT, 112, generator=generator).to(device)
    outcomes = []
    for heads in HEADS:
        result = measure(configs / f"heads{heads}_s{SEED}.json", heads, inputs, changed)
        print(f"heads={heads} metrics={result}", flush=True)
        outcomes.append(result)
    receipt = {
        "schema": "wnn.head-count.probe.v1",
        "status": "passed",
        "qualification": qualification,
        "contract": CONTRACT,
        "results": outcomes,
        "owner": owner,
        "observed_at": dt.datetime.now(dt.UTC).isoformat(),
        "host": platform.node(),
        "device": {
            "name": prop.name,
            "uuid": str(prop.uuid),
            "capability": list(torch.cuda.get_device_capability(0)),
            "visible": os.environ["CUDA_VISIBLE_DEVICES"],
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
        },
    }
    validate_probe(receipt, qualification)
    return receipt


def validate_probe(receipt: dict[str, Any], qualification: dict[str, Any]) -> None:
    """Refuse incomplete geometry evidence and receipts from another source or data split."""
    outcomes = receipt["results"]
    if (
        receipt["schema"] != "wnn.head-count.probe.v1"
        or receipt["status"] != "passed"
        or receipt["qualification"] != qualification
        or receipt["contract"] != CONTRACT
        or not receipt["owner"].strip()
        or "B300" not in receipt["device"]["name"]
        or [r["heads"] for r in outcomes] != list(HEADS)
        or any(r["head_dim"] != 768 // r["heads"] or not checked(r) for r in outcomes)
    ):
        raise ValueError(
            "source-qualified geometry probe failed; inspect the retained measurements"
        )
