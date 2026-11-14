"""Reuse the frozen exporter with four independent sequences per inference batch."""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

SOURCE = Path("/home/vlad/development/wunder/research_tmp/q14_strict_source")


def bounded_loader(*args: Any, **kwargs: Any) -> DataLoader:
    """Reduce batch memory without changing sequence length, row order, or weights."""
    assert kwargs["batch_size"] == 16
    kwargs["batch_size"] = 4
    return DataLoader(*args, **kwargs)


def main() -> None:
    """Apply the same strict precision and producer checks as the original exporter."""
    assert not (Path(__file__).resolve().parents[1] / "LOCAL_GPU_HOLD").exists()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.rnn.fp32_precision = "ieee"
    torch.set_float32_matmul_precision("highest")
    sys.path[:0] = [str(SOURCE / "src"), str(SOURCE / "scripts")]
    name, split = sys.argv[1:3]
    script = SOURCE / "scripts/predict_run.py"
    sys.argv = [str(script), *sys.argv[1:]]
    namespace = runpy.run_path(str(script), run_name="wnn_small_batch_export")
    entry = namespace["main"]
    entry.__globals__["DataLoader"] = bounded_loader
    print("PREDICT_STRICT_FP32 batch4; original source/input/ancestry checks", flush=True)
    with torch.autocast(device_type="cuda", enabled=False):
        entry()
    path = Path(os.environ["WNN_PROJECT_ROOT"]) / "cache" / f"pred_{name}_{split}.json"
    metadata = json.loads(path.read_text())
    metadata["export_batch_size"] = 4
    metadata["batch_override_scope"] = (
        "Independent sequence batching only; frozen producer/model unchanged"
    )
    path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
