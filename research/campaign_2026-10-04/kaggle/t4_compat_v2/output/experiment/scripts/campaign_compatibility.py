"""Bounded device/GRU checks; this is infrastructure evidence, never a score run."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {
        "kind": "compatibility_only",
        "device_requested": args.device,
        "python": platform.python_version(),
        "cgroup": Path("/proc/self/cgroup").read_text(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "WNN_THREADS",
                "WNN_GPU_MEMORY_BYTES",
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
        "checks": [],
    }
    try:
        import torch

        from wnn.nn import RNNModel
        from wnn.resources import configure_torch

        configure_torch()
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but unavailable")
        report.update(
            torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version()
        )
        if args.device == "cuda":
            prop = torch.cuda.get_device_properties(0)
            report["hardware"] = {
                "name": prop.name,
                "total_bytes": prop.total_memory,
                "capability": list(torch.cuda.get_device_capability()),
            }
        torch.manual_seed(3407)
        # A small parity case precedes a bounded representative recurrent step.
        cases = [(1, 64, 16)] if args.device == "cpu" else [(1, 64, 16), (16, 1000, 256)]
        for batch, rows, hidden in cases:
            model = RNNModel(79, hidden=hidden, layers=2, n_out=6).to(args.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            x = torch.randn(batch, rows, 79, device=args.device)
            y = torch.randn(batch, rows, 6, device=args.device)
            initial = model.head[-1].weight.detach().clone()
            with torch.autocast(args.device, enabled=args.device == "cuda", dtype=torch.float16):
                prediction, _ = model(x)
                loss = (prediction.float() - y).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite GRU loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("nonfinite GRU gradient")
            optimizer.step()
            if torch.equal(initial, model.head[-1].weight):
                raise RuntimeError("GRU step did not change the head")
            model.eval()
            with torch.no_grad():
                whole, _ = model(x)
                left, state = model(x[:, : rows // 2])
                right, _ = model(x[:, rows // 2 :], state)
                split = torch.cat([left, right], dim=1)
            torch.testing.assert_close(whole, split, rtol=2e-5, atol=2e-6)
            check = {
                "batch": batch,
                "rows": rows,
                "hidden": hidden,
                "loss": float(loss.detach()),
                "grad_norm": float(grad_norm),
                "split_max_abs": float((whole - split).abs().max()),
            }
            if args.device == "cuda":
                torch.cuda.synchronize()
                check["peak_allocator_bytes"] = torch.cuda.max_memory_allocated()
                check["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            report["checks"].append(check)
            del model, optimizer, x, y, prediction, loss, whole, left, right, split, state
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
