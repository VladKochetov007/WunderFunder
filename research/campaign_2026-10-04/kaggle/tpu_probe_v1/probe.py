from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Filled with the exact RNNModel class by kaggle_prepare_probe.py, never credentials.
MODEL_SOURCE = 'class RNNModel(nn.Module):\n    """Input projection -> stacked GRU -> head fed by GRU output and the projection skip."""\n\n    def __init__(\n        self, n_in: int, hidden: int = 256, layers: int = 2, dropout: float = 0.0, n_out: int = 2,\n        cell: str = "gru", enc: str = "linear", tok_dim: int = 32, att_layers: int = 1, conv_k: int = 16,\n    ):\n        super().__init__()\n        self.register_buffer("mean", torch.zeros(n_in))\n        self.register_buffer("std", torch.ones(n_in))\n        self.inp = nn.Linear(n_in, hidden)\n        # Optional per-row encoders, added as residuals to the linear projection (research screening only;\n        # the native engine supports enc="linear").\n        self.enc = enc\n        if enc == "mlp":\n            self.enc_norm = nn.LayerNorm(hidden)\n            self.enc_up = nn.Linear(hidden, 4 * hidden)\n            self.enc_down = nn.Linear(2 * hidden, hidden)\n        elif enc == "tokens":\n            self.groups = token_groups(n_in)\n            self.tok_in = nn.ModuleList(nn.Linear(b - a, tok_dim) for a, b in self.groups)\n            self.cls = nn.Parameter(torch.zeros(1, 1, tok_dim))\n            layer = nn.TransformerEncoderLayer(tok_dim, 4, 2 * tok_dim, dropout=0.0, activation="gelu",\n                                               batch_first=True, norm_first=True)\n            self.att = nn.TransformerEncoder(layer, att_layers, enable_nested_tensor=False)\n            self.tok_out = nn.Linear(2 * tok_dim, hidden)\n        elif enc == "conv":\n            self.conv_k = conv_k\n            self.conv_dw = nn.Conv1d(hidden, hidden, conv_k, groups=hidden)\n            self.conv_pw = nn.Linear(hidden, hidden)\n        elif enc != "linear":\n            raise ValueError(enc)\n        rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]\n        # Attribute name kept as ``gru`` so existing checkpoints and exporters stay valid.\n        self.gru = rnn(\n            hidden,\n            hidden,\n            num_layers=layers,\n            batch_first=True,\n            dropout=dropout if layers > 1 else 0.0,\n        )\n        self.head = nn.Sequential(\n            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, n_out)\n        )\n\n    def encode(self, xn):\n        u = self.inp(xn)\n        if self.enc == "tokens":\n            b, t, _ = xn.shape\n            flat = xn.reshape(b * t, -1)\n            toks = torch.stack([lin(flat[:, a:c]) for lin, (a, c) in zip(self.tok_in, self.groups)], dim=1)\n            toks = self.att(torch.cat([self.cls.expand(b * t, -1, -1), toks], dim=1))\n            pooled = torch.cat([toks[:, 0], toks[:, 1:].mean(1)], dim=-1)\n            u = u + self.tok_out(pooled).reshape(b, t, -1)\n        z = F.gelu(u)\n        if self.enc == "mlp":\n            a, g = self.enc_up(self.enc_norm(z)).chunk(2, dim=-1)\n            z = z + self.enc_down(F.gelu(a) * torch.sigmoid(g))\n        elif self.enc == "conv":\n            c = F.pad(z.transpose(1, 2), (self.conv_k - 1, 0))  # causal; zero left context at chunk starts\n            z = z + F.gelu(self.conv_pw(self.conv_dw(c).transpose(1, 2)))\n        return z\n\n    def forward(self, x, h=None):\n        z = self.encode((x - self.mean) / self.std)\n        o, h = self.gru(z, h)\n        return self.head(torch.cat([o, z], dim=-1)), h'
MODEL_SHA256 = '0dea50e5aebea7d6670a65b00177e4d2ff6b92fd3adf1af7c0a37d7b8a74d3c3'
OUTPUT = Path("/kaggle/working/tpu_probe")


def write_json(path: Path, value: object) -> None:
    """Persist each stage before another framework can fail."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def package_versions() -> dict[str, str]:
    """Read installed versions without initializing accelerator runtimes."""
    result = {}
    for name in ("jax", "jaxlib", "libtpu", "torch", "torch-xla", "tensorflow", "numpy"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "absent"
    return result


def jax_probe() -> dict[str, object]:
    """Inventory topology and run tiny GRU-equation gradient/Adam checks."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    devices = jax.devices()
    inventory = []
    for device in devices:
        inventory.append(
            {
                "id": device.id,
                "platform": device.platform,
                "kind": device.device_kind,
                "process_index": device.process_index,
                "coords": str(getattr(device, "coords", None)),
                "core_on_chip": str(getattr(device, "core_on_chip", None)),
                "memory_stats": device.memory_stats(),
            }
        )
    write_json(OUTPUT / "jax_inventory.json", inventory)
    if not devices or any(device.platform != "tpu" for device in devices):
        raise RuntimeError("Requested TPU kernel did not expose TPU devices")
    rng = np.random.default_rng(271828)
    weights = rng.normal(0, 0.03, (2, 16, 48)).astype("float32")
    inputs = rng.normal(0, 0.1, (8, 1, 16)).astype("float32")
    results = []
    for dtype in (jnp.float32, jnp.bfloat16):
        w = jax.device_put(jnp.asarray(weights, dtype=dtype), devices[0])
        x = jax.device_put(jnp.asarray(inputs, dtype=dtype), devices[0])

        def loss_fn(params: object, sequence: object) -> object:
            def cell(h: object, xt: object) -> tuple[object, object]:
                xi = xt @ params[0]
                hi = h @ params[1]
                xr, xz, xn = jnp.split(xi, 3, axis=-1)
                hr, hz, hn = jnp.split(hi, 3, axis=-1)
                reset, update = jax.nn.sigmoid(xr + hr), jax.nn.sigmoid(xz + hz)
                candidate = jnp.tanh(xn + reset * hn)
                state = (1 - update) * candidate + update * h
                return state, state

            _, output = jax.lax.scan(cell, jnp.zeros((1, 16), dtype=sequence.dtype), sequence)
            return jnp.mean(output.astype(jnp.float32) ** 2)

        start = time.monotonic()
        loss, gradient = jax.jit(jax.value_and_grad(loss_fn))(w, x)
        gradient.block_until_ready()
        # First Adam update, FP32 moments, no external optimizer dependency.
        grad = gradient.astype(jnp.float32)
        updated = w.astype(jnp.float32) - 1e-3 * grad / (jnp.sqrt(grad * grad) + 1e-8)
        updated.block_until_ready()
        if not np.isfinite(np.asarray(updated)).all():
            raise RuntimeError("Nonfinite JAX GRU Adam update")
        results.append(
            {
                "dtype": str(dtype),
                "loss": float(loss),
                "gradient_norm": float(jnp.linalg.norm(grad)),
                "seconds": time.monotonic() - start,
                "explicit_tensor_bytes": weights.nbytes + inputs.nbytes,
                "memory_stats": devices[0].memory_stats(),
            }
        )
    return {
        "devices": inventory,
        "checks": results,
        "scope": "tiny GRU equations; no repository-model parity claim",
    }


def torch_xla_probe() -> dict[str, object]:
    """Run the exact repository RNNModel class at tiny dimensions on XLA."""
    import torch
    import torch.nn.functional as F
    import torch_xla
    import torch_xla.core.xla_model as xm
    from torch import nn

    if hashlib.sha256(MODEL_SOURCE.encode()).hexdigest() != MODEL_SHA256:
        raise ValueError("Repository model source hash mismatch")
    namespace = {"torch": torch, "F": F, "nn": nn}
    exec(compile(MODEL_SOURCE, "bound_repository_RNNModel", "exec"), namespace)
    device = torch_xla.device()
    checks = []
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(271828)
        model = namespace["RNNModel"](211, hidden=16, layers=2, n_out=23)
        model = model.to(device=device, dtype=dtype)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        inputs = torch.zeros((1, 8, 211), device=device, dtype=dtype)
        started = time.monotonic()
        output, state = model(inputs)
        loss = output.float().square().mean()
        loss.backward()
        optimizer.step()
        xm.mark_step()
        loss_value = float(loss.cpu())
        if not torch.isfinite(output.cpu()).all() or not torch.isfinite(state.cpu()).all():
            raise RuntimeError("Nonfinite actual GRU output/state")
        checks.append(
            {
                "dtype": str(dtype),
                "loss": loss_value,
                "output_shape": list(output.shape),
                "seconds": time.monotonic() - started,
                "memory_info": xm.get_memory_info(device),
            }
        )
    return {
        "device": str(device),
        "checks": checks,
        "model_source_sha256": MODEL_SHA256,
        "scope": "actual class, hidden16; not a training/parity pilot",
    }


def stage(name: str) -> None:
    """Record unsupported operations as failures without installing packages."""
    started = time.monotonic()
    try:
        result = jax_probe() if name == "jax" else torch_xla_probe()
        record = {"status": "passed", "result": result}
    except Exception as exc:
        record = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    record["seconds"] = time.monotonic() - started
    record["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(OUTPUT / f"{name}.json", record)


def main() -> None:
    """Bound separate framework processes; preserve inventory even on timeout."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) == 2:
        stage(sys.argv[1])
        return
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    host = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "affinity_count": len(os.sched_getaffinity(0)),
        "host_memory": {k: memory[k].strip() for k in ("MemTotal", "MemAvailable")},
        "versions": package_versions(),
        "model_source_sha256": MODEL_SHA256,
        "initial_explicit_tensor_limit_bytes": 128 * 1024**2,
        "note": "Compiler/runtime overhead is additional; no preallocation or large tensors requested",
    }
    write_json(OUTPUT / "host.json", host)
    env = os.environ.copy()
    env.update(
        {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.01",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "PJRT_DEVICE": "TPU",
        }
    )
    for name in ("jax", "torch_xla"):
        with (OUTPUT / f"{name}.log").open("w") as log:
            try:
                completed = subprocess.run(
                    [sys.executable, __file__, name],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=300,
                    check=False,
                )
                write_json(OUTPUT / f"{name}_process.json", {"returncode": completed.returncode})
            except subprocess.TimeoutExpired:
                write_json(OUTPUT / f"{name}_process.json", {"status": "timeout", "seconds": 300})
    write_json(
        OUTPUT / "complete.json", {"status": "probe_finished", "scientific_pilot_authorized": False}
    )


if __name__ == "__main__":
    main()
