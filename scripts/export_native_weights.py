"""Export a trained run to the flat float32 file `native/examples/bench_int8.rs` reads.

    .venv/bin/python scripts/export_native_weights.py bd26_h230_f1_s3_ep3@ep2 out.wbin
    .venv/bin/python scripts/export_native_weights.py ab2_E_mix_h384 out384.wbin

Equal-width one-hidden checkpoints retain the ``WNNWTS01`` layout. Other shapes use
``WNNWTS02`` with seven little-endian uint32 values (n_in, encoder, recurrent, layers,
readout, head_layers, n_out), followed by the same tensor order plus the optional
``head.extra`` matrix and bias. Only the final ``head.2`` scored outputs are kept.
Weights are the checkpoint's own tensors; nothing is refit or requantized here.
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def state_dict(run: str, runs_dir: Path) -> dict[str, np.ndarray]:
    """The run's float32 state dict, trimmed to the two scored outputs."""
    import torch

    name, _, epoch = run.partition("@ep")
    ckpt = f"model_ep{int(epoch)}.pt" if epoch else "model.pt"
    ck = torch.load(runs_dir / name / ckpt, map_location="cpu", weights_only=False)
    if ck.get("args", {}).get("readout_activation", "gelu") != "gelu":
        raise ValueError("non-GELU readout is not supported by the native engine")
    sd = {k: np.ascontiguousarray(v.float().numpy()) for k, v in ck["model"].items()}
    sd["head.2.weight"] = np.ascontiguousarray(sd["head.2.weight"][:2])
    sd["head.2.bias"] = np.ascontiguousarray(sd["head.2.bias"][:2])
    return sd


def write_wbin(sd: dict[str, np.ndarray], path: Path) -> None:
    layers = 0
    while f"gru.weight_ih_l{layers}" in sd:
        layers += 1
    if layers == 0:
        raise ValueError("checkpoint has no recurrent layers")
    n_in = sd["mean"].shape[0]
    encoder = sd["inp.bias"].shape[0]
    recurrent = sd["gru.bias_hh_l0"].shape[0] // 3
    readout = sd["head.0.bias"].shape[0]
    head_layers = 2 if "head.extra.weight" in sd else 1
    n_out = sd["head.2.bias"].shape[0]
    order = ["mean", "std", "inp.weight", "inp.bias"]
    for k in range(layers):
        order += [f"gru.{n}_l{k}" for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
    order += ["head.0.weight", "head.0.bias"]
    if head_layers == 2:
        order += ["head.extra.weight", "head.extra.bias"]
    order += ["head.2.weight", "head.2.bias"]
    legacy = encoder == recurrent == readout and head_layers == 1
    dims = (n_in, recurrent, layers, readout, n_out) if legacy else (
        n_in, encoder, recurrent, layers, readout, head_layers, n_out
    )
    with path.open("wb") as out:
        out.write((b"WNNWTS01" + struct.pack("<5I", *dims)) if legacy else
                  (b"WNNWTS02" + struct.pack("<7I", *dims)))
        for key in order:
            if key not in sd:
                raise ValueError(f"checkpoint is missing {key}")
            out.write(np.ascontiguousarray(sd[key], dtype="<f4").tobytes())


def main() -> None:
    run, path = sys.argv[1], Path(sys.argv[2])
    write_wbin(state_dict(run, Path(os.environ.get("WNN_RUNS_DIR", ROOT / "artifacts" / "runs"))), path)
    print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
