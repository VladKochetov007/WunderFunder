"""Report deterministic synthetic matrix and native Q8 storage for student width choices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import wnn_native
except ImportError as exc:  # pragma: no cover - exercised in the built-extension environment
    raise SystemExit("build the native extension before running this probe") from exc

from scripts.export_native_weights import write_wbin
from wnn.nn import RNNModel


def matrix_parameter_count(
    input_width: int,
    encoder: int,
    recurrent: int,
    readout: int,
    layers: int,
    head_layers: int,
) -> int:
    """Count matrix weights, excluding biases and FP32 recurrent state."""
    return (
        input_width * encoder
        + 3 * recurrent * (encoder + recurrent)
        + 6 * (layers - 1) * recurrent**2
        + (recurrent + encoder) * readout
        + 2 * readout
        + (readout**2 if head_layers == 2 else 0)
    )


def packed_dyn8_bytes(rows: int, columns: int) -> int:
    """Mirror DynQ8 storage: 64-row superblocks, 4-column groups and two FP32 row arrays."""
    superblocks = (rows + 63) // 64
    return superblocks * ((columns + 3) // 4) * 256 + superblocks * 64 * 8


def measure(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = RNNModel(
        211,
        hidden=args.recurrent_width,
        layers=args.layers,
        encoder_width=args.encoder_width,
        recurrent_width=args.recurrent_width,
        readout_width=args.readout_width,
        head_layers=args.head_layers,
    ).eval()
    tensors = {
        name: np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
        for name, value in model.state_dict().items()
    }
    weight_elements = sum(
        value.size
        for name, value in tensors.items()
        if name == "inp.weight"
        or name.startswith("gru.weight_")
        or name.startswith("head.")
        and name.endswith(".weight")
    )
    formula_elements = matrix_parameter_count(
        211,
        args.encoder_width,
        args.recurrent_width,
        args.readout_width,
        args.layers,
        args.head_layers,
    )
    if weight_elements != formula_elements:
        raise AssertionError(f"matrix formula {formula_elements} != tensors {weight_elements}")

    native = wnn_native.Predictor(tensors, precision="q8vnni_portable", fast_activations=False)
    q8 = native.quantization_codes()
    matrices: dict[str, dict[str, int]] = {}
    for key, codes in sorted(q8.items()):
        if not key.endswith(".codes"):
            continue
        name = key.removesuffix(".codes")
        rows, columns = codes.shape
        scales = q8[f"{name}.scales"]
        matrices[name] = {
            "rows": int(rows),
            "columns": int(columns),
            "logical_codes_and_scales_bytes": int(codes.nbytes + scales.nbytes),
            "native_packed_bytes": packed_dyn8_bytes(rows, columns),
        }
    storage = {key: int(value) for key, value in native.storage_bytes.items()}
    packed_total = sum(value["native_packed_bytes"] for value in matrices.values())
    if packed_total != storage["dynamic_q8_weights"]:
        raise AssertionError(
            f"matrix packed bytes {packed_total} != native storage {storage['dynamic_q8_weights']}"
        )
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.tobytes())
    bias_elements = sum(value.size for name, value in tensors.items() if name.endswith(".bias"))
    receipt = {
        "format": "wnn.student-independent-width-probe",
        "version": 1,
        "synthetic_weights": True,
        "seed": args.seed,
        "sequence_rows": None,
        "input_width": 211,
        "encoder_width": args.encoder_width,
        "recurrent_width": args.recurrent_width,
        "readout_width": args.readout_width,
        "layers": args.layers,
        "head_layers": args.head_layers,
        "matrix_weight_elements_formula": formula_elements,
        "matrix_weight_elements_tensor_count": int(weight_elements),
        "matrix_weights_fp32_bytes": int(4 * weight_elements),
        "bias_elements_fp32": int(bias_elements),
        "all_exported_tensor_bytes_fp32": int(sum(value.nbytes for value in tensors.values())),
        "weights_sha256": digest.hexdigest(),
        "native_precision": native.precision,
        "native_weight_bytes": int(native.weight_bytes),
        "native_storage_bytes": storage,
        "dynamic_q8_matrices": matrices,
        "claim_limit": "storage-only synthetic weights; no fit, quality or runtime claim",
    }
    if args.export_binary is not None:
        args.export_binary.parent.mkdir(parents=True, exist_ok=True)
        write_wbin(tensors, args.export_binary)
        exported = args.export_binary.read_bytes()
        receipt["wbin_fixture"] = {
            "path": str(args.export_binary),
            "sha256": hashlib.sha256(exported).hexdigest(),
            "bytes": len(exported),
            "format_magic": exported[:8].decode("ascii"),
            "transport": "WNNWTS02 independent-width native example format",
        }
    if args.raw_rows is not None:
        args.raw_rows.parent.mkdir(parents=True, exist_ok=True)
        raw = np.random.default_rng(args.seed + 1).normal(size=(20_000, 112)).astype("<f4")
        raw.tofile(args.raw_rows)
        receipt["raw_callback_fixture"] = {
            "path": str(args.raw_rows),
            "rows": int(raw.shape[0]),
            "raw_columns": int(raw.shape[1]),
            "sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
            "bytes": int(raw.nbytes),
            "synthetic": True,
        }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-width", type=int, required=True)
    parser.add_argument("--recurrent-width", type=int, required=True)
    parser.add_argument("--readout-width", type=int, required=True)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--head-layers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seed", type=int, default=2203)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--export-binary", type=Path)
    parser.add_argument("--raw-rows", type=Path)
    args = parser.parse_args()
    if min(args.encoder_width, args.recurrent_width, args.readout_width, args.layers) <= 0:
        parser.error("all widths and layer count must be positive")
    receipt = measure(args)
    serialized = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
