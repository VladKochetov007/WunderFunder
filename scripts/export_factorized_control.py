"""Export a zero-update SVD rank control with effective weights and original factors."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from train_rnn import make_model

from wnn.splits import file_sha256
from wnn.training_factorized import configure_factorized, factorized_inference_state


def export_control(source: Path, output: Path, rank: int) -> None:
    """Record truncation alone, including the closest dense parameter-budget control."""
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if "args" not in checkpoint or "n_in" not in checkpoint:
        raise ValueError(
            "rank control requires an inference checkpoint, not raw continuation state"
        )
    if checkpoint.get("factorization") or checkpoint.get("quantization"):
        raise ValueError("rank control requires an unfactorized float parent")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite control checkpoint: {output}")
    if rank < 1:
        raise ValueError("rank control requires a positive rank")
    model = make_model(
        checkpoint["args"], checkpoint["n_in"], checkpoint["model"]["head.2.bias"].numel()
    )
    model.load_state_dict(checkpoint["model"])
    configure_factorized(model, rank)
    represented, metadata = factorized_inference_state(model, "zero-update SVD control")
    metadata["source_checkpoint_sha256"] = file_sha256(source)
    metadata["training_updates"] = 0
    checkpoint["model"] = represented
    checkpoint["factorization"] = metadata
    checkpoint["provenance"] = {
        **checkpoint.get("provenance", {}),
        "factorization_control": {
            "source": str(source),
            "source_sha256": metadata["source_checkpoint_sha256"],
            "rank": rank,
            "training_updates": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    torch.save(checkpoint, temporary)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=128)
    args = parser.parse_args()
    export_control(args.source, args.output, args.rank)


if __name__ == "__main__":
    main()
