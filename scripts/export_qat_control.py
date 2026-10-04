"""Export a fixed no-fine-tuning q8 control, preserving exact byte codes and row scales."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from wnn.quantization import PRESETS
from wnn.splits import file_sha256
from wnn.training_qat import effective_qat_state


def export_control(source: Path, output: Path, preset: str) -> None:
    """Quantize an ordinary inference checkpoint once; do not masquerade as a fitted run."""
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if "args" not in checkpoint or "n_in" not in checkpoint:
        raise ValueError(
            "control export requires an inference checkpoint, not training continuation state"
        )
    if checkpoint.get("quantization") is not None:
        raise ValueError(
            "do not repack an already represented checkpoint; preserve its codes/scales"
        )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite control checkpoint: {output}")
    represented, metadata = effective_qat_state(checkpoint["model"], preset, "no-finetune control")
    metadata["source_checkpoint_sha256"] = file_sha256(source)
    metadata["training_updates"] = 0
    metadata["training_rule"] = "none; post-training quantization without updates"
    checkpoint["model"] = represented
    checkpoint["quantization"] = metadata
    checkpoint["provenance"] = {
        **checkpoint.get("provenance", {}),
        "quantization_control": {
            "source": str(source),
            "source_sha256": metadata["source_checkpoint_sha256"],
            "preset": preset,
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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preset", choices=PRESETS, required=True)
    args = parser.parse_args()
    export_control(args.source, args.output, args.preset)


if __name__ == "__main__":
    main()
