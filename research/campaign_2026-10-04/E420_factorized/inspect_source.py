from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from wnn.splits import file_sha256  # noqa: E402


def inspect(source: Path, e409_manifest: Path, output: Path) -> None:
    """Record read-only checkpoint shape/epoch evidence and E409 stream identity."""
    checkpoint: dict[str, Any] = torch.load(source, map_location="cpu", weights_only=False)
    manifest = json.loads(e409_manifest.read_text())
    args = checkpoint["args"]
    model = checkpoint["model"]
    expected_shapes = {
        "inp.weight": [384, 211],
        "gru.weight_ih_l0": [1152, 384],
        "gru.weight_hh_l0": [1152, 384],
        "gru.weight_ih_l1": [1152, 384],
        "gru.weight_hh_l1": [1152, 384],
        "head.2.weight": [23, 384],
        "head.2.bias": [23],
    }
    shapes = {name: list(model[name].shape) for name in expected_shapes}
    if shapes != expected_shapes:
        raise ValueError(f"unexpected W384 checkpoint tensor shapes: {shapes}")
    if args["name"] != "c20261003_w384_s3_f0":
        raise ValueError("checkpoint run identity differs")
    if args["hidden"] != 384 or args["layers"] != 2 or args["seed"] != 3:
        raise ValueError("checkpoint model identity differs")
    if args["valid_fold"] != 0 or args["checkpoint_epoch"] != 3 or checkpoint["epoch"] != 3:
        raise ValueError("checkpoint is not the fixed epoch-3 fold-0 model")
    if args["cell"] != "gru" or args["enc"] != "linear" or checkpoint["n_in"] != 211:
        raise ValueError("checkpoint is outside the factorized native support contract")
    if args.get("online_norm") or args.get("heuristics") or args.get("raw_only"):
        raise ValueError("checkpoint feature pipeline is not raw plus event features")
    if args.get("research_features", "none") != "none" or args.get("prop_input", ""):
        raise ValueError("checkpoint has additional model input features")
    if manifest["experiment"] != "E409" or len(manifest["row_groups"]) != 3:
        raise ValueError("E409 manifest does not bind the required three streams")
    if manifest["checkpoints"].get(str(source.parent / "model_ep3.pt")) is None:
        raise ValueError("E409 did not use an epoch-3 checkpoint from the same run")
    report = {
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": file_sha256(source),
        "e409_manifest": str(e409_manifest),
        "e409_manifest_sha256": file_sha256(e409_manifest),
        "e409_model_ep3_sha256": manifest["checkpoints"][str(source.parent / "model_ep3.pt")],
        "e409_data_path": manifest["data"]["path"],
        "e409_data_sha256": manifest["data"]["sha256"],
        "e409_row_groups": manifest["row_groups"],
        "run": args["name"],
        "epoch": checkpoint["epoch"],
        "checkpoint_epoch": args["checkpoint_epoch"],
        "seed": args["seed"],
        "fold": args["valid_fold"],
        "hidden": args["hidden"],
        "layers": args["layers"],
        "input_width": checkpoint["n_in"],
        "output_width_including_auxiliaries": shapes["head.2.bias"][0],
        "scored_outputs": [0, 1],
        "training_precision": args["precision"],
        "eval_ema": args["eval_ema"],
        "model_shapes": shapes,
        "factorization": checkpoint.get("factorization"),
    }
    output.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: inspect_source.py SOURCE E409_MANIFEST OUTPUT")
    inspect(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))


if __name__ == "__main__":
    main()
