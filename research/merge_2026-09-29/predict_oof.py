"""Run one saved merged model on only its untouched development chain fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(ROOT / "src"))
from data import SEQUENCE_LENGTH, VALID_PATH, SequenceReader  # noqa: E402
from model import ModelConfig, Predictor  # noqa: E402

from wnn.splits import valid_row_groups  # noqa: E402

FOLD_CACHE = ROOT / "cache" / "chain_folds.npz"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_groups(fold: int, n_folds: int = 4) -> np.ndarray:
    """Select one chain fold after removing the contract's reserved holdout."""
    dev_groups, _ = valid_row_groups(VALID_PATH)
    data = np.load(FOLD_CACHE)
    valid = data["split"] == "valid"
    fold_by_group = dict(
        zip(
            data["row_group"][valid].tolist(),
            data["fold"][valid].tolist(),
            strict=True,
        )
    )
    assigned = np.array([fold_by_group[int(group)] % n_folds for group in dev_groups])
    return dev_groups[assigned == fold]


@torch.inference_mode()
def predict(
    checkpoint: Path,
    groups: np.ndarray,
    output: Path,
    device: torch.device,
    fold: int,
    n_folds: int,
) -> None:
    """Write predictions for selected row groups in their original order."""
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = Predictor(ModelConfig(**saved["config"]), saved["mean"], saved["std"])
    model.load_state_dict(saved["state_dict"])
    model.to(device).eval()
    reset = int(saved.get("state_reset", 0) or 0)
    chunk = math.gcd(2_000, reset) if reset else 2_000
    reader = SequenceReader(VALID_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        partial.unlink()
    predictions = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.float32,
        shape=(len(groups), SEQUENCE_LENGTH, 2),
    )
    for index, group in enumerate(groups):
        sequence = reader.read(int(group))
        state = model.initial_state(1, device)
        for start in range(0, SEQUENCE_LENGTH, chunk):
            if reset and start and start % reset == 0:
                state = model.initial_state(1, device)
            x = torch.from_numpy(sequence.features[None, start : start + chunk]).to(device)
            pred, state = model(x, state)
            predictions[index, start : start + chunk] = pred[0, :, :2].float().cpu().numpy()
        if index % 32 == 0:
            logging.info("predicted %d/%d row groups", index + 1, len(groups))
    predictions.flush()
    del predictions
    partial.replace(output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": sha256(checkpoint),
                "fold": fold,
                "n_folds": n_folds,
                "row_groups": groups.tolist(),
                "prediction_shape": [len(groups), SEQUENCE_LENGTH, 2],
                "prediction_dtype": "float32",
                "state_reset": reset,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run deterministic OOF inference under the configured device memory cap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.90, torch.cuda.current_device())
    groups = evaluation_groups(args.fold, args.n_folds)
    logging.info("fold=%d development_sequences=%d", args.fold, len(groups))
    predict(args.checkpoint, groups, args.output, device, args.fold, args.n_folds)


if __name__ == "__main__":
    main()
