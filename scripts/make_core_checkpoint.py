from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE_ROWS = 23


def derive_core(root: Path, source_run: str, epoch: int, destination_run: str) -> dict[str, object]:
    """Remove only family rows and bind the derived checkpoint's actual identity."""
    source_dir = root / "artifacts/runs" / source_run
    destination = root / "artifacts/runs" / destination_run
    if destination.exists():
        raise FileExistsError(destination)
    checkpoint_name = f"model_ep{epoch}.pt"
    source_checkpoint = source_dir / checkpoint_name
    source_provenance = source_dir / "provenance.json"
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    if state["head.2.weight"].shape[0] != BASE_ROWS + 8:
        raise ValueError("expected 31 head rows in the source family checkpoint")
    if state["head.2.bias"].shape != (BASE_ROWS + 8,):
        raise ValueError("source family bias does not match the head rows")
    state["head.2.weight"] = state["head.2.weight"][:BASE_ROWS].clone()
    state["head.2.bias"] = state["head.2.bias"][:BASE_ROWS].clone()
    derivation = {
        "source_run": source_run,
        "source_epoch": epoch,
        "source_checkpoint_sha256": hashlib.sha256(source_checkpoint.read_bytes()).hexdigest(),
        "source_provenance_sha256": hashlib.sha256(source_provenance.read_bytes()).hexdigest(),
        "dropped_family_rows": 8,
    }
    provenance = json.loads(source_provenance.read_text())
    for holder in (
        checkpoint.get("args"),
        checkpoint.get("provenance", {}).get("args"),
        provenance.get("args"),
    ):
        if isinstance(holder, dict):
            holder["distill_family_head"] = False
            holder["distill_family_weight"] = 0.0
    checkpoint["core_derivation"] = derivation
    embedded = checkpoint.get("provenance")
    if isinstance(embedded, dict):
        embedded.pop("checkpoint_sha256", None)
        embedded["core_derivation"] = derivation
    provenance["core_derivation"] = derivation
    destination.mkdir(parents=True)
    torch.save(checkpoint, destination / checkpoint_name)
    checkpoint_hash = hashlib.sha256((destination / checkpoint_name).read_bytes()).hexdigest()
    # The original training hashes remain in the immutable source provenance.
    # This sidecar describes the new bytes; the binary cannot contain its own hash.
    provenance["checkpoint_sha256"] = {checkpoint_name: checkpoint_hash}
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return {"destination": destination_run, "checkpoint_sha256": checkpoint_hash, **derivation}


def main() -> None:
    """Derive a new core run without modifying the source or an existing run."""
    parser = argparse.ArgumentParser(description=derive_core.__doc__)
    parser.add_argument("source_run")
    parser.add_argument("epoch", type=int)
    parser.add_argument("destination_run")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(derive_core(args.root, args.source_run, args.epoch, args.destination_run)))


if __name__ == "__main__":
    main()
