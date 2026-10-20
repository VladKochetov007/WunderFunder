"""Create the predeclared equal epoch-3/4/5 H512 checkpoint mean."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

from h512_swa345_validation import (
    ADMISSION_PATH,
    ADMISSION_SHA,
    DEST_RUN,
    HELPER_PATH,
    HELPER_SHA,
    ROOT_RUN,
    TRAINING_COMMIT,
    sha256,
    validate_inputs,
)


def derive(root: Path, manifest_path: Path) -> dict[str, Any]:
    """Write the create-only mean checkpoint after root accepts the F0 fit."""
    bound = validate_inputs(root, manifest_path)
    root = bound["root"]
    helper = bound["helper"]
    checkpoints = bound["checkpoints"]
    destination = root / "artifacts/runs" / DEST_RUN
    if destination.parent != root / "artifacts/runs" or os.path.lexists(destination):
        raise FileExistsError(f"derived output must be new: {destination}")
    averaged = helper.mean_state([checkpoint["model"] for checkpoint in checkpoints])
    derivation = {
        "schema": "wnn.h512-swa345-derivation.v1",
        "method": "equal_parameter_mean_epochs_3_4_5",
        "source_run_name": ROOT_RUN,
        "destination_run_name": DEST_RUN,
        "source_checkpoints": bound["items"],
        "input_manifest": {
            "path": str(bound["manifest_path"].relative_to(root)),
            "sha256": sha256(bound["manifest_path"]),
        },
        "fit_terminal_receipt": bound["fit_receipt"],
        "source_provenance": bound["provenance_item"],
        "admission_path": str(ADMISSION_PATH),
        "admission_sha256": ADMISSION_SHA,
        "source_training_commit": TRAINING_COMMIT,
        "source_run_identity": bound["identity"],
        "helper_path": str(HELPER_PATH),
        "helper_sha256": HELPER_SHA,
        "derivation_code_sha256": sha256(Path(__file__).resolve()),
        "accumulation": "float64 mean followed by cast to source dtype",
        "fixed_buffers": "mean/std and non-floating tensors must be bit-identical and are preserved",
        "training_run": False,
        "quality_evaluated": False,
    }
    derived = copy.deepcopy(checkpoints[-1])
    derived["model"] = averaged
    derived["weight_derivation"] = derivation
    derived["provenance"]["weight_derivation"] = derivation
    derived["provenance"].pop("checkpoint_sha256", None)
    destination.mkdir()
    output = destination / "model_ep5.pt"
    with output.open("xb") as stream:
        helper.torch.save(derived, stream)
    os.link(output, destination / "model.pt")
    output_sha = sha256(output)
    provenance = copy.deepcopy(bound["source_provenance"])
    provenance["weight_derivation"] = derivation
    provenance["selected_epoch"] = 5
    provenance["checkpoint_sha256"] = {"model_ep5.pt": output_sha, "model.pt": output_sha}
    provenance_path = destination / "provenance.json"
    with provenance_path.open("x") as stream:
        stream.write(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    receipt = {
        **derivation,
        "outputs": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in (output, destination / "model.pt", provenance_path)
        },
    }
    with (destination / "weight_derivation_receipt.json").open("x") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main() -> None:
    """Run only the fixed, no-fit/no-evaluation derivation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="post-fit input hash manifest")
    parser.add_argument("--root", type=Path, required=True, help="project root")
    args = parser.parse_args()
    print(json.dumps(derive(args.root, args.manifest), indent=2))


if __name__ == "__main__":
    main()
