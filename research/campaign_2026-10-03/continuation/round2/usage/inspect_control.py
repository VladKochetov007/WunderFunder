from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]


def digest(path: Path) -> str:
    """Stream a file digest without loading its entire contents."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    """Inspect exact checkpoint metadata and cached index headers before sampling."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    checkpoint_path = REPO / "artifacts/runs/c20261001_d256_s3_f0/model_ep3.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    split_path = Path(args["split_manifest"])
    sidecar = json.loads((REPO / "cache/pred_c20261001_d256_s3_f0_valid.json").read_text())
    with zipfile.ZipFile(REPO / "cache/valid_targets.npz") as archive:
        headers = {}
        for name in archive.namelist():
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
                headers[name] = {
                    "shape": shape,
                    "fortran": fortran,
                    "dtype": str(dtype),
                    "payload_offset": stream.tell(),
                }
    manifest = json.loads(split_path.read_text())
    summary = {
        "checkpoint": str(checkpoint_path.relative_to(REPO)),
        "checkpoint_sha256": digest(checkpoint_path),
        "cached_sidecar_checkpoint_sha256": sidecar["checkpoint_sha256"],
        "checkpoint_epoch": checkpoint.get("epoch"),
        "n_in": checkpoint["n_in"],
        "n_out": checkpoint["model"]["head.2.weight"].shape[0],
        "args": args,
        "readout_present": checkpoint.get("readout") is not None,
        "split_manifest": str(split_path.relative_to(REPO)),
        "split_sha256": digest(split_path),
        "evaluation_count": len(manifest["evaluation"]["valid"]),
        "cache_headers": headers,
        "cached_predictions": {key: value for key, value in sidecar.items() if key != "row_groups"},
        "cached_prediction_shape": np.load(
            REPO / "cache/pred_c20261001_d256_s3_f0_valid.npy", mmap_mode="r"
        ).shape,
        "source_sha256": {
            name: digest(REPO / name)
            for name in [
                "scripts/predict_run.py",
                "scripts/train_rnn.py",
                "src/wnn/nn.py",
                "src/wnn/features.py",
                "src/wnn/splits.py",
            ]
        },
    }
    if summary["checkpoint_sha256"] != sidecar["checkpoint_sha256"]:
        raise ValueError("cached checkpoint is not exact ep3")
    if summary["split_sha256"] != args["split_manifest_sha256"]:
        raise ValueError("checkpoint split identity mismatch")
    (ROOT / "control_identity.json").write_text(json.dumps(summary, indent=2))
    logging.info(
        json.dumps(
            {
                key: summary[key]
                for key in [
                    "checkpoint_epoch",
                    "n_in",
                    "n_out",
                    "readout_present",
                    "cache_headers",
                    "checkpoint_sha256",
                    "evaluation_count",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
