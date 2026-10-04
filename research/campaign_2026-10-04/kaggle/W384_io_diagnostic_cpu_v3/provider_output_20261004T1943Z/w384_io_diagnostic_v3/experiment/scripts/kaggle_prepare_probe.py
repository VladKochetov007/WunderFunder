from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


def prepare(root: Path, destination: Path) -> None:
    """Bind the exact repository class into a private standalone probe."""
    source = (root / "src/wnn/nn.py").read_text()
    tree = ast.parse(source)
    node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RNNModel"
    )
    model = ast.get_source_segment(source, node)
    if model is None:
        raise ValueError("Unable to extract repository RNNModel")
    probe = (root / "scripts/kaggle_tpu_probe.py").read_text()
    digest = hashlib.sha256(model.encode()).hexdigest()
    probe = probe.replace('MODEL_SOURCE = ""', f"MODEL_SOURCE = {model!r}")
    probe = probe.replace('MODEL_SHA256 = ""', f"MODEL_SHA256 = {digest!r}")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "probe.py").write_text(probe)
    metadata = {
        "id": "vladkochetov07/wnn-tpu-probe-20261004",
        "title": "WNN TPU Probe 20261004",
        "code_file": "probe.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": True,
        "enable_internet": False,
        "machine_shape": "TpuV5E8",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    provenance = {
        "model_class_sha256": digest,
        "model_file_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "probe_sha256": hashlib.sha256(probe.encode()).hexdigest(),
        "timeout_seconds": 900,
        "explicit_tensor_limit_bytes": 128 * 1024**2,
        "api_documentation": "https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md",
    }
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    """Create one immutable private kernel directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(Path(__file__).resolve().parents[1], args.destination)


if __name__ == "__main__":
    main()
