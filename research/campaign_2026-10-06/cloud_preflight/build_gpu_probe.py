from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

PACKET_ID = "wnn-gru-device-preflight-20261006-v1"
DATASETS = [
    "vladkochetov07/wnn-jax-realdata-e463-tpu-20261004-v1",
    "vladkochetov07/wunder-funder",
]
CHECKPOINT_SHA256 = "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e"


def sha256_file(path: Path) -> str:
    """Hash a file without reading checkpoints or source bundles into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_model_source(repository: Path) -> str:
    """Extract the exact root GRU classes used by the Kaggle preflight."""
    source_path = repository / "src/wnn/nn.py"
    source = source_path.read_text()
    tree = ast.parse(source)
    names = {"FixedReadoutMixture", "RNNModel"}
    segments = [
        ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    if len(segments) != len(names) or any(segment is None for segment in segments):
        raise ValueError("Root RNNModel source extraction was incomplete")
    extracted = "\n\n".join(segment for segment in segments if segment is not None)
    return (
        "from __future__ import annotations\n\nimport torch\nimport torch.nn.functional as F\nfrom torch import nn\n\n\n"
        + extracted
        + "\n"
    )


def prepare(repository: Path, packet: Path) -> dict[str, object]:
    """Create one immutable private T4 probe packet."""
    packet.mkdir(parents=True, exist_ok=False)
    model_source = extract_model_source(repository)
    (packet / "model_source.py").write_text(model_source)
    root_model_hash = sha256_file(repository / "src/wnn/nn.py")
    probe_source = (Path(__file__).parent / "gpu_probe.py").read_text()
    probe_source = probe_source.replace(
        'ROOT_NN_PY_SHA256 = ""', f'ROOT_NN_PY_SHA256 = "{root_model_hash}"'
    )
    (packet / "probe.py").write_text(probe_source)
    metadata = {
        "id": f"vladkochetov07/{PACKET_ID}",
        "title": PACKET_ID,
        "code_file": "probe.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": DATASETS,
        "competition_sources": [],
        "kernel_sources": [],
    }
    (packet / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {
        "packet_id": PACKET_ID,
        "metadata": metadata,
        "root_nn_py_sha256": root_model_hash,
        "embedded_model_source_sha256": hashlib.sha256(model_source.encode()).hexdigest(),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "packet_files": {
            path.name: sha256_file(path) for path in sorted(packet.iterdir()) if path.is_file()
        },
        "provider_timeout_seconds": 420,
        "payload_timeout_seconds": 360,
        "hardware_request": "NvidiaTeslaT4; actual device count and model recorded at runtime",
        "budget_cap": "0.25 provider GPU-hours maximum; one notebook, one active GPU worker",
        "scientific_scope": "device, checkpoint-load, numerical-parity, one production-chunk backward/update/export only",
    }


def main() -> None:
    """Build the exact reviewable packet and pre-registration manifest."""
    repository = Path(__file__).resolve().parents[3]
    campaign = Path(__file__).resolve().parent
    packet = campaign / "gpu_packet_v1"
    manifest = prepare(repository, packet)
    manifest_path = campaign / "gpu_prereg_v1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"packet={packet}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
