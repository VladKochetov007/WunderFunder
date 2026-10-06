from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

PACKET_ID = "wnn-gru-device-preflight-20261006-v2"
DATASETS = [
    "vladkochetov07/wnn-jax-realdata-e463-tpu-20261004-v1",
    "vladkochetov07/wunder-funder",
]
CHECKPOINT_SHA256 = "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e"


def sha256_file(path: Path) -> str:
    """Hash one packet or checkpoint file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_model_source(repository: Path) -> str:
    """Extract the exact root GRU definitions from the checked-out source."""
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
        "from __future__ import annotations\n\n"
        "import torch\nimport torch.nn.functional as F\nfrom torch import nn\n\n\n"
        + extracted
        + "\n"
    )


def prepare(repository: Path, campaign: Path) -> dict[str, object]:
    """Build a self-contained private T4 probe and immutable preregistration data."""
    packet = campaign / "gpu_packet_v2"
    packet.mkdir(parents=True, exist_ok=False)
    model_source = extract_model_source(repository)
    root_model_hash = sha256_file(repository / "src/wnn/nn.py")
    probe_source = (campaign / "gpu_probe.py").read_text()
    probe_source = probe_source.replace(
        'OUTPUT = Path("/kaggle/working/cloud_preflight_gpu")',
        'OUTPUT = Path("/kaggle/working/cloud_preflight_gpu_v2")',
    )
    probe_source = probe_source.replace(
        'ROOT_NN_PY_SHA256 = ""', f'ROOT_NN_PY_SHA256 = "{root_model_hash}"'
    )
    probe_source = probe_source.replace(
        '        "packet": "wnn-gru-device-preflight-20261006-v1",',
        f'        "packet": "{PACKET_ID}",',
    )
    probe_source = probe_source.replace(
        'raise TimeoutError("payload exceeded 360 seconds")',
        'raise TimeoutError("payload exceeded 300 seconds")',
    )
    probe_source = probe_source.replace("signal.alarm(360)", "signal.alarm(300)")
    probe_source = probe_source.replace(
        "        sys.path.insert(0, str(Path(__file__).resolve().parent))\n"
        "        from model_source import RNNModel\n",
        "        model_namespace: dict[str, object] = {}\n"
        '        exec(compile(MODEL_SOURCE, "<embedded-root-gru>", "exec"), model_namespace)\n'
        '        RNNModel = model_namespace["RNNModel"]\n',
    )
    marker = "STARTED = time.monotonic()\n"
    probe_source = probe_source.replace(
        marker,
        "MODEL_SOURCE = " + repr(model_source) + "\n\n" + marker,
    )
    if "from model_source import" in probe_source or "<embedded-root-gru>" not in probe_source:
        raise RuntimeError("self-contained model-source transport was not applied")
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
        "revision_of": "wnn-gru-device-preflight-20261006-v1",
        "revision_reason": "v1 terminal Python failure showed Kaggle script kernels do not expose sibling uploaded files next to /kaggle/src/script.py; v2 embeds the exact model classes in its single uploaded code_file.",
        "provider_timeout_seconds": 360,
        "payload_timeout_seconds": 300,
        "hardware_request": "NvidiaTeslaT4; actual device count, model, VRAM, and host RAM are recorded at runtime",
        "budget_cap": "This v2 is capped at 360 provider seconds; cumulative first-stage GPU cap is 0.25 device-hours including v1; one notebook, one active GPU worker.",
        "scientific_scope": "actual-device inventory, checkpoint-load, short CPU/GPU parity, one synthetic production-chunk backward/update/export only; no scientific fit or score",
        "failure_durability": "startup and host admission precede framework import; exception phase and traceback persist in terminal.json and complete.json from finally; provider logs are downloaded separately",
    }


def main() -> None:
    """Create the new versioned packet without modifying v1 evidence."""
    repository = Path(__file__).resolve().parents[3]
    campaign = Path(__file__).resolve().parent
    manifest = prepare(repository, campaign)
    output = campaign / "gpu_prereg_v2.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"packet={campaign / 'gpu_packet_v2'}")
    print(f"manifest={output}")


if __name__ == "__main__":
    main()
