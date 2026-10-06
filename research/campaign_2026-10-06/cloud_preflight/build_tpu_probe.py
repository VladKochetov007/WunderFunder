from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

PACKET_ID = "wnn-e463-real-tpu-fullshape-preflight-20261006-v1"
DATASETS = [
    "vladkochetov07/wnn-jax-realdata-e463-tpu-20261004-v1",
    "vladkochetov07/wunder-funder",
]
SOURCE_SHA256 = {
    "model_ep3.pt": "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e",
    "model_ep3.jax.npz": "677f604fd86d993c3d4a66a10638755253bfe52f1f578009dbd781c73fe6d04a",
    "model_ep3.jax.npz.json": "3e65d2a88352d22bc90d88b829f05beee3faa7a66b9de41c7507f521aab12668",
    "wnn_jax/rnn.py": "294eff11d906fe434dba2c4d0ed4741ad2191f756c4a35204474cb229ff939dc",
}


def sha256_file(path: Path) -> str:
    """Hash a packet file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(campaign: Path) -> dict[str, object]:
    """Write a one-file private Kaggle TPU packet and its preregistration manifest."""
    packet = campaign / "tpu_packet_v1"
    packet.mkdir(parents=True, exist_ok=False)
    source = campaign / "tpu_probe.py"
    shutil.copyfile(source, packet / "probe.py")
    metadata = {
        "id": f"vladkochetov07/{PACKET_ID}",
        "title": PACKET_ID,
        "code_file": "probe.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": True,
        "enable_internet": False,
        "machine_shape": "TpuV5E8",
        "dataset_sources": DATASETS,
        "competition_sources": [],
        "kernel_sources": [],
    }
    (packet / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    packet_hashes = {
        path.name: sha256_file(path) for path in sorted(packet.iterdir()) if path.is_file()
    }
    return {
        "packet_id": PACKET_ID,
        "metadata": metadata,
        "packet_files": packet_hashes,
        "verified_private_inputs": {
            "dataset_sources": DATASETS,
            "epoch3_torch_checkpoint_sha256": SOURCE_SHA256["model_ep3.pt"],
            "epoch3_jax_npz_sha256": SOURCE_SHA256["model_ep3.jax.npz"],
            "epoch3_jax_npz_sidecar_sha256": SOURCE_SHA256["model_ep3.jax.npz.json"],
            "exact_wnn_jax_rnn_source_sha256": SOURCE_SHA256["wnn_jax/rnn.py"],
        },
        "revision_reason": "Fresh E463 device packet after historical dead-keyword, CPU-fallback/RSS, and empty-log failures; asserts actual 8-device TPU backend, writes pre-framework startup/host telemetry, runs bounded child with stdout/stderr log and terminal status, checks checkpoint/source hashes and full shape, and refuses unknown/low HBM headroom.",
        "provider_timeout_seconds": 600,
        "payload_timeout_seconds": 540,
        "worker_timeout_seconds": 480,
        "first_stage_tpu_cap_hours": 0.25,
        "maximum_new_provider_tpu_seconds": 600,
        "hardware_request": "TpuV5E8; requires exactly eight local TPU devices and records actual device kinds and per-device HBM stats",
        "shape_gate": {
            "global_batch": 16,
            "per_tpu_batch": 2,
            "chunk_rows": 2000,
            "input_width": 211,
            "hidden": 256,
            "layers": 2,
            "output_width": 23,
            "updates": 1,
            "objective": "synthetic finite diagnostic objective only; no score or scientific fit",
        },
        "headroom": "host and every TPU must retain at least 20 percent free capacity at start; after the full-shape update every TPU peak use must remain at or below 80 percent of reported HBM capacity",
        "durability": "startup/host records precede JAX import; child writes inventory and stage outputs, parent captures worker stdout/stderr and always writes terminal.json/complete.json in finally; bounded child is terminated if parent deadline arrives",
    }


def main() -> None:
    """Create only the new TPU packet and reviewable prelaunch manifest."""
    campaign = Path(__file__).resolve().parent
    manifest = prepare(campaign)
    output = campaign / "tpu_prereg_v1.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"packet={campaign / 'tpu_packet_v1'}")
    print(f"manifest={output}")


if __name__ == "__main__":
    main()
