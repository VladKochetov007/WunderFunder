from __future__ import annotations

import ast
import hashlib
import json
import zipfile
from pathlib import Path

PACKET = Path(__file__).resolve().parent
UPLOAD = PACKET / "upload_dataset"
BUNDLE = PACKET / "build_inputs" / "source_bundle"
KERNEL_DIR = PACKET / "kernel_push_runner"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    archive = UPLOAD / "source_bundle.zip"
    manifest_path = UPLOAD / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    dataset = json.loads((UPLOAD / "dataset-metadata.json").read_text())
    kernel_path = KERNEL_DIR / "kernel.py"
    kernel_metadata = json.loads((KERNEL_DIR / "kernel-metadata.json").read_text())
    kernel_text = kernel_path.read_text()
    members = {}
    with zipfile.ZipFile(archive) as zipped:
        assert set(zipped.namelist()) == set(manifest["files"])
        for name in zipped.namelist():
            raw = zipped.read(name)
            descriptor = manifest["files"][name]
            assert len(raw) == descriptor["size_bytes"]
            assert sha(raw) == descriptor["sha256"]
            assert raw == (BUNDLE / name).read_bytes()
            if name.endswith(".py"):
                ast.parse(raw.decode("utf-8"), filename=name)
            members[name] = {"size_bytes": len(raw), "sha256": sha(raw)}
    assert sha(archive.read_bytes()) == manifest["archive_sha256"]
    assert manifest["dataset_id"] == dataset["id"]
    assert manifest["private"] is True
    assert all(manifest[name] is False for name in ("enable_gpu", "enable_tpu", "enable_internet"))
    assert kernel_metadata["dataset_sources"] == [manifest["dataset_id"]]
    assert kernel_metadata["is_private"] is True
    assert all(kernel_metadata[name] is False for name in ("enable_gpu", "enable_tpu", "enable_internet"))
    assert kernel_metadata["id"].endswith("-runner")
    assert kernel_metadata["title"] == kernel_metadata["id"].split("/", 1)[-1]
    assert manifest["archive_sha256"] in kernel_text
    assert sha(manifest_path.read_bytes()) in kernel_text
    combo_test = (BUNDLE / "tests/test_distributed_combo.py").read_text()
    worker = (BUNDLE / "worker.py").read_text()
    combo_ast = ast.parse(combo_test)
    state_ast = ast.parse((BUNDLE / "tests/test_distributed_state_boundary.py").read_text())
    count = sum(
        isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        for tree in (combo_ast, state_ast)
        for node in tree.body
    )
    assert count == 4
    assert 'summary["tests"] == 4' in worker
    assert 'summary["skips"] == 0' in worker
    assert "expect_clipping=False, weight_decay=0.0" in combo_test
    assert "global gradient norm" in combo_test
    assert "np.zeros_like(np.asarray(gradients))" in combo_test
    assert "np.asarray(initial)" in combo_test
    assert "np.isfinite(np.asarray(first_moments))" in combo_test
    assert 'OUT = Path("/kaggle/working/e465_sharding_cpu_v3")' in worker
    assert "WORKER_SECONDS = 600" in kernel_text
    assert "JOB_SECONDS = 780" in kernel_text
    assert "RSS_LIMIT_BYTES = 4 * 1024**3" in kernel_text
    assert "/kaggle/working/e465_sharding_cpu_v3" in kernel_text
    assert "CUDA_VISIBLE_DEVICES\": \"\"" in kernel_text

    old = PACKET.parent / "E465_sharding_cpu_v2" / "build_inputs" / "source_bundle"
    unchanged = [
        "src/wnn_jax/bridge.py",
        "src/wnn_jax/distributed_combo.py",
        "src/wnn_jax/rnn.py",
        "tests/test_distributed_state_boundary.py",
    ]
    for name in unchanged:
        assert (old / name).read_bytes() == (BUNDLE / name).read_bytes(), name
    validation = {
        "status": "static_packet_valid_pending_root_review",
        "validation_scope": "stdlib-only archive/member/source/resource/privacy checks; no JAX/Torch import or tests run locally",
        "dataset_id": manifest["dataset_id"],
        "kernel_id": kernel_metadata["id"],
        "archive_sha256": sha(archive.read_bytes()),
        "manifest_sha256": sha(manifest_path.read_bytes()),
        "kernel_sha256": sha(kernel_path.read_bytes()),
        "kernel_metadata_sha256": sha((KERNEL_DIR / "kernel-metadata.json").read_bytes()),
        "members": members,
        "test_count": count,
        "implementation_members_byte_identical_to_failed_v2": unchanged,
        "allzero_case_contract": "finite clamped loss; exactly zero global norm and gradients; weight decay zero so parameters remain unchanged; optimizer count exactly one; finite AdamW moments; no expectation of clipping for zero-mass input",
        "private_offline_cpu_only": True,
        "worker_seconds": 600,
        "whole_job_seconds": 780,
        "aggregate_rss_limit_bytes": 4294967296,
        "provider_dataset_upload": "not uploaded; source-only review candidate",
        "provider_kernel_push": "not pushed; root review pending",
        "scientific_result": False,
    }
    (PACKET / "validation.prepush.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
