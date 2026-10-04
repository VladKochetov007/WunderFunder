from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

PACKET = Path(__file__).resolve().parent
KERNEL_DIR = PACKET / "kernel_push"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    code_path = KERNEL_DIR / "kernel.py"
    metadata_path = KERNEL_DIR / "kernel-metadata.json"
    source = code_path.read_text()
    ast.parse(source, filename=str(code_path))
    metadata = json.loads(metadata_path.read_text())
    plan = json.loads((PACKET / "plan.v1.json").read_text())
    w384 = Path("research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v3/bundle")
    bundle_manifest = json.loads((w384 / "bundle-manifest.json").read_text())
    checks = {
        "kernel_private": metadata["is_private"] is True,
        "gpu_disabled": metadata["enable_gpu"] is False,
        "tpu_disabled": metadata["enable_tpu"] is False,
        "internet_disabled": metadata["enable_internet"] is False,
        "kernel_id_title_match": metadata["id"].split("/", 1)[-1] == metadata["title"],
        "dataset_inputs_exact": metadata["dataset_sources"] == plan["provider_inputs"],
        "kernel_slug_matches": metadata["id"] == "vladkochetov07/wnn-w384-io-diagnostic-cpu-20261004-v1-runner",
        "bundle_archive_pin_matches": bundle_manifest["archive_sha256"] in source,
        "bundle_manifest_pin_matches": sha(w384 / "bundle-manifest.json") in source,
        "hash_block_size_matches_controller": "HASH_BLOCK_BYTES = 8 * 1024**2" in source,
        "portable_hash_block_size_matches_source": "block_bytes=1 << 20" in source,
        "calls_frozen_portable_loader": "splits.load_campaign_split(Path(split_value), root)" in source,
        "two_worker_cpu_affinity": "portable_check_processes" not in source and "cpus=record[\"cpu_affinity\"][:2]" in source,
        "rss_cap_4gib": "RSS_LIMIT_BYTES = 4 * 1024**3" in source,
        "whole_wall_cap_3600": "JOB_LIMIT_SECONDS = 3600" in source,
        "worker_cap_3500": "WORKER_LIMIT_SECONDS = 3500" in source,
        "receipt_never_short_circuits": '"eligible_to_skip_shared_portability_hashes":False' in source,
        "torch_only_import": "import torch" in source and "import jax" not in source,
        "no_training_entrypoint_invoked": '"scripts/kaggle_sequence.py"' not in source and '"scripts/train_rnn.py"' not in source,
        "reports_no_copy": '"copy_performed":False' in source,
        "no_copy_api_calls": all(
            marker not in source
            for marker in ("shutil.copy", "shutil.copy2", "shutil.copyfile", "shutil.copytree")
        ),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    validation = {
        "status": "static_packet_valid_pending_root_review",
        "scope": "stdlib-only AST/source/metadata/plan checks; no data read, JAX/Torch import, or local ML operation",
        "checks": checks,
        "kernel_id": metadata["id"],
        "metadata_sha256": sha(metadata_path),
        "kernel_sha256": sha(code_path),
        "plan_sha256": sha(PACKET / "plan.v1.json"),
        "expected_bundle_manifest_sha256": sha(w384 / "bundle-manifest.json"),
        "expected_bundle_archive_sha256": bundle_manifest["archive_sha256"],
        "dataset_sources": metadata["dataset_sources"],
        "provider_mutation": "none; source-only review candidate",
        "training_or_score": False,
        "scientific_result": False,
    }
    (PACKET / "validation.prepush.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
