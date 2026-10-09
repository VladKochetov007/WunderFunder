from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

CANDIDATE_ROOT = Path(__file__).resolve().parent
BASE_SEALER = CANDIDATE_ROOT.parent / "d02_muon_loss_successor" / "seal_d02_candidate.py"
CONTROL_KERNEL = "vladkochetov07/wnn-loss-pair-gpu-20261009-v4"
R4_READER_SHA256 = "35431e48583b1523188b3d6084fa12be23e7379560a323411655bf561507a693"
R4_PACKET_SHA256 = "dd1f57dc288b8b75df349f150170c6fc60cdff474f2f6f14897baa60b9fcc16a"
R4_KERNEL_SHA256 = "86346f9c1addefbd38d3dcec52b18f514698a9514e7917aeb6081cc32f7eb039"
R4_ASSET_MANIFEST_SHA256 = "8ef360536ca117232198a4ee6dd8fabf072fb7bfad9812a83972d8bfb750f584"
R4_ROOT_REVIEW_SHA256 = "701987320ad045a4688e633cef19d59fb8b940cb1a75f53b089deaeb0a070e47"
R4_COLLECTION_RECEIPT_SHA256 = "1bed2d7befbdd25b487f15f0b8788b8e514783058a2c7d5f683b31ff70a7500c"
R4_ATTEMPT_RESULT_SHA256 = "b708bf12e32673b9f0e516dfa486733f7c356797e7dc0ae4f4c65178acf09232"
R4_RUNTIME_CACHE_RECEIPT_SHA256 = "f6452f4b03abf7c66a219008458e431a8c1b6a2ffba961f63e5f57d1e936076f"
R4_MODEL_SHA256 = {
    "wmse": "71fcba3d9eaff3710f8fd42f3778ef74cd425f11977668a5575c21563a82c309",
    "huber_d1": "b61b5026386273b11db871f06f09eda7654af5814c066e850274e32d9cdfca77",
}
R4_SOURCE_COMMITS = {
    "reader_fix": "1bc0fbacf53cdc08a8c4664c6f66f38f8980092c",
    "packet_kernel": "cc59806075863f41fe08c7a73a14c289a4bd4c39",
    "opaque_source_payload": "1f4e08f02ff3c8abbc20ecc7e7dc369427810143",
}


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_base_sealer():
    spec = importlib.util.spec_from_file_location("d02_r1_sealer", BASE_SEALER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load frozen base sealer: {BASE_SEALER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The R1 package implementation is reused with explicit R2 identifiers and local template.
    module.__file__ = str(Path(__file__).resolve())
    module.KERNEL_TITLE = "WNN D02 Muon Loss GPU 20261009 V2"
    module.KERNEL_ID = "vladkochetov07/wnn-d02-muon-loss-gpu-20261009-v2"
    module.SOURCE_DATASET_TITLE = "WNN D02 Muon Loss Code 20261009 R2"
    module.SOURCE_DATASET_ID = "vladkochetov07/wnn-d02-muon-loss-code-20261009-r2"
    module.CONTROL_GPU_KERNEL = CONTROL_KERNEL
    module.KERNEL_FILENAME = "wnn-d02-muon-loss-gpu-20261009-v2.py"
    module.BOOTSTRAP_PATH = Path("kernel") / module.KERNEL_FILENAME
    return module


def seal_candidate(candidate: Path, source_commit: str) -> dict[str, Any]:
    base = load_base_sealer()
    packet = base.seal_candidate(candidate, source_commit)
    packet["schema"] = "wnn.d02.kaggle-muon-loss-packet.v2"
    packet["r4_control_parent"] = {
        "schema": "wnn.d02.r4-control-parent-binding.v1",
        "state": "fits_verified_export_recovery_pending",
        "kernel_id": CONTROL_KERNEL,
        "packet_sha256": R4_PACKET_SHA256,
        "kernel_source_sha256": R4_KERNEL_SHA256,
        "source_asset_manifest_sha256": R4_ASSET_MANIFEST_SHA256,
        "source_commits": R4_SOURCE_COMMITS,
        "reader_module_sha256": R4_READER_SHA256,
        "completed_fits_review": {
            "path": "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ROOT_GPU_R4_COMPLETED_FITS_REVIEW_R1.json",
            "sha256": R4_ROOT_REVIEW_SHA256,
            "state": "completed_fits_verified_export_pending",
        },
        "operator_terminal_collection_receipt_sha256": R4_COLLECTION_RECEIPT_SHA256,
        "terminal_attempt_result_sha256": R4_ATTEMPT_RESULT_SHA256,
        "cache_runtime_validation_receipt_sha256": R4_RUNTIME_CACHE_RECEIPT_SHA256,
        "arms": {
            arm: {
                "checkpoint_sha256": model_sha,
                "checkpoint_size_bytes": 3_922_155,
                "fixed_epoch": 1,
                "completed_updates": 220,
                "skipped_updates": 0,
                "normalization_sha256": "d096115cb59a7462608d5d11df29ba77142aac03527f26fbb2a8534077e57849",
                "sample_orders_sha256": "cbb972a29278617890ac3e51d9f4d3cb954d392192aef3ebe044b8a71a8ec523",
                "prediction_export": "pending R5 export-only recovery",
            }
            for arm, model_sha in R4_MODEL_SHA256.items()
        },
        "shared_prefit_and_schedule": {
            "initial_state_identical_across_arms": True,
            "sample_orders_sha256": "cbb972a29278617890ac3e51d9f4d3cb954d392192aef3ebe044b8a71a8ec523",
            "schedule_rows": 220,
            "updates_verified_per_arm": 220,
            "normalization_sha256": "d096115cb59a7462608d5d11df29ba77142aac03527f26fbb2a8534077e57849",
        },
        "training_contract_sha256": None,
        "control_contract_admitted": False,
        "quality_score": None,
        "note": "Do not fit or enable push until R5 restores the export/training contract and root reviews its exact hash and mounted artifacts.",
    }
    packet["experiment"]["batch_tail_semantics"] = (
        "final batch has 30 real sequences plus 2 zero-weight padded sequences"
    )
    base.write_json(candidate / "gpu_loss_pair.packet.json", packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE_ROOT)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    packet = seal_candidate(args.candidate, args.source_commit)
    packet_bytes = (args.candidate / "gpu_loss_pair.packet.json").read_bytes()
    code_manifest_sha = packet["source"]["code_manifest_sha256"]
    print(
        json.dumps(
            {
                "packet_id": packet["packet_id"],
                "packet_sha256": sha256_bytes(packet_bytes),
                "code_manifest_sha256": code_manifest_sha,
                "root_admission_sha256": packet["root_admission_sha256"],
                "control_kernel": packet["control_gpu_kernel"],
                "control_contract_sha256": None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
