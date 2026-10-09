from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

CANDIDATE_ROOT = Path(__file__).resolve().parent
BASE_SEALER = CANDIDATE_ROOT / "seal_d02_candidate_base.py"
KERNEL_TITLE = "WNN D02 Muon Loss GPU 20261009 V5"
KERNEL_ID = "vladkochetov07/wnn-d02-muon-loss-gpu-20261009-v5"
SOURCE_DATASET_TITLE = "WNN D02 Muon Loss Code 20261009 R5"
SOURCE_DATASET_ID = "vladkochetov07/wnn-d02-muon-loss-code-20261009-r5"
CONTROL_KERNEL = "vladkochetov07/wnn-loss-pair-export-recovery-20261009-r1"
CONTROL_CONTRACT_SHA256 = "5b63ed39aa6046e9f02878e7dcb4b48982240bffa45ec09b7466dfb9eeac63f7"
CONTROL_OUTPUT_RELATIVE = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "cloud_delivery/gpu_loss_pair_r5_export/terminal_collect_r1/attempt_001/outputs"
)
R5_REVIEW_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ROOT_GPU_R5_TERMINAL_REVIEW_R1.json"
)
R5_REVIEW_SHA256 = "60e9770de432a6b39d3ae41c818d60b2d47b5298f630b33ee3483a0e7fecc7f9"
R5_RECEIPT_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "cloud_delivery/gpu_loss_pair_r5_export/GPU_R5_TERMINAL_OPERATOR_RECEIPT_R1.json"
)
R5_RECEIPT_SHA256 = "4c85b6090692f4ac452cdcdfc7488443b1432f110ca00a9b452d53b33c786b4e"
R5_GRADE_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "grading_r1/LOSS_PAIR_R5_GRADE_R2.json"
)
R5_GRADE_SHA256 = "94c8bb8031be7e4776a7b80f694d852551165bb0b52f7f9df5d287c8b3b46f34"
R5_EXPORT_RECEIPT_SHA256 = "791a3bbdde92034ec83ff1e220d698367731276c4ad9faa723c15ad16640f986"
R4_FAILURE_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "d02_muon_loss_successor_r4_admitted_r2/GPU_D02_R4_TERMINAL_OPERATOR_RECEIPT_R1.json"
)
R4_FAILURE_SHA256 = "546c267622e6968d318e6ff5d27bb1e4011e3766c3431d9f34420e9b3bafeba8"
R5_SHARED_ROLES = {
    "cache_manifest": {
        "path": "shared/cache_manifest.json",
        "sha256": "7400b75504f592011577dd3b126e9813749c21ce9126e5c4fa04c8e6b61e8f3b",
        "size_bytes": 12722,
    },
    "cache_reader_profile": {
        "path": "shared/cache_reader_profile_runtime.json",
        "sha256": "fbac76445ff91dbe8d8336bc3ab2bc3e2ddddd3c4be90d25550c478ea774091e",
        "size_bytes": 3160,
    },
    "cache_validation_receipt": {
        "path": "shared/cache_validation_runtime.json",
        "sha256": "bf43ade9f08fb0447fc1df18dbb91f1d4723c828eaa3c14a7f5bffd73049f6ed",
        "size_bytes": 3421,
    },
    "device_profile_gate": {
        "path": "shared/device_profile_gate.json",
        "sha256": "70b8b40e994a74387a0b7e9055a532f21fbbb2ea516ead3ef3201d1ad3c575ae",
        "size_bytes": 3204,
    },
    "fold0_asset_validation_receipt": {
        "path": "shared/fold0_assets_validation.json",
        "sha256": "93ecabd180a608d72f111b9886e8804cb084d4ace694de57d188b745c33dbd43",
        "size_bytes": 6204,
    },
    "initial_state": {
        "path": "shared/initial_state.npz",
        "sha256": "10e4b56002b52faaaf55dbfcbd28b37f479d2dbd69e727740250dbb90d873af1",
        "size_bytes": 3908280,
    },
    "normalization": {
        "path": "shared/normalization.npz",
        "sha256": "d096115cb59a7462608d5d11df29ba77142aac03527f26fbb2a8534077e57849",
        "size_bytes": 2188,
    },
    "normalization_metadata": {
        "path": "shared/normalization.json",
        "sha256": "0e4762ebb51f95589b93e79545c769376b42afaa2e7069f81b199164dd59cf96",
        "size_bytes": 4274,
    },
    "onecycle_trace": {
        "path": "shared/schedule_trace.json",
        "sha256": "a46a5beb2484b3549b6fdb926b885ecbcaf4a61b1cec60dd3750760eafff27de",
        "size_bytes": 67395,
    },
    "profile_inputs": {
        "path": "shared/profile_inputs.npz",
        "sha256": "f042ae97ddebac87c1023758ccd84d9b2923e5cc6ae3ac878596e84bfde9a1c6",
        "size_bytes": 54592760,
    },
    "profile_inputs_receipt": {
        "path": "shared/profile_inputs_receipt.json",
        "sha256": "6620c5f34c13fd9310a4aaa7d27e0631b9bdf8d05ceb0459863cfed1746596ab",
        "size_bytes": 4509,
    },
    "sample_orders": {
        "path": "shared/sample_orders.json",
        "sha256": "cbb972a29278617890ac3e51d9f4d3cb954d392192aef3ebe044b8a71a8ec523",
        "size_bytes": 148958,
    },
    "source_stage_receipt": {
        "path": "shared/source_stage_receipt.json",
        "sha256": "557cd0a61f1129f811dc104398c3c9c7295c3343a1c9a302018fcb588c0c1de7",
        "size_bytes": 34299,
    },
    "source_verification": {
        "path": "shared/source_verification.json",
        "sha256": "1ccd91ccec85bd157cbd0244ae9032316b88d781a0caf19daf7a027f3a908567",
        "size_bytes": 4511,
    },
}
R5_ARMS = {
    "wmse": {
        "run_name": "loss_pair_r3_wmse",
        "checkpoint_sha256": "71fcba3d9eaff3710f8fd42f3778ef74cd425f11977668a5575c21563a82c309",
        "checkpoint_size_bytes": 3922155,
        "epoch1_sha256": "da9763cf34280224561f6115baf413fa9779bf9381a590088a9a04e8bc01bc90",
        "provenance_sha256": "c3e99c0c310288450f86fd007f8dfb9ffe80a5acd13857ebce455be83803477c",
        "prediction_sha256": "8b695aac4efd43c3a10ea398a836a3460769963e198f879bd8beb10a4b01fd5d",
        "prediction_size_bytes": 74880128,
        "prediction_sidecar_sha256": "7ebf2025b7110fa4458cfd863abd5fd0c726cd177df460d8e0b5d80f276ac4c8",
    },
    "huber_d1": {
        "run_name": "loss_pair_r3_huber_d1",
        "checkpoint_sha256": "b61b5026386273b11db871f06f09eda7654af5814c066e850274e32d9cdfca77",
        "checkpoint_size_bytes": 3922155,
        "epoch1_sha256": "6424b906c5b894598665843f043d7cdb28774211f8d9a0eb2474d20e37562390",
        "provenance_sha256": "18e71073b7ac70fdac9aa57a3bd15fc0c55122472f9ace3eeeda0ab3befe8fe5",
        "prediction_sha256": "cf763bcc1ed3dd6d2f299fd647b7967112d25a8cbafb9e13b56885739591d5f3",
        "prediction_size_bytes": 74880128,
        "prediction_sidecar_sha256": "ba391743f7e490cb57cf250e3026256a58b44aa75787c1884777ebbb4f54201b",
    },
}


def _load_base_sealer() -> ModuleType:
    """Load the repository's frozen D02 packager with R5 identifiers."""
    spec = importlib.util.spec_from_file_location("d02_r5_base", BASE_SEALER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base sealer: {BASE_SEALER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__file__ = str(Path(__file__).resolve())
    module.KERNEL_TITLE = KERNEL_TITLE
    module.KERNEL_ID = KERNEL_ID
    module.SOURCE_DATASET_TITLE = SOURCE_DATASET_TITLE
    module.SOURCE_DATASET_ID = SOURCE_DATASET_ID
    module.CONTROL_GPU_KERNEL = CONTROL_KERNEL
    module.KERNEL_FILENAME = "wnn-d02-muon-loss-gpu-20261009-v5.py"
    module.BOOTSTRAP_PATH = Path("kernel") / module.KERNEL_FILENAME
    return module


def _source_manifest_sha256(base: ModuleType, source_commit: str) -> str:
    """Compute the frozen code-manifest digest before binding its admission."""
    code_root = CANDIDATE_ROOT / "code"
    rows = [
        {
            "path": path.relative_to(code_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": base.sha256_file(path),
        }
        for path in base.source_files(code_root)
    ]
    manifest = {
        "schema": "wnn.kaggle-loss-pair-code-manifest",
        "format_version": 1,
        "source_commit": source_commit,
        "trainer_base_commit": base.CPU_SOURCE_COMMIT,
        "files": rows,
    }
    raw = base.canonical_json(manifest)
    return hashlib.sha256(raw).hexdigest()


def _pending_admission(source_manifest_sha256: str) -> dict[str, object]:
    """Bind the exact producer while keeping D02 provider actions disabled."""
    return {
        "schema": "wnn.d02.root-admission.v1",
        "kernel_id": KERNEL_ID,
        "push_allowed": False,
        "conditional_fit_allowed": False,
        "source_code_manifest_sha256": source_manifest_sha256,
        "gpu_control_contract_sha256": CONTROL_CONTRACT_SHA256,
        "control_gpu_kernel": CONTROL_KERNEL,
        "runtime_seconds": 3300,
    }


def _r5_parent_binding() -> dict[str, object]:
    """Record the exact reviewed R5 AdamW producer and its fixed screen result."""
    return {
        "schema": "wnn.d02.r5-control-parent-binding.v1",
        "state": "root_reviewed_complete_fits_exports_and_fixed_grade",
        "kernel_id": CONTROL_KERNEL,
        "kernel_version": 1,
        "training_contract": {
            "path_under_kernel_output": "out/training_contract.json",
            "sha256": CONTROL_CONTRACT_SHA256,
            "cache_manifest_sha256": "7400b75504f592011577dd3b126e9813749c21ce9126e5c4fa04c8e6b61e8f3b",
            "cache_key": "949587f77794935c08589bd5d5e81641062de547526b92fc1b01bc4109cd82a6",
            "membership_sha256": "dbd9ea6d7b0b752495745ea1ebd4bd858af9305c40f0ada7de7386c5359f3acd",
            "validation_receipt_sha256": "bf43ade9f08fb0447fc1df18dbb91f1d4723c828eaa3c14a7f5bffd73049f6ed",
            "parsed_cache_reader_sha256": "35431e48583b1523188b3d6084fa12be23e7379560a323411655bf561507a693",
        },
        "root_terminal_review": {"path": R5_REVIEW_PATH, "sha256": R5_REVIEW_SHA256},
        "operator_terminal_receipt": {
            "path": R5_RECEIPT_PATH,
            "sha256": R5_RECEIPT_SHA256,
            "export_receipt_sha256": R5_EXPORT_RECEIPT_SHA256,
            "files_verified": 149,
            "bytes_verified": 298070207,
        },
        "fixed_grade": {
            "path": R5_GRADE_PATH,
            "sha256": R5_GRADE_SHA256,
            "metric_id": "global_wp_clip2_v1",
            "evaluation_groups": 468,
            "components": 141,
            "wmse_wp": 0.6060019230032425,
            "huber_d1_wp": 0.6012391352737978,
            "equal_blend_wp": 0.6054802014069163,
            "interpretation": (
                "Exposed fold-0 development panel; fixed equal blend was not searched. "
                "The result does not establish public transfer or reject the Huber loss."
            ),
        },
        "shared_artifacts": R5_SHARED_ROLES,
        "arms": {
            name: {
                **values,
                "fixed_epoch": 1,
                "completed_updates": 220,
                "skipped_updates": 0,
                "normalization_sha256": R5_SHARED_ROLES["normalization"]["sha256"],
                "sample_orders_sha256": R5_SHARED_ROLES["sample_orders"]["sha256"],
                "prediction": {
                    "path": f"out/arms/{name}/predictions_valid.npy",
                    "sha256": values["prediction_sha256"],
                    "size_bytes": values["prediction_size_bytes"],
                    "sidecar_sha256": values["prediction_sidecar_sha256"],
                },
            }
            for name, values in R5_ARMS.items()
        },
        "training_repeated": False,
        "d02_push_admitted": False,
    }


def seal_candidate(source_commit: str) -> dict[str, object]:
    """Seal the first-batch profile producer fix without changing the fit recipe."""
    base = _load_base_sealer()
    (CANDIDATE_ROOT / "kernel").mkdir(exist_ok=True)
    source_manifest_sha256 = _source_manifest_sha256(base, source_commit)
    admission_path = CANDIDATE_ROOT / "root_admission.json"
    base.write_json(admission_path, _pending_admission(source_manifest_sha256))
    packet = base.seal_candidate(CANDIDATE_ROOT, source_commit, admission_path)
    packet["schema"] = "wnn.d02.kaggle-muon-loss-packet.v5"
    packet["r5_control_parent"] = _r5_parent_binding()
    packet["control_gpu_kernel"] = CONTROL_KERNEL
    packet["recovery_parent"] = {
        "kernel_id": "vladkochetov07/wnn-d02-muon-loss-gpu-20261009-v4",
        "failure_evidence": {"path": R4_FAILURE_PATH, "sha256": R4_FAILURE_SHA256},
        "diagnosis": (
            "The WMSE profile gate wrote profile_inputs.npz, but the Huber gate did not: "
            "the trainer's first-batch export condition was WMSE-only while the worker "
            "required the artifact for both loss-specific gates."
        ),
        "fit_attempted": False,
        "scientific_recipe_changed": False,
    }
    packet["root_admission_required_before_dataset_creation_or_kernel_push"] = True
    packet["experiment"]["control_basis"] = {
        "training_contract_sha256": CONTROL_CONTRACT_SHA256,
        "kernel": CONTROL_KERNEL,
        "retrain": False,
        "science_recipe_change": False,
    }
    packet["experiment"]["batch_tail_semantics"] = (
        "final batch has 30 real sequences plus 2 zero-weight padded sequences"
    )
    packet_path = CANDIDATE_ROOT / "gpu_loss_pair.packet.json"
    base.write_json(packet_path, packet)
    return packet


def main() -> int:
    """Seal code, asset manifest, bootstrap and packet from one committed source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    packet = seal_candidate(args.source_commit)
    packet_path = CANDIDATE_ROOT / "gpu_loss_pair.packet.json"
    print(
        json.dumps(
            {
                "packet_id": packet["packet_id"],
                "packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
                "source_code_manifest_sha256": packet["source"]["code_manifest_sha256"],
                "bootstrap_sha256": packet["kernel"]["code_file_sha256"],
                "metadata_sha256": packet["kernel"]["metadata_sha256"],
                "control_kernel": CONTROL_KERNEL,
                "control_contract_sha256": CONTROL_CONTRACT_SHA256,
                "push_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
