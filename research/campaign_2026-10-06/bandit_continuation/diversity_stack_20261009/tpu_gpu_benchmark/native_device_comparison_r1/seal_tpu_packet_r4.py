"""Seal a small deterministic source closure and fail-closed Kaggle TPU packet."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path
from typing import Any

BASE_COMMIT = "d5daaa0f90b0362877522703c7e17aa78c19cac9"
R2_SOURCE_COMMIT = "9f20145cc4e91a326525324a8266decb912959ae"
R3_SOURCE_COMMIT = "e5f761ce"
NATIVE_R4_SOURCE_COMMIT = "58f17259ae2a359165e6a0698d187c852183a241"
R4_SOURCE_COMMIT = "1bc0fbacf53cdc08a8c4664c6f66f38f8980092c"
R4_PACKET_COMMIT = "cc59806075863f41fe08c7a73a14c289a4bd4c39"
R5_SOURCE_COMMIT = "2b10067281ed7d6d2e5f6ae722072c49d793dde0"
R5_REHEARSAL_COMMIT = "9d380d37ffdf1f3d4da0cd5a0a15527f6cfd4129"
R4_READER_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "cloud_delivery/gpu_loss_pair_r4/code/src/wnn/parsed_sequence_cache_reader.py"
)
R4_READER_SHA256 = "35431e48583b1523188b3d6084fa12be23e7379560a323411655bf561507a693"
R4_PACKET_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "cloud_delivery/gpu_loss_pair_r4/gpu_loss_pair.packet.json"
)
R4_KERNEL_METADATA_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "cloud_delivery/gpu_loss_pair_r4/kernel/kernel-metadata.json"
)
R4_SOURCE_ASSET_MANIFEST_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "cloud_delivery/gpu_loss_pair_r4/source_asset/assets-manifest.json"
)
R4_BRIDGE_TEST_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "cloud_delivery/gpu_loss_pair_r4/tests/test_parsed_cache_receipt_bridge.py"
)
R4_BRIDGE_TEST_SHA256 = "5602f765c37b6e199b0f399191138dbda8a815c457d9b4172e758c56318a4c9f"
R4_KERNEL_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "cloud_delivery/gpu_loss_pair_r4/kernel/wnn-loss-pair-gpu-20261009-v4.py"
)
R4_PACKET_SHA256 = "dd1f57dc288b8b75df349f150170c6fc60cdff474f2f6f14897baa60b9fcc16a"
R4_KERNEL_SHA256 = "86346f9c1addefbd38d3dcec52b18f514698a9514e7917aeb6081cc32f7eb039"
R4_KERNEL_METADATA_SHA256 = "ced255b82699ff6d354b12e7b7687d122814a3b75f80827213dadbc7830e567d"
R4_SOURCE_ASSET_MANIFEST_SHA256 = "8ef360536ca117232198a4ee6dd8fabf072fb7bfad9812a83972d8bfb750f584"
R5_PACKET_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "cloud_delivery/gpu_loss_pair_r5_export/gpu_loss_pair_export_recovery.packet.json"
)
R5_KERNEL_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "cloud_delivery/gpu_loss_pair_r5_export/kernel/"
    "wnn-loss-pair-export-recovery-20261009-r1.py"
)
R5_KERNEL_METADATA_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "cloud_delivery/gpu_loss_pair_r5_export/kernel/kernel-metadata.json"
)
R5_SOURCE_ASSET_MANIFEST_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "cloud_delivery/gpu_loss_pair_r5_export/source_asset_stage/assets-manifest.json"
)
R5_EXPORT_TEST_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "cloud_delivery/gpu_loss_pair_r5_export/tests/test_gpu_loss_pair_r5_export.py"
)
R5_REHEARSAL_TEST_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "cloud_delivery/gpu_loss_pair_r5_export/tests/test_gpu_loss_pair_r5_shared_rehearsal.py"
)
R5_REHEARSAL_RECEIPT_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "cloud_delivery/gpu_loss_pair_r5_export/shared_artifact_rehearsal_receipt_R1.json"
)
R5_PACKET_SHA256 = "ce58293e8124ef7e51d1e32c773c286ada5ca180c3e81a308bc31a95c3c108cc"
R5_KERNEL_SHA256 = "ddb6401a6bf3b59a8242e387d11da9c7dbc1713236e085bb3cade27de775fc19"
R5_KERNEL_METADATA_SHA256 = "e57d5b86eca15818b18c0fd11d01190f5f4cd00ec868187f7ef89a1527dc9494"
R5_SOURCE_ASSET_MANIFEST_SHA256 = "0d9d5b6d53017b8ef837c369d7aaa47ce2a676c37a1165cda71d6735b446bcfd"
R5_EXPORT_TEST_SHA256 = "5ee4ef5bd535881bb3e2ed6649d368c4f0f14cc5e84a7b1ea2550800bff20fce"
R5_REHEARSAL_TEST_SHA256 = "b517c3cc6f3a317b944288a51dabbdb311029b65ac63d62f0ae3df43a75adb40"
R5_REHEARSAL_RECEIPT_SHA256 = "92a55122f6b067302858028583c9f71da99e8b7669be47e56d81b3128b37b3bb"
R5_KERNEL = "vladkochetov07/wnn-loss-pair-export-recovery-20261009-r1"
R5_RECOVERY_ASSET_SHA256 = "0d9d5b6d53017b8ef837c369d7aaa47ce2a676c37a1165cda71d6735b446bcfd"
R5_OUTPUT_CONTRACT_SHA256 = "5b63ed39aa6046e9f02878e7dcb4b48982240bffa45ec09b7466dfb9eeac63f7"
R5_ROOT_REVIEW_SHA256 = "60e9770de432a6b39d3ae41c818d60b2d47b5298f630b33ee3483a0e7fecc7f9"
PREFIX = "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/tpu_gpu_benchmark/native_device_comparison_r1"
KERNEL_TITLE = "WNN Native Device TPU 20261009 V4"
KERNEL_ID = "vladkochetov07/wnn-native-device-tpu-20261009-v4"
BASE_FILES = tuple(
    f"src/wnn/{name}.py"
    for name in (
        "__init__",
        "data",
        "splits",
        "parsed_sequence_cache",
        "features",
        "gmetric",
        "nn",
        "event_content",
        "numeric_ple",
    )
)
OWN_FILES = tuple(
    f"{PREFIX}/{name}"
    for name in (
        "jax_step.py",
        "tpu_worker.py",
        "run_device_parity.py",
        "run_native_profile.py",
        "rnn_native.py",
        "tpu_packet_runtime.py",
        "tests/test_feature_transform_contract.py",
        "tests/test_gru256_feature_adapter_parity.py",
        "tests/test_native_precision_tiny.py",
        "tests/test_native_admission.py",
    )
)
CORE_FILE = "research/campaign_2026-10-04/staging/jax_tpu_e465/src/wnn_jax/rnn.py"
R4_EXTRA_FILES = (
    f"{PREFIX}/tpu_packet_runtime_r4.py",
    f"{PREFIX}/fit_headroom_gate_r4.py",
    f"{PREFIX}/tests/test_native_r4_binding.py",
    f"{PREFIX}/tests/test_fit_headroom_gate_r4.py",
    f"{PREFIX}/CONTRACT_R4.md",
)
R3_PACKET_PATH = f"{PREFIX}/sealed_candidate_r3/packet.json"
R3_SOURCE_MANIFEST_PATH = f"{PREFIX}/sealed_candidate_r3/source_manifest.json"
R3_SOURCE_ARCHIVE_PATH = f"{PREFIX}/sealed_candidate_r3/source.tar.gz"
R3_PACKET_SHA256 = "31667abbd6b7741293741fcbcebe9567ef725f549767348cdaafa2e755dfbd5d"
R3_SOURCE_MANIFEST_SHA256 = "a7cd50522db8a9dd221fd8a637e1f5042eb356b16ac324df2068dc3296036f50"
R3_SOURCE_ARCHIVE_SHA256 = "d19355d757cdf9f3d7524aa11c923b52be298f7a97a4a0194a5d78c205426945"
R3_SEAL_RECEIPT_PATH = f"{PREFIX}/PACKET_SEAL_R3.json"
Record = dict[str, Any]  # Source and root-admission JSON schemas are heterogeneous.


def canonical(value: Record) -> bytes:
    """Encode independent byte identity with intentional actual newlines."""
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def sha(payload: bytes) -> str:
    """Return an archive or small frozen metadata digest."""
    return hashlib.sha256(payload).hexdigest()


def seal(
    repo: Path,
    output: Path,
    admission_path: Path | None,
    control_contract_path: Path,
) -> Record:
    """Freeze exact source, mount IDs, source pins and optional root authority."""
    if output.exists():
        raise FileExistsError(output)
    files = {
        name: subprocess.run(
            ["git", "-C", str(repo), "show", f"{BASE_COMMIT}:{name}"],
            capture_output=True,
            check=True,
        ).stdout
        for name in BASE_FILES
    }
    reader = subprocess.run(
        ["git", "-C", str(repo), "show", f"{R4_SOURCE_COMMIT}:{R4_READER_PATH}"],
        capture_output=True,
        check=True,
    ).stdout
    if sha(reader) != R4_READER_SHA256:
        raise ValueError("frozen R4 parsed-cache reader differs from its audited SHA-256")
    files["src/wnn/parsed_sequence_cache_reader.py"] = reader
    r4_packet = subprocess.run(
        ["git", "-C", str(repo), "show", f"{R4_PACKET_COMMIT}:{R4_PACKET_PATH}"],
        capture_output=True,
        check=True,
    ).stdout
    r4_kernel = subprocess.run(
        ["git", "-C", str(repo), "show", f"{R4_PACKET_COMMIT}:{R4_KERNEL_PATH}"],
        capture_output=True,
        check=True,
    ).stdout
    r4_bridge_test = subprocess.run(
        ["git", "-C", str(repo), "show", f"{R4_SOURCE_COMMIT}:{R4_BRIDGE_TEST_PATH}"],
        capture_output=True,
        check=True,
    ).stdout
    r4_metadata = subprocess.run(
        ["git", "-C", str(repo), "show", f"{R4_PACKET_COMMIT}:{R4_KERNEL_METADATA_PATH}"],
        capture_output=True,
        check=True,
    ).stdout
    r4_asset_manifest = subprocess.run(
        ["git", "-C", str(repo), "show", f"{R4_PACKET_COMMIT}:{R4_SOURCE_ASSET_MANIFEST_PATH}"],
        capture_output=True,
        check=True,
    ).stdout
    if (
        sha(r4_packet) != R4_PACKET_SHA256
        or sha(r4_kernel) != R4_KERNEL_SHA256
        or sha(r4_metadata) != R4_KERNEL_METADATA_SHA256
        or sha(r4_asset_manifest) != R4_SOURCE_ASSET_MANIFEST_SHA256
    ):
        raise ValueError("frozen R4 packet or producer kernel differs from its audited SHA-256")
    if sha(r4_bridge_test) != R4_BRIDGE_TEST_SHA256:
        raise ValueError("frozen R4 cache bridge test differs from its audited SHA-256")
    files.update(
        {
            "producer_evidence/r4/gpu_loss_pair.packet.json": r4_packet,
            "producer_evidence/r4/wnn-loss-pair-gpu-20261009-v4.py": r4_kernel,
            "producer_evidence/r4/kernel-metadata.json": r4_metadata,
            "producer_evidence/r4/source-assets-manifest.json": r4_asset_manifest,
            "producer_evidence/r4/test_parsed_cache_receipt_bridge.py": r4_bridge_test,
        }
    )
    r5_paths = (
        R5_PACKET_PATH,
        R5_KERNEL_PATH,
        R5_KERNEL_METADATA_PATH,
        R5_SOURCE_ASSET_MANIFEST_PATH,
        R5_EXPORT_TEST_PATH,
    )
    r5_files = {
        name: subprocess.run(
            ["git", "-C", str(repo), "show", f"{R5_SOURCE_COMMIT}:{name}"],
            capture_output=True,
            check=True,
        ).stdout
        for name in r5_paths
    }
    rehearsal_test = subprocess.run(
        ["git", "-C", str(repo), "show", f"{R5_REHEARSAL_COMMIT}:{R5_REHEARSAL_TEST_PATH}"],
        capture_output=True,
        check=True,
    ).stdout
    rehearsal_receipt = subprocess.run(
        ["git", "-C", str(repo), "show", f"{R5_REHEARSAL_COMMIT}:{R5_REHEARSAL_RECEIPT_PATH}"],
        capture_output=True,
        check=True,
    ).stdout
    r5_packet = r5_files[R5_PACKET_PATH]
    r5_kernel = r5_files[R5_KERNEL_PATH]
    r5_metadata = r5_files[R5_KERNEL_METADATA_PATH]
    r5_asset_manifest = r5_files[R5_SOURCE_ASSET_MANIFEST_PATH]
    r5_export_test = r5_files[R5_EXPORT_TEST_PATH]
    if (
        sha(r5_packet) != R5_PACKET_SHA256
        or sha(r5_kernel) != R5_KERNEL_SHA256
        or sha(r5_metadata) != R5_KERNEL_METADATA_SHA256
        or sha(r5_asset_manifest) != R5_SOURCE_ASSET_MANIFEST_SHA256
        or sha(r5_export_test) != R5_EXPORT_TEST_SHA256
        or sha(rehearsal_test) != R5_REHEARSAL_TEST_SHA256
        or sha(rehearsal_receipt) != R5_REHEARSAL_RECEIPT_SHA256
    ):
        raise ValueError("frozen R5 packet, output source, or rehearsal evidence changed")
    control_contract = control_contract_path.resolve(strict=True).read_bytes()
    if sha(control_contract) != R5_OUTPUT_CONTRACT_SHA256:
        raise ValueError("actual completed R5 training contract differs from the root pin")
    for name in OWN_FILES:
        files[name] = subprocess.run(
            ["git", "-C", str(repo), "show", f"{R2_SOURCE_COMMIT}:{name}"],
            capture_output=True,
            check=True,
        ).stdout
    files[CORE_FILE] = subprocess.run(
        ["git", "-C", str(repo), "show", f"{BASE_COMMIT}:{CORE_FILE}"],
        capture_output=True,
        check=True,
    ).stdout
    r3_predecessor_paths = (
        R3_PACKET_PATH,
        R3_SOURCE_MANIFEST_PATH,
        R3_SOURCE_ARCHIVE_PATH,
        R3_SEAL_RECEIPT_PATH,
    )
    r3_predecessor = {
        name: subprocess.run(
            ["git", "-C", str(repo), "show", f"{R3_SOURCE_COMMIT}:{name}"],
            capture_output=True,
            check=True,
        ).stdout
        for name in r3_predecessor_paths
    }
    if (
        sha(r3_predecessor[R3_PACKET_PATH]) != R3_PACKET_SHA256
        or sha(r3_predecessor[R3_SOURCE_MANIFEST_PATH]) != R3_SOURCE_MANIFEST_SHA256
        or sha(r3_predecessor[R3_SOURCE_ARCHIVE_PATH]) != R3_SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError("sealed R3 TPU predecessor differs from its reviewed exact hashes")
    files.update(
        {
            "predecessor_evidence/r3/packet.json": r3_predecessor[R3_PACKET_PATH],
            "predecessor_evidence/r3/source_manifest.json": r3_predecessor[
                R3_SOURCE_MANIFEST_PATH
            ],
            "predecessor_evidence/r3/source.tar.gz": r3_predecessor[R3_SOURCE_ARCHIVE_PATH],
            "predecessor_evidence/r3/PACKET_SEAL_R3.json": r3_predecessor[
                R3_SEAL_RECEIPT_PATH
            ],
        }
    )
    for name in R4_EXTRA_FILES:
        payload = (repo / name).read_bytes()
        committed = subprocess.run(
            ["git", "-C", str(repo), "show", f"{NATIVE_R4_SOURCE_COMMIT}:{name}"],
            capture_output=True,
            check=True,
        ).stdout
        if payload != committed:
            raise ValueError(f"R4 source is not identical to its frozen commit: {name}")
        files[name] = payload
    files.update(
        {
            "producer_evidence/r5/gpu_loss_pair_export_recovery.packet.json": r5_packet,
            "producer_evidence/r5/wnn-loss-pair-export-recovery-20261009-r1.py": r5_kernel,
            "producer_evidence/r5/kernel-metadata.json": r5_metadata,
            "producer_evidence/r5/source-assets-manifest.json": r5_asset_manifest,
            "producer_evidence/r5/test_gpu_loss_pair_r5_export.py": r5_export_test,
            "producer_evidence/r5/test_gpu_loss_pair_r5_shared_rehearsal.py": rehearsal_test,
            "producer_evidence/r5/shared_artifact_rehearsal_receipt_R1.json": rehearsal_receipt,
            "producer_evidence/r5/training_contract.json": control_contract,
        }
    )
    source_manifest = {
        "schema": "wnn.native-device-r5-producer-binding.v2",
        "revision": "R4",
        "base_commit": BASE_COMMIT,
        "reused_native_implementation_commit": R2_SOURCE_COMMIT,
        "native_tpu_r4_source_commit": NATIVE_R4_SOURCE_COMMIT,
        "source_sealer": {
            "path": f"{PREFIX}/seal_tpu_packet_r4.py",
            "sha256": sha(Path(__file__).read_bytes()),
        },
        "native_tpu_r3_predecessor": {
            "source_commit": R3_SOURCE_COMMIT,
            "packet_path": "predecessor_evidence/r3/packet.json",
            "packet_sha256": R3_PACKET_SHA256,
            "source_manifest_sha256": R3_SOURCE_MANIFEST_SHA256,
            "source_archive_sha256": R3_SOURCE_ARCHIVE_SHA256,
            "changes": [
                "require root-specified device, host, cgroup, and disk reserve bytes",
                "record actual full-shape per-device memory capacity/peak evidence",
                "defer fits when profile or live headroom evidence is missing or insufficient",
                "monitor live host/cgroup/disk reserves while the matched fit runs",
            ],
        },
        "gpu_fit_producer_r4": {
            "source_commit": R4_SOURCE_COMMIT,
            "packet_commit": R4_PACKET_COMMIT,
            "packet_path": "producer_evidence/r4/gpu_loss_pair.packet.json",
            "packet_sha256": R4_PACKET_SHA256,
            "kernel_code_path": "producer_evidence/r4/wnn-loss-pair-gpu-20261009-v4.py",
            "kernel_code_sha256": R4_KERNEL_SHA256,
            "kernel_metadata_sha256": R4_KERNEL_METADATA_SHA256,
            "source_asset_manifest_sha256": R4_SOURCE_ASSET_MANIFEST_SHA256,
            "kernel": "vladkochetov07/wnn-loss-pair-gpu-20261009-v4",
            "source_asset": "vladkochetov07/wnn-loss-pair-code-20261009-r4",
            "status": "completed_fits_retained_and_root_reviewed_in_R4",
        },
        "gpu_export_recovery_producer_r5": {
            "source_commit": R5_SOURCE_COMMIT,
            "packet_path": "producer_evidence/r5/gpu_loss_pair_export_recovery.packet.json",
            "packet_sha256": R5_PACKET_SHA256,
            "kernel_code_path": "producer_evidence/r5/wnn-loss-pair-export-recovery-20261009-r1.py",
            "kernel_code_sha256": R5_KERNEL_SHA256,
            "kernel_metadata_sha256": R5_KERNEL_METADATA_SHA256,
            "source_asset_manifest_sha256": R5_SOURCE_ASSET_MANIFEST_SHA256,
            "shared_artifact_rehearsal_receipt_sha256": R5_REHEARSAL_RECEIPT_SHA256,
            "kernel": R5_KERNEL,
            "training_contract_path": "producer_evidence/r5/training_contract.json",
            "training_contract_sha256": R5_OUTPUT_CONTRACT_SHA256,
            "root_terminal_review_sha256": R5_ROOT_REVIEW_SHA256,
            "recovery_asset_manifest_sha256": R5_RECOVERY_ASSET_SHA256,
            "status": "completed_export_only_and_root_reviewed; native TPU admission remains separate",
        },
        "parsed_cache_reader_override": {
            "commit": R4_SOURCE_COMMIT,
            "source_path": R4_READER_PATH,
            "packaged_path": "src/wnn/parsed_sequence_cache_reader.py",
            "sha256": R4_READER_SHA256,
            "bridge_test_path": R4_BRIDGE_TEST_PATH,
            "packaged_bridge_test_path": "producer_evidence/r4/test_parsed_cache_receipt_bridge.py",
            "bridge_test_sha256": R4_BRIDGE_TEST_SHA256,
            "focused_tests": "5 passed in 3.18s",
        },
        "files": [
            {"path": name, "size_bytes": len(payload), "sha256": sha(payload)}
            for name, payload in sorted(files.items())
        ],
    }
    manifest_bytes = canonical(source_manifest)
    files["source_manifest.json"] = manifest_bytes
    pending = {
        "schema": "wnn.loss-pair-tpu.root-admission.v1",
        "push_allowed": False,
        "conditional_fit_allowed": False,
        "source_manifest_sha256": sha(manifest_bytes),
        "gpu_training_contract_sha256": R5_OUTPUT_CONTRACT_SHA256,
        "gpu_root_terminal_review_sha256": R5_ROOT_REVIEW_SHA256,
        "runtime_seconds": 3300,
        "evaluation_export_reserve_seconds": 600,
        "cache_manifest_sha256": "7400b75504f592011577dd3b126e9813749c21ce9126e5c4fa04c8e6b61e8f3b",
        "cache_key": "949587f77794935c08589bd5d5e81641062de547526b92fc1b01bc4109cd82a6",
        "cpu_kernel": "vladkochetov07/wnn-parsed-cache-source-stage-20261009-v1",
        "gpu_kernel": R5_KERNEL,
        "native_bf16_fit_allowed": False,
        "tpu_compute_precision": "fp32",
        "fit_headroom_policy": None,
    }
    admission = json.loads(admission_path.read_text()) if admission_path else pending
    if admission["source_manifest_sha256"] != sha(manifest_bytes):
        raise ValueError("root admission binds a different source closure")
    if admission.get("gpu_training_contract_sha256") != R5_OUTPUT_CONTRACT_SHA256:
        raise ValueError("root admission does not bind the exact completed R5 control contract")
    if admission.get("conditional_fit_allowed") is True:
        from fit_headroom_gate_r4 import validate_policy

        _policy, policy_errors = validate_policy(admission.get("fit_headroom_policy"))
        if policy_errors:
            raise ValueError("conditional fit admission needs explicit device/host/cgroup/disk reserves")
    files["root_admission.json"] = canonical(admission)
    buf = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buf, mode="wb", filename="", mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w") as archive,
    ):
        for name, payload in sorted(files.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(payload)
            entry.mode = 0o644
            entry.mtime = 0
            archive.addfile(entry, io.BytesIO(payload))
    archive_bytes = buf.getvalue()
    code = """from __future__ import annotations
import base64, gzip, hashlib, io, json, os, signal, subprocess, sys, tarfile, time, traceback
from pathlib import Path
WORK=Path(os.environ.get('KAGGLE_WORKING','/kaggle/working'))
WORK.mkdir(parents=True,exist_ok=True)
(WORK/'boot_marker.json').write_text(json.dumps({'event':'tpu_boot','unix':time.time()}))
PAYLOAD=base64.b64decode('__PAYLOAD__')
if hashlib.sha256(PAYLOAD).hexdigest()!='__ARCHIVE_SHA__': raise ValueError('source archive bytes changed')
SOURCE=WORK/'tpu_source'
SOURCE.mkdir(exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(PAYLOAD),mode='r:gz') as archive:
    for member in archive.getmembers():
        relative=Path(member.name)
        if relative.is_absolute() or '..' in relative.parts or not member.isfile(): raise ValueError('unsafe frozen source entry')
    archive.extractall(SOURCE,filter='data')
sys.path.insert(0,str(SOURCE/'__PREFIX__'))
from fit_headroom_gate_r4 import evaluate_live_resources, sample_live_resources
ADMISSION=json.loads((SOURCE/'root_admission.json').read_text())
command=[sys.executable,str(SOURCE/'__PREFIX__/tpu_packet_runtime_r4.py'),'--admission',str(SOURCE/'root_admission.json'),'--source-manifest',str(SOURCE/'source_manifest.json'),'--input-root',os.environ.get('KAGGLE_INPUT','/kaggle/input'),'--output-root',str(WORK/'out')]
if os.environ.get('WNN_TPU_OFFLINE')=='1': command.append('--offline')
try:
    started=time.monotonic()
    with (WORK/'runtime.log').open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            if time.monotonic()-started>3300:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL)
                raise TimeoutError('finite TPU worker deadline exceeded')
            phase=WORK/'out/phase.json'
            phase_record=json.loads(phase.read_text()) if phase.exists() else {'phase':'startup'}
            memory=Path('/sys/fs/cgroup/memory.current')
            (WORK/'heartbeat.json').write_text(json.dumps({'pid':child.pid,'elapsed_seconds':time.monotonic()-started,'phase':phase_record,'memory_current_bytes':int(memory.read_text()) if memory.exists() else None,'unix':time.time()}))
            if phase_record.get('phase')=='matched_fit_pair':
                try:
                    live=sample_live_resources(WORK/'out')
                    guard=evaluate_live_resources(ADMISSION.get('fit_headroom_policy'),live)
                except Exception as error:
                    live={'sampling_error':str(error)}
                    guard={'passed':False,'errors':['live_resource_sampling_failed'],'live_checks':{}}
                if not guard['passed']:
                    stop={'schema':'wnn.loss-pair-tpu.live-fit-headroom-stop.v1','status':'stopped_fit_for_headroom','phase':phase_record,'live_resources':live,'guard':guard,'unix':time.time()}
                    (WORK/'out/resource_guard_stop.json').write_text(json.dumps(stop,sort_keys=True,indent=2)+'\\n')
                    os.killpg(child.pid,signal.SIGTERM)
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL)
                    break
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: pass
        if child.returncode: raise subprocess.CalledProcessError(child.returncode,command)
    (WORK/'final_status.json').write_text(json.dumps({'status':'complete','unix':time.time()}))
except BaseException as error:
    (WORK/'final_status.json').write_text(json.dumps({'status':'failed','error':str(error),'unix':time.time()}))
    traceback.print_exc()
    raise
"""
    code = (
        code.replace("__PAYLOAD__", base64.b64encode(archive_bytes).decode())
        .replace("__ARCHIVE_SHA__", sha(archive_bytes))
        .replace("__PREFIX__", PREFIX)
    )
    if len(code.encode()) >= 1_000_000:
        raise ValueError("kernel bootstrap exceeds conservative1MB envelope")
    slug = re.sub(r"[^a-z0-9]+", "-", KERNEL_TITLE.lower()).strip("-")
    if KERNEL_ID.split("/")[1] != slug:
        raise ValueError("canonical kernel ID differs from title-derived slug")
    output.mkdir(parents=True)
    kernel = output / "kernel"
    kernel.mkdir()
    code_name = slug + ".py"
    (kernel / code_name).write_text(code)
    metadata = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": code_name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": True,
        "machine_shape": "TpuV5E8",
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "kernel_sources": [pending["cpu_kernel"], pending["gpu_kernel"]],
    }
    (kernel / "kernel-metadata.json").write_bytes(canonical(metadata))
    (output / "source_manifest.json").write_bytes(manifest_bytes)
    (output / "source.tar.gz").write_bytes(archive_bytes)
    (output / "root_admission.json").write_bytes(canonical(admission))
    packet = {
        "schema": "wnn.loss-pair-tpu.packet.v1",
        "status": "review_required_not_admitted"
        if not admission["push_allowed"]
        else "root_admitted",
        "kernel_id": KERNEL_ID,
        "kernel_title": KERNEL_TITLE,
        "source_manifest_sha256": sha(manifest_bytes),
        "source_archive_sha256": sha(archive_bytes),
        "bootstrap_sha256": sha(code.encode()),
        "bootstrap_bytes": len(code.encode()),
        "metadata_sha256": sha(canonical(metadata)),
        "root_admission_sha256": sha(canonical(admission)),
        "kernel_sources": metadata["kernel_sources"],
        "gpu_training_contract_sha256": R5_OUTPUT_CONTRACT_SHA256,
        "gpu_export_kernel": R5_KERNEL,
        "runtime_seconds": 3300,
        "provider_seconds_envelope": 3600,
        "data_download_to_laptop": False,
        "source_files": len(source_manifest["files"]),
    }
    (output / "packet.json").write_bytes(canonical(packet))
    return packet


def main() -> None:
    """Prepare a packet only; source sealing never dispatches a provider job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[6])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--control-contract", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(seal(args.repo, args.output, args.admission, args.control_contract)))


if __name__ == "__main__":
    main()
