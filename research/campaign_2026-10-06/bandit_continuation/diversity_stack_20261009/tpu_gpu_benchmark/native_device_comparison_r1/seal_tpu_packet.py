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
R4_COMMIT = "1bc0fbacf53cdc08a8c4664c6f66f38f8980092c"
R4_READER_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "cloud_delivery/gpu_loss_pair_r4/code/src/wnn/parsed_sequence_cache_reader.py"
)
R4_READER_SHA256 = "35431e48583b1523188b3d6084fa12be23e7379560a323411655bf561507a693"
R4_PACKET_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "cloud_delivery/gpu_loss_pair_r4/gpu_loss_pair.packet.json"
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
PREFIX = "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/tpu_gpu_benchmark/native_device_comparison_r1"
KERNEL_TITLE = "WNN Native Device TPU 20261009 V2"
KERNEL_ID = "vladkochetov07/wnn-native-device-tpu-20261009-v2"
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
Record = dict[str, Any]  # Source and root-admission JSON schemas are heterogeneous.


def canonical(value: Record) -> bytes:
    """Encode independent byte identity with intentional actual newlines."""
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def sha(payload: bytes) -> str:
    """Return an archive or small frozen metadata digest."""
    return hashlib.sha256(payload).hexdigest()


def seal(repo: Path, output: Path, admission_path: Path | None) -> Record:
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
        ["git", "-C", str(repo), "show", f"{R4_COMMIT}:{R4_READER_PATH}"],
        capture_output=True,
        check=True,
    ).stdout
    if sha(reader) != R4_READER_SHA256:
        raise ValueError("frozen R4 parsed-cache reader differs from its audited SHA-256")
    files["src/wnn/parsed_sequence_cache_reader.py"] = reader
    r4_packet = subprocess.run(
        ["git", "-C", str(repo), "show", f"{R4_COMMIT}:{R4_PACKET_PATH}"],
        capture_output=True,
        check=True,
    ).stdout
    r4_kernel = subprocess.run(
        ["git", "-C", str(repo), "show", f"{R4_COMMIT}:{R4_KERNEL_PATH}"],
        capture_output=True,
        check=True,
    ).stdout
    r4_bridge_test = subprocess.run(
        ["git", "-C", str(repo), "show", f"{R4_COMMIT}:{R4_BRIDGE_TEST_PATH}"],
        capture_output=True,
        check=True,
    ).stdout
    if sha(r4_packet) != R4_PACKET_SHA256 or sha(r4_kernel) != R4_KERNEL_SHA256:
        raise ValueError("frozen R4 packet or producer kernel differs from its audited SHA-256")
    if sha(r4_bridge_test) != R4_BRIDGE_TEST_SHA256:
        raise ValueError("frozen R4 cache bridge test differs from its audited SHA-256")
    files.update(
        {
            "producer_evidence/r4/gpu_loss_pair.packet.json": r4_packet,
            "producer_evidence/r4/wnn-loss-pair-gpu-20261009-v4.py": r4_kernel,
            "producer_evidence/r4/test_parsed_cache_receipt_bridge.py": r4_bridge_test,
        }
    )
    for name in (*OWN_FILES, CORE_FILE):
        files[name] = (repo / name).read_bytes()
    source_manifest = {
        "base_commit": BASE_COMMIT,
        "gpu_producer": {
            "source_commit": R4_COMMIT,
            "packet_path": "producer_evidence/r4/gpu_loss_pair.packet.json",
            "packet_sha256": R4_PACKET_SHA256,
            "kernel_code_path": "producer_evidence/r4/wnn-loss-pair-gpu-20261009-v4.py",
            "kernel_code_sha256": R4_KERNEL_SHA256,
            "kernel": "vladkochetov07/wnn-loss-pair-gpu-20261009-v4",
            "source_asset": "vladkochetov07/wnn-loss-pair-code-20261009-r4",
            "training_contract_sha256": None,
            "status": "sealed_candidate_waiting_root_review_and_admission",
        },
        "parsed_cache_reader_override": {
            "commit": R4_COMMIT,
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
        "gpu_training_contract_sha256": None,
        "runtime_seconds": 3300,
        "evaluation_export_reserve_seconds": 600,
        "cache_manifest_sha256": "7400b75504f592011577dd3b126e9813749c21ce9126e5c4fa04c8e6b61e8f3b",
        "cache_key": "949587f77794935c08589bd5d5e81641062de547526b92fc1b01bc4109cd82a6",
        "cpu_kernel": "vladkochetov07/wnn-parsed-cache-source-stage-20261009-v1",
        "gpu_kernel": "vladkochetov07/wnn-loss-pair-gpu-20261009-v4",
        "native_bf16_fit_allowed": False,
        "tpu_compute_precision": "fp32",
    }
    admission = json.loads(admission_path.read_text()) if admission_path else pending
    if admission["source_manifest_sha256"] != sha(manifest_bytes):
        raise ValueError("root admission binds a different source closure")
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
command=[sys.executable,str(SOURCE/'__PREFIX__/tpu_packet_runtime.py'),'--admission',str(SOURCE/'root_admission.json'),'--source-manifest',str(SOURCE/'source_manifest.json'),'--input-root',os.environ.get('KAGGLE_INPUT','/kaggle/input'),'--output-root',str(WORK/'out')]
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
            memory=Path('/sys/fs/cgroup/memory.current')
            (WORK/'heartbeat.json').write_text(json.dumps({'pid':child.pid,'elapsed_seconds':time.monotonic()-started,'phase':json.loads(phase.read_text()) if phase.exists() else {'phase':'startup'},'memory_current_bytes':int(memory.read_text()) if memory.exists() else None,'unix':time.time()}))
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
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admission", type=Path)
    args = parser.parse_args()
    print(json.dumps(seal(args.repo, args.output, args.admission)))


if __name__ == "__main__":
    main()
