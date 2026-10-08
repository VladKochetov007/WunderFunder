from __future__ import annotations

import hashlib
import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

SOURCE_DATASET = 'vladkochetov07/e529-gru512-768-muon-fit-v5r1-source-20261008'
EXPECTED_MANIFEST_SHA256 = 'ab7d84b3c0f38697bada309f324182abd0b393ca1e5abcfde7bbe46e89eff0cc'
REQUIRED_FILES = ('GRU512_768_MUON_FIT_RECIPE_V5R1.json', 'README.md', 'ROOT_FIT_AUTHORIZATION.json', 'ROOT_FIT_SOURCE_REVIEW.json', 'capacity_arm_worker.py', 'capacity_fit_worker.py', 'capacity_worker.py', 'fold0.json', 'kernel-metadata.json', 'kernel.py', 'portable_inputs.json', 'prior_input_identity_v2.json', 'producer_evidence/v4/cohort_terminal_receipt.json', 'producer_evidence/v4/input_identity_receipt.json', 'producer_evidence/v4/manifest.json', 'producer_evidence/v4/muon512-profile-argv.json', 'producer_evidence/v4/muon512-profile-digest-metrics.json', 'producer_evidence/v4/muon512-profile-gpu-samples.json', 'producer_evidence/v4/muon512-profile.log', 'producer_evidence/v4/muon768-profile-argv.json', 'producer_evidence/v4/muon768-profile-digest-metrics.json', 'producer_evidence/v4/muon768-profile-gpu-samples.json', 'producer_evidence/v4/muon768-profile.log', 'producer_evidence/v4/provider_kernel_log.log', 'producer_evidence/v4/root_posthoc_profile_review.json', 'producer_evidence/v4/terminal_collection_review_request.json', 'source/source_archive.tar', 'source/source_archive_manifest.json', 'train_entry.py', 'verified_digest_handoff.py')
PACKET_ID = 'E529_H232_GRU512_768_MUON_SEED9_FIT_V5R1_20261008'
WORKER_CAP_SECONDS = 26000

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()

def source_root(input_root: Path) -> Path:
    owner, slug = SOURCE_DATASET.split("/", 1)
    candidates = (input_root / "datasets" / owner / slug, input_root / slug)
    matches = []
    for candidate in candidates:
        manifest = candidate / "assets-manifest.json"
        if manifest.is_file() and sha256_file(manifest) == EXPECTED_MANIFEST_SHA256:
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError("expected exactly one mount of the hash-bound private packet")
    return matches[0]

def materialize(input_root: Path, packet_root: Path) -> None:
    dataset_root = source_root(input_root)
    manifest_path = dataset_root / "assets-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != SOURCE_DATASET or manifest.get("private") is not True:
        raise ValueError("private source dataset identity or visibility changed")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("private packet file manifest is malformed")
    by_path = {}
    for entry in entries:
        logical = str(entry.get("path", ""))
        relative = PurePosixPath(logical)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts or "\\" in logical:
            raise ValueError("private packet contains an unsafe path")
        if logical in by_path:
            raise ValueError("private packet repeats a file path")
        by_path[logical] = entry
    if tuple(sorted(by_path)) != REQUIRED_FILES:
        raise ValueError("private packet closure differs from the reviewed file set")
    temporary = packet_root.with_name(packet_root.name + ".partial")
    if packet_root.exists() or temporary.exists():
        raise FileExistsError("packet destination already exists")
    temporary.mkdir(parents=True)
    complete = False
    try:
        for logical in REQUIRED_FILES:
            entry = by_path[logical]
            digest = str(entry.get("sha256", ""))
            filename = str(entry.get("filename", ""))
            size = entry.get("size")
            if len(digest) != 64 or filename != digest + ".wnn":
                raise ValueError("private packet member is not content-addressed")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError("private packet member size is invalid")
            source = dataset_root / filename
            if source.is_symlink() or not source.is_file() or source.stat().st_size != size:
                raise ValueError("private packet member type or size changed")
            relative = PurePosixPath(logical)
            target = temporary.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            copied = 0
            digest_state = hashlib.sha256()
            with source.open("rb") as reader, target.open("xb") as writer:
                while block := reader.read(1 << 20):
                    writer.write(block)
                    digest_state.update(block)
                    copied += len(block)
            if copied != size or digest_state.hexdigest() != digest:
                raise ValueError("private packet member digest changed during materialization")
        temporary.replace(packet_root)
        complete = True
    finally:
        if not complete:
            shutil.rmtree(temporary, ignore_errors=True)

def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

def stop_child(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)

def run_worker(packet_root: Path, started: float) -> dict:
    worker_log = Path("/kaggle/working/gru_capacity_v5r1_worker.log")
    terminal_path = Path("/kaggle/working/gru_capacity_v5r1_terminal.json")
    deadline = started + WORKER_CAP_SECONDS
    command = [sys.executable, str(packet_root / "capacity_fit_worker.py"), "--packet-root", str(packet_root)]
    prefixes = (
        "GRU_CAPACITY_PHASE ",
        "GRU_CAPACITY_DEVICE ",
        "GRU_CAPACITY_TERMINAL ",
        '{"event": "campaign_resource_heartbeat"',
    )
    status = "running"
    with worker_log.open("wb") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=packet_root,
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            start_new_session=True,
        )
        if process.stdout is None:
            raise RuntimeError("fit worker stdout pipe was not created")
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        buffer = bytearray()
        exit_code = None
        try:
            while True:
                if time.monotonic() >= deadline:
                    stop_child(process)
                    status = "timeout"
                    break
                for key, _ in selector.select(timeout=1.0):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    log_stream.write(chunk)
                    log_stream.flush()
                    buffer.extend(chunk)
                    while b"\n" in buffer:
                        raw, _, tail = buffer.partition(b"\n")
                        buffer = bytearray(tail)
                        line = raw.decode("utf-8", errors="replace")
                        if line.startswith(prefixes):
                            print(line, flush=True)
                exit_code = process.poll()
                if exit_code is not None and not selector.get_map():
                    break
            if buffer:
                line = buffer.decode("utf-8", errors="replace")
                if line.startswith(prefixes):
                    print(line, flush=True)
            if status != "timeout":
                status = "complete" if exit_code == 0 else "worker_error"
        finally:
            selector.close()
            if process.poll() is None:
                stop_child(process)
            process.stdout.close()
    worker_receipt = {}
    if terminal_path.is_file():
        worker_receipt = json.loads(terminal_path.read_text(encoding="utf-8"))
    result = {
        "schema": "wnn-gru-capacity-v5r1-driver-terminal-v1",
        "packet_id": PACKET_ID,
        "status": status,
        "worker_exit_code": exit_code,
        "worker_receipt_status": worker_receipt.get("status"),
        "worker_seconds": round(time.monotonic() - started, 3),
        "worker_log": str(worker_log),
        "fit_started": worker_receipt.get("fit_started", False),
    }
    write_json(Path("/kaggle/working/gru_capacity_v5r1_driver_terminal.json"), result)
    print("GRU_CAPACITY_DRIVER_TERMINAL " + json.dumps(result, sort_keys=True), flush=True)
    return result

def main() -> None:
    started = time.monotonic()
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["WNN_KERNEL_DEADLINE_MONOTONIC"] = repr(started + WORKER_CAP_SECONDS)
    print("GRU_CAPACITY_PHASE " + json.dumps({"phase":"bootstrap","status":"running","arm":"cohort","elapsed_s":0.0}), flush=True)
    packet_root = Path("/kaggle/working/gru_capacity_v5r1_packet")
    try:
        materialize(Path("/kaggle/input"), packet_root)
        print("GRU_CAPACITY_PHASE " + json.dumps({"phase":"bootstrap","status":"pass","arm":"cohort","elapsed_s":round(time.monotonic()-started,1)}), flush=True)
        receipt = run_worker(packet_root, started)
    except BaseException as error:
        failure = {"schema":"wnn-gru-capacity-v5r1-driver-terminal-v1","packet_id":PACKET_ID,"status":"bootstrap_error","error_type":type(error).__name__,"worker_seconds":round(time.monotonic()-started,3),"fit_started":False}
        write_json(Path("/kaggle/working/gru_capacity_v5r1_driver_terminal.json"), failure)
        print("GRU_CAPACITY_DRIVER_TERMINAL " + json.dumps(failure, sort_keys=True), flush=True)
        raise
    if receipt.get("status") != "complete" or receipt.get("worker_receipt_status") not in {"source_review_pending","fit_authorization_pending","fit_complete"}:
        raise RuntimeError("V5R1 fit worker did not reach a declared terminal status")

if __name__ == "__main__":
    main()
