"""Verify, profile and conditionally train the frozen GRU capacity cohort."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import selectors
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from verified_digest_handoff import FileIdentity, VerifiedDigestHandoff


PACKET_ID = "E529_H232_GRU512_768_MUON_SEED9_PROFILE_V3_20261008"
RECIPE_NAME = "GRU512_768_MUON_PROFILE_RECIPE_V3.json"
AUTH_NAME = "ROOT_PROFILE_AUTHORIZATION.json"
WORKER_CAP_SECONDS = 5200
FINALIZATION_RESERVE_SECONDS = 300
PROFILE_WARMUP_BATCHES = 3
PROFILE_TIMED_BATCHES = 20
CHUNK_UPDATES_PER_BATCH = 10
FIT_PROJECTION_SAFETY = 1.20
FIT_STARTUP_RESERVE_SECONDS = 60
PREDICTION_RESERVE_SECONDS = 420
ARM_SPECS = (
    {"id": "muon512", "hidden": 512, "optimizer": "muon", "primary": True},
    {"id": "muon768", "hidden": 768, "optimizer": "muon", "primary": True},
    {"id": "adamw512", "hidden": 512, "optimizer": "adamw", "primary": False},
)
RAW_DATA_ROOT = Path("/kaggle/input/datasets/vladkochetov07/wunder-funder")
ASSET_DATA_ROOT = Path("/kaggle/input/datasets/vladkochetov07/wnn-c20261004-fold0-assets")
RAW_DATA_FALLBACK = Path("/kaggle/input/wunder-funder")
ASSET_DATA_FALLBACK = Path("/kaggle/input/wnn-c20261004-fold0-assets")
ASSET_MANIFEST_SHA256 = "19fe0eb6445553c2adff00e821a4fa16ca5969473ca8be420c32cb8e4b4b72c7"
GPU_DEVICE_INDEX = 0


def sha256_file(path: Path) -> str:
    """Hash a regular file in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Atomically write a JSON receipt without emitting its contents to the live log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_relative(raw: str) -> PurePosixPath:
    """Reject absolute, traversing or noncanonical packet paths."""
    relative = PurePosixPath(raw)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or "." in relative.parts
    ):
        raise ValueError("packet contains an unsafe relative path")
    return relative


def _file_binding(packet_root: Path, path: str, expected: dict[str, object]) -> None:
    relative = safe_relative(path)
    target = packet_root.joinpath(*relative.parts)
    if not target.is_file():
        raise FileNotFoundError(f"bound packet file is missing: {path}")
    if target.stat().st_size != int(expected["bytes"]):
        raise ValueError(f"bound packet file size changed: {path}")
    if sha256_file(target) != expected["sha256"]:
        raise ValueError(f"bound packet file digest changed: {path}")


def _emit(
    started: float,
    phase: str,
    *,
    arm: str = "cohort",
    status: str = "running",
    fields: dict[str, int | float | str | bool | None] | None = None,
) -> None:
    """Print only allowlisted phase, arm, status and bounded scalar telemetry."""
    elapsed = max(0.0, time.monotonic() - started)
    allowed_phases = {
        "bootstrap",
        "source_verify",
        "input_mount",
        "input_verify",
        "profile",
        "profile_gate",
        "fit",
        "checkpoint_validate",
        "prediction_export",
        "prediction_validate",
        "terminal",
    }
    allowed_arms = {"cohort", "muon512", "muon768", "adamw512"}
    allowed_status = {"running", "pass", "fail", "skipped", "timeout", "profile_only"}
    if phase not in allowed_phases or arm not in allowed_arms or status not in allowed_status:
        raise ValueError("phase telemetry field is outside the fixed allowlist")
    safe: dict[str, object] = {
        "phase": phase,
        "arm": arm,
        "status": status,
        "elapsed_s": round(elapsed, 1),
    }
    if fields:
        allowed_fields = {
            "remaining_s",
            "batch",
            "step_wall_ms",
            "optimizer_attempts",
            "completed_updates",
            "amp_skips",
            "gpu_peak_used_mb",
            "gpu_min_free_mb",
            "fit_projection_s",
            "export_reserve_s",
            "worker_cap_s",
            "exit_code",
        }
        if set(fields) - allowed_fields:
            raise ValueError("phase telemetry includes an unapproved field")
        for key, value in fields.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("phase telemetry values must be finite")
            safe[key] = value
    print("GRU_CAPACITY_PHASE " + json.dumps(safe, sort_keys=True), flush=True)


def _mount_root(primary: Path, fallback: Path) -> Path:
    """Resolve one exact declared Kaggle mount; never search ambiguous basenames."""
    for root in (primary, fallback):
        if root.is_dir():
            return root.resolve(strict=True)
    raise FileNotFoundError("one declared Kaggle input mount is unavailable")


def _sha_matches(path: Path, expected_sha256: str, expected_bytes: int) -> FileIdentity:
    """Hash one mounted source once and return the unchanged identity it verifies."""
    before = FileIdentity.from_stat(path.stat())
    if before.size != expected_bytes:
        raise ValueError(f"mounted input size changed: {path.name}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"mounted input digest changed: {path.name}")
    after = FileIdentity.from_stat(path.stat())
    if after != before:
        raise ValueError(f"mounted input changed while hashing: {path.name}")
    return after


def _prior_stat_match(
    path: Path, prior: dict[str, object], expected_sha256: str
) -> tuple[bool, FileIdentity | None]:
    """Match an earlier full digest only when the exact canonical path and stat tuple recur."""
    binding = prior.get("read_only_file_stat_binding")
    if not isinstance(binding, dict):
        return False, None
    resolved = path.resolve(strict=True)
    if str(resolved) != binding.get("canonical_path"):
        return False, None
    stat_result = resolved.stat()
    current = FileIdentity.from_stat(stat_result)
    expected = FileIdentity(
        device=int(binding["device"]),
        inode=int(binding["inode"]),
        size=int(binding["size"]),
        modified_ns=int(binding["mtime_ns"]),
        changed_ns=int(binding["ctime_ns"]),
        mode=int(binding["mode"]),
    )
    return current == expected and prior.get("sha256") == expected_sha256, current


def _verify_raw_source(
    handoff: VerifiedDigestHandoff,
    source: Path,
    logical_name: str,
    expected: dict[str, object],
    prior_observation: dict[str, object] | None,
) -> dict[str, object]:
    """Verify raw bytes by exact prior identity or a full streaming SHA-256 fallback."""
    expected_sha = str(expected["sha256"])
    expected_bytes = int(expected["bytes"])
    prior = prior_observation or {}
    matches, current = _prior_stat_match(source, prior, expected_sha)
    if matches and current is not None:
        candidate = handoff.candidate_from_prior_identity(
            source, prior_identity=current, prior_sha256=expected_sha
        )
        if candidate is None:
            matches = False
        else:
            handoff.authorize(candidate, expected_sha256=expected_sha, registry_sha256=expected_sha)
    if not matches:
        observed = handoff.hash_file(source)
        if observed.bytes_hashed != expected_bytes:
            raise ValueError(f"raw input byte count changed: {logical_name}")
        handoff.authorize(observed, expected_sha256=expected_sha, registry_sha256=expected_sha)
        current = observed.identity
    if current is None:
        current = FileIdentity.from_stat(source.resolve(strict=True).stat())
    return {
        "logical_name": logical_name,
        "path": str(source.resolve(strict=True)),
        "sha256": expected_sha,
        "bytes": expected_bytes,
        "identity": asdict(current),
        "digest_source": "prior_exact_stat_identity" if matches else "current_full_content_hash",
        "bytes_hashed_this_attempt": 0 if matches else expected_bytes,
    }


def _asset_entries(asset_root: Path) -> dict[str, dict[str, object]]:
    """Read only the mounted asset manifest and index unique logical names."""
    manifest_path = asset_root / "assets-manifest.json"
    if sha256_file(manifest_path) != ASSET_MANIFEST_SHA256:
        raise ValueError("reusable fold0 asset manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "vladkochetov07/wnn-c20261004-fold0-assets":
        raise ValueError("mounted asset manifest names a different dataset")
    entries: dict[str, dict[str, object]] = {}
    for item in manifest.get("files", []):
        logical = str(item["path"])
        if logical in entries:
            raise ValueError("mounted asset manifest repeats a logical path")
        entries[logical] = item
    return entries


def _link_source(source: Path, target: Path, verified_identity: FileIdentity) -> None:
    """Link an already hashed or exact-stat-verified input without rereading its bytes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    resolved_source = source.resolve(strict=True)
    if FileIdentity.from_stat(resolved_source.stat()) != verified_identity:
        raise ValueError("verified input identity changed before project linking")
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve(strict=True) == resolved_source:
            return
        raise FileExistsError("project input destination already exists")
    target.symlink_to(resolved_source)
    if target.resolve(strict=True) != resolved_source:
        raise RuntimeError("project input link changed its source")
    if FileIdentity.from_stat(target.stat()) != verified_identity:
        raise ValueError("verified input identity changed while project linking")


def _safe_extract_source(packet_root: Path, project_root: Path) -> dict[str, str]:
    """Validate every archived regular file before extracting it under a fresh root."""
    manifest_path = packet_root / "source/source_archive_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != manifest.get("member_count"):
        raise ValueError("source archive manifest is malformed")
    expected = {str(item["path"]): item for item in members}
    if len(expected) != len(members):
        raise ValueError("source archive manifest contains duplicate paths")
    project_root.mkdir(parents=True, exist_ok=False)
    source_hashes: dict[str, str] = {}
    archive_path = packet_root / "source/source_archive.tar"
    with tarfile.open(archive_path, "r:") as archive:
        archive_members = archive.getmembers()
        if {member.name for member in archive_members} != set(expected):
            raise ValueError("source archive member set differs from its manifest")
        for member in archive_members:
            relative = safe_relative(member.name)
            entry = expected[member.name]
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("source archive contains a nonregular member")
            if member.size != int(entry["bytes"]):
                raise ValueError("source archive member size differs from its manifest")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("source archive member cannot be read")
            destination = project_root.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            bytes_written = 0
            with source, destination.open("xb") as target:
                while block := source.read(1 << 20):
                    digest.update(block)
                    target.write(block)
                    bytes_written += len(block)
            if bytes_written != member.size or digest.hexdigest() != entry["sha256"]:
                raise ValueError("source archive member digest differs from its manifest")
            source_hashes[member.name] = digest.hexdigest()
    return source_hashes


def _prepare_project(packet_root: Path, project_root: Path) -> dict[str, object]:
    """Extract own code and link the exact shared mounted input closure."""
    source_hashes = _safe_extract_source(packet_root, project_root)
    raw_root = _mount_root(RAW_DATA_ROOT, RAW_DATA_FALLBACK)
    asset_root = _mount_root(ASSET_DATA_ROOT, ASSET_DATA_FALLBACK)
    asset_map = _asset_entries(asset_root)
    portable_path = packet_root / "portable_inputs.json"
    portable = json.loads(portable_path.read_text(encoding="utf-8"))
    sources = portable.get("sources")
    if not isinstance(sources, dict) or set(sources) != {
        "datasets/train.parquet",
        "datasets/valid.parquet",
        "cache/chain_folds.npz",
        "cache/chains.npz",
        "cache/train_targets.npz",
        "cache/valid_targets.npz",
    }:
        raise ValueError("portable source closure differs from the frozen six-file manifest")
    prior = json.loads((packet_root / "prior_input_identity_v2.json").read_text(encoding="utf-8"))
    if prior.get("status") != "matched" or prior.get("split_sha256") != portable.get(
        "split_manifest_sha256"
    ):
        raise ValueError("prior raw input identity is not bound to the frozen split")
    handoff = VerifiedDigestHandoff((raw_root / "train.parquet", raw_root / "valid.parquet"))
    raw_receipts = []
    for logical in ("datasets/train.parquet", "datasets/valid.parquet"):
        split = Path(logical).stem
        source = raw_root / f"{split}.parquet"
        expected = sources[logical]
        prior_observation = prior.get("observed", {}).get(logical)
        if prior_observation is None:
            raise ValueError("prior raw input receipt does not include both source files")
        receipt = _verify_raw_source(handoff, source, logical, expected, prior_observation)
        raw_receipts.append(receipt)
        raw_identity = FileIdentity(**receipt["identity"])
        _link_source(source, project_root / logical, raw_identity)

    asset_hashes = {}
    for logical in (
        "cache/chain_folds.npz",
        "cache/chains.npz",
        "cache/train_targets.npz",
        "cache/valid_targets.npz",
    ):
        item = asset_map.get(logical)
        if item is None:
            raise FileNotFoundError(f"required fold0 cache is absent: {logical}")
        expected = sources[logical]
        if item.get("sha256") != expected.get("sha256") or int(item.get("size", -1)) != int(
            expected["original_identity"]["bytes"]
        ):
            raise ValueError(
                f"asset manifest binding differs from the portable manifest: {logical}"
            )
        source = asset_root / str(item["filename"])
        asset_identity = _sha_matches(
            source, str(expected["sha256"]), int(expected["original_identity"]["bytes"])
        )
        _link_source(source, project_root / logical, asset_identity)
        asset_hashes[logical] = {
            "bytes": asset_identity.size,
            "sha256": str(expected["sha256"]),
            "identity": asdict(asset_identity),
        }

    split_destination = project_root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    split_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(packet_root / "fold0.json", split_destination)
    portable_destination = project_root / "research/campaign_2026-10-04/portable_inputs.json"
    portable_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(portable_path, portable_destination)
    split_sha = sha256_file(split_destination)
    portable_sha = sha256_file(portable_destination)
    if split_sha != portable.get("split_manifest_sha256"):
        raise ValueError("copied campaign split digest changed")
    if portable_sha != sha256_file(portable_path):
        raise ValueError("copied portable manifest digest changed")

    input_receipt = project_root / "input_identity_receipt.json"
    write_json(
        input_receipt,
        {
            "schema": "wnn-gru-capacity-input-identity-v1",
            "portable_manifest_sha256": portable_sha,
            "split_manifest_sha256": split_sha,
            "raw_sources": [
                {
                    "path": row["path"],
                    "sha256": row["sha256"],
                    "bytes": row["bytes"],
                    "identity": row["identity"],
                }
                for row in raw_receipts
            ],
            "asset_sources": asset_hashes,
            "source_hashes": source_hashes,
        },
    )
    return {
        "raw_root": str(raw_root),
        "asset_root": str(asset_root),
        "raw_sources": raw_receipts,
        "asset_sources": asset_hashes,
        "source_hashes": source_hashes,
        "input_identity_receipt": str(input_receipt),
        "project_root": str(project_root),
    }


class GpuMemorySampler:
    """Sample one visible GPU's used/free memory while a child command runs."""

    def __init__(self, interval_seconds: float = 5.0) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[tuple[int, int, int]] = []

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={GPU_DEVICE_INDEX}",
                        "--query-gpu=index,memory.used,memory.free,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                rows = list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))
                if len(rows) == 1 and len(rows[0]) == 4:
                    device, used, free, total = (int(part.strip()) for part in rows[0])
                    if device != GPU_DEVICE_INDEX:
                        raise ValueError("GPU memory sample came from an unselected device")
                    self.samples.append((used, free, total))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, int | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if not self.samples:
            return {"gpu_peak_used_mb": None, "gpu_min_free_mb": None, "gpu_total_mb": None}
        return {
            "gpu_peak_used_mb": max(sample[0] for sample in self.samples),
            "gpu_min_free_mb": min(sample[1] for sample in self.samples),
            "gpu_total_mb": min(sample[2] for sample in self.samples),
        }


def _child_env(
    project_root: Path, arm: str, phase: str, receipt_root: Path, input_identity: Path
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "WNN_PROJECT_ROOT": str(project_root),
            "WNN_ENTRY_MODE": "predict" if phase == "predict" else "train",
            "WNN_INPUT_IDENTITY_RECEIPT": str(input_identity),
            "WNN_PORTABLE_DATA_MANIFEST": str(
                project_root / "research/campaign_2026-10-04/portable_inputs.json"
            ),
            "WNN_DIGEST_METRICS_PATH": str(receipt_root / f"{arm}-{phase}-digest-metrics.json"),
            "WNN_THREADS": "4",
            "WNN_LOADER_WORKERS": "0",
            "WNN_LOADER_RAM_GB": "0.30",
            "WNN_GPU_FRACTION": "0.75",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def _parse_child_line(line: str, arm: str, phase: str, started: float) -> dict[str, object]:
    """Parse only trainer records needed for gates and sanitized progress."""
    for prefix, key in (
        ("PROFILE ", "profile"),
        ("PROFILE_UPDATE_EVIDENCE ", "profile_update_evidence"),
        ("OPTIMIZER_COMPUTE_PRECISION ", "optimizer_precision"),
        ("FIT ", "fit_epoch"),
    ):
        if line.startswith(prefix):
            payload = json.loads(line[len(prefix) :])
            if not isinstance(payload, dict):
                raise ValueError("trainer emitted malformed structured evidence")
            if key == "profile":
                safe_fields = {
                    "step_wall_ms": payload.get("step_wall"),
                }
                _emit(started, "profile", arm=arm, status="running", fields=safe_fields)
            elif key == "profile_update_evidence":
                _emit(
                    started,
                    "profile_gate",
                    arm=arm,
                    status="pass" if payload.get("valid_for_throughput") else "fail",
                    fields={
                        "optimizer_attempts": payload.get("optimizer_attempts"),
                        "completed_updates": payload.get("optimizer_completed_updates"),
                        "amp_skips": payload.get("amp_skipped_updates"),
                    },
                )
            elif key == "fit_epoch":
                updates = payload.get("update_diagnostics", {}).get("epoch", {})
                _emit(
                    started,
                    "fit",
                    arm=arm,
                    fields={
                        "optimizer_attempts": updates.get("optimizer_attempts"),
                        "completed_updates": updates.get("optimizer_completed_updates"),
                        "amp_skips": updates.get("amp_skipped_updates"),
                    },
                )
            return {key: payload}
    if line.startswith("SELECTED_FIXED "):
        return {"selected_fixed": line.removeprefix("SELECTED_FIXED ").strip()}
    return {}


def _run_child(
    *,
    packet_root: Path,
    project_root: Path,
    receipt_root: Path,
    input_identity: Path,
    arm: str,
    phase: str,
    run_name: str,
    argv: list[str],
    worker_started: float,
    worker_deadline: float,
) -> dict[str, object]:
    """Run one exact CLI in a killable subprocess with streamed, sanitized heartbeats."""
    receipt_root.mkdir(parents=True, exist_ok=True)
    argv_path = receipt_root / f"{arm}-{phase}-argv.json"
    write_json(argv_path, {"arm": arm, "phase": phase, "argv": argv})
    log_path = receipt_root / f"{arm}-{phase}.log"
    metrics_path = receipt_root / f"{arm}-{phase}-gpu-samples.json"
    env = _child_env(project_root, arm, phase, receipt_root, input_identity)
    env["WNN_DIGEST_METRICS_PATH"] = str(receipt_root / f"{arm}-{phase}-digest-metrics.json")
    entry = packet_root / "train_entry.py"
    cmd = [sys.executable, str(entry), *argv]
    available = worker_deadline - time.monotonic() - FINALIZATION_RESERVE_SECONDS
    if available <= 0:
        return {"status": "not_started_deadline", "exit_code": None, "log": str(log_path)}
    sampler = GpuMemorySampler()
    sampler.start()
    markers: dict[str, object] = {}
    timed_out = False
    exit_code: int | None = None
    buffer = bytearray()
    last_heartbeat = time.monotonic()
    with log_path.open("wb") as log_stream:
        process = subprocess.Popen(
            cmd,
            cwd=project_root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            start_new_session=True,
        )
        if process.stdout is None:
            raise RuntimeError("trainer child stdout pipe was not created")
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while True:
                now = time.monotonic()
                if now >= worker_deadline - FINALIZATION_RESERVE_SECONDS:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=5)
                    break
                events = selector.select(timeout=1.0)
                for key, _ in events:
                    chunk = os.read(key.fd, 65_536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    log_stream.write(chunk)
                    log_stream.flush()
                    buffer.extend(chunk)
                    while b"\n" in buffer:
                        raw_line, _, tail = buffer.partition(b"\n")
                        buffer = bytearray(tail)
                        line = raw_line.decode("utf-8", errors="replace")
                        markers.update(_parse_child_line(line, arm, phase, worker_started))
                if time.monotonic() - last_heartbeat >= 20:
                    remaining = max(0.0, worker_deadline - time.monotonic())
                    _emit(
                        worker_started,
                        phase if phase != "predict" else "prediction_export",
                        arm=arm,
                        fields={"remaining_s": round(remaining, 1)},
                    )
                    last_heartbeat = time.monotonic()
                exit_code = process.poll()
                if exit_code is not None and not selector.get_map():
                    break
            if buffer:
                line = buffer.decode("utf-8", errors="replace")
                markers.update(_parse_child_line(line, arm, phase, worker_started))
            if exit_code is None:
                exit_code = process.poll()
        finally:
            selector.close()
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            process.stdout.close()
    gpu = sampler.stop()
    write_json(metrics_path, gpu)
    status = "timeout" if timed_out else "complete" if exit_code == 0 else "error"
    return {
        "status": status,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "log": str(log_path),
        "argv_receipt": str(argv_path),
        "gpu": gpu,
        "markers": markers,
        "worker_elapsed_seconds": round(time.monotonic() - worker_started, 3),
    }


def _profile_argv(arm: dict[str, object], batch: int, split_path: Path) -> list[str]:
    """Build the exact archived trainer arguments for one full-shape profile."""
    run_name = str(arm["run_name"])
    argv = [
        "--name",
        run_name,
        "--device",
        "cuda",
        "--arch",
        "gru",
        "--cell",
        "gru",
        "--enc",
        "linear",
        "--hidden",
        str(arm["hidden"]),
        "--layers",
        "2",
        "--dropout",
        "0",
        "--seed",
        "9",
        "--loss",
        "wmse",
        "--loss-norm",
        "chunk",
        "--train-source",
        "train+valid",
        "--train-seqs",
        "10607",
        "--split-manifest",
        str(split_path),
        "--valid-fold",
        "0",
        "--epochs",
        "4",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "3",
        "--no-eval-during-fit",
        "--batch",
        str(batch),
        "--batch-order",
        "seeded",
        "--chunk",
        "2000",
        "--lr",
        "0.001",
        "--wd",
        "0.01",
        "--optim",
        str(arm["optimizer"]),
        "--muon-scope",
        "rnn",
        "--sched",
        "onecycle",
        "--precision",
        "fp16",
        "--amp-init-scale",
        "4096.0",
        "--profile-warmup",
        str(PROFILE_WARMUP_BATCHES),
        "--profile-steps",
        str(PROFILE_TIMED_BATCHES),
    ]
    if arm["optimizer"] == "muon":
        argv += ["--muon-ortho-dtype-policy", "native_bf16_if_supported_else_fp32_v1"]
    return argv


def _fit_argv(arm: dict[str, object], batch: int, split_path: Path) -> list[str]:
    """Build the complete fit command by removing only the profile-specific CLI tokens."""
    profile = _profile_argv(arm, batch, split_path)
    remove_values = {"--profile-warmup", "--profile-steps"}
    output: list[str] = []
    skip_next = False
    for token in profile:
        if skip_next:
            skip_next = False
            continue
        if token in remove_values:
            skip_next = True
            continue
        output.append(token)
    if skip_next:
        raise AssertionError("profile command ended without a profile-option value")
    if set(output) & remove_values:
        raise AssertionError("profile command deriver retained a profile-only option")
    return output


def _predict_argv(run_name: str, split_path: Path) -> list[str]:
    """Build feature-only FP32 prediction argv over the frozen fold evaluation groups."""
    return [
        run_name,
        "valid",
        "--epoch",
        "3",
        "--output-name",
        run_name,
        "--campaign-split-manifest",
        str(split_path),
        "--inference-precision",
        "fp32",
    ]


def _parse_profile_result(result: dict[str, object], arm: str) -> dict[str, object]:
    """Apply the frozen 230-update strict profile gate to the trainer's actual receipts."""
    if result.get("status") != "complete" or result.get("exit_code") != 0:
        raise RuntimeError(f"profile subprocess failed for {arm}")
    markers = result.get("markers", {})
    profile = markers.get("profile")
    evidence = markers.get("profile_update_evidence")
    if not isinstance(profile, dict) or not isinstance(evidence, dict):
        raise ValueError(f"profile evidence is incomplete for {arm}")
    step_wall = float(profile.get("step_wall", float("nan")))
    if not math.isfinite(step_wall) or step_wall <= 0:
        raise ValueError(f"profile timing is not finite and positive for {arm}")
    expected = {
        "optimizer_attempts": 230,
        "optimizer_completed_updates": 230,
        "amp_skipped_updates": 0,
        "loss_checks": 230,
        "grad_scaler_enabled": True,
        "loss_finite": True,
        "gradient_finite": True,
        "valid_for_throughput": True,
    }
    if any(evidence.get(key) != value for key, value in expected.items()):
        raise ValueError(f"profile update contract failed for {arm}")
    return {
        "step_wall_ms": step_wall,
        "profile_metrics": profile,
        "update_evidence": evidence,
        "optimizer_precision": markers.get("optimizer_precision"),
        "gpu_samples": result.get("gpu"),
        "log": result.get("log"),
        "argv_receipt": result.get("argv_receipt"),
    }


def _validate_precision_policy(
    arm: dict[str, object], profile: dict[str, object], hardware: dict[str, object]
) -> None:
    """Check that Muon actually used the declared hardware-resolved compute dtype."""
    if arm["optimizer"] != "muon":
        return
    metadata = profile.get("optimizer_precision")
    if not isinstance(metadata, dict):
        raise ValueError("Muon profile omitted its resolved orthogonalization precision")
    capability = hardware.get("compute_capability")
    native_expected = False
    if isinstance(capability, str):
        major = int(capability.split(".", 1)[0])
        native_expected = major >= 8 and bool(metadata.get("torch_cuda_bf16_supported"))
    expected_dtype = "bfloat16" if native_expected else "float32"
    if metadata.get("policy") != "native_bf16_if_supported_else_fp32_v1":
        raise ValueError("Muon resolved a different orthogonalization policy")
    if metadata.get("selected_dtype") != expected_dtype:
        raise ValueError("Muon resolved dtype disagrees with the current device policy")
    if not isinstance(metadata.get("groups"), list) or not metadata["groups"]:
        raise ValueError("Muon precision metadata has no matrix groups")


def _memory_failure(result: dict[str, object]) -> bool:
    """Recognize only explicit accelerator allocation failures for the B4 fallback."""
    log = Path(str(result.get("log", "")))
    if not log.is_file():
        return False
    content = log.read_text(encoding="utf-8", errors="replace")
    markers = ("CUDA out of memory", "torch.OutOfMemoryError", "CUDNN_STATUS_ALLOC_FAILED")
    return any(marker in content for marker in markers)


def _parse_nvidia_smi_rows(stdout: str) -> list[dict[str, object]]:
    """Parse NVIDIA's index-qualified CSV inventory and reject malformed rows."""
    rows = [row for row in csv.reader(io.StringIO(stdout), skipinitialspace=True) if row]
    if not rows:
        raise ValueError("nvidia-smi returned no accelerator rows")
    parsed: list[dict[str, object]] = []
    seen_indices: set[int] = set()
    for row in rows:
        if len(row) != 6:
            raise ValueError("nvidia-smi inventory row has an unexpected field count")
        try:
            index = int(row[0].strip())
            total_memory = int(row[4].strip())
            free_memory = int(row[5].strip())
        except ValueError as error:
            raise ValueError(
                "nvidia-smi inventory has a nonnumeric index or memory value"
            ) from error
        if index < 0 or total_memory <= 0 or free_memory < 0 or index in seen_indices:
            raise ValueError("nvidia-smi inventory has an invalid or repeated device index")
        seen_indices.add(index)
        parsed.append(
            {
                "index": index,
                "name": row[1].strip(),
                "uuid": row[2].strip(),
                "compute_capability": row[3].strip(),
                "total_memory_mb": total_memory,
                "free_memory_mb": free_memory,
            }
        )
    return parsed


def _parse_selected_nvidia_smi_row(stdout: str, expected_index: int) -> dict[str, object]:
    """Require one nvidia-smi row for the explicitly selected physical GPU."""
    rows = _parse_nvidia_smi_rows(stdout)
    if len(rows) != 1 or rows[0]["index"] != expected_index:
        raise ValueError("nvidia-smi selected a different physical accelerator")
    return rows[0]


def _gpu_hardware() -> dict[str, object]:
    """Record full GPU inventory and pin all profiling telemetry to physical GPU 0."""
    query = "index,name,uuid,compute_cap,memory.total,memory.free"
    inventory_result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    inventory = _parse_nvidia_smi_rows(inventory_result.stdout)
    selected_result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={GPU_DEVICE_INDEX}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    selected = _parse_selected_nvidia_smi_row(selected_result.stdout, GPU_DEVICE_INDEX)
    inventory_match = next((row for row in inventory if row["index"] == GPU_DEVICE_INDEX), None)
    if inventory_match is None or any(
        inventory_match[key] != selected[key]
        for key in ("name", "uuid", "compute_capability", "total_memory_mb")
    ):
        raise ValueError("selected physical GPU does not match the recorded inventory")
    return {
        "selected_physical_gpu_index": GPU_DEVICE_INDEX,
        "selected_physical_gpu": selected,
        "gpu_inventory": inventory,
        "name": selected["name"],
        "uuid": selected["uuid"],
        "compute_capability": selected["compute_capability"],
        "total_memory_mb": selected["total_memory_mb"],
        "free_memory_mb_before_profile": selected["free_memory_mb"],
    }


def _profile_projection_seconds(profile: dict[str, object], batch: int) -> float:
    """Project one fit from its measured full-shape outer-batch wall time."""
    batches_per_epoch = 11940 // batch
    updates = math.ceil(20000 / 2000)
    if batch == 8:
        batches_per_epoch = 1492
    elif batch == 4:
        batches_per_epoch = 2985
    else:
        raise ValueError("unsupported hardware-selected batch")
    if updates != CHUNK_UPDATES_PER_BATCH:
        raise AssertionError("fixed training sequence/chunk geometry changed")
    step_seconds = float(profile["step_wall_ms"]) / 1000.0
    return step_seconds * batches_per_epoch * 4 * FIT_PROJECTION_SAFETY


def _component_map(split_manifest: dict[str, object], source: str) -> dict[int, int]:
    """Return unique row-group-to-component identities for one source namespace."""
    sequences = split_manifest.get("sequences", {}).get(source)
    if not isinstance(sequences, list):
        raise ValueError(f"split manifest has no {source} sequence identities")
    mapping: dict[int, int] = {}
    for row in sequences:
        group_id = int(row["row_group"])
        component_id = int(row["component"])
        if group_id in mapping:
            raise ValueError(f"split manifest repeats a {source} row-group identity")
        mapping[group_id] = component_id
    return mapping


def _validate_source_qualified_fit_membership(
    split_manifest: dict[str, object],
    observed_train_order: list[int],
    observed_valid_order: list[int],
) -> dict[str, object]:
    """Validate ordered fit pools and exclusion/components in their source namespaces."""
    fit = split_manifest.get("fit", {})
    excluded = split_manifest.get("excluded", {})
    evaluation = split_manifest.get("evaluation", {})
    expected_train_order = list(map(int, fit.get("train", [])))
    expected_valid_order = list(map(int, fit.get("valid", [])))
    if observed_train_order != expected_train_order or observed_valid_order != expected_valid_order:
        raise ValueError("checkpoint ordered fit membership differs from the frozen fold")

    excluded_train = set(map(int, excluded.get("train", [])))
    excluded_valid = set(map(int, excluded.get("valid", [])))
    excluded_corrupt_train = set(map(int, excluded.get("corrupt_train", [])))
    evaluation_valid = set(map(int, evaluation.get("valid", [])))
    if excluded_valid != evaluation_valid:
        raise ValueError("valid-source exclusion differs from the frozen evaluation membership")

    observed_train = set(observed_train_order)
    observed_valid = set(observed_valid_order)
    train_overlap = observed_train & (excluded_train | excluded_corrupt_train)
    valid_overlap = observed_valid & excluded_valid
    if train_overlap or valid_overlap:
        raise ValueError("fit membership overlaps excluded groups in its own source namespace")

    train_components = _component_map(split_manifest, "train")
    valid_components = _component_map(split_manifest, "valid")
    expected_train_groups = set(train_components)
    expected_valid_groups = set(valid_components)
    if observed_train | excluded_train != expected_train_groups:
        raise ValueError("train-source fit/excluded pools do not partition frozen clean groups")
    if observed_valid | excluded_valid != expected_valid_groups:
        raise ValueError("valid-source fit/excluded pools do not partition frozen groups")
    if observed_train & excluded_train or observed_valid & excluded_valid:
        raise ValueError("same-source fit and excluded pools overlap")
    if excluded_corrupt_train & (observed_train | excluded_train | expected_train_groups):
        raise ValueError("corrupt train group is mixed into the frozen clean-source membership")

    fit_components = {train_components[group] for group in observed_train} | {
        valid_components[group] for group in observed_valid
    }
    evaluation_components = {valid_components[group] for group in evaluation_valid}
    component_overlap = fit_components & evaluation_components
    if component_overlap:
        raise ValueError("fit and evaluation share a frozen connected component")

    return {
        "status": "pass",
        "train_fit_groups": len(observed_train),
        "valid_fit_groups": len(observed_valid),
        "train_excluded_groups": len(excluded_train),
        "valid_evaluation_groups": len(evaluation_valid),
        "fit_component_count": len(fit_components),
        "evaluation_component_count": len(evaluation_components),
        "cross_source_alias_overlap_counts": {
            "train_fit_vs_valid_evaluation": len(observed_train & evaluation_valid),
            "valid_fit_vs_train_excluded": len(observed_valid & excluded_train),
        },
    }


def _tensor_tree_bitwise_equal(left: object, right: object, torch: object) -> bool:
    """Compare checkpoint state recursively, preserving tensor dtype and exact bytes."""
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            return False
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            return False
        left_bytes = left.detach().cpu().contiguous().view(torch.uint8)
        right_bytes = right.detach().cpu().contiguous().view(torch.uint8)
        return torch.equal(left_bytes, right_bytes)
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict) or left.keys() != right.keys():
            return False
        return all(_tensor_tree_bitwise_equal(left[key], right[key], torch) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(_tensor_tree_bitwise_equal(a, b, torch) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def _validate_saved_checkpoint_pair(
    epoch_checkpoint: dict[str, object],
    alias_checkpoint: dict[str, object],
    expected_args: dict[str, object],
    torch: object,
) -> None:
    """Require equal producer-saved state/config despite distinct torch-save bytes."""
    for field in ("epoch", "n_in", "args", "readout_tilted"):
        if epoch_checkpoint.get(field) != alias_checkpoint.get(field):
            raise ValueError(f"fixed epoch checkpoint and model alias differ in {field}")
    for checkpoint in (epoch_checkpoint, alias_checkpoint):
        if checkpoint.get("epoch") != 3 or checkpoint.get("n_in") != 211:
            raise ValueError("checkpoint epoch or input width differs from the recipe")
        args = checkpoint.get("args", {})
        if any(args.get(key) != value for key, value in expected_args.items()):
            raise ValueError("checkpoint command arguments differ from the frozen recipe")
    for field in ("model", "readout"):
        if not _tensor_tree_bitwise_equal(
            epoch_checkpoint.get(field), alias_checkpoint.get(field), torch
        ):
            raise ValueError(
                f"fixed epoch checkpoint and model alias tensor state differs in {field}"
            )


def _verify_fit_artifacts(
    project_root: Path,
    arm: dict[str, object],
    run_name: str,
    batch: int,
    split_manifest: dict[str, object],
    source_hashes: dict[str, str],
) -> dict[str, object]:
    """Validate frozen epoch, exact fit membership/update counts and finite checkpoint tensors."""
    run_root = project_root / "artifacts/runs" / run_name
    checkpoint_path = run_root / "model_ep3.pt"
    alias_path = run_root / "model.pt"
    history_path = run_root / "history.json"
    provenance_path = run_root / "provenance.json"
    for path in (checkpoint_path, alias_path, history_path, provenance_path):
        if not path.is_file():
            raise FileNotFoundError(f"fit output is missing: {path.name}")
    checkpoint_hash = sha256_file(checkpoint_path)
    alias_hash = sha256_file(alias_path)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or len(history) != 4:
        raise ValueError("fit history does not contain four completed epochs")
    fit_groups = provenance.get("fit_row_groups", {})
    observed_train_order = list(map(int, fit_groups.get("train", [])))
    observed_valid_order = list(map(int, fit_groups.get("valid", [])))
    observed_train = set(observed_train_order)
    observed_valid = set(observed_valid_order)
    membership_validation = _validate_source_qualified_fit_membership(
        split_manifest, observed_train_order, observed_valid_order
    )
    expected_eval = set(map(int, split_manifest["evaluation"]["valid"]))
    if (
        provenance.get("split_manifest_sha256")
        != "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
    ):
        raise ValueError("fit provenance is bound to a different split manifest")
    if provenance.get("trainer_source_sha256") != source_hashes["scripts/train_rnn.py"]:
        raise ValueError("fit provenance trainer source hash changed")
    if provenance.get("model_source_sha256") != source_hashes["src/wnn/nn.py"]:
        raise ValueError("fit provenance model source hash changed")
    expected_batches = 1492 if batch == 8 else 2985
    expected_attempts = expected_batches * CHUNK_UPDATES_PER_BATCH
    provenance_epochs = provenance.get("update_diagnostics", [])
    if not isinstance(provenance_epochs, list) or len(provenance_epochs) != 4:
        raise ValueError("fit provenance does not contain four completed epoch receipts")
    epoch_receipts = []
    sample_orders = []
    for index, row in enumerate(history):
        if row.get("epoch") != index:
            raise ValueError("fit history epoch order is incomplete")
        update = row.get("update_diagnostics", {}).get("epoch", {})
        provenance_epoch = provenance_epochs[index]
        if provenance_epoch.get("epoch_index") != index or provenance_epoch.get("epoch") != update:
            raise ValueError(f"history and provenance update receipts differ in epoch {index}")
        sample_order = provenance_epoch.get("sample_order_identity")
        if not isinstance(sample_order, dict):
            raise ValueError(f"sample-order identity is missing in epoch {index}")
        seeded_order = sample_order.get("seeded_order")
        if (
            sample_order.get("sampler_class") != "EpochOrderSampler"
            or sample_order.get("batch_size") != batch
            or sample_order.get("drop_last") is not True
            or not isinstance(seeded_order, dict)
            or seeded_order.get("seed") != 9
            or seeded_order.get("epoch") != index
            or seeded_order.get("antithetic") is not False
            or seeded_order.get("shared_dropped_tail") is not True
        ):
            raise ValueError(
                f"realized sampler receipt differs from the frozen recipe in epoch {index}"
            )
        normalization = provenance_epoch.get("normalization_identity")
        if normalization != provenance.get("normalization_identity"):
            raise ValueError(f"normalization identity changed during fitting in epoch {index}")
        required = {
            "optimizer_attempts": expected_attempts,
            "optimizer_completed_updates": expected_attempts,
            "amp_skipped_updates": 0,
            "completed_batches": expected_batches,
        }
        if any(update.get(key) != value for key, value in required.items()):
            raise ValueError(f"fit update contract failed in epoch {index}")
        epoch_receipts.append(update)
        sample_orders.append(
            {
                "epoch": index,
                "sampler_class": sample_order["sampler_class"],
                "batch_size": sample_order["batch_size"],
                "drop_last": sample_order["drop_last"],
                "seeded_order": seeded_order,
            }
        )
    if provenance.get("selected_epoch") != 3:
        raise ValueError("provenance did not select the frozen epoch-3 checkpoint")
    producer_hashes = provenance.get("checkpoint_sha256", {})
    if (
        producer_hashes.get("model_ep3.pt") != checkpoint_hash
        or producer_hashes.get("model.pt") != alias_hash
    ):
        raise ValueError("producer checkpoint digests disagree with the two saved files")
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    alias_checkpoint = torch.load(alias_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    expected_args = {
        "name": run_name,
        "seed": 9,
        "hidden": int(arm["hidden"]),
        "layers": 2,
        "batch": batch,
        "batch_order": "seeded",
        "optim": str(arm["optimizer"]),
        "precision": "fp16",
        "amp_init_scale": 4096.0,
        "loss": "wmse",
        "loss_norm": "chunk",
        "train_source": "train+valid",
        "valid_fold": 0,
        "epochs": 4,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 3,
        "no_eval_during_fit": True,
    }
    if args.get("split_manifest_sha256") != provenance.get("split_manifest_sha256"):
        raise ValueError("checkpoint args and provenance bind different split manifests")
    _validate_saved_checkpoint_pair(checkpoint, alias_checkpoint, expected_args, torch)
    tensors = checkpoint.get("model", {})
    for tensor in tensors.values():
        if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all().item():
            raise ValueError("checkpoint has a nonfinite model tensor")
    if arm["optimizer"] == "muon":
        precision = provenance.get("orthogonalization_precision")
        if (
            not isinstance(precision, dict)
            or precision.get("policy") != "native_bf16_if_supported_else_fp32_v1"
        ):
            raise ValueError("Muon's realized orthogonalization precision is missing")
    return {
        "status": "pass",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "model_alias_sha256": alias_hash,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_pair_state_identity": "bitwise_tensor_state_and_config_match",
        "fit_membership_validation": membership_validation,
        "history_sha256": sha256_file(history_path),
        "provenance_sha256": sha256_file(provenance_path),
        "train_group_count": len(observed_train),
        "train_ordered_membership_sha256": hashlib.sha256(
            json.dumps(observed_train_order, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "valid_fit_group_count": len(observed_valid),
        "valid_fit_ordered_membership_sha256": hashlib.sha256(
            json.dumps(observed_valid_order, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "evaluation_group_count": len(expected_eval),
        "epoch_update_receipts": epoch_receipts,
        "sample_order_receipts": sample_orders,
        "normalization_identity": provenance.get("normalization_identity"),
        "orthogonalization_precision": provenance.get("orthogonalization_precision"),
    }


def _validate_prediction(
    project_root: Path,
    run_name: str,
    evaluation_groups: list[int],
    expected_checkpoint_sha256: str,
) -> dict[str, object]:
    """Validate finite FP32 feature-only predictions in exact frozen row-group order."""
    import numpy as np

    prediction_path = project_root / "cache" / f"pred_{run_name}_valid.npy"
    sidecar_path = prediction_path.with_suffix(".json")
    if not prediction_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("feature-only prediction or sidecar is missing")
    predictions = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    if predictions.shape != (len(evaluation_groups), 20_000, 2) or predictions.dtype != np.float32:
        raise ValueError("feature-only prediction shape or dtype differs from the recipe")
    for start in range(0, len(evaluation_groups), 8):
        if not np.isfinite(predictions[start : start + 8]).all():
            raise ValueError("feature-only prediction contains a nonfinite value")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("run") != run_name or sidecar.get("epoch") != 3:
        raise ValueError("prediction sidecar names a different run or fixed epoch")
    if sidecar.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError("prediction sidecar is not bound to the validated fixed checkpoint")
    if sidecar.get("row_groups") != evaluation_groups:
        raise ValueError("prediction sidecar row-group order differs from the frozen evaluation")
    if (
        sidecar.get("campaign_split_manifest_sha256")
        != "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
    ):
        raise ValueError("prediction sidecar is bound to a different campaign split")
    if sidecar.get("inference_precision") != "fp32":
        raise ValueError("prediction export did not use FP32 inference")
    return {
        "status": "pass",
        "prediction": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "prediction_bytes": prediction_path.stat().st_size,
        "sidecar": str(sidecar_path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "row_group_count": len(evaluation_groups),
        "membership_sha256": sidecar.get("evaluation_membership_sha256"),
        "label_mask_reads": False,
    }


def _profile_arm(
    *,
    arm: dict[str, object],
    batch: int,
    split_path: Path,
    packet_root: Path,
    project_root: Path,
    receipt_root: Path,
    input_identity: Path,
    worker_started: float,
    worker_deadline: float,
) -> dict[str, object]:
    """Run and strictly validate one actual full-shape profile."""
    argv = _profile_argv(arm, batch, split_path)
    result = _run_child(
        packet_root=packet_root,
        project_root=project_root,
        receipt_root=receipt_root,
        input_identity=input_identity,
        arm=str(arm["id"]),
        phase="profile",
        run_name=str(arm["run_name"]),
        argv=argv,
        worker_started=worker_started,
        worker_deadline=worker_deadline,
    )
    parsed = _parse_profile_result(result, str(arm["id"]))
    return {**parsed, "subprocess": result}


def _profile_batch(
    *,
    arms: list[dict[str, object]],
    batch: int,
    split_path: Path,
    packet_root: Path,
    project_root: Path,
    receipt_root: Path,
    input_identity: Path,
    worker_started: float,
    worker_deadline: float,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    """Profile every arm at one common batch, preserving each failure receipt."""
    passed: dict[str, dict[str, object]] = {}
    failures: list[dict[str, object]] = []
    for arm in arms:
        result = _run_child(
            packet_root=packet_root,
            project_root=project_root,
            receipt_root=receipt_root,
            input_identity=input_identity,
            arm=str(arm["id"]),
            phase="profile",
            run_name=str(arm["run_name"]),
            argv=_profile_argv(arm, batch, split_path),
            worker_started=worker_started,
            worker_deadline=worker_deadline,
        )
        try:
            parsed = _parse_profile_result(result, str(arm["id"]))
            passed[str(arm["id"])] = {**parsed, "subprocess": result}
        except (RuntimeError, ValueError) as error:
            failures.append(
                {
                    "arm": str(arm["id"]),
                    "error_type": type(error).__name__,
                    "memory_oom": _memory_failure(result),
                    "exit_code": result.get("exit_code"),
                    "log": result.get("log"),
                }
            )
    return passed, failures


def _verify_packet(packet_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Verify exact recipe, source, mounts and runtime bytes before reading inputs."""
    recipe_path = packet_root / RECIPE_NAME
    auth_path = packet_root / AUTH_NAME
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    authorization = json.loads(auth_path.read_text(encoding="utf-8"))
    if recipe.get("packet_id") != PACKET_ID or authorization.get("packet_id") != PACKET_ID:
        raise ValueError("packet identity differs")
    if authorization.get("conditional_fit_authorized") or authorization.get("fit_arm_ids"):
        raise PermissionError("V3 profile-only packet cannot authorize any fit arm")
    if recipe.get("schema") != "wnn-gru-optimizer-capacity-recipe-v2":
        raise ValueError("unsupported capacity recipe schema")
    if authorization.get("schema") != "wnn-gru-capacity-root-conditional-authorization-v2":
        raise ValueError("unsupported root authorization schema")
    for path, binding in recipe["source"]["file_bindings"].items():
        _file_binding(packet_root, str(path), binding)
    if sha256_file(packet_root / RECIPE_NAME) != authorization.get("recipe_sha256"):
        raise ValueError("root authorization does not bind this exact recipe")
    expected_auth = {
        "source_archive_sha256": recipe["source"]["source_archive"]["sha256"],
        "source_manifest_sha256": recipe["source"]["source_manifest"]["sha256"],
        "portable_manifest_sha256": recipe["inputs"]["portable_manifest"]["sha256"],
        "split_manifest_sha256": recipe["inputs"]["split_manifest"]["sha256"],
        "prior_input_identity_sha256": recipe["inputs"]["prior_input_identity"]["sha256"],
        "kernel_metadata_sha256": recipe["source"]["file_bindings"]["kernel-metadata.json"][
            "sha256"
        ],
        "worker_sha256": recipe["source"]["file_bindings"]["capacity_worker.py"]["sha256"],
        "entry_sha256": recipe["source"]["file_bindings"]["kernel.py"]["sha256"],
        "train_entry_sha256": recipe["source"]["file_bindings"]["train_entry.py"]["sha256"],
        "digest_handoff_sha256": recipe["source"]["file_bindings"]["verified_digest_handoff.py"][
            "sha256"
        ],
    }
    if any(authorization.get(key) != value for key, value in expected_auth.items()):
        raise ValueError("root authorization source/input hashes disagree with the recipe")
    return recipe, authorization


def _check_authorized_review(packet_root: Path, authorization: dict[str, object]) -> None:
    """Require both explicit profile authorization and its exact independent review receipt."""
    if not authorization.get("authorized") or not authorization.get("profile_authorized"):
        raise PermissionError("root has not authorized the profile action")
    review_name = safe_relative(str(authorization["root_review_receipt"]))
    review_path = packet_root.joinpath(*review_name.parts)
    if not review_path.is_file() or sha256_file(review_path) != authorization.get(
        "root_review_receipt_sha256"
    ):
        raise PermissionError("root independent review receipt is missing or has changed")
    max_worker = int(authorization.get("max_worker_seconds", 0))
    if max_worker <= 0 or max_worker > WORKER_CAP_SECONDS:
        raise PermissionError("root authorization exceeds the reviewed worker deadline")


def _host_snapshot(root: Path) -> dict[str, object]:
    """Capture bounded host-memory and disk headroom snapshots."""
    disk = shutil.disk_usage(root)
    memory: dict[str, int | None] = {"total_bytes": None, "available_bytes": None}
    try:
        import psutil

        virtual = psutil.virtual_memory()
        memory = {"total_bytes": int(virtual.total), "available_bytes": int(virtual.available)}
    except ImportError:
        pass
    return {
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "memory": memory,
    }


def _evaluate_batch_profiles(
    profiles: dict[str, dict[str, object]],
    arms: list[dict[str, object]],
    hardware: dict[str, object],
) -> None:
    """Require all declared profile arms to pass exact numeric and device gates."""
    if set(profiles) != {str(arm["id"]) for arm in arms}:
        raise ValueError("not every arm has a profile at the common selected batch")
    for arm in arms:
        _validate_precision_policy(arm, profiles[str(arm["id"])], hardware)
        samples = profiles[str(arm["id"])].get("gpu_samples")
        if not isinstance(samples, dict) or samples.get("gpu_peak_used_mb") is None:
            raise ValueError("profile lacks actual GPU memory telemetry")


def _fit_arm(
    *,
    arm: dict[str, object],
    batch: int,
    split_path: Path,
    packet_root: Path,
    project_root: Path,
    receipt_root: Path,
    input_identity: Path,
    worker_started: float,
    worker_deadline: float,
    split_manifest: dict[str, object],
    source_hashes: dict[str, str],
    evaluation_groups: list[int],
) -> dict[str, object]:
    """Run one exact four-epoch fit, fixed checkpoint, and 468-row feature export."""
    arm_id = str(arm["id"])
    run_name = str(arm["run_name"])
    result: dict[str, object] = {"arm": arm_id, "run_name": run_name}
    fit = _run_child(
        packet_root=packet_root,
        project_root=project_root,
        receipt_root=receipt_root,
        input_identity=input_identity,
        arm=arm_id,
        phase="fit",
        run_name=run_name,
        argv=_fit_argv(arm, batch, split_path),
        worker_started=worker_started,
        worker_deadline=worker_deadline,
    )
    result["fit_subprocess"] = fit
    if fit.get("status") != "complete" or fit.get("exit_code") != 0:
        result["status"] = "fit_failed"
        return result
    try:
        fit_validation = _verify_fit_artifacts(
            project_root, arm, run_name, batch, split_manifest, source_hashes
        )
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        result["status"] = "fit_contract_failed"
        result["fit_validation_error_type"] = type(error).__name__
        return result
    result["fit_validation"] = fit_validation
    prediction = _run_child(
        packet_root=packet_root,
        project_root=project_root,
        receipt_root=receipt_root,
        input_identity=input_identity,
        arm=arm_id,
        phase="predict",
        run_name=run_name,
        argv=_predict_argv(run_name, split_path),
        worker_started=worker_started,
        worker_deadline=worker_deadline,
    )
    result["prediction_subprocess"] = prediction
    if prediction.get("status") != "complete" or prediction.get("exit_code") != 0:
        result["status"] = "prediction_failed"
        return result
    try:
        prediction_validation = _validate_prediction(
            project_root,
            run_name,
            evaluation_groups,
            str(fit_validation["checkpoint_sha256"]),
        )
    except (OSError, ValueError, KeyError) as error:
        result["status"] = "prediction_contract_failed"
        result["prediction_validation_error_type"] = type(error).__name__
        return result
    result["prediction_validation"] = prediction_validation
    result["status"] = "complete"
    return result


def _validate_pair_consistency(fit_results: dict[str, object]) -> dict[str, object]:
    """Require completed arms to share ordered membership and sampler receipts."""
    required = ("muon512", "muon768")
    if any(fit_results.get(arm, {}).get("status") != "complete" for arm in required):
        return {"status": "incomplete", "required_arms": list(required)}
    completed = [
        result
        for result in fit_results.values()
        if isinstance(result, dict) and result.get("status") == "complete"
    ]
    baseline = completed[0]["fit_validation"]
    comparison_keys = (
        "train_ordered_membership_sha256",
        "valid_fit_ordered_membership_sha256",
        "sample_order_receipts",
        "normalization_identity",
        "epoch_update_receipts",
    )
    for result in completed[1:]:
        validation = result["fit_validation"]
        if any(validation.get(key) != baseline.get(key) for key in comparison_keys):
            raise ValueError("completed arms disagree on fit membership or sampler contract")
    return {
        "status": "pass",
        "arms": [result["arm"] for result in completed],
        "shared_train_ordered_membership_sha256": baseline["train_ordered_membership_sha256"],
        "shared_valid_ordered_membership_sha256": baseline["valid_fit_ordered_membership_sha256"],
        "shared_sample_order_receipts": baseline["sample_order_receipts"],
        "shared_normalization_identity": baseline["normalization_identity"],
    }


def run(packet_root: Path) -> dict[str, object]:
    """Execute only a root-approved, bounded profile and its separately authorized fit."""
    started = time.monotonic()
    deadline = started + WORKER_CAP_SECONDS
    receipt_root = Path("/kaggle/working/gru512_768_muon_capacity_v3_profile_receipts")
    receipt_root.mkdir(parents=True, exist_ok=True)
    terminal_path = receipt_root / "cohort_terminal_receipt.json"
    outcome: dict[str, object] = {
        "packet_id": PACKET_ID,
        "status": "starting",
        "fit_started": False,
        "worker_seconds": 0.0,
    }
    try:
        _emit(started, "bootstrap")
        recipe, authorization = _verify_packet(packet_root)
        outcome.update(
            {
                "recipe_sha256": sha256_file(packet_root / RECIPE_NAME),
                "authorization_sha256": sha256_file(packet_root / AUTH_NAME),
                "source_archive_sha256": recipe["source"]["source_archive"]["sha256"],
                "source_manifest_sha256": recipe["source"]["source_manifest"]["sha256"],
            }
        )
        if not authorization.get("authorized") or not authorization.get("profile_authorized"):
            outcome["status"] = "authorization_pending"
            outcome["worker_seconds"] = time.monotonic() - started
            write_json(terminal_path, outcome)
            _emit(started, "terminal", status="skipped")
            return outcome
        _check_authorized_review(packet_root, authorization)
        if time.monotonic() >= deadline - FINALIZATION_RESERVE_SECONDS:
            raise TimeoutError("worker deadline expired before source preparation")

        project_root = Path("/kaggle/working/gru512_768_muon_capacity_v3_profile_project")
        _emit(started, "source_verify")
        prepared = _prepare_project(packet_root, project_root)
        _emit(started, "input_mount", status="pass")
        host_before = _host_snapshot(project_root)
        hardware = _gpu_hardware()
        source_hashes = prepared["source_hashes"]
        split_path = project_root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
        split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
        evaluation_groups = list(map(int, split_manifest["evaluation"]["valid"]))
        if len(evaluation_groups) != 468 or evaluation_groups != sorted(set(evaluation_groups)):
            raise ValueError(
                "frozen fold evaluation membership differs from the 468-group contract"
            )
        expected_fit = recipe["fit_membership"]
        if (
            len(split_manifest["fit"]["train"]) != expected_fit["train_groups"]
            or len(split_manifest["fit"]["valid"]) != expected_fit["valid_fit_groups"]
        ):
            raise ValueError("frozen split fitting membership differs from the recipe")

        input_receipt_path = Path(prepared["input_identity_receipt"])
        outcome.update(
            {
                "input_receipt": str(input_receipt_path),
                "input_receipt_sha256": sha256_file(input_receipt_path),
                "host_before_profile": host_before,
                "hardware": hardware,
            }
        )
        arm_defs = [dict(arm) for arm in ARM_SPECS]
        required_arm_defs = [arm for arm in arm_defs if arm["primary"]]
        optional_arm_defs = [arm for arm in arm_defs if not arm["primary"]]
        for arm in arm_defs:
            arm["run_name"] = (
                f"bd26_gru{arm['hidden']}_full_s9_f0_{arm['optimizer']}_seeded_"
                f"b{{batch}}_amp4096_capacityv3profile_20261008"
            )

        profiles_by_batch: dict[str, object] = {}
        batch_failures: dict[str, object] = {}
        optional_profile_failures: list[dict[str, object]] = []
        selected_batch: int | None = None
        for batch in (8, 4):
            arms_at_batch = [
                {**arm, "run_name": str(arm["run_name"]).format(batch=batch)}
                for arm in required_arm_defs
            ]
            profiles, failures = _profile_batch(
                arms=arms_at_batch,
                batch=batch,
                split_path=split_path,
                packet_root=packet_root,
                project_root=project_root,
                receipt_root=receipt_root,
                input_identity=input_receipt_path,
                worker_started=started,
                worker_deadline=deadline,
            )
            profiles_by_batch[str(batch)] = profiles
            batch_failures[str(batch)] = failures
            if not failures:
                _evaluate_batch_profiles(profiles, arms_at_batch, hardware)
                selected_batch = batch
                outcome["selected_batch"] = batch
                optional_arms_at_batch = [
                    {**arm, "run_name": str(arm["run_name"]).format(batch=batch)}
                    for arm in optional_arm_defs
                ]
                optional_profiles, optional_profile_failures = _profile_batch(
                    arms=optional_arms_at_batch,
                    batch=batch,
                    split_path=split_path,
                    packet_root=packet_root,
                    project_root=project_root,
                    receipt_root=receipt_root,
                    input_identity=input_receipt_path,
                    worker_started=started,
                    worker_deadline=deadline,
                )
                for arm in optional_arms_at_batch:
                    arm_id = str(arm["id"])
                    if arm_id in optional_profiles:
                        try:
                            _evaluate_batch_profiles(
                                {arm_id: optional_profiles[arm_id]}, [arm], hardware
                            )
                        except (KeyError, ValueError) as error:
                            optional_profile_failures.append(
                                {"arm": arm_id, "error_type": type(error).__name__}
                            )
                            del optional_profiles[arm_id]
                profiles_by_batch[str(batch)] = {**profiles, **optional_profiles}
                outcome["profiles"] = profiles_by_batch
                outcome["profile_failures"] = batch_failures
                outcome["optional_profile_failures"] = optional_profile_failures
                break
            if batch == 8 and failures and all(failure["memory_oom"] for failure in failures):
                _emit(started, "profile_gate", status="running", fields={"batch": 4})
                continue
            outcome["status"] = "profile_failed"
            outcome["profiles"] = profiles_by_batch
            outcome["profile_failures"] = batch_failures
            raise RuntimeError("a non-memory or B4 profile gate failed")

        if selected_batch is None:
            outcome["status"] = "profile_failed"
            outcome["profiles"] = profiles_by_batch
            outcome["profile_failures"] = batch_failures
            raise RuntimeError("no common hardware-selected batch passed all profiles")

        auth_fit = bool(authorization.get("conditional_fit_authorized"))
        fit_ids = authorization.get("fit_arm_ids", [])
        if not auth_fit:
            outcome["status"] = "profile_complete_fit_not_authorized"
            outcome["fit_started"] = False
            outcome["worker_seconds"] = time.monotonic() - started
            write_json(terminal_path, outcome)
            _emit(started, "terminal", status="profile_only")
            return outcome
        if authorization.get("selected_batch") != selected_batch:
            raise PermissionError("root fit authorization selects a different profiled batch")
        if not isinstance(fit_ids, list) or not {"muon512", "muon768"}.issubset(set(fit_ids)):
            raise PermissionError(
                "root fit authorization must include the complete primary Muon pair"
            )
        if not set(fit_ids).issubset({"muon512", "muon768", "adamw512"}):
            raise PermissionError("root fit authorization includes an unknown arm")

        selected_profiles = profiles_by_batch[str(selected_batch)]
        if "adamw512" in fit_ids and "adamw512" not in selected_profiles:
            raise PermissionError(
                "root fit authorization includes an AdamW arm without a passing profile"
            )
        projected: dict[str, float] = {}
        for arm in arm_defs:
            arm_id = str(arm["id"])
            if arm_id in selected_profiles:
                projected[arm_id] = _profile_projection_seconds(
                    selected_profiles[arm_id], selected_batch
                )
        fit_arm_defs = [arm for arm in arm_defs if arm["id"] in fit_ids]
        pair_projection = sum(projected[arm["id"]] for arm in fit_arm_defs)
        fit_setup_reserve = FIT_STARTUP_RESERVE_SECONDS * len(fit_arm_defs)
        export_reserve = PREDICTION_RESERVE_SECONDS * len(fit_arm_defs)
        elapsed = time.monotonic() - started
        projected_total = (
            elapsed
            + pair_projection
            + fit_setup_reserve
            + export_reserve
            + FINALIZATION_RESERVE_SECONDS
        )
        max_authorized = min(int(authorization["max_worker_seconds"]), WORKER_CAP_SECONDS)
        outcome["profile_projection"] = {
            "per_arm_fit_seconds": projected,
            "authorized_fit_arms": fit_ids,
            "projected_fit_seconds": pair_projection,
            "fit_startup_reserve_seconds": fit_setup_reserve,
            "prediction_export_reserve_seconds": export_reserve,
            "elapsed_before_fit_seconds": elapsed,
            "worker_finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
            "projected_total_worker_seconds": projected_total,
            "authorized_worker_seconds": max_authorized,
        }
        if projected_total > max_authorized:
            outcome["status"] = "profile_complete_fit_projection_rejected"
            outcome["fit_started"] = False
            outcome["worker_seconds"] = time.monotonic() - started
            write_json(terminal_path, outcome)
            _emit(
                started,
                "profile_gate",
                status="skipped",
                fields={
                    "fit_projection_s": round(projected_total, 1),
                    "worker_cap_s": max_authorized,
                },
            )
            _emit(started, "terminal", status="profile_only")
            return outcome

        outcome["status"] = "fit_started"
        outcome["fit_started"] = True
        fit_results: dict[str, object] = {}
        for arm in fit_arm_defs:
            run_arm = {**arm, "run_name": str(arm["run_name"]).format(batch=selected_batch)}
            fit_results[str(arm["id"])] = _fit_arm(
                arm=run_arm,
                batch=selected_batch,
                split_path=split_path,
                packet_root=packet_root,
                project_root=project_root,
                receipt_root=receipt_root,
                input_identity=input_receipt_path,
                worker_started=started,
                worker_deadline=deadline,
                split_manifest=split_manifest,
                source_hashes=source_hashes,
                evaluation_groups=evaluation_groups,
            )
            write_json(receipt_root / f"{arm['id']}-fit-terminal.json", fit_results[str(arm["id"])])
            if time.monotonic() >= deadline - FINALIZATION_RESERVE_SECONDS:
                break
        outcome["fit_results"] = fit_results
        primary_done = all(
            fit_results.get(name, {}).get("status") == "complete" for name in ("muon512", "muon768")
        )
        try:
            pair_consistency = _validate_pair_consistency(fit_results)
        except (KeyError, TypeError, ValueError) as error:
            pair_consistency = {"status": "fail", "error_type": type(error).__name__}
        outcome["paired_fit_consistency"] = pair_consistency
        optional_requested = "adamw512" in fit_ids
        optional_done = (
            not optional_requested or fit_results.get("adamw512", {}).get("status") == "complete"
        )
        if pair_consistency.get("status") == "fail":
            outcome["status"] = "fit_pair_contract_failed"
        else:
            outcome["status"] = (
                "fit_complete"
                if primary_done and pair_consistency.get("status") == "pass" and optional_done
                else "fit_partial"
            )
        outcome["worker_seconds"] = time.monotonic() - started
        write_json(terminal_path, outcome)
        _emit(started, "terminal", status="pass" if outcome["status"] == "fit_complete" else "fail")
        return outcome
    except BaseException as error:
        outcome.setdefault("status", "worker_error")
        outcome["error_type"] = type(error).__name__
        outcome["worker_seconds"] = time.monotonic() - started
        try:
            write_json(terminal_path, outcome)
        except OSError:
            pass
        _emit(started, "terminal", status="fail")
        raise
