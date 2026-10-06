from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import math
import subprocess
import tarfile
import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PACKET_DIR = HERE / "packet"
TRAINER_COMMIT = "69b2331825e193a070015ca4182872724595209a"
TRAINER_SHA256 = "8a29459e998f7f5d6a03b7a4b87c6f4b569002e65b8c87127c37138360909cbf"
SPLIT_PATH = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
ASSETS_MANIFEST = ROOT / "research/campaign_2026-10-04/kaggle/fold0_assets/assets-manifest.json"
PORTABLE_INPUTS = ROOT / "research/campaign_2026-10-04/portable_inputs.json"
SOURCE_FAMILIES = ROOT / "research/campaign_2026-10-06/bandit_continuation/SOURCE_FAMILIES.json"
ASSETS_SHA256 = "19fe0eb6445553c2adff00e821a4fa16ca5969473ca8be420c32cb8e4b4b72c7"
PORTABLE_SHA256 = "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c"
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
SOURCE_FAMILIES_SHA256 = "499a66270d71db6e8721a7e20b42be8704cbc3c47c31b55d3eef2ebe8846b24d"
SOURCE_MANIFEST_SHA256 = "851898475dfa701f2089ec76cfe9592538aa2e7f9b70ebcb68c0a614e332d6fb"
SOURCE_CHECKPOINT_SHA256 = "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e"
PROFILE_ALLOCATOR_BYTES = 4_300_000_000
MAX_PAIR_SECONDS = 6_300


def sha256_bytes(value: bytes) -> str:
    """Return a stable identity for exact packet and trainer source bytes."""
    return hashlib.sha256(value).hexdigest()


def validate_profile_receipt(
    receipt_bytes: bytes,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Validate E508's profile structure and exact caller-supplied file identity."""
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise ValueError("expected E508 profile hash must be a lowercase SHA-256")
    if sha256_bytes(receipt_bytes) != expected_sha256:
        raise ValueError("E508 profile receipt bytes do not match the supplied digest")
    receipt = json.loads(receipt_bytes)
    if not isinstance(receipt, dict):
        raise ValueError("E508 profile receipt must be a JSON object")
    expected_counts = {"train": 10_535, "valid": 1_405}
    expected_targets = {"train": 10_606, "valid": 1_873}
    if (
        receipt.get("packet") != "E508_RAW_BASELINE_V1"
        or receipt.get("status") != "profile_ready_awaiting_controller_review"
        or receipt.get("fit_admitted") is not False
        or receipt.get("fit_stages_started") != []
        or receipt.get("split_sha256") != SPLIT_SHA256
        or receipt.get("fit_group_counts") != expected_counts
        or receipt.get("evaluation_valid_groups") != 468
        or receipt.get("target_row_counts") != expected_targets
    ):
        raise ValueError("receipt is not the E508 fold-0 no-fit profile result")
    profile = receipt.get("profile")
    if not isinstance(profile, dict) or not isinstance(profile.get("metrics"), dict):
        raise ValueError("E508 result omits measured profile fields")
    try:
        raw_step_wall = profile["metrics"]["step_wall"]
        raw_peak_reserved = profile["peak_reserved_bytes"]
        raw_projected = profile["estimated_pair_fit_and_export_seconds"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("E508 profile fields are incomplete or malformed") from error
    if (
        isinstance(raw_step_wall, bool)
        or not isinstance(raw_step_wall, (int, float))
        or isinstance(raw_peak_reserved, bool)
        or not isinstance(raw_peak_reserved, int)
        or isinstance(raw_projected, bool)
        or not isinstance(raw_projected, (int, float))
    ):
        raise ValueError("E508 profile values must use JSON numeric types")
    step_wall = float(raw_step_wall)
    peak_reserved = raw_peak_reserved
    projected = float(raw_projected)
    if (
        not math.isfinite(step_wall)
        or step_wall <= 0
        or not 0 <= peak_reserved <= PROFILE_ALLOCATOR_BYTES
        or not math.isfinite(projected)
        or not 0 < projected <= MAX_PAIR_SECONDS
    ):
        raise ValueError("E508 timing projection or allocator peak exceeds E510 admission bounds")
    hardware = receipt.get("hardware")
    devices = hardware.get("cuda_devices") if isinstance(hardware, dict) else None
    if (
        not isinstance(hardware, dict)
        or hardware.get("cuda_available") is not True
        or not isinstance(devices, list)
        or not devices
        or not isinstance(devices[0], dict)
        or "Tesla T4" not in str(devices[0].get("name", ""))
    ):
        raise ValueError("E508 profile did not verify the declared T4 runtime")
    return receipt


def source_archive() -> tuple[bytes, dict[str, str]]:
    """Read only the pinned production trainer and its WNN source modules."""
    archive = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "archive",
            "--format=tar",
            TRAINER_COMMIT,
            "scripts/train_rnn.py",
            "src/wnn",
        ],
        check=True,
        capture_output=True,
    ).stdout
    identities: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        for member in bundle.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"unexpected non-file in pinned archive: {member.name}")
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise ValueError(f"source archive member is unreadable: {member.name}")
            identities[member.name] = sha256_bytes(extracted.read())
    if identities.get("scripts/train_rnn.py") != TRAINER_SHA256:
        raise ValueError("pinned production trainer bytes differ from the sealed hash")
    if "src/wnn/splits.py" not in identities or "src/wnn/portability.py" not in identities:
        raise ValueError("pinned source archive omits campaign split validation")
    return archive, identities


def sha256_file(path: Path) -> str:
    """Hash a potentially large downloaded terminal output in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def controller_review_targets(
    profile_receipt: bytes, profile_receipt_sha256: str
) -> dict[str, object]:
    """Compute exact E510 code identities for the external controller review."""
    validate_profile_receipt(profile_receipt, expected_sha256=profile_receipt_sha256)
    for path, expected in (
        (SPLIT_PATH, SPLIT_SHA256),
        (ASSETS_MANIFEST, ASSETS_SHA256),
        (PORTABLE_INPUTS, PORTABLE_SHA256),
        (SOURCE_FAMILIES, SOURCE_FAMILIES_SHA256),
    ):
        if sha256_bytes(path.read_bytes()) != expected:
            raise ValueError(f"review input identity changed: {path}")
    source, source_hashes = source_archive()
    worker = (HERE / "packet_worker.py").read_bytes()
    kernel = kernel_source(
        source,
        source_hashes,
        worker,
        profile_receipt,
        profile_receipt_sha256,
        SOURCE_FAMILIES.read_bytes(),
    )
    return {
        "kernel_sha256": sha256_bytes(kernel),
        "worker_sha256": sha256_bytes(worker),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "trainer_sha256": source_hashes["scripts/train_rnn.py"],
        "source_archive_sha256": sha256_bytes(source),
    }


def validate_controller_review_receipt(
    review_bytes: bytes,
    *,
    expected_review_sha256: str,
    profile_receipt_sha256: str,
    terminal_output_sha256: str,
    code_identities: dict[str, object],
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Require separate controller approval tied to E508 evidence and E510 code."""
    if sha256_bytes(review_bytes) != expected_review_sha256:
        raise ValueError("controller review receipt bytes do not match the supplied digest")
    review = json.loads(review_bytes)
    if not isinstance(review, dict):
        raise ValueError("controller review receipt must be a JSON object")
    if (
        review.get("format") != "wnn.controller.e510-review.v1"
        or review.get("version") != 1
        or review.get("status") != "approved"
        or review.get("packet_id") != "E510_RAW_TEACHER_PAIR_V1"
    ):
        raise ValueError("separate controller review receipt is missing or not approved")
    profile_decision = review.get("profile_review")
    if not isinstance(profile_decision, dict) or profile_decision.get("status") != "approved":
        raise ValueError("controller did not approve the actual E508 profile")
    expected_profile = {
        "packet_id": "E508_RAW_BASELINE_V1",
        "receipt_sha256": profile_receipt_sha256,
        "terminal_output_sha256": terminal_output_sha256,
    }
    for key, value in expected_profile.items():
        if profile_decision.get(key) != value:
            raise ValueError(f"controller receipt does not bind E508 {key}")
    code_review = review.get("e510_code_review")
    if not isinstance(code_review, dict) or code_review.get("status") != "approved":
        raise ValueError("controller did not approve the revised E510 code")
    for key, value in code_identities.items():
        if code_review.get(key) != value:
            raise ValueError(f"controller receipt does not bind E510 {key}")

    budget = review.get("production_budget")
    if (
        not isinstance(budget, dict)
        or budget.get("status") != "approved"
        or budget.get("provider") != "kaggle"
        or budget.get("provider_watch_seconds") != 7_200
        or budget.get("worker_wall_cap_seconds") != 6_300
        or budget.get("worst_case_dual_t4_gpu_hours") != 4.0
        or budget.get("ram_headroom_fraction") != 0.2
        or budget.get("ram_monitor_required") is not True
        or budget.get("fresh_quota_checked") is not True
    ):
        raise ValueError("controller production-budget approval is incomplete")
    try:
        remaining_hours = float(budget["gpu_hours_remaining"])
        reserved_hours = float(budget["gpu_hours_reserved"])
        quota_time = datetime.fromisoformat(
            str(budget["quota_checked_at_utc"]).replace("Z", "+00:00")
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("controller quota evidence is incomplete") from error
    if (
        not math.isfinite(remaining_hours)
        or not math.isfinite(reserved_hours)
        or remaining_hours < 4.0
        or reserved_hours < 4.0
        or remaining_hours < reserved_hours
        or quota_time.tzinfo is None
    ):
        raise ValueError("controller quota evidence does not cover the frozen pair budget")
    current_time = now_utc or datetime.now(UTC)
    quota_age = current_time - quota_time.astimezone(UTC)
    if quota_age < timedelta(0) or quota_age > timedelta(hours=6):
        raise ValueError("provider quota evidence is stale or future-dated")
    return review


def compressed_source(source: bytes) -> str:
    """Encode compressed bytes as source-safe base85 for Kaggle's single script."""
    return base64.b85encode(zlib.compress(source, level=9)).decode("ascii")


def kernel_source(
    archive: bytes,
    source_hashes: dict[str, str],
    worker: bytes,
    profile_receipt: bytes,
    profile_receipt_sha256: str,
    source_families: bytes,
) -> bytes:
    """Embed the reviewed E508 profile receipt as a runtime admission gate."""
    constants = (
        f"PACKET_ID = {'E510_RAW_TEACHER_PAIR_V1'!r}\n"
        f"SOURCE_ARCHIVE = {compressed_source(archive)!r}\n"
        f"SOURCE_ARCHIVE_SHA256 = {sha256_bytes(archive)!r}\n"
        f"SOURCE_HASHES = {source_hashes!r}\n"
        f"WORKER_CODE = {compressed_source(worker)!r}\n"
        f"WORKER_SHA256 = {sha256_bytes(worker)!r}\n"
        f"PROFILE_RECEIPT = {compressed_source(profile_receipt)!r}\n"
        f"PROFILE_RECEIPT_SHA256 = {profile_receipt_sha256!r}\n"
        f"SOURCE_FAMILIES = {compressed_source(source_families)!r}\n"
        f"SOURCE_FAMILIES_SHA256 = {sha256_bytes(source_families)!r}\n"
        "GLOBAL_TIMEOUT_SECONDS = 6300\n"
    )
    template = r'''from __future__ import annotations

import base64
import codecs
import contextlib
import hashlib
import io
import json
import os
import select
import signal
import subprocess
import sys
import tarfile
import time
import zlib
from pathlib import Path, PurePosixPath

__CONSTANTS__

RUNTIME = Path("/kaggle/working/e510_raw_teacher_pair")
PROJECT = RUNTIME / "source"
ENTRY = {"packet": PACKET_ID, "started_unix": time.time(), "pid": os.getpid()}


def write_json(path: Path, value: dict[str, object]) -> None:
    """Persist wrapper state atomically before starting Torch work."""
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def decode_source(value: str, expected_hash: str) -> bytes:
    """Decode one embedded source and refuse byte drift before execution."""
    decoded = zlib.decompress(base64.b85decode(value.encode("ascii")))
    if hashlib.sha256(decoded).hexdigest() != expected_hash:
        raise ValueError("embedded packet payload hash mismatch")
    return decoded


def extract_sources(archive: bytes) -> None:
    """Write regular files from the pinned archive without unsafe extraction."""
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        for member in bundle.getmembers():
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe source archive member: {member.name}")
            path = PROJECT.joinpath(*relative.parts)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"unexpected non-file source member: {member.name}")
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise ValueError(f"source member cannot be read: {member.name}")
            data = extracted.read()
            expected = SOURCE_HASHES.get(member.name)
            if expected is None or hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"source file hash mismatch: {member.name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    if len(SOURCE_HASHES) != sum(1 for path in PROJECT.rglob("*") if path.is_file()):
        raise ValueError("extracted source tree count differs from the build receipt")


def stop_worker(process: subprocess.Popen[bytes]) -> None:
    """Terminate native work by process group, then kill after a fixed grace."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main() -> None:
    """Run the E510 worker only with the embedded controller-reviewed profile."""
    print(json.dumps({"event": "kernel_entry", **ENTRY}), flush=True)
    RUNTIME.mkdir(parents=True, exist_ok=False)
    write_json(RUNTIME / "wrapper_status.json", {**ENTRY, "phase": "source_unpack"})
    process: subprocess.Popen[bytes] | None = None
    try:
        archive = decode_source(SOURCE_ARCHIVE, SOURCE_ARCHIVE_SHA256)
        worker = decode_source(WORKER_CODE, WORKER_SHA256)
        profile_receipt = decode_source(PROFILE_RECEIPT, PROFILE_RECEIPT_SHA256)
        source_families = decode_source(SOURCE_FAMILIES, SOURCE_FAMILIES_SHA256)
        if hashlib.sha256(source_families).hexdigest() != "499a66270d71db6e8721a7e20b42be8704cbc3c47c31b55d3eef2ebe8846b24d":
            raise ValueError("embedded SOURCE_FAMILIES registration hash mismatch")
        PROJECT.mkdir(parents=True, exist_ok=False)
        extract_sources(archive)
        worker_path = PROJECT / "packet_worker.py"
        worker_path.write_bytes(worker)
        (RUNTIME / "e508_profile_receipt.json").write_bytes(profile_receipt)
        (RUNTIME / "SOURCE_FAMILIES.json").write_bytes(source_families)
        env = os.environ.copy()
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "WNN_E510_RUNTIME": str(RUNTIME),
                "WNN_E508_PROFILE_RECEIPT_SHA256": PROFILE_RECEIPT_SHA256,
                "PYTHONPATH": os.pathsep.join(
                    [str(PROJECT / "src"), str(PROJECT / "scripts"), env.get("PYTHONPATH", "")]
                ),
            }
        )
        print(
            json.dumps(
                {
                    "event": "source_verified",
                    "trainer_commit": "69b2331825e193a070015ca4182872724595209a",
                    "trainer_sha256": SOURCE_HASHES["scripts/train_rnn.py"],
                    "worker_sha256": WORKER_SHA256,
                    "e508_profile_receipt_sha256": PROFILE_RECEIPT_SHA256,
                    "source_families_sha256": SOURCE_FAMILIES_SHA256,
                    "source_files": len(SOURCE_HASHES),
                }
            ),
            flush=True,
        )
        log_path = RUNTIME / "worker.log"
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, str(worker_path)],
                cwd=PROJECT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,
                bufsize=0,
                start_new_session=True,
            )
            output = process.stdout
            if output is None:
                raise RuntimeError("worker stdout pipe was not created")
            output_fd = output.fileno()
            os.set_blocking(output_fd, False)
            decoder = codecs.getincrementaldecoder("utf-8")()

            def forward_output(data: bytes, *, final: bool = False) -> None:
                text = decoder.decode(data, final=final)
                if text:
                    log.write(text)
                    log.flush()
                    sys.stdout.write(text)
                    sys.stdout.flush()

            deadline = time.monotonic() + GLOBAL_TIMEOUT_SECONDS
            timed_out = False
            while process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    write_json(
                        RUNTIME / "wrapper_status.json",
                        {**ENTRY, "phase": "hard_timeout", "timeout_seconds": GLOBAL_TIMEOUT_SECONDS},
                    )
                    stop_worker(process)
                    break
                ready, _, _ = select.select([output_fd], [], [], min(15.0, remaining))
                if ready:
                    with contextlib.suppress(BlockingIOError):
                        forward_output(os.read(output_fd, 65_536))
            while True:
                try:
                    chunk = os.read(output_fd, 65_536)
                except BlockingIOError:
                    break
                if not chunk:
                    break
                forward_output(chunk)
            forward_output(b"", final=True)
            return_code = process.wait()
        write_json(
            RUNTIME / "wrapper_status.json",
            {
                **ENTRY,
                "phase": "timed_out" if timed_out else "worker_finished",
                "worker_return_code": return_code,
                "worker_log": str(log_path),
                "finished_unix": time.time(),
            },
        )
        if timed_out or return_code != 0:
            raise SystemExit(124 if timed_out else return_code)
    except BaseException as error:
        failure: dict[str, object] = {
            **ENTRY,
            "phase": "wrapper_failed",
            "error": repr(error),
            "finished_unix": time.time(),
        }
        if process is not None and process.poll() is None:
            try:
                stop_worker(process)
            except Exception as stop_error:
                failure["child_cleanup_error"] = repr(stop_error)
        write_json(RUNTIME / "wrapper_status.json", failure)
        raise


if __name__ == "__main__":
    main()
'''
    result = template.replace("__CONSTANTS__", constants)
    ast.parse(result)
    return result.encode("utf-8")


def training_args(name: str, seed: int) -> list[str]:
    """Mirror the worker's fixed raw-only fit command for controller review."""
    return [
        "--name",
        name,
        "--device",
        "cuda:0",
        "--arch",
        "gru",
        "--cell",
        "gru",
        "--hidden",
        "128",
        "--layers",
        "2",
        "--dropout",
        "0",
        "--seed",
        str(seed),
        "--raw-only",
        "--loss",
        "wmse",
        "--loss-norm",
        "chunk",
        "--train-source",
        "train+valid",
        "--train-seqs",
        "10607",
        "--split-manifest",
        "research/campaign_2026-09-29/artifacts/splits/fold0.json",
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
        "16",
        "--chunk",
        "2000",
        "--lr",
        "0.001",
        "--wd",
        "0.01",
        "--optim",
        "adamw",
        "--sched",
        "onecycle",
        "--precision",
        "fp16",
        "--init",
        "",
        "--distill",
        "",
        "--distill-boot-h",
        "0",
        "--propensity",
        "",
        "--prop-input",
        "",
        "--prop-power",
        "1",
        "--prop-eps",
        "0",
        "--valid-true-mask",
        "-1",
        "--ema-teacher",
        "0",
        "--aux-struct",
        "0",
        "--aux-next",
        "0",
        "--aux-tier",
        "0",
        "--aux-mask",
        "0",
        "--aux-sign",
        "0",
        "--aux-quote",
        "0",
        "--aux",
        "",
        "--aux-weight",
        "0",
        "--swa-frac",
        "0",
        "--fast",
    ]


def build_packet(
    profile_receipt_path: Path | None = None,
    *,
    profile_receipt_sha256: str | None = None,
    terminal_output_path: Path | None = None,
    controller_review_receipt_path: Path | None = None,
    controller_review_receipt_sha256: str | None = None,
) -> dict[str, object]:
    """Materialize only after a separate controller receipt approves evidence, code and budget."""
    if (
        profile_receipt_path is None
        or profile_receipt_sha256 is None
        or terminal_output_path is None
        or controller_review_receipt_path is None
        or controller_review_receipt_sha256 is None
    ):
        raise ValueError("E510 requires actual E508 files and a separate controller review receipt")
    receipt_bytes = profile_receipt_path.read_bytes()
    profile = validate_profile_receipt(
        receipt_bytes,
        expected_sha256=profile_receipt_sha256,
    )
    terminal_output_sha256 = sha256_file(terminal_output_path)
    identities = controller_review_targets(receipt_bytes, profile_receipt_sha256)
    review_bytes = controller_review_receipt_path.read_bytes()
    review = validate_controller_review_receipt(
        review_bytes,
        expected_review_sha256=controller_review_receipt_sha256,
        profile_receipt_sha256=profile_receipt_sha256,
        terminal_output_sha256=terminal_output_sha256,
        code_identities=identities,
    )
    if PACKET_DIR.exists():
        raise FileExistsError(f"packet already exists; refusing to overwrite: {PACKET_DIR}")
    for path, expected in (
        (SPLIT_PATH, SPLIT_SHA256),
        (ASSETS_MANIFEST, ASSETS_SHA256),
        (PORTABLE_INPUTS, PORTABLE_SHA256),
        (SOURCE_FAMILIES, SOURCE_FAMILIES_SHA256),
    ):
        if sha256_bytes(path.read_bytes()) != expected:
            raise ValueError(f"reviewed input identity changed: {path}")
    source, source_hashes = source_archive()
    worker = (HERE / "packet_worker.py").read_bytes()
    kernel = kernel_source(
        source,
        source_hashes,
        worker,
        receipt_bytes,
        profile_receipt_sha256,
        SOURCE_FAMILIES.read_bytes(),
    )
    if sha256_bytes(kernel) != identities["kernel_sha256"]:
        raise RuntimeError("kernel source changed after the controller reviewed its identity")
    metadata = {
        "code_file": "kernel.py",
        "competition_sources": [],
        "dataset_sources": [
            "vladkochetov07/wnn-c20261004-fold0-assets",
            "vladkochetov07/wunder-funder",
            "vladkochetov07/wnn-jax-realdata-e463-tpu-20261004-v1",
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": "vladkochetov07/wnn-raw-teacher-pair-e510-20261007",
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "title": "wnn-raw-teacher-pair-e510-20261007",
    }
    profile_argv = [
        "python",
        "scripts/train_rnn.py",
        *training_args("e510_raw_profile_s3_f0", 3),
        "--profile-warmup",
        "1",
        "--profile-steps",
        "3",
    ]
    fits = {
        str(seed): [
            "python",
            "scripts/train_rnn.py",
            *training_args(f"bd26_samefold_s{seed}_f0", seed),
        ]
        for seed in (3, 4)
    }
    PACKET_DIR.mkdir(parents=True)
    kernel_path = PACKET_DIR / "kernel.py"
    metadata_path = PACKET_DIR / "kernel-metadata.json"
    kernel_path.write_bytes(kernel)
    metadata_bytes = (json.dumps(metadata, indent=2) + "\n").encode()
    metadata_path.write_bytes(metadata_bytes)
    receipt: dict[str, object] = {
        "packet_id": "E510_RAW_TEACHER_PAIR_V1",
        "status": "prepared_private_not_uploaded",
        "kernel_id": metadata["id"],
        "metadata": metadata,
        "packet_path": PACKET_DIR.relative_to(ROOT).as_posix(),
        "packet_sha256": {
            "kernel.py": sha256_bytes(kernel),
            "kernel-metadata.json": sha256_bytes(metadata_bytes),
            "packet_worker.py": sha256_bytes(worker),
            "source_archive.tar": sha256_bytes(source),
        },
        "external_profile_gate": {
            "packet": "E508_RAW_BASELINE_V1",
            "receipt_sha256": profile_receipt_sha256,
            "terminal_output_sha256": terminal_output_sha256,
            "controller_review_receipt_sha256": controller_review_receipt_sha256,
            "controller_review_receipt_path": str(controller_review_receipt_path),
            "approval_status": review["status"],
            "status": profile["status"],
            "fit_admitted": profile["fit_admitted"],
            "fit_stages_started": profile["fit_stages_started"],
            "split_sha256": profile["split_sha256"],
            "profile": profile["profile"],
            "only_after_controller_review": True,
        },
        "controller_reviewed_code_identities": identities,
        "trainer": {
            "commit": TRAINER_COMMIT,
            "source_sha256": source_hashes["scripts/train_rnn.py"],
            "source_files_sha256": source_hashes,
            "archive_bytes": len(source),
        },
        "frozen_recipe": {
            "runs": ["bd26_samefold_s3_f0", "bd26_samefold_s4_f0"],
            "arch": "scratch GRU, hidden=128, layers=2, dropout=0, raw-only event block",
            "seeds": [3, 4],
            "fit_epochs": 4,
            "checkpoint": {"policy": "fixed", "zero_based_epoch": 3},
            "objective": "wMSE with clipped labels and original clipped-magnitude weights",
            "optimizer": "AdamW / OneCycle, lr=0.001, wd=0.01",
            "batch": 16,
            "chunk_rows": 2000,
            "fast": True,
            "evaluation_during_fit": False,
            "disabled": [
                "initializer",
                "distillation",
                "EMA/SWA",
                "propensity",
                "mask weighting",
                "auxiliary losses",
            ],
        },
        "fold0": {
            "manifest_path": SPLIT_PATH.relative_to(ROOT).as_posix(),
            "manifest_sha256": SPLIT_SHA256,
            "fit_train_groups": 10535,
            "fit_valid_groups": 1405,
            "evaluation_valid_groups_excluded": 468,
            "corrupt_train_group_excluded": 528,
            "legacy_train_seqs_ceiling": 10607,
        },
        "outputs": {
            "checkpoint_epoch_zero_based": 3,
            "run_provenance_and_checkpoint_sha256": True,
            "feature_only_prediction_exports": {
                "train_rows": 10606,
                "valid_rows": 1873,
                "shape": "(target-cache rows, 20000, 2)",
                "dtype": "float32",
                "row_order": "cache/{split}_targets.npz row_group order",
            },
            "realized_source_snapshot": "e510_source_families_realized.json",
            "baseline_status": "produced_by_e510",
            "snapshot_sha256_pinned_by_controller_before_cache_build": True,
        },
        "commands": {
            "profile_argv": profile_argv,
            "fit_argv_by_seed": fits,
            "kaggle_push_command_prepared_not_run": f"./.venv/bin/kaggle kernels push -p {PACKET_DIR.relative_to(ROOT).as_posix()} -t 7200",
            "git_push_run": False,
            "kaggle_push_run": False,
        },
        "guardrails": {
            "separate_controller_receipt_binds_e508_profile_and_terminal_output": True,
            "controller_receipt_binds_exact_e510_kernel_worker_and_builder_hashes": True,
            "controller_receipt_approves_fresh_kaggle_quota_and_production_budget": True,
            "e508_profile_receipt_hash_gate_before_any_fit": True,
            "profile_status_must_be_reviewed_no_fit": True,
            "internal_profile_then_serial_pair": True,
            "worker_process_group_timeout_seconds": 6300,
            "provider_watch_seconds": 7200,
            "automatic_oom_retry": False,
            "durable_phase_logs_and_finally_receipt": True,
            "recursive_input_discovery_by_exact_identity": True,
            "actual_split_group_membership_asserted": True,
            "evaluation_labels_read": False,
            "controller_must_review_packet_before_push": True,
        },
    }
    (PACKET_DIR / "review_packet.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (HERE / "README.md").write_text(readme(receipt))
    return receipt


def readme(receipt: dict[str, object]) -> str:
    """Write an operator handoff that keeps approval and provider push separate."""
    profile_gate = receipt["external_profile_gate"]
    commands = receipt["commands"]
    return f"""# E510 raw teacher pair

This private packet was prepared from the frozen trainer at `{TRAINER_COMMIT}`.
It has **not** been pushed or run. A separate controller receipt approves E508's
profile result and terminal output and binds their hashes to this exact E510
kernel, worker, and builder. The builder refuses packet creation unless that
receipt also approves a fresh Kaggle quota check and the frozen production
budget. At runtime the worker rechecks E508 profile bytes, status, fold
membership, timing, and allocator peak before input discovery. Its provider watch
is 7,200 seconds and its worker process-group deadline is 6,300 seconds. The two
teachers run serially on `cuda:0` after a short in-kernel profile passes.

The pair is an independent same-fold raw-label baseline: fresh raw-only GRU(128,
two layers), seeds 3/4, four epochs, fixed zero-based epoch 3, wMSE with the
trainer's original clipped-label-magnitude weights, AdamW/OneCycle at lr 0.001,
batch 16, chunk 2,000, and `--fast`. Initialization, distillation, propensity,
EMA/SWA, mask weighting, and auxiliary losses are disabled. Fits use exactly
10,535 train and 1,405 validation groups, excluding the 468 evaluation groups
and corrupt group 528. The legacy `--train-seqs 10607` argument remains only a
ceiling; the fold manifest provides actual membership.

The worker recursively discovers mounted inputs by exact manifest/hash and byte
identity, checks Parquet footers, and symlinks raw data without copying it. After
both fits it exports train/valid float32 predictions in full cache row order using
feature-only projections, then writes `e510_source_families_realized.json` with
the two checkpoint and provenance hashes plus the sealed six existing teacher
identities. The controller must pin that snapshot's resulting hash before building
or fitting any C1-C4 cache/recipe.

Prepared command (not executed):

```sh
{commands["kaggle_push_command_prepared_not_run"]}
```

Kernel SHA-256: `{receipt["packet_sha256"]["kernel.py"]}`<br>
Trainer source SHA-256: `{TRAINER_SHA256}`<br>
E508 profile SHA-256: `{profile_gate["receipt_sha256"]}`<br>
E508 terminal output SHA-256: `{profile_gate["terminal_output_sha256"]}`<br>
Controller review receipt SHA-256: `{profile_gate["controller_review_receipt_sha256"]}`<br>
SOURCE_FAMILIES SHA-256: `{SOURCE_FAMILIES_SHA256}`
"""


def main() -> None:
    """Preview code hashes or build only with an independent controller review."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--e508-profile-receipt", type=Path, required=True)
    parser.add_argument("--e508-profile-receipt-sha256", required=True)
    parser.add_argument("--review-targets-only", action="store_true")
    parser.add_argument("--e508-terminal-output", type=Path)
    parser.add_argument("--controller-review-receipt", type=Path)
    parser.add_argument("--controller-review-receipt-sha256")
    args = parser.parse_args()
    if args.review_targets_only:
        profile_bytes = args.e508_profile_receipt.read_bytes()
        print(
            json.dumps(
                controller_review_targets(profile_bytes, args.e508_profile_receipt_sha256), indent=2
            )
        )
        return
    if (
        args.e508_terminal_output is None
        or args.controller_review_receipt is None
        or args.controller_review_receipt_sha256 is None
    ):
        parser.error(
            "packet creation requires E508 terminal output and separate controller approval"
        )
    print(
        json.dumps(
            build_packet(
                args.e508_profile_receipt,
                profile_receipt_sha256=args.e508_profile_receipt_sha256,
                terminal_output_path=args.e508_terminal_output,
                controller_review_receipt_path=args.controller_review_receipt,
                controller_review_receipt_sha256=args.controller_review_receipt_sha256,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
