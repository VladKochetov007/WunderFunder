from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import subprocess
import tarfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PACKET_DIR = HERE / "packet"
TRAINER_COMMIT = "69b2331825e193a070015ca4182872724595209a"
TRAINER_SHA256 = "8a29459e998f7f5d6a03b7a4b87c6f4b569002e65b8c87127c37138360909cbf"
SPLIT_PATH = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
ASSETS_MANIFEST = ROOT / "research/campaign_2026-10-04/kaggle/fold0_assets/assets-manifest.json"
PORTABLE_INPUTS = ROOT / "research/campaign_2026-10-04/portable_inputs.json"
ASSETS_SHA256 = "19fe0eb6445553c2adff00e821a4fa16ca5969473ca8be420c32cb8e4b4b72c7"
PORTABLE_SHA256 = "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c"
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
SOURCE_MANIFEST_SHA256 = "851898475dfa701f2089ec76cfe9592538aa2e7f9b70ebcb68c0a614e332d6fb"
SOURCE_CHECKPOINT_SHA256 = "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e"


def sha256_bytes(value: bytes) -> str:
    """Return a stable identity for exact packet and trainer source bytes."""
    return hashlib.sha256(value).hexdigest()


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
                raise ValueError(f"unexpected non-file in pinned source archive: {member.name}")
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise ValueError(f"source archive member is unreadable: {member.name}")
            identities[member.name] = sha256_bytes(extracted.read())
    if identities.get("scripts/train_rnn.py") != TRAINER_SHA256:
        raise ValueError("pinned production trainer bytes differ from reviewed SHA-256")
    if "src/wnn/splits.py" not in identities or "src/wnn/portability.py" not in identities:
        raise ValueError("pinned source archive omits campaign split validation")
    return archive, identities


def compressed_source(source: bytes) -> str:
    """Encode compressed bytes as source-safe base85 for Kaggle's single script."""
    return base64.b85encode(zlib.compress(source, level=9)).decode("ascii")


def profile_estimate_seconds(step_wall_ms: float) -> float:
    """Estimate the frozen two-seed fit plus full exports from profile timing."""
    import math

    if not math.isfinite(step_wall_ms) or step_wall_ms <= 0:
        raise ValueError("profile step_wall must be finite and positive")
    fit_batches = 2 * (10_535 + 1_405) // 16 * 4
    export_batches = 2 * (math.ceil(10_606 / 16) + math.ceil(1_873 / 16))
    return (fit_batches + 0.75 * export_batches) * step_wall_ms / 1000.0 + 900.0


def kernel_source(
    archive: bytes,
    source_hashes: dict[str, str],
    worker: bytes,
) -> bytes:
    """Create a self-contained kernel entry that inventories inputs before training."""
    constants = (
        f"PACKET_ID = {'e508_raw_teacher_pair'!r}\n"
        f"SOURCE_ARCHIVE = {compressed_source(archive)!r}\n"
        f"SOURCE_ARCHIVE_SHA256 = {sha256_bytes(archive)!r}\n"
        f"SOURCE_HASHES = {source_hashes!r}\n"
        f"WORKER_CODE = {compressed_source(worker)!r}\n"
        f"WORKER_SHA256 = {sha256_bytes(worker)!r}\n"
        "GLOBAL_TIMEOUT_SECONDS = 3600\n"
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

RUNTIME = Path("/kaggle/working/e508_raw_teacher_pair")
PROJECT = RUNTIME / "source"
ENTRY = {"packet": PACKET_ID, "started_unix": time.time(), "pid": os.getpid()}


def write_json(path: Path, value: dict[str, object]) -> None:
    """Persist wrapper status atomically before starting any Torch work."""
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
    """Write regular files from the pinned archive without unsafe path extraction."""
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
    """Terminate native work by process group, then kill it after a fixed grace."""
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
    """Launch one profile-gated paired cohort with a parent-enforced hard timeout."""
    print(json.dumps({"event": "kernel_entry", **ENTRY}), flush=True)
    RUNTIME.mkdir(parents=True, exist_ok=False)
    write_json(RUNTIME / "wrapper_status.json", {**ENTRY, "phase": "source_unpack"})
    process: subprocess.Popen[bytes] | None = None
    try:
        archive = decode_source(SOURCE_ARCHIVE, SOURCE_ARCHIVE_SHA256)
        worker = decode_source(WORKER_CODE, WORKER_SHA256)
        PROJECT.mkdir(parents=True, exist_ok=False)
        extract_sources(archive)
        worker_path = PROJECT / "packet_worker.py"
        worker_path.write_bytes(worker)
        env = os.environ.copy()
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
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


def build_packet() -> dict[str, object]:
    """Build an immutable private packet locally; never invoke Kaggle or Git push."""
    if PACKET_DIR.exists():
        raise FileExistsError(f"packet already exists; refusing to overwrite: {PACKET_DIR}")
    for path, expected in (
        (SPLIT_PATH, SPLIT_SHA256),
        (ASSETS_MANIFEST, ASSETS_SHA256),
        (PORTABLE_INPUTS, PORTABLE_SHA256),
    ):
        if sha256_bytes(path.read_bytes()) != expected:
            raise ValueError(f"reviewed input identity changed: {path}")
    source, source_hashes = source_archive()
    worker = (HERE / "packet_worker.py").read_bytes()
    kernel = kernel_source(source, source_hashes, worker)
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
        "id": "vladkochetov07/wnn-raw-teacher-pair-e508-20261007",
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "title": "wnn-raw-teacher-pair-e508-20261007",
    }
    PACKET_DIR.mkdir(parents=True)
    kernel_path = PACKET_DIR / "kernel.py"
    metadata_path = PACKET_DIR / "kernel-metadata.json"
    kernel_path.write_bytes(kernel)
    metadata_bytes = (json.dumps(metadata, indent=2) + "\n").encode("utf-8")
    metadata_path.write_bytes(metadata_bytes)
    profile_argv = [
        "python",
        "scripts/train_rnn.py",
        *training_args("e508_raw_profile_s3_f0", 3, profile=True),
    ]
    fits = {
        str(seed): ["python", "scripts/train_rnn.py", *training_args(f"e508_raw_s{seed}_f0", seed)]
        for seed in (3, 4)
    }
    receipt: dict[str, object] = {
        "packet_id": "E508_RAW_BASELINE_V1",
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
        "trainer": {
            "commit": TRAINER_COMMIT,
            "source_sha256": source_hashes["scripts/train_rnn.py"],
            "source_files_sha256": source_hashes,
            "archive_bytes": len(source),
        },
        "frozen_recipe": {
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
                "distillation and bootstrap",
                "EMA and SWA",
                "propensity weighting/input",
                "mask weighting",
                "all auxiliary losses",
                "class balancing",
            ],
        },
        "profile_gate": {
            "seed": 3,
            "warmup_batches": 1,
            "measured_batches": 3,
            "finite_step_wall_required": True,
            "allocator_bytes": 4_300_000_000,
            "profile_kernel_wall_seconds": 3_600,
            "profile_provider_watch_seconds": 4_200,
            "followup_pair_wall_cap_seconds": 6_300,
            "followup_provider_watch_seconds": 7_200,
            "maximum_gpu_charge_if_dual_t4_profile_gpu_hours": 2.34,
            "maximum_gpu_charge_if_dual_t4_pair_gpu_hours": 4.0,
            "seed_schedule": "follow-up pair serial on cuda:0 after controller review; no concurrent allocation",
            "no_oom_retry": True,
            "fit_launch_in_this_packet": False,
            "controller_must_review_actual_profile_result": True,
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
        "input_map": {
            "discovery": "recursive /kaggle/input inventory by basename, exact size, and sealed digest; never assume slug mount path",
            "assets_manifest_sha256": ASSETS_SHA256,
            "portable_inputs_sha256": PORTABLE_SHA256,
            "raw": {
                "dataset": "vladkochetov07/wunder-funder",
                "train_bytes": 29_240_219_011,
                "valid_bytes": 5_615_509_128,
                "footer_hashes_checked": True,
                "full_hash": "only through the frozen portable registry when source metadata differs",
            },
            "audit_only_source": {
                "dataset": "vladkochetov07/wnn-jax-realdata-e463-tpu-20261004-v1",
                "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
                "checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
                "consumed_for_training": False,
            },
            "data_copy_bytes": 0,
        },
        "prediction_exports": {
            "splits": ["train", "valid"],
            "rows": {"train": 10606, "valid": 1873},
            "shape": "(target-cache rows, 20000, 2)",
            "dtype": "float32",
            "method": "feature-only parquet reads; label/mask fields excluded from projection",
            "paired_prediction_bytes_estimate": 3_993_280_000,
            "row_order": "cache/{split}_targets.npz row_group order, full row order required by cache validator",
        },
        "commands": {
            "profile_argv": profile_argv,
            "fit_argv_by_seed": fits,
            "kaggle_push_command_prepared_not_run": f"./.venv/bin/kaggle kernels push -p {PACKET_DIR.relative_to(ROOT).as_posix()} -t 4200",
            "git_push_run": False,
            "kaggle_push_run": False,
        },
        "guardrails": {
            "profile_before_full_fit": True,
            "profile_process_group_timeout_seconds": 3_600,
            "followup_fit_process_group_timeout_seconds": 6_300,
            "term_grace_seconds": 20,
            "automatic_oom_retry": False,
            "phase_status_and_logs_durable": True,
            "prediction_sha_checkpoint_sha_and_row_order_bound": True,
            "source_checkpoint_used": False,
            "evaluation_labels_read": False,
            "controller_admission_review_required": True,
            "actual_profile_review_precedes_any_fit_packet": True,
        },
        "immutable_inputs": {
            "fold0_manifest_sha256": SPLIT_SHA256,
            "fold0_assets_manifest_sha256": ASSETS_SHA256,
            "portable_inputs_sha256": PORTABLE_SHA256,
            "e463_source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "audit_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        },
    }
    (PACKET_DIR / "review_packet.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (HERE / "README.md").write_text(readme(receipt))
    return receipt


def training_args(name: str, seed: int, *, profile: bool = False) -> list[str]:
    """Mirror packet_worker.training_argv for reviewer-visible frozen commands."""
    args = [
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
    if profile:
        args.extend(("--profile-warmup", "1", "--profile-steps", "3"))
    return args


def readme(receipt: dict[str, object]) -> str:
    """Write an operator-facing handoff with status and one reviewed command."""
    packet = str(receipt["packet_path"])
    return f"""# E508 raw-only baseline teacher pair

This is a private Kaggle profile-gate packet prepared locally from the frozen
production trainer at `{TRAINER_COMMIT}`. It is reviewable but has **not** been
pushed or run. This kernel runs the short profile and then stops for controller
review; it cannot launch either scientific teacher fit.

It trains fresh raw-only GRU(128, 2-layer) teachers for seeds 3 and 4. Both use
the exact fold-0 fitting sets: 10,535 train groups and 1,405 valid groups. The
468 evaluation valid groups and corrupt train group 528 remain excluded from
training. The legacy `--train-seqs 10607` value is only a ceiling; the campaign
manifest supplies the actual members.

Each seed uses four full epochs and the predeclared fixed zero-based epoch 3
checkpoint, with wMSE, original clipped-label-magnitude weights, AdamW/OneCycle
at lr 0.001, batch 16, chunk 2,000, and `--fast`. Initialization, distillation,
propensity weighting, EMA, SWA, mask weighting, and auxiliary heads are disabled.
The trainer never reads evaluation labels during fitting.

At startup the kernel recursively inventories `/kaggle/input`; it resolves the
fold-0 asset manifest, raw parquet tables, and audit-only E463 source manifest by
content identity rather than assuming a mount directory. The raw files are
symlinked, not copied. Their predeclared Parquet footer hashes are checked; the
frozen portable registry triggers full-file checks only if source metadata differs.
The E463 checkpoint is checked against its manifest and is not used to fit this
scratch baseline.

The kernel runs one short seed-3 profile only. It requires a finite multi-step
timing, the declared T4 device, enough host/device/disk headroom, and a projected
pair/export runtime below 6,300 seconds. It emits a measured profile result and
exits with `profile_ready_awaiting_controller_review`; no scientific fit is
admitted by this packet. On a dual-T4 allocation the profile job has a 4,200-second
provider-watch budget (at most 2.34 allocated GPU-hours). A later, separate fit
packet can use at most 6,300 seconds; its 7,200-second provider watch bounds the
worst-case dual-T4 charge at 4 GPU-hours. The profile wrapper enforces a
3,600-second process-group timeout, records durable phase files and logs, and
never retries an OOM automatically. Its parent uses nonblocking byte reads, so
the wall-clock deadline does not depend on output newlines.

The frozen follow-up recipe exports train and valid predictions in the existing
full target-cache row order as float32 `(rows, 20000, 2)` arrays. Its exporter
projects feature columns only and records the fixed checkpoint hash, prediction
hash, split hash, row groups, and label/mask-read flags in sidecars. Build that
fit packet only after the controller reviews the actual E508 profile receipt.

Prepared command (not executed):

```sh
./.venv/bin/kaggle kernels push -p {packet} -t 4200
```

Packet kernel SHA-256: `{receipt["packet_sha256"]["kernel.py"]}`<br>
Trainer source SHA-256: `{TRAINER_SHA256}`<br>
Fold-0 manifest SHA-256: `{SPLIT_SHA256}`
"""


def main() -> None:
    """Materialize one immutable packet and print its review receipt."""
    print(json.dumps(build_packet(), indent=2))


if __name__ == "__main__":
    main()
