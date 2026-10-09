"""Assemble the bounded V5R2 fit packet and its private Kaggle source dataset."""

from __future__ import annotations

import hashlib
import json
import shutil
import textwrap
from pathlib import Path

from build_fit_archive import _archive_source

PACKET_ID = "E529_H232_GRU512_768_MUON_SEED9_FIT_V5R2_20261009"
SOURCE_DATASET = "vladkochetov07/e529-gru512-768-muon-fit-v5r2-source-20261009"
KERNEL_SLUG = "e529-gru512-768-muon-fit-v5r2-20261009"
KERNEL_CAP_SECONDS = 26_000
PACKET_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
    "/gru512_768_muon_fit_v5r2_20261009/packet"
)
EVIDENCE_FILES = (
    "cohort_terminal_receipt.json",
    "input_identity_receipt.json",
    "muon512-profile-argv.json",
    "muon512-profile-digest-metrics.json",
    "muon512-profile-gpu-samples.json",
    "muon512-profile.log",
    "muon768-profile-argv.json",
    "muon768-profile-digest-metrics.json",
    "muon768-profile-gpu-samples.json",
    "muon768-profile.log",
    "provider_kernel_log.log",
    "root_posthoc_profile_review.json",
    "terminal_collection_review_request.json",
)
CPU_THREADS_PER_ARM = 2
AGGREGATE_CPU_THREADS = 4


def sha256_file(path: Path) -> str:
    """Hash a regular file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    """Write deterministic, reviewable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _profile_manifest(packet_root: Path) -> dict[str, object]:
    evidence_root = packet_root / "producer_evidence/v4"
    files: dict[str, dict[str, object]] = {}
    for name in EVIDENCE_FILES:
        path = evidence_root / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"required V4 profile evidence is missing: {name}")
        relative = f"producer_evidence/v4/{name}"
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    review_path = evidence_root / "root_posthoc_profile_review.json"
    manifest = {
        "schema": "wnn-gru-capacity-v4-profile-evidence-v1",
        "profile_reuse_only": True,
        "fit_authorized_by_v4": False,
        "root_review_sha256": sha256_file(review_path),
        "files": files,
    }
    write_json(evidence_root / "manifest.json", manifest)
    return manifest


def _recipe(packet_root: Path, profile_manifest: dict[str, object]) -> dict[str, object]:
    source_recipe = json.loads(
        (packet_root / "GRU512_768_MUON_PROFILE_RECIPE_V4.json").read_text(encoding="utf-8")
    )
    recipe = source_recipe
    recipe["schema"] = "wnn-gru-optimizer-capacity-fit-recipe-v1"
    recipe["packet_id"] = PACKET_ID
    recipe["parent_packet"] = "E529_H232_GRU512_768_MUON_SEED9_FIT_V5R1_20261008"
    recipe["profile_evidence_parent_packet"] = "E529_H232_GRU512_768_MUON_SEED9_PROFILE_V4_20261008"
    recipe["scientific_question"] = (
        "At fixed seed, clean-fold membership, B8 objective and four-epoch budget, "
        "does the independently implemented Muon recurrent update support a useful "
        "GRU width ladder from 512 to 768?"
    )
    recipe["arms"] = [arm for arm in recipe["arms"] if arm["id"] in {"muon512", "muon768"}]
    recipe["comparison_scope"] = {
        "claim_scope": "bounded clean-fold fit screen; not a public-score forecast or submission claim",
        "primary_pair": ["muon512", "muon768"],
        "pairing": "same seed-9 initialization protocol, ordered clean-fold fit pools, seeded epoch order, B8, loss, clipping, schedule, precision and four epochs; independent fresh model and optimizer per process",
        "profile_reuse": "reuse only the successful actual V4 full-shape B8 Muon profiles; do not rerun profiles",
        "all_data_teacher_family": "not tested by this clean-fold capacity pair",
    }
    recipe["authorization"] = {
        "source_review": "pending independent root review",
        "fit": "disabled until an exact root conditional-fit receipt is embedded",
        "provider_push": "pending root packet review and fresh quota admission",
        "grading": "root-owned; only the frozen 468-group evaluation population is exported",
    }
    recipe["status"] = "fixed-b8-fit-only-source-prepared-pending-root-review-and-conditional-authorization"
    recipe["training"]["batch_selection_rule"] = (
        "Fixed B8 for both fit arms, selected by the two completed actual V4 B8 profiles. "
        "V5R2 performs no profile rerun and has no batch or single-arm fallback."
    )
    recipe["training"]["initial_batch"] = 8
    recipe["training"]["fallback_batch"] = None
    recipe["profile"]["profile_reused_from_parent"] = True
    recipe["profile"]["profile_rerun_in_this_packet"] = False
    recipe["profile"]["precision_note"] = (
        "The trainer command used FP16 autocast with GradScaler init scale 4096. "
        "OPTIMIZER_COMPUTE_PRECISION.selected_dtype records Muon orthogonalization "
        "dtype separately; the measured T4 policy selected FP32 orthogonalization."
    )
    recipe["profile"]["required_evidence"] = [
        "actual V4 PROFILE and PROFILE_UPDATE_EVIDENCE from the exact archived trainer",
        "230 attempts, 230 completed updates, 0 AMP skips and 230 loss checks per required arm",
        "actual V4 argv records bind FP16 autocast, GradScaler scale 4096, B8, 3 warmup and 20 timed batches",
        "finite profile losses and gradients with GradScaler enabled",
        "Muon orthogonalization precision metadata matches the declared hardware policy; it is distinct from model autocast precision",
        "stable one-device T4 identity, positive finite timing, measured GPU memory and host headroom",
    ]
    recipe["fit_and_export_gate"] = {
        "fit_arms": ["muon512", "muon768"],
        "fit": "fresh seed-9 initialization; exactly four epochs; fixed epoch-3 checkpoint; no profile rerun; B8 has 1492 outer batches per epoch and ten optimizer updates per batch",
        "strict_fit_contract": "exact per-epoch attempt and batch counts, zero AMP skips, finite checkpoint parameters, same frozen split/membership/sampler receipts and exact command arguments in both arms",
        "prediction_export": "FP32 feature-only inference on exactly the frozen 468 clean-fold evaluation groups; no labels or scoring masks are read; no score is computed in the worker",
        "artifact_delivery": "checkpoint, fit/provenance receipts, prediction array and sidecar are required per arm; each arm is independently preserved if its pair mate fails",
    }
    recipe["profile_projection"] = {
        "basis": "actual V4 full-shape PROFILE median outer-sequence-batch wall time measured with 4 CPU threads per arm; B8 has 1492 batches per epoch, ten chunk updates per batch, four epochs, and 1.20 safety multiplier. V5R2 shares the observed 4-CPU host across two workers at 2 threads each, so this is a projection reference, not a timing guarantee.",
        "fit_seconds": {"muon512": 13213.224, "muon768": 20493.420},
        "reserves_per_arm_seconds": {"startup": 60, "checkpoint_prediction_export": 420, "runner_finalization": 300},
        "max_kernel_seconds": KERNEL_CAP_SECONDS,
        "max_arm_seconds": {"muon512": 15500, "muon768": 22500},
        "source": "V4 root-reviewed profile evidence manifest and logs, preserved unchanged under producer_evidence/v4",
        "admission": "require actual two-T4 and host 4-CPU/16-GiB simultaneous-worker preflight for the pair; each worker is allocated 2 CPU threads, using the observed dual-T4 host's full 4-CPU affinity",
    }
    recipe["runtime"] = {
        **recipe["runtime"],
        "cpu_threads": CPU_THREADS_PER_ARM,
        "aggregate_cpu_threads": AGGREGATE_CPU_THREADS,
        "cpu_allocation_change": "V4 profile timings used 4 CPU threads per arm; this fit-only successor uses 2 threads per arm so both independently bound T4 workers fit the measured 4-CPU Kaggle host. Fit wall projections are not guaranteed under this shared CPU allocation.",
        "initial_execution_scope": "fixed-B8 fit-only; reuse reviewed V4 profiles, with no profile rerun",
        "provider": "Kaggle with actual Tesla T4 device identities measured at startup",
        "physical_devices": "one UUID-bound physical T4 per independent arm process",
        "provider_timeout_seconds": KERNEL_CAP_SECONDS,
        "worker_wall_cap_seconds": KERNEL_CAP_SECONDS,
        "max_arm_deadline_seconds": recipe["profile_projection"]["max_arm_seconds"],
        "root_fit_authorization": "required and hash-bound; no implicit fallback or profile-only authorization",
        "requested_kernel": f"vladkochetov07/{KERNEL_SLUG}",
        "requested_source_dataset": SOURCE_DATASET,
    }
    recipe["profile_evidence"] = {
        "manifest_path": "producer_evidence/v4/manifest.json",
        "manifest_sha256": sha256_file(packet_root / "producer_evidence/v4/manifest.json"),
        "profile_status": "both actual B8 profiles completed 230 updates with zero skips; V4 failed only in its post-profile consumer and did not fit either arm",
        "profile_values": {
            arm: {
                "outer_batch_median_ms": values["step_wall_ms"],
                "completed_updates": values["updates"],
                "amp_skips": values["skips"],
                "model_precision": "fp16_autocast",
                "amp_init_scale": 4096.0,
                "grad_scaler_enabled": True,
                "orthogonalization_policy": "native_bf16_if_supported_else_fp32_v1",
                "orthogonalization_dtype": "float32",
                "native_bf16_selected": False,
                "observed_peak_gpu_used_mb": values["peak_gpu_used_mb"],
            }
            for arm, values in {
                "muon512": {"step_wall_ms": 1845.01, "updates": 230, "skips": 0, "peak_gpu_used_mb": 1039},
                "muon768": {"step_wall_ms": 2861.57, "updates": 230, "skips": 0, "peak_gpu_used_mb": 1663},
            }.items()
        },
    }
    recipe["source"] = {
        "parent_source_archive_sha256": source_recipe["source"]["source_archive"]["sha256"],
        "source_archive": {
            "path": "source/source_archive.tar",
            "bytes": (packet_root / "source/source_archive.tar").stat().st_size,
            "sha256": sha256_file(packet_root / "source/source_archive.tar"),
        },
        "source_manifest": {
            "path": "source/source_archive_manifest.json",
            "bytes": (packet_root / "source/source_archive_manifest.json").stat().st_size,
            "sha256": sha256_file(packet_root / "source/source_archive_manifest.json"),
        },
        "parent_source_commit": "ead2eb7ce4108bd1ae4fa1fd04b2d5469cfe656c",
        "entry_modules": [
            "capacity_fit_worker.py",
            "capacity_arm_worker.py",
            "capacity_worker.py",
            "train_entry.py",
            "verified_digest_handoff.py",
        ],
        "build_scope": "own-project frozen V4 archive plus reviewed V5 launch fixes and the identified V5R2 two-thread-per-arm allocation; no mutable-main trainer source or teammates' code/weights",
    }
    recipe["transport"] = {
        "dataset": SOURCE_DATASET,
        "dataset_manifest": "source_dataset_upload/assets-manifest.json",
        "kernel_entrypoint": "kernel_upload/kernel.py",
        "raw_data_copy": False,
        "mounted_raw_bytes": recipe["inputs"]["source_members"]["datasets/train.parquet"]["bytes"]
        + recipe["inputs"]["source_members"]["datasets/valid.parquet"]["bytes"],
        "mounted_cache_bytes": sum(
            recipe["inputs"]["source_members"][name]["bytes"]
            for name in INPUT_CACHE_NAMES
        ),
        "shared_loader": False,
        "shared_identity_passes": 1,
    }
    return recipe


INPUT_CACHE_NAMES = (
    "cache/chain_folds.npz",
    "cache/chains.npz",
    "cache/train_targets.npz",
    "cache/valid_targets.npz",
)


def _outer_kernel_source(dataset_manifest_sha256: str, files: list[str]) -> str:
    required = tuple(sorted(files))
    required_repr = repr(required)
    return textwrap.dedent(
        f'''\
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

        SOURCE_DATASET = {SOURCE_DATASET!r}
        EXPECTED_MANIFEST_SHA256 = {dataset_manifest_sha256!r}
        REQUIRED_FILES = {required_repr}
        PACKET_ID = {PACKET_ID!r}
        WORKER_CAP_SECONDS = {KERNEL_CAP_SECONDS}

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
            by_path = {{}}
            for entry in entries:
                logical = str(entry.get("path", ""))
                relative = PurePosixPath(logical)
                if relative.is_absolute() or not relative.parts or ".." in relative.parts or "\\\\" in logical:
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
            path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\\n", encoding="utf-8")

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
            worker_log = Path("/kaggle/working/gru_capacity_v5r2_worker.log")
            terminal_path = Path("/kaggle/working/gru_capacity_v5r2_terminal.json")
            deadline = started + WORKER_CAP_SECONDS
            command = [sys.executable, str(packet_root / "capacity_fit_worker.py"), "--packet-root", str(packet_root)]
            prefixes = (
                "GRU_CAPACITY_PHASE ",
                "GRU_CAPACITY_DEVICE ",
                "GRU_CAPACITY_TERMINAL ",
                '{{"event": "campaign_resource_heartbeat"',
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
                            while b"\\n" in buffer:
                                raw, _, tail = buffer.partition(b"\\n")
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
            worker_receipt = {{}}
            if terminal_path.is_file():
                worker_receipt = json.loads(terminal_path.read_text(encoding="utf-8"))
            result = {{
                "schema": "wnn-gru-capacity-v5r2-driver-terminal-v1",
                "packet_id": PACKET_ID,
                "status": status,
                "worker_exit_code": exit_code,
                "worker_receipt_status": worker_receipt.get("status"),
                "worker_seconds": round(time.monotonic() - started, 3),
                "worker_log": str(worker_log),
                "fit_started": worker_receipt.get("fit_started", False),
            }}
            write_json(Path("/kaggle/working/gru_capacity_v5r2_driver_terminal.json"), result)
            print("GRU_CAPACITY_DRIVER_TERMINAL " + json.dumps(result, sort_keys=True), flush=True)
            return result

        def main() -> None:
            started = time.monotonic()
            os.environ["PYTHONUNBUFFERED"] = "1"
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["WNN_KERNEL_DEADLINE_MONOTONIC"] = repr(started + WORKER_CAP_SECONDS)
            print("GRU_CAPACITY_PHASE " + json.dumps({{"phase":"bootstrap","status":"running","arm":"cohort","elapsed_s":0.0}}), flush=True)
            packet_root = Path("/kaggle/working/gru_capacity_v5r2_packet")
            try:
                materialize(Path("/kaggle/input"), packet_root)
                print("GRU_CAPACITY_PHASE " + json.dumps({{"phase":"bootstrap","status":"pass","arm":"cohort","elapsed_s":round(time.monotonic()-started,1)}}), flush=True)
                receipt = run_worker(packet_root, started)
            except BaseException as error:
                failure = {{"schema":"wnn-gru-capacity-v5r2-driver-terminal-v1","packet_id":PACKET_ID,"status":"bootstrap_error","error_type":type(error).__name__,"worker_seconds":round(time.monotonic()-started,3),"fit_started":False}}
                write_json(Path("/kaggle/working/gru_capacity_v5r2_driver_terminal.json"), failure)
                print("GRU_CAPACITY_DRIVER_TERMINAL " + json.dumps(failure, sort_keys=True), flush=True)
                raise
            if receipt.get("status") != "complete" or receipt.get("worker_receipt_status") not in {{"source_review_pending","fit_authorization_pending","fit_complete"}}:
                raise RuntimeError("V5R2 fit worker did not reach a declared terminal status")

        if __name__ == "__main__":
            main()
        '''
    )


def build(packet_root: Path, repo_root: Path) -> dict[str, object]:
    """Build V5R2 content-addressed source assets and a pending, fail-closed recipe."""
    _archive_source(packet_root, repo_root)
    profile_manifest = _profile_manifest(packet_root)
    write_json(packet_root / "ROOT_FIT_SOURCE_REVIEW.json", {
        "schema": "wnn-gru-capacity-fit-source-review-v1",
        "packet_id": PACKET_ID,
        "status": "pending",
        "independent_review_required": True,
    })
    write_json(packet_root / "ROOT_FIT_AUTHORIZATION.json", {
        "schema": "wnn-gru-capacity-fit-authorization-v1",
        "packet_id": PACKET_ID,
        "fit_authorized": False,
        "fit_arm_ids": [],
        "selected_batch": 8,
        "max_worker_seconds": 0,
        "profile_projection_seconds": {},
        "root_source_review_sha256": "",
        "recipe_sha256": "",
        "source_archive_sha256": "",
        "source_manifest_sha256": "",
        "profile_evidence_manifest_sha256": "",
        "root_review_sha256": "",
    })
    recipe = _recipe(packet_root, profile_manifest)
    recipe_path = packet_root / "GRU512_768_MUON_FIT_RECIPE_V5R2.json"
    write_json(recipe_path, recipe)

    internal_kernel = packet_root / "kernel.py"
    internal_kernel.write_text(
        "from __future__ import annotations\n\n"
        "import os\nimport time\nfrom pathlib import Path\n\n"
        "from capacity_fit_worker import run\n\n"
        "if __name__ == '__main__':\n"
        "    os.environ.setdefault('WNN_KERNEL_DEADLINE_MONOTONIC', repr(time.monotonic() + 26000))\n"
        "    run(Path(__file__).resolve().parent)\n",
        encoding="utf-8",
    )
    metadata = {
        "code_file": "kernel.py",
        "competition_sources": [],
        "dataset_sources": [SOURCE_DATASET, "vladkochetov07/wunder-funder", "vladkochetov07/wnn-c20261004-fold0-assets"],
        "enable_gpu": True,
        "enable_internet": False,
        "id": f"vladkochetov07/{KERNEL_SLUG}",
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "title": KERNEL_SLUG,
    }
    write_json(packet_root / "kernel-metadata.json", metadata)
    write_json(packet_root / "GRU512_768_MUON_FIT_RECIPE_V5R2.json", recipe)

    recipe["source"]["file_bindings"] = {}
    binding_paths = [
        "capacity_worker.py",
        "capacity_fit_worker.py",
        "capacity_arm_worker.py",
        "train_entry.py",
        "verified_digest_handoff.py",
        "kernel.py",
        "kernel-metadata.json",
        "source/source_archive.tar",
        "source/source_archive_manifest.json",
        "fold0.json",
        "portable_inputs.json",
        "prior_input_identity_v2.json",
        "producer_evidence/v4/manifest.json",
    ]
    recipe["source"]["file_bindings"] = {
        name: {"bytes": (packet_root / name).stat().st_size, "sha256": sha256_file(packet_root / name)}
        for name in binding_paths
    }
    recipe["profile_evidence"]["manifest_sha256"] = sha256_file(
        packet_root / "producer_evidence/v4/manifest.json"
    )
    write_json(recipe_path, recipe)

    readme = """# GRU512/768 Muon capacity fit V5R2\n\nThis identified V5R2 successor preserves the V5 scientific recipe and reuses two actual V4 B8 profiles; it performs no profile rerun. It fits the frozen seed-9, clean-fold, FP16 (GradScaler scale 4096) Muon GRU512/768 pair for four epochs with fixed epoch 3, then exports FP32 feature-only predictions for exactly 468 frozen evaluation groups. The worker computes no score. V4 profiles completed 230 updates each with zero skips; V4 itself did not fit because its precision metadata consumer read the wrong JSON nesting.\n\nTwo simultaneous workers are required. The measured dual-T4 Kaggle host exposes four CPUs and 30 GiB RAM. Each independently initialized arm is allocated two CPU threads, for four aggregate, and receives a distinct physical Tesla T4 UUID. Startup checks that the visible CUDA device is the assigned T4 and reports that device's memory. The preflight verifies actual host CPU/RAM and distinct GPUs before raw/cache identity work. The V4 wall-time projections were measured at four CPU threads per arm; under this two-thread-per-arm shared-host allocation they are reference estimates, not timing guarantees.\n\nMounted input bytes are never copied into the worker directory. The worker verifies the six exact mounted inputs once, reusing the V4 content digest only when the exact path/stat identity matches; each arm rechecks the frozen stat and symlink binding. The loader is not shared; the host may reuse ordinary OS page cache.\n\nThis is a clean-fold capacity screen, not a public forecast and not the missing all-data heterogeneous teacher-family replication. No fit is authorized until root writes an independent review and exact conditional authorization into this packet.\n"""
    (packet_root / "README.md").write_text(readme, encoding="utf-8")
    recipe["source"]["file_bindings"]["README.md"] = {
        "bytes": (packet_root / "README.md").stat().st_size,
        "sha256": sha256_file(packet_root / "README.md"),
    }
    write_json(recipe_path, recipe)

    upload_root = packet_root / "source_dataset_upload"
    if upload_root.exists():
        shutil.rmtree(upload_root)
    upload_root.mkdir(parents=True)
    payload_files = [
        "capacity_worker.py", "capacity_fit_worker.py", "capacity_arm_worker.py",
        "train_entry.py", "verified_digest_handoff.py", "kernel.py", "kernel-metadata.json",
        "README.md", "GRU512_768_MUON_FIT_RECIPE_V5R2.json", "ROOT_FIT_SOURCE_REVIEW.json",
        "ROOT_FIT_AUTHORIZATION.json", "source/source_archive.tar", "source/source_archive_manifest.json",
        "fold0.json", "portable_inputs.json", "prior_input_identity_v2.json",
        "producer_evidence/v4/manifest.json",
        *[f"producer_evidence/v4/{name}" for name in EVIDENCE_FILES],
    ]
    descriptors = []
    for relative in sorted(set(payload_files)):
        source = packet_root / relative
        digest = sha256_file(source)
        destination = upload_root / f"{digest}.wnn"
        shutil.copyfile(source, destination)
        descriptors.append({"path": relative, "size": source.stat().st_size, "sha256": digest, "filename": destination.name})
    manifest = {"version": 1, "dataset": SOURCE_DATASET, "private": True, "files": descriptors}
    manifest_path = upload_root / "assets-manifest.json"
    write_json(manifest_path, manifest)
    write_json(upload_root / "dataset-metadata.json", {
        "id": SOURCE_DATASET,
        "title": SOURCE_DATASET.split("/", 1)[1],
        "isPrivate": True,
        "licenses": [{"name": "other"}],
    })
    outer_driver = _outer_kernel_source(sha256_file(manifest_path), [row["path"] for row in descriptors])
    outer_root = packet_root / "kernel_upload"
    outer_root.mkdir(exist_ok=True)
    (outer_root / "kernel.py").write_text(outer_driver, encoding="utf-8")
    shutil.copyfile(packet_root / "kernel-metadata.json", outer_root / "kernel-metadata.json")
    driver_sha = sha256_file(outer_root / "kernel.py")
    receipt = {
        "schema": "wnn-gru-capacity-fit-v5r2-build-receipt-v1",
        "packet_id": PACKET_ID,
        "status": "pending_root_source_review_and_fit_authorization",
        "parent_packet": "E529_H232_GRU512_768_MUON_SEED9_FIT_V5R1_20261008",
        "source_dataset": SOURCE_DATASET,
        "source_dataset_manifest_sha256": sha256_file(manifest_path),
        "source_dataset_files": len(descriptors),
        "kernel_slug": metadata["id"],
        "kernel_title": metadata["title"],
        "kernel_driver_sha256": driver_sha,
        "recipe_sha256": sha256_file(recipe_path),
        "source_archive_sha256": sha256_file(packet_root / "source/source_archive.tar"),
        "source_manifest_sha256": sha256_file(packet_root / "source/source_archive_manifest.json"),
        "profile_evidence_manifest_sha256": sha256_file(packet_root / "producer_evidence/v4/manifest.json"),
        "fit_started": False,
        "profile_rerun": False,
        "provider_upload_or_push": False,
    }
    write_json(packet_root / "PACKET_BUILD_RECEIPT.json", receipt)
    return receipt


def main() -> None:
    """Build this exact packet from the frozen V4 source snapshot."""
    script_path = Path(__file__).resolve()
    repo_root = next(parent for parent in script_path.parents if (parent / "AGENTS.md").is_file())
    packet_root = repo_root / PACKET_RELATIVE
    build(packet_root, repo_root)


if __name__ == "__main__":
    main()
