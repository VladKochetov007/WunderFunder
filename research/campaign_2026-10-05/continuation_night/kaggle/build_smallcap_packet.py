from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
WIDTH = HERE.parent / "width"
PACKET_SLUG = "wnn-night-smallcap-w256w384-20261006-v1"
PACKET_DIR = HERE / "packets" / PACKET_SLUG
SOURCE_PAYLOAD = ROOT / "research/campaign_2026-10-05/w384_v7/bundle/kernel/kernel.py"
SOURCE_MANIFEST = ROOT / "research/campaign_2026-10-05/w384_v7/bundle/bundle-manifest.json"
SOURCE_ROOT = "research/campaign_2026-10-05/w384_v7/bundle"
V3R1_REVIEW = (
    "research/campaign_2026-10-05/continuation_night/kaggle/packets/"
    "wnn-night-w384-serial-20261006-v3r1/review_packet.json"
)
V3R1_CRITIC = "research/campaign_2026-10-05/continuation_night/admission/W384_V3R1_INDEPENDENT_REVIEW_GO.md"
RECOVERY_ATTEMPT = 2
RECOVERY_LIMIT = 2
GPU_ALLOCATOR_BYTES = 2_500_000_000
GPU_STOP_BYTES = 2_800_000_000
GPU_HARD_LIMIT_BYTES = 3_000_000_000
CPU_BUDGET = 4
WORKER_THREADS = 3
RAM_PROCESS_STOP_BYTES = 10 * 1024**3
PAYLOAD_TIMEOUT_SECONDS = 31_800
WATCHDOG_TIMEOUT_SECONDS = 31_500
PROVIDER_TIMEOUT_SECONDS = 32_100


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_path(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"expected one {label} patch point; observed {source.count(old)}")
    return source.replace(old, new, 1)


def load_previous_builder():
    path = HERE / "build_packets.py"
    spec = importlib.util.spec_from_file_location("continuation_night_packet_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the reviewed packet source helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pair_payload(previous_builder) -> bytes:
    """Reuse the original pinned private bundle with a pair runner entry point."""
    payload = previous_builder.gpu_payload(256).decode()
    payload = replace_once(
        payload,
        'command = [sys.executable, "/kaggle/working/continuation_runtime/phase.py", "--width", "256", "--manifest", str(matches[0])]',
        'command = [sys.executable, "/kaggle/working/continuation_runtime/smallcap_phase.py", "--manifest", str(matches[0])]',
        "pair phase entry",
    )
    payload = replace_once(
        payload,
        "timeout_seconds=17_900,",
        f"timeout_seconds={WATCHDOG_TIMEOUT_SECONDS:,},",
        "whole-kernel watchdog",
    )
    ast.parse(payload)
    return payload.encode()


def entry_source(previous_builder, sources: dict[str, bytes]) -> bytes:
    """Pin source hashes and reserve four CPU cores including telemetry overhead."""
    hashes = {name: sha(data) for name, data in sources.items()}
    encoded = {
        name: base64.b85encode(zlib.compress(data, level=9)).decode()
        for name, data in sources.items()
    }
    constants = (
        f"PACKET = {PACKET_SLUG!r}\n"
        f"SOURCES = {encoded!r}\n"
        f"HASHES = {hashes!r}\n"
        f"CPU_COUNT = {CPU_BUDGET}\n"
        f"WORKER_THREADS = {WORKER_THREADS}\n"
        f"TIMEOUT = {PAYLOAD_TIMEOUT_SECONDS}\n"
        f"RSS_BYTES = {RAM_PROCESS_STOP_BYTES}\n"
        f"CLEANUP_LINKS = {[]!r}\n"
    )
    entry = previous_builder.ENTRY.replace("import base64\n", constants + "\nimport base64\n", 1)
    entry = replace_once(
        entry,
        "if len(allowed) < CPU_COUNT:\n    raise RuntimeError(\"insufficient CPU affinity\")\nos.sched_setaffinity(0, allowed[:CPU_COUNT])",
        "if not allowed:\n    raise RuntimeError(\"no CPU affinity is available\")\nactive_cpu_count = min(CPU_COUNT, len(allowed))\nos.sched_setaffinity(0, allowed[:active_cpu_count])",
        "aggregate CPU affinity",
    )
    entry = replace_once(
        entry,
        "environment[key] = str(CPU_COUNT)",
        "environment[key] = str(min(WORKER_THREADS, active_cpu_count))",
        "trainer thread-pool limit",
    )
    entry = replace_once(
        entry,
        'for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):',
        'for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "WNN_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS", "ARROW_NUM_THREADS"):',
        "all worker thread-pool caps",
    )
    ast.parse(entry)
    return entry.encode()


def build_packet() -> dict[str, object]:
    """Create a new private packet; never contact Kaggle or modify prior packets."""
    if PACKET_DIR.exists():
        raise FileExistsError(f"packet identity already exists: {PACKET_DIR}")
    previous_builder = load_previous_builder()
    source_files = {
        "payload.py": pair_payload(previous_builder),
        "bounded_loader.py": (WIDTH / "bounded_loader.py").read_bytes(),
        "derive_source.py": (WIDTH / "derive_source.py").read_bytes(),
        "smallcap_phase.py": (WIDTH / "smallcap_phase.py").read_bytes(),
        "smallcap_host_guard.py": (WIDTH / "smallcap_host_guard.py").read_bytes(),
    }
    hashes = {name: sha(data) for name, data in source_files.items()}
    kernel_bytes = entry_source(previous_builder, source_files)
    metadata = json.loads((ROOT / SOURCE_ROOT / "kernel/kernel-metadata.json").read_text())
    metadata.update(
        id=f"vladkochetov07/{PACKET_SLUG}",
        title=PACKET_SLUG,
        is_private=True,
        enable_gpu=True,
        enable_internet=False,
        enable_tpu=False,
        competition_sources=[],
    )
    if metadata["enable_internet"] or metadata["competition_sources"]:
        raise ValueError("small-cap packet must remain private and offline")
    if "vladkochetov07/wunder-funder" not in metadata["dataset_sources"]:
        raise ValueError("the actual user-created source data is missing")

    PACKET_DIR.mkdir(parents=True, exist_ok=False)
    metadata_bytes = (json.dumps(metadata, indent=2) + "\n").encode()
    (PACKET_DIR / "kernel.py").write_bytes(kernel_bytes)
    (PACKET_DIR / "kernel-metadata.json").write_bytes(metadata_bytes)
    receipt: dict[str, object] = {
        "status": "prepared_private_not_uploaded_pending_smallcap_critic",
        "packet_id": metadata["id"],
        "packet_path": PACKET_DIR.relative_to(ROOT).as_posix(),
        "push_command": f"./.venv/bin/kaggle kernels push -p {PACKET_DIR.relative_to(ROOT).as_posix()}",
        "kernel_sha256": sha(kernel_bytes),
        "metadata_sha256": sha(metadata_bytes),
        "embedded_source_sha256": hashes,
        "immutable_inputs": {
            "bundle_manifest": {
                "path": SOURCE_MANIFEST.relative_to(ROOT).as_posix(),
                "sha256": sha_path(SOURCE_MANIFEST),
            },
            "original_kernel_payload": {
                "path": SOURCE_PAYLOAD.relative_to(ROOT).as_posix(),
                "sha256": sha_path(SOURCE_PAYLOAD),
            },
        },
        "source_dataset_ids": metadata["dataset_sources"],
        "competition_sources": [],
        "network_access": False,
        "science_preregistration": {
            "status": "frozen_in_packet_before_any_provider_action",
            "question": "Does recurrent width 384 improve over width 256 on the matched fold-1 recipe?",
            "primary_factor": "hidden width: 256 versus 384",
            "matched_settings": {
                "seed": 3,
                "fold": 1,
                "batch_size": "common batch 16; common batch 8 only after classified B16 VRAM failure",
                "chunk": 2000,
                "epochs": 4,
                "optimizer": "adamw",
                "precision": "fp16 training; fixed epoch-3 FP32 prediction export",
                "data_source": "train+valid",
                "training_workers": 0,
                "loader_ram_gib": 1,
            },
            "preregistered_resource_changes_from_v3r1": [
                "batch size 32 to 16 for both widths; common batch 8 is the sole preregistered fallback for a classified B16 VRAM failure",
                "one visible training GPU; W256 and W384 probes and fits run serially",
            ],
            "batch_fallback_rule": "If either B16 profile stops at the aggregate VRAM gate or reports a CUDA allocation OOM, discard the B16 pair, run both W256 and W384 B8 resource probes, and fit only if both B8 probes pass. Any other B16 failure is fail-closed. No further fallback is allowed.",
            "score_comparison_rule": "new B16 baseline; no score comparison to any B32 recipe or result",
            "fit_gate": "both production-shape one-batch resource probes must pass before either fit starts",
            "evaluation": "root-owned paired full fold-1 scoring after both prediction exports exist",
        },
        "resource_policy": {
            "aggregate_cpu_budget": 5,
            "provider_cpu_affinity_cap": CPU_BUDGET,
            "trainer_thread_limit": WORKER_THREADS,
            "supervisor_watchdog_telemetry_cpu_allowance": 1,
            "root_watcher_cpu_reserve": 1,
            "local_cpu_parent_quota_during_provider_job": "100%",
            "local_training_cpu_overlap": False,
            "allocated_gpu_count": "one or two; all devices charged in aggregate monitor",
            "training_visible_gpu_count": 1,
            "aggregate_gpu_hard_limit_decimal_bytes": GPU_HARD_LIMIT_BYTES,
            "pytorch_allocator_limit_decimal_bytes": GPU_ALLOCATOR_BYTES,
            "observed_aggregate_gpu_stop_decimal_bytes": GPU_STOP_BYTES,
            "prelaunch_aggregate_gpu_use_limit_decimal_bytes": 400_000_000,
            "gpu_poll_seconds": 1,
            "worker_tree_rss_stop_bytes": RAM_PROCESS_STOP_BYTES,
            "rss_reserve_bytes": 2 * 1024**3,
            "host_target_fraction": 0.70,
            "host_stop_fraction": 0.85,
            "profile_before_fit": True,
            "failed_profile_starts_no_full_fit": True,
            "provider_guard_affinity_cpus": 1,
            "trainer_worker_affinity_cpus": 3,
            "local_gpu_must_be_idle_during_provider_gpu_work": True,
            "local_and_provider_gpu_probe_overlap": False,
        },
        "provider_timeout_seconds": PROVIDER_TIMEOUT_SECONDS,
        "whole_payload_timeout_seconds": PAYLOAD_TIMEOUT_SECONDS,
        "provider_recovery_attempt": RECOVERY_ATTEMPT,
        "maximum_provider_recoveries": RECOVERY_LIMIT,
        "supersedes_unlaunched_packet_for_attempt": "wnn-night-w384-serial-20261006-v3r1",
        "previous_v3r1_packet_sha256": sha_path(ROOT / V3R1_REVIEW),
        "previous_v3r1_critic_sha256": sha_path(ROOT / V3R1_CRITIC),
        "old_packet_files_preserved": True,
        "provider_pushes_for_this_revision": 0,
        "provider_launches_for_this_revision": 0,
        "fresh_provider_quota_and_root_reservation_required": True,
        "fresh_independent_smallcap_critic_review_required": True,
        "no_reservation_or_quota_is_granted_by_this_packet": True,
        "new_output_ids": {
            "pair": "c20261006night_smallcap_w256w384_adamw_s3_f1",
            "candidate_batch_sizes": [16, 8],
            "width_256_probe_b16": "c20261006night_smallcap_w256w384_adamw_s3_f1_b16_w256_probe",
            "width_256_fit_b16": "c20261006night_smallcap_w256w384_adamw_s3_f1_b16_w256_fit",
            "width_384_probe_b16": "c20261006night_smallcap_w256w384_adamw_s3_f1_b16_w384_probe",
            "width_384_fit_b16": "c20261006night_smallcap_w256w384_adamw_s3_f1_b16_w384_fit",
            "width_256_probe_b8": "c20261006night_smallcap_w256w384_adamw_s3_f1_b8_w256_probe",
            "width_256_fit_b8": "c20261006night_smallcap_w256w384_adamw_s3_f1_b8_w256_fit",
            "width_384_probe_b8": "c20261006night_smallcap_w256w384_adamw_s3_f1_b8_w384_probe",
            "width_384_fit_b8": "c20261006night_smallcap_w256w384_adamw_s3_f1_b8_w384_fit",
        },
        "review_gate": "fresh root reservation and independent small-cap critic GO before provider push",
    }
    review_bytes = (json.dumps(receipt, indent=2) + "\n").encode()
    (PACKET_DIR / "review_packet.json").write_bytes(review_bytes)
    receipt["review_packet_sha256"] = sha(review_bytes)
    return receipt


def main() -> None:
    print(json.dumps(build_packet(), indent=2))


if __name__ == "__main__":
    main()
