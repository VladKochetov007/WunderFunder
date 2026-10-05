"""Independently validate and score the six remote E482 T4 arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PACKET = HERE / "prepared_v1"
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
E481_DIR = ROOT / "research/campaign_2026-10-05/recurrent_activation_curvature"
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E470_DIR))
sys.path.insert(0, str(E474_DIR))
sys.path.insert(0, str(E481_DIR))

import fit_activation_optimizer as e470  # noqa: E402
import score_recurrent_curvature_v2 as e481grade  # noqa: E402

from wnn.gmetric import METRIC_ID  # noqa: E402

KERNEL_REF = "vladkochetov07/wnn-e482-t4-sixfit-20261005-v2-runner"
DATASET_SLUG = "vladkochetov07/e482-t4-sixfit-20261005-v2"
INPUT_DATASET = "vladkochetov07/wunder-funder"
SEED = 271_828
OPTIMIZERS = ("adamw", "muon_adamw")
SHAPES = ("baseline", "sine", "cubic")
EXPECTED_CELLS = {(SEED, optimizer, shape) for optimizer in OPTIMIZERS for shape in SHAPES}
FEATURES, HIDDEN, ROWS = 42, 64, 20_000
CHUNK_ROWS, UPDATES, PARAMETERS = 256, 474, 61_058
TELEMETRY_STEPS = (0, 46, 236, 473)
PAIR_SHAPES = (("sine", "baseline"), ("cubic", "baseline"), ("cubic", "sine"))
BOOTSTRAP_COUNT, BOOTSTRAP_SEED = 2_000, 271_828
DATASET_SHA256 = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
COHORT_SHA256 = "068103c71663bc445aab943e314c5be1d9a094f7f83456fdb406fbdc1012a6f2"
WORKER_SHA256 = "ad8bc19fd46a1394b7f9346eeb7451e14f7e4a6fc6bb643fdb6584d9566e4982"
BUILDER_SHA256 = "9acb6bd62b43f6cae66a09847ed9d2ff8dc37e6d3f12e009a23ede4ccd6fed54"
TESTS_SHA256 = "94d1049c3bdf3d2c9bd23095a1e13c5352c1b6fcf441b01334267d4f8f38bc2c"
ARCHIVE_SHA256 = "6cf83debabd43ef46365d224492c31f3c33aa579f07190ae387dc5384ccd4245"
MANIFEST_SHA256 = "64796bd729527c70a0be5402979be1bd5e5d0c525207cc57fb2fb0a3c38aa473"
KERNEL_SHA256 = "2d630c8c8d6d43d23c0053bc4c3f4d6c1a22ce293510754ac14312529fa2cf0c"
KERNEL_METADATA_SHA256 = "8935ac6dfe870891c8c6d2fc4308e0dac2bdf17c697528d5ed3ffe7e39894eb0"
BUILD_RECEIPT_SHA256 = "877514b74290f391fcd388b71f95d841f1a872804efb2c32857863644b6f5db3"
E470_SHA256 = "871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9"
E474_SHA256 = "40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068"
E481_SHA256 = "3d97d2d07b116b3cf04b365c33e3d3caed2c96c35108bd28f3a6ff63b54d65cc"
E481_GRADER_SHA256 = "ce439bb122d640279f8b199e7ee3af002f4ca70e16d6a171cf47e9587be9a4f3"
GMETRIC_SHA256 = "2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa"
EVALUATION_SHA256 = "96cebf2c8d564493f5a3a866b8d879964247f885587a455ebb1449e937230d0e"
RAM_LIMIT_BYTES = 6 * 1024**3
CONTROLLER_RESERVE_BYTES = 1 * 1024**3
GPU_ALLOCATOR_BYTES = 2 * 1024**3
GPU_STOP_BYTES = 3_000_000_000
HOST_STOP_FRACTION = 0.9
ADMISSION_DEADLINE = datetime(2026, 10, 5, 7, 0, tzinfo=UTC)
OOM_COUNTERS = ("oom", "oom_kill", "oom_group_kill")


def file_sha256(path: Path) -> str:
    """Hash a file incrementally to bound memory during provenance checks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    """Hash a JSON object with stable key ordering and compact encoding."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def finite_number(value: object, label: str) -> float:
    """Return a finite numeric value or reject malformed receipt data."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def packet_identities() -> dict[str, str]:
    """Verify the local, reviewed private packet and return its exact identities."""
    build_path = PACKET / "BUILD_RECEIPT.json"
    manifest_path = PACKET / "bundle/bundle-manifest.json"
    archive_path = PACKET / "bundle/bundle.wnn"
    kernel_path = PACKET / "bundle/kernel/kernel.py"
    metadata_path = PACKET / "bundle/kernel/kernel-metadata.json"
    observed = {
        "build_receipt": file_sha256(build_path),
        "bundle_manifest": file_sha256(manifest_path),
        "bundle_archive": file_sha256(archive_path),
        "kernel_source": file_sha256(kernel_path),
        "kernel_metadata": file_sha256(metadata_path),
        "worker": file_sha256(HERE / "e482_remote.py"),
        "builder": file_sha256(HERE / "prepare_e482.py"),
        "tests": file_sha256(HERE / "test_e482_packet.py"),
    }
    expected = {
        "build_receipt": BUILD_RECEIPT_SHA256,
        "bundle_manifest": MANIFEST_SHA256,
        "bundle_archive": ARCHIVE_SHA256,
        "kernel_source": KERNEL_SHA256,
        "kernel_metadata": KERNEL_METADATA_SHA256,
        "worker": WORKER_SHA256,
        "builder": BUILDER_SHA256,
        "tests": TESTS_SHA256,
    }
    if observed != expected:
        raise ValueError(f"local reviewed E482 packet identity differs: {observed}")
    receipt = json.loads(build_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    external_paths = set(manifest.get("external_mounts", {}))
    for item in manifest.get("files", []):
        relative = item.get("path")
        if not isinstance(relative, str):
            raise ValueError("E482 packet manifest has an invalid file path")
        path = ROOT / relative
        if relative in external_paths:
            if relative != "datasets/valid.parquet" or item.get("sha256") != DATASET_SHA256:
                raise ValueError("E482 packet has an unexpected external input")
            continue
        if (
            not path.is_file()
            or path.stat().st_size != item.get("size")
            or file_sha256(path) != item.get("sha256")
        ):
            raise ValueError(f"local E482 packet source differs from posted manifest: {relative}")
    if (
        receipt.get("status") != "local_source_packet_built_not_uploaded_not_launched"
        or receipt.get("private") is not True
        or receipt.get("contains_raw_dataset") is not False
        or receipt.get("dataset_slug") != DATASET_SLUG
        or receipt.get("kernel_slug") != KERNEL_REF
        or manifest.get("private") is not True
        or manifest.get("no_raw_data_in_archive") is not True
        or manifest.get("external_mounts", {}).get("datasets/valid.parquet")
        != {"dataset": INPUT_DATASET, "filename": "valid.parquet"}
        or metadata.get("id") != KERNEL_REF
        or metadata.get("title") != "wnn-e482-t4-sixfit-20261005-v2-runner"
        or metadata.get("is_private") is not True
        or metadata.get("enable_gpu") is not True
        or metadata.get("enable_internet") is not False
    ):
        raise ValueError("local E482 packet privacy, dataset, or kernel identity differs")
    return observed


def parse_utc(value: object, label: str) -> datetime:
    """Parse an explicit UTC timestamp from the provider proof."""
    if not isinstance(value, str):
        raise ValueError(f"{label} is not an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is not an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} has no timezone")
    return parsed.astimezone(UTC)


def output_file_inventory(outputs: object) -> list[dict[str, object]]:
    """Rehash exactly the provider proof's root-relative output inventory."""
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("provider output inventory is empty or malformed")
    allowed_roots = {
        "research/campaign_2026-10-05/activation_t4_replication_v2/remote_v1/",
        "research/campaign_2026-10-05/activation_t4_replication_v2/provider_metadata_v1/",
    }
    allowed_singletons = {
        "research/campaign_2026-10-05/activation_t4_replication_v2/provider_complete_status_v1.json",
        "research/campaign_2026-10-05/activation_t4_replication_v2/PROVIDER_ADMISSION.json",
    }
    remote_root = "research/campaign_2026-10-05/activation_t4_replication_v2/remote_v1/"
    expected_paths = set(allowed_singletons) | {
        "research/campaign_2026-10-05/activation_t4_replication_v2/provider_metadata_v1/kernel-metadata.json",
        "research/campaign_2026-10-05/activation_t4_replication_v2/provider_metadata_v1/wnn-e482-t4-sixfit-20261005-v2-runner.py",
        f"{remote_root}e482-results/admission.json",
        f"{remote_root}e482-results/e482_receipt.json",
        f"{remote_root}e482-results/fits/fit_adamw.log",
        f"{remote_root}e482-results/fits/fit_muon_adamw.log",
        f"{remote_root}e482-results/probes/probe_adamw.log",
        f"{remote_root}e482-results/probes/probe_muon_adamw.log",
        f"{remote_root}experiment/kaggle_specs/e482_spec.json",
        f"{remote_root}experiment/research/campaign_2026-10-05/target_mechanics/cohort.json",
        f"{remote_root}wnn-e482-t4-sixfit-20261005-v2-runner.log",
    }
    for optimizer in OPTIMIZERS:
        expected_paths.add(f"{remote_root}e482-results/fits/{optimizer}/fit_receipt.json")
        expected_paths.add(f"{remote_root}e482-results/probes/{optimizer}/probe.json")
        for shape in SHAPES:
            expected_paths.add(
                f"{remote_root}e482-results/fits/{optimizer}/seed{SEED}_{optimizer}_{shape}.pt"
            )
    entries, seen = [], set()
    for item in outputs:
        if not isinstance(item, dict):
            raise ValueError("provider output inventory entry is malformed")
        relative = item.get("path")
        if (
            not isinstance(relative, str)
            or relative in seen
            or not (
                relative in allowed_singletons
                or any(relative.startswith(root) for root in allowed_roots)
            )
        ):
            raise ValueError("provider output path is duplicate or outside the registered roots")
        seen.add(relative)
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"provider output is missing or a symlink: {relative}")
        if path.suffix not in {".json", ".log", ".pt", ".py"}:
            raise ValueError(f"unregistered provider output extension: {path.name}")
        entry = {
            "path": relative,
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        if item != entry:
            raise ValueError(f"provider output SHA/size differs: {relative}")
        entries.append(entry)
    if seen != expected_paths:
        raise ValueError("provider output inventory omits or adds a registered run artifact")
    return sorted(entries, key=lambda entry: str(entry["path"]))


def validate_provider_proof(proof_path: Path, run_root: Path, packet: dict[str, str]) -> dict:
    """Bind local downloaded artifacts to one successful private Kaggle kernel."""
    resolved_run = run_root.resolve(strict=True)
    if proof_path.resolve(strict=True).is_relative_to(resolved_run):
        raise ValueError("provider proof must be stored outside the downloaded output tree")
    proof = json.loads(proof_path.read_text())
    observation_path = ROOT / proof.get("status_observation_file", "")
    metadata_path = ROOT / proof.get("provider_metadata_file", "")
    if (
        observation_path.is_symlink()
        or metadata_path.is_symlink()
        or not observation_path.is_file()
        or not metadata_path.is_file()
    ):
        raise ValueError("provider status observation or posted metadata is missing/unsafe")
    observation = json.loads(observation_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    download_root = (ROOT / proof.get("download_root", "")).resolve(strict=True)
    if download_root != run_root.resolve(strict=True):
        raise ValueError("provider proof download root differs from the supplied run directory")
    if (
        proof.get("kernel_ref") != KERNEL_REF
        or proof.get("status") != "COMPLETE"
        or proof.get("private") is not True
        or proof.get("bundle_dataset") != DATASET_SLUG
        or proof.get("input_dataset") != INPUT_DATASET
        or observation.get("kernel_ref") != KERNEL_REF
        or observation.get("status") != "COMPLETE"
        or observation.get("exit_code") != 0
        or not isinstance(observation.get("stdout"), str)
        or metadata.get("id") != KERNEL_REF
        or metadata.get("title") != "wnn-e482-t4-sixfit-20261005-v2-runner"
        or metadata.get("is_private") is not True
        or metadata.get("enable_gpu") is not True
        or metadata.get("enable_tpu") is not False
        or metadata.get("enable_internet") is not False
        or metadata.get("dataset_sources") != [DATASET_SLUG, INPUT_DATASET]
        or metadata.get("code_file") != "wnn-e482-t4-sixfit-20261005-v2-runner.py"
    ):
        raise ValueError("provider proof does not bind the exact successful private E482 job")
    submitted = parse_utc(proof.get("submitted_utc"), "submitted_utc")
    terminal = parse_utc(proof.get("observed_terminal_utc"), "observed_terminal_utc")
    if submitted >= ADMISSION_DEADLINE or terminal < submitted:
        raise ValueError("E482 provider submission/completion timestamps violate admission")
    observed_utc = parse_utc(observation.get("observed_utc"), "status observed_utc")
    if abs(terminal.timestamp() - observed_utc.timestamp()) > 300:
        raise ValueError("provider status observation time differs from terminal receipt")
    actual_outputs = output_file_inventory(proof.get("outputs"))
    if len(actual_outputs) != 23:
        raise ValueError("provider output inventory count differs from completed-run packet")
    output_by_path = {item["path"]: item for item in actual_outputs}
    posted_code_path = "research/campaign_2026-10-05/activation_t4_replication_v2/provider_metadata_v1/wnn-e482-t4-sixfit-20261005-v2-runner.py"
    posted_metadata_path = "research/campaign_2026-10-05/activation_t4_replication_v2/provider_metadata_v1/kernel-metadata.json"
    if (
        output_by_path[posted_code_path]["sha256"] != packet["kernel_source"]
        or output_by_path[posted_metadata_path]["sha256"] != packet["kernel_metadata"]
        or file_sha256(ROOT / posted_code_path) != packet["kernel_source"]
    ):
        raise ValueError(
            "posted private kernel source/metadata differs from reviewed bundle: "
            f"code={output_by_path[posted_code_path]['sha256']}/{packet['kernel_source']}, "
            f"metadata={output_by_path[posted_metadata_path]['sha256']}/{packet['kernel_metadata']}"
        )
    return {
        "provider_proof_sha256": file_sha256(proof_path),
        "kernel_ref": KERNEL_REF,
        "kernel_status": "COMPLETE",
        "submitted_utc": submitted.isoformat(),
        "observed_terminal_utc": terminal.isoformat(),
        "downloaded_output_count": len(actual_outputs),
        "downloaded_outputs_sha256": canonical_json_sha256(actual_outputs),
    }


def assert_oom_unchanged(before: object, after: object, label: str) -> None:
    """Require cgroup OOM counters to be present and unchanged during one stage."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError(f"{label} lacks cgroup memory-event snapshots")
    for key in OOM_COUNTERS:
        left, right = before.get(key), after.get(key)
        if type(left) is not int or type(right) is not int or left < 0 or right != left:
            raise ValueError(f"{label} cgroup OOM counter changed or is malformed: {key}")


def _host_used_fraction(host: dict, label: str) -> float:
    """Validate a host snapshot and return its used-memory fraction."""
    capacity = host.get("capacity_bytes")
    available = host.get("available_bytes")
    affinity = host.get("affinity")
    if (
        type(capacity) is not int
        or type(available) is not int
        or capacity <= 0
        or available < 0
        or available > capacity
        or not isinstance(affinity, list)
        or len(affinity) < 3
    ):
        raise ValueError(f"{label} host snapshot is malformed or underprovisioned")
    used_fraction = (capacity - available) / capacity
    if used_fraction >= HOST_STOP_FRACTION:
        raise ValueError(f"{label} host memory reached its 90% stop")
    if min(int(capacity * 0.8), available - int(capacity * 0.2)) < (
        2 * RAM_LIMIT_BYTES + CONTROLLER_RESERVE_BYTES
    ):
        raise ValueError(f"{label} host lacks worker/support headroom")
    return used_fraction


def _device_map(devices: object, label: str, require_idle: bool) -> dict[str, dict]:
    """Validate two physical T4 records and return them by UUID."""
    if not isinstance(devices, list) or len(devices) != 2:
        raise ValueError(f"{label} must contain exactly two physical devices")
    result = {}
    for device in devices:
        if not isinstance(device, dict):
            raise ValueError(f"{label} contains a malformed device")
        uuid = device.get("uuid")
        if (
            not isinstance(uuid, str)
            or not uuid
            or uuid in result
            or "T4" not in str(device.get("name", ""))
            or type(device.get("index")) is not int
            or type(device.get("total_bytes")) is not int
            or type(device.get("used_bytes")) is not int
            or device["index"] not in (0, 1)
            or device["total_bytes"] <= GPU_STOP_BYTES
            or device["used_bytes"] < 0
            or device["used_bytes"] >= GPU_STOP_BYTES
            or (require_idle and device["used_bytes"] > 128 * 1024**2)
        ):
            raise ValueError(f"{label} contains an invalid/used T4 identity")
        result[uuid] = device
    if {record["index"] for record in result.values()} != {0, 1}:
        raise ValueError(f"{label} device indexes differ from the registered map")
    return result


def validate_supervision(resources: object, label: str, initial: bool = False) -> dict[str, dict]:
    """Validate exit, RAM, host, GPU and OOM evidence for a two-worker stage."""
    if not isinstance(resources, dict):
        raise ValueError(f"{label} resource receipt is missing")
    start, end = resources.get("resource_start"), resources.get("resource_end")
    if not isinstance(start, dict) or not isinstance(end, dict):
        raise ValueError(f"{label} resource snapshots are missing")
    assert_oom_unchanged(start.get("memory_events"), end.get("memory_events"), label)
    start_host, end_host = start.get("host"), end.get("host")
    start_used = _host_used_fraction(start_host, f"{label} start")
    end_used = _host_used_fraction(end_host, f"{label} end")
    start_devices = _device_map(start.get("devices"), f"{label} start", initial)
    end_devices = _device_map(end.get("devices"), f"{label} end", False)
    if set(start_devices) != set(end_devices):
        raise ValueError(f"{label} physical T4 inventory changed during work")
    if resources.get("exit_codes") != {optimizer: 0 for optimizer in OPTIMIZERS}:
        raise ValueError(f"{label} worker exit codes are not both zero")
    peak_rss = resources.get("peak_process_tree_rss_bytes_by_optimizer")
    if not isinstance(peak_rss, dict) or set(peak_rss) != set(OPTIMIZERS):
        raise ValueError(f"{label} lacks per-worker process-tree RSS receipts")
    if any(
        type(value) is not int or value < 0 or value >= RAM_LIMIT_BYTES
        for value in peak_rss.values()
    ):
        raise ValueError(f"{label} worker process-tree RSS crossed 6 GiB")
    thresholds = resources.get("thresholds", {})
    if thresholds != {
        "ram_worker_bytes": RAM_LIMIT_BYTES,
        "gpu_physical_stop_bytes": GPU_STOP_BYTES,
        "host_stop_fraction": HOST_STOP_FRACTION,
    }:
        raise ValueError(f"{label} recorded resource stop thresholds differ")
    return {
        "uuids": start_devices,
        "start_host_used_fraction": start_used,
        "end_host_used_fraction": end_used,
        "peak_rss": peak_rss,
    }


def expected_sources() -> dict[str, str]:
    """Recompute the exact remote worker source identity map locally."""
    sources = {
        "e481": E481_DIR / "fit_recurrent_curvature.py",
        "e470": E470_DIR / "fit_activation_optimizer.py",
        "e474": E474_DIR / "fit_activation_placement.py",
        "muon_optimizer": ROOT / "src/wnn/optim.py",
        "remote_worker": HERE / "e482_remote.py",
    }
    observed = {name: file_sha256(path) for name, path in sources.items()}
    fixed = {
        "e481": E481_SHA256,
        "e470": E470_SHA256,
        "e474": E474_SHA256,
        "remote_worker": WORKER_SHA256,
    }
    for name, digest in fixed.items():
        if observed[name] != digest:
            raise ValueError(f"local E482 source identity changed: {name}")
    return observed


def validate_scorer_sources() -> dict[str, str]:
    """Pin the independent E481 reconstruction and official scoring helpers."""
    paths = {
        "e481_reconstruction": E481_DIR / "score_recurrent_curvature_v2.py",
        "official_gmetric": ROOT / "src/wnn/gmetric.py",
        "official_evaluation": ROOT / "src/wnn/evaluation.py",
    }
    observed = {name: file_sha256(path) for name, path in paths.items()}
    expected = {
        "e481_reconstruction": E481_GRADER_SHA256,
        "official_gmetric": GMETRIC_SHA256,
        "official_evaluation": EVALUATION_SHA256,
    }
    if observed != expected:
        raise ValueError(f"independent E482 scorer dependencies changed: {observed}")
    return observed


def canonical_fit_nodes() -> list[dict]:
    """Read only the canonical cohort metadata and return its 24 fit roles."""
    if file_sha256(e470.COHORT_PATH) != COHORT_SHA256:
        raise ValueError("E482 cohort manifest bytes changed")
    document = json.loads(e470.COHORT_PATH.read_text())
    nodes = [node for node in document["cohort"] if node["role"] == "fit"]
    if len(nodes) != 24 or len({node["component"] for node in nodes}) != 24:
        raise ValueError("canonical E482 fit cohort is incomplete")
    return nodes


def validate_row_order(arm: dict, nodes: list[dict], label: str) -> list[int]:
    """Validate seed-shuffled fit traversal and its sampled batch/chunk trace."""
    order = np.arange(len(nodes))
    np.random.default_rng(SEED).shuffle(order)
    expected_order = [nodes[int(index)]["row_group"] for index in order]
    if arm.get("fit_row_groups") != expected_order:
        raise ValueError(f"{label} full fit-row traversal differs")
    telemetry = arm.get("telemetry")
    if not isinstance(telemetry, list) or [
        row.get("update_index_zero_based") for row in telemetry
    ] != list(TELEMETRY_STEPS):
        raise ValueError(f"{label} telemetry cadence differs")
    for row in telemetry:
        step = row["update_index_zero_based"]
        batch = step // math.ceil(ROWS / CHUNK_ROWS)
        expected_batch = expected_order[batch * 4 : (batch + 1) * 4]
        expected_chunk = (step % math.ceil(ROWS / CHUNK_ROWS)) * CHUNK_ROWS
        if row.get("fit_row_groups") != expected_batch or row.get("chunk_start") != expected_chunk:
            raise ValueError(f"{label} sampled batch or chunk provenance differs")
        for field in ("loss", "state_abs_max"):
            finite_number(row.get(field), f"{label} telemetry {field}")
        for field in ("preclip_gradient_rms", "parameter_update_rms"):
            families = row.get(field)
            if not isinstance(families, dict) or set(families) != {"inp", "gru", "head"}:
                raise ValueError(f"{label} telemetry families differ: {field}")
            for family, value in families.items():
                if finite_number(value, f"{label} {field}/{family}") < 0:
                    raise ValueError(f"{label} {field}/{family} is negative")
    return expected_order


def validate_checkpoint(checkpoint: Path, arm: dict, optimizer: str, shape: str) -> None:
    """Check checkpoint hash, model keys, identity buffers, and finite tensors."""
    expected_name = f"seed{SEED}_{optimizer}_{shape}.pt"
    if arm.get("checkpoint") != expected_name or checkpoint.name != expected_name:
        raise ValueError(f"unregistered E482 checkpoint name: {checkpoint.name}")
    if not checkpoint.is_file() or file_sha256(checkpoint) != arm.get("checkpoint_sha256"):
        raise ValueError(f"E482 checkpoint SHA differs: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(
        not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
        for value in state.values()
    ):
        raise ValueError(f"E482 checkpoint has nonfinite or malformed tensors: {checkpoint.name}")
    e481grade.validate_identity_buffers(state)
    model = e481grade.CurvatureModel(shape)
    if set(state) != set(model.state_dict()):
        raise ValueError(f"E482 checkpoint keys differ from independent model: {checkpoint.name}")
    if (
        sum(value.numel() for name, value in state.items() if name not in {"mean", "std"})
        != PARAMETERS
    ):
        raise ValueError(f"E482 checkpoint parameter count differs: {checkpoint.name}")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"E482 checkpoint tensor shapes differ: {checkpoint.name}") from exc


def validate_probe_receipt(
    receipt: dict,
    optimizer: str,
    sources: dict[str, str],
    device: dict,
) -> dict[tuple[int, str, str], dict]:
    """Validate one optimizer's three actual T4 probes and native parity proof."""
    if (
        receipt.get("status") != "probe_passed"
        or receipt.get("phase") != "probe"
        or receipt.get("optimizer") != optimizer
        or receipt.get("fit_only") is not True
        or receipt.get("test_rows_loaded") is not False
        or receipt.get("source_sha256") != sources
        or receipt.get("dataset_sha256") != DATASET_SHA256
        or receipt.get("cohort_sha256") != COHORT_SHA256
        or receipt.get("assigned_device", {}).get("uuid") != device["uuid"]
        or receipt.get("device_properties", {}).get("uuid") != device["uuid"]
        or "T4" not in str(receipt.get("assigned_device", {}).get("name", ""))
        or receipt.get("native_parity", {}).get("optimizer") != optimizer
        or receipt.get("native_parity", {}).get(
            "output_state_input_gradient_parameter_gradient_and_clipped_step"
        )
        != "passed"
    ):
        raise ValueError(f"E482 {optimizer} probe identity/parity differs")
    for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
        value = receipt.get(field)
        if type(value) is not int or value < 0 or value > GPU_ALLOCATOR_BYTES:
            raise ValueError(f"E482 {optimizer} probe exceeded allocator limit: {field}")
    # E482 v2 serializes precision metadata after its context managers restore
    # the caller's flags. The frozen worker source, not those restored fields,
    # binds the actual IEEE forward/backward path and unchanged BF16 Muon step.
    by_key = {}
    for arm in receipt.get("arms", []):
        key = (arm.get("seed"), arm.get("optimizer"), arm.get("shape"))
        if key not in EXPECTED_CELLS or key in by_key:
            raise ValueError(f"unknown or duplicate E482 probe cell: {key}")
        if (
            arm.get("batch") != 4
            or arm.get("chunk_rows") != CHUNK_ROWS
            or arm.get("updates") != 1
            or type(arm.get("changed_parameter_tensors")) is not int
            or arm["changed_parameter_tensors"] <= 0
            or arm.get("parameters") != PARAMETERS
        ):
            raise ValueError(f"E482 {key} is not a finite changed B4×256 probe")
        finite_number(arm.get("loss"), f"E482 {key} probe loss")
        telemetry = arm.get("telemetry_step0", {})
        if telemetry.get("update_index_zero_based") != 0:
            raise ValueError(f"E482 {key} probe telemetry is not update zero")
        for field in ("preclip_gradient_rms", "parameter_update_rms"):
            families = telemetry.get(field)
            if not isinstance(families, dict) or set(families) != {"inp", "gru", "head"}:
                raise ValueError(f"E482 {key} probe telemetry families differ: {field}")
            for family, value in families.items():
                if finite_number(value, f"E482 {key} {field}/{family}") < 0:
                    raise ValueError(f"E482 {key} {field}/{family} is negative")
        by_key[key] = arm
    if set(by_key) != {(SEED, optimizer, shape) for shape in SHAPES}:
        raise ValueError(f"E482 {optimizer} probe omits registered shapes")
    return by_key


def validate_fit_receipt(
    receipt: dict,
    optimizer: str,
    sources: dict[str, str],
    device: dict,
    run_root: Path,
    nodes: list[dict],
) -> dict[tuple[int, str, str], dict]:
    """Validate one remote worker's three complete fit arms and checkpoint bytes."""
    if (
        receipt.get("status") != "three_fit_arms_complete_test_unread"
        or receipt.get("phase") != "fit"
        or receipt.get("fit_only") is not True
        or receipt.get("test_rows_loaded") is not False
        or receipt.get("planned_fit_count") != 3
        or receipt.get("batch") != 4
        or receipt.get("chunk_rows") != CHUNK_ROWS
        or receipt.get("updates_per_arm") != UPDATES
        or receipt.get("optimizer") != optimizer
        or receipt.get("seed") != SEED
        or receipt.get("dataset_sha256") != DATASET_SHA256
        or receipt.get("cohort_sha256") != COHORT_SHA256
        or receipt.get("source_sha256") != sources
        or receipt.get("assigned_device", {}).get("uuid") != device["uuid"]
        or "T4" not in str(receipt.get("assigned_device", {}).get("name", ""))
    ):
        raise ValueError(f"E482 {optimizer} fit receipt identity differs")
    canonical = [node for node in canonical_fit_nodes()]
    fit_manifest = receipt.get("fit_only_cohort", {})
    if fit_manifest.get("cohort") != canonical or not fit_manifest.get("normalization"):
        raise ValueError(f"E482 {optimizer} fit-only cohort differs from canonical 24 groups")
    if receipt.get("parent_source_sha256") != {"e470": E470_SHA256, "e474": E474_SHA256}:
        raise ValueError(f"E482 {optimizer} parent helper identities differ")
    if (
        receipt.get("architecture")
        != "E481 frozen; 42 features, two-layer H64 GRU, 61,058 parameters"
    ):
        raise ValueError(f"E482 {optimizer} model architecture differs")
    order_by_arm = {}
    by_key = {}
    for arm in receipt.get("arms", []):
        key = (arm.get("seed"), arm.get("optimizer"), arm.get("shape"))
        if key not in EXPECTED_CELLS or key in by_key:
            raise ValueError(f"unknown or duplicate E482 fit cell: {key}")
        if arm.get("updates") != UPDATES or arm.get("parameters") != PARAMETERS:
            raise ValueError(f"E482 fit cadence/size differs for {key}")
        order_by_arm[key] = validate_row_order(arm, nodes, f"E482 {key}")
        checkpoint = run_root / "fits" / optimizer / str(arm.get("checkpoint", ""))
        validate_checkpoint(checkpoint, arm, optimizer, str(key[2]))
        finite_number(arm.get("fit_seconds"), f"E482 {key} fit_seconds")
        by_key[key] = arm
    if set(by_key) != {(SEED, optimizer, shape) for shape in SHAPES}:
        raise ValueError(f"E482 {optimizer} fit receipt omits registered shapes")
    if len({tuple(order) for order in order_by_arm.values()}) != 1:
        raise ValueError(f"E482 {optimizer} recipes used different fit permutations")
    # run_fits serializes precision fields after the FP32 context exits. The
    # backend values here are restored defaults, not fit precision evidence.
    return by_key


def validate_remote_run(
    run_root: Path, provider_proof_path: Path
) -> tuple[dict, dict, dict, dict, dict]:
    """Validate all remote fit/probe/provider proof before any test loader call."""
    if not run_root.is_dir() or (run_root / "E482_FAILURE.json").exists():
        raise ValueError("E482 output directory is missing or contains a failure receipt")
    packet = packet_identities()
    scorer_sources = validate_scorer_sources()
    provider = validate_provider_proof(provider_proof_path, run_root, packet)
    sources = expected_sources()
    if file_sha256(e470.DATA_PATH) != DATASET_SHA256:
        raise ValueError("local E482 dataset bytes differ")
    nodes = canonical_fit_nodes()
    manifest, fit_x, _fit_y, _fit_need, _fit_selected, fit_nodes = e470.load_cohort(
        include_test=False
    )
    if fit_nodes != nodes or fit_x.shape != (24, ROWS, FEATURES) or not np.isfinite(fit_x).all():
        raise ValueError("local E482 loader did not return exact finite 24-sequence fit data")
    result_root = run_root / "e482-results"
    receipt_path = result_root / "e482_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    admission = json.loads((result_root / "admission.json").read_text())
    if (
        receipt.get("kind") != "E482 six-recipe private T4 replication"
        or receipt.get("status") != "six_fits_complete_test_unread"
        or receipt.get("fit_only") is not True
        or receipt.get("test_rows_loaded") is not False
        or receipt.get("fit_count") != 6
        or receipt.get("seed") != SEED
        or receipt.get("optimizers") != list(OPTIMIZERS)
        or receipt.get("shapes") != list(SHAPES)
        or receipt.get("dataset_sha256") != DATASET_SHA256
        or receipt.get("cohort_sha256") != COHORT_SHA256
        or receipt.get("source_sha256") != sources
        or admission.get("inputs")
        != {"dataset_sha256": DATASET_SHA256, "cohort_sha256": COHORT_SHA256}
        or admission.get("devices") != receipt.get("devices_initial")
        or admission.get("host") != receipt.get("host_initial")
    ):
        raise ValueError("E482 orchestrator/admission receipt identity differs")
    devices = _device_map(receipt.get("devices_initial"), "initial admission", True)
    before_fit_devices = _device_map(receipt.get("before_fit_devices"), "pre-fit admission", True)
    if set(devices) != set(before_fit_devices):
        raise ValueError("E482 physical GPU UUIDs changed before fit admission")
    _host_used_fraction(receipt.get("host_initial", {}), "initial admission")
    if admission.get("host_budget_bytes", 0) < 2 * RAM_LIMIT_BYTES + CONTROLLER_RESERVE_BYTES:
        raise ValueError("E482 admission host headroom receipt is below requirement")
    if admission.get("after_input_hash_host_budget_bytes", 0) < (
        2 * RAM_LIMIT_BYTES + CONTROLLER_RESERVE_BYTES
    ):
        raise ValueError("E482 post-input host headroom receipt is below requirement")
    probe_stage = validate_supervision(receipt.get("probe_resources"), "probe stage", True)
    fit_stage = validate_supervision(receipt.get("fit_resources"), "fit stage", True)
    if set(probe_stage["uuids"]) != set(devices) or set(fit_stage["uuids"]) != set(devices):
        raise ValueError("E482 supervised GPU inventory differs from initial admission")
    device_for_optimizer = {
        "adamw": next(device for device in devices.values() if device["index"] == 0),
        "muon_adamw": next(device for device in devices.values() if device["index"] == 1),
    }
    if len({device["uuid"] for device in device_for_optimizer.values()}) != 2:
        raise ValueError("E482 optimizer workers were not assigned distinct T4s")

    probe_files, fit_files = {}, {}
    for optimizer in OPTIMIZERS:
        probe_path = result_root / "probes" / optimizer / "probe.json"
        fit_path = result_root / "fits" / optimizer / "fit_receipt.json"
        probe_files[optimizer] = json.loads(probe_path.read_text())
        fit_files[optimizer] = json.loads(fit_path.read_text())
    if receipt.get("probes") != [probe_files[name] for name in OPTIMIZERS]:
        raise ValueError("E482 top-level probe receipts differ from downloaded worker receipts")
    if receipt.get("fits") != [fit_files[name] for name in OPTIMIZERS]:
        raise ValueError("E482 top-level fit receipts differ from downloaded worker receipts")
    probe_cells = {}
    fit_cells = {}
    for optimizer in OPTIMIZERS:
        device = device_for_optimizer[optimizer]
        probe_cells.update(
            validate_probe_receipt(probe_files[optimizer], optimizer, sources, device)
        )
        fit_cells.update(
            validate_fit_receipt(fit_files[optimizer], optimizer, sources, device, result_root, nodes)
        )
        if probe_files[optimizer].get("assigned_device", {}).get("uuid") != fit_files[
            optimizer
        ].get("assigned_device", {}).get("uuid"):
            raise ValueError(f"E482 {optimizer} changed T4 between successful probe and fits")
        if fit_files[optimizer].get("source_sha256") != probe_files[optimizer].get("source_sha256"):
            raise ValueError(f"E482 {optimizer} fit source changed after probe")
    if set(probe_cells) != EXPECTED_CELLS or set(fit_cells) != EXPECTED_CELLS:
        raise ValueError("E482 remote run is missing one or more of its six registered cells")
    fit_normalization = fit_files["adamw"]["fit_only_cohort"]
    if fit_files["muon_adamw"].get("fit_only_cohort") != fit_normalization:
        raise ValueError("E482 optimizer workers used different fit-only cohort/normalization")
    for worker in fit_files.values():
        if (
            worker.get("peak_allocated_bytes", GPU_ALLOCATOR_BYTES + 1) > GPU_ALLOCATOR_BYTES
            or worker.get("peak_reserved_bytes", GPU_ALLOCATOR_BYTES + 1) > GPU_ALLOCATOR_BYTES
        ):
            raise ValueError("E482 full-fit worker exceeded its CUDA allocator cap")
    # Independently verify the fit-only normalization and identity checkpoint buffers.
    fit_mean, fit_scale = e481grade.validate_normalizer(
        fit_normalization, fit_x, nodes, fit_normalization, fit_x, nodes
    )
    if fit_normalization != manifest:
        raise ValueError("remote fit manifest differs from the locally loaded fit-only manifest")
    proof = {
        "provider": provider,
        "packet": packet,
        "source_sha256": sources,
        "independent_scorer_sources": scorer_sources,
        "remote_receipt_sha256": file_sha256(receipt_path),
        "admission_sha256": file_sha256(result_root / "admission.json"),
        "fit_only_normalizer_mean_sha256": hashlib.sha256(
            fit_mean.astype("<f4").tobytes()
        ).hexdigest(),
        "fit_only_normalizer_scale_sha256": hashlib.sha256(
            fit_scale.astype("<f4").tobytes()
        ).hexdigest(),
        "validated_cells": len(fit_cells),
        "test_rows_loaded": False,
        "test_predictions_scored": False,
    }
    return receipt, fit_cells, proof, fit_normalization, {"fit_x": fit_x, "fit_nodes": nodes}


def score_validated_run(
    run_root: Path,
    cells: dict[tuple[int, str, str], dict],
    fit_manifest: dict,
    fit_x: np.ndarray,
    fit_nodes: list[dict],
    device: torch.device,
) -> tuple[dict, dict]:
    """Load test sequences only after provenance validation and score all six arms."""
    if file_sha256(e470.DATA_PATH) != DATASET_SHA256:
        raise ValueError("E482 dataset identity changed before test loading")
    all_manifest, all_x, all_y, _need, all_selected, all_nodes = e470.load_cohort(include_test=True)
    canonical = json.loads(e470.COHORT_PATH.read_text())["cohort"]
    if all_nodes != canonical or all_x.shape != (40, ROWS, FEATURES):
        raise ValueError("E482 score loader role/order differs from canonical cohort")
    mean, scale = e481grade.validate_normalizer(
        fit_manifest, fit_x, fit_nodes, all_manifest, all_x, all_nodes
    )
    test_positions = [index for index, node in enumerate(all_nodes) if node["role"] == "test"]
    test_nodes = [all_nodes[index] for index in test_positions]
    if len(test_nodes) != 16 or len({node["component"] for node in test_nodes}) != 16:
        raise ValueError("E482 evaluation cohort is not the fixed 16 exposed components")
    test_x, clipped, selected = (
        all_x[test_positions],
        all_y[test_positions],
        all_selected[test_positions],
    )
    raw_targets = e481grade.read_raw_test_targets(test_nodes, clipped, selected)
    row_counts = selected.sum(axis=1).astype(np.int64)
    if np.any(row_counts <= 0):
        raise ValueError("every E482 evaluation component must contain selected rows")
    component_ids = np.asarray([node["component"] for node in test_nodes], dtype=np.int64)
    order = np.argsort(component_ids, kind="stable")
    sorted_components = component_ids[order]
    multiplicities = e481grade.bootstrap_counts(sorted_components)
    scores, moments, predictions = {}, {}, {}
    for optimizer in OPTIMIZERS:
        for shape in SHAPES:
            cell = (SEED, optimizer, shape)
            arm = cells[cell]
            key = f"{optimizer}__{shape}"
            prediction = e481grade.predict_sequences(
                run_root / "e482-results" / "fits" / optimizer / arm["checkpoint"], shape, test_x, device
            )
            if not np.isfinite(prediction).all():
                raise ValueError(f"E482 raw predictions are nonfinite: {cell}")
            metric = e481grade.global_wp(raw_targets, prediction, selected)
            if metric.get("metric_id") != METRIC_ID or metric.get("selected_rows") != int(
                row_counts.sum()
            ):
                raise ValueError(f"official E482 metric metadata differs: {cell}")
            component_moments, _ = e481grade.paired_component_moments(
                raw_targets, prediction, prediction, selected
            )
            scores[key] = {
                "optimizer": optimizer,
                "shape": shape,
                "wp": float(metric["wp"]),
                "t0": float(metric["t0"]),
                "t1": float(metric["t1"]),
                "selected_rows": int(metric["selected_rows"]),
                "component_row_counts": row_counts.tolist(),
                "parameters": PARAMETERS,
                "updates": UPDATES,
            }
            moments[key] = component_moments[order]
            predictions[key] = prediction[selected].copy()
    arm_draws = {
        key: np.asarray(
            [e481grade.pool_moments(value, multiplicity) for multiplicity in multiplicities],
            dtype=np.float64,
        )
        for key, value in moments.items()
    }
    contrasts = e481grade.registered_contrasts(
        {key: value["wp"] for key, value in scores.items()}, arm_draws
    )
    if len(scores) != 6 or len(contrasts) != 10:
        raise AssertionError("E482 registered score/contrast inventory differs")
    selected_targets = raw_targets[selected]
    arm_names = [f"{optimizer}__{shape}" for optimizer in OPTIMIZERS for shape in SHAPES]
    exports = {
        "arm_names": np.asarray(arm_names),
        "targets_raw": selected_targets,
        "predictions_raw": np.stack([predictions[key] for key in arm_names]),
        "selected_mask": selected,
        "component_ids": component_ids,
        "sorted_component_ids": sorted_components,
        "row_groups": np.asarray([node["row_group"] for node in test_nodes], dtype=np.int64),
        "selected_row_counts": row_counts,
        "component_bootstrap_multiplicities": multiplicities,
        "arm_bootstrap_wp": np.stack([arm_draws[key] for key in arm_names]),
        "contrast_names": np.asarray(list(contrasts)),
        "contrast_bootstrap_diffs": np.stack(
            [
                arm_draws["muon_adamw__cubic"]
                - arm_draws["muon_adamw__sine"]
                - arm_draws["adamw__cubic"]
                + arm_draws["adamw__sine"]
                if name == "curvature_cubic_minus_sine_x_muon_minus_adamw"
                else arm_draws[contrast["candidate"]] - arm_draws[contrast["reference"]]
                for name, contrast in contrasts.items()
            ]
        ),
    }
    result = {
        "kind": "independent E482 remote T4 six-recipe grader",
        "metric_id": METRIC_ID,
        "scope": "one fit seed; same 16 previously exposed components; exploratory only",
        "seed": SEED,
        "validated_fit_cells": 6,
        "contrast_count": 10,
        "fit_only_proof": "passed_before_test_loader",
        "normalizer": {
            "source": "independent raw fit-role derivation compared with E470 loader",
            "fit_mean_sha256": hashlib.sha256(mean.astype("<f4").tobytes()).hexdigest(),
            "fit_scale_sha256": hashlib.sha256(scale.astype("<f4").tobytes()).hexdigest(),
            "identity_model_buffers_required": True,
        },
        "component_ids": component_ids.tolist(),
        "sorted_component_ids": sorted_components.tolist(),
        "row_groups": [node["row_group"] for node in test_nodes],
        "selected_row_counts": row_counts.tolist(),
        "selected_rows_total": int(row_counts.sum()),
        "arm_scores": scores,
        "contrasts": contrasts,
        "bootstrap": {
            "method": "official centered weighted-Pearson moments and shared component draws",
            "n_boot": BOOTSTRAP_COUNT,
            "seed": BOOTSTRAP_SEED,
            "multiplicities_shared_across_all_six_arms_and_contrasts": True,
            "resampling_unit": "numeric-sorted fixed component ID",
            "no_seed_pooling": True,
        },
    }
    return result, exports


def main() -> None:
    """Validate fit-only evidence first; score only with an explicit later request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--provider-proof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-fit-only", action="store_true")
    mode.add_argument("--score", action="store_true")
    args = parser.parse_args()
    run_root, output = args.run.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite E482 grader output: {output}")
    receipt, cells, proof, fit_manifest, fit_data = validate_remote_run(
        run_root, args.provider_proof.resolve()
    )
    if args.verify_fit_only:
        print(
            json.dumps(
                {
                    "status": "all_six_remote_fit_receipts_and_checkpoints_verified_test_unread",
                    "fit_cells": len(cells),
                    "test_rows_loaded": False,
                    "test_predictions_scored": False,
                    "provider_proof_sha256": proof["provider"]["provider_proof_sha256"],
                    "remote_receipt_sha256": proof["remote_receipt_sha256"],
                },
                indent=2,
            ),
            flush=True,
        )
        return
    if args.device == "cuda":
        from wnn.resources import configure_torch

        configure_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA scoring requested but unavailable")
    torch.set_num_threads(1)
    score, exports = score_validated_run(
        run_root,
        cells,
        fit_manifest,
        fit_data["fit_x"],
        fit_data["fit_nodes"],
        torch.device(args.device),
    )
    output.mkdir(parents=True)
    score["provider_proof"] = proof["provider"]
    score["remote_receipt_sha256"] = proof["remote_receipt_sha256"]
    score["source_sha256"] = {
        "grader": file_sha256(Path(__file__)),
        **validate_scorer_sources(),
    }
    export_path = output / "selected_exports.npz"
    temporary_export = output / "selected_exports.tmp.npz"
    np.savez_compressed(temporary_export, **exports)
    temporary_export.replace(export_path)
    score["selected_exports_path"] = export_path.name
    score["selected_exports_sha256"] = file_sha256(export_path)
    score["fit_only_proof"] = proof
    temp_result = output / "scores.tmp.json"
    temp_result.write_text(json.dumps(score, indent=2, allow_nan=False) + "\n")
    temp_result.replace(output / "scores.json")
    print(
        json.dumps(
            {
                "output": str(output),
                "arms": len(score["arm_scores"]),
                "contrasts": score["contrast_count"],
                "selected_rows": score["selected_rows_total"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
