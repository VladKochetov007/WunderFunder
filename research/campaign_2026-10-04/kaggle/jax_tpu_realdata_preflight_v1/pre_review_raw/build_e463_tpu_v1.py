from __future__ import annotations

import ast
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "research/campaign_2026-10-04/kaggle/jax_realdata_preflight_v3"
HERE = ROOT / "research/campaign_2026-10-04/kaggle/jax_tpu_realdata_preflight_v1"
OWNER = "vladkochetov07"
SLUG = "wnn-jax-realdata-e463-tpu-20261004-v1"
KERNEL_SLUG = f"{SLUG}-runner"
RSS_LIMIT = 12 * 1024**3
LAUNCHER_RESERVE = 2 * 1024**3
MAX_RUNTIME = 3600
PROVIDER_TIMEOUT = 4200


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def patch_preflight(source: str) -> str:
    source = source.replace("import time\n", "import time\nimport threading\n", 1)
    old = '''    devices = jax.devices()
    if len(devices) != 1:
        raise RuntimeError(f"preflight requires exactly one visible JAX device, got {devices}")
    if str(getattr(devices[0], "platform", "unknown")) != args.backend:
        raise RuntimeError(
            f"requested backend {args.backend!r}, but JAX selected {devices[0]}"
        )
'''
    new = '''    devices = jax.devices(args.backend)
    expected_device_count = 8 if args.backend == "tpu" else 1
    if len(devices) != expected_device_count:
        raise RuntimeError(f"preflight requires {expected_device_count} {args.backend} devices, got {devices}")
    if any(str(getattr(device, "platform", "unknown")) != args.backend for device in devices):
        raise RuntimeError(f"requested backend {args.backend!r}, got devices {devices}")
    if args.backend == "tpu" and any("TPU v5 lite" not in str(getattr(device, "device_kind", "")) for device in devices):
        raise RuntimeError(f"preflight requires eight TPU v5 lite devices, got {devices}")

    memory_path = Path(os.environ["WNN_TPU_MEMORY_SAMPLES"])
    stop_memory_sampler = threading.Event()

    def sample_tpu_memory(phase: str) -> None:
        records = []
        for device in devices:
            raw_stats = device.memory_stats()
            if not isinstance(raw_stats, dict):
                raise RuntimeError(f"TPU {device} memory_stats is unavailable")
            stats = {str(key): int(value) for key, value in raw_stats.items()}
            limit = stats.get("bytes_limit")
            if not limit or limit <= 0:
                raise RuntimeError(f"TPU {device} has no positive bytes_limit: {stats}")
            observed = max(stats.get("bytes_in_use", 0), stats.get("peak_bytes_in_use", 0))
            records.append({"device": _device_record(device), "memory_stats": stats,
                            "allocator_target_fraction": 0.80, "observed_hard_stop_fraction": 0.90,
                            "observed_hard_stop_bytes": int(limit * 0.90),
                            "observed_bytes_in_use_or_peak": observed})
        try:
            current_rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            current_rss = 0
        entry = {"phase": phase, "wall_time_epoch_seconds": time.time(),
                 "elapsed_seconds": time.monotonic() - started,
                 "current_process_rss_bytes": current_rss, "devices": records}
        with memory_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\\n")
            stream.flush()
            os.fsync(stream.fileno())
        if any(row["observed_bytes_in_use_or_peak"] >= row["observed_hard_stop_bytes"] for row in records):
            os._exit(86)

    def memory_sampler() -> None:
        try:
            while not stop_memory_sampler.is_set():
                sample_tpu_memory("periodic")
                stop_memory_sampler.wait(5.0)
        except BaseException as error:
            try:
                with memory_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"phase": "memory_sampler_failed", "error": repr(error)}) + "\\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                os._exit(87)

    sample_tpu_memory("jax_devices_validated")
    threading.Thread(target=memory_sampler, name="tpu-memory-guard", daemon=True).start()
'''
    if source.count(old) != 1:
        raise ValueError("JAX device validation block changed")
    source = source.replace(old, new, 1)
    old_main = '''    report = run(args)
    print(json.dumps({"status": report["status"], "metrics": str(args.output / "metrics.json")}))
'''
    new_main = '''    if args.backend == "tpu":
        os.environ["JAX_PLATFORMS"] = "tpu"
        import jax
        devices = jax.devices("tpu")
        if len(devices) != 8:
            raise RuntimeError(f"expected exactly eight TPU devices, got {devices}")
        with jax.default_device(devices[0]):
            report = run(args)
    else:
        report = run(args)
    print(json.dumps({"status": report["status"], "metrics": str(args.output / "metrics.json")}))
'''
    if source.count(old_main) != 1:
        raise ValueError("realdata main block changed")
    source = source.replace(old_main, new_main, 1)
    checkpoints = [
        ('        _emit_milestone(f"{precision}_gradient_update_start")\n',
         '        sample_tpu_memory(f"{precision}_gradient_update_start")\n        _emit_milestone(f"{precision}_gradient_update_start")\n'),
        ('        loss = jax.block_until_ready(loss)\n',
         '        loss = jax.block_until_ready(loss)\n        sample_tpu_memory(f"{precision}_gradient_update_complete")\n'),
        ('        _emit_milestone(f"{precision}_evaluation_forward_start")\n',
         '        sample_tpu_memory(f"{precision}_evaluation_forward_start")\n        _emit_milestone(f"{precision}_evaluation_forward_start")\n'),
        ('        _emit_milestone(f"{precision}_evaluation_complete", scored_rows=int(score_mask.sum()), evaluation_score=evaluation_score)\n',
         '        sample_tpu_memory(f"{precision}_evaluation_complete")\n        _emit_milestone(f"{precision}_evaluation_complete", scored_rows=int(score_mask.sum()), evaluation_score=evaluation_score)\n'),
    ]
    for before, after in checkpoints:
        if source.count(before) != 1:
            raise ValueError(f"expected one phase checkpoint to instrument: {before.strip()}")
        source = source.replace(before, after, 1)
    return source


def patch_kernel(source: str, manifest_sha: str, archive_sha: str) -> str:
    source = source.replace("wnn-jax-realdata-e463-20261004-v3", SLUG)
    source = source.replace("jax_realdata_preflight_e463_v3", "jax_realdata_preflight_e463_tpu_v1")
    source = source.replace("RSS_LIMIT_BYTES = 4 * 1024**3", f"RSS_LIMIT_BYTES = {RSS_LIMIT}")
    source = source.replace("MAX_RUNTIME_SECONDS = 1800", f"MAX_RUNTIME_SECONDS = {MAX_RUNTIME}")
    source = source.replace("# E-463 private CPU-only JAX real-data one-update preflight; no TPU or score claim.",
                            "# E-463 bounded private JAX TPU real-data one-update preflight; no fit or score claim.")
    source = source.replace('manifest.get("tpu") is not False', 'manifest.get("tpu") is not True')
    source = source.replace("1e3201f4203f7ae42da6ed98648b13c215ce8fcf8d7b96d6dd356f9d36fdad97", manifest_sha)
    source = source.replace("31fb466d31a734ffcd31e82bc7826b039821e5bc368e5f09662caba64530796e", archive_sha)
    source = source.replace('"JAX_PLATFORMS": "cpu"', '"JAX_PLATFORMS": "tpu"')
    source = source.replace('"OMP_NUM_THREADS": "1"', '"OMP_NUM_THREADS": "8"')
    source = source.replace('"MKL_NUM_THREADS": "1"', '"MKL_NUM_THREADS": "8"')
    source = source.replace('"OPENBLAS_NUM_THREADS": "1"', '"OPENBLAS_NUM_THREADS": "8"')
    source = source.replace('"NUMEXPR_NUM_THREADS": "1"', '"NUMEXPR_NUM_THREADS": "8"')
    source = source.replace("intra_op_parallelism_threads=1", "intra_op_parallelism_threads=8")
    source = source.replace('"--backend", "cpu"', '"--backend", "tpu"')
    source = source.replace('"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",', '"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",\n        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",\n        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.8",')
    start = source.index("def validate_cpu_host(output: Path) -> tuple[bool, dict]:")
    end = source.index("\ndef main() -> int:", start)
    helper = '''def validate_tpu_host(output: Path) -> tuple[bool, dict]:
    host_bytes = mem_total_bytes()
    cgroup = cgroup_memory_limit()
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    parent_rss = own_rss_bytes()
    physical_ceiling = int(host_bytes * 0.8) if host_bytes else 0
    limits = [physical_ceiling] if physical_ceiling else []
    if cgroup["limit_bytes"] is not None:
        limits.append(cgroup["limit_bytes"])
    effective = min(limits) if limits else 0
    selected = affinity[:8]
    required = RSS_LIMIT_BYTES + LAUNCHER_RESERVE_BYTES
    record = {"host_mem_total_bytes": host_bytes, "host_reserve_fraction": 0.20,
              "host_memory_ceiling_after_reserve_bytes": physical_ceiling,
              "cgroup_memory": cgroup, "effective_memory_ceiling_bytes": effective,
              "aggregate_rss_stop_bytes": RSS_LIMIT_BYTES,
              "launcher_reserve_bytes": LAUNCHER_RESERVE_BYTES,
              "required_memory_budget_bytes": required,
              "launcher_current_rss_bytes": parent_rss,
              "available_cpu_affinity": affinity, "selected_cpu_affinity": selected,
              "selected_cpu_count": len(selected), "backend_requested_before_jax_import": "tpu",
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "guard_passed": bool(host_bytes and cgroup["path"] and len(selected) == 8
                  and effective >= required and parent_rss < LAUNCHER_RESERVE_BYTES
                  and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
                  and os.environ.get("JAX_PLATFORMS") == "tpu")}
    write_json(output / "host_resource_guard.json", record)
    return record["guard_passed"], record
'''
    source = source[:start] + helper + source[end:]
    old = '''    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        if allowed:
            os.sched_setaffinity(0, {allowed[0]})
    guard_ok, guard = validate_cpu_host(OUTPUT)
'''
    new = '''    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        if len(allowed) >= 8:
            os.sched_setaffinity(0, set(allowed[:8]))
    guard_ok, guard = validate_tpu_host(OUTPUT)
'''
    if source.count(old) != 1:
        raise ValueError("kernel affinity/resource guard block changed")
    source = source.replace(old, new, 1)
    source = source.replace('os.environ["WNN_PREFLIGHT_MILESTONE"] = str(OUTPUT / "phase_milestones.jsonl")',
                            'os.environ["WNN_PREFLIGHT_MILESTONE"] = str(OUTPUT / "phase_milestones.jsonl")\n    os.environ["WNN_TPU_MEMORY_SAMPLES"] = str(OUTPUT / "tpu_memory_samples.jsonl")')
    source = source.replace('if runtime["jax_backend"] != "cpu" or len(runtime["jax_devices"]) != 1:',
                            'if runtime["jax_backend"] != "tpu" or len(runtime["jax_devices"]) != 8 or any("TPU v5 lite" not in d["kind"] for d in runtime["jax_devices"]):')
    source = source.replace('"classification": "bounded real-data JAX CPU preflight; no TPU/HBM or scientific gain claim"',
                            '"classification": "bounded one-update real-data JAX TPU systems preflight; no fitted model or scientific gain claim"')
    source = source.replace('"CPU result does not validate TPU execution or HBM",',
                            '"one TPU v5 lite device performs the diagnostic update; all eight are inventoried and monitored",')
    return source


def build() -> dict:
    HERE.mkdir(parents=True, exist_ok=True)
    for name in ("upload_dataset", "kernel_push", "source_bundle"):
        (HERE / name).mkdir(exist_ok=True)
    upload = HERE / "upload_dataset"; kernel_dir = HERE / "kernel_push"; source_dir = HERE / "source_bundle"
    prior_manifest = json.loads((BASE / "upload_dataset/source-manifest.json").read_text())
    prior_spec = json.loads((BASE / "bundle_spec.json").read_text())
    payload: dict[str, bytes] = {}
    with zipfile.ZipFile(BASE / "upload_dataset/source_bundle.zip") as archive:
        if set(archive.namelist()) != set(prior_manifest["files"]):
            raise ValueError("E463 source archive differs from frozen manifest")
        for name in archive.namelist(): payload[name] = archive.read(name)
    payload["scripts/realdata_preflight.py"] = patch_preflight(payload["scripts/realdata_preflight.py"].decode()).encode()
    for name, data in payload.items():
        target = source_dir / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    archive_path = upload / "source_bundle.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 4, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    archive_sha = sha(archive_path.read_bytes())
    manifest = json.loads(json.dumps(prior_manifest))
    manifest.update({"attempt": 1, "evidence_id": "E-463-TPU-preflight", "dataset": f"{OWNER}/{SLUG}", "archive_sha256": archive_sha,
        "private": True, "internet": False, "gpu": False, "tpu": True,
        "classification": "bounded one-update real-data JAX TPU systems preflight; diagnostic only; no fitted model or scientific gain claim",
        "resource_contract": {"aggregate_rss_limit_bytes": RSS_LIMIT, "launcher_reserve_bytes": LAUNCHER_RESERVE,
            "host_reserve_fraction": 0.20, "provider_timeout_seconds": PROVIDER_TIMEOUT,
            "whole_kernel_runtime_limit_seconds": MAX_RUNTIME, "cpu_affinity_max": 8, "cpu_threads_max": 8,
            "backend": "TPU", "visible_tpu_devices": 8, "selected_compute_tpu_devices": 1,
            "tpu_device_kind": "TPU v5 lite", "allocator_target_fraction": 0.80,
            "per_device_observed_hbm_hard_stop_fraction": 0.90, "memory_stats_interval_seconds": 5,
            "internet": False, "gpu": False, "tpu": True, "cuda_hidden": True}})
    manifest["files"]["scripts/realdata_preflight.py"] = {"sha256": sha(payload["scripts/realdata_preflight.py"]), "size_bytes": len(payload["scripts/realdata_preflight.py"]),
        "source_path": "research/campaign_2026-10-04/kaggle/jax_tpu_realdata_preflight_v1/source_bundle/scripts/realdata_preflight.py"}
    manifest_bytes = dump(manifest); manifest_sha = sha(manifest_bytes)
    (upload / "source-manifest.json").write_bytes(manifest_bytes)
    (upload / "dataset-metadata.json").write_bytes(dump({"id": f"{OWNER}/{SLUG}", "title": SLUG, "isPrivate": True, "licenses": [{"name": "other"}]}))
    kernel = patch_kernel((BASE / "kernel_push/kernel.py").read_text(), manifest_sha, archive_sha)
    (kernel_dir / "kernel.py").write_text(kernel)
    metadata = json.loads((BASE / "kernel_push/kernel-metadata.json").read_text())
    metadata.update({"id": f"{OWNER}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "enable_gpu": False,
        "enable_tpu": True, "enable_internet": False, "machine_shape": "TpuV5E8",
        "dataset_sources": [f"{OWNER}/{SLUG}", *metadata["dataset_sources"][1:]]})
    (kernel_dir / "kernel-metadata.json").write_bytes(dump(metadata))
    spec = json.loads(json.dumps(prior_spec))
    spec.update({"attempt": 1, "evidence_id": "E-463-TPU-preflight", "dataset_slug": f"{OWNER}/{SLUG}",
        "dataset_archive_sha256": archive_sha, "manifest_sha256": manifest_sha, "kernel_slug": f"{OWNER}/{KERNEL_SLUG}",
        "kernel_code_path": "kernel_push/kernel.py", "kernel_code_sha256": sha((kernel_dir / "kernel.py").read_bytes()),
        "kernel_metadata_sha256": sha((kernel_dir / "kernel-metadata.json").read_bytes()),
        "private": True, "gpu": False, "tpu": True, "internet": False, "status": "prepared_not_uploaded_or_launched",
        "resources": {"backend": "TPU", "visible_tpu_devices": 8, "selected_compute_tpu_devices": 1,
            "tpu_device_kind": "TPU v5 lite", "aggregate_rss_stop_bytes": RSS_LIMIT, "launcher_reserve_bytes": LAUNCHER_RESERVE,
            "host_reserve_fraction": 0.20, "cpu_affinity_max": 8, "cpu_threads_max": 8, "allocator_target_fraction": 0.80,
            "per_device_observed_hbm_hard_stop_fraction": 0.90, "memory_stats_interval_seconds": 5,
            "provider_timeout_seconds": PROVIDER_TIMEOUT, "whole_kernel_runtime_limit_seconds": MAX_RUNTIME,
            "internet": False, "gpu": False, "tpu": True},
        "protocol": {"backend": "tpu", "batch_size": 4, "chunk_rows": 1000, "precision_arms": ["fp32", "bf16"],
            "updates_per_arm": 1, "expected_fit_groups": [["valid", 1], ["valid", 0], ["train", 529], ["train", 0]],
            "expected_group_batch_sha256": "65918af8c55326923a96878d800e5c35ab3e891700af9139d74f4ca03db274b3",
            "same_scientific_computation_as": "E-463 CPU v3; CPU result is not an eligibility condition",
            "one_device_executes_update": True, "all_eight_devices_inventoried_and_memory_monitored": True,
            "diagnostic_only": True, "no_fit_or_scientific_gain_claim": True}})
    (HERE / "bundle_spec.json").write_bytes(dump(spec))
    packet = {"evidence_id": "E-463-TPU-preflight", "attempt": 1, "dataset": f"{OWNER}/{SLUG}", "kernel": f"{OWNER}/{KERNEL_SLUG}",
        "archive_sha256": archive_sha, "manifest_sha256": manifest_sha, "kernel_sha256": spec["kernel_code_sha256"],
        "kernel_metadata_sha256": spec["kernel_metadata_sha256"], "source_file_count": len(payload),
        "external_input_count": len(manifest["external_inputs"]), "source_preflight_sha256": manifest["files"]["scripts/realdata_preflight.py"]["sha256"],
        "provider_timeout_seconds": PROVIDER_TIMEOUT, "whole_kernel_runtime_limit_seconds": MAX_RUNTIME,
        "nominal_max_tpu_quota_seconds": PROVIDER_TIMEOUT, "aggregate_rss_stop_bytes": RSS_LIMIT,
        "launcher_reserve_bytes": LAUNCHER_RESERVE, "host_reserve_fraction": 0.20, "cpu_affinity_max": 8,
        "visible_tpu_devices": 8, "compute_tpu_devices": 1, "allocator_target_fraction": 0.80,
        "observed_hbm_stop_fraction": 0.90, "memory_sampling_interval_seconds": 5,
        "scientific_computation_unchanged": True, "actual_tpu_training_or_gain_claim": False, "pushed": False}
    (HERE / "review_packet.json").write_bytes(dump(packet))
    ast.parse((kernel_dir / "kernel.py").read_text()); ast.parse((source_dir / "scripts/realdata_preflight.py").read_text())
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == set(manifest["files"])
        for name, data in payload.items(): assert sha(archive.read(name)) == sha(data)
    (HERE / "validation.prepush.json").write_bytes(dump({"status": "prepared_not_uploaded_or_launched", "python_ast_parses": True,
        "archive_members_match_manifest": True, "archive_sha256": archive_sha, "manifest_sha256": manifest_sha,
        "kernel_sha256": spec["kernel_code_sha256"], "kernel_metadata_sha256": spec["kernel_metadata_sha256"]}))
    return packet


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
