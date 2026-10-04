from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import resource
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

KERNEL_ID = "vladkochetov07/wnn-w384-io-diagnostic-cpu-20261004-v2-runner"
BUNDLE_SLUG = "wnn-w384-fold1-dualt4-20261004-v3"
EXPECTED_BUNDLE_SHA256 = "cfaed9a32399348d5bb1a13b817cd219c69b0f72adb03de7496735a173094a05"
EXPECTED_BUNDLE_MANIFEST_SHA256 = "890fe16b96e65852fca1ea418f5bf125694b253630ee41227d5d9d2e6ad1abcf"
OUTPUT = Path("/kaggle/working/w384_io_diagnostic_v1")
WORKER_LIMIT_SECONDS = 3500
JOB_LIMIT_SECONDS = 3600
RSS_LIMIT_BYTES = 4 * 1024**3
HASH_BLOCK_BYTES = 8 * 1024**2


def now_ns() -> int:
    return time.time_ns()


def sha256_file(path: Path, *, block_bytes: int, deadline: float, on_progress=None) -> tuple[str, int, float]:
    started = time.monotonic()
    digest = hashlib.sha256()
    bytes_read = 0
    next_progress = 1 << 30
    with path.open("rb") as stream:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("diagnostic wall budget reached during file hash")
            if rss_self() > RSS_LIMIT_BYTES:
                raise MemoryError("diagnostic RSS limit reached during file hash")
            block = stream.read(block_bytes)
            if not block:
                break
            digest.update(block)
            bytes_read += len(block)
            if bytes_read >= next_progress and on_progress is not None:
                on_progress(bytes_read)
                next_progress += 1 << 30
    return digest.hexdigest(), bytes_read, time.monotonic() - started


def rss_self() -> int:
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def group_rss(group_id: int) -> int:
    total = 0
    page_size = os.sysconf("SC_PAGE_SIZE")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].strip().split()
            if int(fields[2]) == group_id:
                total += int((entry / "statm").read_text().split()[1]) * page_size
        except (OSError, ValueError, IndexError):
            continue
    return total


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def event(path: Path, phase: str, **fields) -> None:
    row = {"utc_ns": now_ns(), "monotonic_ns": time.monotonic_ns(), "phase": phase, **fields}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def file_stat(path: Path) -> dict:
    st = path.stat()
    lst = path.lstat()
    vfs = os.statvfs(path)
    return {
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "is_symlink": path.is_symlink(),
        "symlink_target": os.readlink(path) if path.is_symlink() else None,
        "device": st.st_dev,
        "inode": st.st_ino,
        "size_bytes": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "ctime_ns": st.st_ctime_ns,
        "lstat_device": lst.st_dev,
        "lstat_inode": lst.st_ino,
        "readonly_mount_flag": bool(vfs.f_flag & getattr(os, "ST_RDONLY", 1)),
        "filesystem_free_bytes": vfs.f_bavail * vfs.f_frsize,
        "filesystem_total_bytes": vfs.f_blocks * vfs.f_frsize,
    }


def safe_extract_bundle(archive: Path, manifest: dict, destination: Path) -> None:
    external = set(manifest["external_mounts"])
    expected = {item["path"]: item for item in manifest["files"] if item["path"] not in external}
    expected.update({name: {"path": name} for name in manifest["spec_paths"]})
    seen = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ValueError("unsafe path in frozen experiment bundle")
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError("undeclared, unsupported, or duplicate bundle member")
            seen.add(member.name)
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError("bundle member has no stream")
            h = hashlib.sha256()
            size = 0
            with source, target.open("wb") as output:
                while True:
                    block = source.read(1 << 20)
                    if not block:
                        break
                    h.update(block)
                    output.write(block)
                    size += len(block)
            desc = expected[member.name]
            if desc.get("size", size) != size or desc.get("sha256", h.hexdigest()) != h.hexdigest():
                raise ValueError(f"bundle member identity mismatch: {member.name}")
    if seen != set(expected):
        raise ValueError(f"bundle member set mismatch: missing={sorted(set(expected)-seen)}")


def mount_external_inputs(root: Path, mounts: dict, created_links: list[tuple[Path, Path]]) -> dict[str, Path]:
    bound = {}
    for logical, mapping in mounts.items():
        dataset = mapping["dataset"] if isinstance(mapping, dict) else "vladkochetov07/wunder-funder"
        filename = mapping["filename"] if isinstance(mapping, dict) else mapping
        slug = dataset.split("/", 1)[1]
        candidates = list(Path("/kaggle/input").glob(f"**/{slug}/{filename}"))
        if len(candidates) != 1:
            raise RuntimeError(f"expected exactly one mounted file for {dataset}/{filename}; found {len(candidates)}")
        target = root / logical
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise RuntimeError(f"refusing to replace pre-existing experiment path {logical}")
        target.symlink_to(candidates[0])
        created_links.append((target, candidates[0]))
        bound[logical] = candidates[0]
    return bound


def cleanup_mount_symlinks(created_links: list[tuple[Path, Path]], log: Path) -> list[str]:
    removed = []
    for target, candidate in reversed(created_links):
        outcome = "preserved_identity_changed"
        try:
            if target.is_symlink() and target.resolve(strict=True) == candidate.resolve(strict=True):
                target.unlink()
                removed.append(str(target))
                outcome = "removed_diagnostic_created_link"
            elif not target.exists() and not target.is_symlink():
                outcome = "already_absent"
        except FileNotFoundError:
            outcome = "already_absent"
        event(log, "mount_symlink_cleanup", target=str(target), candidate=str(candidate), outcome=outcome)
    return removed


def controller_verify(root: Path, specs: list[dict], log: Path, deadline: float, records: list[dict]) -> list[dict]:
    for spec in specs:
        for category, descriptors in (("code", spec["code"]), ("input", spec["inputs"])):
            for item in descriptors:
                path = root / item["path"]
                before = file_stat(path)
                event(log, "controller_hash_start", spec=spec["name"], category=category, relative_path=item["path"], expected_bytes=item["size"], stat_before=before)
                def progress(n):
                    event(log, "controller_hash_progress", spec=spec["name"], relative_path=item["path"], bytes_read=n, rss_bytes=rss_self())
                digest, size, elapsed = sha256_file(path, block_bytes=HASH_BLOCK_BYTES, deadline=deadline, on_progress=progress)
                after = file_stat(path)
                ok = digest == item["sha256"] and size == item["size"]
                row = {"spec":spec["name"],"category":category,"relative_path":item["path"],"size_bytes":size,"sha256":digest,"expected_sha256":item["sha256"],"elapsed_seconds":elapsed,"throughput_bytes_per_second":size/max(elapsed,1e-9),"verified":ok,"stat_before":before,"stat_after":after,"rss_after_bytes":rss_self()}
                event(log, "controller_hash_finish", **row)
                records.append(row)
                if not ok:
                    raise ValueError(f"dependency hash mismatch: {item['path']}")
    return records


def child_portable(root_value: str, split_value: str, registry_value: str, name: str, log_value: str, cpu: int, deadline: float) -> int:
    root = Path(root_value)
    log = Path(log_value)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})
    os.environ.update({"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1","WNN_PORTABLE_DATA_MANIFEST":registry_value,"PYTHONDONTWRITEBYTECODE":"1"})
    event(log, "child_start", worker=name, cpu_affinity=sorted(os.sched_getaffinity(0)), rss_bytes=rss_self())
    started = time.monotonic()
    event(log, "torch_import_start", worker=name, rss_bytes=rss_self())
    import torch
    cuda_initialized = bool(torch.cuda.is_initialized())
    cuda_available = bool(torch.cuda.is_available())
    event(log, "torch_import_finish", worker=name, elapsed_seconds=time.monotonic()-started, rss_bytes=rss_self(), torch_version=torch.__version__, cuda_initialized=cuda_initialized, cuda_available=cuda_available, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    sys.path.insert(0, str(root / "src"))
    import wnn.portability as portability
    import wnn.splits as splits
    portability._content_digest.cache_clear()
    registry = json.loads(Path(registry_value).read_text())
    expected_by_resolved_path = {}
    for logical, descriptor in registry["sources"].items():
        expected_by_resolved_path[str((root / logical).resolve())] = descriptor.get("sha256")
    def timed_hash(path: Path) -> str:
        source = Path(path)
        event(log, "portable_hash_start", worker=name, path=str(source), stat_before=file_stat(source), rss_bytes=rss_self())
        digest, size, elapsed = sha256_file(source, block_bytes=1 << 20, deadline=deadline, on_progress=lambda n:event(log,"portable_hash_progress",worker=name,path=str(source),bytes_read=n,rss_bytes=rss_self()))
        expected = expected_by_resolved_path.get(str(source.resolve()))
        event(log, "portable_hash_finish", worker=name, path=str(source), bytes_read=size, sha256=digest, expected_sha256=expected, elapsed_seconds=elapsed, throughput_bytes_per_second=size/max(elapsed,1e-9), verified=(expected is None or digest==expected), stat_after=file_stat(source), rss_bytes=rss_self())
        if expected is not None and digest != expected:
            raise ValueError(f"portable hash mismatch: {source}")
        return digest
    portability.sha256_file = timed_hash
    splits.file_sha256 = timed_hash
    load_started = time.monotonic()
    cpu_before = resource.getrusage(resource.RUSAGE_SELF)
    event(log, "load_campaign_split_start", worker=name, split=split_value, registry=registry_value, rss_bytes=rss_self())
    split = splits.load_campaign_split(Path(split_value), root)
    cpu_after = resource.getrusage(resource.RUSAGE_SELF)
    event(log, "load_campaign_split_finish", worker=name, elapsed_seconds=time.monotonic()-load_started, user_cpu_seconds=cpu_after.ru_utime-cpu_before.ru_utime, system_cpu_seconds=cpu_after.ru_stime-cpu_before.ru_stime, fold=split.fold, train_groups=len(split.train_groups), valid_groups=len(split.valid_groups), evaluation_groups=len(split.evaluation_groups), rss_bytes=rss_self())
    if time.monotonic() >= deadline:
        raise TimeoutError("portable validation passed deadline")
    return 0


def child_entry() -> int:
    argv = sys.argv
    log = Path(argv[6]) if len(argv) > 6 else Path("/tmp/w384_io_diagnostic_child_error.jsonl")
    name = argv[5] if len(argv) > 5 else "unknown"
    try:
        if len(argv) != 9 or argv[1] != "--portable-child":
            raise ValueError("invalid portable-child argv shape")
        _, _, root, split, registry, name, log_value, cpu, deadline = argv
        log = Path(log_value)
        return child_portable(root, split, registry, name, str(log), int(cpu), float(deadline))
    except BaseException as exc:
        event(Path(log), "child_failed", worker=name, error_type=type(exc).__name__, error=str(exc), rss_bytes=rss_self())
        return 70


def discover_version_markers() -> list[str]:
    markers=[]
    for root in Path("/kaggle/input").glob("**"):
        if root.is_dir() and root.name in {".dataset-metadata","dataset-metadata"}:
            markers.append(str(root))
    return markers[:20]


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    log=OUTPUT/"events.jsonl"
    started=time.monotonic(); deadline=started+WORKER_LIMIT_SECONDS
    if hasattr(os,"sched_setaffinity"):
        allowed=sorted(os.sched_getaffinity(0))
        if len(allowed)<2: raise RuntimeError("CPU diagnostic requires two allowed CPUs")
        os.sched_setaffinity(0,set(allowed[:2]))
    os.environ.update({"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1","PYTHONDONTWRITEBYTECODE":"1"})
    record={"kernel_id":KERNEL_ID,"started_utc_ns":now_ns(),"cpu_affinity":sorted(os.sched_getaffinity(0)),"cpu_count":os.cpu_count(),"memory_total_bytes":None,"rss_cap_bytes":RSS_LIMIT_BYTES,"worker_limit_seconds":WORKER_LIMIT_SECONDS,"job_limit_seconds":JOB_LIMIT_SECONDS,"torch_import":None,"controller_hashes":[],"portable_workers":[],"receipt_reuse_candidate":None}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            record["memory_total_bytes"]=int(line.split()[1])*1024; break
    cgroup=Path("/sys/fs/cgroup/memory.max")
    record["cgroup_memory_max"]=cgroup.read_text().strip() if cgroup.exists() else None
    record["dataset_version_markers_observed"]=discover_version_markers()
    record["dataset_versions_exactly_observable"]=False
    event(log,"host_preflight",cpu_affinity=record["cpu_affinity"],memory_total_bytes=record["memory_total_bytes"],cgroup_memory_max=record["cgroup_memory_max"],gpu_hidden=True,tpu_enabled=False,internet_enabled=False,dataset_version_markers=record["dataset_version_markers_observed"])
    created_mount_links: list[tuple[Path, Path]] = []
    try:
        found=list(Path("/kaggle/input").glob(f"**/{BUNDLE_SLUG}/bundle-manifest.json"))
        if len(found)!=1: raise RuntimeError(f"expected one W384 v3 bundle manifest, found {len(found)}")
        manifest_path=found[0]; dataset_root=manifest_path.parent
        manifest_digest,manifest_size,manifest_elapsed=sha256_file(manifest_path,block_bytes=HASH_BLOCK_BYTES,deadline=deadline)
        manifest=json.loads(manifest_path.read_text()); archive=dataset_root/"bundle.wnn"
        bundle_hash,size,elapsed=sha256_file(archive,block_bytes=HASH_BLOCK_BYTES,deadline=deadline)
        event(log,"bundle_manifest_verified",path=str(manifest_path),size_bytes=manifest_size,sha256=manifest_digest,elapsed_seconds=manifest_elapsed,expected_sha256=EXPECTED_BUNDLE_MANIFEST_SHA256,verified=manifest_digest==EXPECTED_BUNDLE_MANIFEST_SHA256,stat=file_stat(manifest_path))
        event(log,"bundle_archive_verified",path=str(archive),size_bytes=size,sha256=bundle_hash,elapsed_seconds=elapsed,expected_sha256=manifest.get("archive_sha256"),verified=(bundle_hash==EXPECTED_BUNDLE_SHA256==manifest.get("archive_sha256")),stat=file_stat(archive))
        if manifest_digest!=EXPECTED_BUNDLE_MANIFEST_SHA256 or manifest.get("private") is not True or bundle_hash!=EXPECTED_BUNDLE_SHA256 or manifest.get("archive_sha256")!=EXPECTED_BUNDLE_SHA256:
            raise ValueError("frozen private W384 bundle identity mismatch")
        experiment=OUTPUT/"experiment"; experiment.mkdir()
        safe_extract_bundle(archive,manifest,experiment)
        mounts=mount_external_inputs(experiment,manifest["external_mounts"],created_mount_links)
        split_logical="research/campaign_2026-09-29/artifacts/splits/fold1.json"
        registry_logical="research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v3/portable_inputs_f1.json"
        split_path=experiment/split_logical; registry_path=experiment/registry_logical
        specs=[]
        for width in (256,384):
            spec_path=experiment/f"kaggle_specs/c20261004_w{width}_adamw_s3_f1_t4v3_probe.json"
            specs.append(json.loads(spec_path.read_text()))
        record["mounted_inputs"]={k:{"path":str(v),"stat":file_stat(v)} for k,v in mounts.items()}
        # Profile trainer import and portable source validation first so the
        # bounded run records those boundaries even if the controller's exact
        # full-file dependency pass consumes the remaining wall budget.
        children=[]
        cpus=record["cpu_affinity"][:2]
        names=["c20261004_w256_adamw_s3_f1_t4v3_probe","c20261004_w384_adamw_s3_f1_t4v3_probe"]
        for name,cpu in zip(names,cpus,strict=True):
            child_log=OUTPUT/f"{name}_portable.jsonl"
            cmd=[sys.executable,str(Path(__file__).resolve()),"--portable-child",str(experiment),str(split_path),str(registry_path),name,str(child_log),str(cpu),str(deadline)]
            proc=subprocess.Popen(cmd,start_new_session=True,env=os.environ.copy(),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            children.append({"name":name,"pid":proc.pid,"pgrp":proc.pid,"cpu":cpu,"log":str(child_log),"process":proc})
        peak_rss=0; stop_reason=None
        while any(row["process"].poll() is None for row in children):
            if time.monotonic()-started>JOB_LIMIT_SECONDS-10 or time.monotonic()>deadline:
                stop_reason="wall_time_guard"; break
            total=rss_self()+sum(group_rss(row["pgrp"]) for row in children)
            peak_rss=max(peak_rss,total)
            event(log,"portable_worker_heartbeat",active=[{"name":r["name"],"pid":r["pid"],"cpu":r["cpu"],"returncode":r["process"].poll()} for r in children],aggregate_rss_bytes=total,peak_rss_bytes=peak_rss)
            if total>RSS_LIMIT_BYTES:
                stop_reason="aggregate_rss_guard"; break
            time.sleep(1.0)
        if stop_reason:
            for row in children:
                if row["process"].poll() is None:
                    try: os.killpg(row["pgrp"],signal.SIGTERM)
                    except ProcessLookupError: pass
            end=time.monotonic()+5
            while time.monotonic()<end and any(r["process"].poll() is None for r in children): time.sleep(.1)
            for row in children:
                if row["process"].poll() is None:
                    try: os.killpg(row["pgrp"],signal.SIGKILL)
                    except ProcessLookupError: pass
        for row in children:
            row["returncode"]=row["process"].wait()
            del row["process"]
        record["portable_workers"]=children
        record["aggregate_peak_rss_bytes"]=peak_rss
        record["stop_reason"]=stop_reason
        record["status"]="complete" if stop_reason is None and all(r["returncode"]==0 for r in children) else (stop_reason or "partial_or_failed")
        if stop_reason is None:
            try:
                controller_verify(experiment,specs,log,deadline,record["controller_hashes"])
            except TimeoutError as exc:
                record["status"]="time_budget_partial"
                record["stop_reason"]=str(exc)
        mounted_stats={logical:file_stat(path) for logical,path in mounts.items()}
        stable_fields=("resolved_path","device","inode","size_bytes","mtime_ns","ctime_ns")
        stable_full_reads=all(
            all(row["stat_before"][key]==row["stat_after"][key] for key in stable_fields)
            for row in record["controller_hashes"]
        )
        record["receipt_reuse_candidate"]={
            "full_integrity_hashes_completed":len(record["controller_hashes"])==sum(len(s["code"])+len(s["inputs"]) for s in specs),
            "all_input_mounts_readonly":all(v["readonly_mount_flag"] for v in mounted_stats.values()),
            "all_full_read_stats_stable":stable_full_reads,
            "mounted_input_stats_after_workers":mounted_stats,
            "dataset_version_exactly_observable":False,
            "eligible_to_skip_shared_portability_hashes":False,
            "reason":"Candidate must bind actual full-read hashes to exact readonly mount path/stat and provider dataset version; mounted version identity was not independently exposed. This diagnostic never changes or bypasses shared source verification.",
        }
        unique_inputs={item["path"]:item["size"] for spec in specs for item in spec["inputs"]}
        free_bytes=os.statvfs(OUTPUT).f_bavail*os.statvfs(OUTPUT).f_frsize
        record["copy_and_rehash_capacity_assessment"]={
            "unique_declared_input_bytes":sum(unique_inputs.values()),
            "working_filesystem_free_bytes":free_bytes,
            "required_free_bytes_with_20pct_headroom":int(sum(unique_inputs.values())/0.8),
            "copy_feasible_by_capacity":free_bytes>=int(sum(unique_inputs.values())/0.8),
            "copy_performed":False,
            "note":"Capacity-only assessment; no input files were copied or modified.",
        }
    except TimeoutError as exc:
        record["status"]="time_budget_partial"; record["stop_reason"]=str(exc)
    except BaseException as exc:
        record["status"]="failed"; record["failure"]={"type":type(exc).__name__,"detail":str(exc)}
    finally:
        try:
            record["removed_mount_symlinks"]=cleanup_mount_symlinks(created_mount_links,log)
        except BaseException as exc:
            record["mount_symlink_cleanup_failure"]={"type":type(exc).__name__,"detail":str(exc)}
    record["elapsed_seconds"]=time.monotonic()-started
    write_json(OUTPUT/"summary.json",record)
    event(log,"job_terminal",status=record.get("status"),elapsed_seconds=record["elapsed_seconds"],stop_reason=record.get("stop_reason"))
    return 0


if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1]=="--portable-child":
        raise SystemExit(child_entry())
    raise SystemExit(main())
