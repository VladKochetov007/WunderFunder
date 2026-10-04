from __future__ import annotations
import ast, hashlib, json, zipfile, re
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT/'research/campaign_2026-10-04/kaggle/jax_realdata_preflight_v2'
HERE = ROOT/'research/campaign_2026-10-04/kaggle/jax_realdata_preflight_v3'
SLUG = 'wnn-jax-realdata-e463-20261004-v3'
OWNER = 'vladkochetov07'


def sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def dump(value: object) -> bytes: return (json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
def wstr(value: str) -> bytes: return value.encode()


def instrument_preflight(source: str) -> str:
    # Diagnostics are additive; the bounded AST harness strips only this helper and its calls
    # and compares the remaining computational AST to the frozen v2 source.
    helper = '''\n\ndef _emit_milestone(phase: str, **details: Any) -> None:\n    """Persist a flushed process/RSS checkpoint for fail-closed CPU preflight diagnostics."""\n    milestone_path = os.environ.get("WNN_PREFLIGHT_MILESTONE")\n    if not milestone_path:\n        return\n    current_rss = 0\n    try:\n        current_rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")\n    except (OSError, ValueError, IndexError):\n        pass\n    record = {\n        "phase": phase,\n        "wall_time_epoch_seconds": time.time(),\n        "elapsed_seconds": time.monotonic() - _MILESTONE_STARTED,\n        "current_process_rss_bytes": current_rss,\n        "ru_maxrss_platform_units": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),\n        "ru_maxrss_unit": "KiB on Linux",\n        **details,\n    }\n    with open(milestone_path, "a", encoding="utf-8") as stream:\n        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\\n")\n        stream.flush()\n        os.fsync(stream.fileno())\n\n\n_MILESTONE_STARTED = time.monotonic()\n_emit_milestone("realdata_import_start")\n'''
    marker = 'import numpy as np\n'
    assert source.count(marker)==1
    source = source.replace(marker, helper+'\n'+marker+'\n_emit_milestone("realdata_import_complete")\n',1)
    def once(old: str,new: str) -> None:
        nonlocal source
        assert source.count(old)==1, f'expected one marker: {old[:80]!r}, got {source.count(old)}'
        source=source.replace(old,new,1)
    once('    import jax\n', '    _emit_milestone("jax_import_start")\n    import jax\n')
    once('    from wnn_jax.rnn import adamw_init, export_npz_state, forward_chunk, load_npz_state\n', '    from wnn_jax.rnn import adamw_init, export_npz_state, forward_chunk, load_npz_state\n    _emit_milestone("jax_and_bridge_import_complete")\n')
    once('    export_sidecar = _verify_initial_export(args.initial_npz, args.initial_checkpoint)\n', '    _emit_milestone("initial_checkpoint_hash_verification_start")\n    export_sidecar = _verify_initial_export(args.initial_npz, args.initial_checkpoint)\n    _emit_milestone("initial_checkpoint_hash_verification_complete")\n')
    once('    groups = load_fold_groups(args.split_manifest)\n', '    _emit_milestone("fold_split_load_start")\n    groups = load_fold_groups(args.split_manifest)\n    _emit_milestone("fold_split_load_complete", split_manifest_sha256=groups.manifest_sha256)\n')
    once('    params = load_npz_state(args.initial_npz, *dimensions)\n', '    _emit_milestone("model_parameter_initialization_start")\n    params = load_npz_state(args.initial_npz, *dimensions)\n    _emit_milestone("model_parameter_initialization_complete")\n')
    once('    store = RawSequenceStore(root)\n', '    _emit_milestone("raw_data_store_initialization_start")\n    store = RawSequenceStore(root)\n    _emit_milestone("raw_data_store_initialization_complete")\n')
    once('    fit_samples = [\n', '    _emit_milestone("fit_sequence_and_target_load_start", expected_groups=[list(group) for group in ordered_fit])\n    fit_samples = [\n')
    once('    fit_host = collate_sequences(fit_samples)\n', '    _emit_milestone("fit_sequence_and_target_load_complete")\n    fit_host = collate_sequences(fit_samples)\n    _emit_milestone("fit_batch_collation_complete")\n')
    once('    eval_samples = [\n', '    _emit_milestone("evaluation_sequence_load_start", groups=list(selected_eval))\n    eval_samples = [\n')
    once('    eval_batch = {\n', '    _emit_milestone("evaluation_sequence_load_complete")\n    eval_batch = {\n')
    once('    if not eval_mask.any():\n', '    _emit_milestone("evaluation_mask_build_complete", scored_rows=int(eval_mask.sum()))\n    if not eval_mask.any():\n')
    once('        updated, optimizer, _, _, loss = train_combo_chunk(\n', '        _emit_milestone(f"{precision}_gradient_update_start")\n        updated, optimizer, _, _, loss = train_combo_chunk(\n')
    once('        loss = jax.block_until_ready(loss)\n', '        loss = jax.block_until_ready(loss)\n        _emit_milestone(f"{precision}_gradient_update_complete")\n')
    once('        npz_path = output / f"model_{precision}.npz"\n', '        _emit_milestone(f"{precision}_finite_loss_and_optimizer_checks_complete", train_loss=loss_value, adamw_steps=step_count)\n        npz_path = output / f"model_{precision}.npz"\n')
    once('        prediction, _ = forward_chunk(\n', '        _emit_milestone(f"{precision}_evaluation_forward_start")\n        prediction, _ = forward_chunk(\n')
    once('        arm_metrics[precision] = {\n', '        _emit_milestone(f"{precision}_evaluation_complete", scored_rows=int(score_mask.sum()), evaluation_score=evaluation_score)\n        arm_metrics[precision] = {\n')
    once('    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n")\n', '    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n")\n    _emit_milestone("preflight_metrics_written", metrics_path=str(report_path))\n')
    return source


def make_resource_helpers() -> str:
    return """
LAUNCHER_RESERVE_BYTES = 2 * 1024**3

def mem_total_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None

def cgroup_memory_limit() -> dict:
    candidates = []
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            hierarchy, controllers, rel = line.split(":", 2)
            rel_path = PurePosixPath(rel.lstrip("/"))
            if hierarchy == "0" and not controllers:
                candidates.extend([Path("/sys/fs/cgroup") / rel_path / "memory.max", Path("/sys/fs/cgroup/memory.max")])
            if "memory" in controllers.split(","):
                candidates.extend([Path("/sys/fs/cgroup/memory") / rel_path / "memory.limit_in_bytes", Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")])
    except (OSError, ValueError):
        pass
    candidates.extend([Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")])
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            raw = candidate.read_text().strip()
        except OSError:
            continue
        if raw == "max":
            return {"path": str(candidate), "raw": raw, "limit_bytes": None, "unlimited": True}
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0:
            continue
        unlimited = value >= 1 << 60
        return {"path": str(candidate), "raw": raw, "limit_bytes": None if unlimited else value, "unlimited": unlimited}
    return {"path": None, "raw": None, "limit_bytes": None, "unlimited": None}

def validate_cpu_host(output: Path) -> tuple[bool, dict]:
    host_bytes = mem_total_bytes()
    cgroup = cgroup_memory_limit()
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    parent_rss = own_rss_bytes()
    physical_ceiling = int(host_bytes * 0.8) if host_bytes else 0
    limits = [physical_ceiling] if physical_ceiling else []
    if cgroup["limit_bytes"] is not None:
        limits.append(cgroup["limit_bytes"])
    effective = min(limits) if limits else 0
    required = RSS_LIMIT_BYTES + LAUNCHER_RESERVE_BYTES
    record = {
        "host_mem_total_bytes": host_bytes,
        "host_reserve_fraction": 0.20,
        "host_memory_ceiling_after_reserve_bytes": physical_ceiling,
        "cgroup_memory": cgroup,
        "effective_memory_ceiling_bytes": effective,
        "aggregate_rss_stop_bytes": RSS_LIMIT_BYTES,
        "launcher_reserve_bytes": LAUNCHER_RESERVE_BYTES,
        "required_memory_budget_bytes": required,
        "launcher_current_rss_bytes": parent_rss,
        "cpu_affinity": affinity,
        "required_cpu_count": 1,
        "backend_requested_before_jax_import": "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "guard_passed": bool(host_bytes and cgroup["path"] and len(affinity) == 1 and effective >= required and parent_rss < LAUNCHER_RESERVE_BYTES and os.environ.get("CUDA_VISIBLE_DEVICES") == ""),
    }
    write_json(output / "host_resource_guard.json", record)
    return record["guard_passed"], record
"""

def instrument_kernel(source: str, manifest_sha: str, archive_sha: str) -> str:
    source=source.replace('wnn-jax-realdata-e463-20261004-v2','wnn-jax-realdata-e463-20261004-v3')
    source=source.replace('jax_realdata_preflight_e463_v2','jax_realdata_preflight_e463_v3')
    source=source.replace('f8602ae4d2214164f0d7b5ec5c29bcbb6cc7b9d7b19fed686231cdb293a5a28a',manifest_sha)
    source=source.replace('b6954032fadd8c1d8f292f9e18e05ca7b28ff8dfd6c48cb05fc1aea1f946f7e7',archive_sha)
    source=source.replace('RSS_LIMIT_BYTES = 2 * 1024**3','RSS_LIMIT_BYTES = 4 * 1024**3')
    # The worker is embedded as a literal. Add its own flushed phase/RSS checkpoints.
    source=source.replace('import runpy\nimport sys\nimport time\n', 'import runpy\nimport resource\nimport sys\nimport time\n')
    worker_anchor='started = time.monotonic()\n\n\ndef write_json(path: Path, value: object) -> None:\n'
    worker_insert='''started = time.monotonic()\nMILESTONE_PATH = OUT / "phase_milestones.jsonl"\n\n\ndef milestone(phase: str, **details: object) -> None:\n    try:\n        rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")\n    except (OSError, ValueError, IndexError):\n        rss = 0\n    record = {"phase": phase, "elapsed_seconds": time.monotonic() - started, "wall_time_epoch_seconds": time.time(),\n              "current_process_rss_bytes": rss, "ru_maxrss_platform_units": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),\n              "ru_maxrss_unit": "KiB on Linux", **details}\n    with MILESTONE_PATH.open("a", encoding="utf-8") as stream:\n        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\\\\n")\n        stream.flush()\n        os.fsync(stream.fileno())\n\n\ndef write_json(path: Path, value: object) -> None:\n'''
    assert source.count(worker_anchor)==1
    source=source.replace(worker_anchor,worker_insert,1)
    source=source.replace('def main() -> int:\n    versions = {"python": sys.version}\n', 'def main() -> int:\n    milestone("worker_entry_before_dependency_probe")\n    versions = {"python": sys.version}\n')
    source=source.replace('    import jax\n    runtime["jax_backend"] = jax.default_backend()\n', '    milestone("jax_import_start")\n    import jax\n    milestone("jax_import_complete")\n    runtime["jax_backend"] = jax.default_backend()\n')
    source=source.replace('    started_run = time.monotonic()\n', '    milestone("bridge_import_and_backend_validation_complete", jax_backend=runtime["jax_backend"], device_count=len(runtime["jax_devices"]))\n    started_run = time.monotonic()\n')
    source=source.replace('    worker = OUTPUT / "worker.py"\n', '    os.environ["WNN_PREFLIGHT_MILESTONE"] = str(OUTPUT / "phase_milestones.jsonl")\n    worker = OUTPUT / "worker.py"\n')
    # Add conservative pre-JAX host/cgroup guard helpers.
    insert_at='def main() -> int:\n    overall_started = time.monotonic()\n'
    assert source.count(insert_at)==1
    source=source.replace(insert_at,make_resource_helpers()+'\n\n'+insert_at,1)
    main_anchor='''    if hasattr(os, "sched_getaffinity"):\n        allowed = sorted(os.sched_getaffinity(0))\n        if allowed:\n            os.sched_setaffinity(0, {allowed[0]})\n    manifests = list(Path("/kaggle/input").glob(f"**/{DATASET_SLUG}/source-manifest.json"))\n'''
    main_insert='''    if hasattr(os, "sched_getaffinity"):\n        allowed = sorted(os.sched_getaffinity(0))\n        if allowed:\n            os.sched_setaffinity(0, {allowed[0]})\n    guard_ok, guard = validate_cpu_host(OUTPUT)\n    if not guard_ok:\n        write_json(OUTPUT / "runner_result.json", {"status": "failed_host_resource_guard", "host_resource_guard": guard,\n                                                     "overall_runtime_limit_seconds": MAX_RUNTIME_SECONDS})\n        return 37\n    manifests = list(Path("/kaggle/input").glob(f"**/{DATASET_SLUG}/source-manifest.json"))\n'''
    assert source.count(main_anchor)==1
    source=source.replace(main_anchor,main_insert,1)
    return source


def main() -> None:
    dataset_dir=HERE/'upload_dataset'; kernel_dir=HERE/'kernel_push'; src_dir=HERE/'source_bundle'
    dataset_dir.mkdir(exist_ok=False); kernel_dir.mkdir(); src_dir.mkdir()
    old_manifest=json.loads((BASE/'upload_dataset/source-manifest.json').read_text())
    old_spec=json.loads((BASE/'bundle_spec.json').read_text())
    payload={}
    with zipfile.ZipFile(BASE/'upload_dataset/source_bundle.zip') as z:
        assert set(z.namelist())==set(old_manifest['files'])
        for name in z.namelist():payload[name]=z.read(name)
    old_preflight=payload['scripts/realdata_preflight.py'].decode()
    new_preflight=instrument_preflight(old_preflight)
    payload['scripts/realdata_preflight.py']=wstr(new_preflight)
    for name,data in payload.items():
        target=src_dir/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    # Stable ZIP member names/timestamps and file descriptors.
    archive=dataset_dir/'source_bundle.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(payload.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,10,4,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            z.writestr(info,data)
    archive_sha=sha(archive.read_bytes())
    manifest=json.loads(json.dumps(old_manifest))
    manifest.update({'attempt':3,'dataset':f'{OWNER}/{SLUG}','archive_sha256':archive_sha,
      'classification':'bounded real-data JAX CPU one-update FP32/BF16 preflight with host memory guard and flushed phase/RSS milestones; diagnostic only; no TPU validation or scientific gain claim; attempt 3'})
    manifest['resource_contract'].update({'aggregate_rss_limit_bytes':4*1024**3,'launcher_reserve_bytes':2*1024**3,
      'host_reserve_fraction':0.20,'provider_timeout_seconds':2100,'whole_kernel_runtime_limit_seconds':1800,
      'cpu_affinity':1,'backend':'CPU','internet':False,'gpu':False,'tpu':False})
    manifest['files']['scripts/realdata_preflight.py']={'sha256':sha(payload['scripts/realdata_preflight.py']),'size_bytes':len(payload['scripts/realdata_preflight.py']),'source_path':'research/campaign_2026-10-04/kaggle/jax_realdata_preflight_v3/source_bundle/scripts/realdata_preflight.py'}
    manifest_bytes=dump(manifest); manifest_sha=sha(manifest_bytes)
    (dataset_dir/'source-manifest.json').write_bytes(manifest_bytes)
    (dataset_dir/'dataset-metadata.json').write_bytes(dump({'id':f'{OWNER}/{SLUG}','title':SLUG,'isPrivate':True,'licenses':[{'name':'other'}]}))
    kernel=instrument_kernel((BASE/'kernel_push/kernel.py').read_text(),manifest_sha,archive_sha)
    (kernel_dir/'kernel.py').write_text(kernel)
    meta=json.loads((BASE/'kernel_push/kernel-metadata.json').read_text())
    meta.update({'id':f'{OWNER}/{SLUG}-runner','title':f'{SLUG}-runner'})
    meta['dataset_sources'][0]=f'{OWNER}/{SLUG}'
    (kernel_dir/'kernel-metadata.json').write_bytes(dump(meta))
    spec=json.loads(json.dumps(old_spec));spec.update({'attempt':3,'dataset_slug':f'{OWNER}/{SLUG}','dataset_archive_sha256':archive_sha,'manifest_sha256':manifest_sha,'kernel_slug':meta['id'],'kernel_code_sha256':sha((kernel_dir/'kernel.py').read_bytes()),'kernel_metadata_sha256':sha((kernel_dir/'kernel-metadata.json').read_bytes()),'status':'prepared_not_uploaded_or_launched'})
    spec['resources']['aggregate_rss_stop_bytes']=4*1024**3
    spec['resources']['launcher_reserve_bytes']=2*1024**3
    spec['resources']['host_reserve_fraction']=0.20
    spec['resources']['provider_timeout_seconds']=2100
    spec['resources']['whole_kernel_runtime_limit_seconds']=1800
    spec['resources']['host_inventory_pre_jax']=True
    spec['resources']['flushed_phase_rss_milestones']=True
    spec['source_closure']['scripts/realdata_preflight.py']={'sha256':sha(payload['scripts/realdata_preflight.py']),'size_bytes':len(payload['scripts/realdata_preflight.py']),'source_path':'research/campaign_2026-10-04/kaggle/jax_realdata_preflight_v3/source_bundle/scripts/realdata_preflight.py'}
    (HERE/'bundle_spec.json').write_bytes(dump(spec))
    packet={'evidence_id':'E-463','attempt':3,'dataset':f'{OWNER}/{SLUG}','kernel':meta['id'],'archive_sha256':archive_sha,'manifest_sha256':manifest_sha,
      'kernel_sha256':spec['kernel_code_sha256'],'kernel_metadata_sha256':spec['kernel_metadata_sha256'],'private':True,'gpu':False,'tpu':False,'internet':False,
      'provider_timeout_seconds':2100,'whole_kernel_runtime_limit_seconds':1800,'aggregate_rss_limit_bytes':4*1024**3,
      'launcher_reserve_bytes':2*1024**3,'host_reserve_fraction':0.20,'source_file_count':len(payload),'external_input_count':len(manifest['external_inputs']),
      'source_preflight_sha256':sha(payload['scripts/realdata_preflight.py']),'prior_v2_source_sha256':old_manifest['files']['scripts/realdata_preflight.py']['sha256'],
      'prior_v2_error_record':'research/campaign_2026-10-04/kaggle/jax_realdata_preflight_v2/provider_error_E463_v2.json',
      'objective_params_unchanged':True,'fit_groups':'valid:1,valid:0,train:529,train:0','batch_size':4,'chunk_rows':1000,'precision_arms':['fp32','bf16'],'updates_per_arm':1,
      'finite_loss_gradient_moment_prediction_checkpoint_and_source_hash_checks_preserved':True,'phase_milestones':['realdata_import_start','realdata_import_complete','jax_import_start','jax_import_complete','fit_sequence_and_target_load_start','fit_sequence_and_target_load_complete','model_parameter_initialization_start','model_parameter_initialization_complete','fp32_gradient_update_start','fp32_gradient_update_complete','bf16_gradient_update_start','bf16_gradient_update_complete','fp32_evaluation_complete','bf16_evaluation_complete'],
      'pushed':False}
    (HERE/'review_packet.json').write_bytes(dump(packet))
    print(json.dumps({'dataset':f'{OWNER}/{SLUG}','archive_sha256':archive_sha,'manifest_sha256':manifest_sha,'kernel_sha256':spec['kernel_code_sha256'],'kernel_metadata_sha256':spec['kernel_metadata_sha256'],'source_file_count':len(payload),'external_count':len(manifest['external_inputs'])},indent=2))

if __name__=='__main__':main()
