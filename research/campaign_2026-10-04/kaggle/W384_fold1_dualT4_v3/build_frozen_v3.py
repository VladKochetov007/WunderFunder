from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
V2 = ROOT / "research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v2"
HERE = ROOT / "research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v3"
OLD = "research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v2"
NEW = "research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v3"
OLD_SLUG = "wnn-w384-fold1-dualt4-20261004-v2"
SLUG = "wnn-w384-fold1-dualt4-20261004-v3"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha(value: object) -> str:
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def dump(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def rename(value: str) -> str:
    return value.replace(OLD, NEW).replace("t4v2", "t4v3").replace(OLD_SLUG, SLUG)


def write_archive(path: Path, members: dict[str, bytes]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
            with tarfile.open(fileobj=gz, mode="w|") as tar:
                for name, data in sorted(members.items()):
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mode = 0o644
                    tar.addfile(info, io.BytesIO(data))


def normalize_spec_descriptor(spec: dict) -> dict:
    return {
        "code": sorted(spec["code"], key=lambda item: item["path"]),
        "inputs": sorted(spec["inputs"], key=lambda item: item["path"]),
    }


def make_phase_runner(source: bytes) -> bytes:
    text = source.decode().replace(OLD, NEW).replace("t4v2", "t4v3")
    text = text.replace(
        'CAMPAIGN = "research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v3"',
        'CAMPAIGN = "research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v3"',
    )
    text = text.replace(
        'probe_summary = launch_runner(probe_names, OUTPUT / "probe_phase")\n'
        '        probe_rows = validate_workers(probe_summary, probe=True)\n'
        '        probe_devices = {\n'
        '            width: probe_rows[WORKER_NAMES[(width, True)]]["device"]["uuid"] for width in WIDTHS\n'
        '        }\n',
        'probe_evidence_path = ROOT / f"{CAMPAIGN}/probe_reuse_evidence.json"\n'
        '        probe_evidence = json.loads(probe_evidence_path.read_text())\n'
        '        fit_specs = {width: json.loads((ROOT / f"kaggle_specs/{WORKER_NAMES[(width, False)]}.json").read_text()) for width in WIDTHS}\n'
        '        expected_spec_identity = {str(width): {"code_sha256": fit_specs[width]["code_sha256"], "input_sha256": canonical_sha(fit_specs[width]["inputs"])} for width in WIDTHS}\n'
        '        if probe_evidence.get("fit_spec_identity") != expected_spec_identity:\n'
        '            raise RuntimeError("Reused probe source/input identity differs from this fit bundle")\n'
        '        current_uuid_set = {device["uuid"] for device in devices}\n'
        '        old_uuid_set = {probe_evidence["probes"][str(width)]["device"]["uuid"] for width in WIDTHS}\n'
        '        if current_uuid_set == old_uuid_set:\n'
        '            probe_devices = {width: probe_evidence["probes"][str(width)]["device"]["uuid"] for width in WIDTHS}\n'
        '            evidence["probe_phase"] = {"reused": True, "evidence_sha256": sha256_file(probe_evidence_path), "source_identity": probe_evidence["source_identity"], "workers": probe_evidence["probes"], "gate": "reused successful one-batch production probes only when source/input signatures and both physical T4 UUIDs match"}\n'
        '            fit_names = [WORKER_NAMES[(width, False)] for width in WIDTHS]\n'
        '            if devices[0]["uuid"] == probe_devices[WIDTHS[1]]:\n'
        '                fit_names.reverse()\n'
        '        else:\n'
        '            probe_summary = launch_runner(probe_names, OUTPUT / "probe_phase")\n'
        '            probe_rows = validate_workers(probe_summary, probe=True)\n'
        '            probe_devices = {width: probe_rows[WORKER_NAMES[(width, True)]]["device"]["uuid"] for width in WIDTHS}\n'
        '            evidence["probe_phase"] = {"reused": False, "concurrency_admitted": probe_summary["concurrency_admitted"], "workers": {row["name"]: {"device": row["device"], "peak_gpu_bytes": row["peak_gpu_bytes"], "peak_ram_bytes": row["peak_ram_bytes"], "status": row["status"], "returncode": row["returncode"]} for row in probe_summary["finished"]}, "gate": "fresh corrected-path probes completed on distinct T4s and each measured under 12 GiB operating / 13 GiB stop"}\n'
        '            fit_names = [WORKER_NAMES[(width, False)] for width in WIDTHS]\n'
        '            if devices[0]["uuid"] == probe_devices[WIDTHS[1]]:\n'
        '                fit_names.reverse()\n'
        '        probe_rows = None\n',
    )
    # The old inline phase evidence block is now populated above, so remove its duplicate assignment.
    text = text.replace(
        '        evidence["probe_phase"] = {\n'
        '            "concurrency_admitted": probe_summary["concurrency_admitted"],\n'
        '            "workers": {\n'
        '                row["name"]: {\n'
        '                    "device": row["device"],\n'
        '                    "peak_gpu_bytes": row["peak_gpu_bytes"],\n'
        '                    "peak_ram_bytes": row["peak_ram_bytes"],\n'
        '                    "status": row["status"],\n'
        '                    "returncode": row["returncode"],\n'
        '                }\n'
        '                for row in probe_summary["finished"]\n'
        '            },\n'
        '            "gate": "both probes completed on distinct T4s and each measured under 12 GiB operating / 13 GiB stop; only then may fitting begin",\n'
        '        }\n\n'
        '        fit_names = [WORKER_NAMES[(width, False)] for width in WIDTHS]\n',
        '',
    )
    text = text.replace(
        '        fit_names = [WORKER_NAMES[(width, False)] for width in WIDTHS]\n'
        '        fit_summary = launch_runner(fit_names, OUTPUT / "fit_phase")',
        '        fit_summary = launch_runner(fit_names, OUTPUT / "fit_phase")',
    )
    if 'probe_reuse_evidence.json' not in text or 'probe_rows = None' not in text:
        raise RuntimeError("Could not safely inject the probe reuse gate into phase_runner.py")
    if 'def canonical_sha' not in text:
        text = text.replace(
            'def sha256_file(path: Path) -> str:\n',
            'def canonical_sha(value: object) -> str:\n'
            '    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()\n\n\n'
            'def sha256_file(path: Path) -> str:\n',
        )
    check_function = '''def validate_reused_probe_evidence(probe_evidence: dict, devices: list[dict]) -> dict[int, str]:
    """Fail closed unless both archived probes are complete, safe, and on these T4 UUIDs."""
    probes = probe_evidence.get("probes")
    if set(probes or {}) != {str(width) for width in WIDTHS}:
        raise RuntimeError("Reused probe evidence does not contain both widths")
    uuids = set()
    bound = {}
    for width in WIDTHS:
        row = probes[str(width)]
        steps = row.get("step_report", {})
        event = row.get("single_batch_update", {})
        device = row.get("device", {})
        if steps.get("status") != "complete" or len(steps.get("steps", [])) != 1 or steps["steps"][0].get("returncode") != 0:
            raise RuntimeError(f"Prior width-{width} production probe did not complete")
        if event.get("completed_batches") != 1 or event.get("optimizer_attempts", 0) < 1 or event.get("optimizer_completed_updates", 0) < 1 or event.get("amp_skipped_updates") != 0:
            raise RuntimeError(f"Prior width-{width} probe lacks a clean optimizer update")
        if "T4" not in device.get("name", "") or not device.get("uuid"):
            raise RuntimeError(f"Prior width-{width} probe did not bind a Tesla T4")
        expected_missing = [f"research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v2/plans/{row.get('name')}_steps.json"]
        if row.get("result_returncode") != 0 or row.get("result_missing") != expected_missing or row.get("prior_missing_expected") != expected_missing:
            raise RuntimeError(f"Prior width-{width} probe failure was not limited to the known report-path mapping")
        expected_step_report = f"research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v2/{row.get('name')}_steps.json"
        if row.get("step_report_path") != expected_step_report:
            raise RuntimeError(f"Prior width-{width} probe report path is not the matching generated file")
        if row.get("peak_gpu_bytes", STOP_GPU_BYTES) > OPERATING_GPU_BYTES or row.get("peak_gpu_bytes", STOP_GPU_BYTES) >= STOP_GPU_BYTES:
            raise RuntimeError(f"Prior width-{width} probe exceeded the GPU memory gate")
        if row.get("peak_ram_bytes", WORKER_RAM_BYTES + 1) > WORKER_RAM_BYTES:
            raise RuntimeError(f"Prior width-{width} probe exceeded the RAM cap")
        if not row.get("artifact_hashes_verified"):
            raise RuntimeError(f"Prior width-{width} probe artifacts were not verified")
        uuids.add(device["uuid"])
        bound[width] = device["uuid"]
    current = {device["uuid"] for device in devices}
    if len(uuids) != len(WIDTHS) or current != uuids:
        raise RuntimeError("Current physical T4 UUIDs differ from successful archived probes")
    return bound


'''
    if 'def validate_reused_probe_evidence' not in text:
        text = text.replace('def main() -> None:\n', check_function + 'def main() -> None:\n')
    text = text.replace(
        '        current_uuid_set = {device["uuid"] for device in devices}\n'
        '        old_uuid_set = {probe_evidence["probes"][str(width)]["device"]["uuid"] for width in WIDTHS}\n'
        '        if current_uuid_set == old_uuid_set:\n'
        '            probe_devices = {width: probe_evidence["probes"][str(width)]["device"]["uuid"] for width in WIDTHS}\n',
        '        old_uuid_set = {probe_evidence["probes"][str(width)]["device"]["uuid"] for width in WIDTHS}\n'
        '        if {device["uuid"] for device in devices} == old_uuid_set:\n'
        '            probe_devices = validate_reused_probe_evidence(probe_evidence, devices)\n',
    )
    return text.encode()


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    if (HERE / "bundle").exists() or (HERE / "kernel_push").exists():
        raise FileExistsError("v3 bundle outputs already exist")
    source_manifest = json.loads((V2 / "bundle/bundle-manifest.json").read_text())
    source_payload: dict[str, bytes] = {}
    with tarfile.open(V2 / "bundle/bundle.wnn", "r:gz") as archive:
        for item in archive:
            stream = archive.extractfile(item)
            if stream is not None:
                source_payload[item.name] = stream.read()

    v2_results = V2 / "provider_output_v2/results/probe_phase"
    v2_inventory = json.loads((v2_results / "inventory.json").read_text())
    old_specs = {}
    with tarfile.open(V2 / "bundle/bundle.wnn", "r:gz") as archive:
        for item in archive:
            if item.name.startswith("kaggle_specs/") and item.name.endswith(".json"):
                old_specs[item.name] = json.load(archive.extractfile(item))
    if len(old_specs) != 4:
        raise RuntimeError("Frozen v2 archive must contain the exact four fit/probe specs")

    old_phase = f"{OLD}/phase_runner.py"
    new_phase = f"{NEW}/phase_runner.py"
    phase = make_phase_runner(source_payload.pop(old_phase))
    payload: dict[str, bytes] = {}
    for old_name, data in source_payload.items():
        new_name = rename(old_name)
        try:
            decoded = data.decode()
        except UnicodeDecodeError:
            payload[new_name] = data
        else:
            payload[new_name] = decoded.replace(OLD, NEW).replace("t4v2", "t4v3").encode()
    payload[new_phase] = phase

    specs = []
    plan_members = [name for name in list(payload) if "/plans/" in name and name.endswith(".json")]
    for name in plan_members:
        value = json.loads(payload[name])
        value["report"] = value["report"].replace(OLD, NEW).replace("t4v2", "t4v3")
        for step in value["steps"]:
            step["argv"] = [x.replace(OLD, NEW).replace("t4v2", "t4v3") if isinstance(x, str) else x for x in step["argv"]]
        payload[name] = dump(value)

    for old_name, raw in old_specs.items():
        spec = raw
        spec["name"] = spec["name"].replace("t4v2", "t4v3")
        spec["argv"] = [x.replace(OLD, NEW).replace("t4v2", "t4v3") if isinstance(x, str) else x for x in spec["argv"]]
        spec["artifacts"] = [rename(x).replace(f"{NEW}/plans/", f"{NEW}/") for x in spec["artifacts"]]
        spec["environment"] = {k: (v.replace(OLD, NEW).replace("t4v2", "t4v3") if isinstance(v, str) else v) for k, v in spec["environment"].items()}
        for group in ("code", "inputs"):
            for item in spec[group]:
                item["path"] = item["path"].replace(OLD, NEW).replace("t4v2", "t4v3")
                if item["path"] == new_phase:
                    item.update(size=len(phase), sha256=sha(phase))
                elif item["path"] in payload:
                    item.update(size=len(payload[item["path"]]), sha256=sha(payload[item["path"]]))
        spec["code_sha256"] = canonical_sha(sorted(spec["code"], key=lambda item: item["path"]))
        specs.append(spec)
        payload[f"kaggle_specs/{spec['name']}.json"] = (json.dumps(spec, sort_keys=True) + "\n").encode()

    specs.sort(key=lambda item: item["name"])
    fit_spec_identity = {
        str(width): {
            "code_sha256": next(item["code_sha256"] for item in specs if item["name"].startswith(f"c20261004_w{width}_") and item["name"].endswith("_fit")),
            "input_sha256": canonical_sha(next(item["inputs"] for item in specs if item["name"].startswith(f"c20261004_w{width}_") and item["name"].endswith("_fit"))),
        }
        for width in (256, 384)
    }
    previous = {}
    for width in (256, 384):
        old_name = f"c20261004_w{width}_adamw_s3_f1_t4v2_probe"
        result_path = v2_results / old_name / "result.json"
        if not result_path.is_file():
            result_path = v2_results / old_name / "failure.json"
        result = json.loads(result_path.read_text())
        step_path = V2 / "provider_output_v2/experiment/research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v2" / f"{old_name}_steps.json"
        steps = json.loads(step_path.read_text())
        worker_log = (v2_results / old_name / "worker.log").read_text()
        device = result.get("device")
        if device is None:
            device_index = 0 if width == 256 else 1
            device = next(item for item in v2_inventory["devices"] if item["index"] == device_index)
        if steps.get("status") != "complete" or any(item.get("returncode") != 0 for item in steps["steps"]) or "amp_skipped_updates\": 0" not in worker_log:
            raise RuntimeError(f"Prior probe evidence is incomplete for width {width}")
        fit_line = next((line for line in worker_log.splitlines() if line.startswith("FIT ")), None)
        if fit_line is None:
            raise RuntimeError(f"Prior probe lacks the single-batch optimizer record for width {width}")
        fit_event = json.loads(fit_line[4:])["update_diagnostics"]["epoch"]
        artifacts_verified = True
        for artifact in result.get("artifacts", {}).get("files", []):
            artifact_path = V2 / "provider_output_v2/experiment" / artifact["path"]
            if not artifact_path.is_file() or artifact_path.stat().st_size != artifact["size"] or sha(artifact_path.read_bytes()) != artifact["sha256"]:
                artifacts_verified = False
                break
        if not artifacts_verified:
            raise RuntimeError(f"Provider probe artifact hashes did not verify locally for width {width}")
        previous[str(width)] = {
            "prior_kernel": "vladkochetov07/wnn-w384-fold1-dualt4-20261004-v2-runner",
            "prior_kernel_status": "ERROR",
            "prior_worker_status": "failed_closed_on_declared_output_guard_after_zero_returncode",
            "name": old_name,
            "step_report": steps,
            "step_report_path": f"research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v2/{old_name}_steps.json",
            "device": device,
            "peak_gpu_bytes": result["peak_gpu_bytes"],
            "peak_ram_bytes": result["peak_ram_bytes"],
            "result_artifact_files": result["artifacts"]["files"],
            "result_missing": result.get("artifacts", {}).get("missing", []),
            "prior_missing_expected": [f"research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v2/plans/{old_name}_steps.json"],
            "result_returncode": result.get("returncode", 0),
            "single_batch_update": fit_event,
            "artifact_hashes_verified": artifacts_verified,
            "worker_log_sha256": sha(worker_log.encode()),
            "completion_evidence_scope": "one production-shape batch only; no full fit and no score",
        }
    evidence = {
        "protocol": "reuse v2 one-batch probe evidence only when all v3 fit source/input signatures and both physical T4 UUIDs match; otherwise run fresh corrected-path probes",
        "source_identity": {
            "prior_dataset": source_manifest["dataset"],
            "prior_kernel": "vladkochetov07/wnn-w384-fold1-dualt4-20261004-v2-runner",
            "prior_manifest_sha256": sha((V2 / "bundle/bundle-manifest.json").read_bytes()),
            "prior_archive_sha256": source_manifest["archive_sha256"],
            "fitness_claim": "probe execution evidence only; no epoch/fold score or full-fit result",
        },
        "fit_spec_identity": fit_spec_identity,
        "probes": previous,
    }
    evidence_path = f"{NEW}/probe_reuse_evidence.json"
    payload[evidence_path] = dump(evidence)

    files: dict[str, dict] = {}
    for spec in specs:
        for item in spec["code"] + spec["inputs"]:
            old = files.setdefault(item["path"], item)
            if old != item:
                raise RuntimeError(f"Conflicting descriptors: {item['path']}")
    external = {rename(k): v for k, v in source_manifest["external_mounts"].items()}
    for name, item in files.items():
        if name in external:
            continue
        if name not in payload:
            raise RuntimeError(f"Missing sealed payload {name}")
        if len(payload[name]) != item["size"] or sha(payload[name]) != item["sha256"]:
            raise RuntimeError(f"Payload descriptor mismatch {name}")

    archive_members = {name: data for name, data in payload.items() if name in files or name.startswith("kaggle_specs/") or name == evidence_path}
    bundle = HERE / "bundle"
    bundle.mkdir()
    write_archive(bundle / "bundle.wnn", archive_members)
    manifest = {
        "version": 1,
        "dataset": f"vladkochetov07/{SLUG}",
        "private": True,
        "archive_sha256": sha((bundle / "bundle.wnn").read_bytes()),
        "archive_bytes": (bundle / "bundle.wnn").stat().st_size,
        "expanded_bytes": sum(len(data) for data in archive_members.values()),
        "files": [files[path] for path in sorted(files)] + [{"path": evidence_path, "size": len(payload[evidence_path]), "sha256": sha(payload[evidence_path])}],
        "external_mounts": external,
        "spec_paths": [f"kaggle_specs/{spec['name']}.json" for spec in specs],
    }
    (bundle / "bundle-manifest.json").write_bytes(dump(manifest))
    (bundle / "dataset-metadata.json").write_bytes(dump({"id": manifest["dataset"], "title": SLUG, "licenses": [{"name": "other"}], "isPrivate": True}))

    kernel = (V2 / "bundle/kernel/kernel.py").read_text().replace(OLD, NEW).replace(OLD_SLUG, SLUG).replace(source_manifest["archive_sha256"], manifest["archive_sha256"])
    kernel_dir = HERE / "kernel_push"
    kernel_dir.mkdir()
    (kernel_dir / "kernel.py").write_text(kernel)
    metadata = json.loads((V2 / "bundle/kernel/kernel-metadata.json").read_text())
    metadata["id"] = f"vladkochetov07/{SLUG}-runner"
    metadata["title"] = f"{SLUG}-runner"
    metadata["dataset_sources"][0] = manifest["dataset"]
    (kernel_dir / "kernel-metadata.json").write_bytes(dump(metadata))

    (HERE / "review_packet.json").write_bytes(dump({
        "dataset": manifest["dataset"],
        "kernel": metadata["id"],
        "private": True,
        "accelerator": "2x Nvidia Tesla T4",
        "worker_fit_timeout_seconds": 15600,
        "provider_timeout_seconds": 18000,
        "expected_fit_device_gpu_hours": 8.6666666667,
        "worst_case_device_gpu_hours": 10.0,
        "source_bundle_v2_archive_sha256": source_manifest["archive_sha256"],
        "fresh_bundle_archive_sha256": manifest["archive_sha256"],
        "probe_reuse_evidence_sha256": sha(payload[evidence_path]),
        "probe_reused_only_if_source_and_both_t4_uuids_match": True,
        "fallback_runs_both_corrected_output_path_probes_before_fit": True,
        "only_mapping_fix": "declared probe/fit step-report artifact paths now match the plan report path; removed incorrect /plans/ segment",
        "full_fits_started_in_v2": False,
        "pushed": False,
    }))
    print(json.dumps({"archive_sha256": manifest["archive_sha256"], "archive_bytes": manifest["archive_bytes"], "manifest_sha256": sha((bundle / "bundle-manifest.json").read_bytes()), "kernel_sha256": sha((kernel_dir / "kernel.py").read_bytes()), "metadata_sha256": sha((kernel_dir / "kernel-metadata.json").read_bytes()), "probe_evidence_sha256": sha(payload[evidence_path]), "specs": [spec["name"] for spec in specs]}, indent=2))


if __name__ == "__main__":
    main()
