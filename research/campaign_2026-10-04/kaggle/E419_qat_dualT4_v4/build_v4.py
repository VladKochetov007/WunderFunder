from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath

import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
V3 = ROOT / "research/campaign_2026-10-04/kaggle/E419_qat_dualT4_v3"
FAILURE = V3 / "failure_20261004T1022Z/experiment"
V3_SLUG = "wnn-e419-qat-dualt4-20261004-v3"
SLUG = "wnn-e419-qat-dualt4-20261004-v4"
DATASET = f"vladkochetov07/{SLUG}"
ASSET_DATASET = "vladkochetov07/wnn-c20261004-fold0-assets"
CAMPAIGN = "research/campaign_2026-10-04/kaggle/E419_qat_dualT4_v4"
SPLIT_SHA = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
RECOVERED_REPORT = ROOT / "research/campaign_2026-10-04/checks/e419_fp32_controls_recovered_report.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def descriptor(path: str, root: Path) -> dict:
    p = root / path
    return {"path": path, "size": p.stat().st_size, "sha256": digest(p)}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def safe_link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == source.stat().st_size and digest(destination) == digest(source):
            return
        raise FileExistsError(f"Existing content-addressed file differs: {destination}")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def tensor_identity(checkpoint: Path) -> dict:
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["model"]
    h = hashlib.sha256()
    entries = []
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        raw = value.numpy().tobytes()
        entries.append({"key": key, "dtype": str(value.dtype), "shape": list(value.shape),
                        "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
        h.update(key.encode() + b"\0" + str(value.dtype).encode() + b"\0")
        h.update(json.dumps(list(value.shape), separators=(",", ":")).encode() + b"\0")
        h.update(raw)
    return {"state_tensor_sha256": h.hexdigest(), "tensor_count": len(entries), "tensors": entries}


def extract_v3(v3_manifest: dict, archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    allowed = {x["path"] for x in v3_manifest["files"]} - set(v3_manifest["external_mounts"])
    seen: set[str] = set()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            name = member.name
            if name.startswith("kaggle_specs/"):
                continue
            if name not in allowed or not member.isfile() or name in seen:
                raise ValueError(f"Unexpected v3 archive member: {name}")
            seen.add(name)
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tf.extractfile(member)
            if source is None:
                raise ValueError(f"Missing archive payload: {name}")
            with source, target.open("wb") as out:
                shutil.copyfileobj(source, out, 8 * 1024 * 1024)
    if seen != allowed:
        raise ValueError(f"Frozen v3 archive file mismatch: {len(allowed-seen)} omitted")
    for item in v3_manifest["files"]:
        if item["path"] in v3_manifest["external_mounts"]:
            continue
        p = destination / item["path"]
        if p.stat().st_size != item["size"] or digest(p) != item["sha256"]:
            raise ValueError(f"Frozen v3 member hash mismatch: {item['path']}")


def v4_name(text: str) -> str:
    return (text.replace("E419_qat_dualT4_v3", "E419_qat_dualT4_v4")
            .replace("t4v3", "t4v4").replace("t4v2", "t4v4"))


def make_screens(source_root: Path, specs: list[dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for spec in specs:
        for item in spec["inputs"]:
            old = item["path"]
            if not old.endswith(".json") or "_screen.json" not in old:
                continue
            data = json.loads((source_root / old).read_text())
            data = json.loads(json.dumps(data).replace("t4v2", "t4v4").replace("t4v3", "t4v4"))
            new = v4_name(old)
            # Control screen members bind to byte-identical aliases of the saved v3 controls.
            write_json(source_root / new, data)
            result[new] = data
    return result


def set_step(plan: dict, name: str, argv: list[str], at: int | None = None) -> None:
    step = {"name": name, "argv": argv}
    if at is None:
        plan["steps"].append(step)
    else:
        plan["steps"].insert(at, step)


def build_plan(source_root: Path, old_spec: dict, preset: str) -> tuple[dict, list[str]]:
    spec_name = v4_name(old_spec["name"])
    plan_path = f"{CAMPAIGN}/{spec_name}_plan.json"
    old_plan_path = old_spec["argv"][-1]
    old_plan = json.loads((source_root / old_plan_path).read_text())
    names = {s["name"]: s for s in old_plan["steps"]}
    v4_plan = {"version": 1, "report": f"{CAMPAIGN}/{spec_name}_steps.json", "steps": []}
    worker_names = ["recurrent", "recurrent_head"]
    control_name = f"c20261004_e419_fp32_control_{preset}_s3_f0_t4v4"

    # The control score validates cache routing only; its checkpoint/prediction bytes are reused.
    set_step(v4_plan, "reused_v3_control_cached_prediction_single_arm_path_check", [
        "{python}", "scripts/screen_score.py",
        f"{CAMPAIGN}/c20261004_e419_qat_{preset}_s3_f0_t4v4_control_screen.json",
        "--output", f"{CAMPAIGN}/c20261004_e419_qat_{preset}_s3_f0_t4v4_control_score.json",
    ])

    for comparison_preset in worker_names:
        parent_id = f"c20261004_e419_ptq_parent_{comparison_preset}_{preset}_s3_f0_t4v4"
        float_id = f"c20261004_e419_ptq_float_{comparison_preset}_{preset}_s3_f0_t4v4"
        parent_export = {
            "name": f"parent_{comparison_preset}_zero_update_ptq_export",
            "argv": ["{python}", "scripts/export_qat_control.py",
                     "artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt", "--output",
                     f"artifacts/runs/{parent_id}/model.pt", "--preset", comparison_preset],
        }
        set_step(v4_plan, parent_export["name"], parent_export["argv"])
        set_step(v4_plan, f"parent_{comparison_preset}_ptq_full_valid_prediction", [
            "{python}", "scripts/predict_run.py", parent_id, "valid"])
        set_step(v4_plan, f"parent_{comparison_preset}_ptq_single_arm_score", [
            "{python}", "scripts/screen_score.py",
            f"{CAMPAIGN}/c20261004_e419_ptq_parent_{comparison_preset}_{preset}_s3_f0_t4v4_screen.json",
            "--output", f"{CAMPAIGN}/{parent_id}_score.json"])

    for comparison_preset in worker_names:
        float_id = f"c20261004_e419_ptq_float_{comparison_preset}_{preset}_s3_f0_t4v4"
        source_control = f"artifacts/runs/{control_name}/model_ep0.pt"
        set_step(v4_plan, f"float_control_{comparison_preset}_zero_update_ptq_export", [
            "{python}", "scripts/export_qat_control.py", source_control, "--output",
            f"artifacts/runs/{float_id}/model.pt", "--preset", comparison_preset])
        set_step(v4_plan, f"float_control_{comparison_preset}_ptq_full_valid_prediction", [
            "{python}", "scripts/predict_run.py", float_id, "valid"])
        set_step(v4_plan, f"float_control_{comparison_preset}_ptq_single_arm_score", [
            "{python}", "scripts/screen_score.py",
            f"{CAMPAIGN}/c20261004_e419_ptq_float_{comparison_preset}_{preset}_s3_f0_t4v4_screen.json",
            "--output", f"{CAMPAIGN}/{float_id}_score.json"])

    fit = json.loads(json.dumps(names["fixed_one_epoch_qat_fit"]))
    fit["name"] = "fixed_one_epoch_qat_fit"
    fit["argv"] = [v4_name(x) for x in fit["argv"]]
    set_step(v4_plan, fit["name"], fit["argv"])
    set_step(v4_plan, "qat_fixed_epoch0_full_valid_prediction", [
        "{python}", "scripts/predict_run.py", f"c20261004_e419_qat_{preset}_s3_f0_t4v4", "valid", "--epoch", "0"])
    set_step(v4_plan, "qat_frozen_fold0_single_arm_score", [
        "{python}", "scripts/screen_score.py",
        f"{CAMPAIGN}/c20261004_e419_qat_{preset}_s3_f0_t4v4_qat_screen.json",
        "--output", f"{CAMPAIGN}/c20261004_e419_qat_{preset}_s3_f0_t4v4_qat_score.json"])

    # Remove prior v2/v3 screen and plan inputs; v4 artifacts are the only active plan inputs.
    replace = {x["path"] for x in old_spec["inputs"] if x["path"].startswith(f"research/campaign_2026-10-04/kaggle/E419_qat_dualT4_v3/")}
    extra = [plan_path]
    extra += [v4_name(x["path"]) for x in old_spec["inputs"] if "_screen.json" in x["path"]]
    input_paths = [x["path"] for x in old_spec["inputs"] if x["path"] not in replace]
    input_paths = sorted(set(input_paths + extra))
    return {"spec_name": spec_name, "plan_path": plan_path, "plan": v4_plan,
            "input_paths": input_paths, "control_name": control_name}, []


def main() -> None:
    if (HERE / "upload_dataset" / "ready_manifest.json").exists():
        raise FileExistsError(f"A completed v4 bundle already exists: {HERE}")
    source_archive = V3 / "bundle/bundle.wnn"
    v3_manifest = json.loads((V3 / "bundle/bundle-manifest.json").read_text())
    if digest(source_archive) != v3_manifest["archive_sha256"]:
        raise ValueError("Frozen v3 source archive SHA mismatch")
    source_root = HERE / "source_root"
    upload = HERE / "upload_dataset"
    specs_dir = HERE / "specs"
    plans_dir = HERE / "plans"
    specs_dir.mkdir(parents=True, exist_ok=True)
    plans_dir.mkdir(parents=True, exist_ok=True)
    upload.mkdir(parents=True, exist_ok=True)
    if not source_root.exists():
        extract_v3(v3_manifest, source_archive, source_root)

    old_specs = [json.loads(p.read_text()) for p in sorted((V3 / "specs").glob("*.sealed.json"))]
    if len(old_specs) != 2:
        raise ValueError("Expected two sealed v3 QAT worker specs")
    screen_values = make_screens(source_root, old_specs)

    # Save the exact frozen source archive alongside v4 and content-address every reused control artifact.
    safe_link_or_copy(source_archive, upload / "frozen-v3-source.bundle.wnn")
    recovery = json.loads(RECOVERED_REPORT.read_text())
    reuse = {
        "version": 1,
        "source_archive_sha256": v3_manifest["archive_sha256"],
        "source_code_sha256": old_specs[0]["code_sha256"],
        "source_code_paths": [x["path"] for x in old_specs[0]["code"]],
        "recovered_report_sha256": digest(RECOVERED_REPORT),
        "score_role": "path-check-only; the two recovered controls have duplicate model tensors and identical prediction bytes",
        "fold": 0,
        "split_manifest_sha256": SPLIT_SHA,
        "controls": {},
    }
    control_mounts = {}
    control_logicals = []
    for preset in ("recurrent", "recurrent_head"):
        source_run = f"c20261004_e419_fp32_control_{preset}_s3_f0_t4v3"
        alias_run = f"c20261004_e419_fp32_control_{preset}_s3_f0_t4v4"
        sources = {
            "checkpoint": FAILURE / f"artifacts/runs/{source_run}/model_ep0.pt",
            "provenance": FAILURE / f"artifacts/runs/{source_run}/provenance.json",
            "prediction": FAILURE / f"cache/pred_{source_run}_valid.npy",
            "sidecar": FAILURE / f"cache/pred_{source_run}_valid.json",
        }
        alias_paths = {
            "checkpoint": f"artifacts/runs/{alias_run}/model_ep0.pt",
            "provenance": f"artifacts/runs/{alias_run}/provenance.json",
            "prediction": f"cache/pred_{alias_run}_valid.npy",
            "sidecar": f"cache/pred_{alias_run}_valid.json",
        }
        sidecar = json.loads(sources["sidecar"].read_text())
        provenance = json.loads(sources["provenance"].read_text())
        if sidecar["run"] != source_run or sidecar["epoch"] != 0 or sidecar["split"] != "valid":
            raise ValueError(f"Unexpected recovered control sidecar: {source_run}")
        if sidecar["split_manifest_sha256"] != SPLIT_SHA or provenance["split_manifest_sha256"] != SPLIT_SHA:
            raise ValueError(f"Recovered control split mismatch: {source_run}")
        if digest(sources["checkpoint"]) != sidecar["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint/sidecar hash mismatch: {source_run}")
        if digest(sources["prediction"]) != sidecar["prediction_sha256"]:
            raise ValueError(f"Prediction/sidecar hash mismatch: {source_run}")
        if provenance["checkpoint_sha256"].get("model_ep0.pt") != sidecar["checkpoint_sha256"]:
            raise ValueError(f"Provenance/checkpoint hash mismatch: {source_run}")
        files = {}
        for kind, src in sources.items():
            sha = digest(src)
            filename = f"{sha}.wnn"
            safe_link_or_copy(src, upload / filename)
            logical = alias_paths[kind]
            control_mounts[logical] = {"dataset": DATASET, "filename": filename}
            control_logicals.append(logical)
            files[kind] = {"source_path": str(src.relative_to(ROOT)), "alias_path": logical,
                           "sha256": sha, "size": src.stat().st_size,
                           "source_run": source_run, "alias_run": alias_run}
        reuse["controls"][preset] = {
            "source_run": source_run,
            "alias_run": alias_run,
            "checkpoint_sha256": sidecar["checkpoint_sha256"],
            "prediction_sha256": sidecar["prediction_sha256"],
            "sidecar_sha256": digest(sources["sidecar"]),
            "provenance_sha256": digest(sources["provenance"]),
            "split_manifest_sha256": sidecar["split_manifest_sha256"],
            "model_tensor_identity": tensor_identity(sources["checkpoint"]),
            "sidecar_run_retained": source_run,
            "artifacts": files,
        }
    if reuse["controls"]["recurrent"]["prediction_sha256"] != reuse["controls"]["recurrent_head"]["prediction_sha256"]:
        raise ValueError("Expected duplicate recovered FP32 control predictions")
    if reuse["controls"]["recurrent"]["model_tensor_identity"]["state_tensor_sha256"] != reuse["controls"]["recurrent_head"]["model_tensor_identity"]["state_tensor_sha256"]:
        raise ValueError("Recovered FP32 control model tensors are not exact duplicates")

    reuse_path = f"{CAMPAIGN}/control_reuse_provenance.json"
    write_json(source_root / reuse_path, reuse)
    write_json(HERE / "control_reuse_provenance.json", reuse)

    new_plans = {}
    for old_spec in old_specs:
        preset = "recurrent_head" if "recurrent_head" in old_spec["name"] else "recurrent"
        built, _ = build_plan(source_root, old_spec, preset)
        plan = built["plan"]
        plan_path = built["plan_path"]
        write_json(source_root / plan_path, plan)
        write_json(HERE / plan_path.removeprefix(CAMPAIGN + "/"), plan)
        new_plans[preset] = built

    v3_external = dict(v3_manifest["external_mounts"])
    v3_external.update(control_mounts)
    source_entries = {x["path"]: x for x in v3_manifest["files"]}
    # Frozen code and ordinary input bytes are drawn from the unpacked v3 archive.
    code_entries = sorted(old_specs[0]["code"], key=lambda x: x["path"])
    if code_entries != sorted(old_specs[1]["code"], key=lambda x: x["path"]):
        raise ValueError("Worker source closures differ")
    specs = []
    new_screens = set()
    for p in new_plans.values():
        new_screens.update(x for x in p["input_paths"] if "_screen.json" in x)
    screen_desc = {p: descriptor(p, source_root) for p in sorted(new_screens)}
    reuse_desc = descriptor(reuse_path, source_root)
    for old_spec in old_specs:
        preset = "recurrent_head" if "recurrent_head" in old_spec["name"] else "recurrent"
        built = new_plans[preset]
        input_paths = set(built["input_paths"])
        input_paths.add(reuse_path)
        input_paths.update(control_logicals)
        inputs = []
        for name in sorted(input_paths):
            if name in screen_desc:
                item = screen_desc[name]
            elif name == reuse_path:
                item = reuse_desc
            elif name in control_mounts:
                source_path = next(x["source_path"] for c in reuse["controls"].values() for x in c["artifacts"].values() if x["alias_path"] == name)
                src = ROOT / source_path
                item = {"path": name, "size": src.stat().st_size, "sha256": digest(src)}
            elif name.startswith(CAMPAIGN + "/") and name.endswith("_plan.json"):
                item = descriptor(name, source_root)
            else:
                item = source_entries.get(name)
                if item is None:
                    item = descriptor(name, source_root)
            inputs.append({key: item[key] for key in ("path", "size", "sha256")})
        # Imported control artifacts are the only extra outputs; QAT/PTQ outputs remain fully declared.
        artifacts = [v4_name(path) for path in old_spec["artifacts"]]
        for path in control_logicals:
            if path not in artifacts:
                artifacts.append(path)
        artifacts.extend([reuse_path, *[x["path"] for x in inputs if "_screen.json" in x["path"]]])
        spec = {
            "version": 1,
            "name": built["spec_name"],
            "kind": old_spec["kind"],
            "argv": ["{python}", "scripts/kaggle_sequence.py", "--plan", built["plan_path"]],
            "code": code_entries,
            "code_sha256": old_spec["code_sha256"],
            "inputs": inputs,
            "artifacts": sorted(set(artifacts)),
            "limits": dict(old_spec["limits"], timeout_seconds=14_400),
            "environment": dict(old_spec["environment"]),
        }
        spec["inputs"] = sorted(spec["inputs"], key=lambda x: x["path"])
        spec_path = HERE / "specs" / f"{spec['name']}.sealed.json"
        write_json(spec_path, spec)
        specs.append(spec)

    # Make the explicit cost contract conservative: 2 allocated T4s for at most 4h worker runtime.
    budget = {
        "provider_timeout_seconds": 15_300,
        "worker_timeout_seconds": 14_400,
        "physical_gpus": 2,
        "maximum_device_gpu_hours": 8.0,
        "provider_overhead_reserve_hours": 0.5,
        "hard_attempt_cap_gpu_hours": 8.5,
        "quota_remaining_before_push_hours": 26.0,
        "quota_remaining_after_max_attempt_hours": 17.5,
        "weekly_reserve_required_hours": 6.0,
        "width_pair_staged_budget_hours": 9.0,
        "fits_reserve_plus_width": True,
        "launch_gate": "READY input dataset, 2 physical Tesla T4 workers, exact bundle/source/hash validation; one dual-T4 kernel at a time",
        "vrm_host_stop_bytes": 13 * 1024**3,
    }
    write_json(HERE / "budget.json", budget)

    # Build a deterministic archive: frozen v3 source files, v4 plans/screens, and no credentials.
    file_entries = {}
    for item in code_entries:
        file_entries[item["path"]] = item
    for spec in specs:
        for item in spec["inputs"]:
            file_entries[item["path"]] = item
    external = dict(v3_external)
    total_expanded = sum(x["size"] for path, x in file_entries.items() if path not in external)
    archive_path = upload / "bundle.wnn"
    with archive_path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz, tarfile.open(fileobj=gz, mode="w|") as tf:
        for path, item in sorted(file_entries.items()):
            if path in external:
                continue
            src = source_root / path
            info = tarfile.TarInfo(path)
            info.size, info.mode = item["size"], 0o644
            with src.open("rb") as f:
                tf.addfile(info, f)
        for spec in specs:
            data = (json.dumps(spec, sort_keys=True) + "\n").encode()
            info = tarfile.TarInfo(f"kaggle_specs/{spec['name']}.json")
            info.size, info.mode = len(data), 0o644
            tf.addfile(info, io.BytesIO(data))
    manifest = {
        "version": 1,
        "dataset": DATASET,
        "private": True,
        "archive_sha256": digest(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "expanded_bytes": total_expanded,
        "files": sorted(file_entries.values(), key=lambda x: x["path"]),
        "external_mounts": external,
        "spec_paths": [f"kaggle_specs/{s['name']}.json" for s in specs],
        "frozen_v3_source_archive_sha256": v3_manifest["archive_sha256"],
        "control_reuse_manifest": reuse_path,
        "control_reuse_manifest_sha256": digest(source_root / reuse_path),
    }
    write_json(upload / "bundle-manifest.json", manifest)
    write_json(upload / "ready_manifest.json", {
        "archive_sha256": manifest["archive_sha256"],
        "dataset": DATASET,
        "ready_for_preflight": True,
    })
    write_json(upload / "dataset-metadata.json", {
        "id": DATASET, "title": SLUG, "licenses": [{"name": "other"}], "isPrivate": True,
    })
    kernel_dir = HERE / "kernel_push"
    kernel_dir.mkdir(exist_ok=True)
    bootstrap = (source_root / "scripts/kaggle_kernel.py").read_text()
    bootstrap = bootstrap.replace('BUNDLE_SLUG = ""', f"BUNDLE_SLUG = {SLUG!r}")
    bootstrap = bootstrap.replace('EXPECTED_ARCHIVE_SHA256 = ""', f"EXPECTED_ARCHIVE_SHA256 = {manifest['archive_sha256']!r}")
    bootstrap = bootstrap.replace("EXPECTED_EXTERNAL_MOUNTS = None", f"EXPECTED_EXTERNAL_MOUNTS = {external!r}")
    (kernel_dir / "kernel.py").write_text(bootstrap)
    sources = {DATASET}
    for value in external.values():
        sources.add(value["dataset"] if isinstance(value, dict) else "vladkochetov07/wunder-funder")
    kernel_metadata = {
        "id": f"vladkochetov07/{SLUG}-runner",
        "title": f"{SLUG} runner",
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": sorted(sources),
        "competition_sources": [],
        "kernel_sources": [],
    }
    write_json(kernel_dir / "kernel-metadata.json", kernel_metadata)
    # Local exact-source tree is used only for CPU preflight and cache-score smoke tests.
    write_json(HERE / "bundle-manifest.json", manifest)
    write_json(HERE / "kernel-metadata.json", kernel_metadata)
    write_json(HERE / "provenance.json", {
        "status": "prepared_not_pushed",
        "bundle_archive_sha256": manifest["archive_sha256"],
        "frozen_v3_source_archive_sha256": v3_manifest["archive_sha256"],
        "source_code_sha256": old_specs[0]["code_sha256"],
        "control_reuse_manifest_sha256": manifest["control_reuse_manifest_sha256"],
        "spec_sha256": {s["name"]: digest(HERE / "specs" / f"{s['name']}.sealed.json") for s in specs},
        "control_reuse": reuse,
        "budget": budget,
    })
    print(json.dumps({"archive_sha256": manifest["archive_sha256"],
                      "archive_bytes": manifest["archive_bytes"],
                      "expanded_bytes": total_expanded,
                      "controls": reuse["controls"],
                      "specs": [s["name"] for s in specs],
                      "external_mounts": len(external),
                      "dataset_files": len(list(upload.iterdir()))}, indent=2))


if __name__ == "__main__":
    main()
