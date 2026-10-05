from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SRC = ROOT / "research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v4_transport/provider_output_v1/experiment"
CANON_PHASE = SRC / "research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v4/phase_runner.py"
V6_PHASE_REL = "research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v4/phase_runner.py"
V6_RUNNER_REL = "scripts/kaggle_gpu_runner.py"
V6_KERNEL_SOURCE = ROOT / "research/campaign_2026-10-05/w384_v6/bundle/kernel/kernel.py"
V6_KERNEL_SHA256 = "c60dd550411bc9efbe858849b1f12a5136867c5599974075c1f3fef7297b496e"
SPEC_NAMES = [
    "c20261004_w256_adamw_s3_f1_t4v4_probe",
    "c20261004_w256_adamw_s3_f1_t4v4_fit",
    "c20261004_w384_adamw_s3_f1_t4v4_probe",
    "c20261004_w384_adamw_s3_f1_t4v4_fit",
]
GIB = 1024**3


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canon(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected one {label} patch point, found {text.count(old)}")
    return text.replace(old, new)


def replace_assignment(text: str, name: str, value: object) -> str:
    pattern = re.compile(rf"^{re.escape(name)} = .*?$", re.MULTILINE)
    if len(pattern.findall(text)) != 1:
        raise RuntimeError(f"expected one pinned-kernel assignment for {name}")
    return pattern.sub(f"{name} = {value!r}", text, count=1)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    phase = CANON_PHASE.read_text()
    phase = replace_once(phase, 'WORKER_RAM_BYTES = 10 * 1024**3', 'WORKER_RAM_BYTES = 8 * 1024**3', "worker RAM reservation")
    phase = replace_once(phase, '        raise RuntimeError(f"GPU runner failed for phase at {output}")', '''        # Preserve host/GPU state and the admission decision for every phase, including rejection.
        receipts_path = OUTPUT / "admission_receipts.json"
        receipts = json.loads(receipts_path.read_text()) if receipts_path.is_file() else []
        receipts.append({
            "phase": output.name,
            "inventory": json.loads((output / "inventory.json").read_text()) if (output / "inventory.json").is_file() else None,
            "final_inventory": json.loads((output / "final_inventory.json").read_text()) if (output / "final_inventory.json").is_file() else None,
            "admission": json.loads((output / "admission.json").read_text()) if (output / "admission.json").is_file() else None,
            "summary": summary,
            "returncode": completed.returncode,
        })
        write_json(receipts_path, receipts)
        raise RuntimeError(f"GPU runner failed for phase at {output}")''', "phase admission receipt")
    # Save the successful phase receipt too, before validation/gating continues.
    phase = replace_once(phase, '    if summary.get("concurrency_admitted") != 2 or len(summary.get("finished", [])) != 2:', '''    receipts_path = OUTPUT / "admission_receipts.json"
    receipts = json.loads(receipts_path.read_text()) if receipts_path.is_file() else []
    receipts.append({
        "phase": output.name,
        "inventory": json.loads((output / "inventory.json").read_text()) if (output / "inventory.json").is_file() else None,
        "final_inventory": json.loads((output / "final_inventory.json").read_text()) if (output / "final_inventory.json").is_file() else None,
        "admission": json.loads((output / "admission.json").read_text()) if (output / "admission.json").is_file() else None,
        "summary": summary,
        "returncode": completed.returncode,
    })
    write_json(receipts_path, receipts)
    if summary.get("concurrency_admitted") != 2 or len(summary.get("finished", [])) != 2:''', "successful phase receipt")
    # Failed fit admission must be visible in the phase report while retaining the no-fit-before-probes gate.
    phase = replace_once(phase, '        evidence["status"] = "failed_closed"', '        evidence["status"] = "failed_closed"', "failure status")
    phase = replace_once(phase, '        write_json(OUTPUT / "phase_report.json", evidence)', '''        receipt_path = OUTPUT / "admission_receipts.json"
        if receipt_path.is_file():
            evidence["admission_receipts"] = json.loads(receipt_path.read_text())
        write_json(OUTPUT / "phase_report.json", evidence)''', "final receipt inclusion")
    (HERE / "phase_runner.py").write_text(phase)

    runner = (ROOT / "scripts/kaggle_gpu_runner.py").read_text()
    runner = replace_once(runner, "\ndef run(root: Path, spec_paths: list[Path], output: Path, stagger_seconds: float = 30) -> None:", '''
def release_verified_file_cache(root: Path, descriptors: list[dict]) -> dict:
    """Advise DONTNEED only for hashed, declared regular files through read-only FDs."""
    result = {"api": "POSIX_FADV_DONTNEED", "attempted_files": 0, "advised_files": 0,
              "advised_bytes": 0, "failures": [], "status": "complete"}
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return {**result, "status": "unsupported"}
    unique = {item["path"]: item for item in descriptors}
    for name, item in sorted(unique.items()):
        path = root / relative_path(name)
        fd = None
        result["attempted_files"] += 1
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            result["advised_files"] += 1
            result["advised_bytes"] += int(item["size"])
        except OSError as exc:
            result["failures"].append({"path": name, "errno": exc.errno, "error": str(exc)})
        finally:
            if fd is not None:
                os.close(fd)
    if result["failures"]:
        result["status"] = "partial_failure"
    return result


def run(root: Path, spec_paths: list[Path], output: Path, stagger_seconds: float = 30) -> None:''', "bounded per-file cache advice")
    old_preflight = '''    host, devices = host_inventory(), gpu_inventory()
    write_json(output / "inventory.json", {"host": host, "devices": devices})
    concurrency = admit_specs(specs, host, devices)
    print('''
    new_preflight = '''    before_host, before_devices = host_inventory(), gpu_inventory()
    write_json(output / "inventory.json", {"before_validation": {"host": before_host, "devices": before_devices}})
    # Verify exact bytes first; cache hints apply only to these declared input files.
    for spec in specs:
        verify_files(root, spec["code"] + spec["inputs"])
    descriptors = [item for spec in specs for item in spec["code"] + spec["inputs"]]
    cache_advice = release_verified_file_cache(root, descriptors)
    host, devices = host_inventory(), gpu_inventory()
    write_json(output / "inventory.json", {
        "before_validation": {"host": before_host, "devices": before_devices},
        "after_validation_and_cache_advice": {"host": host, "devices": devices},
        "cache_advice": cache_advice,
    })
    try:
        concurrency = admit_specs(specs, host, devices)
    except Exception as exc:
        write_json(output / "admission.json", {
            "status": "rejected", "error_type": type(exc).__name__, "error": str(exc),
            "host": host, "devices": devices, "specs": [s["name"] for s in specs],
            "cache_advice": cache_advice,
            "guard": "min(0.8*capacity, available-0.2*capacity); unchanged",
        })
        raise
    if concurrency < 2:
        write_json(output / "admission.json", {
            "status": "rejected", "error_type": "InsufficientPairedCapacity",
            "error": "Paired width replication requires two concurrent admitted workers",
            "host": host, "devices": devices, "specs": [s["name"] for s in specs],
            "cache_advice": cache_advice,
            "guard": "min(0.8*capacity, available-0.2*capacity); unchanged",
        })
        raise ValueError("Paired phase requires concurrency 2; no worker launched")
    write_json(output / "admission.json", {
        "status": "admitted", "concurrency": concurrency,
        "host": host, "devices": devices, "specs": [s["name"] for s in specs],
        "cache_advice": cache_advice,
        "guard": "min(0.8*capacity, available-0.2*capacity); unchanged",
    })
    print('''
    if old_preflight not in runner:
        raise RuntimeError("could not locate runner admission preflight")
    runner = runner.replace(old_preflight, new_preflight, 1)
    runner = replace_once(runner, '    # Validate the complete batch before any worker can start.\n    for spec in specs:\n        verify_files(root, spec["code"] + spec["inputs"])\n', '', "remove duplicate post-admission validation")
    runner = replace_once(runner, '    finally:\n        write_json(\n            output / "summary.json",', '''    finally:
        try:
            write_json(output / "final_inventory.json", {"host": host_inventory(), "devices": gpu_inventory()})
        except Exception as exc:
            write_json(output / "final_inventory_error.json", {"type": type(exc).__name__, "error": str(exc)})
        write_json(
            output / "summary.json",''', "final host inventory")
    runner = replace_once(runner, '                        "artifacts": export_artifacts(root, spec, job["output"]),', '''                        "artifacts": export_artifacts(root, spec, job["output"]),
                        "host_inventory_after_exit": host_inventory(),
                        "gpu_inventory_after_exit": gpu_inventory(),''', "per-worker exit inventory")
    stale_runner = HERE / "kaggle_gpu_runner.py"
    if stale_runner.exists():
        stale_runner.unlink()
    (HERE / "runtime/scripts").mkdir(parents=True, exist_ok=True)
    (HERE / "runtime/scripts/kaggle_gpu_runner.py").write_text(runner)

    sys.path.insert(0, str(ROOT / "scripts"))
    from kaggle_common import canonical_sha256
    builder_text = (ROOT / "scripts/kaggle_bundle.py").read_text()
    builder_text = builder_text.replace(
        "verify_files(root, list(files.values()))",
        "verify_files(root, [item for name, item in files.items() if name not in external_map])",
    )
    if builder_text == (ROOT / "scripts/kaggle_bundle.py").read_text():
        raise RuntimeError("could not apply external-mount-aware verifier patch")
    (HERE / "bundle_builder.py").write_text(builder_text)
    from bundle_builder import build_bundle

    specs_dir = HERE / "specs"
    specs_dir.mkdir(exist_ok=True)
    for name in SPEC_NAMES:
        spec = json.loads((SRC / f"kaggle_specs/{name}.json").read_text())
        for item in spec["code"]:
            if item["path"].endswith("/phase_runner.py"):
                item["path"] = V6_PHASE_REL
                p = HERE / "phase_runner.py"
            elif item["path"] == "scripts/kaggle_gpu_runner.py":
                p = HERE / "runtime/scripts/kaggle_gpu_runner.py"
            else:
                continue
            item["size"] = p.stat().st_size
            item["sha256"] = sha(p)
        spec["limits"]["ram_bytes"] = 8 * GIB
        spec["code_sha256"] = canonical_sha256(sorted(spec["code"], key=lambda item: item["path"]))
        path = specs_dir / f"{name}.json"
        path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    previous_manifest = json.loads((ROOT / "research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v4_transport/bundle/bundle-manifest.json").read_text())
    external = []
    for logical, value in previous_manifest["external_mounts"].items():
        external.append(f"{logical}={value['dataset']}:{value['filename']}" if isinstance(value, dict) else f"{logical}={value}")
    snapshot = HERE / "bundle_source"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    snapshot.mkdir()
    for name in SPEC_NAMES:
        spec = json.loads((specs_dir / f"{name}.json").read_text())
        for item in spec["code"] + spec["inputs"]:
            if item["path"] in previous_manifest["external_mounts"]:
                continue
            target = snapshot / item["path"]
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if item["path"] == V6_PHASE_REL:
                source = HERE / "phase_runner.py"
            elif item["path"] == V6_RUNNER_REL:
                source = HERE / "runtime/scripts/kaggle_gpu_runner.py"
            else:
                source = SRC / item["path"]
            shutil.copyfile(source, target)
    bundle_dir = HERE / "bundle"
    build_dir = HERE / "bundle.next"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_bundle(snapshot, [specs_dir / f"{name}.json" for name in SPEC_NAMES], build_dir,
                 "vladkochetov07/wnn-w384-fold1-dualt4-20261005-v7", external,
                 ["vladkochetov07/wnn-c20261004-fold1-assets"])
    manifest = json.loads((build_dir / "bundle-manifest.json").read_text())
    if manifest["external_mounts"] != previous_manifest["external_mounts"]:
        raise RuntimeError("The sealed v6 external mount map changed")
    if sha(V6_KERNEL_SOURCE) != V6_KERNEL_SHA256:
        raise RuntimeError("Reviewed v6 kernel bootstrap bytes changed")
    kernel_path = build_dir / "kernel/kernel.py"
    kernel = V6_KERNEL_SOURCE.read_text()
    kernel = replace_once(kernel, '    archive_members = {item["path"] for item in manifest["files"]}', '    archive_members = seen', "actual extracted archive-member set")
    kernel = replace_assignment(kernel, "BUNDLE_SLUG", "wnn-w384-fold1-dualt4-20261005-v7")
    kernel = replace_assignment(kernel, "EXPECTED_ARCHIVE_SHA256", manifest["archive_sha256"])
    kernel = replace_assignment(kernel, "EXPECTED_EXTERNAL_MOUNTS", manifest["external_mounts"])
    if kernel.count("timeout_seconds=25_180") != 1:
        raise RuntimeError("Reviewed v5 whole-job watchdog was not preserved")
    kernel_path.write_text(kernel)
    kernel_metadata_path = build_dir / "kernel/kernel-metadata.json"
    kernel_metadata = json.loads(kernel_metadata_path.read_text())
    if (kernel_metadata["id"] != "vladkochetov07/wnn-w384-fold1-dualt4-20261005-v7-runner"
            or kernel_metadata["enable_gpu"] is not True
            or kernel_metadata["enable_tpu"] is not False
            or kernel_metadata["enable_internet"] is not False
            or kernel_metadata["is_private"] is not True
            or "NvidiaTeslaT4" != kernel_metadata["machine_shape"]):
        raise RuntimeError("Generated private T4 kernel metadata differs from contract")
    if (kernel_metadata["dataset_sources"] != ["vladkochetov07/wnn-w384-fold1-dualt4-20261005-v7",
                                                "vladkochetov07/wnn-c20261004-fold1-assets",
                                                "vladkochetov07/wunder-funder"]):
        raise RuntimeError("Generated kernel dataset sources differ from the reviewed mount contract")
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    build_dir.rename(bundle_dir)
    receipt = {
        "status": "built_offline_not_uploaded", "science": "copied four-epoch W256/W384 fold1 fit and one-batch production probes; code/data/input plans unchanged",
        "worker_ram_bytes": 8 * GIB, "probe_fits_gate": "both fresh probes complete on distinct T4s before fit",
        "source_phase_sha256": sha(CANON_PHASE), "v7_phase_sha256": sha(HERE / "phase_runner.py"),
        "source_runner_sha256": sha(ROOT / "scripts/kaggle_gpu_runner.py"), "v7_runner_sha256": sha(HERE / "runtime/scripts/kaggle_gpu_runner.py"),
        "specs": {name: {"sha256": sha(specs_dir / f"{name}.json"), "limits": json.loads((specs_dir / f"{name}.json").read_text())["limits"]} for name in SPEC_NAMES},
        "bundle_manifest_sha256": sha(bundle_dir / "bundle-manifest.json"),
        "archive_sha256": json.loads((bundle_dir / "bundle-manifest.json").read_text())["archive_sha256"],
        "v6_kernel_template_sha256": V6_KERNEL_SHA256,
        "v7_kernel_sha256": sha(bundle_dir / "kernel/kernel.py"),
        "kernel_metadata_sha256": sha(bundle_dir / "kernel/kernel-metadata.json"),
        "whole_job_watchdog_seconds": 25180,
        "provider_wall_cap_seconds": 25200,
        "kernel_id": json.loads((bundle_dir / "kernel/kernel-metadata.json").read_text())["id"],
        "superseded_candidate_failure_record_sha256": sha(HERE / "SUPERSEDED_CANDIDATE_FAILURE.json"),
        "provider_action": "none; controller review, inventory refresh and independent probe required before any push/fit",
    }
    (HERE / "build_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2))

if __name__ == "__main__":
    main()
