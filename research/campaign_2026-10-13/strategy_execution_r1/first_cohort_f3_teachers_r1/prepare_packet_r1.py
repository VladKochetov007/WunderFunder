"""Build the first seed-3 F3 teacher tranche with the existing Kaggle bundler."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

PACKET = Path("research/campaign_2026-10-13/strategy_execution_r1/first_cohort_f3_teachers_r1")
OLD = Path("research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_fold0_h390_r3_teacher_cohort")
F3_ASSET = Path("artifacts/kaggle_assets/wnn-strategy-f3-t3-assets-20261013-r1")
F3_DATASET = "vladkochetov07/wnn-strategy-clean-t3-f3-assets-20261013-r1"
KERNEL_DATASET = "vladkochetov07/wnn-strategy-f3-t3-seed3-20261013-r1"
SPLIT_SHA = "78157b3347b9e9ba6d2c075d0c53df612e67af9ecd5ee8af7fd7957e406b8343"
PROP_SHA = "8898ef3d31533058c354a16d00632b7c7adca461eaa813f0bed1f59a1fef59b8"
OLD_RUN = "clean_t3_f0_muonall_h384_e3_s3_r3"
NEW_RUN = "clean_t3_f3_muonall_h384_e3_s3_r1"
OLD_PROFILE = "profile_clean_t3_f0_muonall_h384_e3_s3_paired_t4_r3"
NEW_PROFILE = "profile_clean_t3_f3_muonall_h384_e3_s3_t4_r1"


def digest(path: Path, relative_to: Path | None = None) -> dict:
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    name = path.relative_to(relative_to) if relative_to is not None else path
    return {"path": name.as_posix(), "size": path.stat().st_size, "sha256": value}


def patch_kernel_dispatch(kernel: Path, packet: Path, root: Path) -> None:
    """Make the sealed bootstrap call the profile-guarded cohort driver."""
    source = kernel.read_text()
    start = source.index("    command = [\n")
    marker = '    finally:\n        release_mounts(root, manifest["external_mounts"])\n'
    end = source.index(marker, start) + len(marker)
    dispatch = (
        f'    command = [sys.executable, str(root / "{PACKET.as_posix()}/kernel_support/f3_teacher_driver.py")]\n'
        '    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"\n'
        '    environment = os.environ.copy()\n'
        '    environment["WNN_KERNEL_DEADLINE_MONOTONIC"] = repr(kernel_deadline)\n'
        '    try:\n'
        '        require_remaining(kernel_deadline, "teacher pipeline launch")\n'
        '        remaining = remaining_seconds(kernel_deadline)\n'
        '        process = subprocess.Popen(command, cwd=root, env=environment, start_new_session=True)\n'
        '        try:\n'
        '            returncode = process.wait(timeout=remaining)\n'
        '        except subprocess.TimeoutExpired as error:\n'
        '            terminate_process_tree(process)\n'
        '            raise TimeoutError("teacher pipeline exceeded the provider kernel-process budget") from error\n'
        '        if returncode != 0:\n'
        '            raise subprocess.CalledProcessError(returncode, command)\n'
        '        if remaining_seconds(kernel_deadline) <= 0:\n'
        '            raise TimeoutError("teacher pipeline completed outside the provider session window")\n'
        '    finally:\n'
        '        release_mounts(root, manifest["external_mounts"])\n'
    )
    if 'str(root / "kernel_support/f3_teacher_driver.py")' in source:
        raise ValueError("kernel already contains the F3 teacher dispatch")
    kernel.write_text(source[:start] + dispatch + source[end:])
    binding = {
        "schema": "wnn.strategy.f3-teacher-entrypoint-binding.r1",
        "bundle_dataset": KERNEL_DATASET,
        "kernel_dataset_id": f"{KERNEL_DATASET}-runner",
        "title_derived_slug": KERNEL_DATASET.rsplit("/", 1)[1],
        "code_file": "kernel/kernel.py",
        "kernel_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
        "archive_sha256": json.loads((kernel.parent.parent / "bundle-manifest.json").read_text())["archive_sha256"],
        "dispatch": "verified bundle bootstrap starts the sealed profile→fit→FP32-export driver",
        "kernel_source": kernel.relative_to(root).as_posix(),
    }
    (packet / "ENTRYPOINT_BINDING_R1.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n"
    )


def transform(value):
    if isinstance(value, str):
        for old, new in (
            (str(OLD), str(PACKET)),
            (OLD_PROFILE, NEW_PROFILE),
            (OLD_RUN, NEW_RUN),
            ("export_clean_t3_f0_muonall_h384_e3_s3_r3", "export_clean_t3_f3_muonall_h384_e3_s3_r1"),
            ("c20260929_prop_f0", "c20260929_prop_f3"),
            ("fold0.json", "fold3.json"),
            ("09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d", SPLIT_SHA),
            ("b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7", PROP_SHA),
        ):
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [transform(item) for item in value]
    if isinstance(value, dict):
        return {key: transform(item) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    f3_assets = json.loads((root / F3_ASSET / "assets-manifest.json").read_text())
    f3_by_path = {item["path"]: item for item in f3_assets["files"]}
    global_binding = json.loads((root / OLD / "INPUT_BINDING_R3.json").read_text())
    global_mounts = {item["path"]: item["mount"] for item in global_binding["external_inputs"]}

    source_files = [
        "source_snapshot/SOURCE_MANIFEST_H390.json",
        "source_snapshot/scripts/cleanfold_exec.py",
        "source_snapshot/scripts/train_rnn.py",
        "source_snapshot/scripts/predict_run.py",
        "source_snapshot/src/sitecustomize.py",
        *sorted(
            p.relative_to(root / PACKET).as_posix()
            for p in (root / PACKET / "source_snapshot/src/wnn").rglob("*.py")
            if p.is_file()
        ),
    ]
    source_binding = {
        "schema": "wnn.strategy.f3-teacher-source-binding.r1",
        "reference_manifest": digest(root / PACKET / "source_snapshot/SOURCE_MANIFEST_H390.json", root),
        "executed_source_paths": [digest(root / PACKET / name, root) for name in source_files],
        "excluded_uninvoked_helpers": [
            "source_snapshot/scripts/kaggle_common.py",
            "source_snapshot/scripts/kaggle_gpu_runner.py",
            "source_snapshot/scripts/kaggle_kernel.py",
            "These archived helpers are not imported by the sealed F3 driver or train/predict entry points.",
        ],
    }
    (root / PACKET / "SOURCE_BINDING_R1.json").write_text(
        json.dumps(source_binding, indent=2, sort_keys=True) + "\n"
    )
    code_paths = [
        (PACKET / "README.md").as_posix(),
        (PACKET / "TEACHER_COHORT_PLAN_R1.json").as_posix(),
        (PACKET / "INPUT_BINDING_R1.json").as_posix(),
        (PACKET / "KAGGLE_QUOTA_AUDIT_R1.json").as_posix(),
        (PACKET / "SOURCE_BINDING_R1.json").as_posix(),
        (PACKET / "PROFILE_REFERENCE_F0_R3.json").as_posix(),
        (PACKET / "PROFILE_GUARD_CONTRACT_R1.json").as_posix(),
        (PACKET / "kernel_support/f3_teacher_driver.py").as_posix(),
        *((PACKET / name).as_posix() for name in source_files),
        "scripts/kaggle_common.py",
        "scripts/kaggle_gpu_runner.py",
        "scripts/kaggle_kernel.py",
    ]
    code = sorted((digest(root / path, root) for path in code_paths), key=lambda item: item["path"])
    source = {
        "profile": OLD / "jobs/profile_clean_t3_f0_muonall_h384_e3_s3_paired_t4_r3.json",
        "fit": OLD / "jobs/fit_clean_t3_f0_muonall_h384_e3_s3_r3.json",
    }
    for kind in ("fit_train", "fit_valid", "full_valid"):
        source[kind] = OLD / f"jobs/export_clean_t3_f0_muonall_h384_e3_s3_r3_{kind}.json"
    specs = []
    external = {}
    for key, old_spec_path in source.items():
        spec = transform(json.loads((root / old_spec_path).read_text()))
        if key == "profile":
            spec["name"] = NEW_PROFILE
            spec["argv"][spec["argv"].index("--name") + 1] = NEW_PROFILE
            spec["argv"][spec["argv"].index("--valid-fold") + 1] = "3"
            spec["limits"].update({"cpus": 2, "ram_bytes": 8000000000, "gpu_memory_bytes": 12 * 1024**3, "gpu_stop_bytes": 13 * 1024**3, "timeout_seconds": 1800})
        elif key == "fit":
            spec["name"] = f"fit_{NEW_RUN}"
            spec["argv"][spec["argv"].index("--name") + 1] = NEW_RUN
            spec["argv"][spec["argv"].index("--valid-fold") + 1] = "3"
            spec["limits"].update({"cpus": 2, "ram_bytes": 8000000000, "gpu_memory_bytes": 12 * 1024**3, "gpu_stop_bytes": 13 * 1024**3, "timeout_seconds": 20000})
        else:
            spec["name"] = f"export_{NEW_RUN}_{key}"
            spec["limits"].update({"cpus": 2, "ram_bytes": 8000000000, "gpu_memory_bytes": 12 * 1024**3, "gpu_stop_bytes": 13 * 1024**3, "timeout_seconds": 6000})
        spec["code"] = code
        for descriptor in spec["inputs"]:
            path = descriptor["path"]
            if path in f3_by_path:
                item = f3_by_path[path]
                descriptor.update({"size": item["size"], "sha256": item["sha256"]})
            if path.startswith("datasets/"):
                filename = path.split("/", 1)[1]
                external[path] = filename
            elif path in f3_by_path:
                item = f3_by_path[path]
                external[path] = f"{F3_DATASET}:{item['filename']}"
            else:
                mount = global_mounts.get(path)
                if mount is None:
                    raise ValueError(f"no reviewed external asset mapping for {path}")
                if isinstance(mount, dict):
                    external[path] = f"{mount['dataset']}:{mount['filename']}"
                else:
                    external[path] = mount
        sys.path.insert(0, str(root / "scripts"))
        from kaggle_common import canonical_sha256
        spec["code_sha256"] = canonical_sha256(code)
        name = spec["name"]
        destination = root / PACKET / "jobs" / f"{name}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
        specs.append(destination)

    print(json.dumps({"specs": [path.name for path in specs], "unique_external_inputs": len(external), "kernel_dataset": KERNEL_DATASET, "asset_dataset": F3_DATASET}, indent=2))
    if args.build:
        sys.path.insert(0, str(root / "scripts"))
        from kaggle_bundle import build_bundle
        bundle = root / PACKET / "bundle_f3_teacher_seed3_r1"
        build_bundle(root, specs, bundle, KERNEL_DATASET, [f"{path}={mount}" for path, mount in sorted(external.items())], [F3_DATASET, "vladkochetov07/wnn-c20261004-fold0-assets"])
        patch_kernel_dispatch(bundle / "kernel/kernel.py", root / PACKET, root)


if __name__ == "__main__":
    main()
