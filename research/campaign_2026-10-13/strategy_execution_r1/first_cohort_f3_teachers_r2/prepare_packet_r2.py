"""Seal the quota-bounded successor using the existing Kaggle packet builder."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

R1 = Path("research/campaign_2026-10-13/strategy_execution_r1/first_cohort_f3_teachers_r1")
PACKET = Path("research/campaign_2026-10-13/strategy_execution_r1/first_cohort_f3_teachers_r2")
DATASET = "vladkochetov07/wnn-strategy-f3-t3-seed3-20261013-r2"


def file_entry(root: Path, name: str) -> dict:
    path = root / name
    return {
        "path": name,
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def replace_strings(value: object, replacements: tuple[tuple[str, str], ...]) -> object:
    """Rewrite R1 identifiers throughout a copied job specification."""
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: replace_strings(item, replacements) for key, item in value.items()}
    return value


def patch_bootstrap(kernel: Path, root: Path, manifest: dict) -> None:
    source = kernel.read_text()
    start = source.index("    command = [\n")
    marker = '    finally:\n        release_mounts(root, manifest["external_mounts"])\n'
    end = source.index(marker, start) + len(marker)
    dispatch = (
        f'    command = [sys.executable, str(root / "{PACKET.as_posix()}/kernel_support/f3_teacher_driver_r2.py")]\n'
        '    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"\n'
        "    environment = os.environ.copy()\n"
        '    environment["WNN_KERNEL_DEADLINE_MONOTONIC"] = repr(kernel_deadline)\n'
        '    environment["WNN_KERNEL_STARTED_MONOTONIC"] = repr(kernel_started)\n'
        '    environment["WNN_PROJECT_ROOT"] = str(root)\n'
        "    try:\n"
        '        require_remaining(kernel_deadline, "quota-bounded teacher pipeline launch")\n'
        "        remaining = remaining_seconds(kernel_deadline)\n"
        "        process = subprocess.Popen(command, cwd=root, env=environment, start_new_session=True)\n"
        "        try:\n"
        "            returncode = process.wait(timeout=remaining)\n"
        "        except subprocess.TimeoutExpired as error:\n"
        "            terminate_process_tree(process)\n"
        '            raise TimeoutError("quota-bounded teacher pipeline exceeded the provider session") from error\n'
        "        if returncode != 0:\n"
        "            raise subprocess.CalledProcessError(returncode, command)\n"
        "        if remaining_seconds(kernel_deadline) <= 0:\n"
        '            raise TimeoutError("teacher pipeline completed outside the provider session window")\n'
        "    finally:\n"
        '        release_mounts(root, manifest["external_mounts"])\n'
    )
    if 'str(root / "' + PACKET.as_posix() + '/kernel_support/f3_teacher_driver_r2.py")' in source:
        raise ValueError("bootstrap already contains the R2 dispatch")
    kernel.write_text(source[:start] + dispatch + source[end:])
    binding = {
        "schema": "wnn.strategy.f3-teacher-entrypoint-binding.r2",
        "bundle_dataset": DATASET,
        "kernel_dataset_id": f"{DATASET}-runner",
        "title_derived_slug": DATASET.rsplit("/", 1)[1],
        "code_file": "kernel/kernel.py",
        "kernel_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
        "archive_sha256": manifest["archive_sha256"],
        "dispatch": "verified bootstrap starts the quota-admitted profile→fit→three-export driver and passes exact project root/deadline",
        "kernel_source": kernel.relative_to(root).as_posix(),
    }
    (root / PACKET / "ENTRYPOINT_BINDING_R2.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    r1_binding = json.loads((root / R1 / "SOURCE_BINDING_R1.json").read_text())
    source_paths = sorted({entry["path"] for entry in r1_binding["executed_source_paths"]})
    runtime_paths = [
        "scripts/kaggle_common.py",
        "scripts/kaggle_gpu_runner.py",
        "scripts/kaggle_kernel.py",
    ]
    execution_paths = sorted(
        set(
            source_paths
            + runtime_paths
            + [f"{PACKET.as_posix()}/kernel_support/f3_teacher_driver_r2.py"]
        )
    )
    source_binding = {
        "schema": "wnn.strategy.f3-teacher-source-binding.r2",
        "reference_source_manifest": file_entry(
            root, f"{R1.as_posix()}/source_snapshot/SOURCE_MANIFEST_H390.json"
        ),
        "execution_code": [file_entry(root, name) for name in execution_paths],
        "reference_r1_source_binding_sha256": hashlib.sha256(
            (root / R1 / "SOURCE_BINDING_R1.json").read_bytes()
        ).hexdigest(),
        "description": "R2 reuses the exact independently trained source snapshot and existing Kaggle runtime, with only the quota-bounded dispatch driver changed.",
    }
    (root / PACKET / "SOURCE_BINDING_R2.json").write_text(
        json.dumps(source_binding, indent=2, sort_keys=True) + "\n"
    )
    code_paths = [
        f"{PACKET.as_posix()}/README.md",
        f"{PACKET.as_posix()}/KAGGLE_QUOTA_GUARD_R2.json",
        f"{PACKET.as_posix()}/OUTPUT_COLLECTION_PLAN_R2.json",
        f"{PACKET.as_posix()}/SOURCE_BINDING_R2.json",
        *execution_paths,
    ]
    code = sorted((file_entry(root, name) for name in code_paths), key=lambda entry: entry["path"])

    old_bundle = json.loads(
        (root / R1 / "bundle_f3_teacher_seed3_r1/bundle-manifest.json").read_text()
    )
    source_jobs = sorted((root / R1 / "jobs").glob("*.json"))
    specs = []
    names = {}
    for source_job in source_jobs:
        spec = json.loads(source_job.read_text())
        old_name = spec["name"]
        if old_name.startswith("profile_"):
            new_name = "profile_clean_t3_f3_muonall_h384_e3_s3_t4_r2"
            cap = 1800
        elif old_name.startswith("fit_"):
            new_name = "fit_clean_t3_f3_muonall_h384_e3_s3_r2"
            cap = 8500
        else:
            kind = next(
                suffix
                for suffix in ("fit_train", "fit_valid", "full_valid")
                if old_name.endswith("_" + suffix)
            )
            new_name = f"export_clean_t3_f3_muonall_h384_e3_s3_r2_{kind}"
            cap = {"fit_train": 2500, "fit_valid": 900, "full_valid": 1100}[kind]
        replacements = (
            (
                "profile_clean_t3_f3_muonall_h384_e3_s3_t4_r1",
                "profile_clean_t3_f3_muonall_h384_e3_s3_t4_r2",
            ),
            ("clean_t3_f3_muonall_h384_e3_s3_r1", "clean_t3_f3_muonall_h384_e3_s3_r2"),
        )

        spec = replace_strings(spec, replacements)
        spec["name"] = new_name
        spec["code"] = code
        spec["limits"]["timeout_seconds"] = cap
        spec["code_sha256"] = None
        names[new_name] = source_job.name
        (root / PACKET / "jobs").mkdir(exist_ok=True)
        target = root / PACKET / "jobs" / f"{new_name}.json"
        target.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
        specs.append(target)
    sys.path.insert(0, str(root / "scripts"))
    from kaggle_common import canonical_sha256

    for path in specs:
        spec = json.loads(path.read_text())
        spec["code_sha256"] = canonical_sha256(code)
        path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")

    print(json.dumps({"jobs": names, "kernel_dataset": DATASET}, indent=2))
    if not args.build:
        return
    from kaggle_bundle import build_bundle

    external = []
    asset_datasets = set()
    for logical, mapping in sorted(old_bundle["external_mounts"].items()):
        if isinstance(mapping, dict):
            external.append(f"{logical}={mapping['dataset']}:{mapping['filename']}")
            asset_datasets.add(mapping["dataset"])
        else:
            external.append(f"{logical}={mapping}")
    bundle = root / PACKET / "bundle_f3_teacher_seed3_r2"
    build_bundle(root, specs, bundle, DATASET, external, sorted(asset_datasets))
    manifest = json.loads((bundle / "bundle-manifest.json").read_text())
    patch_bootstrap(bundle / "kernel/kernel.py", root, manifest)


if __name__ == "__main__":
    main()
