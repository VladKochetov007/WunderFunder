"""Seal a quota-bounded fit/export successor using the existing packet tools."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

R1 = Path("research/campaign_2026-10-13/strategy_execution_r1/first_cohort_f3_teachers_r1")
PACKET = Path("research/campaign_2026-10-14/cpu_student_frontier_r1/f3_teacher_fit_r5")
DATASET = "vladkochetov07/wnn-f3-teacher-s3-20261014-r5"


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
    source = kernel.read_text().replace(
        "KERNEL_PROCESS_WALL_SECONDS_MAX = 43_200", "KERNEL_PROCESS_WALL_SECONDS_MAX = 10_000"
    )
    start = source.index("    command = [\n")
    marker = '    finally:\n        release_mounts(root, manifest["external_mounts"])\n'
    end = source.index(marker, start) + len(marker)
    dispatch = (
        f'    command = [sys.executable, str(root / "{PACKET.as_posix()}/kernel_support/f3_teacher_driver_r5.py")]\n'
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
    if 'str(root / "' + PACKET.as_posix() + '/kernel_support/f3_teacher_driver_r5.py")' in source:
        raise ValueError("bootstrap already contains the R5 dispatch")
    kernel.write_text(source[:start] + dispatch + source[end:])
    binding = {
        "schema": "wnn.strategy.f3-teacher-entrypoint-binding.r5",
        "bundle_dataset": DATASET,
        "kernel_dataset_id": f"{DATASET}-runner",
        "title_derived_slug": DATASET.rsplit("/", 1)[1],
        "code_file": "kernel/kernel.py",
        "kernel_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
        "archive_sha256": manifest["archive_sha256"],
        "dispatch": "verified bootstrap starts the quota-admitted profile-reuse→fit-only driver and passes exact project root/deadline",
        "kernel_source": kernel.relative_to(root).as_posix(),
    }
    (root / PACKET / "ENTRYPOINT_BINDING_R5.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    r1_binding = json.loads((root / R1 / "SOURCE_BINDING_R1.json").read_text())
    sys.path.insert(0, str(root / PACKET / "kernel_support"))
    from profile_reuse import validate_profile_reuse

    profile_reuse_receipt = validate_profile_reuse(root)
    profile_evidence = sorted(
        f"{PACKET.as_posix()}/profile_reuse_r2/{path.name}"
        for path in (root / PACKET / "profile_reuse_r2").iterdir()
        if path.is_file()
    )
    reuse_binding = {
        "schema": "wnn.strategy.f3-teacher-profile-reuse-binding.r5",
        "validation": profile_reuse_receipt,
        "evidence": [file_entry(root, name) for name in profile_evidence],
    }
    (root / PACKET / "PROFILE_REUSE_BINDING_R5.json").write_text(
        json.dumps(reuse_binding, indent=2, sort_keys=True) + "\n"
    )
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
            + [
                f"{PACKET.as_posix()}/kernel_support/f3_teacher_driver_r5.py",
                f"{PACKET.as_posix()}/kernel_support/profile_reuse.py",
            ]
        )
    )
    source_binding = {
        "schema": "wnn.strategy.f3-teacher-source-binding.r5",
        "reference_source_manifest": file_entry(
            root, f"{R1.as_posix()}/source_snapshot/SOURCE_MANIFEST_H390.json"
        ),
        "execution_code": [file_entry(root, name) for name in execution_paths],
        "reference_r1_source_binding_sha256": hashlib.sha256(
            (root / R1 / "SOURCE_BINDING_R1.json").read_bytes()
        ).hexdigest(),
        "description": "R5 reuses the exact independently trained source snapshot and successful R2 T4 resource profile; it runs only the unchanged teacher fit under a shorter quota envelope. All predictions are owned checkpoint successors after collection; no teacher cache is claimed complete.",
    }
    (root / PACKET / "SOURCE_BINDING_R5.json").write_text(
        json.dumps(source_binding, indent=2, sort_keys=True) + "\n"
    )
    code_paths = [
        f"{PACKET.as_posix()}/README.md",
        f"{PACKET.as_posix()}/KAGGLE_QUOTA_GUARD_R5.json",
        f"{PACKET.as_posix()}/OUTPUT_COLLECTION_PLAN_R5.json",
        f"{PACKET.as_posix()}/PROFILE_REUSE_BINDING_R5.json",
        f"{PACKET.as_posix()}/SOURCE_BINDING_R5.json",
        *profile_evidence,
        *execution_paths,
    ]
    code = sorted((file_entry(root, name) for name in code_paths), key=lambda entry: entry["path"])

    old_bundle = json.loads(
        (root / R1 / "bundle_f3_teacher_seed3_r1/bundle-manifest.json").read_text()
    )
    source_jobs = sorted((root / R1 / "jobs").glob("*.json"))
    specs = []
    names = {}
    jobs_dir = root / PACKET / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    for stale_spec in jobs_dir.glob("*.json"):
        stale_spec.unlink()
    for source_job in source_jobs:
        spec = json.loads(source_job.read_text())
        old_name = spec["name"]
        replacements = (
            (
                "profile_clean_t3_f3_muonall_h384_e3_s3_t4_r1",
                "profile_clean_t3_f3_muonall_h384_e3_s3_t4_r2",
            ),
            ("clean_t3_f3_muonall_h384_e3_s3_r1", "clean_t3_f3_muonall_h384_e3_s3_r5"),
        )
        if old_name.startswith("profile_"):
            continue
        elif old_name.startswith("fit_"):
            new_name = "fit_clean_t3_f3_muonall_h384_e3_s3_r5"
            cap = 8500
        else:
            continue  # This tranche preserves the teacher checkpoint; exports follow collection.

        spec = replace_strings(spec, replacements)
        spec["name"] = new_name
        spec["code"] = code
        spec["limits"]["timeout_seconds"] = cap
        spec["limits"]["ram_bytes"] = 6_000_000_000
        spec["code_sha256"] = None
        names[new_name] = source_job.name
        target = jobs_dir / f"{new_name}.json"
        target.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
        specs.append(target)
    sys.path.insert(0, str(root / "scripts"))
    from kaggle_common import canonical_sha256

    for path in specs:
        spec = json.loads(path.read_text())
        spec["code_sha256"] = canonical_sha256(code)
        path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")

    print(
        json.dumps(
            {"jobs": names, "kernel_dataset": DATASET, "profile_reuse": profile_reuse_receipt},
            indent=2,
        )
    )
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
    bundle = root / PACKET / "bundle_f3_teacher_seed3_r5"
    build_bundle(root, specs, bundle, DATASET, external, sorted(asset_datasets))
    manifest = json.loads((bundle / "bundle-manifest.json").read_text())
    patch_bootstrap(bundle / "kernel/kernel.py", root, manifest)


if __name__ == "__main__":
    main()
