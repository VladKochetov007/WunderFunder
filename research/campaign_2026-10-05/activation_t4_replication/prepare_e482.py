"""Build a content-bound E482 private Kaggle draft without uploading it."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TEMPLATE = ROOT / "research/campaign_2026-10-05/w384_v7/bundle/kernel/kernel.py"
DATASET_SLUG = "vladkochetov07/e482-t4-sixfit-20261005"
INPUT_DATASET = "vladkochetov07/wunder-funder"
DATASET_SHA256 = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
COHORT_SHA256 = "068103c71663bc445aab943e314c5be1d9a094f7f83456fdb406fbdc1012a6f2"
E481_SHA256 = "3d97d2d07b116b3cf04b365c33e3d3caed2c96c35108bd28f3a6ff63b54d65cc"
E470_SHA256 = "871444301fccfa5fda5356dadf80e9aba1205aa91f798839c4bcb8c6704d05b9"
E474_SHA256 = "40d0f309c54c20f54614f413bedbb99a00df11ad4a62c029a0ebf709e9708068"
SHAPES = ("baseline", "sine", "cubic")
OPTIMIZERS = ("adamw", "muon_adamw")
EXTERNAL = {"datasets/valid.parquet": {"dataset": INPUT_DATASET, "filename": "valid.parquet"}}


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks for immutable packet identities."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(root: Path, relative: str) -> dict[str, str | int]:
    """Create a safe root-relative path, byte-count and SHA256 descriptor."""
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError(f"unsafe packet path: {relative}")
    source = root / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    return {"path": relative, "size": source.stat().st_size, "sha256": sha256_file(source)}


def packet_files(root: Path) -> tuple[list[dict], list[dict]]:
    """Return the narrow runtime closure and two explicitly bound input files."""
    code_paths = [
        "research/campaign_2026-10-05/activation_t4_replication/e482_remote.py",
        "research/campaign_2026-10-05/activation_t4_replication/E482_DESIGN.md",
        "research/campaign_2026-10-05/recurrent_activation_curvature/fit_recurrent_curvature.py",
        "research/campaign_2026-10-05/activation_optimizer/fit_activation_optimizer.py",
        "research/campaign_2026-10-05/activation_placement/fit_activation_placement.py",
    ]
    code_paths.extend(
        path.relative_to(root).as_posix() for path in sorted((root / "src/wnn").rglob("*.py"))
    )
    code = [descriptor(root, name) for name in code_paths]
    inputs = [
        descriptor(root, "research/campaign_2026-10-05/target_mechanics/cohort.json"),
        {
            "path": "datasets/valid.parquet",
            "size": (root / "datasets/valid.parquet").stat().st_size,
            "sha256": DATASET_SHA256,
        },
    ]
    if (
        next(item for item in code if item["path"].endswith("fit_recurrent_curvature.py"))["sha256"]
        != E481_SHA256
    ):
        raise ValueError("frozen E481 actor identity changed")
    if (
        next(item for item in code if item["path"].endswith("fit_activation_optimizer.py"))[
            "sha256"
        ]
        != E470_SHA256
    ):
        raise ValueError("frozen E470 helper identity changed")
    if (
        next(item for item in code if item["path"].endswith("fit_activation_placement.py"))[
            "sha256"
        ]
        != E474_SHA256
    ):
        raise ValueError("frozen E474 helper identity changed")
    if inputs[0]["sha256"] != COHORT_SHA256:
        raise ValueError("frozen E482 cohort identity changed")
    return code, inputs


def e482_spec(code: list[dict], inputs: list[dict]) -> dict:
    """Describe the exact six-cell treatment and conservative execution envelope."""
    return {
        "schema": "e482-private-t4-research-v1",
        "identity": "E482-seed271828-six-recipes-two-T4",
        "design": "E482_DESIGN.md",
        "seed": 271828,
        "optimizers": list(OPTIMIZERS),
        "shapes": list(SHAPES),
        "cells": [
            {"seed": 271828, "optimizer": optimizer, "shape": shape}
            for optimizer in OPTIMIZERS
            for shape in SHAPES
        ],
        "architecture": {
            "features": 42,
            "hidden": 64,
            "layers": 2,
            "parameters": 61058,
            "batch": 4,
            "chunk_rows": 256,
            "updates": 474,
            "candidate_maps": {
                "baseline": "u",
                "sine": "u+0.1*sin(pi*u)",
                "cubic": "u+0.1*pi*u*(1-u*u)",
            },
        },
        "cohort": {
            "dataset": INPUT_DATASET,
            "path": "datasets/valid.parquet",
            "dataset_sha256": DATASET_SHA256,
            "cohort_path": inputs[0]["path"],
            "cohort_sha256": COHORT_SHA256,
            "fit_components": 24,
            "test_components": 16,
            "test_rows_read": False,
        },
        "source": {
            "e481_sha256": E481_SHA256,
            "e470_sha256": E470_SHA256,
            "e474_sha256": E474_SHA256,
            "code": code,
            "inputs": inputs,
        },
        "execution": {
            "accelerator": "NvidiaTeslaT4",
            "required_visible_t4_count": 2,
            "optimizer_to_device": {"adamw": 0, "muon_adamw": 1},
            "processes": 2,
            "cpus_per_worker": 1,
            "minimum_available_cpus": 3,
            "ram_per_worker_bytes": 6 * 1024**3,
            "controller_support_reserve_bytes": 1 * 1024**3,
            "host_admission_fraction": 0.8,
            "host_stop_fraction": 0.9,
            "cuda_allocator_bytes_per_worker": 2 * 1024**3,
            "physical_gpu_stop_bytes": 3_000_000_000,
            "kernel_timeout_seconds": 3600,
            "internal_watchdog_seconds": 3500,
            "maximum_device_hours": 2,
            "new_admission_cutoff_utc": "2026-10-05T07:00:00Z",
        },
        "gate": {
            "before_fits": [
                "all six actual T4 B4x256 updates finite and changed",
                "native baseline output/state/input-gradient/parameter-gradient parity",
                "clipped AdamW and Muon optimizer-step parity on both T4s",
                "exact source/cohort/data hashes and resource admission",
            ],
            "probe_failure_blocks_every_full_fit": True,
        },
    }


def canonical_hash(value: object) -> str:
    """Hash a canonical JSON value for the spec envelope."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_tar_members(
    members: list[tarfile.TarInfo], allowed: set[str], external: set[str]
) -> None:
    """Reject unsafe, linked, duplicate, undeclared or omitted archive files."""
    expected = allowed - external
    seen = set()
    for member in members:
        path = PurePosixPath(member.name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in member.name
            or path.as_posix() != member.name
        ):
            raise ValueError(f"unsafe archive path: {member.name}")
        if not member.isfile() or member.name not in expected or member.name in seen:
            raise ValueError(f"unsupported or duplicate archive member: {member.name}")
        seen.add(member.name)
    if seen != expected:
        raise ValueError("archive member inventory is incomplete")


def build_draft(root: Path, destination: Path) -> dict:
    """Assemble a private draft bundle and notebook bootstrap without upload."""
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite E482 packet: {destination}")
    code, inputs = packet_files(root)
    spec = e482_spec(code, inputs)
    spec["spec_sha256"] = canonical_hash(spec)
    file_items = code + inputs
    names = [item["path"] for item in file_items]
    if len(set(names)) != len(names) or set(EXTERNAL) != {
        item["path"] for item in inputs if item["path"] in EXTERNAL
    }:
        raise ValueError("duplicate or unexpected E482 file/external input declarations")
    destination.mkdir(parents=True)
    bundle = destination / "bundle"
    bundle.mkdir()
    archive = bundle / "bundle.wnn"
    spec_bytes = (json.dumps(spec, sort_keys=True, indent=2) + "\n").encode()
    with (
        archive.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as tar,
    ):
        for item in sorted(file_items, key=lambda value: str(value["path"])):
            relative = str(item["path"])
            if relative in EXTERNAL:
                continue
            file_path = root / relative
            if file_path.stat().st_size != item["size"] or sha256_file(file_path) != item["sha256"]:
                raise ValueError(f"file changed while bundling: {relative}")
            info = tarfile.TarInfo(relative)
            info.size = int(item["size"])
            info.mode = 0o644
            with file_path.open("rb") as source:
                tar.addfile(info, source)
        info = tarfile.TarInfo("kaggle_specs/e482_spec.json")
        info.size, info.mode = len(spec_bytes), 0o644
        tar.addfile(info, io.BytesIO(spec_bytes))

    manifest = {
        "version": 1,
        "dataset": DATASET_SLUG,
        "private": True,
        "archive_sha256": sha256_file(archive),
        "archive_bytes": archive.stat().st_size,
        "expanded_bytes": sum(
            int(item["size"]) for item in code + inputs if item["path"] not in EXTERNAL
        )
        + len(spec_bytes),
        "files": code + inputs,
        "external_mounts": EXTERNAL,
        "spec_paths": ["kaggle_specs/e482_spec.json"],
        "spec_sha256": spec["spec_sha256"],
        "no_raw_data_in_archive": True,
        "source_template_sha256": sha256_file(TEMPLATE),
    }
    (bundle / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    manifest_sha256 = sha256_file(bundle / "bundle-manifest.json")
    with tarfile.open(archive, "r:gz") as tar:
        validate_tar_members(
            tar.getmembers(),
            {item["path"] for item in manifest["files"]} | set(manifest["spec_paths"]),
            set(manifest["external_mounts"]),
        )
    (bundle / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": DATASET_SLUG,
                "title": "E482 T4 six-recipe private research bundle",
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    kernel_dir = bundle / "kernel"
    kernel_dir.mkdir()
    template = TEMPLATE.read_text()
    kernel = re.sub(
        r"^BUNDLE_SLUG = .*?$",
        f"BUNDLE_SLUG = {DATASET_SLUG.split('/')[1]!r}",
        template,
        flags=re.M,
    )
    kernel = re.sub(
        r"^EXPECTED_ARCHIVE_SHA256 = .*?$",
        f"EXPECTED_ARCHIVE_SHA256 = {manifest['archive_sha256']!r}",
        kernel,
        flags=re.M,
    )
    kernel = re.sub(
        r"^EXPECTED_EXTERNAL_MOUNTS = .*?$",
        f"EXPECTED_EXTERNAL_MOUNTS = {EXTERNAL!r}",
        kernel,
        flags=re.M,
    )
    kernel = kernel.replace(
        f"EXPECTED_EXTERNAL_MOUNTS = {EXTERNAL!r}",
        f"EXPECTED_EXTERNAL_MOUNTS = {EXTERNAL!r}\nEXPECTED_MANIFEST_SHA256 = {manifest_sha256!r}",
        1,
    )
    old_manifest_read = "    manifest = json.loads(matches[0].read_text())"
    new_manifest_read = (
        "    manifest_bytes = matches[0].read_bytes()\n"
        "    if hashlib.sha256(manifest_bytes).hexdigest() != EXPECTED_MANIFEST_SHA256:\n"
        '        raise ValueError("Private bundle manifest identity mismatch")\n'
        "    manifest = json.loads(manifest_bytes)"
    )
    if kernel.count(old_manifest_read) != 1:
        raise ValueError("reviewed bootstrap manifest-read boundary changed")
    kernel = kernel.replace(old_manifest_read, new_manifest_read)
    function_start = kernel.index("def find_external_candidate(")
    function_end = kernel.index("def _kernel_work()", function_start)
    external_resolver = '''def find_external_candidate(input_root: Path, dataset: str, filename: str) -> Path:
    """Resolve a single regular file inside the authorized mounted dataset root."""
    parts = dataset.split("/") if isinstance(dataset, str) else []
    if (
        len(parts) != 2
        or not all(parts)
        or not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or filename in {".", ".."}
    ):
        raise ValueError("Invalid external dataset file descriptor")
    slug = parts[1]
    candidates = list(input_root.glob(f"**/{slug}/{filename}"))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one authorized input: {dataset}/{filename}")
    input_root_real = input_root.resolve(strict=True)
    candidate = candidates[0]
    dataset_roots = [parent for parent in candidate.parents if parent.name == slug]
    if not dataset_roots:
        raise ValueError("Mounted file has no authorized dataset root")
    dataset_root = dataset_roots[0].resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if (
        not dataset_root.is_relative_to(input_root_real)
        or not resolved.is_file()
        or not resolved.is_relative_to(dataset_root)
    ):
        raise ValueError("Mounted input resolves outside the authorized dataset root")
    return resolved


'''
    kernel = kernel[:function_start] + external_resolver + kernel[function_end:]
    archive_guard = '''def validate_archive_members(members: list, allowed: set[str], external: set[str]) -> None:
    """Reject unsafe, linked, duplicate, undeclared and omitted archive files."""
    expected = allowed - external
    seen = set()
    for member in members:
        path = PurePosixPath(member.name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\\\" in member.name
            or path.as_posix() != member.name
        ):
            raise ValueError("Archive path escapes experiment directory")
        if not member.isfile() or member.name not in expected or member.name in seen:
            raise ValueError("Unsupported, undeclared or duplicate archive member")
        seen.add(member.name)
    if seen != expected:
        raise ValueError("Archive member set is incomplete")


'''
    kernel = kernel.replace(
        "def _kernel_work() -> None:\n", archive_guard + "def _kernel_work() -> None:\n", 1
    )
    archive_loop = '    with tarfile.open(archive, "r:gz") as tar:\n        for member in tar:\n'
    guarded_archive_loop = (
        '    with tarfile.open(archive, "r:gz") as tar:\n'
        "        archive_entries = tar.getmembers()\n"
        '        validate_archive_members(archive_entries, allowed, set(manifest["external_mounts"]))\n'
        "        for member in archive_entries:\n"
    )
    if kernel.count(archive_loop) != 1:
        raise ValueError("reviewed bootstrap archive loop changed")
    kernel = kernel.replace(archive_loop, guarded_archive_loop)
    if kernel.count("    archive_members = seen\n") != 1:
        raise ValueError("reviewed bootstrap extracted-member checkpoint changed")
    kernel = kernel.replace(
        "    archive_members = seen\n",
        '    if seen != allowed - set(manifest["external_mounts"]):\n'
        '        raise ValueError("Extracted archive member inventory is incomplete")\n'
        "    archive_members = seen\n",
        1,
    )
    old_command = 'command = [sys.executable, str(root / "research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v4/phase_runner.py")]'
    new_command = (
        "command = [sys.executable, str(root / "
        '"research/campaign_2026-10-05/activation_t4_replication/e482_remote.py"), '
        '"--output", "/kaggle/working/e482-results"]'
    )
    if kernel.count(old_command) != 1 or kernel.count("timeout_seconds=25_180") != 1:
        raise ValueError("reviewed v7 bootstrap template shape changed")
    kernel = kernel.replace(old_command, new_command).replace(
        "timeout_seconds=25_180", "timeout_seconds=3_500"
    )
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(kernel)
    formatter = root / ".venv/bin/ruff"
    if not formatter.is_file():
        raise FileNotFoundError("the packet builder requires the repository .venv/bin/ruff")
    subprocess.run([str(formatter), "format", str(kernel_path)], check=True, cwd=root)
    metadata = {
        "id": f"vladkochetov07/{DATASET_SLUG.split('/')[1]}-runner",
        "title": "E482 six-recipe private T4 replication (draft)",
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [DATASET_SLUG, INPUT_DATASET],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    receipt = {
        "status": "local_source_packet_built_not_uploaded_not_launched",
        "dataset_slug": DATASET_SLUG,
        "kernel_slug": metadata["id"],
        "archive_sha256": manifest["archive_sha256"],
        "manifest_sha256": sha256_file(bundle / "bundle-manifest.json"),
        "kernel_sha256": sha256_file(kernel_path),
        "metadata_sha256": sha256_file(kernel_dir / "kernel-metadata.json"),
        "spec_sha256": spec["spec_sha256"],
        "template_sha256": manifest["source_template_sha256"],
        "archive_members": len(code) + len(inputs) - 1 + 1,
        "external_input": "datasets/valid.parquet",
        "private": True,
        "contains_raw_dataset": False,
    }
    (destination / "BUILD_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def main() -> None:
    """Build the reviewable packet locally; this command never calls Kaggle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--destination", type=Path, default=HERE / "prepared_v1")
    args = parser.parse_args()
    receipt = build_draft(args.root.resolve(), args.destination.resolve())
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
