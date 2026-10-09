from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

WORK = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
OUT = WORK / "out"
REPO = WORK / "wnn"
SOURCE_DATASET = "vladkochetov07/wnn-d02-muon-loss-code-20261009-r5"
OVERLAY_DATASET = "vladkochetov07/wnn-sampled-recipe-runner-code-20261009-r1"
FOLD0_ASSET_DATASET = "vladkochetov07/wnn-c20261004-fold0-assets"
EXPECTED_FOLD0_ASSET_MANIFEST_SHA256 = (
    "19fe0eb6445553c2adff00e821a4fa16ca5969473ca8be420c32cb8e4b4b72c7"
)
EXPECTED_ASSET_MANIFEST_SHA256 = "9ab15bc1efae1b56e045413d3999c350ffdf75edbc541b49f053b41c3bb264c6"
EXPECTED_ARCHIVE_SHA256 = "7c85e1175f3dfff01fc8ab5b2fee81f449cfb873c8f1303fece8b330ad67dcdc"
EXPECTED_CODE_MANIFEST_SHA256 = "dbd98f4b73f5cdc7dd8df137ae9d0bd23495fd8ee2067da02cc94d45699bd6e9"
EXPECTED_SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
EXPECTED_REGISTRY_SHA256 = "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c"
EXPECTED_MEMBERSHIP_SHA256 = "dbd9ea6d7b0b752495745ea1ebd4bd858af9305c40f0ada7de7386c5359f3acd"
SOURCE_COMMIT = "845405d16dc6b3768080aa99c3663a440d343ba5"
ROOT_ADMISSION_BYTES = base64.b64decode("ewogICJhdXRob3JpdHkiOiAicm9vdDogZXhpc3RpbmcgdXNlciBhdXRob3JpemF0aW9uOyBubyBWYXN0OyBzaXggZmluaXRlIHNjcmVlbmVkIHJlY2lwZXMgb25seSIsCiAgImNhY2hlX2tleSI6ICI5NDk1ODdmNzc3OTQ5MzVjMDg1ODliZDVkNWU4MTY0MTA2MmRlNTQ3NTI2YjkyZmMxYjAxYmM0MTA5Y2Q4MmE2IiwKICAiY2FjaGVfbWFuaWZlc3Rfc2hhMjU2IjogIjc0MDBiNzU1MDRmNTkyMDExNTc3ZGQzYjEyNmU5ODEzNzQ5YzIxY2U5MTI2ZTVjNGZhMDRjOGU2YjYxZThmM2IiLAogICJjYW5kaWRhdGVfcG9vbF9zaGEyNTYiOiAiZTVkMzdkMmU1YjQyN2ZlZTM2NDhiYjZjYWUxYWViY2QxODliZjliNGRiNjQyYTFjMDdiNDdjNjQ3NDM5MzVlZSIsCiAgImNvaG9ydF9zcGVjX3NoYTI1NiI6ICIzYzE4YTk4MTEwOWZiODcxMTIyN2RiNzNmZjgwZTQyYjUzMDk1NDAyM2RmZjc3YmZhYzczNzQ1ZjZlMmM1MjZjIiwKICAiY29uZGl0aW9uYWxfZml0X2FsbG93ZWQiOiB0cnVlLAogICJjb250cm9sX2NvbnRyYWN0X3NoYTI1NiI6ICI1YjYzZWQzOWFhNjA0NmU5ZjAyODc4ZTdkY2I0YjQ4OTgyMjQwYmZmYTQ1ZWMwOWI3NDY2ZGZiOWVlYWM2M2Y3IiwKICAiY3JlYXRlZF91dGMiOiAiMjAyNi0xMC0wOVQwNDozMjo1NS44ODkyNDMrMDA6MDAiLAogICJncHVfY291bnQiOiAyLAogICJncHVfbW9kZWwiOiAiTnZpZGlhVGVzbGFUNCIsCiAgImtlcm5lbF9pZCI6ICJ2bGFka29jaGV0b3YwNy93bm4tc2FtcGxlZC1yZWNpcGUtY29ob3J0LTIwMjYxMDA5LXIxIiwKICAibWF4aW11bV9vdXRwdXRfYnl0ZXMiOiAxMjAwMDAwMDAwMCwKICAicHJvdmlkZXJfc2Vjb25kcyI6IDM2MDAsCiAgInB1c2hfYWxsb3dlZCI6IHRydWUsCiAgInJldmlld19iYXNpcyI6ICJJbmRlcGVuZGVudCBMdW5hIGV4YWN0IHBhY2tldCBQQVNTIGFuZCByb290OHRlc3RzUEFTUzsgcm9vdCBkaXNwYXRjaCByZWNlaXB0IGFuZCBmcmVzaCBzb3VyY2UgcmVhZGluZXNzIHJlcXVpcmVkIGJlZm9yZSBwcm92aWRlciBwdXNoIiwKICAicnVubmVyX292ZXJsYXlfZGF0YXNldCI6ICJ2bGFka29jaGV0b3YwNy93bm4tc2FtcGxlZC1yZWNpcGUtcnVubmVyLWNvZGUtMjAyNjEwMDktcjEiLAogICJydW5uZXJfb3ZlcmxheV9tYW5pZmVzdF9zaGEyNTYiOiAiNGU5NTU2M2Q2ODc5MzY0ZmQ4OTI4ZTgyYmY0YmRlYWJmODAxY2NkMmUwZDBmYTkzODBjYjk0NTVhNzZlN2ExNCIsCiAgInJ1bm5lcl93b3JrZXJfc2hhMjU2IjogIjVmZThkYjY0OTQyNjliOTRkN2E1OTM3M2U3MmYwYmYxOTg2OTc5MmMzYzZhOTUxNTNlMTZkNGU0NWM2MDRlMzEiLAogICJydW50aW1lX3NlY29uZHMiOiAzMzAwLAogICJzY2hlbWEiOiAid25uLnNhbXBsZWQtY29ob3J0LnJvb3QtYWRtaXNzaW9uLnYxIiwKICAic291cmNlX2NvZGVfbWFuaWZlc3Rfc2hhMjU2IjogImRiZDk4ZjRiNzNmNWNkYzdkZDhkZjEzN2FlOWQwYmQyMzQ5NWZkOGVlMjA2N2RhMDJjYzk0ZDQ1Njk5YmQ2ZTkiLAogICJzb3VyY2VfY29tbWl0IjogIjg0NTQwNWQxNmRjNmIzNzY4MDgwYWE5OWMzNjYzYTQ0MGQzNDNiYTUiLAogICJzb3VyY2VfY29tbWl0X3J1bm5lciI6ICI0ODdiNmFlNGZiMGM3ZmM0MDIzMmI5YWE5N2VmNzRjNTc1Njc1ZWJlIiwKICAic291cmNlX2RhdGFzZXQiOiAidmxhZGtvY2hldG92MDcvd25uLWQwMi1tdW9uLWxvc3MtY29kZS0yMDI2MTAwOS1yNSIKfQo=")
ARCHIVE_PATH = "wnn_loss_pair_src.tar.gz"
CODE_MANIFEST_PATH = "source-code-manifest.json"
SPLIT_PATH = "research/campaign_2026-09-29/artifacts/splits/fold0.json"
REGISTRY_PATH = "research/campaign_2026-10-04/portable_inputs.json"
MEMBERSHIP_PATH = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "loss_pair_r1_membership_fold0_f03_s20261009.json"
)
ASSET_PATHS = {ARCHIVE_PATH, CODE_MANIFEST_PATH, SPLIT_PATH, REGISTRY_PATH, MEMBERSHIP_PATH}
OVERLAY_MANIFEST_SHA256 = "4e95563d6879364fd8928e82bf4bdeabf801ccd2e0d0fa9380cb9455a76e7a14"
OVERLAY_FILES = {
    "scripts/sampled_recipe_worker.py",
    "scripts/cohort_spec_r1.json",
}
COHORT_SPEC_SHA256 = "3c18a981109fb8711227db73ff80e42b530954023dff77bfac73745f6e2c526c"
RUNNER_WORKER_SHA256 = "5fe8db6494269b94d7a59373e72f0bf19869792c3c6a95153e16d4e45c604e31"
CANDIDATE_POOL_SHA256 = "e5d37d2e5b427fee3648bb6cae1aebcd189bf9b4db642a1c07b47c64743935ee"
CONTROL_CONTRACT_SHA256 = "5b63ed39aa6046e9f02878e7dcb4b48982240bffa45ec09b7466dfb9eeac63f7"
EXPECTED_CACHE_KEY = "949587f77794935c08589bd5d5e81641062de547526b92fc1b01bc4109cd82a6"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def discover_asset_manifest() -> tuple[Path, dict[str, Any]]:
    found = []
    root_depth = len(INPUT.parts)
    if not INPUT.is_dir():
        raise RuntimeError(f"Kaggle input directory missing: {INPUT}")
    for directory, subdirs, files in os.walk(INPUT, followlinks=False):
        current = Path(directory)
        depth = len(current.parts) - root_depth
        subdirs[:] = sorted(subdirs) if depth < 8 else []
        if "assets-manifest.json" not in files:
            continue
        path = current / "assets-manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("dataset") == SOURCE_DATASET:
            found.append((path, value))
    if len(found) != 1:
        raise RuntimeError(f"expected one private code asset manifest, found {len(found)}")
    path, manifest = found[0]
    if sha256_file(path) != EXPECTED_ASSET_MANIFEST_SHA256:
        raise RuntimeError("mounted code asset manifest SHA-256 differs from the sealed pin")
    if manifest.get("version") != 1 or manifest.get("private") is not True:
        raise RuntimeError("mounted code asset manifest has unsupported metadata")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(ASSET_PATHS):
        raise RuntimeError("mounted code asset payload inventory is incomplete")
    if {row.get("path") for row in files} != ASSET_PATHS:
        raise RuntimeError("mounted code asset paths differ from the sealed inventory")
    for row in files:
        filename = str(row.get("filename", ""))
        if Path(filename).name != filename or not filename.endswith(".wnn"):
            raise RuntimeError("private payload filenames must be flat opaque .wnn names")
        payload = path.parent / filename
        if payload.stat().st_size != int(row["size"]) or sha256_file(payload) != row["sha256"]:
            raise RuntimeError(
                f"mounted private payload does not match its manifest: {row['path']}"
            )
    return path.parent, manifest


def discover_overlay_manifest() -> tuple[Path, dict[str, Any]]:
    found = []
    root_depth = len(INPUT.parts)
    for directory, subdirs, files in os.walk(INPUT, followlinks=False):
        current = Path(directory)
        depth = len(current.parts) - root_depth
        subdirs[:] = sorted(subdirs) if depth < 8 else []
        if "assets-manifest.json" not in files:
            continue
        path = current / "assets-manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("dataset") == OVERLAY_DATASET:
            found.append((path, value))
    if len(found) != 1:
        raise RuntimeError(f"expected one sampled-runner overlay manifest, found {len(found)}")
    path, manifest = found[0]
    if sha256_file(path) != OVERLAY_MANIFEST_SHA256:
        raise RuntimeError("sampled-runner overlay manifest differs from the sealed pin")
    if manifest.get("version") != 1 or manifest.get("private") is not True:
        raise RuntimeError("sampled-runner overlay manifest metadata is unsupported")
    files = manifest.get("files")
    if (
        not isinstance(files, list)
        or len(files) != len(OVERLAY_FILES)
        or {row.get("path") for row in files} != OVERLAY_FILES
    ):
        raise RuntimeError("sampled-runner overlay file inventory differs from the sealed list")
    for row in files:
        filename = str(row.get("filename", ""))
        if Path(filename).name != filename or not filename.endswith(".wnn"):
            raise RuntimeError("sampled-runner overlay payload names must be flat opaque .wnn files")
        payload = path.parent / filename
        if payload.stat().st_size != int(row["size"]) or sha256_file(payload) != row["sha256"]:
            raise RuntimeError(f"sampled-runner overlay payload differs: {row['path']}")
    return path.parent, manifest


def install_runner_overlay(asset_root: Path, manifest: dict[str, Any]) -> None:
    rows = {row["path"]: row for row in manifest["files"]}
    for relative in sorted(OVERLAY_FILES):
        row = rows[relative]
        source = asset_root / row["filename"]
        destination = REPO.joinpath(*PurePosixPath(relative).parts)
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"sampled-runner overlay would replace source code: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if destination.stat().st_size != int(row["size"]) or sha256_file(destination) != row["sha256"]:
            raise RuntimeError(f"installed sampled-runner overlay differs: {relative}")
    if sha256_file(REPO / "scripts/sampled_recipe_worker.py") != RUNNER_WORKER_SHA256:
        raise RuntimeError("installed sampled recipe worker differs from its sealed hash")
    if sha256_file(REPO / "scripts/cohort_spec_r1.json") != COHORT_SPEC_SHA256:
        raise RuntimeError("installed cohort recipe manifest differs from its sealed hash")


def discover_fold0_asset_manifest() -> tuple[Path, dict[str, Any]]:
    found = []
    root_depth = len(INPUT.parts)
    for directory, subdirs, files in os.walk(INPUT, followlinks=False):
        current = Path(directory)
        depth = len(current.parts) - root_depth
        subdirs[:] = sorted(subdirs) if depth < 8 else []
        if "assets-manifest.json" not in files:
            continue
        path = current / "assets-manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("dataset") == FOLD0_ASSET_DATASET:
            found.append((path, value))
    if len(found) != 1:
        raise RuntimeError(f"expected one fold-0 auxiliary asset manifest, found {len(found)}")
    path, manifest = found[0]
    if sha256_file(path) != EXPECTED_FOLD0_ASSET_MANIFEST_SHA256:
        raise RuntimeError("mounted fold-0 asset manifest differs from the sealed pin")
    if manifest.get("version") != 1 or manifest.get("private") is not True:
        raise RuntimeError("mounted fold-0 asset manifest has unsupported metadata")
    return path, manifest


def unpack_source(asset_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if REPO.exists() or REPO.is_symlink():
        raise RuntimeError("source extraction directory already exists")
    entries = {row["path"]: row for row in manifest["files"]}
    code_manifest_path = asset_root / entries[CODE_MANIFEST_PATH]["filename"]
    if sha256_file(code_manifest_path) != EXPECTED_CODE_MANIFEST_SHA256:
        raise RuntimeError("code source manifest SHA-256 differs from the sealed pin")
    code_manifest = json.loads(code_manifest_path.read_text(encoding="utf-8"))
    if (
        code_manifest.get("schema") != "wnn.kaggle-loss-pair-code-manifest"
        or code_manifest.get("format_version") != 1
        or code_manifest.get("source_commit") != SOURCE_COMMIT
    ):
        raise RuntimeError("code source manifest has a different schema or source commit")
    archive_path = asset_root / entries[ARCHIVE_PATH]["filename"]
    if sha256_file(archive_path) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("source archive SHA-256 differs from the sealed pin")
    allowed = {row["path"] for row in code_manifest["files"]}
    if not allowed or len(allowed) != len(code_manifest["files"]):
        raise RuntimeError("source code manifest has duplicate or empty paths")
    REPO.mkdir(parents=True)
    seen: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if (
                relative.is_absolute()
                or not relative.parts
                or any(part in ("", ".", "..") for part in relative.parts)
                or member.name not in allowed
                or member.name in seen
                or not member.isfile()
            ):
                raise RuntimeError("source archive contains an unexpected or unsafe entry")
            destination = REPO.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"source archive member is unreadable: {member.name}")
            with source, destination.open("xb") as output:
                shutil.copyfileobj(source, output, 1 << 20)
            seen.add(member.name)
    if seen != allowed:
        raise RuntimeError("source archive is missing manifest-declared files")
    for row in code_manifest["files"]:
        path = REPO.joinpath(*PurePosixPath(row["path"]).parts)
        if path.stat().st_size != int(row["size_bytes"]) or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"extracted source file differs from its pin: {row['path']}")
    for relative, expected in (
        (SPLIT_PATH, EXPECTED_SPLIT_SHA256),
        (REGISTRY_PATH, EXPECTED_REGISTRY_SHA256),
        (MEMBERSHIP_PATH, EXPECTED_MEMBERSHIP_SHA256),
    ):
        row = entries[relative]
        source = asset_root / row["filename"]
        destination = REPO / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256_file(destination) != expected:
            raise RuntimeError(f"extracted data contract differs from its pin: {relative}")
    for relative in ("datasets", "cache", "artifacts/runs"):
        (REPO / relative).mkdir(parents=True, exist_ok=True)
    return code_manifest


def validate_root_admission(admission: Any) -> int:
    """Require the exact reviewed cohort, source closure, cache, control, and overlay pins."""
    if not isinstance(admission, dict):
        raise RuntimeError("sampled-cohort root admission is not a JSON object")
    expected = {
        "schema": "wnn.sampled-cohort.root-admission.v1",
        "kernel_id": "vladkochetov07/wnn-sampled-recipe-cohort-20261009-r1",
        "source_commit": SOURCE_COMMIT,
        "source_dataset": SOURCE_DATASET,
        "candidate_pool_sha256": CANDIDATE_POOL_SHA256,
        "cohort_spec_sha256": COHORT_SPEC_SHA256,
        "source_code_manifest_sha256": EXPECTED_CODE_MANIFEST_SHA256,
        "cache_manifest_sha256": "7400b75504f592011577dd3b126e9813749c21ce9126e5c4fa04c8e6b61e8f3b",
        "cache_key": EXPECTED_CACHE_KEY,
        "control_contract_sha256": CONTROL_CONTRACT_SHA256,
        "runner_overlay_manifest_sha256": OVERLAY_MANIFEST_SHA256,
        "runner_worker_sha256": RUNNER_WORKER_SHA256,
        "runner_overlay_dataset": OVERLAY_DATASET,
    }
    for key, value in expected.items():
        if admission.get(key) != value:
            raise RuntimeError(f"sampled-cohort root admission differs at {key}")
    if admission.get("push_allowed") is not True or admission.get("conditional_fit_allowed") is not True:
        raise RuntimeError("sampled-cohort root admission does not authorize this bounded run")
    runtime = admission.get("runtime_seconds")
    if type(runtime) is not int or not 0 < runtime <= 3300:
        raise RuntimeError("sampled-cohort runtime exceeds the 3300-second worker cap")
    return runtime


def main() -> int:
    marker = {
        "event": "kernel_boot",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
        "input_root": str(INPUT),
    }
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / "boot_marker.json").open("xb") as stream:
        stream.write((json.dumps(marker, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
        print(json.dumps(marker, sort_keys=True), flush=True)
    try:
        asset_root, manifest = discover_asset_manifest()
        code_manifest = unpack_source(asset_root, manifest)
        overlay_root, overlay_manifest = discover_overlay_manifest()
        install_runner_overlay(overlay_root, overlay_manifest)
        admission = json.loads(ROOT_ADMISSION_BYTES)
        runtime = validate_root_admission(admission)
        if os.environ.get("WNN_SAMPLED_COHORT_OFFLINE") == "1":
            atomic_json(
                WORK / "offline_source_receipt.json",
                {
                    "status": "pass",
                    "archive_sha256": EXPECTED_ARCHIVE_SHA256,
                    "code_manifest_sha256": EXPECTED_CODE_MANIFEST_SHA256,
                    "files_verified": len(code_manifest["files"]),
                    "runner_overlay_manifest_sha256": OVERLAY_MANIFEST_SHA256,
                    "runner_worker_sha256": RUNNER_WORKER_SHA256,
                    "cohort_spec_sha256": COHORT_SPEC_SHA256,
                    "fit_admitted": True,
                    "runtime_seconds": runtime,
                },
            )
            return 0
        fold0_manifest_path, _ = discover_fold0_asset_manifest()
        sys.path.insert(0, str(REPO / "src"))
        from wnn.fold0_assets import verify_and_link_fold0_assets

        fold0_receipt_path = WORK / "fold0_assets_validation.json"
        verify_and_link_fold0_assets(
            fold0_manifest_path,
            REPO / REGISTRY_PATH,
            REPO / SPLIT_PATH,
            REPO / "cache",
            fold0_receipt_path,
        )
        fold0_receipt_sha256 = sha256_file(fold0_receipt_path)
        OUT.mkdir(parents=True, exist_ok=False)
        (OUT / "logs").mkdir()
        bootstrap = {
            "schema": "wnn.diversity.sampled-cohort-bootstrap-receipt.v1",
            "format_version": 1,
            "source_dataset": SOURCE_DATASET,
            "asset_manifest_sha256": EXPECTED_ASSET_MANIFEST_SHA256,
            "archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "code_manifest_sha256": EXPECTED_CODE_MANIFEST_SHA256,
            "source_commit": SOURCE_COMMIT,
            "source_file_count": len(code_manifest["files"]),
            "extracted_files_verified": len(code_manifest["files"]),
            "fold0_asset_dataset": FOLD0_ASSET_DATASET,
            "fold0_asset_manifest_sha256": EXPECTED_FOLD0_ASSET_MANIFEST_SHA256,
            "fold0_asset_validation_receipt": str(fold0_receipt_path),
            "fold0_asset_validation_receipt_sha256": fold0_receipt_sha256,
            "runner_overlay_dataset": OVERLAY_DATASET,
            "runner_overlay_manifest_sha256": OVERLAY_MANIFEST_SHA256,
            "runner_worker_sha256": RUNNER_WORKER_SHA256,
            "cohort_spec_sha256": COHORT_SPEC_SHA256,
            "candidate_pool_sha256": CANDIDATE_POOL_SHA256,
            "root_admission_sha256": hashlib.sha256(ROOT_ADMISSION_BYTES).hexdigest(),
            "root_admission": admission,
            "runtime_seconds": runtime,
        }
        atomic_json(OUT / "bootstrap_receipt.json", bootstrap)
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONPATH": str(REPO / "src"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "WNN_SAMPLED_ROOT_ADMISSION_JSON": ROOT_ADMISSION_BYTES.decode(),
            }
        )
        os.execve(
            sys.executable,
            [sys.executable, str(REPO / "scripts/sampled_recipe_worker.py")],
            environment,
        )
    except BaseException as error:
        failure = {
            "state": "failed_bootstrap",
            "error_type": type(error).__name__,
            "error": str(error),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            (WORK / "failure.json").write_text(json.dumps(failure, sort_keys=True, indent=2) + "\n")
            (WORK / "final_status.json").write_text('{"state":"failed_bootstrap"}\n')
            print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        except OSError:
            pass
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
