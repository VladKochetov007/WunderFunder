"""Bind and submit the one reviewed train-fold dictionary preparation."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[1]
PREP = CAMPAIGN / "preparation"
RECORDS = CAMPAIGN / "records/dictionary_f0_v1"
CONFIG_SHA = "b6d21725a4609ede93a75c5be8420afc51833be2be23e2748ee1a80dc5b94b6c"
MANIFEST_SHA = "0d12d3b24839d6a31cdc957dbd20464335b2ad0a7a87b8396c6eb5be7e281a9f"
ARCHIVE_SHA = "43070da5ec5a8d47e751c85f35bd01263422f20ed207504ab8029b46e800d173"
STARTUP_SHA = "0a80a8ab8d86f389788a97b73636e1cffc3b884cf00d8a731dfbf77eca1fb6ee"
GRU_COLD_IMPORT_SHA = "39ba17533b152409754b315a8eb8c53dbdeec0ac2d8657baa05a53259fdc0ec7"
PROFILE_SHA = "cfafd5b90b480e22513b6b13924e2aadd3921154362bfd38620b04bc2ead43da"
PATH_DIAGNOSTIC_SHA = "e9bec80623c928f6436b728dc31a02286c3c56ac2d01cef060c2f6c8fdb9bfd6"
REMOTE_PATH = PREP / "dictionary_fit_admission_remote.py"
REVIEW_PATH = PREP / "DICTIONARY_ADMISSION_REVIEW.json"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _payload() -> dict[str, object]:
    config_path = PREP / "DICTIONARY_PREPARATION.json"
    manifest_path = PREP / "DICTIONARY_SOURCE_MANIFEST.json"
    archive_path = PREP / "DICTIONARY_SOURCE_51A9B168.tar"
    smoke_path = PREP / "DICTIONARY_STARTUP_SMOKE.json"
    cold_import_path = PREP / "GRU_COLD_IMPORT_RECEIPT_51A9B168.json"
    config_raw, manifest_raw, archive_raw = (
        config_path.read_bytes(),
        manifest_path.read_bytes(),
        archive_path.read_bytes(),
    )
    manifest, config = json.loads(manifest_raw), json.loads(config_raw)
    if (
        _sha(config_raw) != CONFIG_SHA
        or _sha(manifest_raw) != MANIFEST_SHA
        or _sha(archive_raw) != ARCHIVE_SHA
        or manifest.get("source_commit") != "51a9b168584e86ca10ebdc0532465d6158ab19d3"
        or manifest.get("archive_sha256") != ARCHIVE_SHA
        or manifest.get("file_count") != 162
        or config.get("status") != "FROZEN_BEFORE_DATA_EXECUTION_NOT_YET_ADMITTED"
    ):
        raise ValueError("frozen dictionary config, source manifest, or archive changed")
    files = {row["path"]: row["sha256"] for row in manifest["files"]}
    sizes = {row["path"]: row["bytes"] for row in manifest["files"]}
    if len(files) != 162 or len(sizes) != 162 or sum(sizes.values()) != 1_556_908:
        raise ValueError("source manifest inventory differs from the published snapshot")
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        prefix = manifest["source_root"] + "/"
        names = {member.name.removeprefix(prefix): member for member in members}
        if len(members) != 162 or set(names) != set(files) or any(not m.isfile() for m in members):
            raise ValueError("source archive membership differs from its manifest")
        for relative, digest in files.items():
            stream = archive.extractfile(names[relative])
            if stream is None:
                raise ValueError(f"source archive member is unreadable: {relative}")
            content = stream.read()
            if len(content) != sizes[relative] or _sha(content) != digest:
                raise ValueError(f"source archive member failed its manifest hash: {relative}")
    smoke_raw, cold_import_raw = smoke_path.read_bytes(), cold_import_path.read_bytes()
    if _sha(smoke_raw) != STARTUP_SHA or _sha(cold_import_raw) != GRU_COLD_IMPORT_SHA:
        raise ValueError("startup smoke receipt changed after review")
    smoke = json.loads(smoke_raw)
    cold_import = json.loads(cold_import_raw)
    if (
        smoke.get("status") != "PASS"
        or smoke.get("source_archive_sha256") != ARCHIVE_SHA
        or smoke.get("reached_split_boundary") is not True
        or smoke.get("created_cache") is not False
    ):
        raise ValueError("actual-entry startup smoke is not passing")
    entrypoints = {
        "scripts/fit_input_representation.py",
        "scripts/predict_run.py",
        "scripts/train_propensity.py",
        "scripts/train_rnn.py",
    }
    runtime_files = cold_import.get("runtime_file_sha256", {})
    if (
        cold_import.get("status") != "PASS"
        or cold_import.get("runtime_wnn_module_count") != 98
        or cold_import.get("all_runtime_origins_inside_isolated_wnn_copy") is not True
        or cold_import.get("required_semantic_modules_present") is not True
        or cold_import.get("missing_semantic_modules") != []
        or cold_import.get("data_read_sentinels", {}).get("calls") != []
        or not all(
            cold_import.get("data_read_sentinels", {}).get(name) is True
            for name in (
                "SequenceFile.read",
                "numpy.load",
                "numpy.memmap",
                "pyarrow.parquet",
                "torch.load",
            )
        )
        or set(cold_import.get("entrypoint_paths", {})) != entrypoints
        or not entrypoints.issubset(runtime_files)
        or not set(runtime_files).issubset(
            entrypoints
            | {name.removeprefix("src/") for name in files if name.startswith("src/wnn/")}
        )
    ):
        raise ValueError(
            "isolated cold-import receipt does not prove the frozen source starts without data I/O"
        )
    for name, digest in runtime_files.items():
        source_name = name if name.startswith("scripts/") else f"src/{name}"
        if files.get(source_name) != digest:
            raise ValueError(f"cold-import source pin differs from the frozen archive: {name}")
    profile_records = CAMPAIGN / "records/profile_v1"
    acceptance_raw = (profile_records / "ACCEPTANCE.json").read_bytes()
    acceptance = json.loads(acceptance_raw)
    if (
        acceptance.get("result_sha256") != PROFILE_SHA
        or acceptance.get("native_integer_exit_status") != 0
    ):
        raise ValueError("the accepted exact-fold data profile is missing or changed")
    captures = [
        path
        for path in (profile_records / "capture_20261019T140338Z").glob("PROFILE_RESULT.json")
        if _sha(path.read_bytes()) == PROFILE_SHA
    ]
    if len(captures) != 1:
        raise ValueError("the exact accepted profile result must have one local capture")
    profile = _json(captures[0])
    if profile.get("valid_source", {}).get("device") != 64512:
        raise ValueError("the profiled B data device identity changed")
    profile_admission = _json(profile_records / "ADMISSION.json")
    data_stats = profile_admission.get("source_data_stats", {})
    diagnostic_path = CAMPAIGN / "records/propensity_f0_e4_v2/PATH_DIAGNOSTIC_STDOUT.raw.json"
    diagnostic_raw = diagnostic_path.read_bytes()
    if _sha(diagnostic_raw) != PATH_DIAGNOSTIC_SHA:
        raise ValueError("the exact six-source resolution diagnostic changed")
    diagnostic = json.loads(diagnostic_raw)
    rows = diagnostic.get("sources", [])
    if (
        diagnostic.get("output_exists") is not False
        or diagnostic.get("run_exists") is not False
        or len(rows) != 6
    ):
        raise ValueError("path diagnostic is not the expected clean pre-fit observation")
    resolutions: dict[str, dict[str, object]] = {}
    for row in rows:
        relative = row["path"].removeprefix(
            "/home/grinchenkoeg/computer-vision-homework-20261018/data/"
        )
        prior = data_stats.get(relative)
        if (
            prior is None
            or any(row.get(key) != prior.get(key) for key in ("bytes", "mtime_ns", "inode"))
            or row.get("device") != profile["valid_source"]["device"]
            or row.get("registered", {}).get("sha256") != prior.get("registered_sha256")
        ):
            raise ValueError(f"profile and path diagnostic disagree for {relative}")
        resolutions[relative] = {
            key: row[key]
            for key in ("path", "resolved", "islink", "device", "inode", "mtime_ns", "bytes")
        }
    if set(resolutions) != set(data_stats):
        raise ValueError("profile and path diagnostic do not cover the same six files")
    fit_launch = _json(CAMPAIGN / "records/propensity_f0_e4_v3/LAUNCH.json")
    fit_admission = _json(CAMPAIGN / "records/propensity_f0_e4_v3/ADMISSION.json")
    stdout = fit_launch.get("systemctl_after_submission", {}).get("stdout", "")
    invocation = "85b87a83f96945f4b78908f124435a2b"
    if (
        fit_admission.get("unit") != "computer-vision-homework-fg19-propensity-f0-e4-v3.service"
        or f"InvocationID={invocation}" not in stdout
    ):
        raise ValueError("the only allowed concurrent fit does not match its reviewed invocation")
    remote_raw = REMOTE_PATH.read_bytes()
    binder_raw = Path(__file__).read_bytes()
    review = _json(REVIEW_PATH)
    expected_review = {
        "status": "PASS",
        "remote_admission_sha256": _sha(remote_raw),
        "local_binder_sha256": _sha(binder_raw),
        "configuration_sha256": CONFIG_SHA,
        "source_manifest_sha256": MANIFEST_SHA,
        "source_archive_sha256": ARCHIVE_SHA,
        "startup_smoke_sha256": _sha(smoke_raw),
        "profile_result_sha256": PROFILE_SHA,
        "cold_import_receipt_sha256": GRU_COLD_IMPORT_SHA,
        "path_diagnostic_sha256": PATH_DIAGNOSTIC_SHA,
    }
    if any(review.get(key) != value for key, value in expected_review.items()):
        raise ValueError("independent admission review does not bind current files and evidence")
    return {
        "source_commit": manifest["source_commit"],
        "source_files": files,
        "source_sizes": sizes,
        "archive_sha256": ARCHIVE_SHA,
        "source_manifest_sha256": MANIFEST_SHA,
        "archive_bytes": len(archive_raw),
        "archive_b64": base64.b64encode(archive_raw).decode("ascii"),
        "configuration": config,
        "configuration_raw_b64": base64.b64encode(config_raw).decode("ascii"),
        "configuration_sha256": CONFIG_SHA,
        "cold_import_receipt_sha256": GRU_COLD_IMPORT_SHA,
        "profile_data_device": profile["valid_source"]["device"],
        "profile_data_stats": data_stats,
        "profile_data_resolved_paths": resolutions,
        "fit_invocation_id": invocation,
        "profile_result_sha256": PROFILE_SHA,
        "profile_admission_sha256": _sha((profile_records / "ADMISSION.json").read_bytes()),
        "path_diagnostic_sha256": PATH_DIAGNOSTIC_SHA,
        "portable_source_registry_sha256": "1f39cbf8b242495e18baaf5d36eeac907c0ad655e951b0efa6a48580b7699378",
        "review_sha256": _sha(REVIEW_PATH.read_bytes()),
    }


def main() -> None:
    payload = _payload()
    encoded = base64.b64encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    template = REMOTE_PATH.read_text()
    marker = 'PAYLOAD_B64 = "__PAYLOAD_B64__"'
    if template.count(marker) != 1:
        raise ValueError("remote payload marker is missing or duplicated")
    remote = template.replace(marker, "PAYLOAD_B64 = " + repr(encoded.decode("ascii")), 1)
    compile(remote, "dictionary_fit_admission_remote.py", "exec")
    if len(remote) > 8_000_000:
        raise ValueError("remote admission payload exceeded the bounded transfer size")
    RECORDS.mkdir(mode=0o700, parents=False, exist_ok=False)
    (RECORDS / "REMOTE_SUBMISSION.py").write_text(remote)
    (RECORDS / "BINDING.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    command = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ConnectionAttempts=1",
        "beleriand",
        "python3 -B -",
    ]
    try:
        result = subprocess.run(command, input=remote.encode(), capture_output=True, timeout=150)
    except subprocess.TimeoutExpired as error:
        (RECORDS / "STDOUT.raw").write_bytes(error.stdout or b"")
        (RECORDS / "STDERR.raw").write_bytes(error.stderr or b"")
        (RECORDS / "TRANSPORT.json").write_text(
            json.dumps(
                {
                    "returncode": None,
                    "timed_out": True,
                    "unknown_outcome_policy": "inspect exact source, log directory, output paths, and unit; never blindly retry",
                },
                indent=2,
            )
            + "\n"
        )
        raise RuntimeError(
            "SSH timed out; inspect exact records and remote paths before any retry"
        ) from error
    (RECORDS / "STDOUT.raw").write_bytes(result.stdout)
    (RECORDS / "STDERR.raw").write_bytes(result.stderr)
    (RECORDS / "TRANSPORT.json").write_text(
        json.dumps(
            {
                "returncode": result.returncode,
                "unknown_outcome_policy": "inspect exact source, log directory, output paths, and unit; never blindly retry",
            },
            indent=2,
        )
        + "\n"
    )
    try:
        returned = json.loads(result.stdout)
    except json.JSONDecodeError:
        returned = None
    if isinstance(returned, dict):
        if set(returned) != {"ADMISSION.json", "LAUNCH.json"}:
            raise ValueError("remote helper returned unexpected evidence members")
        for name, encoded_raw in returned.items():
            (RECORDS / name).write_bytes(base64.b64decode(encoded_raw, validate=True))
    if result.returncode:
        raise RuntimeError(
            "dictionary submission failed or is unknown; inspect preserved raw records"
        )
    print(
        json.dumps(
            {
                "records": str(RECORDS),
                "status": "submitted; inspect durable admission and launch records",
            }
        )
    )


if __name__ == "__main__":
    main()
