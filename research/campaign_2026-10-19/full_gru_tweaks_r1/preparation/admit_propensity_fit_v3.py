"""Bind the reviewed clean-fit admission and exact Beleriand evidence."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[1]
PREPARATION = CAMPAIGN / "preparation"
PROFILE_RECORDS = CAMPAIGN / "records/profile_v1"
RECORDS = CAMPAIGN / "records/propensity_f0_e4_v3"
PRIOR_RECORDS = CAMPAIGN / "records/propensity_f0_e4_v2"
B = "/home/grinchenkoeg/computer-vision-homework-20261018"
SOURCE_PREFIX = B + "/source-fg19-propensity-235db75/"
PACKET_PREFIX = B + "/research/full_gru_tweaks_r1/profile_v1/"
COMMIT = "235db75ab91e24c69f21e637a83a48512dc674fe"
PROFILE_RESULT_SHA256 = "cfafd5b90b480e22513b6b13924e2aadd3921154362bfd38620b04bc2ead43da"
SPLIT_SHA256 = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
REGISTRY_SHA256 = "1f39cbf8b242495e18baaf5d36eeac907c0ad655e951b0efa6a48580b7699378"
PROFILE_SHA256 = "d5c2c3030f46ba5ea4a22a9d4ad0e5a5ec2af7d53ab729742fc7068d706f4376"
PROFILE_CORE_SHA256 = "fe9f3551085244dd278f89fc61ec0a95853d8f4a71b5b5c8d2d17bb40086d2af"
PATH_DIAGNOSTIC_SHA256 = "e9bec80623c928f6436b728dc31a02286c3c56ac2d01cef060c2f6c8fdb9bfd6"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _raw_bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    return value.encode() if isinstance(value, str) else value


def _bind_files(manifest: dict[str, object]) -> tuple[dict[str, str], dict[str, str]]:
    if manifest.get("commit") != COMMIT:
        raise ValueError("source manifest commit differs from the published source")
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) != 157:
        raise ValueError("source manifest does not contain the expected 157 files")
    source: dict[str, str] = {}
    packet: dict[str, str] = {}
    for path, digest in files.items():
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError("source manifest contains a non-string path or hash")
        if path.startswith(SOURCE_PREFIX):
            relative = path.removeprefix(SOURCE_PREFIX)
            source[relative] = digest
        elif path.startswith(PACKET_PREFIX):
            relative = path.removeprefix(PACKET_PREFIX)
            packet[relative] = digest
        else:
            raise ValueError(f"source manifest contains an out-of-scope path: {path}")
        if ".." in Path(relative).parts or len(digest) != 64:
            raise ValueError("source manifest contains an unsafe path or malformed digest")
    if len(source) != 153 or set(packet) != {
        "propensity_profile.py",
        "propensity_profile_core.py",
        "PROPENSITY_PREREQUISITE.json",
        "PROFILE_REVIEW.json",
    }:
        raise ValueError("source manifest does not bind the expected source and profile files")
    return source, packet


def _accepted_profile(acceptance: dict[str, object]) -> dict[str, object]:
    expected_status = "PASS_ENGINEERING_PROFILE_READY_FOR_BOUNDED_PREREQUISITE_FIT"
    if (
        acceptance.get("status") != expected_status
        or acceptance.get("native_integer_exit_status") != 0
        or acceptance.get("result_sha256") != PROFILE_RESULT_SHA256
    ):
        raise ValueError("the actual-shape profile acceptance is not the expected passing receipt")
    captures = list(PROFILE_RECORDS.glob("capture_*/PROFILE_RESULT.json"))
    matches = [path for path in captures if _sha256(path.read_bytes()) == PROFILE_RESULT_SHA256]
    if len(matches) != 1:
        raise ValueError("the accepted profile result must have one exact local capture")
    profile = _read_json(matches[0])
    if (
        profile.get("status") != "completed_engineering_profile_only"
        or profile.get("source_commit") != COMMIT
        or profile.get("split_manifest_sha256") != SPLIT_SHA256
        or profile.get("profile_sha256") != PROFILE_SHA256
        or profile.get("profile_core_sha256") != PROFILE_CORE_SHA256
    ):
        raise ValueError("the captured profile result does not bind the accepted source")
    return profile


def _resolved_data_map(
    profile_admission: dict[str, object], device: int
) -> dict[str, dict[str, object]]:
    diagnostic_path = PRIOR_RECORDS / "PATH_DIAGNOSTIC_STDOUT.raw.json"
    raw = diagnostic_path.read_bytes()
    if _sha256(raw) != PATH_DIAGNOSTIC_SHA256:
        raise ValueError("the captured B path diagnostic bytes changed")
    diagnostic = json.loads(raw)
    expected = profile_admission.get("source_data_stats", {})
    rows = diagnostic.get("sources", [])
    if (
        diagnostic.get("unit") != "LoadState=not-found\nActiveState=inactive\nSubState=dead\n"
        or diagnostic.get("output_exists") is not False
        or diagnostic.get("run_exists") is not False
        or len(rows) != 6
        or set(expected) != {row.get("path", "").removeprefix(B + "/data/") for row in rows}
    ):
        raise ValueError("path diagnostic is not the captured no-start V2 observation")
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        relative = row["path"].removeprefix(B + "/data/")
        identity = expected[relative]
        registered = row.get("registered", {})
        if (
            row.get("path") != f"{B}/data/{relative}"
            or not Path(row.get("resolved", "")).is_absolute()
            or type(row.get("islink")) is not bool
            or row.get("device") != device
            or any(row.get(key) != identity.get(key) for key in ("bytes", "inode", "mtime_ns"))
            or registered.get("sha256") != identity.get("registered_sha256")
            or registered.get("original_identity", {}).get("bytes") != identity.get("bytes")
            or registered.get("original_identity", {}).get("mtime_ns") != identity.get("mtime_ns")
        ):
            raise ValueError(
                f"captured path resolution disagrees with accepted profile: {relative}"
            )
        result[relative] = {
            "path": row["path"],
            "resolved": row["resolved"],
            "islink": row["islink"],
            "device": row["device"],
            "inode": row["inode"],
            "mtime_ns": row["mtime_ns"],
            "bytes": row["bytes"],
            "registered_sha256": registered["sha256"],
        }
    return result


def _expected_arguments() -> list[str]:
    return [
        "scripts/train_propensity.py",
        "--name",
        "fg19_propensity_clean_f0_seed0_e4",
        "--hidden",
        "64",
        "--epochs",
        "4",
        "--lr",
        "0.002",
        "--seed",
        "0",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "3",
        "--no-eval-during-fit",
        "--split-manifest",
        B + "/splits/fold0.json",
        "--precision-profile",
        "strict_fp32_v1",
        "--loader-workers",
        "0",
    ]


def _payload() -> dict[str, object]:
    manifest_path = PROFILE_RECORDS / "SOURCE_MANIFEST.json"
    manifest = _read_json(manifest_path)
    source, packet = _bind_files(manifest)
    acceptance_path = PROFILE_RECORDS / "ACCEPTANCE.json"
    profile = _accepted_profile(_read_json(acceptance_path))
    config_path = PREPARATION / "PROPENSITY_PREREQUISITE.json"
    config_raw = config_path.read_bytes()
    config = json.loads(config_raw)
    config_sha = _sha256(config_raw)
    if (
        config.get("status") != "declared_not_run"
        or config_sha != profile.get("configuration_sha256")
        or packet.get("PROPENSITY_PREREQUISITE.json") != config_sha
        or config.get("arguments") != _expected_arguments()
        or config.get("run_name") != "fg19_propensity_clean_f0_seed0_e4"
        or config.get("source_commit") != COMMIT
    ):
        raise ValueError("fit configuration does not match the accepted profile and source")
    profile_admission = _read_json(PROFILE_RECORDS / "ADMISSION.json")
    review_path = PREPARATION / "FIT_ADMISSION_REVIEW_V3.json"
    review = _read_json(review_path)
    remote_path = PREPARATION / "propensity_fit_admission_remote_v3.py"
    launch_path = PREPARATION / "propensity_fit_launch_remote_v3.py"
    test_path = PREPARATION / "test_propensity_fit_admission_v3.py"
    local_path = Path(__file__).resolve()
    remote_components_sha = _sha256(remote_path.read_bytes() + b"\0" + launch_path.read_bytes())
    test_sha = _sha256(test_path.read_bytes())
    local_sha = _sha256(local_path.read_bytes())
    if (
        review.get("status") != "PASS"
        or review.get("remote_components_sha256") != remote_components_sha
        or review.get("test_sha256") != test_sha
        or review.get("local_binder_sha256") != local_sha
        or review.get("prerequisite_sha256") != config_sha
        or review.get("profile_result_sha256") != PROFILE_RESULT_SHA256
    ):
        raise ValueError("clean-fit admission does not have a current independent review")
    resolved_data_paths = _resolved_data_map(profile_admission, profile["valid_source"]["device"])
    return {
        "source_commit": COMMIT,
        "source_files": source,
        "file_hashes": {
            **{f"source/{name}": digest for name, digest in source.items()},
            **{f"packet/{name}": digest for name, digest in packet.items()},
        },
        "profile_result_sha256": PROFILE_RESULT_SHA256,
        "profile_source_hashes": profile["propensity_fit_source_sha256"],
        "profile_data_device": profile["valid_source"]["device"],
        "profile_data_stats": profile_admission["source_data_stats"],
        "profile_data_resolved_paths": resolved_data_paths,
        "path_diagnostic_sha256": PATH_DIAGNOSTIC_SHA256,
        "config_sha256": config_sha,
        "fit_arguments": _expected_arguments()[1:],
        "split_sha256": SPLIT_SHA256,
        "registry_sha256": REGISTRY_SHA256,
        "source_manifest_sha256": _sha256(manifest_path.read_bytes()),
        "local_profile_acceptance_sha256": _sha256(acceptance_path.read_bytes()),
        "review_sha256": _sha256(review_path.read_bytes()),
        "remote_components_sha256": remote_components_sha,
        "test_sha256": test_sha,
    }


def main() -> None:
    payload = _payload()
    encoded = base64.b64encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    template = (PREPARATION / "propensity_fit_admission_remote_v3.py").read_text()
    marker = 'PAYLOAD_B64 = "__PAYLOAD_B64__"'
    if template.count(marker) != 1:
        raise ValueError("remote payload marker is missing or duplicated")
    template = template.replace(marker, "PAYLOAD_B64 = " + repr(encoded.decode("ascii")), 1)
    launch = (PREPARATION / "propensity_fit_launch_remote_v3.py").read_text()
    core_b64 = base64.b64encode(template.encode()).decode("ascii")
    launch_b64 = base64.b64encode(launch.encode()).decode("ascii")
    remote = "\n".join(
        [
            "from __future__ import annotations",
            "import base64",
            "namespace = {'__name__': '__main__', '__builtins__': __builtins__}",
            f"exec(compile(base64.b64decode({core_b64!r}, validate=True), 'remote_preflight_v3.py', 'exec'), namespace)",
            f"exec(compile(base64.b64decode({launch_b64!r}, validate=True), 'remote_launch_v3.py', 'exec'), namespace)",
            "namespace['main']()",
            "",
        ]
    )
    compile(remote, "propensity_fit_admission_remote_v3.py", "exec")
    RECORDS.mkdir(parents=True, exist_ok=False)
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
        result = subprocess.run(command, input=remote.encode(), capture_output=True, timeout=120)
    except subprocess.TimeoutExpired as error:
        (RECORDS / "STDOUT.raw").write_bytes(_raw_bytes(error.stdout))
        (RECORDS / "STDERR.raw").write_bytes(_raw_bytes(error.stderr))
        (RECORDS / "TRANSPORT.json").write_text(
            json.dumps(
                {
                    "returncode": None,
                    "timed_out": True,
                    "unknown_outcome_policy": "inspect exact unit; never blindly retry",
                },
                indent=2,
            )
            + "\n"
        )
        raise RuntimeError(
            "SSH timed out; inspect preserved raw records and exact unit; never retry blindly"
        ) from error
    (RECORDS / "STDOUT.raw").write_bytes(result.stdout)
    (RECORDS / "STDERR.raw").write_bytes(result.stderr)
    (RECORDS / "TRANSPORT.json").write_text(
        json.dumps(
            {
                "returncode": result.returncode,
                "unknown_outcome_policy": "inspect exact unit; never blindly retry",
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
        for name, raw in returned.items():
            if name not in {"ADMISSION.json", "LAUNCH.json"}:
                raise ValueError("remote helper returned an unexpected evidence file")
            (RECORDS / name).write_bytes(base64.b64decode(raw, validate=True))
    if result.returncode:
        raise RuntimeError(
            "submission failed or had unknown outcome; inspect preserved raw records and unit"
        )
    print(
        json.dumps(
            {
                "records": str(RECORDS),
                "status": "submitted; inspect raw admission and launch evidence",
            }
        )
    )


if __name__ == "__main__":
    main()
