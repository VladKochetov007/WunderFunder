"""Bind exact teacher-profile source/evidence and optionally submit one SSH admission."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[1]
REPO = CAMPAIGN.parents[2]
SOURCE_REPO = REPO.parent / "WunderFunder-strong-gru-tweaks-20261019"
PREP = CAMPAIGN / "preparation"
PROP_RECORDS = CAMPAIGN / "records/propensity_f0_e4_v3"
RECORDS = CAMPAIGN / "records/teacher_profile_v1"
REMOTE = PREP / "teacher_profile_admission_remote.py"
B = "/home/grinchenkoeg/computer-vision-homework-20261018"
SOURCE = B + "/source-fg19-teacher-af21d6d8"
PACKET = B + "/research/full_gru_tweaks_r1/teacher_profile_v1"
OUTPUT = B + "/research_tmp/fg19_teacher_profile_v1"
COMMIT = "af21d6d8409c46b36830f0bdddd50e67ee17025f"
UNIT = "computer-vision-homework-fg19-teacher-profile-v1.service"
PACKET_FILES = {
    "TEACHER_PROFILE_ADMISSION_REVIEW.json": PREP / "TEACHER_PROFILE_ADMISSION_REVIEW.json",
    "TEACHER_PROFILE_BINDING.json": PREP / "TEACHER_PROFILE_BINDING.json",
    "TEACHER_PROFILE_COLD_IMPORT.json": PREP / "TEACHER_PROFILE_COLD_IMPORT.json",
    "TEACHER_SOURCE_MANIFEST.json": PREP / "TEACHER_SOURCE_MANIFEST.json",
    "TEACHER_PROFILE_SOURCE_PUBLICATION.json": PREP / "TEACHER_PROFILE_SOURCE_PUBLICATION.json",
    "PROPENSITY_PREREQUISITE.json": PREP / "PROPENSITY_PREREQUISITE.json",
    "teacher_profile_admission_remote.py": REMOTE,
    "PROPENSITY_TERMINAL_ACCEPTANCE.json": PROP_RECORDS / "TERMINAL_ACCEPTANCE.json",
    "PROPENSITY_NATIVE_LAUNCH.json": PROP_RECORDS / "LAUNCH.json",
    "PROPENSITY_VERIFY_ADMISSION.json": PROP_RECORDS
    / "terminal_verification_v2/capture/ADMISSION.json",
    "PROPENSITY_VERIFY_LAUNCH.json": PROP_RECORDS / "terminal_verification_v2/capture/LAUNCH.json",
    "PROPENSITY_VERIFICATION.json": PROP_RECORDS
    / "terminal_verification_v2/capture/VERIFICATION.json",
    "PROPENSITY_STOP_CAPTURE.json": PROP_RECORDS / "terminal_exact_stop/STDOUT.raw.json",
}


def _bytes_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_item(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required source/evidence file is not a regular file: {path}")
    raw = path.read_bytes()
    if len(raw) > 8_000_000:
        raise RuntimeError(f"single source/evidence file exceeds the staging cap: {path}")
    return {
        "sha256": _bytes_sha(raw),
        "base64": base64.b64encode(raw).decode("ascii"),
    }


def _git_file(commit: str, name: str) -> bytes:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or name.startswith("/"):
        raise RuntimeError(f"source manifest contains an unsafe path: {name}")
    return subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=SOURCE_REPO)


def _make_reviewed_payload() -> dict[str, object]:
    manifest_path = PREP / "TEACHER_SOURCE_MANIFEST.json"
    manifest_raw = manifest_path.read_bytes()
    manifest_sha = _bytes_sha(manifest_raw)
    manifest = json.loads(manifest_raw)
    if (
        manifest.get("source_commit") != COMMIT
        or manifest.get("source_root") != SOURCE
        or len(manifest.get("files", [])) != 164
        or manifest.get("archive_sha256")
        != "1c2cca36cad3d828dd393382882f690694b578ee2287bd8beb51f7161ff797ad"
    ):
        raise RuntimeError("teacher source manifest is not the published immutable source")

    binding_path = PREP / "TEACHER_PROFILE_BINDING.json"
    binding_raw = binding_path.read_bytes()
    binding = json.loads(binding_raw)
    argv = binding.get("bounded_profile", {}).get("profile_argv")
    if binding.get("schema") != "full-gru-tweaks.teacher-profile-binding.v4" or not isinstance(
        argv, list
    ):
        raise RuntimeError("teacher profile binding is missing or has an unsupported schema")
    if (
        binding.get("bounded_profile", {}).get("fit_scope", {}).get("include_heldout_valid")
        is not False
    ):
        raise RuntimeError("profile binding includes held-out validation")
    if "--valid-fold" not in argv:
        argv = [*argv, "--valid-fold", "0"]
    elif argv[argv.index("--valid-fold") + 1] != "0":
        raise RuntimeError("profile binding does not explicitly select fold 0")

    cold_path = PREP / "TEACHER_PROFILE_COLD_IMPORT.json"
    cold_raw = cold_path.read_bytes()
    cold = json.loads(cold_raw)
    required_entrypoints = {
        "scripts/profile_teacher.py",
        "scripts/train_rnn.py",
        "scripts/train_propensity.py",
        "scripts/predict_run.py",
        "scripts/fit_input_representation.py",
    }
    if (
        cold.get("status") != "PASS"
        or cold.get("runtime_wnn_module_count") != 99
        or not cold.get("all_runtime_origins_inside_isolated_wnn_copy")
        or cold.get("missing_semantic_modules")
        or cold.get("wrong_runtime_origins")
        or set(cold.get("entrypoint_paths", {})) != required_entrypoints
        or cold.get("data_read_sentinels", {}).get("calls") != []
        or any(
            cold.get("data_read_sentinels", {}).get(name) is not True
            for name in (
                "SequenceFile.read",
                "numpy.load",
                "numpy.memmap",
                "pyarrow.parquet",
                "torch.load",
            )
        )
    ):
        raise RuntimeError("teacher source cold-import proof is incomplete or failed")

    publication = json.loads((PREP / "TEACHER_PROFILE_SOURCE_PUBLICATION.json").read_bytes())
    if (
        publication.get("source_commit") != COMMIT
        or publication.get("source_manifest_sha256") != manifest_sha
        or publication.get("test_evidence", {}).get("cold_import_sha256") != _bytes_sha(cold_raw)
    ):
        raise RuntimeError("teacher source publication receipt does not bind the current files")

    review_path = PREP / "TEACHER_PROFILE_ADMISSION_REVIEW.json"
    review_raw = review_path.read_bytes()
    review = json.loads(review_raw)
    helper_sha = _bytes_sha(REMOTE.read_bytes())
    if (
        review.get("status") != "PASS"
        or review.get("remote_helper_sha256") != helper_sha
        or review.get("local_binder_sha256") != _bytes_sha(Path(__file__).read_bytes())
        or review.get("binding_sha256") != _bytes_sha(binding_raw)
        or review.get("source_manifest_sha256") != manifest_sha
        or review.get("cold_import_sha256") != _bytes_sha(cold_raw)
    ):
        raise RuntimeError("independent teacher-profile admission review is missing/stale")

    source_items: dict[str, dict[str, str]] = {}
    for entry in manifest["files"]:
        name = entry["path"]
        raw = _git_file(COMMIT, name)
        if len(raw) != entry["bytes"] or _bytes_sha(raw) != entry["sha256"]:
            raise RuntimeError(f"immutable teacher source bytes changed: {name}")
        source_items[f"{SOURCE}/{name}"] = {
            "sha256": entry["sha256"],
            "base64": base64.b64encode(raw).decode("ascii"),
        }
    if len(source_items) != 164:
        raise RuntimeError("source materialization did not match all 164 pinned files")

    packet_items = {f"{PACKET}/{name}": _file_item(path) for name, path in PACKET_FILES.items()}
    all_files = source_items | packet_items
    if sum(len(item["base64"]) * 3 // 4 for item in all_files.values()) > 30_000_000:
        raise RuntimeError("complete source/evidence payload exceeds its transport cap")

    prop_admission = json.loads((PROP_RECORDS / "ADMISSION.json").read_bytes())
    accepted = json.loads((PROP_RECORDS / "TERMINAL_ACCEPTANCE.json").read_bytes())
    stop_path = PROP_RECORDS / "terminal_exact_stop/STDOUT.raw.json"
    verification_raw = (
        PROP_RECORDS / "terminal_verification_v2/capture/VERIFICATION.json"
    ).read_bytes()
    if (
        prop_admission.get("split_manifest_sha256")
        != "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
        or prop_admission.get("run_name") != "fg19_propensity_clean_f0_seed0_e4"
        or accepted.get("status") != "ACCEPTED"
        or accepted.get("model", {}).get("sha256")
        != "6facf4ca9045118b8e18ca590053c97e2f5f5122c92f228f75002f0c76a2800e"
        or accepted.get("verification_sha256") != _bytes_sha(verification_raw)
    ):
        raise RuntimeError("clean F0 propensity prerequisite is not exactly terminal-verified")

    dataset_map = prop_admission.get("data_resolution_map")
    if not isinstance(dataset_map, dict) or len(dataset_map) != 6:
        raise RuntimeError("exact six-path registered F0 data identity map is missing")
    return {
        "source_root": SOURCE,
        "source_manifest": manifest,
        "source_manifest_sha256": manifest_sha,
        "binding_sha256": _bytes_sha(binding_raw),
        "cold_import_sha256": _bytes_sha(cold_raw),
        "review_sha256": _bytes_sha(review_raw),
        "split_sha256": "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7",
        "registry_sha256": prop_admission["portable_source_registry_sha256"],
        "data_resolution_map": dataset_map,
        "profile_argv": argv,
        "packet_names": list(PACKET_FILES),
        "files": all_files,
        "propensity": {
            "run_name": "fg19_propensity_clean_f0_seed0_e4",
            "fit_invocation_id": "85b87a83f96945f4b78908f124435a2b",
            "verify_invocation_id": "9319de78e74a45f0a01f04b1c14ee042",
            "verification_sha256": _bytes_sha(verification_raw),
            "verification_capture": _file_item(
                PROP_RECORDS / "terminal_verification_v2/capture/VERIFICATION.json"
            ),
            "model_sha256": "6facf4ca9045118b8e18ca590053c97e2f5f5122c92f228f75002f0c76a2800e",
            "acceptance": _file_item(PROP_RECORDS / "TERMINAL_ACCEPTANCE.json"),
            "native_launch": _file_item(PROP_RECORDS / "LAUNCH.json"),
            "verifier_admission": _file_item(
                PROP_RECORDS / "terminal_verification_v2/capture/ADMISSION.json"
            ),
            "verifier_launch": _file_item(
                PROP_RECORDS / "terminal_verification_v2/capture/LAUNCH.json"
            ),
            "stop": _file_item(stop_path),
        },
    }


def _remote_script(payload: dict[str, object]) -> bytes:
    template = REMOTE.read_text()
    if template.count('PAYLOAD_B64 = "__PAYLOAD_B64__"') != 1:
        raise RuntimeError("remote profile admission template does not have one payload marker")
    encoded = base64.b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).decode("ascii")
    return template.replace(
        'PAYLOAD_B64 = "__PAYLOAD_B64__"', f'PAYLOAD_B64 = "{encoded}"'
    ).encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submit", action="store_true", help="submit once over SSH after all local bindings pass"
    )
    args = parser.parse_args()
    payload = _make_reviewed_payload()
    remote = _remote_script(payload)
    RECORDS.mkdir(parents=True, exist_ok=False)
    (RECORDS / "REMOTE_SUBMISSION.py").write_bytes(remote)
    manifest = {
        "schema": "fg19.teacher-profile-source-payload.v1",
        "source_commit": COMMIT,
        "source_manifest_sha256": payload["source_manifest_sha256"],
        "files": {path: item["sha256"] for path, item in payload["files"].items()},
        "review_sha256": payload["review_sha256"],
        "remote_script_sha256": _bytes_sha(remote),
        "unit": UNIT,
        "output": OUTPUT,
        "mode": "submit" if args.submit else "prepared_only",
    }
    (RECORDS / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if not args.submit:
        print(json.dumps({"status": "prepared_only", "records": str(RECORDS), "unit": UNIT}))
        return
    result = subprocess.run(
        [
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
        ],
        input=remote,
        capture_output=True,
        timeout=120,
    )
    (RECORDS / "STDOUT.raw").write_bytes(result.stdout)
    (RECORDS / "STDERR.raw").write_bytes(result.stderr)
    (RECORDS / "TRANSPORT.json").write_text(
        json.dumps(
            {
                "returncode": result.returncode,
                "unknown_outcome_policy": "never retry; inspect exact unit and durable profile receipt",
            },
            indent=2,
        )
        + "\n"
    )
    if result.returncode:
        raise RuntimeError("SSH submission failed or is unknown; preserve and inspect this record")
    outer = json.loads(result.stdout)
    for name, encoded in outer.items():
        raw = base64.b64decode(encoded, validate=True)
        (RECORDS / name).write_bytes(raw)
    print(json.dumps({"status": "submitted", "records": str(RECORDS), "unit": UNIT}))


if __name__ == "__main__":
    main()
