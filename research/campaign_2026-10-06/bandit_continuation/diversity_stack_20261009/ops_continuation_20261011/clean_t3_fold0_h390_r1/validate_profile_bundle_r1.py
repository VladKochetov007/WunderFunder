from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[5]
BUNDLE_DIR = PACKET / "bundle_teacher_profile_r1"
ARCHIVE = BUNDLE_DIR / "bundle.wnn"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from kaggle_common import read_spec, verify_files  # noqa: PLC0415

    manifest = json.loads((BUNDLE_DIR / "bundle-manifest.json").read_text())
    archive_sha = sha256(ARCHIVE)
    if archive_sha != manifest["archive_sha256"]:
        raise ValueError("profile bundle archive digest differs from its manifest")
    kernel = (BUNDLE_DIR / "kernel/kernel.py").read_text()
    pinned = re.search(r"EXPECTED_ARCHIVE_SHA256\s*=\s*'([0-9a-f]{64})'", kernel)
    if not pinned or pinned.group(1) != archive_sha:
        raise ValueError("generated Kaggle kernel does not pin this exact archive")
    external_mounts = manifest.get("external_mounts", {})
    if not external_mounts:
        raise ValueError("profile bundle omitted its mounted data/propensity dependencies")
    if "scripts/kaggle_gpu_runner.py" not in {item["path"] for item in manifest["files"]}:
        raise ValueError("the sealed kernel consumer runner is absent from the archive")

    with tempfile.TemporaryDirectory(prefix="wnn-clean-t3-profile-consumer-") as temp_name:
        extracted = Path(temp_name)
        with tarfile.open(ARCHIVE, "r:*") as archive:
            names = []
            for member in archive.getmembers():
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"unsafe archive path: {member.name}")
                names.append(member.name)
                if not member.isfile():
                    continue
                destination = extracted.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read archive member: {member.name}")
                with source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
        expected_names = ({item["path"] for item in manifest["files"]} - set(external_mounts)) | set(manifest["spec_paths"])
        if not expected_names.issubset(set(names)):
            raise ValueError("archive payload is missing a manifest-bound file")
        for item in manifest["files"]:
            if item["path"] in external_mounts:
                continue
            payload = extracted / item["path"]
            if payload.stat().st_size != item["size"] or sha256(payload) != item["sha256"]:
                raise ValueError(f"archive member differs from its manifest pin: {item['path']}")
        if any("teacher_fit_templates" in name or name.endswith("_e3_s4_r1.json") or name.endswith("_e3_s5_r1.json") for name in names):
            raise ValueError("profile archive accidentally contains a full teacher fit template")

        profile_records = []
        entrypoint = (
            "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
            "ops_continuation_20261011/clean_t3_fold0_h390_r1/source_snapshot/scripts/cleanfold_exec.py"
        )
        code_by_path = {item["path"]: item for item in manifest["files"]}
        for spec_rel in manifest["spec_paths"]:
            spec_path = extracted / spec_rel
            spec = read_spec(spec_path)
            verify_files(extracted, spec["code"])
            argv = spec["argv"]
            if argv[1] != entrypoint or entrypoint not in code_by_path:
                raise ValueError("profile argv does not name the exact packaged child entrypoint")
            if argv[2] != "scripts/train_rnn.py":
                raise ValueError("profile consumer does not invoke the frozen trainer child")
            if argv[argv.index("--profile-steps") + 1] != "25" or argv[argv.index("--profile-warmup") + 1] != "5":
                raise ValueError("profile job step/warmup contract changed")
            if "--save-training-state" in argv or spec["artifacts"]:
                raise ValueError("profile stage must not write or collect a fit checkpoint")
            if spec["environment"].get("WNN_LOADER_WORKERS") != "0":
                raise ValueError("profile job does not enforce the verified synchronous loader")
            if {item["path"] for item in spec["inputs"]} != set(external_mounts):
                raise ValueError("spec inputs differ from the kernel external mount map")
            if not (extracted / entrypoint).is_file():
                raise ValueError("frozen worker child entrypoint is absent after archive extraction")
            profile_records.append({
                "name": spec["name"],
                "spec_path": spec_rel,
                "spec_sha256": sha256(spec_path),
                "source_code_hashes_verified": len(spec["code"]),
                "entrypoint": entrypoint,
                "entrypoint_sha256": code_by_path[entrypoint]["sha256"],
                "profile_steps": 25,
                "profile_warmup": 5,
                "save_training_state": False,
                "loader_workers": 0,
            })
        if len(profile_records) != 2:
            raise ValueError("expected exactly two independent T4 profile jobs")

        # Exercise the exact final archive wrapper and trainer CLI imports,
        # without starting a profile, model fit, provider or queue action.
        env = os.environ.copy()
        env.update({"WNN_PROJECT_ROOT": str(extracted), "WNN_LOADER_WORKERS": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        command = [sys.executable, str(extracted / entrypoint), "scripts/train_rnn.py", "--help"]
        cli = subprocess.run(command, cwd=extracted, env=env, text=True, capture_output=True, check=False)
        if cli.returncode != 0:
            raise RuntimeError(f"frozen trainer CLI failed through archive child entrypoint: {cli.stderr[-2000:]}")
        required_options = ["--profile-steps", "--profile-warmup", "--save-training-state", "--checkpoint-epoch", "--muon-scope"]
        missing = [option for option in required_options if option not in cli.stdout]
        if missing:
            raise ValueError(f"frozen trainer CLI omitted expected options: {missing}")
        cli_record = {
            "argv": [sys.executable, f"<extracted>/{entrypoint}", "scripts/train_rnn.py", "--help"],
            "returncode": cli.returncode,
            "stdout_sha256": hashlib.sha256(cli.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(cli.stderr.encode()).hexdigest(),
            "required_options_present": required_options,
            "scope": "exact final archive child entrypoint imported and ran frozen train_rnn.py argparse help; no profile or fit",
        }
        prediction_command = [sys.executable, str(extracted / entrypoint), "scripts/predict_run.py", "--help"]
        prediction_cli = subprocess.run(prediction_command, cwd=extracted, env=env, text=True, capture_output=True, check=False)
        if prediction_cli.returncode != 0:
            raise RuntimeError(f"frozen prediction CLI failed through archive child entrypoint: {prediction_cli.stderr[-2000:]}")
        prediction_option = "--campaign-full-valid-feature-only"
        if prediction_option not in prediction_cli.stdout:
            raise ValueError("frozen prediction CLI omitted the full-valid feature-only export option")
        prediction_record = {
            "argv": [sys.executable, f"<extracted>/{entrypoint}", "scripts/predict_run.py", "--help"],
            "returncode": prediction_cli.returncode,
            "stdout_sha256": hashlib.sha256(prediction_cli.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(prediction_cli.stderr.encode()).hexdigest(),
            "required_option": prediction_option,
            "scope": "exact final archive child entrypoint imported and ran frozen predict_run.py argparse help; no prediction export",
        }

    receipt = {
        "schema": "wnn.cleanfold_t3_profile_bundle_static_consumer_validation.r1",
        "status": "pass_static_final_archive_consumer_and_payload",
        "scope": "Final built archive SHA, generated kernel pin, tar members, actual kaggle_common spec parser/code verifier, exact child entrypoint, profile-only job contract and external mount closure; no GPU/provider/fit/queue action.",
        "archive_sha256": archive_sha,
        "archive_bytes": ARCHIVE.stat().st_size,
        "generated_kernel_entrypoint": "bundle_teacher_profile_r1/kernel/kernel.py",
        "kernel_archive_sha_matches": True,
        "spec_paths": manifest["spec_paths"],
        "profile_workers": profile_records,
        "frozen_trainer_cli_check": cli_record,
        "frozen_prediction_cli_check": prediction_record,
        "external_mount_count": len(external_mounts),
        "mounted_source_and_data_dataset_ids": sorted({item["dataset"] for item in external_mounts.values() if isinstance(item, dict)}),
        "full_teacher_fits_in_bundle": False,
    }
    output = PACKET / "PROFILE_BUNDLE_STATIC_VALIDATION_R1.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
