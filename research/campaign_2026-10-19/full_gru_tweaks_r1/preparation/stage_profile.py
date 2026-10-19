"""Stage exact reviewed propensity sources and admit one bounded GPU profile."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[1]
REPO = CAMPAIGN.parents[2]
PROP_REPO = REPO.parent / "WunderFunder-tl3-clean-propensity-20261018"
B = "/home/grinchenkoeg/computer-vision-homework-20261018"
SOURCE = B + "/source-fg19-propensity-235db75"
PACKET = B + "/research/full_gru_tweaks_r1/profile_v1"
OUTPUT = B + "/research_tmp/fg19_propensity_profile_v1"
UNIT = "computer-vision-homework-fg19-propensity-profile-v1.service"
COMMIT = "235db75ab91e24c69f21e637a83a48512dc674fe"


def main() -> None:
    """Write create-only transport/source evidence, then make one SSH submission."""
    records = CAMPAIGN / "records" / "profile_v1"
    records.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, str]] = {}
    names = subprocess.check_output(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            COMMIT,
            "--",
            "src/wnn",
            "scripts/train_rnn.py",
            "scripts/train_propensity.py",
            "pyproject.toml",
            "uv.lock",
        ],
        cwd=PROP_REPO,
        text=True,
    ).splitlines()
    for name in names:
        data = subprocess.check_output(["git", "show", f"{COMMIT}:{name}"], cwd=PROP_REPO)
        files[SOURCE + "/" + name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "base64": base64.b64encode(data).decode("ascii"),
        }
    for name in (
        "propensity_profile.py",
        "propensity_profile_core.py",
        "PROPENSITY_PREREQUISITE.json",
        "PROFILE_REVIEW.json",
    ):
        data = (CAMPAIGN / "preparation" / name).read_bytes()
        files[PACKET + "/" + name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "base64": base64.b64encode(data).decode("ascii"),
        }
    review = json.loads((CAMPAIGN / "preparation/PROFILE_REVIEW.json").read_bytes())
    assert review["status"] == "PASS"
    assert review["profile_sha256"] == files[PACKET + "/propensity_profile.py"]["sha256"]
    assert review["profile_core_sha256"] == files[PACKET + "/propensity_profile_core.py"]["sha256"]
    assert (
        review["prerequisite_sha256"] == files[PACKET + "/PROPENSITY_PREREQUISITE.json"]["sha256"]
    )
    assert (
        review["remote_admission_sha256"]
        == hashlib.sha256(
            (CAMPAIGN / "preparation/profile_admission_remote.py").read_bytes()
        ).hexdigest()
    )
    assert review["local_binder_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    remote = (
        "FILES="
        + repr(files)
        + "\n"
        + (CAMPAIGN / "preparation/profile_admission_remote.py").read_text()
    )
    (records / "REMOTE_SUBMISSION.py").write_text(remote)
    (records / "SOURCE_MANIFEST.json").write_text(
        json.dumps(
            {"commit": COMMIT, "files": {k: v["sha256"] for k, v in files.items()}}, indent=2
        )
        + "\n"
    )
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
        input=remote.encode(),
        capture_output=True,
        timeout=120,
    )
    (records / "STDOUT.raw").write_bytes(result.stdout)
    (records / "STDERR.raw").write_bytes(result.stderr)
    (records / "TRANSPORT.json").write_text(
        json.dumps(
            {
                "returncode": result.returncode,
                "unknown_outcome_policy": "no retry; inspect exact unit and durable remote receipt",
            },
            indent=2,
        )
        + "\n"
    )
    if result.returncode:
        raise RuntimeError(
            "Submission failed/unknown; inspect preserved records, never blindly retry"
        )
    for name, encoded in json.loads(result.stdout).items():
        data = base64.b64decode(encoded, validate=True)
        (records / name).write_bytes(data)
    print(
        json.dumps(
            {
                "unit": UNIT,
                "records": str(records),
                "status": "submitted; actual status in raw launch receipt",
            }
        )
    )


if __name__ == "__main__":
    main()
