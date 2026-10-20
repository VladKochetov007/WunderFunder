"""Seal a published canonical inference source into one root-controlled profile packet."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import zlib
from pathlib import Path

C = Path(__file__).resolve().parents[1]
SOURCE_REPO = Path("/Users/nb/Documents/wunder/WunderFunder-strong-gru-tweaks-20261019")
B = Path("/home/grinchenkoeg/computer-vision-homework-20261018")


def main() -> None:
    """Create local transport evidence only; this script never connects to Beleriand."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    commit = args.source_commit
    resolved = subprocess.check_output(
        ["git", "rev-parse", commit], cwd=SOURCE_REPO, text=True
    ).strip()
    assert commit == resolved and len(commit) == 40
    names = subprocess.check_output(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            commit,
            "--",
            "src/wnn",
            "pyproject.toml",
            "uv.lock",
            "scripts/train_rnn.py",
            "scripts/train_propensity.py",
            "scripts/profile_teacher.py",
            "scripts/predict_run.py",
            "scripts/fit_input_representation.py",
            "scripts/build_distill_cache.py",
        ],
        cwd=SOURCE_REPO,
        text=True,
    ).splitlines()
    files = {}
    for name in names:
        raw = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=SOURCE_REPO)
        files[name] = dict(
            bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
            base64=base64.b64encode(raw).decode("ascii"),
        )
    contract_raw = (C / "preparation/TEACHER_CACHE_EXECUTION_CONTRACT_R1.json").read_bytes()
    contract = json.loads(contract_raw)
    assert (
        hashlib.sha256(contract_raw).hexdigest()
        == "dbcbe194f3ceb16d499f1c68bc8d862a16c32af54a13f57db75284691d8aabb9"
    )
    acceptances = {}
    for teacher in contract["teachers"]:
        raw = (C / teacher["acceptance_path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == teacher["acceptance_sha256"]
        acceptances[teacher["run"]] = base64.b64encode(raw).decode("ascii")
    gates = (C / "records/teacher_profile_v2/REMOTE_SUBMISSION.py").read_bytes()
    assert (
        hashlib.sha256(gates).hexdigest()
        == "65151d60f69edc66d935d323813965906cacd57cb843eb3d70746348f16775ce"
    )
    source_root = B / f"source-fg19-cache-{commit[:8]}"
    payload = dict(
        source_commit=commit,
        source_root=str(source_root),
        files=files,
        contract=contract,
        contract_sha256=hashlib.sha256(contract_raw).hexdigest(),
        acceptances=acceptances,
        gates_zlib=base64.b64encode(zlib.compress(gates)).decode("ascii"),
        gates_sha256=hashlib.sha256(gates).hexdigest(),
    )
    template_path = C / "preparation/teacher_cache_profile_remote_v1.py"
    template = template_path.read_text()
    assert template.count("__PAYLOAD_B64__") == 1
    body = template.replace(
        "__PAYLOAD_B64__", base64.b64encode(json.dumps(payload).encode()).decode()
    ).encode()
    compile(body, "profile_packet", "exec")
    output = C / "preparation/teacher_cache_profile_v1"
    output.mkdir(exist_ok=False)
    (output / "REMOTE_SUBMISSION.py").write_bytes(body)
    manifest = dict(
        schema="fg19.teacher-cache-source-manifest.v1",
        source_commit=commit,
        source_root=str(source_root),
        files=[
            dict(path=name, bytes=item["bytes"], sha256=item["sha256"])
            for name, item in files.items()
        ],
    )
    manifest_raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (output / "SOURCE_MANIFEST.json").write_bytes(manifest_raw)
    receipt = dict(
        schema="fg19.teacher-cache-profile-local-preparation.v1",
        status="LOCAL_PREPARATION_NOT_LAUNCHED",
        source_commit=commit,
        source_root_b=str(source_root),
        source_files=len(files),
        source_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        source_payload_bytes=sum(item["bytes"] for item in files.values()),
        transport_bytes=len(body),
        transport_sha256=hashlib.sha256(body).hexdigest(),
        template_sha256=hashlib.sha256(template.encode()).hexdigest(),
        contract_sha256=payload["contract_sha256"],
        reused_gates_sha256=payload["gates_sha256"],
        scope="Source/metadata only; body compiled; root must inspect final source/payload/review before one SSH.",
    )
    (output / "PREPARATION.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
