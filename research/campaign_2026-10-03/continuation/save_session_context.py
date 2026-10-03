"""Save private restoration context and verify extraction into paused staging."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from datetime import datetime
from pathlib import Path

from save_progress_artifacts import ROOT, STATE


def main() -> None:
    """Archive explicit regular context files, then independently check every member."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore-dir", type=Path, required=True)
    parser.add_argument("--chain", type=Path, required=True)
    args = parser.parse_args()
    snapshot = ROOT / (STATE / "LATEST_SNAPSHOT").read_text().strip()
    destination = args.restore_dir.resolve(strict=True)
    continuation = STATE.relative_to(ROOT)
    if not (destination / continuation / "RESOURCE_FAILURE").is_file():
        raise ValueError("requires existing paused source restore")
    chain = args.chain.resolve(strict=True)
    chain.relative_to(ROOT / ".claude/chains")
    paths = sorted((STATE / "briefs").glob("*.txt")) + [STATE / "claude_agents.json", chain]
    archive = snapshot / "session-context.tar"
    if archive.exists():
        raise FileExistsError(archive)
    records = []
    with tarfile.open(archive, "w") as bundle:
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"context must be a regular file: {path}")
            data = path.read_bytes()
            if len(data) > 2 * 1024**2:
                raise ValueError("context file exceeds bounds")
            name = str(path.relative_to(ROOT))
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o600
            bundle.addfile(info, io.BytesIO(data))
            records.append(
                {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            )
        body = json.dumps({"members": records}, indent=2).encode()
        info = tarfile.TarInfo("session_context_manifest.json")
        info.size, info.mode = len(body), 0o600
        bundle.addfile(info, io.BytesIO(body))
    archive.chmod(0o600)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    with (snapshot / "SHA256SUMS").open("a") as output:
        output.write(f"{checksum}  session-context.tar\n")
    with tarfile.open(archive, "r:") as bundle:
        members = bundle.getmembers()
        expected = {record["path"]: record for record in records}
        if {member.name for member in members} != set(expected) | {"session_context_manifest.json"}:
            raise ValueError("unexpected context member")
        if len(members) != len(expected) + 1:
            raise ValueError("duplicate context member")
        for member in members:
            if not member.isfile() or member.name not in expected:
                if member.name == "session_context_manifest.json" and member.isfile():
                    continue
                raise ValueError("unexpected context type")
            source = bundle.extractfile(member)
            assert source is not None
            data = source.read()
            entry = expected[member.name]
            if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError("context member mismatch")
            target = destination / member.name
            if target.exists():
                raise FileExistsError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(0o600)
    report = {
        "status": "PASS",
        "paused": True,
        "members": len(records),
        "sha256": checksum,
        "services_started": 0,
        "context_captured_at": datetime.now().astimezone().isoformat(),
        "chain": str(chain.relative_to(ROOT)),
        "scope": "Private context extraction only; source and progress verification recorded separately.",
        "external_artifacts_restored": False,
    }
    (snapshot / "CONTEXT_VERIFICATION.json").write_text(json.dumps(report, indent=2) + "\n")
    source_progress = snapshot / "SOURCE_PROGRESS_VERIFICATION.json"
    if source_progress.is_file():
        restored = json.loads(source_progress.read_text())
        if restored.get("status") != "PASS" or restored.get("paused") is not True:
            raise ValueError("source/progress verification did not pass paused")
        combined = {
            "status": "PASS",
            "source_progress": restored,
            "context": report,
            "paused": True,
            "services_started": 0,
            "standalone_training_restore": False,
            "scope": "Individually timestamped source, numerical progress and private context actually extracted and hash-verified; live state is not atomic.",
        }
        (snapshot / "RECOVERY_VERIFICATION.json").write_text(json.dumps(combined, indent=2) + "\n")
    else:
        (snapshot / "RECOVERY_VERIFICATION.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
